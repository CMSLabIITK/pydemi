"""
pydemi.core.partition
=====================
Assignment of voxels to atoms (spec §9), behind one interface.

Every scheme yields memory-bounded chunks of (voxel, atom, weight, distance,
direction) pairs, where distance and direction are measured from THAT atom's
nucleus (the periodic image the scheme used) to the voxel. Hard schemes emit
one pair per voxel with weight 1; smooth schemes emit several with weights
summing to 1. A site-restricted sum is then one code path for every scheme,

    X^(i) = sum over pairs with atom i of  w * g(field_k, r_ik, u_ik),

i.e. the indicator 1(i(k) = i) replaced by w_i(r_k).

Schemes
-------
``nearest``    i(k) = argmin_i |r_k - R_i|                        (default)
``power``      i(k) = argmin_i |r_k - R_i|^2 - R_i^2, R_i the covalent radius
``becke``      Becke fuzzy cells, pure geometry
``hirshfeld``  w_i(r) = rho_free_i(r - R_i) / sum_j rho_free_j(r - R_j)

Becke in periodic solids
------------------------
Becke's product P_i = prod_{j != i} s(mu_ij) was built for molecules. In a
solid the number of competitors at distance R grows as R^2 while each factor
approaches 1 only as (d/R)^8, so the weights converge slowly with the number
of images in the product (FeNi3: max weight error 1.6e-2 with 60
competitors, 1.0e-3 with 300). Weights are assigned to the BECKE_CELLS
nearest images and every product runs over the BECKE_K nearest images;
both are part of the definition and recorded in the metadata. Hirshfeld,
whose free-atom densities decay exponentially, has no such problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from ..constants import BECKE_CELLS, BECKE_K, MIN_PAIR_WEIGHT
from ..data import atomic_number, covalent_radius, default_zval
from ..io.base import Structure, VolumetricData
from .geometry import NearestAtom, assign_atoms, geometry_of, image_points, reps_for
from .grid import cart_coords

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class PairChunk:
    voxel: I64          # (M,) flat voxel index
    atom: I64           # (M,) atom index
    weight: F64         # (M,)
    distance: F64       # (M,) Angstrom, voxel to this atom's image
    direction: F64      # (M, 3) unit vector nucleus -> voxel


class Partition:
    scheme = "abstract"

    def __init__(self, n_atoms: int) -> None:
        self.n_atoms = n_atoms

    def pairs(self) -> Iterator[PairChunk]:
        raise NotImplementedError

    def site_sum(self, f: NDArray[Any]) -> F64:
        """sum_k w_i(k) f_k for every atom i."""
        flat = np.asarray(f, dtype=np.float64).ravel()
        out = np.zeros(self.n_atoms)
        for p in self.pairs():
            out += np.bincount(p.atom, weights=p.weight * flat[p.voxel], minlength=self.n_atoms)
        return out


class HardPartition(Partition):
    """Every voxel belongs wholly to one atom (nearest atom or power diagram)."""

    def __init__(self, assignment: NearestAtom, n_atoms: int, scheme: str, chunk: int = 1 << 20):
        super().__init__(n_atoms)
        self.scheme = scheme
        self.assignment = assignment
        self.labels = assignment.atom_index.ravel()
        self.chunk = chunk

    def pairs(self) -> Iterator[PairChunk]:
        dist = self.assignment.distance.ravel()
        direction = self.assignment.direction.reshape(-1, 3)
        n = dist.size
        for s in range(0, n, self.chunk):
            v = np.arange(s, min(s + self.chunk, n), dtype=np.int64)
            yield PairChunk(v, self.labels[v], np.ones(v.size), dist[v], direction[v])

    def site_sum(self, f: NDArray[Any]) -> F64:
        return np.asarray(np.bincount(self.labels, weights=np.asarray(f, np.float64).ravel(),
                                      minlength=self.n_atoms), dtype=np.float64)


def power_radii(vd: VolumetricData) -> F64:
    """Covalent radius of every atom (the power-diagram weights)."""
    return np.array([covalent_radius(s) for s in vd.structure.species], dtype=np.float64)


def partition_of(vd: VolumetricData, scheme: str) -> Partition:
    """The partition ``scheme`` for ``vd``, cached on ``vd``."""
    key = ("partition", scheme)
    if key not in vd.cache:
        n = vd.structure.n_atoms
        if scheme == "nearest":
            vd.cache[key] = HardPartition(geometry_of(vd), n, "nearest")
        elif scheme == "power":
            vd.cache[key] = HardPartition(assign_atoms(vd.shape, vd.structure, power_radii(vd)),
                                          n, "power")
        elif scheme == "becke":
            vd.cache[key] = BeckePartition(vd.shape, vd.structure)
        elif scheme == "hirshfeld":
            vd.cache[key] = HirshfeldPartition(vd.shape, vd.structure, hirshfeld_tables(vd))
        else:
            raise ValueError(f"unknown partition {scheme!r}")
    out: Partition = vd.cache[key]
    return out


# ----------------------------------------------------------------------
# Becke
# ----------------------------------------------------------------------

def _renormalized(w: F64, row: I64, n_rows: int) -> F64:
    """Kept pair weights rescaled so every voxel's weights sum to exactly 1."""
    total = np.bincount(row, weights=w, minlength=n_rows)
    return np.asarray(w / total[row], dtype=np.float64)


def _becke_step(mu: F64) -> F64:
    for _ in range(3):
        mu = mu * (1.5 - 0.5 * mu * mu)
    return np.asarray(0.5 * (1.0 - mu))


class BeckePartition(Partition):
    """Becke fuzzy cells (pure geometry, no size adjustment).

    w_i(r) = P_i(r) / sum_j P_j(r),  P_i = prod_{j != i} s(mu_ij),
    mu_ij = (|r - R_i| - |r - R_j|) / |R_i - R_j|,  s(mu) = (1 - f(f(f(mu)))) / 2,
    f(p) = 3p/2 - p^3/2  (Becke, J. Chem. Phys. 88, 2547 (1988)).
    """

    scheme = "becke"

    def __init__(self, shape: tuple[int, int, int], structure: Structure, k: int = BECKE_K,
                 cells: int = BECKE_CELLS, chunk: int = 1 << 13) -> None:
        super().__init__(structure.n_atoms)
        if cells < 1 or k < cells:
            raise ValueError(f"need 1 <= cells <= k, got cells={cells}, k={k}")
        self.shape, self.structure, self.k, self.cells, self.chunk = shape, structure, k, cells, chunk

    def pairs(self) -> Iterator[PairChunk]:
        x_all = cart_coords(self.shape, self.structure.lattice).reshape(-1, 3)
        reps = (1, 1, 1)
        pts, owner, _ = image_points(self.structure, reps)
        tree = cKDTree(pts)
        for s in range(0, len(x_all), self.chunk):
            x = x_all[s:s + self.chunk]
            while True:
                kk = min(self.k, len(pts))
                d, idx = tree.query(x, k=kk)
                d, idx = np.asarray(d).reshape(len(x), -1), np.asarray(idx).reshape(len(x), -1)
                needed = reps_for(float(d[:, -1].max()), self.structure.lattice)
                if all(a <= b for a, b in zip(needed, reps)):
                    break
                reps = (max(needed[0], reps[0]), max(needed[1], reps[1]), max(needed[2], reps[2]))
                pts, owner, _ = image_points(self.structure, reps)
                tree = cKDTree(pts)
            P = pts[idx]
            nc = min(self.cells, d.shape[1])
            R = np.linalg.norm(P[:, :nc, None, :] - P[:, None, :, :], axis=-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                mu = (d[:, :nc, None] - d[:, None, :]) / R
            sij = _becke_step(np.nan_to_num(mu))
            sij[:, np.arange(nc), np.arange(nc)] = 1.0
            cell = sij.prod(axis=2)
            w = cell / cell.sum(axis=1, keepdims=True)
            m, j = np.nonzero(w > MIN_PAIR_WEIGHT)
            wk = _renormalized(w[m, j], m, len(x))
            diff = x[m] - P[m, j]
            dist = d[m, j]
            safe = np.where(dist > 1e-12, dist, 1.0)
            direction = np.where(dist[:, None] > 1e-12, diff / safe[:, None], 0.0)
            yield PairChunk(np.asarray(s + m, dtype=np.int64), owner[idx][m, j], wk, dist,
                            direction)


# ----------------------------------------------------------------------
# Hirshfeld
# ----------------------------------------------------------------------

class HirshfeldPartition(Partition):
    """w_i(r) = rho_free_i(r - R_i) / sum_j rho_free_j(r - R_j), over all periodic images.

    Uses the same free-atom tables and image sums as the promolecule; a voxel
    where every free-atom density vanishes goes wholly to its nearest image.
    """

    scheme = "hirshfeld"

    def __init__(self, shape: tuple[int, int, int], structure: Structure,
                 tables: Mapping[str, Any]) -> None:
        super().__init__(structure.n_atoms)
        self.shape, self.structure, self.tables = shape, structure, tables

    def pairs(self) -> Iterator[PairChunk]:
        from ..fields.deformation import atom_density_blocks
        for b in atom_density_blocks(self.shape, self.structure, self.tables):
            dens = b.density.copy()
            total = dens.sum(axis=1)
            empty = total <= 0
            if empty.any():
                rows = np.flatnonzero(empty)
                dens[rows, np.argmin(b.distance[empty], axis=1)] = 1.0
                total[empty] = 1.0
            w = dens / total[:, None]
            m, c = np.nonzero(w > MIN_PAIR_WEIGHT)
            wk = _renormalized(w[m, c], m, len(b.voxel))
            dist = b.distance[m, c]
            safe = np.where(dist > 1e-12, dist, 1.0)
            direction = np.where(dist[:, None] > 1e-12, b.diff[m, c] / safe[:, None], 0.0)
            yield PairChunk(b.voxel[m], b.owner[c], wk, dist, direction)


def hirshfeld_tables(vd: VolumetricData) -> Mapping[str, Any]:
    """Free-atom tables matching the density: all electrons (all-electron density) or ZVAL."""
    from ..fields.deformation import reference_tables
    part = "total" if vd.density_source == "all_electron" else "valence"
    return reference_tables(vd.structure, "tabulated", part, vd.zval)


def reference_electrons(vd: VolumetricData) -> F64:
    """Z_i of the Hirshfeld charge: the atomic number for an all-electron density, ZVAL for a
    pseudo-density (documented correction: q_i = Z_i - N_i needs the electrons the density holds)."""
    if vd.density_source == "all_electron":
        return np.array([float(atomic_number(s)) for s in vd.structure.species])
    z = vd.zval or {}
    return np.array([float(z[s]) if s in z else default_zval(s) for s in vd.structure.species])


def hirshfeld_charges(vd: VolumetricData) -> F64:
    """Hirshfeld atomic charges q_i = Z_i - int w_i rho dV (for comparison with Bader charges).

    Z_i is ZVAL for a PAW pseudo-density (CHGCAR) and the atomic number for
    AECCAR0 + AECCAR2. With a CHGCAR the free-atom weights keep their
    all-electron shape inside the augmentation spheres, so the partition of
    the pseudized density there is approximate.
    """
    part = partition_of(vd, "hirshfeld")
    N = part.site_sum(vd.rho.data) * vd.rho.dV
    return np.asarray(reference_electrons(vd) - N, dtype=np.float64)
