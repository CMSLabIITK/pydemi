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

All three need only the structure (and, for ``power``, the tabulated
covalent radii).

Becke in periodic solids
------------------------
Becke's product P_i = prod_{j != i} s(mu_ij) was built for molecules. In a
solid the number of competitors at distance R grows as R^2 while each factor
approaches 1 only as (d/R)^8, so the weights converge slowly with the number
of images in the product (FeNi3: max weight error 1.6e-2 with 60
competitors, 1.0e-3 with 300). Weights are assigned to the BECKE_CELLS
nearest images and every product runs over the BECKE_K nearest images;
both are part of the definition and recorded in the metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from ..constants import BECKE_CELLS, BECKE_K, MIN_PAIR_WEIGHT
from ..data import covalent_radius
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
