"""
pydemi.partition
----------------
Assignment of grid voxels to atoms (reference Family H).

Every scheme exposes one interface, :meth:`Partition.pairs`: memory-bounded
chunks of (voxel, atom, weight, distance, direction) pairs, where distance
and direction are measured from *that* atom's nucleus (the nearest periodic
image of it that the scheme used) to the voxel. Hard schemes emit one pair
per voxel with weight 1; soft schemes emit several with weights summing to
1. A per-site quantity is then a weighted sum over pairs,

    X^(i) = F( sum over pairs with atom i of w * g(rho_k, r_ik, ...) ),

so nearest-atom, power-diagram and Becke results come from the same code.

Schemes
-------
``nearest``  i(k) = argmin_i |r_k - R_i|                 (current default)
``power``    i(k) = argmin_i |r_k - R_i|^2 - R_i^rad^2    (entry 99)
``becke``    Becke fuzzy cells, w_i(r) = P_i / sum_j P_j  (entry 100)
``hirshfeld`` w_i(r) = rho_i^free(r) / sum_j rho_j^free(r)  (entry 101)

Becke (J. Chem. Phys. 88, 2547 (1988)): P_i = prod_{j != i} s(mu_ij),
mu_ij = (r_i - r_j) / R_ij, s(mu) = (1 - f(f(f(mu)))) / 2, f(p) = 1.5 p - 0.5 p^3.
The optional heteronuclear size adjustment uses per-atom radii.

Becke in periodic solids (opt-in)
---------------------------------
Becke's scheme was built for molecules, where the product over all atoms is
finite. In a periodic solid a far atom at distance R has s = 1 - O((d/R)^8)
only once R >> d, and the number of such atoms grows as R^2, so the weights
converge slowly with the number of atoms in the product. Measured on an fcc
FeCoNiCr cell: max weight error ~2e-2 with 60 competitors, ~9e-3 with 100,
~1e-3 with 300. The weight itself is always carried by the few nearest
images (weight beyond the 8 nearest < 1e-9).

So the implementation separates the two roles: weights are assigned to the
``cells`` nearest images, and each product runs over the ``k`` nearest
images (``k >= cells``). Truncating the product only, not the candidate
set, avoids the much larger error of dropping equidistant competitors from
a kept cell. ``k`` is part of the definition and must be reported with any
Becke result. Because converged Becke is too slow for production grids,
Becke is opt-in; the default smooth partition is Hirshfeld (entry 101),
which is naturally local.
"""

from dataclasses import dataclass
import itertools
from typing import Iterator, Optional

import numpy as np
from scipy.spatial import cKDTree

from .geometry import NearestAtom, _image_points, _reps_for
from .grid import Grid
from .structure import Structure


@dataclass(frozen=True, eq=False)
class PairChunk:
    voxel: np.ndarray      # (M,) flat voxel index
    atom: np.ndarray       # (M,) atom index
    weight: np.ndarray     # (M,)
    distance: np.ndarray   # (M,) Angstrom, voxel to this atom's nucleus image
    direction: np.ndarray  # (M, 3) unit vector nucleus -> voxel


class Partition:
    scheme = "abstract"

    def __init__(self, n_atoms: int):
        self.n_atoms = n_atoms

    def pairs(self, chunk: int = 1 << 18) -> Iterator[PairChunk]:
        raise NotImplementedError

    def site_sum(self, f) -> np.ndarray:
        """sum_k w_i(k) f_k for every atom i."""
        f = np.asarray(f, float).ravel()
        out = np.zeros(self.n_atoms)
        for p in self.pairs():
            out += np.bincount(p.atom, weights=p.weight * f[p.voxel], minlength=self.n_atoms)
        return out

    def site_count(self) -> np.ndarray:
        """Effective number of voxels per atom, sum_k w_i(k)."""
        out = np.zeros(self.n_atoms)
        for p in self.pairs():
            out += np.bincount(p.atom, weights=p.weight, minlength=self.n_atoms)
        return out


class HardPartition(Partition):
    """Each voxel belongs wholly to one atom (nearest-atom or power diagram)."""

    def __init__(self, assignment: NearestAtom, n_atoms: int, scheme: str = "nearest"):
        super().__init__(n_atoms)
        self.scheme = scheme
        self.assignment = assignment
        self.labels = assignment.atom_index.ravel()

    def pairs(self, chunk: int = 1 << 20) -> Iterator[PairChunk]:
        dist = self.assignment.distance.ravel()
        direction = self.assignment.direction.reshape(-1, 3)
        n = dist.size
        for s in range(0, n, chunk):
            v = np.arange(s, min(s + chunk, n))
            yield PairChunk(v, self.labels[v], np.ones(v.size), dist[v], direction[v])

    def site_sum(self, f) -> np.ndarray:
        return np.bincount(self.labels, weights=np.asarray(f, float).ravel(),
                           minlength=self.n_atoms)

    def site_count(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=self.n_atoms).astype(float)


def _becke_step(mu):
    for _ in range(3):
        mu = mu * (1.5 - 0.5 * mu * mu)   # not mu ** 3: the generic power is ~20x slower
    return 0.5 * (1.0 - mu)


class BeckePartition(Partition):
    """Becke weights for each voxel's ``cells`` nearest atom images, with each
    cell product over its ``k`` nearest images (see module docstring)."""

    scheme = "becke"

    def __init__(self, grid: Grid, structure: Structure, k: int = 60, cells: int = 8,
                 radii: Optional[np.ndarray] = None, min_weight: float = 1e-10):
        super().__init__(structure.n_atoms)
        if cells < 1 or k < cells:
            raise ValueError(f"need 1 <= cells <= k, got cells={cells}, k={k}")
        self.grid = grid
        self.structure = structure
        self.k = int(k)
        self.cells = int(cells)
        self.min_weight = min_weight
        self.radii = None if radii is None else np.asarray(radii, float)
        self._reps = (1, 1, 1)
        self._build()

    def _build(self):
        self._pts, self._owner = _image_points(self.structure, self._reps)
        self._tree = cKDTree(self._pts)

    def _query(self, x):
        """k nearest images, widening the image range until provably complete."""
        k = min(self.k, self._pts.shape[0])
        while True:
            d, idx = self._tree.query(x, k=k, workers=-1)
            d, idx = d.reshape(len(x), -1), idx.reshape(len(x), -1)
            needed = _reps_for(float(d[:, -1].max()), self.structure)
            if all(n <= r for n, r in zip(needed, self._reps)):
                return d, idx
            self._reps = tuple(max(n, r) for n, r in zip(needed, self._reps))
            self._build()
            k = min(self.k, self._pts.shape[0])

    def _adjust(self, mu, owners, nc):
        # Becke's heteronuclear adjustment: nu = mu + a (1 - mu^2)
        if self.radii is None:
            return mu
        R = self.radii[owners]
        chi = R[:, :nc, None] / R[:, None, :]
        u = (chi - 1.0) / (chi + 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(np.abs(u) > 1e-12, u / (u * u - 1.0), 0.0)
        a = np.clip(a, -0.5, 0.5)
        return mu + a * (1.0 - mu * mu)

    def pairs(self, chunk: int = 1 << 13) -> Iterator[PairChunk]:
        x_all = self.grid.cart_coords().reshape(-1, 3)
        for s in range(0, x_all.shape[0], chunk):
            x = x_all[s:s + chunk]
            d, idx = self._query(x)
            P = self._pts[idx]                                 # (M, K, 3)
            owners = self._owner[idx]                          # (M, K)
            nc = min(self.cells, d.shape[1])
            R = np.linalg.norm(P[:, :nc, None, :] - P[:, None, :, :], axis=-1)  # (M, nc, K)
            with np.errstate(divide="ignore", invalid="ignore"):
                mu = (d[:, :nc, None] - d[:, None, :]) / R
            mu = self._adjust(np.nan_to_num(mu), owners, nc)
            sij = _becke_step(mu)
            sij[:, np.arange(nc), np.arange(nc)] = 1.0
            cell = sij.prod(axis=2)                            # (M, nc)
            w = cell / cell.sum(axis=1, keepdims=True)
            keep = w > self.min_weight
            m, j = np.nonzero(keep)
            diff = x[m] - P[m, j]
            dist = d[m, j]
            with np.errstate(invalid="ignore", divide="ignore"):
                direction = np.where(dist[:, None] > 1e-12, diff / dist[:, None], 0.0)
            yield PairChunk(s + m, owners[m, j], w[m, j], dist, direction)


TABLE_POINTS = 1 << 15


class HirshfeldPartition(Partition):
    """Hirshfeld weights from spherical free-atom densities (entry 101).

    ``radial`` maps element -> (r in Angstrom, density). Every atom image
    within ``r_cut`` of a voxel contributes. The free-atom densities decay
    exponentially, so the weights converge quickly with r_cut -- measured
    on fcc FeCoNiCr, max weight error 5e-3 at 3.5 A, 7e-5 at 5.6 A, 9e-6 at
    6.5 A (the default) -- unlike Becke's products (module docstring).

    The grid is processed in blocks of voxels: candidate atom images are
    found once per block (within r_cut plus the block radius), then all
    voxel-image distances are computed densely. A voxel farther than r_cut
    from every image goes to its nearest one.
    """

    scheme = "hirshfeld"

    def __init__(self, grid: Grid, structure: Structure, radial: dict, r_cut: float = 6.5,
                 block: int = 8, min_weight: float = 1e-10):
        super().__init__(structure.n_atoms)
        self.grid = grid
        self.structure = structure
        self.r_cut = float(r_cut)
        self.block = int(block)
        self.min_weight = min_weight
        self._elem = structure.element_index
        # densities tabulated on a uniform distance grid for fast lookup
        # (log-log interpolation of the radial data onto it, once)
        self._step = self.r_cut / (TABLE_POINTS - 1)
        d_tab = np.arange(TABLE_POINTS) * self._step
        self._tables = []
        for e in structure.elements:
            r, n = radial[e]
            r, n = np.asarray(r, float), np.asarray(n, float)
            keep = (r > 0) & (n > 0)
            lr, ln = np.log(r[keep]), np.log(n[keep])
            tab = np.exp(np.interp(np.log(np.maximum(d_tab, r[keep][0])), lr, ln, right=-np.inf))
            self._tables.append(np.append(tab, 0.0))       # guard slot for d >= r_cut
        step = grid.lattice / np.array(grid.shape)[:, None]
        corners = np.array(list(itertools.product((0, 1), repeat=3))) * (self.block - 1)
        cvec = corners @ step
        self._block_radius = float(np.linalg.norm(cvec - cvec.mean(0), axis=1).max())
        reach = self.r_cut + self._block_radius
        self._pts, self._owner = _image_points(structure, _reps_for(reach, structure))
        self._tree = cKDTree(self._pts)

    def density(self, element_index: int, d: np.ndarray) -> np.ndarray:
        """Free-atom density at distances ``d`` (0 beyond r_cut), linear in the table."""
        tab = self._tables[element_index]
        t = np.minimum(d / self._step, TABLE_POINTS - 1)
        i = t.astype(np.int64)
        f = t - i
        return tab[i] * (1.0 - f) + tab[i + 1] * f

    def pairs(self, chunk=None) -> Iterator[PairChunk]:
        shape, b = self.grid.shape, self.block
        lat = self.grid.lattice
        n1, n2, n3 = shape
        for i0 in range(0, n1, b):
            for j0 in range(0, n2, b):
                for k0 in range(0, n3, b):
                    ii, jj, kk = (np.arange(i0, min(i0 + b, n1)), np.arange(j0, min(j0 + b, n2)),
                                  np.arange(k0, min(k0 + b, n3)))
                    I, J, K = np.meshgrid(ii, jj, kk, indexing="ij")
                    frac = np.stack([I / n1, J / n2, K / n3], axis=-1).reshape(-1, 3)
                    x = frac @ lat
                    cand = np.array(self._tree.query_ball_point(x.mean(0), self.r_cut + self._block_radius),
                                    dtype=int)
                    voxel = np.ravel_multi_index((I.ravel(), J.ravel(), K.ravel()), shape)
                    P = self._pts[cand]
                    diff = x[:, None, :] - P[None, :, :]                  # (M, C, 3)
                    d = np.linalg.norm(diff, axis=-1)
                    elem = self._elem[self._owner[cand]]
                    dens = np.zeros_like(d)
                    for e in np.unique(elem):
                        cols = elem == e
                        dens[:, cols] = self.density(e, d[:, cols])
                    total = dens.sum(axis=1)
                    empty = total <= 0
                    if empty.any():                                    # nearest image takes it
                        nearest = np.argmin(d[empty], axis=1)
                        dens[np.flatnonzero(empty), nearest] = 1.0
                        total[empty] = 1.0
                    w = dens / total[:, None]
                    m, c = np.nonzero(w > self.min_weight)
                    dist = d[m, c]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        direction = np.where(dist[:, None] > 1e-12, diff[m, c] / dist[:, None], 0.0)
                    yield PairChunk(voxel[m], self._owner[cand[c]], w[m, c], dist, direction)
