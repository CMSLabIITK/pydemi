"""
pydemi.geometry
---------------
The single geometry pass every descriptor shares: for each voxel k, the
distance r_k to the nearest nucleus (minimum image), the unit vector r_hat_k
from that nucleus to the voxel, and the index i(k) of that atom.

Minimum image
-------------
Wrapping fractional differences with ``round`` is only the true minimum
image for near-orthogonal cells; in a strongly skewed cell it can miss a
closer image. Here periodic images of the atoms are loaded into a KD-tree
and the image range is widened until it provably covers every voxel's
nearest atom (the largest nearest-distance found bounds how far any
voxel can be from its nearest atom).

Power diagram
-------------
:func:`assign_atoms` with ``power_radii`` assigns each voxel to
argmin_i |x - R_i|^2 - r_i^2 (radius-weighted Voronoi, reference entry 99).
Lifting every atom image to 4-D with coordinate sqrt(C - r_i^2), C = max r^2,
turns that into an ordinary nearest-neighbour query:
|(x, 0) - (R_i, sqrt(C - r_i^2))|^2 = |x - R_i|^2 - r_i^2 + C. The lifted
distance also bounds the Euclidean one, so the same image-range argument
holds. Distances and directions are then measured to the assigned atom,
which need not be the nearest one.
"""

from dataclasses import dataclass
import itertools

import numpy as np
from scipy.spatial import cKDTree

from .grid import Grid
from .structure import Structure


@dataclass(frozen=True, eq=False)
class NearestAtom:
    distance: np.ndarray    # (nx, ny, nz) Angstrom
    direction: np.ndarray   # (nx, ny, nz, 3) unit vector nucleus -> voxel; 0 at a nucleus
    atom_index: np.ndarray  # (nx, ny, nz) int, index into structure.species

    @property
    def shape(self) -> tuple:
        return self.distance.shape


def _image_points(structure: Structure, reps):
    frac = structure.frac_coords % 1.0
    shifts = np.array(list(itertools.product(*[range(-r, r + 1) for r in reps])),
                      dtype=float)
    pts = (frac[None, :, :] + shifts[:, None, :]).reshape(-1, 3) @ structure.lattice
    owner = np.tile(np.arange(structure.n_atoms), len(shifts))
    return pts, owner


def _reps_for(distance: float, structure: Structure):
    # a displacement of length d changes fractional coordinate j by at most
    # d * |column j of A^{-1}|; atoms and voxels both sit in [0, 1)
    h = np.linalg.norm(structure.inv_lattice, axis=0)
    return tuple(int(np.ceil(distance * hj)) + 1 for hj in h)


def assign_atoms(grid: Grid, structure: Structure, power_radii=None,
                 chunk: int = 1 << 20, workers: int = -1) -> NearestAtom:
    """Assign every voxel of ``grid`` to an atom.

    ``power_radii=None`` gives the nearest atom; per-atom radii (Angstrom,
    shape (n_atoms,)) give the power diagram. Returns the distance,
    direction and index of the assigned atom.
    """
    if not np.allclose(grid.lattice, structure.lattice):
        raise ValueError("grid and structure lattices differ")

    if power_radii is None:
        lift = np.zeros(structure.n_atoms)
    else:
        w = np.asarray(power_radii, float) ** 2
        if w.shape != (structure.n_atoms,):
            raise ValueError(f"need {structure.n_atoms} power radii, got {w.shape}")
        lift = np.sqrt(w.max() - w)

    x = grid.cart_coords().reshape(-1, 3)
    x4 = np.hstack([x, np.zeros((x.shape[0], 1))])
    reps = (1, 1, 1)
    while True:
        pts, owner = _image_points(structure, reps)
        tree = cKDTree(np.hstack([pts, lift[owner][:, None]]))
        dist = np.empty(x.shape[0])
        idx = np.empty(x.shape[0], dtype=np.int64)
        for s in range(0, x.shape[0], chunk):
            dist[s:s + chunk], idx[s:s + chunk] = tree.query(x4[s:s + chunk],
                                                             workers=workers)
        needed = _reps_for(float(dist.max()), structure)
        if all(n <= r for n, r in zip(needed, reps)):
            break
        reps = tuple(max(n, r) for n, r in zip(needed, reps))

    diff = x - pts[idx]
    dist = np.linalg.norm(diff, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        direction = np.where(dist[:, None] > 1e-12, diff / dist[:, None], 0.0)

    shape = grid.shape
    return NearestAtom(
        distance=dist.reshape(shape),
        direction=direction.reshape(shape + (3,)),
        atom_index=owner[idx].reshape(shape),
    )


def nearest_atom(grid: Grid, structure: Structure, **kwargs) -> NearestAtom:
    """Nearest-atom distance, direction and index for every voxel of ``grid``."""
    return assign_atoms(grid, structure, None, **kwargs)
