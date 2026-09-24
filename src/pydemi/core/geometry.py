"""
pydemi.core.geometry
====================
The geometry pass every descriptor shares (spec §5), shells, and the
periodic-image machinery behind the promolecule and the partitions.

Nearest-atom pass
-----------------
For every voxel k: the distance r_k to the nearest nucleus under the
minimum-image convention, the unit vector u_k from that nucleus to the
voxel (Cartesian), and the index i_k of that atom.

Atom images are loaded into a ``scipy.spatial.cKDTree`` and each voxel's
Cartesian position is queried for the single nearest image, which maps back
to its parent atom. Instead of a fixed 3x3x3 supercell -- which can miss the
nearest image in a strongly sheared cell -- the image range starts at one
cell in each direction and is widened until it provably covers every voxel:
a displacement of length d changes fractional coordinate a by at most
d |column a of A^{-1}|, so images out to the largest nearest distance found
are enough.

Power diagram
-------------
:func:`assign_atoms` with ``power_radii`` assigns voxels to
argmin_i |x - R_i|^2 - R_i^2 (radius-weighted Voronoi). Lifting every image
to 4-D with coordinate sqrt(C - R_i^2), C = max R^2, turns that into an
ordinary nearest-neighbour query, and the lifted distance bounds the
Euclidean one, so the same image-range argument holds.

Shells
------
core r <= c1, bond c1 < r <= c2, interstitial r > c2, with c1 = 0.8 and
c2 = 1.5 Angstrom by default, or c = s * R_e with R_e the covalent radius of
the nearest atom's element (radius-scaled mode).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from ..constants import SHELL_C1, SHELL_C2
from ..data import covalent_radius
from ..io.base import Lattice, Structure, VolumetricData
from .grid import cart_coords

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, eq=False)
class NearestAtom:
    """Result of the geometry pass: all arrays are on the density grid."""

    distance: F64        # (n1, n2, n3) Angstrom
    direction: F64       # (n1, n2, n3, 3) unit vector nucleus -> voxel; 0 at a nucleus
    atom_index: I64      # (n1, n2, n3)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.distance.shape)


def image_points(structure: Structure, reps: Sequence[int]) -> tuple[F64, I64, I64]:
    """Cartesian positions of every atom image with lattice shifts in [-reps, reps].

    Returns (points (M, 3), owner atom (M,), integer shift (M, 3)); atoms are
    first wrapped into [0, 1).
    """
    frac = structure.frac_coords % 1.0
    shifts = np.array(list(itertools.product(*[range(-int(r), int(r) + 1) for r in reps])),
                      dtype=np.int64)
    pts = (frac[None, :, :] + shifts[:, None, :]).reshape(-1, 3) @ structure.lattice.matrix
    owner = np.tile(np.arange(structure.n_atoms, dtype=np.int64), len(shifts))
    shift = np.repeat(shifts, structure.n_atoms, axis=0)
    return np.asarray(pts, dtype=np.float64), owner, shift


def reps_for(distance: float, lattice: Lattice) -> tuple[int, int, int]:
    """Image range that contains every image within ``distance`` of any point in the cell."""
    h = np.linalg.norm(lattice.inverse, axis=0)
    r = [int(np.ceil(distance * hj)) + 1 for hj in h]
    return (r[0], r[1], r[2])


def assign_atoms(shape: Sequence[int], structure: Structure,
                 power_radii: Optional[F64] = None, chunk: int = 1 << 20) -> NearestAtom:
    """Assign every voxel to an atom: nearest (``power_radii=None``) or power diagram.

    Distance and direction are measured to the assigned atom's image.
    """
    n = structure.n_atoms
    if power_radii is None:
        lift = np.zeros(n)
    else:
        w = np.asarray(power_radii, dtype=np.float64) ** 2
        if w.shape != (n,):
            raise ValueError(f"need {n} power radii, got shape {w.shape}")
        lift = np.sqrt(w.max() - w)
    x = cart_coords(shape, structure.lattice).reshape(-1, 3)
    x4 = np.hstack([x, np.zeros((len(x), 1))])
    reps = (1, 1, 1)
    while True:
        pts, owner, _ = image_points(structure, reps)
        tree = cKDTree(np.hstack([pts, lift[owner][:, None]]))
        dist = np.empty(len(x))
        idx = np.empty(len(x), dtype=np.int64)
        for s in range(0, len(x), chunk):
            d, i = tree.query(x4[s:s + chunk])
            dist[s:s + chunk], idx[s:s + chunk] = d, i
        needed = reps_for(float(dist.max()), structure.lattice)
        if all(a <= b for a, b in zip(needed, reps)):
            break
        reps = (max(needed[0], reps[0]), max(needed[1], reps[1]), max(needed[2], reps[2]))
    diff = x - pts[idx]
    r = np.linalg.norm(diff, axis=1)
    safe = np.where(r > 1e-12, r, 1.0)
    u = np.where(r[:, None] > 1e-12, diff / safe[:, None], 0.0)
    s3 = tuple(int(m) for m in shape)
    return NearestAtom(r.reshape(s3), u.reshape(s3 + (3,)), owner[idx].reshape(s3))


def nearest_atom(shape: Sequence[int], structure: Structure) -> NearestAtom:
    """The geometry pass: nearest-nucleus distance, direction and index for every voxel."""
    return assign_atoms(shape, structure, None)


# ----------------------------------------------------------------------
# shells
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Shells:
    """Radial shells about the nearest nucleus.

    Absolute mode (default): c1, c2 in Angstrom. Radius-scaled mode
    (``scaled=True``): c1, c2 are multiples of the covalent radius of the
    nearest atom's element.
    """

    c1: float = SHELL_C1
    c2: float = SHELL_C2
    scaled: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.c1 < self.c2):
            raise ValueError(f"need 0 <= c1 < c2, got c1={self.c1}, c2={self.c2}")

    def atom_cutoffs(self, structure: Structure) -> tuple[F64, F64]:
        """(c1_i, c2_i) per atom, Angstrom."""
        if not self.scaled:
            ones = np.ones(structure.n_atoms)
            return self.c1 * ones, self.c2 * ones
        R = np.array([covalent_radius(s) for s in structure.species], dtype=np.float64)
        return self.c1 * R, self.c2 * R

    def key(self) -> tuple[float, float, bool]:
        return (float(self.c1), float(self.c2), bool(self.scaled))


@dataclass(frozen=True, eq=False)
class ShellMasks:
    core: BoolArray
    bond: BoolArray
    interstitial: BoolArray


def shell_masks(geometry: NearestAtom, structure: Structure, shells: Shells) -> ShellMasks:
    c1, c2 = shells.atom_cutoffs(structure)
    r = geometry.distance
    c1k, c2k = c1[geometry.atom_index], c2[geometry.atom_index]
    return ShellMasks(core=r <= c1k, bond=(r > c1k) & (r <= c2k), interstitial=r > c2k)


# ----------------------------------------------------------------------
# image sums over voxel blocks (promolecule, Hirshfeld)
# ----------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class ImageBlock:
    """One block of voxels with every atom image within the cutoff of any of them."""

    voxel: I64           # (M,) flat voxel indices
    points: F64          # (M, 3) Cartesian voxel positions
    diff: F64            # (M, C, 3) voxel - image
    distance: F64        # (M, C)
    owner: I64           # (C,) atom of each image


def image_blocks(shape: Sequence[int], structure: Structure, cutoff: float,
                 block: int = 8) -> Iterator[ImageBlock]:
    """Blocks of ``block``^3 voxels with the atom images that can lie within ``cutoff``.

    Candidates are found once per block (cutoff plus the block's radius),
    then all voxel-image distances in the block are computed densely.
    """
    n1, n2, n3 = (int(m) for m in shape)
    lat = structure.lattice.matrix
    step = lat / np.array([n1, n2, n3], dtype=np.float64)[:, None]
    corners = np.array(list(itertools.product((0, 1), repeat=3)), dtype=np.float64) * (block - 1)
    cvec = corners @ step
    radius = float(np.linalg.norm(cvec - cvec.mean(0), axis=1).max())
    reach = cutoff + radius
    pts, owner, _ = image_points(structure, reps_for(reach, structure.lattice))
    tree = cKDTree(pts)
    for i0 in range(0, n1, block):
        for j0 in range(0, n2, block):
            for k0 in range(0, n3, block):
                I, J, K = np.meshgrid(np.arange(i0, min(i0 + block, n1)),
                                      np.arange(j0, min(j0 + block, n2)),
                                      np.arange(k0, min(k0 + block, n3)), indexing="ij")
                frac = np.stack([I / n1, J / n2, K / n3], axis=-1).reshape(-1, 3)
                x = frac @ lat
                cand = np.array(tree.query_ball_point(x.mean(0), reach), dtype=np.int64)
                voxel = np.ravel_multi_index((I.ravel(), J.ravel(), K.ravel()), (n1, n2, n3))
                diff = x[:, None, :] - pts[cand][None, :, :]
                yield ImageBlock(np.asarray(voxel, dtype=np.int64), x, diff,
                                 np.linalg.norm(diff, axis=-1), owner[cand])


# ----------------------------------------------------------------------
# cached on VolumetricData (computed once per structure, spec §5, §13)
# ----------------------------------------------------------------------

def geometry_of(vd: VolumetricData) -> NearestAtom:
    """The nearest-atom pass for ``vd``'s grid, cached on ``vd``."""
    key = ("geometry",)
    if key not in vd.cache:
        vd.cache[key] = nearest_atom(vd.shape, vd.structure)
    result: NearestAtom = vd.cache[key]
    return result


def shells_of(vd: VolumetricData, shells: Shells) -> ShellMasks:
    """Shell masks for ``vd`` and ``shells``, cached on ``vd``."""
    key = ("shells", shells.key())
    if key not in vd.cache:
        vd.cache[key] = shell_masks(geometry_of(vd), vd.structure, shells)
    result: ShellMasks = vd.cache[key]
    return result
