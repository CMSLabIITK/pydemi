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

Ties
----
Voxels exactly equidistant from two or more atom images -- common for atoms on
high-symmetry grid points -- are resolved by a geometric rule: among the
images within GEOMETRY_EPS of the nearest, the one with the lexicographically
largest fractional displacement (voxel - image) wins. The rule does not depend
on atom numbering or image enumeration, so it makes the same physical choice
in a cell, its supercell, a translated and a rotated copy -- which the
invariance requirements of the specification need.

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

from ..constants import GEOMETRY_EPS, SHELL_C1, SHELL_C2
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


def _rank(dk: F64, disp: F64, n_pick: int, eps: float) -> I64:
    """Column indices of the ``n_pick`` first candidates in (distance, -displacement) order.

    Candidates within ``eps`` of the smallest remaining distance are tied and
    ordered by lexicographically largest fractional displacement.
    """
    rows = np.arange(dk.shape[0])
    remaining = np.ones(dk.shape, dtype=bool)
    out = np.empty((dk.shape[0], n_pick), dtype=np.int64)
    for pick in range(n_pick):
        d = np.where(remaining, dk, np.inf)
        cand = remaining & (d <= d.min(axis=1, keepdims=True) + eps)
        for c in range(3):
            v = np.where(cand, disp[:, :, c], -np.inf)
            cand &= v >= v.max(axis=1, keepdims=True) - 1e-9
        choice = np.argmax(cand, axis=1)
        out[:, pick] = choice
        remaining[rows, choice] = False
    return out


def _nearest_images(tree: cKDTree, x: F64, pts4: F64, lattice: Lattice, n_pick: int,
                    chunk: int = 1 << 20) -> tuple[F64, I64]:
    """Distances and indices of the ``n_pick`` nearest images, ties broken geometrically."""
    k0 = min(n_pick + 1, len(pts4))
    dist = np.empty((len(x), n_pick))
    idx = np.empty((len(x), n_pick), dtype=np.int64)
    for s in range(0, len(x), chunk):
        d, i = tree.query(x[s:s + chunk], k=k0)
        d, i = np.asarray(d).reshape(-1, k0), np.asarray(i).reshape(-1, k0)
        tied: NDArray[np.bool_] = np.zeros(len(d), dtype=bool)
        if k0 > n_pick:
            tied = d[:, n_pick] - d[:, n_pick - 1] <= GEOMETRY_EPS
        for p in range(1, n_pick):
            tied |= d[:, p] - d[:, p - 1] <= GEOMETRY_EPS
        dist[s:s + chunk], idx[s:s + chunk] = d[:, :n_pick], i[:, :n_pick]
        t = np.flatnonzero(tied)
        if t.size:
            k = min(max(2 * k0, 8), len(pts4))
            while True:
                dk, ik = tree.query(x[s + t], k=k)
                dk, ik = np.asarray(dk).reshape(-1, k), np.asarray(ik).reshape(-1, k)
                if k >= len(pts4) or np.all(dk[:, -1] - dk[:, n_pick - 1] > GEOMETRY_EPS):
                    break
                k = min(2 * k, len(pts4))
            disp = (x[s + t][:, None, :3] - pts4[ik][:, :, :3]) @ lattice.inverse
            order = _rank(dk, disp, n_pick, GEOMETRY_EPS)
            rows = np.arange(len(t))[:, None]
            dist[s + t], idx[s + t] = dk[rows, order], ik[rows, order]
    return dist, idx


def assign_atoms(shape: Sequence[int], structure: Structure,
                 power_radii: Optional[F64] = None) -> NearestAtom:
    """Assign every voxel to an atom: nearest (``power_radii=None``) or power diagram.

    Distance and direction are measured to the assigned atom's image; exact
    ties are resolved geometrically (module docstring).
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
        pts4 = np.hstack([pts, lift[owner][:, None]])
        tree = cKDTree(pts4)
        dist, idx = _nearest_images(tree, x4, pts4, structure.lattice, 1)
        needed = reps_for(float(dist.max()), structure.lattice)
        if all(a <= b for a, b in zip(needed, reps)):
            break
        reps = (max(needed[0], reps[0]), max(needed[1], reps[1]), max(needed[2], reps[2]))
    best = idx[:, 0]
    diff = x - pts[best]
    r = np.linalg.norm(diff, axis=1)
    safe = np.where(r > 1e-12, r, 1.0)
    u = np.where(r[:, None] > 1e-12, diff / safe[:, None], 0.0)
    s3 = tuple(int(m) for m in shape)
    return NearestAtom(r.reshape(s3), u.reshape(s3 + (3,)), owner[best].reshape(s3))


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
    """Shells about the nearest nucleus; boundaries tested with GEOMETRY_EPS tolerance."""
    c1, c2 = shells.atom_cutoffs(structure)
    r = geometry.distance
    c1k = c1[geometry.atom_index] + GEOMETRY_EPS
    c2k = c2[geometry.atom_index] + GEOMETRY_EPS
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


# ----------------------------------------------------------------------
# nearest-neighbour bond census
# ----------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class BondCensus:
    """First-shell neighbour pairs: atom j (any image, shift t) of atom i.

    j is a neighbour of i when |R_j + t - R_i| <= (1 + tol) d_i, with d_i the
    distance from i to its closest neighbour -- a per-atom first shell, the
    unambiguous convention in multi-element cells. Each pair is counted once.
    """

    i: I64
    j: I64
    shift: I64          # (n_bonds, 3) lattice translation of j's image (stored coordinates)
    length: F64
    shift_wrapped: I64  # the same translation for positions wrapped into [0, 1)

    def midpoints_frac(self, structure: Structure) -> F64:
        f = structure.frac_coords
        return np.asarray(0.5 * (f[self.i] + f[self.j] + self.shift), dtype=np.float64)


def bond_census(structure: Structure, tol: float) -> BondCensus:
    n = structure.n_atoms
    frac = structure.frac_coords
    wrap = np.floor(frac)
    centres = (frac - wrap) @ structure.lattice.matrix
    reps = (1, 1, 1)
    while True:
        pts, owner, shift = image_points(structure, reps)
        tree = cKDTree(pts)
        d2, _ = tree.query(centres, k=2)
        d_min = np.asarray(d2)[:, 1]
        needed = reps_for((1.0 + tol) * float(d_min.max()), structure.lattice)
        if all(a <= b for a, b in zip(needed, reps)):
            break
        reps = (max(needed[0], reps[0]), max(needed[1], reps[1]), max(needed[2], reps[2]))
    pairs: dict[tuple[int, int, tuple[int, int, int]], float] = {}
    for i in range(n):
        for k in tree.query_ball_point(centres[i], (1.0 + tol) * d_min[i] + 1e-9):
            j = int(owner[k])
            t = tuple(int(x) for x in shift[k])
            length = float(np.linalg.norm(pts[k] - centres[i]))
            if length < 1e-9:
                continue
            neg = (-t[0], -t[1], -t[2])
            key = (i, j, (t[0], t[1], t[2])) if (i, j, t) <= (j, i, neg) else (j, i, neg)
            pairs[key] = length
    keys = sorted(pairs)
    if not keys:
        z = np.zeros(0, dtype=np.int64)
        empty = np.zeros((0, 3), dtype=np.int64)
        return BondCensus(z, z, empty, np.zeros(0), empty)
    ii = np.array([k[0] for k in keys], dtype=np.int64)
    jj = np.array([k[1] for k in keys], dtype=np.int64)
    # shifts are relative to the wrapped positions; express them for the stored coordinates
    tw = np.array([k[2] for k in keys], dtype=np.int64)
    tt = (tw - wrap[jj] + wrap[ii]).astype(np.int64)
    return BondCensus(ii, jj, tt, np.array([pairs[k] for k in keys]), tw)


def bond_census_of(vd: VolumetricData, tol: float) -> BondCensus:
    key = ("bond_census", float(tol))
    if key not in vd.cache:
        vd.cache[key] = bond_census(vd.structure, tol)
    out: BondCensus = vd.cache[key]
    return out


def pair_regions(shape: Sequence[int], structure: Structure) -> tuple[I64, I64, I64]:
    """Second-order Voronoi assignment: for every voxel, its two nearest atom images.

    Returns (atom_a, atom_b, relative shift t (N, 3)) with the pair written
    canonically as in :class:`BondCensus` (wrapped positions): the region of
    pair (i, j, t) holds the voxels whose nearest two images are atom i and
    atom j shifted by t. These regions tile the cell; ties are resolved as in
    the geometry pass.
    """
    x = cart_coords(shape, structure.lattice).reshape(-1, 3)
    reps = (1, 1, 1)
    while True:
        pts, owner, shift = image_points(structure, reps)
        tree = cKDTree(pts)
        d, idx = _nearest_images(tree, x, pts, structure.lattice, 2)
        needed = reps_for(float(d[:, 1].max()), structure.lattice)
        if all(a <= b for a, b in zip(needed, reps)):
            break
        reps = (max(needed[0], reps[0]), max(needed[1], reps[1]), max(needed[2], reps[2]))
    a, b = owner[idx[:, 0]], owner[idx[:, 1]]
    t = shift[idx[:, 1]] - shift[idx[:, 0]]
    swap = (b < a) | ((b == a) & _lex_less(-t, t))
    i = np.where(swap, b, a)
    j = np.where(swap, a, b)
    t = np.where(swap[:, None], -t, t)
    return i, j, t


def _lex_less(u: I64, v: I64) -> NDArray[np.bool_]:
    """Row-wise lexicographic u < v."""
    out = np.zeros(len(u), dtype=bool)
    decided = np.zeros(len(u), dtype=bool)
    for c in range(u.shape[1]):
        lt = (u[:, c] < v[:, c]) & ~decided
        gt = (u[:, c] > v[:, c]) & ~decided
        out |= lt
        decided |= lt | gt
    return out
