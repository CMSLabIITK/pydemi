"""
pydemi.core.geometry
====================
The geometry pass every descriptor shares, the regions, and the
periodic-image machinery behind the partitions and the bond census.

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

Regions (sphere rule)
---------------------
Around every nucleus i a core sphere of radius a R_i and an outer sphere of
radius b R_i, with R_i the covalent radius. A voxel is core if it lies in any
core sphere, bond if it lies in any outer sphere but in no core sphere, and
interstitial otherwise. With s = min_i r_i / R_i: core s <= a, bond
a < s <= b, interstitial s > b. This is the same as testing r against a R_i
and b R_i for the atom with the smallest r_i / R_i (:func:`sphere_assignment`),
so the masks and the per-site sums stay a lookup on one assignment. The
default factors SPHERE_A = 0.79 and SPHERE_B = 1.14 are the 1st and 99th
percentiles of d / (R_A + R_B) over the nearest neighbour of every atom of the
6,059-structure dataset: the core spheres of bonded atoms almost never overlap,
and almost no bond has interstitial voxels on its line.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from ..constants import GEOMETRY_EPS, SPHERE_A, SPHERE_B
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


def _nearest_images(tree: cKDTree, x: F64, pts: F64, lattice: Lattice, n_pick: int,
                    chunk: int = 1 << 20) -> tuple[F64, I64]:
    """Distances and indices of the ``n_pick`` nearest images, ties broken geometrically."""
    k0 = min(n_pick + 1, len(pts))
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
            k = min(max(2 * k0, 8), len(pts))
            while True:
                dk, ik = tree.query(x[s + t], k=k)
                dk, ik = np.asarray(dk).reshape(-1, k), np.asarray(ik).reshape(-1, k)
                if k >= len(pts) or np.all(dk[:, -1] - dk[:, n_pick - 1] > GEOMETRY_EPS):
                    break
                k = min(2 * k, len(pts))
            disp = (x[s + t][:, None, :] - pts[ik]) @ lattice.inverse
            order = _rank(dk, disp, n_pick, GEOMETRY_EPS)
            rows = np.arange(len(t))[:, None]
            dist[s + t], idx[s + t] = dk[rows, order], ik[rows, order]
    return dist, idx


def nearest_atom(shape: Sequence[int], structure: Structure) -> NearestAtom:
    """The geometry pass: nearest-nucleus distance, direction and index for every voxel.

    Distance and direction are measured to the nearest atom's image; exact
    ties are resolved geometrically (module docstring).
    """
    x = cart_coords(shape, structure.lattice).reshape(-1, 3)
    reps = (1, 1, 1)
    while True:
        pts, owner, _ = image_points(structure, reps)
        tree = cKDTree(pts)
        dist, idx = _nearest_images(tree, x, pts, structure.lattice, 1)
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


def sphere_assignment(shape: Sequence[int], structure: Structure) -> NearestAtom:
    """Assign every voxel to the atom with the smallest r_i / R_i (R_i the covalent radius).

    R_i is the same for every atom of one element, so the minimum is taken over
    one nearest-image search per element. Ties within an element follow the
    geometric rule of the nearest-atom pass; ties between elements (r_i / R_i
    equal within GEOMETRY_EPS / 10) go to the element first in alphabetical
    order. Distance and direction are measured to the assigned atom's image.
    """
    x = cart_coords(shape, structure.lattice).reshape(-1, 3)
    species = [str(e) for e in structure.species]
    best_s = np.full(len(x), np.inf)
    best_atom = np.zeros(len(x), dtype=np.int64)
    best_pt = np.zeros((len(x), 3))
    for el in sorted(set(species)):
        R = covalent_radius(el)
        sub = np.array([k for k, e in enumerate(species) if e == el], dtype=np.int64)
        reps = (1, 1, 1)
        while True:
            pts, owner, _ = image_points(structure, reps)
            keep = np.isin(owner, sub)
            p, o = pts[keep], owner[keep]
            dist, idx = _nearest_images(cKDTree(p), x, p, structure.lattice, 1)
            needed = reps_for(float(dist.max()), structure.lattice)
            if all(a <= b for a, b in zip(needed, reps)):
                break
            reps = (max(needed[0], reps[0]), max(needed[1], reps[1]), max(needed[2], reps[2]))
        s = dist[:, 0] / R
        better = s < best_s - GEOMETRY_EPS / 10
        best_s[better] = s[better]
        best_atom[better] = o[idx[better, 0]]
        best_pt[better] = p[idx[better, 0]]
    diff = x - best_pt
    r = np.linalg.norm(diff, axis=1)
    safe = np.where(r > 1e-12, r, 1.0)
    u = np.where(r[:, None] > 1e-12, diff / safe[:, None], 0.0)
    s3 = tuple(int(m) for m in shape)
    return NearestAtom(r.reshape(s3), u.reshape(s3 + (3,)), best_atom.reshape(s3))


# ----------------------------------------------------------------------
# regions (sphere rule)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Shells:
    """The core, bond and interstitial regions of the sphere rule (module docstring).

    ``a``, ``b``: radii of the core and the outer sphere of every atom, as
    multiples of its covalent radius.
    """

    a: float = SPHERE_A
    b: float = SPHERE_B

    def __post_init__(self) -> None:
        if not (0.0 <= self.a < self.b):
            raise ValueError(f"need 0 <= a < b, got a={self.a}, b={self.b}")

    def atom_cutoffs(self, structure: Structure) -> tuple[F64, F64]:
        """(a R_i, b R_i) per atom, Angstrom."""
        R = np.array([covalent_radius(s) for s in structure.species], dtype=np.float64)
        return self.a * R, self.b * R

    def key(self) -> tuple[float, float]:
        return (float(self.a), float(self.b))


@dataclass(frozen=True, eq=False)
class ShellMasks:
    core: BoolArray
    bond: BoolArray
    interstitial: BoolArray


def shell_masks(spheres: NearestAtom, structure: Structure, shells: Shells) -> ShellMasks:
    """The regions from the sphere-rule assignment ``spheres`` (:func:`sphere_assignment`).

    Boundaries are tested with GEOMETRY_EPS tolerance.
    """
    c1, c2 = shells.atom_cutoffs(structure)
    r = spheres.distance
    c1k = c1[spheres.atom_index] + GEOMETRY_EPS
    c2k = c2[spheres.atom_index] + GEOMETRY_EPS
    return ShellMasks(core=r <= c1k, bond=(r > c1k) & (r <= c2k), interstitial=r > c2k)


# ----------------------------------------------------------------------
# cached on VolumetricData (computed once per structure)
# ----------------------------------------------------------------------

def geometry_of(vd: VolumetricData) -> NearestAtom:
    """The nearest-atom pass for ``vd``'s grid, cached on ``vd``."""
    key = ("geometry",)
    if key not in vd.cache:
        vd.cache[key] = nearest_atom(vd.shape, vd.structure)
    result: NearestAtom = vd.cache[key]
    return result


def sphere_geometry_of(vd: VolumetricData) -> NearestAtom:
    """The sphere-rule assignment for ``vd``'s grid, cached on ``vd``."""
    key = ("geometry", "spheres")
    if key not in vd.cache:
        vd.cache[key] = sphere_assignment(vd.shape, vd.structure)
    result: NearestAtom = vd.cache[key]
    return result


def shells_of(vd: VolumetricData, shells: Shells) -> ShellMasks:
    """Region masks for ``vd`` and ``shells``, cached on ``vd``."""
    key = ("shells", shells.key())
    if key not in vd.cache:
        vd.cache[key] = shell_masks(sphere_geometry_of(vd), vd.structure, shells)
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
