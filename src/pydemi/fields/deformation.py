"""
pydemi.fields.deformation
=========================
Promolecule construction and the deformation density (spec §6.1):

    delta_rho(r) = rho_crystal(r) - sum_i rho_free[element(i)](r - R_i)

The promolecule is a superposition of spherical free-atom densities placed
at the nuclei and summed over periodic images out to where each free-atom
density drops below PROMOLECULE_TOL (1e-6 e/Angstrom^3) -- not just the
parent cell.

Reference sources (``deformation_reference``; the one used is recorded in
the metadata)
-----------------
``"aeccar0"``    for an all-electron density (AECCAR0 + AECCAR2). AECCAR0
                 is the frozen-core density, an element-wise superposition
                 of atomic cores; it supplies the core part of the
                 promolecule, and the tabulated free-atom VALENCE densities
                 (ZVAL electrons per atom) the rest:
                     promolecule = AECCAR0 + sum_i rho_free_val,i
                 so delta_rho = AECCAR2 - sum_i rho_free_val,i.
                 (Documented correction: AECCAR0 alone is not a free-atom
                 superposition -- rho - AECCAR0 would be the whole valence
                 density, not a deformation density.)
``"tabulated"``  radial free-atom densities shipped with pydemi
                 (``data/free_atoms.npz``: spherical LDA atoms): all
                 electrons for an all-electron density, the ZVAL
                 highest-energy electrons for a PAW pseudo-density. For a
                 CHGCAR the free-atom valence keeps its all-electron shape
                 inside the PAW augmentation spheres, where the CHGCAR is
                 pseudized, so delta_rho there reflects the pseudization as
                 well as bonding (see the ``paw`` extension).
``"custom"``     a directory of per-element radial densities from the user's
                 own isolated-atom calculations: ``<Element>.dat`` (or .txt,
                 .csv), two columns r [Angstrom], rho [e/Angstrom^3].
``"auto"``       ``"aeccar0"`` when AECCAR0 was read, otherwise ``"tabulated"``
                 (the preference order of the specification).

Interpolation: nearest neighbour in radius on a fine uniform table built
once per element by a cubic spline of the radial data.

Sampling: the promolecule is point-sampled on the grid. A nucleus that sits
exactly on a grid point over-counts the cusp of its free-atom density (one
O atom, valence: +5.2% at 0.15 A spacing, +0.18% at 0.06 A); off the grid
points the integral is exact to 0.1%. The resulting int delta_rho dV is
reported as ``def_charge_mismatch`` in the metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from ..constants import IMAGE_SUM_BLOCK, PROMOLECULE_TOL
from ..core.geometry import image_blocks
from ..data import DATA_DIR, default_zval
from ..io.base import Structure

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]

#: step (Angstrom) of the uniform radial tables evaluated by nearest-neighbour lookup
TABLE_STEP = 2e-4


# ----------------------------------------------------------------------
# radial references
# ----------------------------------------------------------------------

@lru_cache(maxsize=1)
def _tabulated() -> dict[str, NDArray[np.floating]]:
    with np.load(DATA_DIR / "free_atoms.npz") as z:
        return {k: z[k] for k in z.files}


def tabulated_radial(element: str, part: str, zval: Optional[float] = None) -> tuple[F64, F64]:
    """(r [Angstrom], rho [e/Angstrom^3]) of the shipped free atom.

    ``part="total"``: all electrons. ``part="valence"``: the ``zval`` (default
    :func:`pydemi.data.default_zval`) highest-energy electrons.
    """
    t = _tabulated()
    if f"{element}_r" not in t:
        raise KeyError(f"no tabulated free-atom density for {element!r}")
    r = np.asarray(t[f"{element}_r"], dtype=np.float64)
    orb = np.asarray(t[f"{element}_orbitals"], dtype=np.float64)
    dens = np.asarray(t[f"{element}_density"], dtype=np.float64)
    occ = orb[:, 2]
    if part == "total":
        w = occ
    elif part == "valence":
        z = default_zval(element) if zval is None else float(zval)
        w = np.zeros_like(occ)
        remaining = z
        for k in np.argsort(-orb[:, 3], kind="stable"):          # highest energy first
            take = min(occ[k], remaining)
            w[k] = take
            remaining -= take
            if remaining <= 1e-9:
                break
        if remaining > 1e-9:
            raise ValueError(f"ZVAL {z} exceeds the {occ.sum()} electrons of {element}")
    else:
        raise ValueError(f"part must be 'total' or 'valence', got {part!r}")
    return r, np.asarray(w @ dens, dtype=np.float64)


class CustomReference:
    """Per-element radial densities from files ``<Element>.dat|.txt|.csv`` in ``directory``."""

    def __init__(self, directory: Union[str, Path]) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"custom reference directory {self.directory} not found")

    def radial(self, element: str) -> tuple[F64, F64]:
        for ext in (".dat", ".txt", ".csv"):
            p = self.directory / f"{element}{ext}"
            if p.exists():
                a = np.loadtxt(p, delimiter="," if ext == ".csv" else None, comments="#")
                return np.asarray(a[:, 0], np.float64), np.asarray(a[:, 1], np.float64)
        raise FileNotFoundError(f"{self.directory}: no radial density file for {element}")


@dataclass(frozen=True, eq=False)
class RadialTable:
    """A radial density on a uniform grid 0 .. r_max, evaluated by nearest neighbour."""

    values: F64          # values[k] = rho(k * step); the last entry is 0 (guard)
    step: float
    r_max: float

    def __call__(self, d: F64) -> F64:
        k = np.minimum(np.rint(d / self.step).astype(np.int64), self.values.size - 1)
        return np.asarray(self.values[k], dtype=np.float64)


def radial_table(r: F64, rho: F64, tol: float = PROMOLECULE_TOL,
                 step: float = TABLE_STEP) -> RadialTable:
    """Uniform table from (r, rho) by cubic spline, cut where rho drops below ``tol`` for good."""
    r = np.asarray(r, np.float64)
    rho = np.asarray(rho, np.float64)
    order = np.argsort(r)
    r, rho = r[order], rho[order]
    above = np.flatnonzero(rho >= tol)
    r_max = float(r[above[-1] + 1]) if above.size and above[-1] + 1 < r.size else float(r[-1])
    grid = np.arange(0.0, r_max + step, step)
    spline = CubicSpline(r, rho, extrapolate=False)
    vals = spline(np.clip(grid, r[0], r[-1]))
    vals = np.where(np.isfinite(vals), np.maximum(vals, 0.0), 0.0)
    vals[grid > r_max] = 0.0
    return RadialTable(np.append(vals, 0.0), step, r_max)


# ----------------------------------------------------------------------
# image sums
# ----------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class AtomDensityBlock:
    """Free-atom densities of every contributing atom image at a block of voxels."""

    voxel: I64          # (M,)
    density: F64        # (M, C) rho_free of image c at voxel m
    owner: I64          # (C,) atom of each image
    distance: F64       # (M, C)
    diff: F64           # (M, C, 3) voxel - image


def atom_density_blocks(shape: Sequence[int], structure: Structure,
                        tables: Mapping[str, RadialTable],
                        block: int = IMAGE_SUM_BLOCK) -> Iterator[AtomDensityBlock]:
    """Free-atom densities of all images within each element's cutoff, block by block."""
    cutoff = max(t.r_max for t in tables.values())
    elem = np.array([structure.species[i] for i in range(structure.n_atoms)])
    for b in image_blocks(shape, structure, cutoff, block):
        dens = np.zeros_like(b.distance)
        img_elem = elem[b.owner]
        for e in np.unique(img_elem):
            cols = img_elem == e
            dens[:, cols] = tables[str(e)](b.distance[:, cols])
        yield AtomDensityBlock(b.voxel, dens, b.owner, b.distance, b.diff)


def promolecule(shape: Sequence[int], structure: Structure,
                tables: Mapping[str, RadialTable]) -> F64:
    """sum over atoms and periodic images of the free-atom densities, on the grid."""
    n = int(np.prod(shape))
    out = np.zeros(n)
    for b in atom_density_blocks(shape, structure, tables):
        out[b.voxel] += b.density.sum(axis=1)
    return out.reshape(tuple(int(m) for m in shape))


def reference_tables(structure: Structure, kind: str, part: str,
                     zval: Optional[Mapping[str, float]] = None,
                     custom: Optional[str] = None) -> dict[str, RadialTable]:
    """One radial table per element for ``kind`` ("tabulated" / "custom")."""
    out: dict[str, RadialTable] = {}
    ref = None
    if kind == "custom":
        if custom is None:
            raise ValueError("the custom reference needs a directory")
        ref = CustomReference(custom)
    for e in structure.elements:
        if ref is not None:
            r, rho = ref.radial(e)
        else:
            r, rho = tabulated_radial(e, part, None if zval is None else zval.get(e))
        out[e] = radial_table(r, rho)
    return out


def split_shell_warning(element: str, zval: float) -> Optional[str]:
    """Message when ZVAL splits a partly-counted shell of the tabulated atom (else None)."""
    t = _tabulated()
    orb = np.asarray(t[f"{element}_orbitals"])
    remaining = zval
    for k in np.argsort(-orb[:, 3], kind="stable"):
        take = min(orb[k, 2], remaining)
        if 0 < take < orb[k, 2]:
            return f"ZVAL {zval} for {element} counts {take} of the {orb[k, 2]} electrons of one shell"
        remaining -= take
        if remaining <= 1e-9:
            return None
    return None


def radial_profile(values: NDArray[np.floating], structure: Structure, bins: int = 400
                   ) -> tuple[F64, F64]:
    """Spherical average about the single atom of an isolated-atom calculation.

    Builds a ``custom`` reference table from a one-atom-in-a-box density:
    returns (bin centres [Angstrom], mean density per bin).
    """
    from ..core.geometry import nearest_atom
    if structure.n_atoms != 1:
        raise ValueError("an isolated-atom profile needs a one-atom cell")
    r = nearest_atom(values.shape, structure).distance.ravel()
    edges = np.linspace(0.0, 0.5 * float(structure.lattice.heights.min()), bins + 1)
    k = np.digitize(r, edges) - 1
    ok = (k >= 0) & (k < bins)
    total = np.bincount(k[ok], weights=np.asarray(values).ravel()[ok], minlength=bins)
    count = np.bincount(k[ok], minlength=bins)
    centres = 0.5 * (edges[1:] + edges[:-1])
    keep = count > 0
    return centres[keep], total[keep] / count[keep]


__all__ = ["tabulated_radial", "CustomReference", "RadialTable", "radial_table",
           "atom_density_blocks", "promolecule", "reference_tables", "radial_profile",
           "split_shell_warning"]
