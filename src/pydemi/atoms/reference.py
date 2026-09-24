"""
pydemi.atoms.reference
----------------------
Free-atom reference densities for the promolecule (Family A) and the
Hirshfeld partition (entry 101).

Which electrons
---------------
The reference must describe the same electrons as the field it is
compared with:

``part="total"``    all electrons -- for AECCAR0 + AECCAR2 (``rho_ae``), the
                    route the reference recommends.
``part="valence"``  the ZVAL highest-energy electrons -- for CHGCAR, which
                    holds only the POTCAR valence. The free-atom valence
                    orbitals have their true all-electron shape inside the
                    PAW core radius, where CHGCAR is pseudized. For 3d
                    metals that mismatch dominates the deformation density
                    (see :mod:`pydemi.descriptors.deformation`); use
                    AECCARs, or :class:`IsolatedAtomReference`.

ZVAL should come from the POTCAR (``read_potcar_zval``; ``Engine.from_vasp_dir``
does this when a POTCAR is present). Without one, :func:`default_zval`
counts the electrons outside the preceding noble-gas core, treating a
filled 4f14 / 5f14 shell as core -- which matches many but not all VASP
POTCARs (e.g. Na_pv, K_sv and Ga_d differ). A mismatch between the summed
ZVAL and the CHGCAR electron count is reported as ``def_charge_mismatch``.

Densities are computed with :mod:`pydemi.atoms.solver` on first use and
cached in memory and on disk (``$PYDEMI_CACHE_DIR``, default
``~/.cache/pydemi``; set ``PYDEMI_CACHE_DIR=""`` to disable the disk cache).
"""

import os
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from ..constants import BOHR_ANGSTROM, DENSITY_FROM_AU
from .solver import L_SYMBOL, RadialGrid, solve_atom

NOBLE = {2: "He", 10: "Ne", 18: "Ar", 36: "Kr", 54: "Xe", 86: "Rn"}
CACHE_VERSION = 1


def _configuration(element: str):
    from pymatgen.core import Element
    el = Element(element)
    return el.Z, [(n, L_SYMBOL[l], float(f)) for n, l, f in el.full_electronic_structure]


def default_zval(element: str) -> float:
    """Electrons outside the preceding noble-gas core; a filled f14 counts as core."""
    Z, config = _configuration(element)
    core_Z = max((z for z in NOBLE if z < Z), default=0)
    _, core = _configuration(NOBLE[core_Z]) if core_Z else (0, [])
    core_shells = {(n, l) for n, l, _ in core}
    val = sum(f for n, l, f in config
              if (n, l) not in core_shells and not (l == 3 and f == 14.0))
    return float(val)


def read_potcar_zval(path) -> list:
    """ZVAL of every PAW dataset in a POTCAR or OUTCAR, in file order.

    Reads the ``POMASS = ...; ZVAL = ...`` header line each dataset has; the
    OUTCAR's later ``ZVAL =`` summary lines are not counted twice.
    """
    text = Path(path).read_text(errors="replace")
    header = re.findall(r"POMASS\s*=\s*[-\d.]+\s*;\s*ZVAL\s*=\s*([-\d.]+)", text)
    if header:
        return [float(v) for v in header]
    return [float(v) for v in re.findall(r"ZVAL\s*=\s*([-\d.]+)", text)]


def read_potcar_rcore(path) -> list:
    """RCORE (outermost PAW cutoff radius, bohr) of every dataset in a POTCAR or OUTCAR."""
    text = Path(path).read_text(errors="replace")
    return [float(v) for v in re.findall(r"RCORE\s*=\s*([-\d.]+)", text)]


class AtomicLDAReference:
    """Spherical LDA free-atom densities (pydemi's own atomic solver)."""

    pseudized = False     # all-electron shapes, also for the valence part

    def __init__(self, correlation: str = "pw92", grid: RadialGrid = RadialGrid(),
                 cache_dir: Optional[str] = None):
        self.correlation = correlation
        self.grid = grid
        if cache_dir is None:
            cache_dir = os.environ.get("PYDEMI_CACHE_DIR",
                                       str(Path.home() / ".cache" / "pydemi"))
        self.cache_dir = Path(cache_dir) / "atoms" if cache_dir else None
        self._memo = {}

    def _cache_path(self, element):
        if self.cache_dir is None:
            return None
        g = self.grid
        tag = f"{element}_{self.correlation}_{g.n}_{g.x_min:.4f}_{g.x_max:.4f}_v{CACHE_VERSION}"
        return self.cache_dir / f"{tag}.npz"

    def atom(self, element: str) -> dict:
        """{'r' (bohr), 'orbitals': [(n, l, occ, eps)], 'densities' (n_orb, n_r)}."""
        if element in self._memo:
            return self._memo[element]
        path = self._cache_path(element)
        if path is not None and path.exists():
            data = np.load(path)
            out = {"r": data["r"], "orbitals": [tuple(o) for o in data["orbitals"]],
                   "densities": data["densities"]}
        else:
            Z, config = _configuration(element)
            res = solve_atom(Z, config, self.grid, correlation=self.correlation)
            if not res.converged:
                warnings.warn(f"free-atom SCF for {element} did not converge "
                              f"(residual {res.extra['residual']:.1e})", stacklevel=2)
            out = {"r": res.r,
                   "orbitals": [(o[0], o[1], o[2], o[3]) for o in res.orbitals],
                   "densities": np.array([o[4] for o in res.orbitals])}
            if path is not None:
                # write-then-rename, so parallel workers never read a partial file
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
                np.savez_compressed(tmp, r=out["r"], orbitals=np.array(out["orbitals"]),
                                    densities=out["densities"])
                os.replace(tmp, path)
        self._memo[element] = out
        return out

    def radial(self, element: str, part: str = "total", zval: Optional[float] = None):
        """(r in Angstrom, density in e/Angstrom^3) for the requested electrons."""
        a = self.atom(element)
        occ = np.array([o[2] for o in a["orbitals"]])
        if part == "total":
            weights = occ
        elif part == "valence":
            zval = default_zval(element) if zval is None else float(zval)
            order = np.argsort([-o[3] for o in a["orbitals"]])   # highest energy first
            weights = np.zeros_like(occ)
            remaining = zval
            for k in order:
                take = min(occ[k], remaining)
                weights[k] = take
                remaining -= take
                if remaining <= 1e-9:
                    break
            if remaining > 1e-9:
                raise ValueError(f"ZVAL {zval} exceeds the {occ.sum()} electrons of {element}")
            partial = (weights > 0) & (weights < occ)
            if partial.any():
                warnings.warn(f"ZVAL {zval} for {element} splits an orbital shell", stacklevel=2)
        else:
            raise ValueError(f"part must be 'total' or 'valence', got {part!r}")
        n_au = weights @ a["densities"]
        return a["r"] * BOHR_ANGSTROM, n_au * DENSITY_FROM_AU

    def electrons(self, element: str, part: str = "total", zval: Optional[float] = None) -> float:
        a = self.atom(element)
        if part == "total":
            return float(sum(o[2] for o in a["orbitals"]))
        return default_zval(element) if zval is None else float(zval)


class IsolatedAtomReference:
    """Reference route 3: spherically averaged isolated-atom calculations.

    ``chgcars`` maps element -> CHGCAR of one atom in a large box, run with
    the production POTCAR, so the reference carries exactly the same PAW
    pseudization as the crystal CHGCAR and it cancels in delta_rho.
    Optionally ``aeccars`` maps element -> (AECCAR0, AECCAR2) paths for the
    all-electron ("total") part. Radial profiles are bin-averaged about
    the atom's position out to half the shortest box height.
    """

    pseudized = True

    def __init__(self, chgcars: Mapping[str, str], aeccars: Optional[Mapping] = None,
                 bins: int = 400):
        self.chgcars = dict(chgcars)
        self.aeccars = dict(aeccars or {})
        self.bins = bins
        self._memo = {}

    def _profile(self, cd):
        from ..geometry import nearest_atom
        from ..grid import Grid
        if cd.structure.n_atoms != 1:
            raise ValueError("isolated-atom reference needs a one-atom cell")
        grid = Grid(cd.structure.lattice, cd.shape)
        r = nearest_atom(grid, cd.structure).distance.ravel()
        heights = 1.0 / np.linalg.norm(cd.structure.inv_lattice, axis=0)
        edges = np.linspace(0.0, 0.5 * heights.min(), self.bins + 1)
        k = np.digitize(r, edges) - 1
        inside = (k >= 0) & (k < self.bins)
        total = np.bincount(k[inside], weights=cd.total.ravel()[inside], minlength=self.bins)
        count = np.bincount(k[inside], minlength=self.bins)
        centres = 0.5 * (edges[1:] + edges[:-1])
        keep = count > 0
        return centres[keep], total[keep] / count[keep]

    def radial(self, element: str, part: str = "total", zval: Optional[float] = None):
        key = (element, part)
        if key not in self._memo:
            from ..io.vasp import read_aeccar, read_chgcar
            if part == "valence":
                cd = read_chgcar(self.chgcars[element], read_spin=False)
            elif element in self.aeccars:
                cd = read_aeccar(*self.aeccars[element])
            else:
                raise KeyError(f"no AECCAR pair for {element}; the total part needs one")
            self._memo[key] = self._profile(cd)
        return self._memo[key]

    def electrons(self, element: str, part: str = "total", zval: Optional[float] = None) -> float:
        r, n = self.radial(element, part)
        return float(np.sum(4.0 * np.pi * r ** 2 * n * np.gradient(r)))


@lru_cache(maxsize=1)
def default_reference() -> AtomicLDAReference:
    return AtomicLDAReference()


def resolve_zval(elements, zval: Optional[Mapping[str, float]] = None) -> dict:
    """Per-element valence counts: explicit values first, the default rule otherwise."""
    zval = dict(zval or {})
    return {e: float(zval[e]) if e in zval else default_zval(e) for e in elements}
