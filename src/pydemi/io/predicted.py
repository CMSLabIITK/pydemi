"""
pydemi.io.predicted
===================
Densities predicted by a machine-learning model from the crystal structure.

A predicted density enters pydemi through the same data model as a DFT one:
the prediction of a model trained on VASP CHGCARs is a prediction of the PAW
pseudo-density (``density_source="pseudo"``), on a grid of the cell, in
electrons / Angstrom^3. What differs is recorded in ``sources``:
``origin="predicted"``, the model, and the factor by which the density was
rescaled to the nominal electron count.

* :func:`vasp_grid_shape` gives the grid VASP would use for the density
  (NGXF, NGYF, NGZF) from the lattice and the plane-wave cutoff alone, so a
  prediction needs no DFT calculation to fix its grid. For PREC = Accurate
  it reproduces the CHGCAR grid of all 6,059 runs of the paper's dataset
  (ENCUT = 500 eV, VASP 5.3.3 and 5.4.4).
* :func:`write_predicted` stores a prediction as ``.npz`` (lattice, species,
  fractional coordinates, density, model metadata): the exchange format
  between the model, which lives outside pydemi, and pydemi.
* :func:`read_predicted` turns such a file, or arrays, into
  :class:`VolumetricData`, with optional charge renormalization.

pydemi does not depend on any ML framework: the model writes the file, and
pydemi reads it.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from .base import FloatArray, Grid, Lattice, Structure, VolumetricData, as_structure

PathLike = Union[str, Path]

# VASP's own constants (constant.inc), so that grid sizes on a rounding
# boundary come out as in VASP.
_RYTOEV = 13.605826
_AUTOA = 0.529177249
_WFACT = {"accurate": 2.0, "normal": 1.5}
FORMAT_VERSION = 1


def _fft_friendly(n: int) -> bool:
    for p in (2, 3, 5, 7):
        while n % p == 0:
            n //= p
    return n == 1


def _next_even_fft(n: int) -> int:
    n = max(int(n), 2)
    while n % 2 or not _fft_friendly(n):
        n += 1
    return n


def vasp_grid_shape(lattice: Union[Lattice, ArrayLike], encut: float = 500.0,
                    prec: str = "accurate") -> Tuple[int, int, int]:
    """Charge-density grid (NGXF, NGYF, NGZF) that VASP uses for this cell.

    ``lattice``: a :class:`Lattice` or a 3x3 matrix of lattice vectors (rows,
    Angstrom); ``encut``: plane-wave cutoff in eV; ``prec``: "accurate"
    (validated) or "normal".

    VASP 5 sets the wavefunction grid along each lattice vector to the
    smallest even size with prime factors 2, 3, 5, 7 that is at least
    nint(2 WFACT x), with x = |a_i| sqrt(ENCUT / Ry) / (2 pi) in atomic units
    and WFACT = 2 for Accurate (1.5 for Normal), and doubles it for the
    density grid. The Accurate rule reproduces every grid of the paper's
    dataset; Normal follows the same code path but has not been checked
    against calculations.
    """
    key = prec.lower()
    if key not in _WFACT:
        raise ValueError(f"prec must be 'accurate' or 'normal', got {prec!r}")
    matrix = lattice.matrix if isinstance(lattice, Lattice) else np.asarray(lattice, dtype=float)
    lengths = np.linalg.norm(np.asarray(matrix, dtype=float).reshape(3, 3), axis=1)
    x = math.sqrt(encut / _RYTOEV) * (lengths / _AUTOA) / (2.0 * math.pi)
    coarse = [_next_even_fft(int(math.floor(2.0 * _WFACT[key] * xi + 0.5))) for xi in x]
    return (2 * coarse[0], 2 * coarse[1], 2 * coarse[2])


def write_predicted(path: PathLike, structure: Any, rho: ArrayLike, model: str = "",
                    **metadata: Any) -> Path:
    """Store a predicted density (electrons / Angstrom^3) with its structure as ``.npz``.

    ``rho`` has the grid shape (n1, n2, n3), value [i, j, k] at fractional
    position (i/n1, j/n2, k/n3). ``model`` and any further keyword (checkpoint,
    grid rule, ...) are kept as JSON metadata and end up in ``sources``.
    """
    s = as_structure(structure)
    data = np.asarray(rho)
    if data.ndim != 3:
        raise ValueError(f"rho must be a 3-D grid, got shape {data.shape}")
    meta = {"format_version": FORMAT_VERSION, "model": model, **metadata}
    p = Path(path)
    if p.suffix != ".npz":
        p = p.with_suffix(".npz")
    np.savez_compressed(p, lattice=s.lattice.matrix, species=np.array(s.species),
                        frac_coords=s.frac_coords, rho=data, metadata=json.dumps(meta))
    return p


def _load(path: PathLike) -> tuple[Structure, FloatArray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as f:
        missing = {"lattice", "species", "frac_coords", "rho"} - set(f.files)
        if missing:
            raise ValueError(f"{path}: not a pydemi prediction file (missing {sorted(missing)})")
        meta = json.loads(str(f["metadata"])) if "metadata" in f.files else {}
        structure = Structure(Lattice(f["lattice"]), [str(x) for x in f["species"]], f["frac_coords"])
        return structure, np.asarray(f["rho"], dtype=np.float64), meta


def read_predicted(path: Optional[PathLike] = None, *, structure: Any = None,
                   rho: Optional[ArrayLike] = None, zval: Optional[Mapping[str, float]] = None,
                   paw_radii: Optional[Mapping[str, float]] = None, renormalize: bool = True,
                   model: Optional[str] = None) -> VolumetricData:
    """A predicted density as :class:`VolumetricData`.

    Give either ``path`` (a file of :func:`write_predicted`) or ``structure``
    and ``rho`` (electrons / Angstrom^3 on a grid of the cell).

    ``zval`` (electrons) and ``paw_radii`` (Angstrom) are the per-element
    values of the PAW datasets the model was trained on, as for
    :func:`pydemi.read_vasp`; elements they do not cover fall back to
    :func:`pydemi.data.default_zval` and the covalent radius, recorded in the
    metadata.

    A predicted density does not integrate exactly to the valence electron
    count N = sum ZVAL. With ``renormalize=True`` (default) it is rescaled
    to N, which is what a workflow that starts from the structure alone
    knows; the factor is recorded as ``sources["charge_scale"]`` and the raw
    count as ``sources["n_electrons_raw"]``.
    """
    meta: dict[str, Any] = {}
    if path is not None:
        if structure is not None or rho is not None:
            raise ValueError("give path, or structure and rho, not both")
        s, data, meta = _load(path)
    else:
        if structure is None or rho is None:
            raise ValueError("give path, or both structure and rho")
        s, data = as_structure(structure), np.asarray(rho, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"rho must be a 3-D grid, got shape {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError("the predicted density contains NaN or inf")

    grid = Grid(np.ascontiguousarray(data), s.lattice)
    elements = s.elements
    zv = {e: float(zval[e]) for e in elements if e in zval} if zval else {}
    radii = {e: float(paw_radii[e]) for e in elements if e in paw_radii} if paw_radii else {}
    label = model if model is not None else str(meta.get("model", ""))
    sources = {"origin": "predicted", "rho": f"predicted:{label}" if label else "predicted"}
    if path is not None:
        sources["file"] = str(path)
    for k in ("checkpoint", "grid_rule"):
        if k in meta:
            sources[k] = str(meta[k])
    if zv:
        sources["zval"] = "table"
    if radii:
        sources["paw_radii"] = "table"

    n_raw = float(data.sum() * grid.dV)
    sources["n_electrons_raw"] = f"{n_raw:.6f}"
    if renormalize:
        from ..data import default_zval
        missing = [e for e in elements if e not in zv]
        if missing:
            warnings.warn(f"no ZVAL for {missing}: renormalizing with pydemi.data.default_zval, "
                          "which is wrong for some PAW datasets; pass zval=", stacklevel=2)
        n_ref = sum(zv[e] if e in zv else default_zval(e) for e in s.species)
        if n_raw <= 0:
            raise ValueError(f"predicted density integrates to {n_raw:.4g} electrons; cannot renormalize")
        scale = n_ref / n_raw
        grid = Grid(grid.data * scale, s.lattice)
        sources["charge_scale"] = f"{scale:.8f}"
    else:
        sources["charge_scale"] = "1"

    return VolumetricData(s, grid, density_source="pseudo", zval=zv or None,
                          paw_radii=radii or None, sources=sources)


def predicted_grid(structure: Any, encut: float = 500.0,
                   prec: str = "accurate") -> Tuple[Tuple[int, int, int], FloatArray]:
    """(grid shape, Cartesian probe positions of shape (n1, n2, n3, 3)) for a structure.

    The positions are those of the density grid, r_ijk = (i/n1, j/n2, k/n3) A,
    at which a model has to predict the density.
    """
    s = as_structure(structure)
    shape = vasp_grid_shape(s.lattice, encut, prec)
    u = np.meshgrid(*(np.arange(n) / n for n in shape), indexing="ij")
    frac = np.stack(u, axis=-1)
    return shape, np.asarray(frac @ s.lattice.matrix, dtype=np.float64)


__all__: Sequence[str] = ["vasp_grid_shape", "write_predicted", "read_predicted", "predicted_grid"]
