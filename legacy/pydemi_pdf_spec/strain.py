"""
pydemi.strain
-------------
Entry 107 -- strain response of the density, from SCF runs at +/- epsilon.
A validation experiment on a subset, not a production feature.

The strained cells have different lattices, so the comparison is made in
fractional coordinates on a common grid shape, with the charge per voxel
n_k = rho_k V / N (electrons). A uniform dilation leaves n unchanged, so
what remains is genuine redistribution:

    drho_deps     = sum_k |n+_k - n-_k| / (2 eps Q)    fraction of the charge that
                                                        moves per unit strain
    drho_deps_l2  = ||n+ - n-||_2 / (2 eps ||n0||_2)    relative L2 form

n0 is the unstrained run when given, else (n+ + n-)/2. Grids of different
shape are Fourier-resampled to the shape of ``plus``.
"""

from typing import Optional, Union

import numpy as np

from .engine import RHO, Engine
from .resample import fourier_resample


def _charge_per_voxel(engine: Engine, field: str, shape) -> np.ndarray:
    f = engine[field]
    v = f.values if f.grid.shape == tuple(shape) else fourier_resample(f.values, shape)
    return v * f.grid.volume / v.size


def strain_response(minus: Union[Engine, str], plus: Union[Engine, str], eps: float,
                    unstrained: Optional[Union[Engine, str]] = None,
                    field: str = RHO) -> dict:
    """Entry 107 from engines (or CHGCAR paths) at strain -eps and +eps."""
    load = lambda e: e if isinstance(e, Engine) else Engine.from_chgcar(e, read_spin=False)
    em, ep = load(minus), load(plus)
    if em.structure.species != ep.structure.species:
        raise ValueError("strained runs must have the same atoms in the same order")
    if not np.allclose(em.structure.frac_coords % 1, ep.structure.frac_coords % 1, atol=0.05):
        raise ValueError("fractional coordinates differ by more than 0.05: not the same cell under strain")
    shape = ep[field].grid.shape
    nm, np_ = _charge_per_voxel(em, field, shape), _charge_per_voxel(ep, field, shape)
    n0 = (_charge_per_voxel(load(unstrained), field, shape) if unstrained is not None
          else 0.5 * (nm + np_))
    diff = np_ - nm
    Q = n0.sum()
    return {
        "drho_deps": float(np.abs(diff).sum() / (2.0 * eps * Q)),
        "drho_deps_l2": float(np.linalg.norm(diff) / (2.0 * eps * np.linalg.norm(n0))),
        "charge_drift": float((np_.sum() - nm.sum()) / Q),
    }
