"""
pydemi.descriptors.potential
----------------------------
F2 -- electrostatic potential descriptors (reference entries 74-76).

Two sources, kept under separate names so a column never mixes them:

``V_*``   the potential read from LOCPOT (set LVHAR / LVTOT). Which
          potential that is depends on the VASP flags; LVHAR gives the
          Hartree + ionic electrostatic potential the reference intends.
``VH_*``  the electronic Hartree potential solved from rho by one FFT:

              V_H(G) = 4 pi rho(G) / |G|^2   (G != 0),   V_H(0) = 0

          in eV (e^2 / 4 pi eps0 = 14.40 eV A). It omits the ionic term,
          so it measures where electrons pile up, not the full ESP.

Descriptors, for either source:

    *_spread    (75)  std over atoms of the potential at each nucleus
    *_int_min   (76)  minimum of the potential over the interstitial shell

The G = 0 term is arbitrary (the cell-average potential is set to zero),
so ``*_int_min`` is relative to the cell average; ``*_spread`` is not
affected. Site values (74) come from exact Fourier interpolation of the
grid potential at the nuclear positions: :func:`site_potentials`.
"""

from typing import Optional

import numpy as np

from ..constants import COULOMB_EV_ANGSTROM
from ..engine import POT, RHO, Engine
from ..field import Field
from ..shells import Shells

HARTREE = "hartree_potential"


def hartree_potential(engine: Engine, field: str = RHO) -> Field:
    """Solve Poisson's equation for ``field``; registered as ``hartree_potential``."""
    name = HARTREE if field == RHO else f"{HARTREE}_{field}"
    if name not in engine:
        f = engine[field]
        grid = f.grid
        F = np.fft.rfftn(f.values)
        G2 = np.broadcast_to(grid.g_squared(), F.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            VG = np.where(G2 > 0, 4.0 * np.pi * COULOMB_EV_ANGSTROM * F / G2, 0.0)
        engine.add_field(name, np.fft.irfftn(VG, s=grid.shape, axes=(0, 1, 2)))
    return engine[name]


def fourier_interpolate(values: np.ndarray, frac_points: np.ndarray,
                        batch: int = 256) -> np.ndarray:
    """Trigonometric interpolation of a periodic grid field at fractional points.

    Exact at grid points. Separable: the first contraction is one matrix
    product over all points in a batch, the other two are small einsums.
    """
    F = np.fft.fftn(values) / values.size
    n1, n2, n3 = values.shape
    freqs = [np.fft.fftfreq(n, 1.0 / n) for n in values.shape]
    u = np.asarray(frac_points, float).reshape(-1, 3) % 1.0
    out = np.empty(len(u))
    for s in range(0, len(u), batch):
        ub = u[s:s + batch]
        p1, p2, p3 = (np.exp(2j * np.pi * np.outer(freqs[ax], ub[:, ax])) for ax in range(3))
        A = (F.reshape(-1, n3) @ p3).reshape(n1, n2, -1)     # (n1, n2, M)
        B = np.einsum("ijm,jm->im", A, p2)                   # (n1, M)
        out[s:s + batch] = np.real(np.einsum("im,im->m", B, p1))
    return out


def site_potentials(engine: Engine, source: str = "hartree", field: str = RHO) -> np.ndarray:
    """Potential at each nucleus (entry 74), eV. ``source``: "hartree" or "locpot"."""
    if source == "hartree":
        values = hartree_potential(engine, field).values
    elif source == "locpot":
        values = engine[POT].values
    else:
        raise ValueError(f"source must be 'hartree' or 'locpot', got {source!r}")
    return fourier_interpolate(values, engine.structure.frac_coords)


def _stats(engine, values, sites, prefix, shells):
    inter = engine.shell_masks(values.shape, shells).interstitial
    return {
        f"{prefix}_spread": float(np.std(sites)),
        f"{prefix}_int_min": float(values[inter].min()) if inter.any() else float("nan"),
    }


def potential_family(engine: Engine, field: str = RHO,
                     shells: Optional[Shells] = None) -> dict:
    """F2: VH_* always; V_* too when a LOCPOT is loaded."""
    out = _stats(engine, hartree_potential(engine, field).values,
                 site_potentials(engine, "hartree", field), "VH", shells)
    if POT in engine:
        out.update(_stats(engine, engine[POT].values,
                          site_potentials(engine, "locpot"), "V", shells))
    return out
