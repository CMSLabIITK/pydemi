"""
pydemi.descriptors.tier1
------------------------
Tier 1 charge-density descriptors (reference entries 1-15), plus the
recommended variants for the three flagged ambiguities:

    lnf            voxel count      ->  lnf_rho (charge-weighted)
    moment_ratio   m2/m1 (length)   ->  moment_ratio_scale_free (m2/m1^2)
    zeta_over_rvar zeta/sigma_r^2   ->  zeta_over_sigma_r

and one pydemi proposal: lap_concentration as specified is identically 1/2
for a periodic density, so ``lap_concentration_valence`` restricts it to
r > c1, where it is informative.

All come from one field plus the shared geometry pass. ``field`` may be
``"rho"`` (CHGCAR, pseudo-valence + augmentation) or ``"rho_ae"``
(AECCAR0 + AECCAR2), which removes the POTCAR-dependent near-nucleus shape
from m1.
"""

from typing import Optional

import numpy as np

from ..engine import RHO, Engine
from ..shells import Shells
from .primitives import (gradient_anisotropy, negative_fraction,
                         negative_magnitude_share, radial_moments, safe_div,
                         shell_fractions)


def tier1(engine: Engine, field: str = RHO, shells: Optional[Shells] = None) -> dict:
    f = engine[field]
    shape = f.grid.shape
    rho = f.values
    geo = engine.geometry(shape)
    masks = engine.shell_masks(shape, shells)
    lap = f.laplacian

    Q_tot = f.integral()
    m1, m2, var = radial_moments(rho, geo.distance)
    f_core, f_bond, f_int = shell_fractions(rho, masks)
    zeta = gradient_anisotropy(f.gradient, geo.direction)
    lnf = negative_fraction(lap)
    lnf_rho = negative_fraction(lap, weights=rho)
    sigma_r = float(np.sqrt(var)) if var >= 0 else float("nan")

    return {
        "zeta": zeta,
        "m1": m1,
        "m2": m2,
        "sigma_r2": var,
        "f_core": f_core,
        "f_bond": f_bond,
        "f_int": f_int,
        "lnf": lnf,
        "lnf_rho": lnf_rho,
        "fint_over_lnf": safe_div(f_int, lnf),
        "fint_over_lnf_rho": safe_div(f_int, lnf_rho),
        "moment_ratio": safe_div(m2, m1),
        "moment_ratio_scale_free": safe_div(m2, m1 * m1),
        "radial_cv": safe_div(sigma_r, m1),
        "zeta_over_rvar": safe_div(zeta, var),
        "zeta_over_sigma_r": safe_div(zeta, sigma_r),
        "charge_per_m1": safe_div(Q_tot, m1),
        "lap_concentration": negative_magnitude_share(lap),
        "lap_concentration_valence": negative_magnitude_share(lap[~masks.core]),
        "bond_int_ratio": safe_div(f_bond, f_int),
        "Q_tot": Q_tot,
    }
