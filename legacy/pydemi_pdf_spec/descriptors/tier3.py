"""
pydemi.descriptors.tier3
------------------------
Tier 3 interaction terms and transforms (reference entries 32-39).

The reference recommends reclassifying most of this tier: 33-37 are
combinatorial pairings (kind ``cross_term``) to be reported only if an
independent SHAP/correlation analysis ranks them; 38-39 are regression
preprocessing (kind ``preprocessing``). Entry 32 is kept, with the
core-excluded ``laplacian_std_valence`` alongside it, which measures bonding
heterogeneity instead of near-nucleus numerical spikes.
"""

from typing import Optional

import numpy as np

from ..engine import RHO, Engine
from ..shells import Shells
from .primitives import safe_div


def tier3(engine: Engine, t1: dict, t2: dict, field: str = RHO,
          shells: Optional[Shells] = None) -> dict:
    f = engine[field]
    lap = f.laplacian
    not_core = ~engine.shell_masks(f.grid.shape, shells).core
    lnf, zeta = t1["lnf"], t1["zeta"]

    return {
        "laplacian_std": float(np.std(lap)),
        "laplacian_std_valence": float(np.std(lap[not_core])) if not_core.any() else float("nan"),
        "vec_x_lnf": t2["mean_vec"] * lnf,
        "elneg_x_lnf": t2["mean_elneg"] * lnf,
        "vec_over_rvar": safe_div(t2["mean_vec"], t1["sigma_r2"]),
        "bond_over_lnf": safe_div(t1["f_bond"], lnf),
        "lnf_x_m1": lnf * t1["m1"],
        "sqrt_zeta": float(np.sqrt(zeta)) if zeta >= 0 else float("nan"),
        "log_lnf": float(np.log(lnf)) if lnf > 0 else float("nan"),
    }
