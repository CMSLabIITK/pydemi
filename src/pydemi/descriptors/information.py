"""
pydemi.descriptors.information
------------------------------
F6 -- information-theoretic measures (reference entries 86-89), on the
shape function rho~ = rho / N_e (integral 1), in atomic units:

    S     = - int rho~ ln rho~ dV          shannon_entropy
    I_F   =   int |grad rho~|^2 / rho~ dV  fisher_information   (1/bohr^2)
    D     =   int rho~^2 dV                disequilibrium       (1/bohr^3)
    C_LMC =   D e^S                        LMC_complexity       (dimensionless)

S, I_F and D depend on the length unit (S shifts by 3 ln(unit), I_F and D
scale), so the unit is fixed to bohr as the reference specifies. C_LMC is
unit-free. Negative pseudo-density values are clipped to 0 (0 ln 0 = 0);
voxels below the density floor are left out of the Fisher integral.
"""

import numpy as np

from ..constants import ANGSTROM_BOHR, DENSITY_TO_AU, GRADIENT_TO_AU
from ..engine import RHO, Engine
from .kinetic import RHO_FLOOR_AU


def information_family(engine: Engine, field: str = RHO) -> dict:
    f = engine[field]
    dV = f.grid.dV * ANGSTROM_BOHR ** 3                 # bohr^3
    rho = np.maximum(f.values, 0.0) * DENSITY_TO_AU     # e / bohr^3
    N = rho.sum() * dV
    if N <= 0:
        nan = float("nan")
        return {"shannon_entropy": nan, "fisher_information": nan,
                "disequilibrium": nan, "LMC_complexity": nan}
    p = rho / N
    grad2 = (f.gradient_norm * GRADIENT_TO_AU / N) ** 2
    pos = p > 0
    fisher = rho > RHO_FLOOR_AU

    S = -float(np.sum(p[pos] * np.log(p[pos])) * dV)
    D = float(np.sum(p * p) * dV)
    return {
        "shannon_entropy": S,
        "fisher_information": float(np.sum(grad2[fisher] / p[fisher]) * dV),
        "disequilibrium": D,
        "LMC_complexity": D * float(np.exp(S)),
    }
