"""
pydemi.descriptors.hessian
--------------------------
Descriptors from the density Hessian eigenvalues lambda1 <= lambda2 <= lambda3.

F3 -- non-covalent interactions (reference entries 77-79), with the reduced
density gradient (dimensionless, so independent of the length unit):

    s = |grad rho| / (2 (3 pi^2)^{1/3} rho^{4/3})

    f_NCI                 (1/N) sum_k 1(s_k < 0.5 and rho_k < 0.05 a.u.)
    NCI_attractive        fraction of NCI voxels with lambda2 < 0
    sign_lambda2_rho_mean mean of sign(lambda2) rho over NCI voxels (e/bohr^3)

F4 -- whole-grid ellipticity (reference entries 80-81), over bonding-shell
voxels where the density is concentrated in two directions (lambda2 < 0):

    epsilon_k = lambda1 / lambda2 - 1
    ellip_bond_avg, ellip_bond_std

Near a bond path with lambda2 -> 0^- the ratio diverges, so the mean can be
dominated by a few voxels; ``ellip_bond_median`` is reported alongside as a
robust companion (a pydemi addition, not in the reference).
"""

from typing import Optional

import numpy as np

from ..engine import RHO, Engine
from ..shells import Shells
from .kinetic import kinetic_terms

NCI_S_MAX = 0.5
NCI_RHO_MAX_AU = 0.05
_S_CONST = 2.0 * (3.0 * np.pi ** 2) ** (1.0 / 3.0)


def reduced_gradient(engine: Engine, field: str = RHO) -> np.ndarray:
    """s on the grid; inf where the density is below the floor."""
    kt = kinetic_terms(engine, field)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.sqrt(kt.grad2) / (_S_CONST * kt.rho ** (4.0 / 3.0))
    return np.where(kt.valid, s, np.inf)


def nci_family(engine: Engine, field: str = RHO, s_max: float = NCI_S_MAX,
               rho_max_au: float = NCI_RHO_MAX_AU) -> dict:
    """F3: entries 77-79."""
    kt = kinetic_terms(engine, field)
    nci = (reduced_gradient(engine, field) < s_max) & (kt.rho < rho_max_au)
    out = {"f_NCI": float(np.mean(nci))}
    if not nci.any():
        out.update(NCI_attractive=float("nan"), sign_lambda2_rho_mean=float("nan"))
        return out
    lam2 = engine[field].hessian_eigenvalues[..., 1][nci]
    out["NCI_attractive"] = float(np.mean(lam2 < 0))
    out["sign_lambda2_rho_mean"] = float(np.mean(np.sign(lam2) * kt.rho[nci]))
    return out


def ellipticity_family(engine: Engine, field: str = RHO,
                       shells: Optional[Shells] = None) -> dict:
    """F4: entries 80-81, plus the median."""
    f = engine[field]
    ev = f.hessian_eigenvalues
    sel = engine.shell_masks(f.grid.shape, shells).bond & (ev[..., 1] < 0)
    if not sel.any():
        nan = float("nan")
        return {"ellip_bond_avg": nan, "ellip_bond_std": nan, "ellip_bond_median": nan}
    eps = ev[..., 0][sel] / ev[..., 1][sel] - 1.0
    return {
        "ellip_bond_avg": float(np.mean(eps)),
        "ellip_bond_std": float(np.std(eps)),
        "ellip_bond_median": float(np.median(eps)),
    }
