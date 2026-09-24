"""
pydemi.descriptors.spin
-----------------------
Family E -- spin-density descriptors (reference entries 63-71), from the
magnetization density m = rho_up - rho_down (CHGCAR block 2), in mu_B and
Angstrom:

    M_abs          sum_k |m_k| dV                         (63)
    M_net          |sum_k m_k dV|                         (64)
    m1_spin        sum |m| r / sum |m|                    (65)
    sigma_r2_spin  sum |m| r^2 / sum |m| - m1_spin^2      (66)
    f_bond_spin    sum_bond |m| / sum |m|                 (67)
    mu_site_i      sum_{k in i} m_k dV                    (68, per site)
    mu_site_std    spread of the site moments             (69)
    spin_frustration  1 - |sum_i mu_i| / sum_i |mu_i|     (70)
    spin_charge_correlation  Pearson r(rho_k, |m_k|)      (71)

Non-collinear runs (three magnetization blocks) are handled as vectors:
|m| is the vector norm, M_net and mu_i are vector sums, mu_site_std is
sqrt(mean_i |mu_i - mean mu|^2) (the signed std in the collinear case), and
spin_frustration uses vector norms -- so a non-collinear arrangement is not
misread as collinear cancellation.

Sentinels
---------
A non-spin-polarized run returns 0 for every entry (not NaN), with
``is_spin_polarized = 0``. A spin-polarized run whose total |m| is below
``MAGNETIC_TOL`` mu_B per atom keeps its (tiny) M_abs, M_net and
mu_site_std but returns 0 for the ratio-type entries, which would otherwise
be ratios of numerical noise; ``is_magnetic`` flags which case applies.
"""

import numpy as np

from ..engine import MAG, MAG_ABS, RHO, Engine
from ..io.vasp import SPIN_NONCOLLINEAR, SPIN_NONE
from ..shells import Shells
from .primitives import radial_moments, safe_div

MAGNETIC_TOL = 0.01  # mu_B per atom

_RATIO_KEYS = ("m1_spin", "sigma_r2_spin", "f_bond_spin", "spin_frustration",
               "spin_charge_correlation")
_ALL_KEYS = ("M_abs", "M_net") + _RATIO_KEYS[:3] + ("mu_site_std",) + _RATIO_KEYS[3:]


def _magnetization(engine: Engine) -> np.ndarray:
    """(n_comp, nx, ny, nz): one component collinear, three non-collinear."""
    if engine.spin_mode == SPIN_NONCOLLINEAR:
        return np.asarray(engine.magnetization_vector, float)
    return engine[MAG].values[None]


def site_moments(engine: Engine, scheme: str = "nearest", **partition_kwargs) -> np.ndarray:
    """Entry 68: per-site moments, (n_atoms,) collinear or (n_atoms, 3) non-collinear."""
    if engine.spin_mode == SPIN_NONE:
        return np.zeros(engine.structure.n_atoms)
    m = _magnetization(engine)
    grid = engine[MAG_ABS].grid
    part = engine.partition(grid.shape, scheme, **partition_kwargs)
    mu = np.stack([part.site_sum(c) * grid.dV for c in m], axis=-1)
    return mu[:, 0] if mu.shape[1] == 1 else mu


def spin_family(engine: Engine, field: str = RHO, shells=None) -> dict:
    if engine.spin_mode == SPIN_NONE:
        out = {k: 0.0 for k in _ALL_KEYS}
        out.update(is_spin_polarized=0, is_magnetic=0)
        return out

    absm_field = engine[MAG_ABS]
    grid = absm_field.grid
    absm = absm_field.values
    geo = engine.geometry(grid.shape)
    masks = engine.shell_masks(grid.shape, shells)
    m = _magnetization(engine)

    M_abs = float(absm.sum() * grid.dV)
    M_net = float(np.linalg.norm(m.reshape(m.shape[0], -1).sum(axis=1) * grid.dV))
    mu = site_moments(engine).reshape(engine.structure.n_atoms, -1)  # (n, 1) or (n, 3)
    mu_std = float(np.sqrt(np.mean(np.sum((mu - mu.mean(axis=0)) ** 2, axis=1))))

    out = {"M_abs": M_abs, "M_net": M_net, "mu_site_std": mu_std,
           "is_spin_polarized": 1,
           "is_magnetic": int(M_abs > MAGNETIC_TOL * engine.structure.n_atoms)}
    if not out["is_magnetic"]:
        out.update({k: 0.0 for k in _RATIO_KEYS})
        return {k: out[k] for k in _ALL_KEYS + ("is_spin_polarized", "is_magnetic")}

    m1, _, var = radial_moments(absm, geo.distance)
    rho = engine[field].values
    if rho.shape == absm.shape and np.std(rho) > 0 and np.std(absm) > 0:
        corr = float(np.corrcoef(rho.ravel(), absm.ravel())[0, 1])
    else:
        corr = float("nan")
    norms = np.linalg.norm(mu, axis=1)
    out.update({
        "m1_spin": m1,
        "sigma_r2_spin": var,
        "f_bond_spin": safe_div(absm[masks.bond].sum(), absm.sum()),
        "spin_frustration": 1.0 - safe_div(np.linalg.norm(mu.sum(axis=0)), norms.sum()),
        "spin_charge_correlation": corr,
    })
    return {k: out[k] for k in _ALL_KEYS + ("is_spin_polarized", "is_magnetic")}
