"""
pydemi.descriptors.sites
------------------------
Family C -- site-resolved heterogeneity (reference entries 53-59) -- and the
partition-sensitivity variants of Family H (entries 99-100).

For a whole-cell descriptor X, its site version restricts every sum to the
voxels (or, for a soft partition, the weighted voxels) of atom i:

    m1^(i)     = sum_k w_i(k) rho_k r_ik / sum_k w_i(k) rho_k
    f_bond^(i) = sum_{k: c1 < r_ik <= c2} w_i(k) rho_k / sum_k w_i(k) rho_k
    zeta^(i)   = 1 - sum_k w_i(k) |grad rho_k . r_hat_ik| / sum_k w_i(k) |grad rho_k|

with r_ik measured to atom i. One pass over the partition's pairs
accumulates every sum; the whole-cell value is the sum over sites, so with
the nearest-atom partition it reproduces Tier 1 exactly.

Statistics per site quantity X (population statistics, ddof = 0):

    X_site_std (53-55), X_site_range (56), X_site_max / X_site_min (57),
    X_var_within (58), X_var_between (59), X_within_share

The last three are one variance decomposition (a one-way ANOVA):

    Var_i X^(i) = sum_e w_e Var_{i in e} X^(i)  +  sum_e w_e (Xbar_e - Xbar)^2
                  ----------- within ---------     ---------- between ---------

with w_e the fraction of atoms of element e. The reference writes the
within term as a plain mean over elements; the atom-fraction weights are
what make the two terms add up exactly, so they are used here.
X_within_share = within / total is the single ratio the reference
recommends reporting.

Small cells give std over very few sites, and within-element variance
needs several same-element sites; ``n_atoms`` is reported alongside.
"""

from typing import Optional

import numpy as np

from ..engine import RHO, Engine
from ..shells import Shells

SITE_QUANTITIES = ("m1", "f_bond", "zeta")
SCHEMES = ("power", "hirshfeld")  # Becke is opt-in: add "becke" to schemes
WHOLE_CELL = ("m1", "m2", "sigma_r2", "f_core", "f_bond", "f_int", "zeta")


def site_sums(engine: Engine, field: str = RHO, scheme: str = "nearest",
              shells: Optional[Shells] = None, **partition_kwargs) -> dict:
    """Weighted per-atom sums for every site quantity, in one pass."""
    f = engine[field]
    shape = f.grid.shape
    part = engine.partition(shape, scheme, **_hirshfeld_part(scheme, field, partition_kwargs))
    shells = shells if shells is not None else engine.shells
    c1, c2 = shells.atom_cutoffs(engine.structure)
    rho = f.values.ravel()
    grad = f.gradient.reshape(-1, 3)
    gnorm = f.gradient_norm.ravel()

    n = engine.structure.n_atoms
    keys = ("rho", "rho_r", "rho_r2", "core", "bond", "int", "radial", "gnorm")
    acc = {k: np.zeros(n) for k in keys}

    def add(key, atom, w):
        acc[key] += np.bincount(atom, weights=w, minlength=n)

    for p in part.pairs():
        a, r = p.atom, p.distance
        wr = p.weight * rho[p.voxel]
        add("rho", a, wr)
        add("rho_r", a, wr * r)
        add("rho_r2", a, wr * r * r)
        core = r <= c1[a]
        inter = r > c2[a]
        add("core", a, np.where(core, wr, 0.0))
        add("int", a, np.where(inter, wr, 0.0))
        add("bond", a, np.where(~core & ~inter, wr, 0.0))
        radial = np.abs(np.einsum("ij,ij->i", grad[p.voxel], p.direction))
        add("radial", a, p.weight * radial)
        add("gnorm", a, p.weight * gnorm[p.voxel])
    acc["dV"] = f.grid.dV
    return acc


def _hirshfeld_part(scheme, field, kwargs):
    """Hirshfeld weights must describe the same electrons as the field."""
    from ..engine import RHO_AE
    if scheme == "hirshfeld" and "part" not in kwargs:
        kwargs = dict(kwargs, part="total" if field == RHO_AE else "valence")
    return kwargs


def _ratio(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den != 0, num / den, np.nan)


def site_values(sums: dict) -> dict:
    """Per-site X^(i) arrays from :func:`site_sums`."""
    m1 = _ratio(sums["rho_r"], sums["rho"])
    m2 = _ratio(sums["rho_r2"], sums["rho"])
    return {
        "charge": sums["rho"] * sums["dV"],
        "m1": m1,
        "m2": m2,
        "sigma_r2": m2 - m1 * m1,
        "f_core": _ratio(sums["core"], sums["rho"]),
        "f_bond": _ratio(sums["bond"], sums["rho"]),
        "f_int": _ratio(sums["int"], sums["rho"]),
        "zeta": 1.0 - _ratio(sums["radial"], sums["gnorm"]),
    }


def whole_cell_values(sums: dict) -> dict:
    """Tier 1 radial descriptors from the site sums (sum over atoms)."""
    t = {k: float(np.sum(v)) for k, v in sums.items() if k != "dV"}
    m1 = float(_ratio(t["rho_r"], t["rho"]))
    m2 = float(_ratio(t["rho_r2"], t["rho"]))
    return {
        "m1": m1, "m2": m2, "sigma_r2": m2 - m1 * m1,
        "f_core": float(_ratio(t["core"], t["rho"])),
        "f_bond": float(_ratio(t["bond"], t["rho"])),
        "f_int": float(_ratio(t["int"], t["rho"])),
        "zeta": 1.0 - float(_ratio(t["radial"], t["gnorm"])),
    }


def variance_decomposition(x, groups):
    """(within, between, within_share) over finite x; groups are labels.

    ``within`` and ``within_share`` are NaN when no element has two or more
    sites: a single site carries no within-element information, and 0 would
    falsely read as "no configurational disorder".
    """
    x = np.asarray(x, float)
    groups = np.asarray(groups)
    keep = np.isfinite(x)
    x, groups = x[keep], groups[keep]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    labels, counts = np.unique(groups, return_counts=True)
    mean = x.mean()
    within = between = 0.0
    for g in labels:
        xe = x[groups == g]
        w = xe.size / x.size
        within += w * xe.var()
        between += w * (xe.mean() - mean) ** 2
    if counts.max() < 2:
        return float("nan"), float(between), float("nan")
    total = within + between
    scale = max(mean * mean, 1e-300)
    share = within / total if total > 1e-14 * scale else float("nan")
    return float(within), float(between), float(share)


def site_statistics(values: dict, species, quantities=SITE_QUANTITIES,
                    suffix: str = "") -> dict:
    out = {}
    for X in quantities:
        x = np.asarray(values[X], float)
        finite = x[np.isfinite(x)]
        nan = float("nan")
        within, between, share = variance_decomposition(x, species)
        out.update({
            f"{X}_site_std{suffix}": float(finite.std()) if finite.size else nan,
            f"{X}_site_range{suffix}": float(np.ptp(finite)) if finite.size else nan,
            f"{X}_site_max{suffix}": float(finite.max()) if finite.size else nan,
            f"{X}_site_min{suffix}": float(finite.min()) if finite.size else nan,
            f"{X}_var_within{suffix}": within,
            f"{X}_var_between{suffix}": between,
            f"{X}_within_share{suffix}": share,
        })
    return out


def site_family(engine: Engine, field: str = RHO, shells: Optional[Shells] = None) -> dict:
    """Family C with the nearest-atom partition."""
    values = site_values(site_sums(engine, field, "nearest", shells))
    out = site_statistics(values, engine.structure.species)
    out["n_atoms"] = engine.structure.n_atoms
    return out


def partition_family(engine: Engine, field: str = RHO, shells: Optional[Shells] = None,
                     schemes=SCHEMES, **partition_kwargs) -> dict:
    """Family H: Tier 1 radial descriptors and Family C statistics under other
    partitions, suffixed ``_<scheme>``.

    Default: the power diagram and Hirshfeld (the smooth scheme; ~30 s at
    a 180^3 grid). Becke is opt-in because converged weights are slow in
    periodic solids; see :mod:`pydemi.partition`. Comparing with the
    nearest-atom values is the partition-sensitivity axis the reference
    asks for.
    """
    out = {}
    for scheme in schemes:
        kw = {k: v for k, v in partition_kwargs.items()
              if (scheme == "becke" and k in ("k", "cells", "radii"))
              or (scheme == "hirshfeld" and k in ("r_cut", "part"))}
        sums = site_sums(engine, field, scheme, shells, **kw)
        out.update({f"{k}_{scheme}": v for k, v in whole_cell_values(sums).items()})
        out.update(site_statistics(site_values(sums), engine.structure.species,
                                   suffix=f"_{scheme}"))
    return out


def site_charges(engine: Engine, scheme: str = "nearest", field: str = RHO,
                 **partition_kwargs) -> np.ndarray:
    """Electrons assigned to each atom by ``scheme``."""
    f = engine[field]
    part = engine.partition(f.grid.shape, scheme, **_hirshfeld_part(scheme, field, partition_kwargs))
    return part.site_sum(f.values) * f.grid.dV


def hirshfeld_charges(engine: Engine, field: Optional[str] = None, **partition_kwargs) -> np.ndarray:
    """Hirshfeld atomic charges q_i = N_i - int w_i rho dV (entry 101).

    N_i is Z for the all-electron field (AECCAR, used when loaded) and the
    POTCAR ZVAL for CHGCAR. Only the AECCAR route, or CHGCAR with an
    :class:`~pydemi.atoms.reference.IsolatedAtomReference`, gives charges
    comparable with Bader: with all-electron-shaped reference weights the
    pseudized CHGCAR valence is not partitioned consistently (on 31 VASP
    binaries the sign of the charge transfer followed the electronegativity
    difference in only 58% of cases). That case warns.
    """
    import warnings
    from ..engine import RHO_AE
    from ..elements import atomic_number
    if field is None:
        field = RHO_AE if RHO_AE in engine else RHO
    if field == RHO and not getattr(engine.reference, "pseudized", False):
        warnings.warn("Hirshfeld charges of CHGCAR with all-electron reference shapes are "
                      "biased by PAW pseudization; load AECCARs or use an "
                      "IsolatedAtomReference", stacklevel=2)
    Q = site_charges(engine, "hirshfeld", field, **partition_kwargs)
    zval = engine.valence()
    N = np.array([atomic_number(s) if field == RHO_AE else zval[s]
                  for s in engine.structure.species], float)
    return N - Q
