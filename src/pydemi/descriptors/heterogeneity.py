"""
pydemi.descriptors.heterogeneity
================================
Heterogeneity domain (spec §8.4): a meta-operator over per-site descriptors.

For a base descriptor X with per-site values X^(i) (site aggregation, see
:mod:`pydemi.operators.sitestats`), :func:`heterogeneity` registers

    X_site_std              std over sites of X^(i)
    X_site_range            max_i X^(i) - min_i X^(i)
    X_site_max, X_site_min
    X_within_element_var    sum_e w_e Var_{i in e}(X^(i))
    X_between_element_var   sum_e w_e (Xbar_e - Xbar)^2

with w_e the fraction of sites of element e, so that
Var_i(X) = within + between exactly (one-way ANOVA). Applied to m1, f_bond,
zeta and the site moment mu. Every site-restricted sum honours the
``partition`` option.

Sentinels: a within-element variance is NaN when no element has two or more
sites (no within-element information; the per-element site counts are in
the metadata); a between-element variance is 0.0 for a single-element
structure; both flagged. mu statistics are 0.0 for a non-magnetic structure.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..core.partition import partition_of
from ..io.base import VolumetricData
from ..operators.sitestats import (SiteStatistics, SiteSums, site_f_bond, site_m1, site_statistics,
                                   site_sums, site_zeta)
from .magnetic import is_magnetic, site_moment_magnitudes
from .registry import Result, Sentinel, field_derivatives, is_uniform, metadata_hook, options, register

F64 = NDArray[np.float64]
PerSite = Callable[[VolumetricData], "F64 | Sentinel"]


def density_site_sums(vd: VolumetricData) -> SiteSums:
    """Site sums of rho under the requested partition (cached on vd)."""
    o = options(vd)
    key = ("site_sums", "rho", o.partition, o.shells.key(), o.derivative_backend, o.fd_order)
    if key not in vd.cache:
        c1, c2 = o.shells.atom_cutoffs(vd.structure)
        vd.cache[key] = site_sums(partition_of(vd, o.partition), vd.rho.data,
                                  field_derivatives(vd, "rho").gradient, c1, c2, vd.rho.dV)
    out: SiteSums = vd.cache[key]
    return out


def _per_site_m1(vd: VolumetricData) -> "F64 | Sentinel":
    return site_m1(density_site_sums(vd))


def _per_site_f_bond(vd: VolumetricData) -> "F64 | Sentinel":
    return site_f_bond(density_site_sums(vd))


def _per_site_zeta(vd: VolumetricData) -> "F64 | Sentinel":
    if is_uniform(vd.rho.data):
        return Sentinel(0.0, "uniform_density")
    return site_zeta(density_site_sums(vd))


def _per_site_mu(vd: VolumetricData) -> "F64 | Sentinel":
    if not is_magnetic(vd):
        return Sentinel(0.0, "non_magnetic")
    return site_moment_magnitudes(vd)


def _stats(vd: VolumetricData, per_site: PerSite, base: str) -> "SiteStatistics | Sentinel":
    o = options(vd)
    key = ("site_stats", base, o.partition, o.shells.key(), o.derivative_backend, o.fd_order,
           o.laplacian_method)
    if key not in vd.cache:
        x = per_site(vd)
        if isinstance(x, Sentinel):
            vd.cache[key] = x
        else:
            vd.cache[key] = site_statistics(x, vd.structure.species)
    out: "SiteStatistics | Sentinel" = vd.cache[key]
    return out


def heterogeneity(base: str, per_site: PerSite, units: str, formula: str, requires: list[str],
                  skip: tuple[str, ...] = ()) -> None:
    """Register the six heterogeneity statistics of the per-site descriptor ``base``."""
    sq = f"({units})^2" if units != "dimensionless" else "dimensionless"
    entries = {
        "site_std": ("std over sites of", units, "std"),
        "site_range": ("max - min over sites of", units, "range"),
        "site_max": ("max over sites of", units, "max"),
        "site_min": ("min over sites of", units, "min"),
        "within_element_var": ("sum_e w_e Var_{i in e} of", sq, "within"),
        "between_element_var": ("sum_e w_e (mean_e - mean)^2 of", sq, "between"),
    }
    for suffix, (text, unit, attr) in entries.items():
        name = f"{base}_{suffix}"
        if name in skip:
            continue
        cases: dict[str, float] = {"uniform_density": 0.0, "non_magnetic": 0.0}
        if attr == "within":
            cases["one_site_per_element"] = float("nan")
        if attr == "between":
            cases["single_element"] = 0.0

        def fn(vd: VolumetricData, attr: str = attr) -> Result:
            st = _stats(vd, per_site, base)
            if isinstance(st, Sentinel):
                return st
            value = float(getattr(st, attr))
            if attr == "within" and st.max_sites_per_element < 2:
                return Sentinel(float("nan"), "one_site_per_element")
            if attr == "between" and st.n_elements == 1:
                return Sentinel(0.0, "single_element")
            return value

        register(name=name, domain="heterogeneity", field="rho" if base != "mu" else "magnetization",
                 requires=requires + ["partition"], units=unit, range=(-np.inf, np.inf),
                 sentinel_cases=cases,
                 doc=f"{name} = {text} X^(i),  X^(i) = {formula}\n\n"
                     "Site-restricted sums use the partition weights w_i(r_k); w_e is the "
                     "fraction of sites of element e (spec §8.4).")(fn)


heterogeneity("m1", _per_site_m1, "Angstrom",
              "sum_k w_i(k) rho_k r_ik / sum_k w_i(k) rho_k", ["geometry"])
heterogeneity("f_bond", _per_site_f_bond, "dimensionless",
              "sum_{c1 < r_ik <= c2} w_i(k) rho_k / sum_k w_i(k) rho_k", ["geometry", "shells"])
heterogeneity("zeta", _per_site_zeta, "dimensionless",
              "1 - sum_k w_i(k) |grad rho_k . u_ik| / sum_k w_i(k) |grad rho_k|",
              ["gradient", "geometry"])
# mu_site_std itself belongs to the magnetic domain (spec §8.3)
heterogeneity("mu", _per_site_mu, "mu_B",
              "mu_i = sum_k w_i(k) m_k dV (signed; |mu_i| for non-collinear runs)",
              ["magnetization"], skip=("mu_site_std",))


@metadata_hook
def _site_metadata(vd: VolumetricData) -> dict[str, object]:
    counts = vd.structure.site_counts()
    return {"n_elements": len(counts), "max_sites_per_element": max(counts.values())}
