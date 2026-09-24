"""
pydemi.operators.sitestats
==========================
Site aggregation and the variance decomposition (spec §7, §8.4).

Site aggregation restricts every sum to one atom's voxels -- for a smooth
partition, weights them by w_i(r_k) -- and returns per-site arrays. For a
field f with gradient grad f:

    m1^(i)     = sum_k w_i(k) f_k r_ik / sum_k w_i(k) f_k
    f_bond^(i) = sum_{k: c1 < r_ik <= c2} w_i(k) f_k / sum_k w_i(k) f_k
    zeta^(i)   = 1 - sum_k w_i(k) |grad f_k . u_ik| / sum_k w_i(k) |grad f_k|
    Q^(i)      = sum_k w_i(k) f_k dV

with r_ik and u_ik measured to atom i. One pass over the partition's pairs
accumulates every sum.

Variance decomposition (one-way ANOVA over elements, population variances)

    Var_i(X) = sum_e w_e Var_{i in e}(X)  +  sum_e w_e (Xbar_e - Xbar)^2
               ---------- within ---------    ----------- between ---------

with w_e the fraction of sites of element e. The between term is the
w_e-weighted variance of the element means, and the within term the
w_e-weighted mean of the within-element variances; these weights are what
make the identity exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ..constants import GEOMETRY_EPS
from ..core.partition import Partition

F64 = NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class SiteSums:
    """Per-site weighted sums of one field (see module docstring)."""

    total: F64       # sum w f
    r1: F64          # sum w f r
    r2: F64          # sum w f r^2
    core: F64        # sum_{r <= c1} w f
    bond: F64        # sum_{c1 < r <= c2} w f
    radial: F64      # sum w |grad f . u|
    gnorm: F64       # sum w |grad f|
    dV: float


def site_sums(partition: Partition, field: NDArray[Any], gradient: Optional[NDArray[Any]],
              c1: F64, c2: F64, dV: float) -> SiteSums:
    """Accumulate every site-restricted sum in one pass over ``partition``'s pairs."""
    n = partition.n_atoms
    f = np.asarray(field, dtype=np.float64).ravel()
    g = None if gradient is None else np.asarray(gradient, dtype=np.float64).reshape(-1, 3)
    gn = None if g is None else np.linalg.norm(g, axis=1)
    acc = {k: np.zeros(n) for k in ("total", "r1", "r2", "core", "bond", "radial", "gnorm")}

    def add(key: str, atom: NDArray[np.int64], w: F64) -> None:
        acc[key] += np.bincount(atom, weights=w, minlength=n)

    for p in partition.pairs():
        a, r = p.atom, p.distance
        wf = p.weight * f[p.voxel]
        add("total", a, wf)
        add("r1", a, wf * r)
        add("r2", a, wf * r * r)
        core = r <= c1[a] + GEOMETRY_EPS
        add("core", a, np.where(core, wf, 0.0))
        add("bond", a, np.where(~core & (r <= c2[a] + GEOMETRY_EPS), wf, 0.0))
        if g is not None and gn is not None:
            add("radial", a, p.weight * np.abs(np.einsum("ij,ij->i", g[p.voxel], p.direction)))
            add("gnorm", a, p.weight * gn[p.voxel])
    return SiteSums(dV=dV, **acc)


def _ratio(num: F64, den: F64) -> F64:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(np.where(den != 0, num / den, np.nan), dtype=np.float64)


def site_m1(s: SiteSums) -> F64:
    return _ratio(s.r1, s.total)


def site_f_bond(s: SiteSums) -> F64:
    return _ratio(s.bond, s.total)


def site_zeta(s: SiteSums) -> F64:
    return 1.0 - _ratio(s.radial, s.gnorm)


def site_charge(s: SiteSums) -> F64:
    return np.asarray(s.total * s.dV, dtype=np.float64)


@dataclass(frozen=True)
class SiteStatistics:
    std: float
    range: float
    max: float
    min: float
    within: float           # NaN when no element has two or more sites
    between: float
    n_sites: int
    n_elements: int
    max_sites_per_element: int


def variance_decomposition(x: NDArray[Any], groups: Sequence[str]) -> tuple[float, float]:
    """(within, between) of the ANOVA identity; population variances, site-fraction weights."""
    xv = np.asarray(x, dtype=np.float64)
    g = np.asarray(groups)
    mean = float(xv.mean())
    within = between = 0.0
    for e in dict.fromkeys(g.tolist()):
        xe = xv[g == e]
        w = xe.size / xv.size
        within += w * float(xe.var())
        between += w * (float(xe.mean()) - mean) ** 2
    return within, between


def site_statistics(x: NDArray[Any], groups: Sequence[str]) -> SiteStatistics:
    """std, range, max, min and the variance decomposition of per-site values ``x``."""
    xv = np.asarray(x, dtype=np.float64)
    g = np.asarray(groups)
    counts = [int((g == e).sum()) for e in dict.fromkeys(g.tolist())]
    within, between = variance_decomposition(xv, list(g))
    return SiteStatistics(
        std=float(xv.std()), range=float(np.ptp(xv)), max=float(xv.max()), min=float(xv.min()),
        within=within if max(counts) >= 2 else float("nan"), between=between,
        n_sites=int(xv.size), n_elements=len(counts), max_sites_per_element=max(counts))
