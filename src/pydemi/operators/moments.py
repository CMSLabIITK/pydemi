"""
pydemi.operators.moments
========================
Radial moments of any field about the nearest nucleus (spec §7):

    m_n = sum_k w_k r_k^n / sum_k w_k,     w = field ("signed") or |field| ("abs")

NaN when sum_k w_k = 0; the descriptor layer turns that into a flagged sentinel.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import NDArray

Weight = Literal["signed", "abs"]


def _weights(field: NDArray[Any], weight: Weight) -> NDArray[Any]:
    if weight not in ("signed", "abs"):
        raise ValueError(f"weight must be 'signed' or 'abs', got {weight!r}")
    return np.abs(field) if weight == "abs" else np.asarray(field)


def radial_moment(field: NDArray[Any], r: NDArray[Any], order: int, weight: Weight = "abs",
                  mask: Optional[NDArray[np.bool_]] = None) -> float:
    """m_n = sum_k w_k r_k^n / sum_k w_k (over ``mask`` voxels when given)."""
    w = _weights(field, weight)
    rr = np.asarray(r)
    if mask is not None:
        w, rr = w[mask], rr[mask]
    total = float(np.sum(w, dtype=np.float64))
    if total == 0.0 or not np.isfinite(total):
        return float("nan")
    return float(np.sum(w * rr ** order, dtype=np.float64) / total)


def radial_variance(field: NDArray[Any], r: NDArray[Any], weight: Weight = "abs",
                    mask: Optional[NDArray[np.bool_]] = None) -> float:
    """sigma_r2 = m2 - m1^2."""
    m1 = radial_moment(field, r, 1, weight, mask)
    m2 = radial_moment(field, r, 2, weight, mask)
    return m2 - m1 * m1
