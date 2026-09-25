"""
pydemi.operators.laplacian
==========================
Laplacian-derived statistics of any field (spec §7):

    sign fraction            (1/N) sum_k 1(lap f_k < 0)
    charge-weighted variant  sum_{lap f < 0} w_k / sum_k w_k
    concentration            sum_{lap f < 0} |lap f_k| / sum_k |lap f_k|

On a periodic grid int lap f dV = 0 exactly (the G = 0 Fourier term of a
Laplacian vanishes, and every periodic difference stencil sums to zero), so
the negative and positive parts of lap f cancel and the concentration of the
whole cell is identically 1/2 for ANY field. It becomes informative only
when restricted to a region (``mask``), e.g. the valence region r > c1.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


def negative_fraction(lap: NDArray[Any], mask: Optional[NDArray[np.bool_]] = None) -> float:
    """(1/N) sum_k 1(lap f_k < 0)."""
    v = np.asarray(lap) if mask is None else np.asarray(lap)[mask]
    if v.size == 0:
        return float("nan")
    return float(np.mean(v < 0))


def weighted_negative_fraction(lap: NDArray[Any], weights: NDArray[Any]) -> float:
    """sum_{lap f < 0} w_k / sum_k w_k."""
    w = np.asarray(weights)
    total = float(np.sum(w, dtype=np.float64))
    if total == 0.0 or not np.isfinite(total):
        return float("nan")
    return float(np.sum(w[np.asarray(lap) < 0], dtype=np.float64) / total)


def concentration(lap: NDArray[Any], mask: Optional[NDArray[np.bool_]] = None) -> float:
    """sum_{lap f < 0} |lap f_k| / sum_k |lap f_k| (identically 1/2 over the whole cell)."""
    v = np.asarray(lap) if mask is None else np.asarray(lap)[mask]
    a = np.abs(v)
    total = float(np.sum(a, dtype=np.float64))
    if total == 0.0 or not np.isfinite(total):
        return float("nan")
    return float(np.sum(a[v < 0], dtype=np.float64) / total)
