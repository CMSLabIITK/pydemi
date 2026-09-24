"""
pydemi.operators.fractions
==========================
Shell fractions of any field (spec §7):

    f_shell = sum_{k in shell} w_k / sum_k w_k

NaN when sum_k w_k = 0.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


def shell_fraction(weights: NDArray[Any], shell: NDArray[np.bool_],
                   within: Optional[NDArray[np.bool_]] = None) -> float:
    """sum_{k in shell} w_k / sum_k w_k; ``within`` restricts both sums (e.g. to Delta rho > 0)."""
    w = np.asarray(weights)
    if within is not None:
        w = np.where(within, w, 0.0)
    total = float(np.sum(w, dtype=np.float64))
    if total == 0.0 or not np.isfinite(total):
        return float("nan")
    return float(np.sum(w[shell], dtype=np.float64) / total)
