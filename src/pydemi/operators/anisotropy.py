"""
pydemi.operators.anisotropy
===========================
Directional structure of any field's gradient (spec §7):

    gradient anisotropy   zeta = 1 - sum_k |grad f_k . u_k| / sum_k |grad f_k|
    anisotropy tensor     T_ab = sum_k d_a f_k d_b f_k / sum_k |grad f_k|^2     (tr T = 1)

u_k is the unit vector from the nearest nucleus to voxel k. zeta = 0 when
every gradient is radial (a superposition of well-separated spherical atoms)
and grows as density concentrates off the radial directions. Both are NaN
when the gradient vanishes everywhere (uniform density).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


def gradient_anisotropy(gradient: NDArray[Any], direction: NDArray[Any],
                        mask: Optional[NDArray[np.bool_]] = None) -> float:
    """zeta = 1 - sum_k |grad f_k . u_k| / sum_k |grad f_k| (over ``mask`` voxels when given)."""
    g: NDArray[Any] = np.asarray(gradient).reshape(-1, 3)
    u: NDArray[Any] = np.asarray(direction).reshape(-1, 3)
    if mask is not None:
        keep = np.asarray(mask).ravel()
        g, u = g[keep], u[keep]
    denom = float(np.sum(np.linalg.norm(g, axis=1), dtype=np.float64))
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")
    radial = float(np.sum(np.abs(np.einsum("ij,ij->i", g, u)), dtype=np.float64))
    return 1.0 - radial / denom


def anisotropy_tensor(gradient: NDArray[Any]) -> NDArray[np.float64]:
    """T_ab = sum_k d_a f_k d_b f_k / sum_k |grad f_k|^2; NaN-filled when the gradient vanishes."""
    g = np.asarray(gradient, dtype=np.float64).reshape(-1, 3)
    T = g.T @ g
    trace = float(np.trace(T))
    if trace == 0.0 or not np.isfinite(trace):
        return np.full((3, 3), np.nan)
    return np.asarray(T / trace, dtype=np.float64)


def fractional_anisotropy(T: NDArray[Any]) -> float:
    """sqrt(3/2) ||T - I/3||_F / ||T||_F: 0 isotropic, 1 when every gradient is parallel."""
    Tm = np.asarray(T, dtype=np.float64)
    norm = float(np.linalg.norm(Tm))
    if norm == 0.0 or not np.isfinite(norm):
        return float("nan")
    return float(np.sqrt(1.5) * np.linalg.norm(Tm - np.eye(3) / 3.0) / norm)
