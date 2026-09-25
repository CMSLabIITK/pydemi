"""
pydemi.descriptors.primitives
-----------------------------
Field-agnostic reductions shared by every descriptor family: weighted
radial moments, shell fractions, gradient anisotropy, sign fractions.
Tier 1 runs them on rho; the spin, deformation and ELF families run the
same functions on |m|, |delta rho| and ELF.

All reductions return plain floats. Undefined ratios (zero denominator)
return NaN rather than being regularized with an epsilon.
"""

import numpy as np


def safe_div(a: float, b: float) -> float:
    a, b = float(a), float(b)
    if b == 0.0 or not np.isfinite(b) or not np.isfinite(a):
        return float("nan")
    return a / b


def radial_moments(weights, r):
    """Weighted (m1, m2, sigma_r2) of the distance field ``r``.

    m_n = sum_k w_k r_k^n / sum_k w_k
    """
    w = np.asarray(weights, float).ravel()
    r = np.asarray(r, float).ravel()
    W = w.sum()
    m1 = safe_div((w * r).sum(), W)
    m2 = safe_div((w * r * r).sum(), W)
    return m1, m2, m2 - m1 * m1


def shell_fractions(weights, masks):
    """(f_core, f_bond, f_int): share of the total weight in each shell."""
    w = np.asarray(weights, float)
    W = w.sum()
    return (safe_div(w[masks.core].sum(), W),
            safe_div(w[masks.bond].sum(), W),
            safe_div(w[masks.interstitial].sum(), W))


def gradient_anisotropy(gradient, direction, mask=None) -> float:
    """zeta = 1 - sum |g . r_hat| / sum |g|, optionally over ``mask`` voxels."""
    g = np.asarray(gradient, float).reshape(-1, 3)
    u = np.asarray(direction, float).reshape(-1, 3)
    if mask is not None:
        keep = np.asarray(mask, bool).ravel()
        g, u = g[keep], u[keep]
    radial = np.abs(np.einsum("ij,ij->i", g, u)).sum()
    return 1.0 - safe_div(radial, np.linalg.norm(g, axis=1).sum())


def negative_fraction(values, weights=None) -> float:
    """Fraction of voxels (or of total weight) where ``values`` < 0."""
    v = np.asarray(values, float)
    if weights is None:
        return float(np.mean(v < 0))
    w = np.asarray(weights, float)
    return safe_div(w[v < 0].sum(), w.sum())


def negative_magnitude_share(values) -> float:
    """sum_{v<0} |v| / sum |v|: the continuous analogue of negative_fraction."""
    v = np.abs(np.asarray(values, float))
    neg = np.asarray(values) < 0
    return safe_div(v[neg].sum(), v.sum())
