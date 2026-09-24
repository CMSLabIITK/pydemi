"""
pydemi.core.derivatives
=======================
Gradient, Laplacian and Hessian on a periodic, possibly non-orthogonal grid
(spec §4).

Let A hold the lattice vectors as rows and ``B = inv(A)^T`` the reciprocal
vectors b1, b2, b3 as rows, WITHOUT the 2 pi factor. With fractional
coordinates u = (u1, u2, u3):

    gradient    grad f  = sum_a b_a (df/du_a)
    Laplacian   lap f   = sum_{a,b} G[a, b] d^2f/(du_a du_b),   G = B B^T
    Hessian     H_cart  = B^T H_frac B,   H_frac[a, b] = d^2f/(du_a du_b)

The Laplacian includes the off-diagonal metric terms (``method="metric"``,
the default). ``method="diagonal"`` keeps only G[a, a] d^2f/du_a^2, which
is exact only for orthorhombic cells; it exists solely so the error of that
common shortcut can be quantified.

Backends
--------
``"fft"`` (default)
    Spectral derivatives: d/du_a -> 2 pi i m_a for integer frequency m_a.
    Exact for band-limited data (plane-wave grid densities are), no stencil
    error. The Nyquist frequency of an even axis is dropped from every
    odd-order factor, so derivatives of real data stay real and the
    Hessian symmetric. Sharp features (PAW core peaks, an all-electron
    cusp) ring: the spectral Laplacian oscillates around a cusp, and in
    near-empty regions its absolute round-off (~1e-16 of the peak value)
    decides the sign, so sign-thresholded descriptors (lnf) are noisy there.
``"fd"``
    Central finite differences of order 2, 4 (default), 6 or 8 with modular
    index wrapping. Mixed derivatives apply the first-derivative stencil
    along both axes.

Grid-convergence results differ between the backends, so the backend is
recorded in the output metadata.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import fft as sfft

from ..constants import FD_ORDER, HESSIAN_CHUNK
from ..io.base import FloatArray, Lattice

Backend = Literal["fft", "fd"]
LaplacianMethod = Literal["metric", "diagonal"]

# central-difference coefficients c_k, k = 1..p/2 (first) and k = 0..p/2 (second)
_FIRST = {
    2: (1 / 2,),
    4: (2 / 3, -1 / 12),
    6: (3 / 4, -3 / 20, 1 / 60),
    8: (4 / 5, -1 / 5, 4 / 105, -1 / 280),
}
_SECOND = {
    2: (-2.0, 1.0),
    4: (-5 / 2, 4 / 3, -1 / 12),
    6: (-49 / 18, 3 / 2, -3 / 20, 1 / 90),
    8: (-205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560),
}

#: packed storage order of the symmetric 3x3 fractional Hessian
PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _check(backend: str, order: int) -> None:
    if backend not in ("fft", "fd"):
        raise ValueError(f"derivative backend must be 'fft' or 'fd', got {backend!r}")
    if backend == "fd" and order not in _FIRST:
        raise ValueError(f"finite-difference order must be one of {sorted(_FIRST)}, got {order}")


# ----------------------------------------------------------------------
# spectral helpers
# ----------------------------------------------------------------------

def _frequencies(shape: Sequence[int]) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Per-axis integer frequencies on the rfftn half-grid (broadcastable).

    Returns (m_full, m_odd); m_odd has the Nyquist frequency of every even
    axis zeroed, for use in odd-order factors.
    """
    m_full, m_odd = [], []
    for ax, n in enumerate(shape):
        m = sfft.rfftfreq(n, 1.0 / n) if ax == 2 else sfft.fftfreq(n, 1.0 / n)
        mo = m.copy()
        if n % 2 == 0:
            mo[np.isclose(np.abs(mo), n / 2)] = 0.0
        s = [1, 1, 1]
        s[ax] = m.size
        m_full.append(m.reshape(s))
        m_odd.append(mo.reshape(s))
    return m_full, m_odd


def _rfft(f: FloatArray) -> NDArray[Any]:
    return np.asarray(sfft.rfftn(f, axes=(0, 1, 2)))


def _irfft(F: NDArray[Any], shape: Sequence[int], dtype: Any) -> FloatArray:
    return np.asarray(sfft.irfftn(F, s=tuple(shape), axes=(0, 1, 2)), dtype=dtype)


def g_squared(lattice: Lattice, shape: Sequence[int]) -> NDArray[np.float64]:
    """|G|^2 = (2 pi)^2 |B^T m|^2 on the rfftn half-grid (1/Angstrom^2).

    Consistent with the spectral Laplacian: Nyquist terms are dropped from
    the cross terms, as in :func:`laplacian`.
    """
    m_full, m_odd = _frequencies(shape)
    G = lattice.metric
    out: NDArray[np.float64] = np.zeros(np.broadcast_shapes(*(m.shape for m in m_full)))
    for a in range(3):
        for b in range(3):
            mm = m_full[a] ** 2 if a == b else m_odd[a] * m_odd[b]
            out = out + (2 * np.pi) ** 2 * G[a, b] * mm
    return out


# ----------------------------------------------------------------------
# fractional derivatives
# ----------------------------------------------------------------------

def _d1_fd(f: FloatArray, axis: int, n: int, order: int) -> FloatArray:
    out = np.zeros_like(f)
    for k, c in enumerate(_FIRST[order], start=1):
        out += c * (np.roll(f, -k, axis) - np.roll(f, k, axis))
    return out * n


def _d2_fd(f: FloatArray, axis: int, n: int, order: int) -> FloatArray:
    coeffs = _SECOND[order]
    out = coeffs[0] * f
    for k, c in enumerate(coeffs[1:], start=1):
        out = out + c * (np.roll(f, -k, axis) + np.roll(f, k, axis))
    return out * (n * n)


def frac_gradient(f: FloatArray, backend: Backend = "fft", order: int = FD_ORDER) -> list[FloatArray]:
    """[df/du_1, df/du_2, df/du_3]."""
    _check(backend, order)
    shape = f.shape
    if backend == "fd":
        return [_d1_fd(f, a, shape[a], order) for a in range(3)]
    F = _rfft(f)
    _, m_odd = _frequencies(shape)
    return [_irfft(2j * np.pi * m_odd[a] * F, shape, f.dtype) for a in range(3)]


def frac_hessian(f: FloatArray, backend: Backend = "fft", order: int = FD_ORDER,
                 pairs: Sequence[tuple[int, int]] = PAIRS) -> dict[tuple[int, int], FloatArray]:
    """{(a, b): d^2 f/(du_a du_b)} for the requested index pairs (a <= b)."""
    _check(backend, order)
    shape = f.shape
    out: dict[tuple[int, int], FloatArray] = {}
    if backend == "fd":
        first: dict[int, FloatArray] = {}
        for a, b in pairs:
            if a == b:
                out[(a, b)] = _d2_fd(f, a, shape[a], order)
            else:
                if a not in first:
                    first[a] = _d1_fd(f, a, shape[a], order)
                out[(a, b)] = _d1_fd(first[a], b, shape[b], order)
        return out
    F = _rfft(f)
    m_full, m_odd = _frequencies(shape)
    for a, b in pairs:
        mm = m_full[a] ** 2 if a == b else m_odd[a] * m_odd[b]
        out[(a, b)] = _irfft(-(2 * np.pi) ** 2 * mm * F, shape, f.dtype)
    return out


# ----------------------------------------------------------------------
# Cartesian derivatives
# ----------------------------------------------------------------------

def gradient(f: FloatArray, lattice: Lattice, backend: Backend = "fft",
             order: int = FD_ORDER) -> FloatArray:
    """Cartesian gradient, shape (n1, n2, n3, 3), units [f]/Angstrom."""
    du = frac_gradient(f, backend, order)
    B = lattice.reciprocal
    out = np.empty(f.shape + (3,), dtype=f.dtype)
    for c in range(3):
        out[..., c] = B[0, c] * du[0] + B[1, c] * du[1] + B[2, c] * du[2]
    return out


def laplacian(f: FloatArray, lattice: Lattice, method: LaplacianMethod = "metric",
              backend: Backend = "fft", order: int = FD_ORDER) -> FloatArray:
    """Cartesian Laplacian, units [f]/Angstrom^2.

    ``method="metric"``: sum_{a,b} G[a, b] d^2f/(du_a du_b), G = B B^T (exact).
    ``method="diagonal"``: sum_a G[a, a] d^2f/du_a^2 (exact only for orthorhombic cells).
    """
    if method not in ("metric", "diagonal"):
        raise ValueError(f"laplacian method must be 'metric' or 'diagonal', got {method!r}")
    G = lattice.metric
    pairs = [(a, b) for a, b in PAIRS if a == b or (method == "metric" and abs(G[a, b]) > 0.0)]
    H = frac_hessian(f, backend, order, pairs)
    out = np.zeros_like(f)
    for (a, b), h in H.items():
        out += (G[a, a] if a == b else 2.0 * G[a, b]) * h
    return out


def cartesian_hessian_packed(H_frac: dict[tuple[int, int], FloatArray],
                             lattice: Lattice) -> dict[tuple[int, int], FloatArray]:
    """H_cart = B^T H_frac B, component by component (packed, c <= d)."""
    B = lattice.reciprocal
    out: dict[tuple[int, int], FloatArray] = {}
    for c, d in PAIRS:
        acc = np.zeros_like(next(iter(H_frac.values())))
        for a in range(3):
            for b in range(3):
                coef = B[a, c] * B[b, d]
                if coef != 0.0:
                    acc += coef * H_frac[(min(a, b), max(a, b))]
        out[(c, d)] = acc
    return out


def hessian(f: FloatArray, lattice: Lattice, backend: Backend = "fft",
            order: int = FD_ORDER) -> FloatArray:
    """Cartesian Hessian, shape (n1, n2, n3, 3, 3), symmetric, units [f]/Angstrom^2."""
    packed = cartesian_hessian_packed(frac_hessian(f, backend, order), lattice)
    out = np.empty(f.shape + (3, 3), dtype=f.dtype)
    for (c, d), h in packed.items():
        out[..., c, d] = h
        out[..., d, c] = h
    return out


def eigenvalues_packed(packed: dict[tuple[int, int], FloatArray],
                       chunk: int = HESSIAN_CHUNK) -> FloatArray:
    """Ascending eigenvalues lambda1 <= lambda2 <= lambda3 of a packed symmetric field.

    Vectorized ``eigvalsh`` over chunks of voxels, so the full (N, 3, 3)
    array is never held at once.
    """
    first = next(iter(packed.values()))
    shape = first.shape
    flat = {k: v.reshape(-1) for k, v in packed.items()}
    n = first.size
    out = np.empty((n, 3), dtype=first.dtype)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        M = np.empty((e - s, 3, 3), dtype=first.dtype)
        for (c, d), v in flat.items():
            M[:, c, d] = v[s:e]
            M[:, d, c] = v[s:e]
        out[s:e] = np.linalg.eigvalsh(M)
    return out.reshape(shape + (3,))


def hessian_eigenvalues(f: FloatArray, lattice: Lattice, backend: Backend = "fft",
                        order: int = FD_ORDER) -> FloatArray:
    """(n1, n2, n3, 3) ascending eigenvalues of the Cartesian Hessian."""
    return eigenvalues_packed(cartesian_hessian_packed(frac_hessian(f, backend, order), lattice))
