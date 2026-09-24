"""
pydemi.fields.density
=====================
The density and magnetization fields, and the derivative cache every field
shares.

``rho``     read directly (electrons / Angstrom^3)
``abs_m``   |m| for collinear, |m_vec| for non-collinear runs; the zero field
            when the run is not spin-polarized

:func:`derivatives` returns a :class:`Derivatives` object for any field. It
computes the gradient, Laplacian and Hessian eigenvalues at most once per
(field, backend, order) and caches them on the VolumetricData (spec §13): the
fractional Hessian is formed once, gives both the Laplacian and the
eigenvalues, and is freed afterwards.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..constants import FD_ORDER
from ..core import derivatives as D
from ..io.base import FloatArray, Lattice, VolumetricData


def rho(vd: VolumetricData) -> FloatArray:
    return vd.rho.data


def abs_m(vd: VolumetricData) -> FloatArray:
    """|m| on the grid; zeros for a non-spin-polarized run."""
    key = ("field", "abs_m")
    if key not in vd.cache:
        if vd.magnetization_vector is not None:
            mx, my, mz = (g.data for g in vd.magnetization_vector)
            vd.cache[key] = np.sqrt(mx * mx + my * my + mz * mz)
        elif vd.magnetization is not None:
            vd.cache[key] = np.abs(vd.magnetization.data)
        else:
            vd.cache[key] = np.zeros_like(vd.rho.data)
    out: FloatArray = vd.cache[key]
    return out


class Derivatives:
    """Lazily computed, cached derivatives of one field."""

    def __init__(self, values: FloatArray, lattice: Lattice, backend: D.Backend, order: int) -> None:
        self.values = values
        self.lattice = lattice
        self.backend: D.Backend = backend
        self.order = order
        self._gradient: Optional[FloatArray] = None
        self._gradient_norm: Optional[FloatArray] = None
        self._laplacian: dict[str, FloatArray] = {}
        self._eigenvalues: Optional[FloatArray] = None

    @property
    def gradient(self) -> FloatArray:
        """(n1, n2, n3, 3) Cartesian gradient."""
        if self._gradient is None:
            self._gradient = D.gradient(self.values, self.lattice, self.backend, self.order)
        return self._gradient

    @property
    def gradient_norm(self) -> FloatArray:
        if self._gradient_norm is None:
            self._gradient_norm = np.linalg.norm(self.gradient, axis=-1)
        return self._gradient_norm

    def laplacian(self, method: D.LaplacianMethod = "metric") -> FloatArray:
        if method not in self._laplacian:
            self._laplacian[method] = D.laplacian(self.values, self.lattice, method,
                                                  self.backend, self.order)
        return self._laplacian[method]

    @property
    def hessian_eigenvalues(self) -> FloatArray:
        """(n1, n2, n3, 3) ascending lambda1 <= lambda2 <= lambda3."""
        if self._eigenvalues is None:
            H_frac = D.frac_hessian(self.values, self.backend, self.order)
            if "metric" not in self._laplacian:
                G = self.lattice.metric
                lap = np.zeros_like(self.values)
                for (a, b), h in H_frac.items():
                    lap += (G[a, a] if a == b else 2.0 * G[a, b]) * h
                self._laplacian["metric"] = lap
            packed = D.cartesian_hessian_packed(H_frac, self.lattice)
            del H_frac
            self._eigenvalues = D.eigenvalues_packed(packed)
        return self._eigenvalues


def derivatives(vd: VolumetricData, name: str, values: Callable[[VolumetricData], FloatArray],
                backend: D.Backend = "fft", order: int = FD_ORDER) -> Derivatives:
    """Cached :class:`Derivatives` of field ``name`` (``values(vd)`` supplies its data)."""
    key = ("derivatives", name, backend, int(order) if backend == "fd" else 0)
    if key not in vd.cache:
        vd.cache[key] = Derivatives(values(vd), vd.lattice, backend, order)
    out: Derivatives = vd.cache[key]
    return out
