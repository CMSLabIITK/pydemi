"""
pydemi.fields.potential
=======================
The electronic Hartree potential on the density grid (spec §6.3), in eV,
from the CHGCAR density by one FFT:

    V_H(G) = 4 pi rho(G) / |G|^2   for G != 0,   V_H(0) = 0,    |G| = 2 pi |B^T n|

for integer triples n, times e^2 / (4 pi eps0) = 14.40 eV Angstrom. This is
the potential energy of an electron in the field of the other electrons
only. The G = 0 term is arbitrary; the potential is referenced to its cell
average, so potential descriptors do not depend on that convention.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import fft as sfft

from ..constants import COULOMB_EV_ANGSTROM
from ..io.base import FloatArray, Lattice

F64 = NDArray[np.float64]


def g_squared_full(lattice: Lattice, shape: Sequence[int]) -> F64:
    """|G|^2 = (2 pi)^2 |B^T n|^2 on the rfftn half-grid, every frequency included."""
    n1, n2, n3 = (int(m) for m in shape)
    m = [sfft.fftfreq(n1, 1.0 / n1)[:, None, None], sfft.fftfreq(n2, 1.0 / n2)[None, :, None],
         sfft.rfftfreq(n3, 1.0 / n3)[None, None, :]]
    B = lattice.reciprocal
    out = np.zeros((n1, n2, n3 // 2 + 1))
    for c in range(3):
        Gc = 2 * np.pi * (B[0, c] * m[0] + B[1, c] * m[1] + B[2, c] * m[2])
        out = out + Gc * Gc
    return np.asarray(out, dtype=np.float64)


def _poisson(F: NDArray[np.complex128], G2: F64, shape: Sequence[int]) -> F64:
    with np.errstate(divide="ignore", invalid="ignore"):
        VG = np.where(G2 > 0, 4.0 * np.pi * COULOMB_EV_ANGSTROM * F / G2, 0.0)
    return np.asarray(sfft.irfftn(VG, s=tuple(shape), axes=(0, 1, 2)), dtype=np.float64)


def hartree_potential(rho: FloatArray, lattice: Lattice) -> F64:
    """Electronic Hartree potential (eV), cell average zero."""
    F = np.asarray(sfft.rfftn(np.asarray(rho, dtype=np.float64), axes=(0, 1, 2)))
    return _poisson(F, g_squared_full(lattice, rho.shape), rho.shape)


def centred(values: FloatArray) -> F64:
    """``values`` shifted to zero cell average."""
    v = np.asarray(values, dtype=np.float64)
    return np.asarray(v - v.mean(), dtype=np.float64)
