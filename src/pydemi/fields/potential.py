"""
pydemi.fields.potential
=======================
Electrostatic potentials on the density grid (spec §6.3), in eV.

Electronic Hartree term, one FFT:

    V_H(G) = 4 pi rho(G) / |G|^2   for G != 0,   V_H(0) = 0,    |G| = 2 pi |B^T n|

for integer triples n, times e^2 / (4 pi eps0) = 14.40 eV Angstrom. This is
the potential energy of an electron in the field of the other electrons
only. The full electrostatic potential (``"esp"``) adds the ionic term from
Gaussian-smeared ionic charges Z_i (ZVAL for a pseudo-density, the atomic
number for an all-electron density) of width ION_GAUSSIAN_WIDTH:

    V_esp(G) = 4 pi [rho(G) - rho_ion(G)] / |G|^2,
    rho_ion(G) = (1/V) sum_i Z_i exp(-|G|^2 sigma^2 / 2) exp(-i G . R_i)

The G = 0 term is arbitrary; every potential is referenced to its cell
average (a LOCPOT read from file is shifted the same way), so potential
descriptors do not depend on that convention.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import fft as sfft

from ..constants import COULOMB_EV_ANGSTROM, ION_GAUSSIAN_WIDTH
from ..io.base import FloatArray, Lattice, Structure

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


def ionic_form(structure: Structure, charges: F64, shape: Sequence[int],
               sigma: float = ION_GAUSSIAN_WIDTH) -> NDArray[np.complex128]:
    """Grid-FFT coefficients of the Gaussian ionic charge density (same convention as rfftn(rho))."""
    n1, n2, n3 = (int(m) for m in shape)
    lat = structure.lattice
    G2 = g_squared_full(lat, shape)
    m = [sfft.fftfreq(n1, 1.0 / n1), sfft.fftfreq(n2, 1.0 / n2), sfft.rfftfreq(n3, 1.0 / n3)]
    S = np.zeros(G2.shape, dtype=np.complex128)
    for q, u in zip(charges, structure.frac_coords):
        p = [np.exp(-2j * np.pi * mm * u[ax]) for ax, mm in enumerate(m)]
        S += q * p[0][:, None, None] * p[1][None, :, None] * p[2][None, None, :]
    # rho_ion(G) / (1/V) per unit volume, times N for the grid-FFT convention
    return np.asarray(S * np.exp(-0.5 * G2 * sigma * sigma) * (n1 * n2 * n3) / lat.volume)


def electrostatic_potential(rho: FloatArray, structure: Structure, charges: F64,
                            sigma: float = ION_GAUSSIAN_WIDTH) -> F64:
    """Hartree + ionic potential energy of an electron (eV), cell average zero."""
    shape = rho.shape
    F = np.asarray(sfft.rfftn(np.asarray(rho, dtype=np.float64), axes=(0, 1, 2)))
    F = F - ionic_form(structure, charges, shape, sigma)
    return _poisson(F, g_squared_full(structure.lattice, shape), shape)


def centred(values: FloatArray) -> F64:
    """``values`` shifted to zero cell average."""
    v = np.asarray(values, dtype=np.float64)
    return np.asarray(v - v.mean(), dtype=np.float64)


def ion_charges(species: Sequence[str], all_electron: bool,
                zval: Optional[dict[str, float]] = None) -> F64:
    """Z_i for the ionic term: atomic number (all-electron) or ZVAL (pseudo-density)."""
    from ..data import atomic_number, default_zval
    if all_electron:
        return np.array([float(atomic_number(s)) for s in species])
    z = zval or {}
    return np.array([float(z[s]) if s in z else default_zval(s) for s in species])
