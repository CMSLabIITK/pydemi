"""The density grid VASP would use for a cell, from the lattice and ENCUT alone.

A structure-only prediction needs a grid; using the one of the DFT calculations the
model was trained on (PREC = Accurate, ENCUT = 500 eV) makes predicted and DFT
descriptors comparable. The rule reproduces the CHGCAR grid of all 6,059 runs of the
training dataset (VASP 5.3.3 and 5.4.4).
"""

from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from pydemi.io.base import Lattice

# VASP's own constants (constant.inc), so sizes on a rounding boundary come out as in VASP
_RYTOEV = 13.605826
_AUTOA = 0.529177249
_WFACT = {"accurate": 2.0, "normal": 1.5}


def _fft_friendly(n: int) -> bool:
    for p in (2, 3, 5, 7):
        while n % p == 0:
            n //= p
    return n == 1


def _next_even_fft(n: int) -> int:
    n = max(int(n), 2)
    while n % 2 or not _fft_friendly(n):
        n += 1
    return n


def vasp_grid_shape(lattice: Any, encut: float = 500.0, prec: str = "accurate") -> Tuple[int, int, int]:
    """Charge-density grid (NGXF, NGYF, NGZF) that VASP uses for this cell.

    ``lattice``: a :class:`~pydemi.Lattice` or a 3x3 matrix with the lattice vectors as
    rows (Angstrom); ``encut`` in eV; ``prec`` "accurate" (validated) or "normal".

    VASP 5 takes the wavefunction grid along a_i as the smallest even size with prime
    factors 2, 3, 5, 7 that is at least nint(2 WFACT x_i), with x_i = |a_i| sqrt(ENCUT/Ry)
    / (2 pi) in atomic units and WFACT = 2 for Accurate (1.5 for Normal), and doubles it
    for the density grid.
    """
    key = prec.lower()
    if key not in _WFACT:
        raise ValueError(f"prec must be 'accurate' or 'normal', got {prec!r}")
    matrix = lattice.matrix if isinstance(lattice, Lattice) else np.asarray(lattice, dtype=float)
    lengths = np.linalg.norm(np.asarray(matrix, dtype=float).reshape(3, 3), axis=1)
    x = math.sqrt(encut / _RYTOEV) * (lengths / _AUTOA) / (2.0 * math.pi)
    coarse = [_next_even_fft(int(math.floor(2.0 * _WFACT[key] * xi + 0.5))) for xi in x]
    return (2 * coarse[0], 2 * coarse[1], 2 * coarse[2])


def grid_points(lattice: Lattice, shape: Tuple[int, int, int], start: int, stop: int) -> NDArray[np.float64]:
    """Cartesian positions of grid points ``start:stop`` in C order; point [i, j, k] is at
    fractional (i/n1, j/n2, k/n3)."""
    idx = np.unravel_index(np.arange(start, stop), shape)
    frac = np.stack([i / n for i, n in zip(idx, shape)], axis=1)
    return np.asarray(frac @ lattice.matrix, dtype=np.float64)
