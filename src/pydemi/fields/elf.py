"""
pydemi.fields.elf
=================
ELF reconstruction from rho alone (Tsirelson & Stash, Chem. Phys. Lett. 351,
142 (2002)) and the local energy densities that share its kinetic term
(spec §6.2, §8.1). Everything here is in ATOMIC UNITS: rho is converted to
e/bohr^3 (gradient to e/bohr^4, Laplacian to e/bohr^5) first.

    C_F   = (3/10) (3 pi^2)^(2/3) ~= 2.871234
    t_P   = C_F rho^(5/3) + (1/72) |grad rho|^2 / rho + (1/6) lap rho    (= g)
    D_P   = t_P - |grad rho|^2 / (8 rho)
    D_h   = C_F rho^(5/3)
    ELF_D = 1 / (1 + (D_P / D_h)^2)

    g = t_P,   v = (1/4) lap rho - 2 g,   H = g + v                      (Abramov)

Guards: negative (PAW pseudo-)density values are clipped to 0; the
denominators rho and D_h are clamped at ELF_DENOMINATOR_FLOOR; voxels with
rho below RHO_FLOOR_AU are marked invalid -- ELF_D is set to 0 there and
they are left out of every average over g, H and s. ELF_D is therefore in
[0, 1] everywhere by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..constants import (C_F, DENSITY_TO_AU, ELF_DENOMINATOR_FLOOR, GRADIENT_TO_AU,
                         LAPLACIAN_TO_AU, RHO_FLOOR_AU)
from ..io.base import FloatArray

BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, eq=False)
class KineticTerms:
    rho: FloatArray        # e/bohr^3, clipped at 0
    grad2: FloatArray      # |grad rho|^2, e^2/bohr^8
    lap: FloatArray        # e/bohr^5
    valid: BoolArray       # rho > RHO_FLOOR_AU
    tf: FloatArray         # D_h = C_F rho^(5/3)
    w8: FloatArray         # |grad rho|^2 / (8 rho)
    g: FloatArray          # t_P (Kirzhnits); meaningful where valid


def kinetic_terms(rho_A: FloatArray, gradient_norm_A: FloatArray, laplacian_A: FloatArray
                  ) -> KineticTerms:
    """Kirzhnits kinetic terms from rho, |grad rho| and lap rho in Angstrom units."""
    rho = np.maximum(rho_A, 0.0) * DENSITY_TO_AU
    grad2 = (gradient_norm_A * GRADIENT_TO_AU) ** 2
    lap = laplacian_A * LAPLACIAN_TO_AU
    valid = rho > RHO_FLOOR_AU
    r = np.maximum(rho, ELF_DENOMINATOR_FLOOR)
    tf = C_F * rho ** (5.0 / 3.0)
    w = grad2 / r
    g = tf + w / 72.0 + lap / 6.0
    return KineticTerms(rho, grad2, lap, valid, tf, w / 8.0, g)


def elf_d(kt: KineticTerms) -> FloatArray:
    """ELF_D = 1 / (1 + (D_P / D_h)^2), 0 where the density is below the floor."""
    chi = (kt.g - kt.w8) / np.maximum(kt.tf, ELF_DENOMINATOR_FLOOR)
    out = 1.0 / (1.0 + chi * chi)
    return np.asarray(np.where(kt.valid, out, 0.0), dtype=kt.rho.dtype)


def energy_densities(kt: KineticTerms) -> tuple[FloatArray, FloatArray, FloatArray]:
    """(g, v, H) in hartree / bohr^3: v = lap rho / 4 - 2 g, H = g + v."""
    v = kt.lap / 4.0 - 2.0 * kt.g
    return kt.g, v, kt.g + v
