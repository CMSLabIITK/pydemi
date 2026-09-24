"""
pydemi.descriptors.kinetic
--------------------------
Fields built from the second-order Kirzhnits kinetic energy density,
all in atomic units (hartree, bohr):

    g = t_P = C_F rho^{5/3} + (1/72) |grad rho|^2 / rho + (1/6) lap rho

F1 -- approximate ELF (Tsirelson & Stash, Chem. Phys. Lett. 351, 142 (2002)):

    D_P   = t_P - |grad rho|^2 / (8 rho)
    ELF_D = 1 / (1 + (D_P / (C_F rho^{5/3}))^2)

F5 -- local energy densities (Abramov, Acta Cryst. A53, 264 (1997)):

    v = (1/4) lap rho - 2 g        (local virial theorem)
    H = g + v = (1/4) lap rho - g

    f_H_negative  (1/N_bond) sum_{bond} 1(H_k < 0)   Cremer-Kraka criterion
    H_bond_mean   <H_k> over the bonding shell        hartree / bohr^3
    G_over_rho    <g_k / rho_k> over the bonding shell  hartree / electron

The kinetic terms are computed once per source field and cached on the
engine, so F5 after F1 is free.

Low density
-----------
Negative pseudo-density values (possible in PAW CHGCARs) are clipped to 0,
and voxels with rho below ``RHO_FLOOR_AU`` are marked invalid: ELF_D is set
to 0 there and they are excluded from every F5 average. The gradient
expansion is not meaningful at such densities anyway.

These are approximations derived for all-electron densities; applied to
the PAW pseudo-density they are unreliable inside the core shell, which is
why every descriptor here is restricted to the bonding shell.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..constants import C_F, DENSITY_TO_AU, GRADIENT_TO_AU, LAPLACIAN_TO_AU
from ..engine import RHO, Engine
from ..field import Field
from ..shells import Shells

RHO_FLOOR_AU = 1e-8
ELF_D = "elf_d"


@dataclass(frozen=True, eq=False)
class KineticTerms:
    rho: np.ndarray      # e / bohr^3, clipped at 0
    grad2: np.ndarray    # |grad rho|^2, e^2 / bohr^8
    lap: np.ndarray      # e / bohr^5
    valid: np.ndarray    # rho > RHO_FLOOR_AU
    tf: np.ndarray       # C_F rho^{5/3}
    weizsacker8: np.ndarray  # |grad rho|^2 / (8 rho), 0 where invalid
    g: np.ndarray        # Kirzhnits t_P, NaN where invalid


def kinetic_terms(engine: Engine, field: str = RHO) -> KineticTerms:
    key = ("kinetic", field)
    if key not in engine.cache:
        f = engine[field]
        rho = np.maximum(f.values, 0.0) * DENSITY_TO_AU
        grad2 = f.gradient_norm ** 2 * GRADIENT_TO_AU ** 2
        lap = f.laplacian * LAPLACIAN_TO_AU
        valid = rho > RHO_FLOOR_AU
        safe = np.where(valid, rho, 1.0)
        tf = C_F * rho ** (5.0 / 3.0)
        w = np.where(valid, grad2 / safe, 0.0)
        g = np.where(valid, tf + w / 72.0 + lap / 6.0, np.nan)
        engine.cache[key] = KineticTerms(rho, grad2, lap, valid, tf, w / 8.0, g)
    return engine.cache[key]


def elf_d_values(kt: KineticTerms) -> np.ndarray:
    """ELF_D on the grid (0 where the density is below the floor)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        chi = (kt.g - kt.weizsacker8) / kt.tf
        elf = 1.0 / (1.0 + chi ** 2)
    return np.where(kt.valid, elf, 0.0)


def elf_d_field(engine: Engine, field: str = RHO) -> Field:
    """Compute ELF_D (entry 72) and register it on the engine as ``elf_d``.

    Registered as a regular field, so Family B descriptors (and anything
    else) can run on it exactly as on a real ELFCAR.
    """
    name = ELF_D if field == RHO else f"{ELF_D}_{field}"
    if name not in engine:
        engine.add_field(name, elf_d_values(kinetic_terms(engine, field)))
    return engine[name]


def energy_densities(engine: Engine, field: str = RHO) -> dict:
    """g, v, H (hartree / bohr^3) on the grid; NaN where invalid."""
    kt = kinetic_terms(engine, field)
    v = kt.lap / 4.0 - 2.0 * kt.g
    return {"g": kt.g, "v": v, "H": kt.g + v}


def energy_family(engine: Engine, field: str = RHO,
                  shells: Optional[Shells] = None) -> dict:
    """F5: entries 83-85."""
    kt = kinetic_terms(engine, field)
    H = energy_densities(engine, field)["H"]
    bond = engine.shell_masks(engine[field].grid.shape, shells).bond & kt.valid
    n = int(bond.sum())
    if n == 0:
        return {"f_H_negative": float("nan"), "H_bond_mean": float("nan"),
                "G_over_rho": float("nan")}
    return {
        "f_H_negative": float(np.mean(H[bond] < 0.0)),
        "H_bond_mean": float(np.mean(H[bond])),
        "G_over_rho": float(np.mean(kt.g[bond] / kt.rho[bond])),
    }

