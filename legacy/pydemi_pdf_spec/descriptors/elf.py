"""
pydemi.descriptors.elf
----------------------
Family B -- ELF descriptors (reference entries 48-52), run on any ELF-like
field: the reconstructed ELF_D (always available, from rho alone) or a
true ELFCAR when one was loaded.

Names carry the source so a column never mixes the two: the reference
names (``f_ELF_localized`` ...) for ELFCAR, and ``ELFD`` in place of
``ELF`` (``f_ELFD_localized`` ...) for the reconstruction.

Core shells show ELF ~ 1 from shell structure, not bonding, so the
fractions and averages are restricted to the bonding shell; ``zeta_ELF``
excludes the core shell because ELF gradients there are sharper and noisier
than rho gradients.
"""

from typing import Optional

import numpy as np
from scipy import ndimage

from ..engine import Engine
from ..shells import Shells
from .primitives import gradient_anisotropy, safe_div

SWEEP_POINTS = (0.25, 0.75)


def threshold_sweep(elf_bond: np.ndarray, thresholds=None):
    """f_ELF(t) = fraction of bonding-shell voxels with ELF > t."""
    t = np.linspace(0.0, 1.0, 101) if thresholds is None else np.asarray(thresholds)
    v = np.sort(np.asarray(elf_bond).ravel())
    if v.size == 0:
        return t, np.full(t.shape, np.nan)
    return t, 1.0 - np.searchsorted(v, t, side="right") / v.size


def sweep_inflection(elf_bond: np.ndarray, bins: int = 50) -> float:
    """Threshold of steepest descent of f_ELF(t), i.e. the modal ELF value.

    -d f_ELF / dt is the ELF distribution over the bonding shell, so its
    maximum (the curve's inflection) is the histogram mode.
    """
    v = np.asarray(elf_bond).ravel()
    if v.size == 0:
        return float("nan")
    hist, edges = np.histogram(v, bins=bins, range=(0.0, 1.0))
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


def elf_family(engine: Engine, field: str, prefix: str,
               shells: Optional[Shells] = None) -> dict:
    """Entries 48-52 for the ELF-like field ``field``; names use ``prefix``."""
    f = engine[field]
    shape = f.grid.shape
    masks = engine.shell_masks(shape, shells)
    elf = f.values
    bond = elf[masks.bond]
    core = elf[masks.core]
    _, curve = threshold_sweep(bond, SWEEP_POINTS)

    out = {
        f"f_{prefix}_localized": float(np.mean(bond > 0.5)) if bond.size else float("nan"),
        f"{prefix}_bond_avg": float(np.mean(bond)) if bond.size else float("nan"),
        f"zeta_{prefix}": gradient_anisotropy(f.gradient, engine.geometry(shape).direction,
                                              mask=~masks.core),
    }
    for t, value in zip(SWEEP_POINTS, curve):
        out[f"{prefix}_threshold_sweep_{int(round(t * 100)):03d}"] = float(value)
    out[f"{prefix}_threshold_sweep_inflection"] = sweep_inflection(bond)
    out[f"{prefix}_core_valence_contrast"] = safe_div(
        np.mean(core) if core.size else np.nan, out[f"{prefix}_bond_avg"])
    return out


def resample(values: np.ndarray, shape) -> np.ndarray:
    """Periodic trilinear resampling of a grid field onto ``shape``.

    Exact subsampling when the target grid divides the source grid (VASP's
    ELFCAR is usually on NGX, half of CHGCAR's NGXF).
    """
    src = np.array(values.shape)
    dst = np.array(shape)
    if np.all(src % dst == 0):
        step = src // dst
        return values[::step[0], ::step[1], ::step[2]]
    idx = np.meshgrid(*[np.arange(n) * s / n for n, s in zip(dst, src)], indexing="ij")
    return ndimage.map_coordinates(values, idx, order=1, mode="grid-wrap")


def elf_fidelity(engine: Engine, elf_field: str = "elf", elfd_field: str = "elf_d",
                 shells: Optional[Shells] = None) -> dict:
    """Entry 73: agreement of ELF_D with ELFCAR on the ELFCAR grid.

    Pearson r and mean absolute error over all voxels and over the bonding
    shell, where the comparison is physically meaningful (both ELFs are
    dominated by shell structure, and ELF_D by PAW pseudization, in the core).
    """
    true = engine[elf_field].values
    approx = resample(engine[elfd_field].values, true.shape)
    bond = engine.shell_masks(true.shape, shells).bond

    def stats(a, b):
        if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan"), float(np.mean(np.abs(a - b))) if a.size else float("nan")
        return float(np.corrcoef(a, b)[0, 1]), float(np.mean(np.abs(a - b)))

    r_all, mae_all = stats(approx.ravel(), true.ravel())
    r_bond, mae_bond = stats(approx[bond], true[bond])
    return {"ELFD_fidelity_r": r_all, "ELFD_fidelity_mae": mae_all,
            "ELFD_fidelity_r_bond": r_bond, "ELFD_fidelity_mae_bond": mae_bond}
