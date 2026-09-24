"""Validation: analytic densities, invariance harness, grid convergence, ELF fidelity."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def elf_fidelity(elf_true: NDArray[Any], elf_reconstructed: NDArray[Any]) -> dict[str, float]:
    """Agreement of a reconstructed ELF with a true one over voxels (spec §6.2).

    Returns Pearson r, mean absolute error and root-mean-square error. A
    validation metric, not a descriptor. Resample both onto one grid first
    (``pydemi.core.grid.linear_resample``): ELFCAR is usually on a coarser grid.
    """
    a = np.asarray(elf_true, dtype=np.float64).ravel()
    b = np.asarray(elf_reconstructed, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"grids differ: {np.shape(elf_true)} vs {np.shape(elf_reconstructed)}")
    d = b - a
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return {"pearson_r": r, "mae": float(np.mean(np.abs(d))), "rmse": float(np.sqrt(np.mean(d * d)))}
