"""
pydemi.validate.convergence
===========================
Grid-convergence harness and shell-cutoff sensitivity (spec §5, §11).

* :func:`grid_convergence` -- recompute every descriptor on Fourier-resampled
  (coarser) copies of one density and report the relative change of each,
  so the adequacy of a grid can be judged per descriptor.
* :func:`analytic_convergence` -- the Slater 1s closed forms of §11 against
  grid spacing, from which :func:`recommended_spacing` reads the coarsest
  spacing that meets a tolerance: the recommended minimum mesh comes out of
  the analytic tests rather than a guess.
* :func:`sensitivity_sweep` -- recompute every shell-dependent descriptor
  over a grid of cutoffs (c1, c2) and return a DataFrame, for the cutoff
  sensitivity curves.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..core.geometry import Shells
from ..core.grid import fourier_resample
from ..io.base import Grid, VolumetricData


def _resampled(vd: VolumetricData, shape: Sequence[int]) -> VolumetricData:
    import dataclasses
    lat = vd.lattice

    def g(grid: Optional[Grid]) -> Optional[Grid]:
        return None if grid is None else Grid(fourier_resample(grid.data, shape), lat)
    mv = None
    if vd.magnetization_vector is not None:
        a, b, c = (Grid(fourier_resample(x.data, shape), lat) for x in vd.magnetization_vector)
        mv = (a, b, c)
    rho = g(vd.rho)
    assert rho is not None
    return dataclasses.replace(vd, rho=rho, magnetization=g(vd.magnetization),
                               elf=None if vd.elf is None else Grid(
                                   np.clip(fourier_resample(vd.elf.data, shape), 0.0, 1.0), lat),
                               potential=g(vd.potential), core_density=g(vd.core_density),
                               magnetization_vector=mv, cache={})


def grid_convergence(vd: VolumetricData, scales: Sequence[float] = (0.9, 0.8),
                     **featurize_kwargs: Any) -> Any:
    """DataFrame: every descriptor on the full grid and on grids scaled by ``scales``
    (points per axis), with the relative change |x_s - x_1| / max(|x_1|, tiny)."""
    import pandas as pd

    from ..descriptors import featurize
    base = featurize(vd, **featurize_kwargs)
    out = pd.DataFrame({"descriptor": list(base), "full": list(base.values())})
    for s in scales:
        shape = [max(4, int(round(n * s))) for n in vd.shape]
        f = featurize(_resampled(vd, shape), **featurize_kwargs)
        out[f"x{s:g}"] = [f[k] for k in base]
        out[f"rel_change_x{s:g}"] = [abs(f[k] - base[k]) / max(abs(base[k]), 1e-300)
                                     for k in base]
    return out


def analytic_convergence(spacings: Iterable[float] = (0.12, 0.09, 0.07, 0.05),
                         zeta: float = 3.0, box: float = 4.0, backend: str = "fd") -> Any:
    """Relative errors of the Slater 1s closed forms (spec §11) against grid spacing."""
    import pandas as pd

    from ..descriptors import featurize
    from .analytic import SlaterSuperposition, cubic_cell, slater_lnf, slater_moment
    exact = {"integral": 1.0, "m1": slater_moment(1, zeta), "m2": slater_moment(2, zeta),
             "lnf": slater_lnf(zeta, box ** 3)}
    rows = []
    for h in spacings:
        n = int(round(box / h))
        vd = SlaterSuperposition(cubic_cell(box), [zeta], tol=1e-30).volumetric((n, n, n))
        f = featurize(vd, domains=["bonding"], derivative_backend=backend)
        got = {"integral": float(vd.rho.data.sum() * vd.rho.dV), "m1": f["m1"], "m2": f["m2"],
               "lnf": f["lnf"]}
        row: dict[str, float] = {"spacing": box / n}
        for k, v in exact.items():
            row[f"{k}_rel_error"] = abs(got[k] - v) / abs(v)
        row["zeta_abs"] = f["zeta"]
        rows.append(row)
    return pd.DataFrame(rows)


def recommended_spacing(table: Any, tol: float = 0.01) -> float:
    """Coarsest spacing in ``table`` (from :func:`analytic_convergence`) at which every
    relative error is below ``tol``; NaN when none is."""
    err_cols = [c for c in table.columns if c.endswith("_rel_error")]
    ok = table[(table[err_cols] < tol).all(axis=1)]
    return float(ok["spacing"].max()) if len(ok) else float("nan")


def sensitivity_sweep(vd: VolumetricData, c1_range: Optional[Iterable[float]] = None,
                      c2_range: Optional[Iterable[float]] = None, scaled: bool = False,
                      **featurize_kwargs: Any) -> Any:
    """Every shell-dependent descriptor over a grid of cutoffs (c1, c2).

    Returns a DataFrame with columns c1, c2 and one per descriptor whose
    registry entry requires ``shells``; pairs with c1 >= c2 are skipped.
    """
    import pandas as pd

    from ..constants import SHELL_C1, SHELL_C2
    from ..descriptors import featurize
    from ..descriptors.registry import REGISTRY
    names = [n for n, s in REGISTRY.items() if "shells" in s.requires]
    c1s = list(c1_range) if c1_range is not None else [SHELL_C1]
    c2s = list(c2_range) if c2_range is not None else [SHELL_C2]
    rows = []
    for c1 in c1s:
        for c2 in c2s:
            if c1 >= c2:
                continue
            f = featurize(vd, shells=Shells(float(c1), float(c2), scaled), **featurize_kwargs)
            rows.append({"c1": float(c1), "c2": float(c2), **{n: f[n] for n in names if n in f}})
    return pd.DataFrame(rows)
