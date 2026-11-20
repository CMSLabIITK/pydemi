"""
pydemi.validate.convergence
===========================
Grid-convergence harness and shell-cutoff sensitivity.

* :func:`grid_convergence` -- recompute every descriptor on Fourier-resampled
  (coarser) copies of one density and report the relative change of each,
  so the adequacy of a grid can be judged per descriptor.
* :func:`analytic_convergence` -- the Slater 1s closed forms against
  grid spacing, from which :func:`recommended_spacing` reads the coarsest
  spacing that meets a tolerance: the recommended minimum mesh comes out of
  the analytic tests rather than a guess.
* :func:`sensitivity_sweep` -- recompute every region-dependent descriptor
  over a grid of sphere factors (a, b) and return a DataFrame, for the
  sensitivity curves.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from ..core.geometry import Shells
from ..core.grid import fourier_resample
from ..io.base import Grid, VolumetricData


def _resampled(vd: VolumetricData, shape: Sequence[int]) -> VolumetricData:
    import dataclasses
    return dataclasses.replace(vd, rho=Grid(fourier_resample(vd.rho.data, shape), vd.lattice),
                               cache={})


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
    """Relative errors of the Slater 1s closed forms against grid spacing."""
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


def sensitivity_sweep(vd: VolumetricData, a_range: Optional[Iterable[float]] = None,
                      b_range: Optional[Iterable[float]] = None, **featurize_kwargs: Any) -> Any:
    """Every region-dependent descriptor over a grid of sphere factors (a, b).

    Returns a DataFrame with columns a, b and one per descriptor whose
    registry entry requires ``shells``; pairs with a >= b are skipped.
    """
    import pandas as pd

    from ..constants import SPHERE_A, SPHERE_B
    from ..descriptors import featurize
    from ..descriptors.registry import REGISTRY
    names = [n for n, s in REGISTRY.items() if "shells" in s.requires]
    a_s = list(a_range) if a_range is not None else [SPHERE_A]
    b_s = list(b_range) if b_range is not None else [SPHERE_B]
    rows = []
    for a in a_s:
        for b in b_s:
            if a >= b:
                continue
            f = featurize(vd, shells=Shells(float(a), float(b)), **featurize_kwargs)
            rows.append({"a": float(a), "b": float(b), **{n: f[n] for n in names if n in f}})
    return pd.DataFrame(rows)
