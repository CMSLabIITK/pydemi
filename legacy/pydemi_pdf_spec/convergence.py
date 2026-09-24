"""
pydemi.convergence
------------------
Per-descriptor grid-convergence report: recompute descriptors on Fourier-
coarsened copies of the grid and flag those that move.

Coarsening by truncating the spectrum reproduces what a smaller FFT mesh
would represent, so if a descriptor changes by more than ``rtol`` between
the native grid and the next-coarser one, the native grid is not
demonstrably converged for it. This is the grid-adequacy test the Euler
check (entry 93) cannot provide: the Euler identity holds on any grid,
whereas the critical-point *counts* here are compared across grids.
"""

from typing import Iterable

import numpy as np

from .descriptors import compute_descriptors
from .engine import Engine
from .resample import resample_engine

# cheap families by default; H (Hirshfeld), A and D are slower or need references
DEFAULT_FAMILIES = ("tier1", "tier3", "B", "F2", "F3", "F4", "F5", "F6", "I1", "G", "I2", "C")


def convergence_report(engine: Engine, factors: Iterable[float] = (1.0, 0.8, 0.6),
                       families: Iterable[str] = DEFAULT_FAMILIES, rtol: float = 0.02,
                       atol: float = 1e-12) -> dict:
    """Descriptors at each grid scale factor, with the change from native.

    Returns {"shapes": [...], "rows": [{"name", "values", "rel_change",
    "converged"}]}, rows sorted by descending relative change. ``rel_change``
    compares the native grid with the next-coarser one; ``converged`` is
    rel_change <= rtol.
    """
    factors = sorted(set(float(f) for f in factors), reverse=True)
    if factors[0] != 1.0:
        factors = [1.0] + factors
    families = tuple(families)
    results, shapes = [], []
    for f in factors:
        eng = engine if f == 1.0 else resample_engine(engine, scale=f)
        shapes.append(eng.grid().shape)
        results.append(compute_descriptors(eng, families=families))
    rows = []
    for name, v0 in results[0].items():
        values = [r.get(name, np.nan) for r in results]
        v1 = values[1] if len(values) > 1 else v0
        if np.isfinite(v0) and np.isfinite(v1):
            rel = abs(v1 - v0) / max(abs(v0), atol)
        else:
            rel = 0.0 if (np.isnan(v0) and np.isnan(v1)) else float("inf")
        rows.append({"name": name, "values": values, "rel_change": float(rel),
                     "converged": bool(rel <= rtol)})
    rows.sort(key=lambda r: -r["rel_change"])
    return {"factors": factors, "shapes": shapes, "rows": rows, "rtol": rtol}


def format_report(report: dict, limit: int = 40) -> str:
    head = "  ".join(f"{str(s):>16s}" for s in report["shapes"])
    lines = [f"{'descriptor':34s} {head}  rel.change",
             "-" * (48 + 18 * len(report["shapes"]))]
    for r in report["rows"][:limit]:
        vals = "  ".join(f"{v:16.6g}" for v in r["values"])
        flag = "" if r["converged"] else "  NOT CONVERGED"
        lines.append(f"{r['name']:34s} {vals}  {r['rel_change']:.2e}{flag}")
    n_bad = sum(not r["converged"] for r in report["rows"])
    lines.append(f"{n_bad} of {len(report['rows'])} descriptors change by more than "
                 f"{report['rtol']:.0%} on the next-coarser grid")
    return "\n".join(lines)
