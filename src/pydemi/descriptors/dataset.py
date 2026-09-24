"""
pydemi.descriptors.dataset
--------------------------
Quantities defined across a dataset rather than per structure.

Entry 82, ``zeta_ellip_agreement``: the correlation of zeta with
ellip_bond_avg over all structures, which validates zeta against the QTAIM
ellipticity it was built to approximate.
"""

import numpy as np
from scipy import stats


def correlation(x, y) -> dict:
    """Pearson and Spearman correlation over finite pairs, with the pair count."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    n = int(keep.sum())
    if n < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": n}
    return {"pearson": float(stats.pearsonr(x[keep], y[keep])[0]),
            "spearman": float(stats.spearmanr(x[keep], y[keep])[0]),
            "n": n}


def zeta_ellip_agreement(rows) -> dict:
    """Entry 82 from an iterable of descriptor dicts (or a DataFrame's records).

    Uses ``ellip_bond_median`` too, since the mean can be dominated by
    near-degenerate voxels.
    """
    rows = list(rows)
    zeta = [r["zeta"] for r in rows]
    return {"mean": correlation(zeta, [r["ellip_bond_avg"] for r in rows]),
            "median": correlation(zeta, [r["ellip_bond_median"] for r in rows])}
