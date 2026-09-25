"""The critical-point census on real densities.

pydemi counts maxima and minima against the 26 neighbours and saddles on the
Freudenthal link (14 neighbours). A census done entirely on the Freudenthal
link closes (Euler characteristic of the 3-torus = 0), so the hybrid census's
euler_consistency equals the number of extrema the two neighbourhoods classify
differently:
    e = (n_max^26 - n_max^14) - (n_min^26 - n_min^14).
This script checks that identity on N random structures and collects the
dataset-wide statistics from the rerun table.

Usage:  python h_topology.py [N=200]
Output: out/topology.json
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import OUT, WORKERS, load, results, sample_ids
from pydemi.operators.topology import FREUDENTHAL_14, extremum_census, lower_mask


def one(mid):
    try:
        rho = load(mid).rho.data
        c = extremum_census(rho)
        m14 = lower_mask(rho, FREUDENTHAL_14)
        n_max14, n_min14 = int((m14 == (1 << 14) - 1).sum()), int((m14 == 0).sum())
        euler14 = n_max14 - c["n_saddle2"] + c["n_saddle1"] - n_min14
        return {"id": mid, "n_voxels": int(rho.size), "euler": c["euler_consistency"], "euler14": euler14,
                "n_max26": c["n_max"], "n_max14": n_max14, "n_min26": c["n_min"], "n_min14": n_min14}
    except Exception as exc:                                    # noqa: BLE001
        return {"id": mid, "error": repr(exc)}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    with ProcessPoolExecutor(WORKERS) as pool:
        rows = [r for r in pool.map(one, sample_ids(n), chunksize=1)]
    ok = [r for r in rows if "error" not in r]
    ident = sum(r["euler"] == (r["n_max26"] - r["n_max14"]) - (r["n_min26"] - r["n_min14"]) for r in ok)
    df = results()
    e = df["euler_consistency"]
    crit = (df["n_max"] + df["n_min"] + df["n_saddle1"] + df["n_saddle2"]) * df["volume"]
    z = e == 0
    out = {
        "sample": {"n": len(ok), "errors": len(rows) - len(ok),
                   "freudenthal_census_closes": sum(r["euler14"] == 0 for r in ok),
                   "identity_holds": ident,
                   "extra_maxima14_median": float(np.median([r["n_max14"] - r["n_max26"] for r in ok])),
                   "extra_minima14_median": float(np.median([r["n_min14"] - r["n_min26"] for r in ok]))},
        "dataset": {"n": len(df), "n_zero": int(z.sum()), "n_positive": int((e > 0).sum()),
                    "n_negative": int((e < 0).sum()),
                    "abs_e_over_critical_points": [float((e.abs() / crit).quantile(q)) for q in (0.5, 0.9)],
                    "rho_min_negative_share_zero_vs_nonzero": [float((df.rho_min[z] < 0).mean()),
                                                               float((df.rho_min[~z] < 0).mean())],
                    "nnm_share_zero_vs_nonzero": [float((df.n_NNM[z] > 0).mean()), float((df.n_NNM[~z] > 0).mean())],
                    "n_with_nnm": int((df.n_NNM > 0).sum()), "n_with_nnm_paw": int((df.n_NNM_paw > 0).sum()),
                    "max_nnm_count": {k: float(v) for k, v in
                                      (df.n_NNM * df.volume).sort_values(ascending=False).head(3).round().items()}},
        "rows": ok}
    (OUT / "topology.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
