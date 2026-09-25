"""Grid convergence on real densities: every descriptor (bonding, structural,
magnetic, heterogeneity) on the full grid and on a Fourier-coarsened copy with
80% of the points per axis, for N random structures.

Usage:  python e_convergence.py [N=30]
Output: out/convergence.csv (one row per structure and descriptor)
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import OUT, WORKERS, load, sample_ids
from pydemi.validate.convergence import grid_convergence

DOMAINS = ["bonding", "structural", "magnetic", "heterogeneity"]


def one(mid):
    try:
        df = grid_convergence(load(mid), scales=(0.8,), domains=DOMAINS)
        df.insert(0, "id", mid)
        return df
    except Exception as exc:                                    # noqa: BLE001
        return pd.DataFrame([{"id": mid, "error": repr(exc)}])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    with ProcessPoolExecutor(WORKERS) as pool:
        df = pd.concat(list(pool.map(one, sample_ids(n, seed=11), chunksize=1)), ignore_index=True)
    df.to_csv(OUT / "convergence.csv", index=False)
    ok = df[df["error"].isna()] if "error" in df else df
    ok = ok[np.isfinite(ok["rel_change_x0.8"])]
    per = ok.groupby("id")["rel_change_x0.8"].apply(lambda s: (s > 0.02).mean())
    print(f"{ok['id'].nunique()} structures, {ok['descriptor'].nunique()} descriptors")
    print(f"share of descriptors changing > 2%: median {per.median():.3f}, range {per.min():.3f}-{per.max():.3f}")
    med = ok.groupby("descriptor")["rel_change_x0.8"].median().sort_values(ascending=False)
    with pd.option_context("display.max_rows", 300):
        print(med.to_string(float_format=lambda x: f"{x:.2e}"))
