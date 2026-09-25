"""The four anisotropy descriptors under every derivative scheme pydemi offers.

For N random structures: FFT (default) and central differences of order 2, 4, 6, 8,
plus FFT with the diagonal Laplacian (which changes ELF_D, hence zeta_ELF, only).

Usage:  python derivative_sensitivity.py [N=100]   -> ../data/derivative_sensitivity.csv
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "paper" / "analysis"))
from common import load, sample_ids  # noqa: E402

from pydemi.descriptors.registry import REGISTRY, Sentinel, make_options  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data"
NAMES = ["zeta", "zeta_ELF", "T_eigenvalues_t1", "T_eigenvalues_t2", "T_eigenvalues_t3", "charge_FA"]
SCHEMES = {"fft": {}, "fd2": {"derivative_backend": "fd", "fd_order": 2},
           "fd4": {"derivative_backend": "fd", "fd_order": 4},
           "fd6": {"derivative_backend": "fd", "fd_order": 6},
           "fd8": {"derivative_backend": "fd", "fd_order": 8},
           "fft_diagonal": {"laplacian_method": "diagonal"}}


def one(mid):
    try:
        vd = load(mid)
        row = {"id": mid}
        for tag, kw in SCHEMES.items():
            v = vd.with_options(make_options(**kw))
            for n in NAMES:
                r = REGISTRY[n].func(v)
                row[f"{n}@{tag}"] = r.value if isinstance(r, Sentinel) else float(r)
        return row
    except Exception as exc:                                    # noqa: BLE001
        return {"id": mid, "error": repr(exc)}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    with ProcessPoolExecutor(8) as ex:
        df = pd.DataFrame(list(ex.map(one, sample_ids(n, seed=31), chunksize=1)))
    df.to_csv(OUT / "derivative_sensitivity.csv", index=False)
    ok = df[df["error"].isna()] if "error" in df else df
    print(len(ok), "structures")
    rows = []
    for name in NAMES:
        ref = ok[f"{name}@fft"]
        for tag in list(SCHEMES)[1:]:
            v = ok[f"{name}@{tag}"]
            rel = (v - ref).abs() / np.maximum(np.maximum(v.abs(), ref.abs()), 1e-12)
            rows.append({"descriptor": name, "scheme": tag, "median": rel.median(), "p90": rel.quantile(0.9),
                         "spearman": ref.rank().corr(v.rank())})
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "derivative_sensitivity_summary.csv", index=False)
    print(s.pivot(index="descriptor", columns="scheme", values="median").to_string(float_format=lambda x: f"{x:.1e}"))
    print(s.pivot(index="descriptor", columns="scheme", values="spearman").round(4).to_string())
