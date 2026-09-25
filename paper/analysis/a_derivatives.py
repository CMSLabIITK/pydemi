"""Derivative schemes on real VASP densities.

For N random structures: every bonding + structural descriptor with the
default FFT derivatives, with 4th- and 2nd-order finite differences, and with
FFT but the diagonal (orthogonal-cell) Laplacian. Also the cell integral of
the discrete Laplacian (FFT and FD4), which vanishes for a periodic density.

Usage:  python a_derivatives.py [N=200]   (--summary-only: redo the summary from out/derivatives.csv)
Output: out/derivatives.csv, out/derivatives_summary.csv
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import pydemi
from common import OUT, WORKERS, load, sample_ids
from pydemi.fields.density import derivatives

SCHEMES = {"fft": {}, "fd4": {"derivative_backend": "fd", "fd_order": 4},
           "fd2": {"derivative_backend": "fd", "fd_order": 2},
           "fft_diagonal": {"laplacian_method": "diagonal"}}
DOMAINS = ["bonding", "structural"]


def one(mid):
    try:
        vd = load(mid)
        lat = vd.structure.lattice.matrix
        cosines = [abs(np.dot(lat[i], lat[j])) / np.linalg.norm(lat[i]) / np.linalg.norm(lat[j])
                   for i, j in ((0, 1), (0, 2), (1, 2))]
        row = {"id": mid, "n_atoms": vd.structure.n_atoms, "grid": "x".join(map(str, vd.shape)),
               "max_cos_between_axes": float(max(cosines))}
        for tag, kw in SCHEMES.items():
            f = pydemi.featurize(vd, domains=DOMAINS, **kw)
            row.update({f"{k}@{tag}": v for k, v in f.items()})
        for backend, order in (("fft", 4), ("fd", 4)):
            lap = derivatives(vd, "rho", lambda v: v.rho.data, backend, order).laplacian("metric")
            row[f"lap_integral_rel@{backend}"] = float(abs(lap.sum()) / np.abs(lap).sum())
        return row
    except Exception as exc:                                    # noqa: BLE001
        return {"id": mid, "error": repr(exc)}


def summary(df):
    names = [c.split("@")[0] for c in df.columns
             if c.endswith("@fft") and not c.startswith("lap_integral_rel")]
    rows = []
    for n in names:
        ref = df[f"{n}@fft"].astype(float)
        for tag in ("fd4", "fd2", "fft_diagonal"):
            v = df[f"{n}@{tag}"].astype(float)
            ok = np.isfinite(ref) & np.isfinite(v)
            rel = ((v - ref).abs() / np.maximum(np.maximum(v.abs(), ref.abs()), 1e-12))[ok]
            if rel.max() < 1e-9:
                continue
            sub = ok & (df["max_cos_between_axes"] > 1e-6) if tag == "fft_diagonal" else ok
            rows.append({"descriptor": n, "scheme": tag, "n": int(ok.sum()),
                         "median": rel.median(), "p95": rel.quantile(0.95), "max": rel.max(),
                         "spearman": ref[ok].rank().corr(v[ok].rank()),
                         "n_nonorthogonal": int(sub.sum())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 200
    if "--summary-only" in sys.argv:
        df = pd.read_csv(OUT / "derivatives.csv", low_memory=False)
    else:
        with ProcessPoolExecutor(WORKERS) as pool:
            df = pd.DataFrame(list(pool.map(one, sample_ids(n), chunksize=1)))
        df.to_csv(OUT / "derivatives.csv", index=False)
    ok = df[df.get("error", pd.Series("", index=df.index)).fillna("") == ""] if "error" in df else df
    print(f"{len(ok)} of {len(df)} structures")
    s = summary(ok)
    s.to_csv(OUT / "derivatives_summary.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_rows", 300):
        print(s.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    for b in ("fft", "fd"):
        print(f"max |int lap| / int |lap| ({b}): {ok[f'lap_integral_rel@{b}'].max():.2e}")
