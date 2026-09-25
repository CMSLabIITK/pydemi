"""grid_ionicity from ChargE3Net-predicted densities against DFT (605 test structures).

The descriptor tables of the ML evaluation (descriptor_eval.py: DFT densities and the
from-scratch model's predictions, same pydemi path) give f_int, lnf and V_spread; the
calibration of calibrate_ionicity.py (verified targets) maps them to grid_ionicity.

Usage:  python ml_ionicity.py DESC_DFT.csv DESC_MODEL.csv TAG   -> ../data/ml_ionicity_TAG.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def main(dft, model, tag):
    cal = json.loads((DATA / "calibration.json").read_text())["verified"]
    out = []
    for src, path in (("dft", dft), ("ml", model)):
        d = pd.read_csv(path)[["name", "f_int", "lnf", "V_spread"]]
        d["fint_over_lnf"] = d["f_int"] / d["lnf"]
        Z = (d[cal["features"]].to_numpy(float) - np.array(cal["mean"])) / np.array(cal["std"])
        b = np.array(cal["coef"])
        d["grid_ionicity"] = 1 / (1 + np.exp(-(b[0] + Z @ b[1:])))
        out.append(d.add_suffix(f"_{src}").rename(columns={f"name_{src}": "name"}))
    m = out[0].merge(out[1], on="name")
    m.to_csv(DATA / f"ml_ionicity_{tag}.csv", index=False)
    rel = lambda a, b: (a - b).abs() / np.maximum(np.maximum(a.abs(), b.abs()), 1e-12)
    for c in ("fint_over_lnf", "V_spread", "grid_ionicity"):
        a, b = m[f"{c}_dft"], m[f"{c}_ml"]
        print(c, "median rel", round(rel(b, a).median(), 4), "p90", round(rel(b, a).quantile(0.9), 4),
              "median abs", round((b - a).abs().median(), 4), "Spearman", round(a.rank().corr(b.rank()), 4))


if __name__ == "__main__":
    main(*sys.argv[1:4])
