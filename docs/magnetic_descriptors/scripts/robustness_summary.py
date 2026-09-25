"""Partition, shell and grid sensitivity summaries of the magnetic descriptors.

From region_shares.csv (150 magnetic structures: mu_site_std and spin_frustration under the
four partitions, f_bond_spin under four shell choices) and the paper's grid-convergence
table (paper/analysis/out/convergence.csv; the magnetic structures among its 30).

Usage:  python robustness_summary.py   -> ../data/robust_summary.csv, ../data/robust_grid.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
PAPER = HERE.parents[2] / "paper" / "analysis" / "out"
NAMES = ["m1_spin", "sigma_r2_spin", "f_bond_spin", "mu_site_std", "spin_frustration",
         "spin_charge_correlation"]


def rel(a, b):
    return (a - b).abs() / np.maximum(np.maximum(a.abs(), b.abs()), 1e-12)


if __name__ == "__main__":
    S = pd.read_csv(DATA / "region_shares.csv")
    rows = []
    for p in ("becke", "power", "hirshfeld"):
        for n in ("mu_site_std", "spin_frustration"):
            a, b = S[n], S[f"{n}@{p}"]
            rows.append((n, p, rel(b, a).median(), rel(b, a).quantile(0.9), a.rank().corr(b.rank())))
    for t in ("abs_0.6_1.3", "abs_1.0_1.8", "scaled_0.6_1.3"):
        a, b = S["f_bond_spin"], S[f"f_bond_spin@{t}"]
        rows.append(("f_bond_spin", t, rel(b, a).median(), rel(b, a).quantile(0.9), a.rank().corr(b.rank())))
    r = pd.DataFrame(rows, columns=["descriptor", "variant", "median_rel", "p90_rel", "spearman"])
    r.to_csv(DATA / "robust_summary.csv", index=False)
    print(r.round(3).to_string())
    c = pd.read_csv(PAPER / "convergence.csv")
    c = c[c["descriptor"].isin(NAMES) & (c["full"] != 0)]
    c.to_csv(DATA / "robust_grid.csv", index=False)
    g = c.groupby("descriptor")["rel_change_x0.8"]
    print(c["id"].nunique(), "magnetic structures in the grid test")
    print(pd.DataFrame({"median": g.apply(lambda s: s.abs().median()),
                        "p90": g.apply(lambda s: s.abs().quantile(0.9))}).round(4).to_string())
