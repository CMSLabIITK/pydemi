"""Tables from the finished jobs of scripts/train.py.

    results/study_v1/summary.csv    every target x feature set x model: test metrics, mean and
                                    std over the seeds (ridge: one fit), LightGBM 5-seed ensemble
    results/study_v1/best.csv       per target: the best (set, model) by mean test R² / ROC-AUC
    results/study_v1/jcim.csv       the five JCIM targets next to the published table
    results/study_v1/shap.csv       per target, LightGBM on 'both': mean |SHAP| per feature, seed mean

    python scripts/summarize.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / os.environ.get("OUT", "results/study_v1")
METRICS = {"regression": ("r2", "mae", "rmse", "median_ae"),
           "classification": ("roc_auc", "accuracy", "f1", "logloss")}

# cross-model benchmark (R², MAE) on the same split, from a manuscript in preparation: read from
# data/benchmark.json ({model: {property: [R², MAE]}}) when present; not distributed
JCIM = {"bulk_modulus": "K", "shear_modulus": "G", "youngs_modulus": "E", "energy_jcim": "E_form",
        "debye_jcim": "θ_D"}
BENCHMARK = ROOT / "data" / "benchmark.json"


def main() -> None:
    jobs = [json.loads(p.read_text()) for p in sorted((OUT / "jobs").glob("*.json"))]
    rows, shap = [], []
    for j in jobs:
        for model, m in j["models"].items():
            row = {"target": j["target"], "label": j["label"], "group": j["group"], "task": j["task"],
                   "units": j["units"], "set": j["set"], "model": model, "n_train": j["n_train"],
                   "n_test": j["n_test"], "n_seeds": len(m["seeds"])}
            for k in METRICS[j["task"]]:
                v = [r[k] for r in m["seeds"]]
                row[k], row[k + "_std"] = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            if "ensemble" in m:
                row.update({f"ens_{k}": m["ensemble"][k] for k in METRICS[j["task"]]})
                row["trees"] = float(np.mean([r["best_iteration"] for r in m["seeds"]]))
            rows.append(row)
        if j["set"] == "both":
            s = pd.DataFrame([r["shap"] for r in j["models"]["lgbm"]["seeds"]]).mean()
            s = s / s.sum()
            shap += [{"target": j["target"], "feature": f, "share": float(v),
                      "kind": "composition" if f.startswith("MagpieData") or f.endswith("-norm") else "descriptor"}
                     for f, v in s.sort_values(ascending=False).items()]
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.csv", index=False)
    s["score"] = np.where(s.task == "classification", s.roc_auc, s.r2)
    best = s.loc[s.groupby("target").score.idxmax(), ["target", "set", "model", "score"]]
    piv = s[s.model == "lgbm"].pivot(index="target", columns="set", values="score")
    best = best.merge(piv.add_prefix("lgbm_"), left_on="target", right_index=True)
    best.to_csv(OUT / "best.csv", index=False)
    pd.DataFrame(shap).to_csv(OUT / "shap.csv", index=False)

    ours = []
    for t, sym in JCIM.items():
        for _, r in s[(s.target == t)].iterrows():
            ours.append({"property": sym, "model": f"{r.model} ({r.set})", "r2": r.r2, "r2_std": r.r2_std,
                         "mae": r.mae, "mae_std": r.mae_std, "n_test": r.n_test, "source": "this study"})
    published = json.loads(BENCHMARK.read_text()) if BENCHMARK.exists() else {}
    pub = [{"property": sym, "model": m, "r2": v[0], "mae": v[1], "source": "benchmark"}
           for m, d in published.items() for sym, v in d.items()]
    pd.DataFrame(ours + pub).to_csv(OUT / "jcim.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(f"{len(jobs)} jobs\n")
        print(best.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
