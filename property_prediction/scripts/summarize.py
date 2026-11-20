"""Tables from the finished jobs of scripts/train.py.

    results/study_v2/summary.csv    every target x model: test metrics, mean and std over the
                                    seeds (ridge: one fit), LightGBM 5-seed ensemble
    results/study_v2/best.csv       per target: the best model by mean test R² / ROC-AUC
    results/study_v2/jcim.csv       the five targets of the comparison of Table 4
    results/study_v2/shap.csv       per target, LightGBM: share of the mean |SHAP| of each
                                    descriptor on the test set, seed mean

    python scripts/summarize.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / os.environ.get("OUT", "results/study_v2")
METRICS = {"regression": ("r2", "mae", "rmse", "median_ae"),
           "classification": ("roc_auc", "accuracy", "f1", "logloss")}
JCIM = {"bulk_modulus": "K", "shear_modulus": "G", "youngs_modulus": "E", "energy_jcim": "E_atoms",
        "debye_jcim": "θ_D"}


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
        s = pd.DataFrame([r["shap"] for r in j["models"]["lgbm"]["seeds"]]).mean()
        s = s / s.sum()
        shap += [{"target": j["target"], "feature": f, "share": float(v)}
                 for f, v in s.sort_values(ascending=False).items()]
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.csv", index=False)
    s["score"] = np.where(s.task == "classification", s.roc_auc, s.r2)
    best = s.loc[s.groupby("target").score.idxmax(), ["target", "model", "score"]]
    best.to_csv(OUT / "best.csv", index=False)
    pd.DataFrame(shap).to_csv(OUT / "shap.csv", index=False)
    jcim = s[s.target.isin(JCIM)].assign(property=lambda x: x.target.map(JCIM))
    jcim[["property", "model", "r2", "r2_std", "mae", "mae_std", "n_test"]].to_csv(OUT / "jcim.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(f"{len(jobs)} jobs\n")
        print(best.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
