"""Test accuracy of the 55-descriptor models vs the reduced sets (LightGBM, 5 seeds).

score = mean over the 5 seeds of test R2 (regression) or ROC AUC (classes).
delta = reduced - all55; its 95 % interval is a paired bootstrap over the test structures
(2,000 resamples) of the seed-averaged predictions, so it shows whether a change is larger
than the test-set noise.
-> redundancy/results/compare.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE / "scripts"))
from redundancy import PROPS  # noqa: E402

SETS = {"all55": ROOT / "results/study_v1/jobs/{t}__desc.json",
        "no_redundant": HERE / "results/train/jobs/{t}__no_redundant.json",
        "important34": HERE / "results/train/jobs/{t}__important34.json",
        "lean": HERE / "results/train/jobs/{t}__lean.json"}
N = {k: len(v) for k, v in json.load(open(HERE / "results/sets.json")).items()}


def load(path):
    j = json.load(open(path))
    cls = j["task"] == "classification"
    seeds = j["models"]["lgbm"]["seeds"]
    key = "roc_auc" if cls else "r2"
    p = np.mean([s["test_p"] for s in seeds], axis=0)
    return {"cls": cls, "y": np.array(j["test_y"], float), "p": p, "ids": j["test_material_id"],
            "score": float(np.mean([s[key] for s in seeds])),
            "mae": None if cls else float(np.mean([s["mae"] for s in seeds])), "label": j["label"],
            "symbol": j["symbol"]}


def main() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for t in PROPS:
        base = load(str(SETS["all55"]).format(t=t))
        f = roc_auc_score if base["cls"] else r2_score
        boot = [rng.integers(0, len(base["y"]), len(base["y"])) for _ in range(2000)]
        row = {"target": t, "symbol": base["symbol"], "metric": "AUC" if base["cls"] else "R2",
               "all55": base["score"], "all55_mae": base["mae"]}
        for s in ("no_redundant", "important34", "lean"):
            path = Path(str(SETS[s]).format(t=t))
            if not path.exists():
                continue
            r = load(path)
            assert r["ids"] == base["ids"]
            dd = [f(base["y"][b], r["p"][b]) - f(base["y"][b], base["p"][b]) for b in boot]
            row |= {s: r["score"], f"{s}_mae": r["mae"], f"{s}_delta": r["score"] - base["score"],
                    f"{s}_lo": float(np.percentile(dd, 2.5)), f"{s}_hi": float(np.percentile(dd, 97.5))}
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results/compare.csv", index=False)
    print(f"{'property':30s} {'all 55':>7s}" + "".join(f"{s + ' (' + str(N[s]) + ')':>30s}"
                                                     for s in ("no_redundant", "important34", "lean")))
    for _, r in out.iterrows():
        line = f"{r.target:30s} {r.all55:7.3f}"
        for s in ("no_redundant", "important34", "lean"):
            if s in r and pd.notna(r.get(s)):
                line += f"   {r[s]:.3f} ({r[f'{s}_delta']:+.3f} [{r[f'{s}_lo']:+.3f},{r[f'{s}_hi']:+.3f}])"
            else:
                line += f"{'-':>30s}"
        print(line)


if __name__ == "__main__":
    main()
