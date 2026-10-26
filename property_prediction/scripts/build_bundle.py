"""The v1 property-model bundle for pydemi.props / the GUI, from the study (scripts/train.py).

Per target, the LightGBM model of the feature set (comp / desc / both) with the lowest
validation loss (mean over the 5 seeds; the test split plays no part in the choice), refit
with seed 0. Its test metrics are those of that one model; the 5-seed mean and spread are
stored next to them. Left out: energy_above_hull (test R² < 0.1 for every set) and the two
benchmark-table columns (duplicates of cohesive-like energy and Debye temperature).

The model text keeps only what prediction reads: the per-node training statistics
(split_gain, internal_value/weight/count, leaf_weight/count) and the tree_sizes header are
dropped, which roughly halves the size; the script checks that the trimmed models give
identical predictions. The bundle is xz-compressed and copied into the package, where
pydemi.props and the GUI read it.

    python scripts/build_bundle.py    # writes results/study_v1/properties_v1.json.xz
                                      # and src/pydemi/properties_v1.json.xz
"""

from __future__ import annotations

import gzip
import json
import lzma
import shutil
import sys
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import pydemi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import SETS, load  # noqa: E402
from train_preview import SHOW  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STUDY = ROOT / "results" / "study_v1"
SKIP = ("energy_above_hull",)
INPUTS = {"comp": "composition", "desc": "descriptors", "both": "descriptors + composition"}
PACKAGE = ROOT.parent / "src" / "pydemi"
# per-node training statistics and the tree-size index: not read by predict
UNUSED = ("split_gain=", "internal_value=", "internal_weight=", "internal_count=",
          "leaf_weight=", "leaf_count=", "tree_sizes=")
METRICS = {"regression": ("mae", "rmse", "r2", "median_ae"),
           "classification": ("roc_auc", "accuracy", "f1", "logloss")}


def trim(model: str, n_features: int) -> str:
    """The model text without what prediction does not read, checked to predict identically."""
    out = "\n".join(line for line in model.split("\n") if not line.startswith(UNUSED))
    x = np.random.default_rng(0).normal(scale=3.0, size=(2000, n_features))
    x[::7, ::5] = np.nan
    if not np.array_equal(lgb.Booster(model_str=model).predict(x), lgb.Booster(model_str=out).predict(x)):
        raise RuntimeError("the trimmed model predicts differently")
    return out


def main() -> None:
    d, sets, meta = load()
    desc = sets["desc"]
    train = (d.split == "train").to_numpy()
    X = d[desc].astype(float)
    lo, hi = X[train].quantile(0.01), X[train].quantile(0.99)
    bundle: dict = {
        "name": "v1 (tuned LightGBM, descriptors and composition)",
        "created": date.today().isoformat(),
        "pydemi": pydemi.__version__, "lightgbm": lgb.__version__,
        "data": "CMS Lab 6,059-structure dataset; targets from properties.csv and Materials Project "
                "(database 2026.04.13)",
        "split": f"train {int(train.sum())} / val {int((d.split == 'val').sum())} / "
                 f"test {int((d.split == 'test').sum())}",
        "features": desc,
        "feature_range": {f: [float(lo[f]), float(hi[f])] for f in desc},
        "composition": json.loads((DATA / "magpie_table.json").read_text()),
        "targets": {},
        "dataset": {"material_id": d.material_id.tolist(), "formula": d.formula.tolist(),
                    "split": d.split.tolist(),
                    "X": [[None if np.isnan(v) else float(v) for v in row] for row in X.to_numpy()],
                    "y": {name: [None if np.isnan(v) else float(v) for v in d[name].to_numpy(dtype=float)]
                          for name in SHOW if name not in SKIP}},
    }
    rows = []
    for name, (label, symbol, group, how, floor) in SHOW.items():
        if name in SKIP:
            continue
        jobs = {fs: json.loads((STUDY / "jobs" / f"{name}__{fs}.json").read_text()) for fs in SETS}
        val = {fs: float(np.mean([r["val_loss"] for r in j["models"]["lgbm"]["seeds"]])) for fs, j in jobs.items()}
        fs = min(val, key=val.get)
        j = jobs[fs]
        task = j["task"]
        seeds = j["models"]["lgbm"]["seeds"]
        s0 = seeds[0]
        with gzip.open(STUDY / "models" / f"{name}__{fs}__s0.txt.gz", "rt") as fh:
            booster = trim(fh.read(), len(j["features"]))
        shap = s0["shap"]
        total = sum(shap.values()) or 1.0
        bundle["targets"][name] = {
            "label": label, "symbol": symbol, "group": group, "task": task,
            "units": meta.at[name, "units"], "description": meta.at[name, "description"],
            "transform": how, "floor": floor, "features": j["features"], "inputs": INPUTS[fs],
            "set": fs, "val_loss": val, "params": j["models"]["lgbm"]["params"],
            "metrics": {k: s0[k] for k in METRICS[task]},
            "metrics_seeds": {k: [float(np.mean([r[k] for r in seeds])), float(np.std([r[k] for r in seeds], ddof=1))]
                              for k in METRICS[task]},
            "n_train": j["n_train"], "n_val": j["n_val"], "n_test": j["n_test"],
            "best_iteration": s0["best_iteration"], "importance_kind": "mean |SHAP| on the test set",
            "importance": {f: v / total for f, v in sorted(shap.items(), key=lambda kv: -kv[1])},
            "test": {"y": j["test_y"], "p": s0["test_p"], "formula": j["test_formula"]},
            "booster": booster,
        }
        key = "roc_auc" if task == "classification" else "r2"
        rows.append({"target": name, "set": fs, key: s0[key], "trees": s0["best_iteration"],
                     **{f"val_{k}": v for k, v in val.items()}})
    out = STUDY / "properties_v1.json.xz"
    with lzma.open(out, "wt", preset=6) as fh:
        json.dump(bundle, fh)
    shutil.copyfile(out, PACKAGE / out.name)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB, and copied it to", PACKAGE)


if __name__ == "__main__":
    main()
