"""Preview property models for the GUI: one LightGBM model per target, descriptors only,
fixed hyper-parameters (no tuning), early stopping on the validation split, scored once on
the test split. Not the study's final models; they exist so the GUI's Properties view can
be tried out.

Writes results/preview_v0/properties_preview_v0.json.gz (the bundle pydemi.props reads)
and results/preview_v0/metrics.csv.

    OMP_NUM_THREADS=16 nice -n 10 python scripts/train_preview.py
"""

from __future__ import annotations

import gzip
import json
import os
import time
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score

import pydemi

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "preview_v0"
THREADS = int(os.environ.get("OMP_NUM_THREADS", "16"))
SEED = 0

# label, symbol, group, transform for training, lower bound of the prediction
SHOW = {
    "bulk_modulus": ("Bulk modulus", "K", "Mechanical", None, 0.0),
    "shear_modulus": ("Shear modulus", "G", "Mechanical", None, 0.0),
    "youngs_modulus": ("Young's modulus", "E", "Mechanical", None, 0.0),
    "poisson_ratio": ("Poisson ratio", "ν", "Mechanical", None, None),
    "anisotropy_index": ("Universal anisotropy", "A_U", "Elastic tensor", "log1p", 0.0),
    "C11": ("Elastic constant", "C11", "Elastic tensor", None, None),
    "C12": ("Elastic constant", "C12", "Elastic tensor", None, None),
    "C44": ("Elastic constant", "C44", "Elastic tensor", None, None),
    "debye_temperature": ("Debye temperature", "θ_D", "Thermal", None, 0.0),
    "sound_velocity_transverse": ("Sound velocity, transverse", "v_t", "Thermal", None, 0.0),
    "sound_velocity_longitudinal": ("Sound velocity, longitudinal", "v_l", "Thermal", None, 0.0),
    "thermal_conductivity_clarke": ("Min. thermal conductivity, Clarke", "κ_Clarke", "Thermal", None, 0.0),
    "thermal_conductivity_cahill": ("Min. thermal conductivity, Cahill", "κ_Cahill", "Thermal", None, 0.0),
    "formation_energy": ("Formation energy", "E_f", "Energetics", None, None),
    "energy_above_hull": ("Energy above hull", "E_hull", "Energetics", None, 0.0),
    "cohesive_energy": ("Cohesive energy", "E_coh", "Energetics", None, None),
    "band_gap": ("Band gap (PBE)", "E_g", "Electronic", None, 0.0),
    "is_metal": ("Metallic", "P(metal)", "Electronic", None, None),
    "is_gap_direct": ("Direct gap", "P(direct)", "Electronic", None, None),
    "dielectric_total": ("Dielectric constant, total", "ε", "Dielectric", "log10", 1.0),
    "dielectric_electronic": ("Dielectric constant, electronic", "ε_∞", "Dielectric", "log10", 1.0),
    "dielectric_ionic": ("Dielectric constant, ionic", "ε_ion", "Dielectric", "log1p", 0.0),
    "refractive_index": ("Refractive index", "n", "Dielectric", "log10", 1.0),  # = sqrt(ε_∞): heavy tail
}
PARAMS = {"learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 20,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
          "lambda_l2": 1.0, "num_threads": THREADS, "seed": SEED, "verbose": -1}


def forward(y: np.ndarray, how: str | None) -> np.ndarray:
    if how == "log10":
        return np.log10(np.clip(y, 1e-12, None))
    if how == "log1p":
        return np.log1p(np.clip(y, 0, None))
    return y


def back(z: np.ndarray, how: str | None) -> np.ndarray:
    if how == "log10":
        return 10.0 ** z
    if how == "log1p":
        return np.expm1(z)
    return z


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feats = list(pydemi.descriptor_names())
    d = pd.read_csv(DATA / "dataset.csv")
    meta = pd.read_csv(DATA / "targets_columns.csv").set_index("column")
    X = d[feats].astype(float)
    train, val, test = (d.split == s for s in ("train", "val", "test"))
    lo, hi = X[train].quantile(0.01), X[train].quantile(0.99)

    bundle: dict = {
        "name": "Preview v0 (descriptors only, untuned LightGBM)",
        "created": date.today().isoformat(),
        "pydemi": pydemi.__version__, "lightgbm": lgb.__version__,
        "data": "CMS Lab 6,059-structure dataset; targets from properties.csv and Materials Project "
                "(database 2026.04.13)",
        "split": f"train {int(train.sum())} / val {int(val.sum())} / test {int(test.sum())} (JCIM 2026 split)",
        "features": feats,
        "feature_range": {f: [float(lo[f]), float(hi[f])] for f in feats},
        "targets": {},
        # every material of the dataset, so the GUI can tell when an input is one of them
        "dataset": {"material_id": d.material_id.tolist(), "formula": d.formula.tolist(),
                    "split": d.split.tolist(),
                    "X": [[None if np.isnan(v) else float(v) for v in row] for row in X.to_numpy()],
                    "y": {name: [None if np.isnan(v) else float(v) for v in d[name].to_numpy(dtype=float)]
                          for name in SHOW}},
    }
    rows = []
    for name, (label, symbol, group, how, floor) in SHOW.items():
        t0 = time.time()
        y = d[name]
        ok = y.notna()
        task = meta.at[name, "task"]
        params = dict(PARAMS, objective="binary" if task == "classification" else "regression",
                      metric="binary_logloss" if task == "classification" else "l2")
        z = y if task == "classification" else forward(y.to_numpy(), how)
        z = pd.Series(z, index=d.index)
        dtr = lgb.Dataset(X[train & ok], z[train & ok], feature_name=feats, free_raw_data=False)
        dva = lgb.Dataset(X[val & ok], z[val & ok], reference=dtr)
        model = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
        p = model.predict(X[test & ok], num_iteration=model.best_iteration)
        yt = y[test & ok].to_numpy()
        if task == "classification":
            metrics = {"roc_auc": float(roc_auc_score(yt, p)),
                       "accuracy": float(accuracy_score(yt, p >= 0.5)),
                       "f1": float(f1_score(yt, p >= 0.5)), "positive_rate": float(yt.mean())}
        else:
            p = back(p, how)
            if floor is not None:
                p = np.maximum(p, floor)
            metrics = {"mae": float(mean_absolute_error(yt, p)),
                       "rmse": float(np.sqrt(np.mean((yt - p) ** 2))),
                       "r2": float(r2_score(yt, p)),
                       "median_ae": float(np.median(np.abs(yt - p)))}
        gain = model.feature_importance("gain", iteration=model.best_iteration)
        imp = {f: float(g / gain.sum()) for f, g in zip(feats, gain)} if gain.sum() > 0 else {}
        bundle["targets"][name] = {
            "label": label, "symbol": symbol, "group": group, "task": task,
            "units": meta.at[name, "units"], "description": meta.at[name, "description"],
            "transform": how, "floor": floor, "metrics": metrics,
            "n_train": int((train & ok).sum()), "n_val": int((val & ok).sum()),
            "n_test": int((test & ok).sum()), "best_iteration": int(model.best_iteration),
            "importance": dict(sorted(imp.items(), key=lambda kv: -kv[1])),
            "test": {"y": [float(v) for v in yt], "p": [float(v) for v in p],
                     "formula": d.loc[test & ok, "formula"].tolist()},
            "booster": model.model_to_string(num_iteration=model.best_iteration),
        }
        rows.append({"target": name, "task": task, **metrics, "n_test": int((test & ok).sum()),
                     "trees": int(model.best_iteration), "seconds": round(time.time() - t0, 1)})
        print(f"{name:30s} " + "  ".join(f"{k}={v:.4g}" for k, v in metrics.items()), flush=True)

    with gzip.open(OUT / "properties_preview_v0.json.gz", "wt") as fh:
        json.dump(bundle, fh)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    print("wrote", OUT / "properties_preview_v0.json.gz",
          f"{(OUT / 'properties_preview_v0.json.gz').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
