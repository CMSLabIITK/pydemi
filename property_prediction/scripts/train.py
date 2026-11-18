"""The property models of the paper: for each of the 17 properties (and the two columns of
Kumar et al. used in the comparison of Table 4), three models on the 55 pydemi descriptors,
with the split of Kumar et al., J. Chem. Inf. Model. 66, 7113 (2026): train 4,847 / val 606 /
test 606.

Models
    ridge  standardised linear model (logistic for classes); alpha / C picked on val
    rf     random forest, 500 trees, fixed settings
    lgbm   LightGBM, Optuna-tuned on val (TPE), early stopping on val
rf and lgbm are refit with 5 seeds; the test split is scored once per fitted model, never
used for choosing anything. Seeds vary bagging and column sampling only: the split is fixed.

One job per target, resumable: results/study_v1/jobs/<target>__desc.json
holds the metrics, best parameters, test predictions and SHAP importances; the LightGBM
boosters go to results/study_v1/models/.

    OMP_NUM_THREADS=1 nice -n 10 python scripts/train.py           # 4 workers x 4 threads
    N_TRIALS=2 SEEDS=1 ONLY=bulk_modulus OUT=results/smoke python scripts/train.py
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error, r2_score,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pydemi

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / os.environ.get("OUT", "results/study_v1")
WORKERS = int(os.environ.get("WORKERS", "4"))
THREADS = int(os.environ.get("THREADS", "4"))
N_TRIALS = int(os.environ.get("N_TRIALS", "50"))
SEEDS = list(range(int(os.environ.get("SEEDS", "5"))))
ONLY = [s for s in os.environ.get("ONLY", "").split(",") if s]
MAX_ROUNDS, PATIENCE = 10000, 200

# label, symbol, group, transform for training, lower bound of the prediction
SHOW = {
    "bulk_modulus": ("Bulk modulus", "K", "Mechanical", None, 0.0),
    "shear_modulus": ("Shear modulus", "G", "Mechanical", None, 0.0),
    "youngs_modulus": ("Young's modulus", "E", "Mechanical", None, 0.0),
    "C11": ("Elastic constant", "C11", "Elastic tensor", None, None),
    "C12": ("Elastic constant", "C12", "Elastic tensor", None, None),
    "C44": ("Elastic constant", "C44", "Elastic tensor", None, None),
    "debye_temperature": ("Debye temperature", "θ_D", "Thermal", None, 0.0),
    "sound_velocity_transverse": ("Sound velocity, transverse", "v_t", "Thermal", None, 0.0),
    "sound_velocity_longitudinal": ("Sound velocity, longitudinal", "v_l", "Thermal", None, 0.0),
    "thermal_conductivity_clarke": ("Min. thermal conductivity, Clarke", "κ_Clarke", "Thermal", None, 0.0),
    "thermal_conductivity_cahill": ("Min. thermal conductivity, Cahill", "κ_Cahill", "Thermal", None, 0.0),
    "formation_energy": ("Formation energy", "E_f", "Energetics", None, None),
    "cohesive_energy": ("Cohesive energy", "E_coh", "Energetics", None, None),
    "band_gap": ("Band gap (PBE)", "E_g", "Electronic", None, 0.0),
    "is_metal": ("Metallic", "P(metal)", "Electronic", None, None),
    "is_gap_direct": ("Direct gap", "P(direct)", "Electronic", None, None),
    "refractive_index": ("Refractive index", "n", "Optical", "log10", 1.0),  # heavy tail
}
PROPS = list(SHOW)

# the energy and Debye columns of Kumar et al. (properties.csv), for the comparison of Table 4
EXTRA = {
    "energy_jcim": ("Energy vs free atoms (JCIM column)", "E_form", "Energetics", None, None,
                    "eV/atom", "formation_energy_per_atom"),
    "debye_jcim": ("Debye temperature (JCIM column)", "θ_D", "Thermal", None, 0.0,
                   "K", "debye_temperature"),
}
TARGETS = {**SHOW, **{k: v[:5] for k, v in EXTRA.items()}}
SETS = ("desc",)


def forward(y: np.ndarray, how: str | None) -> np.ndarray:
    return np.log10(np.clip(y, 1e-12, None)) if how == "log10" else y


def back(z: np.ndarray, how: str | None) -> np.ndarray:
    return 10.0 ** z if how == "log10" else z


def load() -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    d = pd.read_csv(DATA / "dataset.csv")
    p = pd.read_csv(DATA / "properties.csv")
    p = p[["formula_sp"] + [v[6] for v in EXTRA.values()]].rename(
        columns={"formula_sp": "structure", **{v[6]: k for k, v in EXTRA.items()}})
    d = d.merge(p, on="structure", how="left")
    meta = pd.read_csv(DATA / "targets_columns.csv").set_index("column")
    for k, v in EXTRA.items():
        meta.loc[k, ["units", "task", "source"]] = [v[5], "regression", f"properties.csv ({v[6]})"]
    return d, {"desc": list(pydemi.descriptor_names())}, meta


def scores(task: str, y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if task == "classification":
        return {"roc_auc": float(roc_auc_score(y, p)), "accuracy": float(accuracy_score(y, p >= 0.5)),
                "f1": float(f1_score(y, p >= 0.5)), "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)))}
    return {"mae": float(mean_absolute_error(y, p)), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            "r2": float(r2_score(y, p)), "median_ae": float(np.median(np.abs(y - p)))}


def job(target: str, fs: str) -> dict:
    t0 = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    d, sets, meta = load()
    label, symbol, group, how, floor = TARGETS[target]
    task = meta.at[target, "task"]
    cls = task == "classification"
    feats = sets[fs]
    y = d[target].to_numpy(float)
    ok = ~np.isnan(y)
    part = {s: (d.split == s).to_numpy() & ok for s in ("train", "val", "test")}
    X = d[feats].to_numpy(float)
    z = y if cls else forward(y, how)
    Xtr, Xva, Xte = (X[part[s]] for s in ("train", "val", "test"))
    ztr, zva = z[part["train"]], z[part["val"]]
    yte = y[part["test"]]

    def out(pz: np.ndarray) -> np.ndarray:
        """Model output in the target's units (classes: probability)."""
        if cls:
            return pz
        p = back(pz, how)
        return np.maximum(p, floor) if floor is not None else p

    res: dict = {"target": target, "set": fs, "task": task, "label": label, "symbol": symbol,
                 "group": group, "units": meta.at[target, "units"], "transform": how, "floor": floor,
                 "features": feats, "n_train": int(part["train"].sum()), "n_val": int(part["val"].sum()),
                 "n_test": int(part["test"].sum()), "test_y": yte.tolist(),
                 "test_material_id": d.material_id[part["test"]].tolist(),
                 "test_formula": d.formula[part["test"]].tolist(), "models": {}}

    # ridge / logistic: one hyper-parameter on val
    best = None
    for a in np.logspace(-3, 3, 13):
        lin = (LogisticRegression(C=1 / a, max_iter=5000) if cls else Ridge(alpha=a))
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), lin).fit(Xtr, ztr)
        loss = (log_loss(zva, m.predict_proba(Xva)[:, 1]) if cls
                else float(np.sqrt(np.mean((m.predict(Xva) - zva) ** 2))))
        if best is None or loss < best[0]:
            best = (loss, a, m)
    p = best[2].predict_proba(Xte)[:, 1] if cls else out(best[2].predict(Xte))
    res["models"]["ridge"] = {"params": {"alpha": float(best[1])}, "val_loss": float(best[0]),
                              "seeds": [{"seed": 0, **scores(task, yte, p), "test_p": p.tolist()}]}

    # random forest: fixed settings
    runs = []
    for s in SEEDS:
        rf = (RandomForestClassifier(500, max_features="sqrt", n_jobs=THREADS, random_state=s) if cls
              else RandomForestRegressor(500, max_features=1 / 3, n_jobs=THREADS, random_state=s))
        m = make_pipeline(SimpleImputer(strategy="median"), rf).fit(Xtr, ztr)
        p = m.predict_proba(Xte)[:, 1] if cls else out(m.predict(Xte))
        runs.append({"seed": s, **scores(task, yte, p), "test_p": p.tolist()})
    res["models"]["rf"] = {"params": {"n_estimators": 500, "max_features": "sqrt" if cls else "1/3"},
                           "seeds": runs}

    # LightGBM: Optuna on val, then refit with every seed
    base = {"objective": "binary" if cls else "regression", "metric": "binary_logloss" if cls else "l2",
            "num_threads": THREADS, "verbose": -1, "feature_pre_filter": False, "bagging_freq": 1}

    def fit(params: dict) -> lgb.Booster:
        dtr = lgb.Dataset(Xtr, ztr, feature_name=feats, params={"feature_pre_filter": False})
        dva = lgb.Dataset(Xva, zva, reference=dtr)
        return lgb.train(params, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])

    def val_loss(m: lgb.Booster) -> float:
        pv = m.predict(Xva, num_iteration=m.best_iteration)
        return float(log_loss(zva, np.clip(pv, 1e-7, 1 - 1e-7)) if cls else np.sqrt(np.mean((pv - zva) ** 2)))

    def objective(trial: optuna.Trial) -> float:
        params = dict(base, seed=0,
                      learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                      num_leaves=trial.suggest_int("num_leaves", 7, 127, log=True),
                      min_child_samples=trial.suggest_int("min_child_samples", 5, 100, log=True),
                      feature_fraction=trial.suggest_float("feature_fraction", 0.3, 1.0),
                      bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
                      lambda_l1=trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                      lambda_l2=trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True))
        m = fit(params)
        trial.set_user_attr("best_iteration", int(m.best_iteration))
        return val_loss(m)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=N_TRIALS)
    tuned = dict(base, **study.best_params)
    runs, ens = [], []
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        m = fit(dict(tuned, seed=s))
        pz = m.predict(Xte, num_iteration=m.best_iteration)
        ens.append(pz)
        p = out(pz)
        shap = m.predict(Xte, num_iteration=m.best_iteration, pred_contrib=True)[:, :-1]
        runs.append({"seed": s, **scores(task, yte, p), "test_p": p.tolist(),
                     "best_iteration": int(m.best_iteration), "val_loss": val_loss(m),
                     "shap": dict(zip(feats, np.abs(shap).mean(axis=0).tolist()))})
        with gzip.open(OUT / "models" / f"{target}__{fs}__s{s}.txt.gz", "wt") as fh:
            fh.write(m.model_to_string(num_iteration=m.best_iteration))
    p = out(np.mean(ens, axis=0))
    res["models"]["lgbm"] = {"params": study.best_params, "val_loss": float(study.best_value),
                             "trials": len(study.trials), "seeds": runs,
                             "ensemble": {**scores(task, yte, p), "test_p": p.tolist()}}
    res["seconds"] = round(time.time() - t0, 1)
    return res


def run(target: str, fs: str) -> str:
    path = OUT / "jobs" / f"{target}__{fs}.json"
    try:
        res = job(target, fs)
    except Exception:
        return f"FAILED {target} {fs}\n{traceback.format_exc()}"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.rename(path)
    lg = res["models"]["lgbm"]
    key = "roc_auc" if res["task"] == "classification" else "r2"
    summary = "  ".join(f"{m}={np.mean([r[key] for r in v['seeds']]):.3f}" for m, v in res["models"].items())
    return f"done {target:28s} {fs:4s} {key} {summary}  ({res['seconds']:.0f} s, {lg['trials']} trials)"


def main() -> None:
    (OUT / "jobs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(OUT / "train.log"), logging.StreamHandler()])
    d, _, _ = load()
    todo = [(t, fs) for t in TARGETS for fs in SETS if (not ONLY or t in ONLY)
            and not (OUT / "jobs" / f"{t}__{fs}.json").exists()]
    todo.sort(key=lambda j: -d[j[0]].notna().sum())       # big targets first
    logging.info(f"{len(todo)} jobs, {WORKERS} workers x {THREADS} threads, {N_TRIALS} trials, "
                 f"seeds {SEEDS}; lightgbm {lgb.__version__}, optuna {optuna.__version__}, "
                 f"pydemi {pydemi.__version__}")
    with ProcessPoolExecutor(WORKERS) as pool:
        for f in as_completed([pool.submit(run, t, fs) for t, fs in todo]):
            logging.info(f.result())


if __name__ == "__main__":
    main()
