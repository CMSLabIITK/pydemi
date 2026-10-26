"""Penn-model features for the dielectric targets (data/penn_features.csv).

The Penn model ties the electronic dielectric constant to the valence plasma energy and a
gap: eps_inf ~ 1 + (hbar omega_p / E_g)^2, with hbar omega_p = hbar sqrt(n e^2 / (eps0 m_e))
for the valence electron density n. pydemi has n from the CHGCAR (electrons / volume); the
gap is not known at prediction time, so it is the prediction of the v1 band-gap model
(LightGBM on the composition, seed 0, as used by pydemi.props):

    n_valence   valence electrons per cell (sum of ZVAL of the training POTCARs = the
                integral of the CHGCAR; checked on CHECK random structures)
    hw_p        valence plasma energy (eV)
    Eg_pred     predicted PBE band gap (eV); on the train split out-of-fold (5 folds, same
                parameters and rounds), so the dielectric models never train on a gap
                predicted for their own training material
    penn_eps    1 + (hw_p / max(Eg_pred, 0.2 eV))^2

    python scripts/penn_features.py
"""

from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import constants as C
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STUDY = ROOT / "results" / "study_v1"
POTCARS = ROOT.parent / "src" / "pydemi" / "ml" / "potcars.csv"
DATASET = Path("/data/sai/new_charge/6000_data_aug13")
CHECK = int(os.environ.get("CHECK", "20"))
GAP_FLOOR = 0.2
# hbar * sqrt(e^2 / (eps0 m_e)) for n in electrons / Angstrom^3, in eV
HW_P = C.hbar * np.sqrt(C.e ** 2 / (C.epsilon_0 * C.m_e) * 1e30) / C.e


def counts(s: str) -> dict[str, int]:
    return {e: int(n) for e, n in (kv.split(":") for kv in s.split(","))}


def main() -> None:
    with open(POTCARS, newline="") as fh:
        zval = {r["element"]: float(r["zval"]) for r in csv.DictReader(l for l in fh if not l.startswith("#"))}
    desc = pd.read_csv(DATA / "descriptors_56.csv", usecols=["path", "volume", "site_counts"])
    desc["structure"] = desc.path.str.split("/").str[-2]
    d = pd.read_csv(DATA / "dataset.csv", usecols=["structure", "material_id", "split"]).merge(
        desc[["structure", "volume", "site_counts"]], on="structure")
    d["n_valence"] = [sum(n * zval[e] for e, n in counts(s).items()) for s in d.site_counts]
    d["hw_p"] = HW_P * np.sqrt(d.n_valence / d.volume)

    # the band-gap model of the GUI bundle (composition, seed 0) and its out-of-fold twin on train
    job = json.loads((STUDY / "jobs" / "band_gap__comp.json").read_text())
    comp = pd.read_csv(DATA / "composition.csv").drop(columns=["material_id", "split"])
    X = d[["structure"]].merge(comp, on="structure", how="left")[job["features"]].to_numpy(float)
    with gzip.open(STUDY / "models" / "band_gap__comp__s0.txt.gz", "rt") as fh:
        model = lgb.Booster(model_str=fh.read())
    eg = model.predict(X)
    y = pd.read_csv(DATA / "dataset.csv", usecols=["structure", "band_gap"]).set_index("structure") \
        .band_gap.reindex(d.structure).to_numpy()
    tr = ((d.split == "train") & ~np.isnan(y)).to_numpy()
    params = {"objective": "regression", "verbose": -1, "num_threads": 8, "seed": 0, "bagging_freq": 1,
              "feature_pre_filter": False, **job["models"]["lgbm"]["params"]}
    rounds = job["models"]["lgbm"]["seeds"][0]["best_iteration"]
    oof = np.full(len(d), np.nan)
    idx = np.flatnonzero(tr)
    for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(idx)):
        m = lgb.train(params, lgb.Dataset(X[idx[a]], y[idx[a]]), num_boost_round=rounds)
        oof[idx[b]] = m.predict(X[idx[b]])
        print(f"fold {k}: out-of-fold gap MAE {np.mean(np.abs(oof[idx[b]] - y[idx[b]])):.3f} eV")
    train_rows = (d.split == "train").to_numpy()
    eg = np.where(train_rows & ~np.isnan(oof), oof, eg)
    d["Eg_pred"] = np.maximum(eg, 0.0)
    d["penn_eps"] = 1 + (d.hw_p / np.maximum(d.Eg_pred, GAP_FLOOR)) ** 2

    # n_valence against the integral of the CHGCAR
    if CHECK:
        import pydemi
        rng = np.random.default_rng(0)
        worst = 0.0
        for i in rng.choice(len(d), CHECK, replace=False):
            vd = pydemi.read_chgcar(DATASET / d.structure[i] / "CHGCAR")
            n = float(vd.rho.data.sum() * vd.rho.dV)
            worst = max(worst, abs(n - d.n_valence[i]) / d.n_valence[i])
        print(f"n_valence vs CHGCAR integral on {CHECK} structures: largest relative difference {worst:.2e}")
    out = d[["structure", "material_id", "split", "n_valence", "hw_p", "Eg_pred", "penn_eps"]]
    out.to_csv(DATA / "penn_features.csv", index=False)
    print("wrote", DATA / "penn_features.csv", len(out))


if __name__ == "__main__":
    main()
