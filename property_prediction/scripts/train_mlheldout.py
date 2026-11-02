"""Property models evaluated on structures that neither the density model nor the property model saw.

The 605 test structures of the density model (paper/analysis/out/descriptors_ml.csv) become the
test set. Training and validation are the study split without them; the structures of the study's
test set that are not among the 605 are set aside. Same protocol as train.py (descriptors only,
50-trial Optuna LightGBM, 5 seeds). The LightGBM boosters of every seed are then applied to the
descriptors of the DFT and of the ChargE3Net-predicted densities of the 605 structures.

    WORKERS=4 THREADS=4 OUT=results/study_v1_mlheldout FEATURE_SETS=desc python scripts/train_mlheldout.py
-> results/study_v1_mlheldout/jobs/<target>__desc.json, predictions.csv, summary.csv
"""

from __future__ import annotations

import gzip
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402
from train_preview import back  # noqa: E402

ML = train.ROOT.parent / "paper/analysis/out/descriptors_ml.csv"
TARGETS = ["bulk_modulus", "shear_modulus", "youngs_modulus", "C11", "C12", "C44", "debye_temperature",
           "sound_velocity_transverse", "sound_velocity_longitudinal", "thermal_conductivity_clarke",
           "thermal_conductivity_cahill", "formation_energy", "cohesive_energy", "band_gap", "is_metal",
           "is_gap_direct", "refractive_index"]
TARGETS = [t for t in TARGETS if not train.ONLY or t in train.ONLY]
HELD = set(pd.read_csv(ML, usecols=["structure"]).structure)


def patched_load():
    d, sets, meta = ORIG_LOAD()
    held = d.structure.isin(HELD)
    d["split"] = np.where(held, "test", np.where(d.split == "test", "aside", d.split))
    return d, sets, meta


ORIG_LOAD = train.load
train.load = patched_load


def run(target: str) -> str:
    return train.run(target, "desc")


def evaluate() -> None:
    d, sets, meta = train.load()
    ml = pd.read_csv(ML).set_index("structure")
    feats = sets["desc"]
    rows, summ = [], []
    for t in TARGETS:
        label, symbol, group, how, floor = train.TARGETS[t]
        cls = meta.at[t, "task"] == "classification"
        test = d[(d.split == "test") & d[t].notna()]
        X = {"dft": test[feats].to_numpy(float), "ml": ml.loc[test.structure, feats].to_numpy(float)}
        pred = {}
        for k, Xk in X.items():
            ens = []
            for s in train.SEEDS:
                with gzip.open(train.OUT / "models" / f"{t}__desc__s{s}.txt.gz", "rt") as fh:
                    m = lgb.Booster(model_str=fh.read())
                ens.append(m.predict(Xk))
            pz = np.mean(ens, axis=0)
            p = pz if cls else back(pz, how)
            pred[k] = np.maximum(p, floor) if (floor is not None and not cls) else p
        y = test[t].to_numpy(float)
        for i, st in enumerate(test.structure):
            rows.append({"structure": st, "target": t, "true": y[i], "dft_desc": pred["dft"][i],
                         "ml_desc": pred["ml"][i]})
        r = {"target": t, "n_test": len(y), "n_train": int(((d.split == "train") & d[t].notna()).sum())}
        if cls:
            r.update(dft_auc=roc_auc_score(y, pred["dft"]), ml_auc=roc_auc_score(y, pred["ml"]),
                     same_class=float(np.mean((pred["dft"] >= 0.5) == (pred["ml"] >= 0.5))))
        else:
            r.update(dft_r2=r2_score(y, pred["dft"]), ml_r2=r2_score(y, pred["ml"]),
                     dft_mae=mean_absolute_error(y, pred["dft"]), ml_mae=mean_absolute_error(y, pred["ml"]),
                     agree_r2=r2_score(pred["dft"], pred["ml"]))
        summ.append(r)
    pd.DataFrame(rows).to_csv(train.OUT / "predictions.csv", index=False)
    s = pd.DataFrame(summ)
    s.to_csv(train.OUT / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))


def main() -> None:
    out = train.OUT
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "train.log"), logging.StreamHandler()])
    d, _, _ = train.load()
    logging.info(f"split: {d.split.value_counts().to_dict()}; held-out structures {len(HELD)}")
    todo = [t for t in TARGETS if not (out / "jobs" / f"{t}__desc.json").exists()]
    todo.sort(key=lambda t: -d[t].notna().sum())
    with ProcessPoolExecutor(train.WORKERS) as pool:
        for f in as_completed([pool.submit(run, t) for t in todo]):
            logging.info(f.result())
    evaluate()


if __name__ == "__main__":
    main()
