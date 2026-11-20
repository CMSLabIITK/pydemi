"""Descriptor models restricted to the 10 most important descriptors of each target.

The ranking is the mean |SHAP| of the five descriptor-only LightGBM models of study_v2
(models/<target>__desc__s<seed>.txt.gz) on the TRAINING split, averaged over the seeds, so
the test split plays no part in choosing the features. Each target is then retrained with
its top 10 exactly as in train.py (ridge, random forest, 50-trial Optuna LightGBM, 5 seeds).

    OMP_NUM_THREADS=1 WORKERS=8 THREADS=2 nice -n 10 python scripts/train_top10.py
    -> results/study_v2_top10/top10.json, jobs/<target>__top10.json, models/, train.log
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lightgbm as lgb
import numpy as np

os.environ.setdefault("OUT", "results/study_v2_top10")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402

STUDY = train.ROOT / os.environ.get("STUDY", "results/study_v2")
K = int(os.environ.get("TOP", "10"))


def ranking(target: str, d, desc: list[str]) -> list[tuple[str, float]]:
    y = d[target].to_numpy(float)
    X = d.loc[(d.split == "train").to_numpy() & ~np.isnan(y), desc].to_numpy(float)
    imp = np.zeros(len(desc))
    seeds = sorted(STUDY.glob(f"models/{target}__desc__s*.txt.gz"))
    for f in seeds:
        with gzip.open(f, "rt") as fh:
            m = lgb.Booster(model_str=fh.read())
        assert m.feature_name() == desc, f
        imp += np.abs(m.predict(X, pred_contrib=True, num_threads=train.THREADS)[:, :-1]).mean(axis=0)
    imp /= len(seeds)
    order = np.argsort(-imp)
    return [(desc[i], float(imp[i] / imp.sum())) for i in order]


def run(target: str, feats: list[str]) -> str:
    load = train.load

    def patched():
        d, sets, meta = load()
        return d, {**sets, "top10": feats}, meta

    train.load = patched
    return train.run(target, "top10")


def main() -> None:
    out = train.OUT
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "train.log"), logging.StreamHandler()])
    d, sets, _ = train.load()
    path = out / "top10.json"
    if path.exists():
        top = json.loads(path.read_text())
    else:
        top = {}
        with ProcessPoolExecutor(train.WORKERS) as pool:
            jobs = {t: pool.submit(ranking, t, d, sets["desc"]) for t in train.TARGETS}
            for t, f in jobs.items():
                r = f.result()
                top[t] = {"features": [n for n, _ in r[:K]], "share": [s for _, s in r[:K]],
                          "share_top": sum(s for _, s in r[:K]), "ranking": r}
                logging.info(f"ranked {t}: top {K} carry {100 * top[t]['share_top']:.0f} % of mean |SHAP|")
        path.write_text(json.dumps(top, indent=1))
    todo = [t for t in train.TARGETS if (not train.ONLY or t in train.ONLY)
            and not (out / "jobs" / f"{t}__top10.json").exists()]
    todo.sort(key=lambda t: -d[t].notna().sum())
    logging.info(f"{len(todo)} jobs, top {K} by train-set SHAP, {train.WORKERS} workers x "
                 f"{train.THREADS} threads, {train.N_TRIALS} trials, seeds {train.SEEDS}")
    with ProcessPoolExecutor(train.WORKERS) as pool:
        for f in as_completed([pool.submit(run, t, top[t]["features"]) for t in todo]):
            logging.info(f.result())


if __name__ == "__main__":
    main()
