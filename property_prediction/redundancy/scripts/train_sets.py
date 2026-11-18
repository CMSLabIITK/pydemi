"""Retrain the 17 properties without the redundant descriptors of redundancy.py (37 descriptors).

Same protocol as the paper (scripts/train.py: ridge, random forest, 50-trial Optuna LightGBM,
5 seeds, fixed split); only the feature list changes. The 55-descriptor baseline is the
paper's study_v1 'desc' set and is not refitted.
    cd property_prediction && OMP_NUM_THREADS=1 WORKERS=8 THREADS=2 nice -n 10 \
        python redundancy/scripts/train_sets.py
    -> redundancy/results/train/jobs/<target>__<set>.json, models/, train.log
"""

from __future__ import annotations

import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
os.environ.setdefault("OUT", "redundancy/results/train")          # relative to property_prediction/
sys.path.insert(0, str(HERE.parent / "scripts"))
import train  # noqa: E402

sys.path.insert(0, str(HERE / "scripts"))
from redundancy import PROPS  # noqa: E402

SETS = json.load(open(HERE / "results/sets.json"))
RUN = ["no_redundant"]


def run(target: str, fs: str) -> str:
    load = train.load

    def patched():
        d, sets, meta = load()
        return d, {**sets, fs: SETS[fs]}, meta
    train.load = patched
    return train.run(target, fs)


def main() -> None:
    out = train.OUT
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "train.log"), logging.StreamHandler()])
    d, _, _ = train.load()
    todo = [(t, s) for s in RUN for t in PROPS if (not train.ONLY or t in train.ONLY)
            and not (out / "jobs" / f"{t}__{s}.json").exists()]
    todo.sort(key=lambda ts: -d[ts[0]].notna().sum())
    logging.info(f"{len(todo)} jobs, sets { {s: len(SETS[s]) for s in RUN} }, {train.WORKERS} workers x "
                 f"{train.THREADS} threads, {train.N_TRIALS} trials, seeds {train.SEEDS}")
    with ProcessPoolExecutor(train.WORKERS) as pool:
        for f in as_completed([pool.submit(run, t, s) for t, s in todo]):
            logging.info(f.result())


if __name__ == "__main__":
    main()
