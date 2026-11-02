"""Descriptor models restricted to K descriptors, from the training-set SHAP ranking of top10.json.

MODE=own:    each property's own K most important descriptors (as train_top10.py, other K).
MODE=common: one set for every property, the K descriptors with the largest training-set SHAP
             share averaged over the properties in ONLY.
Same protocol as train.py (ridge, random forest, 50-trial Optuna LightGBM, 5 seeds).

    TOP=8 MODE=own    OUT=results/study_v1_top8own    ONLY=... python scripts/train_topk.py
    TOP=8 MODE=common OUT=results/study_v1_top8common ONLY=... python scripts/train_topk.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402

K = int(os.environ.get("TOP", "8"))
MODE = os.environ.get("MODE", "own")
TOP10 = train.ROOT / "results/study_v1_top10/top10.json"
SET = f"top{K}{MODE}"


def run(target: str, feats: list[str]) -> str:
    load = train.load

    def patched():
        d, sets, meta = load()
        return d, {**sets, SET: feats}, meta

    train.load = patched
    return train.run(target, SET)


def main() -> None:
    out = train.OUT
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "train.log"), logging.StreamHandler()])
    rank = json.loads(TOP10.read_text())
    targets = train.ONLY or list(rank)
    if MODE == "own":
        feats = {t: [n for n, _ in rank[t]["ranking"][:K]] for t in targets}
    else:
        share: dict[str, float] = {}
        for t in targets:
            for n, s in rank[t]["ranking"]:
                share[n] = share.get(n, 0.0) + s / len(targets)
        common = sorted(share, key=lambda n: -share[n])[:K]
        feats = {t: common for t in targets}
        logging.info("common set: " + ", ".join(f"{n} {100 * share[n]:.1f} %" for n in common))
    (out / "features.json").write_text(json.dumps(feats, indent=1))
    d, _, _ = train.load()
    todo = [t for t in targets if not (out / "jobs" / f"{t}__{SET}.json").exists()]
    todo.sort(key=lambda t: -d[t].notna().sum())
    logging.info(f"{len(todo)} jobs, {SET}, {train.WORKERS} workers x {train.THREADS} threads")
    with ProcessPoolExecutor(train.WORKERS) as pool:
        for f in as_completed([pool.submit(run, t, feats[t]) for t in todo]):
            logging.info(f.result())


if __name__ == "__main__":
    main()
