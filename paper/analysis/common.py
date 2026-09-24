"""Shared helpers for the paper's numerical analyses (run from this directory)."""
import os
import random
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

DATASET = Path("/data/sai/new_charge/6000_data_aug13")
RESULTS = Path(__file__).resolve().parents[2] / "results" / "descriptors_6000_data_aug13.csv"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def all_ids():
    return sorted(p.parent.name for p in DATASET.glob("*/CHGCAR"))


def sample_ids(n, seed=2026):
    ids = all_ids()
    return random.Random(seed).sample(ids, n)
