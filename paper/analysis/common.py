"""Shared helpers for the paper's numerical analyses (run from this directory).

Structures are read as in the dataset rerun (results/prompt_spec/run_rerun.py):
CHGCAR with ZVAL / RCORE from the run's OUTCAR or, when it has none, from the
dataset's PAW table. Random samples use fixed seeds (2026, 7, 11).
"""
import json
import os
import random
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd  # noqa: E402

import pydemi  # noqa: E402
from pydemi.constants import BOHR_ANGSTROM  # noqa: E402

DATASET = Path("/data/sai/new_charge/6000_data_aug13")
REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "prompt_spec" / "descriptors_6000_data_aug13.csv"
PAW_TABLE = REPO / "results" / "prompt_spec" / "dataset_paw_table.json"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
WORKERS = int(os.environ.get("ANALYSIS_WORKERS", "16"))

_table = json.loads(PAW_TABLE.read_text())
READ_OPTIONS = {"zval": {e: v["zval"] for e, v in _table.items()},
                "paw_radii": {e: v["rcore_bohr"] * BOHR_ANGSTROM for e, v in _table.items()}}


def all_ids():
    return sorted(p.parent.name for p in DATASET.glob("*/CHGCAR"))


def sample_ids(n, seed=2026):
    return random.Random(seed).sample(all_ids(), n)


def load(mid, **kw):
    """One run as the rerun read it (``kw`` go to read_vasp)."""
    return pydemi.read_vasp(DATASET / mid / "CHGCAR", **{**READ_OPTIONS, **kw})


def results():
    """The dataset rerun table, indexed by run name."""
    df = pd.read_csv(RESULTS, low_memory=False)
    df["id"] = df["path"].str.split("/").str[-2]
    return df.set_index("id")
