"""
Rerun of the 6,059-structure dataset with the prompt.md build of pydemi.

All domains + the paw extension, default options (FFT derivatives, nearest
partition, shells 0.8/1.5, deformation reference auto -> tabulated),
CHGCAR only (companions off, so every row comes from the same inputs).
Runs without an OUTCAR take ZVAL / RCORE from dataset_paw_table.json
(built from the dataset's 4,901 OUTCARs by tools/paw_table_from_outcars.py).

Chunks of 500 are written to chunks/ and skipped when present (resumable);
the merged table is descriptors_6000_data_aug13.csv.

Usage (niced, one thread per worker):
    OMP_NUM_THREADS=1 nice -n 10 python results/prompt_spec/run_rerun.py --workers 24
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import pydemi
from pydemi.batch import find_runs
from pydemi.constants import BOHR_ANGSTROM

ROOT = Path("/data/sai/new_charge/6000_data_aug13")
HERE = Path(__file__).resolve().parent
CHUNK = 500


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    table = json.loads((HERE / "dataset_paw_table.json").read_text())
    read_options = {"zval": {e: v["zval"] for e, v in table.items()},
                    "paw_radii": {e: v["rcore_bohr"] * BOHR_ANGSTROM for e, v in table.items()}}
    paths = find_runs(ROOT, "*/CHGCAR")
    (HERE / "chunks").mkdir(exist_ok=True)
    t0 = time.time()
    print(f"pydemi {pydemi.__version__}: {len(paths)} structures, {a.workers} workers", flush=True)
    for k in range(0, len(paths), CHUNK):
        out = HERE / "chunks" / f"part_{k // CHUNK:02d}.csv"
        if out.exists():
            continue
        df = pydemi.featurize_batch(paths[k:k + CHUNK], n_workers=a.workers, progress=True,
                                    read_options=read_options, extensions=["paw"])
        df.to_csv(out, index=False)
        print(f"chunk {k // CHUNK}: {len(df)} rows, {(df['error'] != '').sum()} errors, "
              f"{(time.time() - t0) / 60:.1f} min elapsed", flush=True)
    parts = sorted((HERE / "chunks").glob("part_*.csv"))
    full = pd.concat([pd.read_csv(p, keep_default_na=False, na_values=[""]) for p in parts],
                     ignore_index=True)
    full.to_csv(HERE / "descriptors_6000_data_aug13.csv", index=False)
    print(f"done: {len(full)} rows, {(full['error'].fillna('') != '').sum()} errors, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
