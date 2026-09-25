"""
Add the robust-extension column (ellip_bond_bounded_avg, and its __flag) to the
dataset rerun table without recomputing the other 232 descriptors.

Each structure is read exactly as in run_rerun.py (CHGCAR; ZVAL / RCORE from
the OUTCAR or dataset_paw_table.json), and only ellip_bond_bounded_avg is
evaluated with the default options. The column is inserted in registry order
(after ellip_bond_std); descriptors_6000_data_aug13.csv is rewritten in place
and the previous table kept as descriptors_6000_data_aug13.before_robust.csv.

Usage (niced, one thread per worker):
    OMP_NUM_THREADS=1 nice -n 10 python results/prompt_spec/add_robust_ellipticity.py --workers 16
"""

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import pydemi
from pydemi.constants import BOHR_ANGSTROM
from pydemi.descriptors.registry import REGISTRY, Sentinel, make_options

HERE = Path(__file__).resolve().parent
TABLE = HERE / "descriptors_6000_data_aug13.csv"
NAME = "ellip_bond_bounded_avg"
_paw = json.loads((HERE / "dataset_paw_table.json").read_text())
READ_OPTIONS = {"zval": {e: v["zval"] for e, v in _paw.items()},
                "paw_radii": {e: v["rcore_bohr"] * BOHR_ANGSTROM for e, v in _paw.items()}}


def one(path: str) -> dict:
    t0 = time.perf_counter()
    try:
        vd = pydemi.read_vasp(path, **READ_OPTIONS)
        r = REGISTRY[NAME].func(vd.with_options(make_options(extensions=("robust",))))
        value, flag = (r.value, 1) if isinstance(r, Sentinel) else (float(r), 0)
        return {"path": path, NAME: value, f"{NAME}__flag": flag, "t": time.perf_counter() - t0}
    except Exception as exc:                                      # noqa: BLE001
        return {"path": path, NAME: np.nan, f"{NAME}__flag": np.nan, "error_robust": repr(exc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    df = pd.read_csv(TABLE, low_memory=False)
    if NAME in df.columns:
        print(f"{NAME} already present; nothing to do")
        return
    with ProcessPoolExecutor(a.workers) as ex:
        new = pd.DataFrame(list(ex.map(one, df["path"], chunksize=4)))
    errors = int(new.get("error_robust", pd.Series(dtype=object)).notna().sum())
    shutil.copy(TABLE, HERE / "descriptors_6000_data_aug13.before_robust.csv")
    out = df.merge(new[["path", NAME, f"{NAME}__flag"]], on="path", how="left", validate="1:1")
    cols = list(df.columns)
    cols.insert(cols.index("ellip_bond_std") + 1, NAME)
    cols.append(f"{NAME}__flag")
    out[cols].to_csv(TABLE, index=False)
    print(f"{len(out)} rows; {errors} errors; {int(new[f'{NAME}__flag'].sum())} sentinel; "
          f"median {new[NAME].median():.4f}; {new['t'].sum() / 3600:.1f} CPU-h")


if __name__ == "__main__":
    main()
