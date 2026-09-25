"""Descriptors of the DFT and the ChargE3Net-predicted densities of the 605 test structures.

Every density -- the DFT total density (magnetization dropped) and each model's
prediction on the same grid -- goes through the same pydemi path:
VolumetricData(structure, rho) with ZVAL / RCORE from the dataset PAW table,
domains bonding + structural + heterogeneity, paw extension, default options.
--renorm rescales the prediction to N = sum ZVAL, which is what a workflow
that starts from a structure alone knows.

The densities are the ChargE3Net inputs and full-grid test predictions
(e/Angstrom^3 on the DFT grid, one .npy per structure); out/ml/test_structures.json
holds the cell, species and fractional coordinates of the test split, taken
from the same inputs. pydemi itself does not depend on the model.

Usage:
    python n_ml_descriptors.py dft out/ml/desc_dft.csv
    python n_ml_descriptors.py PRED_DIR out/ml/desc_MODEL.csv [--renorm]
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import pydemi
from pydemi.io.base import Grid, Lattice, Structure, VolumetricData

from common import OUT, READ_OPTIONS, WORKERS

C3N = Path("/home/shubham/charge3net/charge3net/data")
DFT = C3N / "your_dataset_inputs"
PRED = {"scratch": C3N / "test_set_predictions_raw" / "cubes",
        "finetune": C3N / "test_set_predictions_finetune" / "cubes"}
ZVAL, RPAW = READ_OPTIONS["zval"], READ_OPTIONS["paw_radii"]
DOMAINS = ["bonding", "structural", "heterogeneity"]


def one(args):
    name, s, src, renorm = args
    t0 = time.perf_counter()
    try:
        st = Structure(Lattice(np.array(s["cell"])), s["symbols"], np.array(s["frac"]))
        rho = np.load((DFT if src == "dft" else Path(src)) / f"{name}.npy")
        rho = rho[..., 0] if rho.ndim == 4 else rho
        dV = st.volume / rho.size
        n_raw = float(rho.sum() * dV)
        n_ref = float(sum(ZVAL[e] for e in s["symbols"]))
        if renorm:
            rho = rho * (n_ref / n_raw)
        vd = VolumetricData(st, Grid(np.ascontiguousarray(rho, dtype=np.float64), st.lattice),
                            zval={e: ZVAL[e] for e in st.elements},
                            paw_radii={e: RPAW[e] for e in st.elements})
        f, meta = pydemi.featurize(vd, domains=DOMAINS, extensions=["paw"], return_metadata=True)
        return {"name": name, **f, "N": n_raw, "N_ref": n_ref, "euler_consistency": meta["euler_consistency"],
                "def_charge_mismatch": meta["def_charge_mismatch"], "error": "",
                "wall_time_s": time.perf_counter() - t0}
    except Exception as exc:                                      # noqa: BLE001
        return {"name": name, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="'dft', a model name (%s) or a directory of .npy predictions" % ", ".join(PRED))
    ap.add_argument("out")
    ap.add_argument("--renorm", action="store_true")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    src = str(PRED.get(a.src, a.src))
    structures = json.loads((OUT / "ml" / "test_structures.json").read_text())
    jobs = [(n, s, src, a.renorm) for n, s in structures.items()]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, jobs, chunksize=1))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"{a.out}: {len(df)} rows, {(df['error'] != '').sum()} errors")


if __name__ == "__main__":
    main()
