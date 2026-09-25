"""Percolation levels and density floors for the 6,059 structures, with classes and features.

Classes and crystal systems: those of the anisotropy report
(docs/anisotropy_descriptors/data/anisotropy_dataset.csv). Added: the lattice lengths
|a1|, |a2|, |a3| and the grid spacing along each axis (from the CHGCAR header), the
Pauling electronegativity range, and the heaviest block.

Usage:  python dataset_table.py   -> ../data/percolation_dataset.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "docs" / "anisotropy_descriptors" / "scripts"))
from classes import DATASET, header  # noqa: E402

TABLE = REPO / "results" / "prompt_spec" / "descriptors_6000_data_aug13.csv"
CLASSES = REPO / "docs" / "anisotropy_descriptors" / "data" / "anisotropy_dataset.csv"
PERC = ["rho_perc_a", "rho_perc_b", "rho_perc_c", "perc_anisotropy", "rho_min", "rho_min_ratio",
        "rho_min_int", "rho_min_int_ratio"]
OTHER = ["rho_int_mean", "rho_mid_mean", "rho_mid_std", "f_int", "f_bond", "m1", "ELF_bond_avg",
         "f_ELF_localized", "zeta", "charge_FA", "T_eigenvalues_t1", "T_eigenvalues_t3",
         "def_polarity_out", "f_int_def_out", "n_NNM_paw", "shannon_entropy", "M_abs_per_atom",
         "magpie_range_Electronegativity", "magpie_mean_NValence", "magpie_mean_GSbandgap"]


def chi_range(c):
    x = [e.X for e in c.elements if not np.isnan(e.X)]
    return float(max(x) - min(x)) if x else np.nan


def geometry(run):
    lat, _, _ = header(DATASET / run / "CHGCAR")
    return np.linalg.norm(lat, axis=1)


def build():
    df = pd.read_csv(TABLE, low_memory=False)
    df["id"] = df["path"].str.split("/").str[-2]
    cls = pd.read_csv(CLASSES)[["id", "formula", "spacegroup_relaxed", "crystal_system_relaxed",
                                "chem_class", "n_elements"]]
    df = cls.merge(df[["id", "n_atoms", "volume", "magnetic", "grid_shape"] + PERC + OTHER], on="id")
    L = np.array([geometry(r) for r in df["id"]])
    shape = np.array([[int(x) for x in s.split("x")] for s in df["grid_shape"]])
    for k, ax in enumerate("abc"):
        df[f"len_{ax}"] = L[:, k]
        df[f"spacing_{ax}"] = L[:, k] / shape[:, k]
    comp = df["formula"].apply(Composition)
    df["delta_chi"] = comp.apply(chi_range)
    df["heaviest_block"] = comp.apply(lambda c: "f" if any(e.block == "f" for e in c.elements)
                                      else "d" if any(e.block == "d" for e in c.elements) else "sp")
    p = df[["rho_perc_a", "rho_perc_b", "rho_perc_c"]].to_numpy()
    df["rho_perc_min"] = p.min(axis=1)
    df["rho_perc_max"] = p.max(axis=1)
    df["rho_perc_mean"] = p.mean(axis=1)
    df["rho_perc_min_over_mean_rho"] = df["rho_perc_min"] / (df["rho_min_ratio"].where(df["rho_min_ratio"] != 0)
                                                              .pipe(lambda s: df["rho_min"] / s))
    return df


if __name__ == "__main__":
    out = build()
    out.to_csv(HERE.parent / "data" / "percolation_dataset.csv", index=False)
    print(len(out), "structures")
    print(out[PERC + ["rho_perc_min", "rho_perc_min_over_mean_rho", "spacing_a"]].describe(
        percentiles=[0.05, 0.5, 0.95]).T.round(4).to_string())
