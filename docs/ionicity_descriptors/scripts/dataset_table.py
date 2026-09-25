"""The descriptors of this report for the 6,059 structures, with classes and features.

Columns: V_spread, V_int_min, rho_mid_mean, rho_mid_std, lap_concentration,
lap_concentration_valence, f_int, lnf, fint_over_lnf = f_int / lnf (the feature of the
legacy grid_ionicity), the Pauling ionicity 1 - exp(-dchi^2 / 4) with dchi the Pauling
electronegativity range (the legacy compositional `ionicity`), and the spread of the
valence counts over the sites (std over sites of ZVAL_i, from the dataset PAW table).
Classes: those of the anisotropy report.

Usage:  python dataset_table.py   -> ../data/ionicity_dataset.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TABLE = REPO / "results" / "prompt_spec" / "descriptors_6000_data_aug13.csv"
CLASSES = REPO / "docs" / "anisotropy_descriptors" / "data" / "anisotropy_dataset.csv"
PAW = json.loads((REPO / "results" / "prompt_spec" / "dataset_paw_table.json").read_text())
COLS = ["V_spread", "V_int_min", "rho_mid_mean", "rho_mid_std", "lap_concentration",
        "lap_concentration_valence", "f_int", "lnf", "lnf_charge_weighted", "f_bond", "m1",
        "def_polarity_out", "rho_min_int_ratio", "rho_perc_a", "bond_charge_transfer_pair_std",
        "ELF_bond_avg", "zeta", "magpie_range_Electronegativity", "magpie_mean_NValence", "n_atoms",
        "volume", "site_counts", "magnetic"]


def chi_range(c):
    x = [e.X for e in c.elements if not np.isnan(e.X)]
    return float(max(x) - min(x)) if x else np.nan


def zval_std(site_counts):
    counts = dict(item.split(":") for item in site_counts.split(","))       # "Ac:2,Cu:1,Ge:1"
    z = np.concatenate([[PAW[e]["zval"]] * int(n) for e, n in counts.items()])
    return float(np.std(z))


def build():
    df = pd.read_csv(TABLE, low_memory=False)
    df["id"] = df["path"].str.split("/").str[-2]
    cls = pd.read_csv(CLASSES)[["id", "formula", "spacegroup", "spacegroup_relaxed",
                                "crystal_system_relaxed", "chem_class", "n_elements"]]
    df = cls.merge(df[["id"] + COLS], on="id")
    comp = df["formula"].apply(Composition)
    df["delta_chi"] = comp.apply(chi_range)
    df["pauling_ionicity"] = 1 - np.exp(-df["delta_chi"] ** 2 / 4)
    df["fint_over_lnf"] = df["f_int"] / df["lnf"]
    df["zval_site_std"] = df["site_counts"].apply(zval_std)
    df["rho_mid_cv"] = df["rho_mid_std"] / df["rho_mid_mean"]
    return df


if __name__ == "__main__":
    out = build()
    out.to_csv(HERE.parent / "data" / "ionicity_dataset.csv", index=False)
    print(len(out))
    print(out[["V_spread", "rho_mid_std", "rho_mid_cv", "lap_concentration", "lap_concentration_valence",
               "fint_over_lnf", "pauling_ionicity", "zval_site_std"]].describe(percentiles=[.05, .5, .95]).T.round(4))
