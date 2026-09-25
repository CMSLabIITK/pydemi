"""The seven deformation descriptors (whole cell and outside the PAW spheres) for the
6,059 structures, with material classes and composition features.

Classes and crystal systems are those of the anisotropy report
(docs/anisotropy_descriptors/data/anisotropy_dataset.csv, built by its classes.py):
chemical class by the anions present (elemental, intermetallic, boride/carbide, hydride,
pnictide, chalcogenide, oxide, halide), crystal system from the relaxed structure.
Added here: the Pauling electronegativity difference max - min over the elements present
(pymatgen), and whether the structure contains a d- or f-block element.

Usage:  python dataset_table.py   -> ../data/deformation_dataset.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TABLE = REPO / "results" / "prompt_spec" / "descriptors_6000_data_aug13.csv"
CLASSES = REPO / "docs" / "anisotropy_descriptors" / "data" / "anisotropy_dataset.csv"
DEF = ["m1_def", "m2_def", "sigma_r2_def", "f_bond_def", "f_int_def", "f_bond_dep", "def_polarity"]
OUTV = [f"{n}_out" for n in DEF] + ["def_out_volume_fraction"]
OTHER = ["m1", "sigma_r2", "f_core", "f_bond", "f_int", "lnf_charge_weighted", "ELF_bond_avg",
         "f_ELF_localized", "rho_mid_mean", "rho_int_mean", "bond_charge_transfer_pair_mean",
         "bond_charge_transfer_pair_std", "zeta", "charge_FA", "shannon_entropy",
         "M_abs_per_atom", "magpie_range_Electronegativity", "magpie_mean_NValence",
         "magpie_mean_CovalentRadius"]
META = ["def_charge_mismatch", "zval_source", "paw_radii_source", "deformation_reference"]


def chi_range(c: Composition) -> float:
    x = [e.X for e in c.elements if not np.isnan(e.X)]
    return float(max(x) - min(x)) if x else np.nan


def block(c: Composition) -> str:
    b = {e.block for e in c.elements}
    return "f" if "f" in b else "d" if "d" in b else "sp"


def build() -> pd.DataFrame:
    df = pd.read_csv(TABLE, low_memory=False)
    df["id"] = df["path"].str.split("/").str[-2]
    cls = pd.read_csv(CLASSES)[["id", "formula", "spacegroup_relaxed", "crystal_system_relaxed",
                                "chem_class", "n_elements"]]
    df = cls.merge(df[["id", "n_atoms", "volume", "magnetic"] + DEF + OUTV + OTHER + META], on="id")
    comp = df["formula"].apply(Composition)
    df["delta_chi"] = comp.apply(chi_range)
    df["heaviest_block"] = comp.apply(block)
    df["electrons_per_A3"] = np.nan
    return df.drop(columns="electrons_per_A3")


if __name__ == "__main__":
    out = build()
    out.to_csv(HERE.parent / "data" / "deformation_dataset.csv", index=False)
    print(len(out), "structures")
    print(out[DEF + OUTV].describe(percentiles=[0.05, 0.5, 0.95]).T.round(4).to_string())
    print(out["zval_source"].value_counts().to_string())
    print(out["deformation_reference"].value_counts().to_string())
