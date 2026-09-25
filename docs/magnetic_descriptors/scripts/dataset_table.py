"""The magnetic descriptors for the 6,059 structures, with classes and magnetic-element groups.

Classes and crystal systems: those of the anisotropy report
(docs/anisotropy_descriptors/data/anisotropy_dataset.csv). Magnetic-element group, by the
elements present (first match): "4f" (La-Lu), "5f" (Ac-Lr), "3d" (Sc-Zn), "4d/5d"
(Y-Cd, Hf-Hg), else "sp". Every run of the dataset is collinear spin-polarized; a run
counts as magnetic when sum |m| dV > 0.01 mu_B per atom (pydemi MAGNETIC_TOL).

Usage:  python dataset_table.py   -> ../data/magnetic_dataset.csv
"""
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TABLE = REPO / "results" / "prompt_spec" / "descriptors_6000_data_aug13.csv"
CLASSES = REPO / "docs" / "anisotropy_descriptors" / "data" / "anisotropy_dataset.csv"
MAG = ["m1_spin", "sigma_r2_spin", "f_bond_spin", "mu_site_std", "spin_frustration",
       "spin_charge_correlation", "M_abs_per_atom", "M_net_per_atom", "mu_site_range", "mu_site_max",
       "mu_site_min", "mu_within_element_var", "mu_between_element_var"]
OTHER = ["m1", "sigma_r2", "f_bond", "f_core", "zeta", "def_polarity_out", "m1_site_std",
         "magpie_mean_NdValence", "magpie_mean_NfValence", "magpie_mean_NdUnfilled",
         "magpie_mean_NfUnfilled", "magpie_mean_GSmagmom", "magpie_maximum_GSmagmom"]
META = ["magnetic", "M_abs", "M_net", "spin_mode", "partition", "zval_source"]


def group(c: Composition) -> str:
    z = [e.Z for e in c.elements]
    if any(57 <= x <= 71 for x in z):
        return "4f"
    if any(89 <= x <= 103 for x in z):
        return "5f"
    if any(21 <= x <= 30 for x in z):
        return "3d"
    if any(39 <= x <= 48 or 72 <= x <= 80 for x in z):
        return "4d/5d"
    return "sp"


def build() -> pd.DataFrame:
    df = pd.read_csv(TABLE, low_memory=False)
    df["id"] = df["path"].str.split("/").str[-2]
    cls = pd.read_csv(CLASSES)[["id", "formula", "spacegroup_relaxed", "crystal_system_relaxed",
                                "chem_class", "n_elements"]]
    df = cls.merge(df[["id", "n_atoms", "volume"] + MAG + OTHER + META], on="id")
    comp = df["formula"].apply(Composition)
    df["magnetic_group"] = comp.apply(group)
    for el in ("Mn", "Fe", "Co", "Ni", "Cr", "V", "Gd", "U"):
        df[f"has_{el}"] = comp.apply(lambda c, el=el: el in {str(e) for e in c.elements})
    return df


if __name__ == "__main__":
    out = build()
    out.to_csv(HERE.parent / "data" / "magnetic_dataset.csv", index=False)
    m = out[out["magnetic"]]
    print(len(out), "structures,", len(m), "magnetic")
    print(pd.crosstab(out["magnetic_group"], out["magnetic"]).to_string())
    print(pd.crosstab(out["chem_class"], out["magnetic"]).to_string())
    print(m[MAG[:8]].describe(percentiles=[0.05, 0.5, 0.95]).T.round(4).to_string())
