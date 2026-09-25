"""The site-heterogeneity descriptors for the 6,059 structures, with material classes,
site symmetry and composition features.

Classes and crystal systems: those of the anisotropy report
(docs/anisotropy_descriptors/data/anisotropy_dataset.csv). Site symmetry: spglib on the
relaxed structure in the CHGCAR header (symprec 0.01 A, as in the paper's site-symmetry
analysis): the number of symmetry-distinct sites, and whether every element's sites are
all equivalent (then the within-element variance must vanish).

Usage:  python dataset_table.py   -> ../data/heterogeneity_dataset.csv
"""
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import spglib
from pymatgen.core import Composition

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "docs" / "anisotropy_descriptors" / "scripts"))
from classes import DATASET, header  # noqa: E402

from pydemi.data import atomic_number  # noqa: E402

TABLE = REPO / "results" / "prompt_spec" / "descriptors_6000_data_aug13.csv"
CLASSES = REPO / "docs" / "anisotropy_descriptors" / "data" / "anisotropy_dataset.csv"
BASES = ["m1", "f_bond", "zeta", "mu"]
STATS = ["site_std", "site_range", "site_max", "site_min", "within_element_var", "between_element_var"]
HET = [f"{b}_{s}" for b in BASES for s in STATS]
OTHER = ["m1", "f_bond", "zeta", "f_int", "charge_FA", "def_polarity_out", "M_abs_per_atom",
         "magpie_range_CovalentRadius", "magpie_range_Electronegativity", "magpie_range_Number",
         "magpie_avg_dev_CovalentRadius", "magpie_avg_dev_Electronegativity"]
META = ["max_sites_per_element", "partition", "site_counts"]


def symmetry(run):
    lat, pos, sp = header(DATASET / run / "CHGCAR")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = spglib.get_symmetry_dataset((lat, pos, [atomic_number(x) for x in sp]), 0.01)
    eq = np.asarray(ds.equivalent_atoms)
    s = np.array(sp)
    distinct = {e: len(set(eq[s == e])) for e in dict.fromkeys(sp)}
    return {"id": run, "n_distinct_sites": int(len(set(eq))),
            "max_distinct_per_element": int(max(distinct.values())),
            "all_sites_equivalent_per_element": bool(max(distinct.values()) == 1)}


def build():
    df = pd.read_csv(TABLE, low_memory=False)
    df["id"] = df["path"].str.split("/").str[-2]
    cls = pd.read_csv(CLASSES)[["id", "formula", "spacegroup_relaxed", "crystal_system_relaxed",
                                "chem_class", "n_elements"]]
    df = cls.merge(df[["id", "n_atoms", "volume", "magnetic"] + HET + OTHER + META], on="id")
    with ProcessPoolExecutor(8) as ex:
        sym = pd.DataFrame(list(ex.map(symmetry, df["id"], chunksize=20)))
    df = df.merge(sym, on="id")
    comp = df["formula"].apply(Composition)
    df["heaviest_block"] = comp.apply(lambda c: "f" if any(e.block == "f" for e in c.elements)
                                      else "d" if any(e.block == "d" for e in c.elements) else "sp")
    return df


if __name__ == "__main__":
    out = build()
    out.to_csv(HERE.parent / "data" / "heterogeneity_dataset.csv", index=False)
    print(len(out), "structures")
    print(out[HET].describe(percentiles=[0.05, 0.5, 0.95]).T.round(5).to_string())
    print("all sites equivalent per element:", out["all_sites_equivalent_per_element"].sum())
    print("one site per element (within NaN):", out["m1_within_element_var"].isna().sum())
