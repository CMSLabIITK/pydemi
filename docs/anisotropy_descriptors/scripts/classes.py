"""Material classes, crystal systems and the four anisotropy descriptors for the 6,059 structures.

Chemical class (first match wins, by the anions present):
  elemental -> halide (F, Cl, Br, I) -> oxide (O) -> chalcogenide (S, Se, Te)
  -> pnictide (N, P, As) -> hydride (H) -> boride/carbide (B, C) -> intermetallic.
Crystal system two ways: from the space-group number in the run name (<formula>_<number>,
the starting structure) and from spglib on the relaxed structure in the CHGCAR header
(symprec 1e-3 A); the two differ where the relaxation broke the symmetry.

Usage:  python classes.py        -> ../data/anisotropy_dataset.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import spglib
from pymatgen.core import Composition

from pydemi.data import atomic_number

DATASET = Path("/data/sai/new_charge/6000_data_aug13")

HERE = Path(__file__).resolve().parent
TABLE = HERE.parents[2] / "results" / "prompt_spec" / "descriptors_6000_data_aug13.csv"
COLS = ["zeta", "zeta_ELF", "T_eigenvalues_t1", "T_eigenvalues_t2", "T_eigenvalues_t3", "charge_FA"]
OTHER = ["m1", "f_core", "f_bond", "f_int", "lnf_charge_weighted", "ELF_bond_avg", "f_ELF_localized",
         "rho_min_int_ratio", "rho_int_mean", "shannon_entropy", "fisher_information", "perc_anisotropy",
         "def_polarity_out", "M_abs_per_atom", "zeta_site_std"]
ORDER = ["elemental", "intermetallic", "boride/carbide", "hydride", "pnictide", "chalcogenide",
         "oxide", "halide"]
SYSTEMS = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"]


def chem_class(c: Composition) -> str:
    el = {str(e) for e in c.elements}
    if len(el) == 1:
        return "elemental"
    for name, anions in (("halide", {"F", "Cl", "Br", "I"}), ("oxide", {"O"}),
                         ("chalcogenide", {"S", "Se", "Te"}), ("pnictide", {"N", "P", "As"}),
                         ("hydride", {"H"}), ("boride/carbide", {"B", "C"})):
        if el & anions:
            return name
    return "intermetallic"


def crystal_system(sg: int) -> str:
    for hi, name in ((2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"), (142, "tetragonal"),
                     (167, "trigonal"), (194, "hexagonal"), (230, "cubic")):
        if sg <= hi:
            return name
    raise ValueError(sg)


def header(path):
    """Lattice, fractional coordinates and species from a VASP 5 CHGCAR header."""
    with open(path) as fh:
        lines = [next(fh) for _ in range(7)]
        scale = float(lines[1].split()[0])
        lat = np.array([[float(x) for x in lines[k].split()] for k in (2, 3, 4)]) * scale
        els, counts = lines[5].split(), [int(x) for x in lines[6].split()]
        mode = next(fh).strip().lower()
        if mode.startswith("s"):
            mode = next(fh).strip().lower()
        pos = np.array([[float(x) for x in next(fh).split()[:3]] for _ in range(sum(counts))])
    if mode.startswith(("c", "k")):
        pos = pos * scale @ np.linalg.inv(lat)
    return lat, pos, [e.split("_")[0] for e, c in zip(els, counts) for _ in range(c)]


def relaxed_spacegroup(run: str, symprec: float = 1e-3) -> int:
    lat, pos, sp = header(DATASET / run / "CHGCAR")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = spglib.get_symmetry_dataset((lat, pos, [atomic_number(x) for x in sp]), symprec)
    return int(ds.number)


def build() -> pd.DataFrame:
    df = pd.read_csv(TABLE, low_memory=False)
    df["id"] = df["path"].str.split("/").str[-2]
    df["formula"] = df["id"].str.rsplit("_", n=1).str[0]
    df["spacegroup"] = df["id"].str.rsplit("_", n=1).str[1].astype(int)
    comp = df["formula"].apply(Composition)
    df["chem_class"] = comp.apply(chem_class)
    df["crystal_system"] = df["spacegroup"].apply(crystal_system)
    df["spacegroup_relaxed"] = df["id"].apply(relaxed_spacegroup)
    df["crystal_system_relaxed"] = df["spacegroup_relaxed"].apply(crystal_system)
    df["n_elements"] = comp.apply(lambda c: len(c.elements))
    keep = ["id", "formula", "spacegroup", "crystal_system", "spacegroup_relaxed",
            "crystal_system_relaxed", "chem_class", "n_elements", "n_atoms",
            "volume", "magnetic", "grid_shape"] + COLS + OTHER
    return df[keep]


if __name__ == "__main__":
    out = build()
    out.to_csv(HERE.parent / "data" / "anisotropy_dataset.csv", index=False)
    print(out["chem_class"].value_counts().to_string())
    print(out["crystal_system"].value_counts().to_string())
    print("relaxed != named crystal system:", int((out.crystal_system != out.crystal_system_relaxed).sum()))
