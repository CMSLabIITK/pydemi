"""Build the training tables of the property-prediction study from the raw data files.

Inputs (../data): properties.csv (targets of the JCIM 2026 study), mp_summary.csv,
mp_elasticity.csv, mp_cohesive.csv (fetch_mp.py), descriptors_56.csv (run_descriptors.sh).

Outputs (../data):
  targets.csv              one row per material: ids, split, 23 cleaned targets, flags
  targets_columns.csv      what each target column is: units, source, task, count, cleaning
  descriptors.csv          one row per material: ids, split, the 56 pydemi descriptors
  descriptors_columns.csv  domain, units and formula of each descriptor (pydemi.catalogue)
  dataset.csv              descriptors and targets joined, ready for training
  build_report.txt         counts and what every cleaning rule removed

Flags in targets.csv: ``has_f_element`` (lanthanide or actinide present: the cohesive energy
is less reliable there) and ``born_stable`` (1 if the MP stiffness tensor is positive
definite with smallest eigenvalue above 1 GPa, 0 if not, empty without a tensor).

Every row is keyed by ``structure`` (the dataset folder, = formula_sp of properties.csv)
and ``material_id`` (legacy Materials Project id). ``split`` is the train / val / test
split of the JCIM study.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import pydemi  # noqa: E402

F_ELEMENTS = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
              "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm"}

# column, units, source, task, description
TARGETS = [
    ("bulk_modulus", "GPa", "properties.csv (bulk_hill)", "regression", "bulk modulus K, Voigt-Reuss-Hill"),
    ("shear_modulus", "GPa", "properties.csv (shear_hill)", "regression", "shear modulus G, Voigt-Reuss-Hill"),
    ("youngs_modulus", "GPa", "properties.csv", "regression", "Young's modulus E"),
    ("poisson_ratio", "1", "properties.csv", "regression", "Poisson ratio; values > 0.5 removed (impossible for an isotropic solid)"),
    ("anisotropy_index", "1", "computed from the MP stiffness tensor", "regression", "universal elastic anisotropy A_U = 5 G_V/G_R + K_V/K_R - 6 (0 = isotropic)"),
    ("C11", "GPa", "MP elasticity (IEEE frame, 1 GPa precision)", "regression", "elastic constant C11"),
    ("C12", "GPa", "MP elasticity (IEEE frame, 1 GPa precision)", "regression", "elastic constant C12"),
    ("C44", "GPa", "MP elasticity (IEEE frame, 1 GPa precision)", "regression", "elastic constant C44"),
    ("debye_temperature", "K", "MP elasticity", "regression", "Debye temperature (from elastic moduli and density)"),
    ("sound_velocity_transverse", "m/s", "MP elasticity", "regression", "transverse sound velocity (from elastic moduli)"),
    ("sound_velocity_longitudinal", "m/s", "MP elasticity", "regression", "longitudinal sound velocity (from elastic moduli)"),
    ("thermal_conductivity_clarke", "W/(m K)", "MP elasticity", "regression", "minimum thermal conductivity, Clarke model (from elastic moduli)"),
    ("thermal_conductivity_cahill", "W/(m K)", "MP elasticity", "regression", "minimum thermal conductivity, Cahill model (from elastic moduli)"),
    ("formation_energy", "eV/atom", "MP summary (formation_energy_per_atom)", "regression", "formation energy relative to the elemental solids"),
    ("energy_above_hull", "eV/atom", "MP summary", "regression", "energy above the convex hull (0 = thermodynamically stable)"),
    ("cohesive_energy", "eV/atom", "MP get_cohesive_energy", "regression", "energy relative to free atoms (negative = bound); see has_f_element"),
    ("band_gap", "eV", "MP summary", "regression", "PBE band gap"),
    ("is_metal", "0/1", "MP summary", "classification", "1 if metallic (band gap 0)"),
    ("is_gap_direct", "0/1", "MP summary", "classification", "1 if the gap is direct; only for materials with a gap"),
    ("dielectric_total", "1", "MP summary (e_total)", "regression", "total (electronic + ionic) dielectric constant; heavy-tailed, model log10"),
    ("dielectric_electronic", "1", "MP summary (e_electronic)", "regression", "electronic dielectric constant; heavy-tailed, model log10"),
    ("dielectric_ionic", "1", "MP summary (e_ionic)", "regression", "ionic dielectric constant; heavy-tailed, can be 0"),
    ("refractive_index", "1", "MP summary (n)", "regression", "refractive index"),
]
ELASTIC_MP = ["anisotropy_index", "C11", "C12", "C44", "debye_temperature", "sound_velocity_transverse",
              "sound_velocity_longitudinal", "thermal_conductivity_clarke", "thermal_conductivity_cahill"]


def raw_tensors() -> dict[str, np.ndarray]:
    """The unrounded MP stiffness tensors (Voigt 6x6, GPa) by legacy id. The IEEE-format
    tensor in mp_elasticity.csv is rounded to 1 GPa; stability and A_U do not depend on the
    orientation, so they are computed from these."""
    import gzip
    import json
    docs = json.load(gzip.open(DATA / "mp_raw.json.gz", "rt"))["elasticity"]
    out = {}
    for d in docs:
        et = d.get("elastic_tensor") or {}
        if et.get("raw"):
            out[d["material_id_legacy"]] = np.array(et["raw"], dtype=float)
    return out


MIN_EIGENVALUE = 1.0  # GPa: the precision of the MP tensors; below it a tensor is numerically singular


def born_stable(c: np.ndarray) -> bool:
    """Born criterion: the 6x6 stiffness matrix is positive definite (smallest eigenvalue
    above MIN_EIGENVALUE, so that it can be inverted for the Reuss bounds)."""
    return bool(np.all(np.isfinite(c)) and np.linalg.eigvalsh((c + c.T) / 2).min() > MIN_EIGENVALUE)


def anisotropy(c: np.ndarray) -> float:
    """Universal anisotropy A_U = 5 G_V/G_R + K_V/K_R - 6 (Ranganathan & Ostoja-Starzewski 2008)."""
    s = np.linalg.inv(c)
    kv = (c[0, 0] + c[1, 1] + c[2, 2] + 2 * (c[0, 1] + c[1, 2] + c[0, 2])) / 9
    gv = (c[0, 0] + c[1, 1] + c[2, 2] - (c[0, 1] + c[1, 2] + c[0, 2]) + 3 * (c[3, 3] + c[4, 4] + c[5, 5])) / 15
    kr = 1 / (s[0, 0] + s[1, 1] + s[2, 2] + 2 * (s[0, 1] + s[1, 2] + s[0, 2]))
    gr = 15 / (4 * (s[0, 0] + s[1, 1] + s[2, 2]) - 4 * (s[0, 1] + s[1, 2] + s[0, 2]) + 3 * (s[3, 3] + s[4, 4] + s[5, 5]))
    return float(5 * gv / gr + kv / kr - 6)


def main() -> None:
    report: list[str] = []
    P = pd.read_csv(DATA / "properties.csv")
    S = pd.read_csv(DATA / "mp_summary.csv").set_index("material_id_legacy")
    E = pd.read_csv(DATA / "mp_elasticity.csv").set_index("material_id_legacy")
    C = pd.read_csv(DATA / "mp_cohesive.csv").set_index("material_id_legacy")
    ids = P.set_index("material_id")

    t = pd.DataFrame({"structure": P.formula_sp, "material_id": P.material_id,
                      "formula": P.formula_pretty, "split": P.split})
    t = t.set_index("material_id", drop=False)
    elements = S.reindex(t.index)["formula_pretty"].fillna(t.formula)
    t["has_f_element"] = [int(any(e in F_ELEMENTS for e in pymatgen_elements(f))) for f in elements]

    # mechanical targets of the JCIM study, as published
    t["bulk_modulus"] = ids.bulk_hill
    t["shear_modulus"] = ids.shear_hill
    t["youngs_modulus"] = ids.youngs_modulus
    nu = ids.poisson_ratio
    t["poisson_ratio"] = nu.where(nu < 0.5)
    report.append(f"poisson_ratio: removed {int((nu >= 0.5).sum())} values >= 0.5 "
                  f"(kept {int((nu < 0).sum())} negative values: auxetic is physical)")

    # MP elastic data: only where the tensor is trustworthy
    e = E.reindex(t.index)
    sane = e.bulk_modulus_vrh.between(0, 1000) & e.shear_modulus_vrh.between(0, 1000)
    agree = (e.bulk_modulus_vrh - t.bulk_modulus).abs().lt(1e-3) & \
            (e.shear_modulus_vrh - t.shear_modulus).abs().lt(1e-3)
    raw = raw_tensors()
    stable = pd.Series([mid in raw and born_stable(raw[mid]) for mid in t.index], index=t.index)
    moduli_ok = sane & agree          # Debye, sound velocities, conductivities come from K, G only
    tensor_ok = moduli_ok & stable    # C_ij and A_U need the whole tensor
    report.append(f"MP elastic: {int(e.C11.notna().sum())} materials with a tensor; "
                  f"{int((e.C11.notna() & ~sane).sum())} with K or G outside 0-1000 GPa, "
                  f"{int((sane & ~agree).sum())} whose K or G differ from properties.csv: "
                  f"kept {int(moduli_ok.sum())} for the K,G-derived targets; of these "
                  f"{int((moduli_ok & ~stable).sum())} not Born-stable (smallest stiffness eigenvalue <= {MIN_EIGENVALUE:g} GPa), "
                  f"so {int(tensor_ok.sum())} for C11, C12, C44 and A_U")
    t["born_stable"] = stable.where(e.C11.notna()).map({True: 1, False: 0})
    t["anisotropy_index"] = pd.Series({mid: anisotropy(raw[mid]) for mid in t.index[tensor_ok]}
                                      ).reindex(t.index)
    mp_au = e.universal_anisotropy.where(tensor_ok)
    report.append(f"anisotropy_index: computed from the unrounded tensor; MP's stored value differs "
                  f"by more than 0.01 for {int(((t.anisotropy_index - mp_au).abs() > 0.01).sum())} "
                  f"materials and is negative (impossible) for {int((mp_au < 0).sum())}")
    for c in ("C11", "C12", "C44"):
        t[c] = e[c].where(tensor_ok)
    for c in ("debye_temperature", "sound_velocity_transverse", "sound_velocity_longitudinal",
              "thermal_conductivity_clarke", "thermal_conductivity_cahill"):
        t[c] = e[c].where(moduli_ok)

    # thermodynamics, electronic, dielectric
    s = S.reindex(t.index)
    t["formation_energy"] = s.formation_energy_per_atom
    t["energy_above_hull"] = s.energy_above_hull
    t["cohesive_energy"] = C.reindex(t.index).cohesive_energy_per_atom
    t["band_gap"] = s.band_gap
    t["is_metal"] = s.is_metal.map({True: 1, False: 0})
    t["is_gap_direct"] = s.is_gap_direct.map({True: 1, False: 0}).where(s.band_gap > 0)
    t["dielectric_total"] = s.e_total
    t["dielectric_electronic"] = s.e_electronic
    t["dielectric_ionic"] = s.e_ionic
    t["refractive_index"] = s.n
    report.append(f"MP summary missing for {int(s.band_gap.isna().sum())} materials "
                  "(ids no longer in the MP database): their MP targets are empty")

    cols = ["structure", "material_id", "formula", "split", "has_f_element", "born_stable"] + [c[0] for c in TARGETS]
    t = t[cols].reset_index(drop=True)
    t.to_csv(DATA / "targets.csv", index=False)

    meta = pd.DataFrame(TARGETS, columns=["column", "units", "source", "task", "description"])
    meta["n"] = [int(t[c].notna().sum()) for c in meta.column]
    meta["n_train"] = [int(t.loc[t.split == "train", c].notna().sum()) for c in meta.column]
    meta["n_test"] = [int(t.loc[t.split == "test", c].notna().sum()) for c in meta.column]
    meta.to_csv(DATA / "targets_columns.csv", index=False)

    # descriptors
    names = list(pydemi.descriptor_names())
    D = pd.read_csv(DATA / "descriptors_56.csv")
    D = D.assign(structure=D.path.map(lambda p: Path(p).parent.name))
    bad = D.get("error", pd.Series("", index=D.index)).fillna("").astype(str).ne("")
    d = t[["structure", "material_id", "formula", "split"]].merge(D[["structure"] + names], on="structure",
                                                                 how="left", validate="one_to_one")
    d.to_csv(DATA / "descriptors.csv", index=False)
    cat = pydemi.catalogue().set_index("name").loc[names].reset_index()
    cat.to_csv(DATA / "descriptors_columns.csv", index=False)
    report.append(f"descriptors: {len(D)} computed, {int(bad.sum())} errors, "
                  f"{int(d[names].isna().all(axis=1).sum())} materials without descriptors; "
                  f"constant (to 1e-9): {[n for n in names if d[n].max() - d[n].min() < 1e-9]}; "
                  f"columns with NaN: {({n: int(v) for n, v in d[names].isna().sum().items() if v})}")

    ds = d.merge(t.drop(columns=["material_id", "formula", "split"]), on="structure", validate="one_to_one")
    ds.to_csv(DATA / "dataset.csv", index=False)

    report.append(f"rows: targets {len(t)}, descriptors {len(d)}, dataset {len(ds)} "
                  f"(split {t.split.value_counts().to_dict()})")
    report.append("targets: " + ", ".join(f"{r.column} {r.n}" for r in meta.itertuples()))
    (DATA / "build_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


def pymatgen_elements(formula: str) -> list[str]:
    import re
    return re.findall(r"[A-Z][a-z]?", formula)


if __name__ == "__main__":
    main()
