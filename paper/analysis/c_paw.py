"""PAW pseudo-density effects on real VASP CHGCARs."""
import csv
import json
import warnings

import numpy as np

from common import DATASET, OUT, RESULTS
from pydemi import Engine
from pydemi.descriptors.deformation import promolecule
from pydemi.descriptors.sites import hirshfeld_charges
from pydemi.descriptors.topology import ascent_basins, morse_census, topology_family

out = {}

# 1. negative pseudo-densities and the interstitial floor, dataset-wide
rows = list(csv.DictReader(open(RESULTS)))
num = lambda r, c: float(r[c]) if r[c] not in ("", "nan") else np.nan
rmin = np.array([num(r, "rho_min") for r in rows])
ratio = np.array([num(r, "rho_min_ratio") for r in rows])
iratio = np.array([num(r, "rho_min_int_ratio") for r in rows])
out["negative_density"] = {
    "n": len(rows), "n_rho_min_negative": int((rmin < 0).sum()),
    "rho_min_ratio_median": float(np.nanmedian(ratio)),
    "rho_min_int_ratio_negative": int((iratio < 0).sum()),
    "rho_min_int_ratio_median": float(np.nanmedian(iratio)),
    "rho_min_int_nan": int(np.isnan(iratio).sum())}

# 2. pseudized nuclei: CaSi3Pt
eng = Engine.from_vasp_dir(DATASET / "CaSi3Pt_107")
rho = eng["rho"].values
geo = eng.geometry()
c = morse_census(rho)
basin = ascent_basins(rho, eng.structure.lattice, c["lower_mask"])
q = np.bincount(basin.ravel(), weights=rho.ravel(), minlength=rho.size) * eng.grid().dV
maxima = np.flatnonzero(c["maxima"])
d, own = geo.distance.ravel()[maxima], geo.atom_index.ravel()[maxima]
si = [k for k, m in enumerate(maxima) if eng.structure.species[own[k]] == "Si"]
fixed = topology_family(eng)
eng.paw_radii = None
fixed_cutoff = topology_family(eng, r_cut=0.8)
kmin = np.argmin(rho)
out["CaSi3Pt"] = {
    "grid": list(rho.shape), "rho_min": float(rho.min()),
    "rho_min_distance_A": float(geo.distance.ravel()[kmin]),
    "rho_min_nearest": eng.structure.species[geo.atom_index.ravel()[kmin]],
    "si_maxima_distances_A": sorted(float(d[k]) for k in si),
    "si_maxima_charge_e": float(sum(q[maxima[k]] for k in si)),
    "nuclear_maxima_found_for": sorted({eng.structure.species[own[k]] for k in range(len(maxima)) if d[k] < 0.1}),
    "n_NNM_fixed_0.8A": fixed_cutoff["n_NNM"], "Q_NNM_fixed_0.8A": fixed_cutoff["Q_NNM"],
    "n_NNM_paw_cutoff": fixed["n_NNM"], "paw_radii_A": None}
eng2 = Engine.from_vasp_dir(DATASET / "CaSi3Pt_107")
out["CaSi3Pt"]["paw_radii_A"] = eng2.paw_radii

# 3. deformation density of CHGCAR against all-electron free-atom valence: FeNi3
eng = Engine.from_vasp_dir(DATASET / "FeNi3_221")
rho = eng["rho"].values
pro = promolecule(eng)
geo = eng.geometry()
dV = eng.grid().dV
fe = eng.structure.species.index("Fe")
shells = [(0, 0.4), (0.4, 0.8), (0.8, 1.2), (1.2, 1.5), (1.5, 3.0)]
table = []
for lo, hi in shells:
    m = (geo.atom_index == fe) & (geo.distance >= lo) & (geo.distance < hi)
    table.append({"shell_A": [lo, hi], "chgcar_e": float(rho[m].sum() * dV),
                  "free_atom_valence_e": float(pro[m].sum() * dV)})
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from pydemi.descriptors.deformation import deformation_family
    dd = deformation_family(eng, field="rho")
    qh = hirshfeld_charges(eng, "rho")
out["FeNi3"] = {"zval": eng.valence(), "Q_tot": eng["rho"].integral(),
                "promolecule_charge": float(pro.sum() * dV), "fe_radial_table": table,
                "deformation_chgcar_route": {k: dd[k] for k in ("f_bond_dep", "bond_charge_transfer", "def_polarity")},
                "hirshfeld_charges_chgcar": dict(zip([f"{s}{i}" for i, s in enumerate(eng.structure.species)], map(float, qh)))}
(OUT / "paw.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
