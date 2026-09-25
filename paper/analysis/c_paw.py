"""PAW pseudo-density effects on real VASP CHGCARs.

1. Negative pseudo-densities and the interstitial floor, dataset-wide (rerun table).
2. Pseudized nuclei: the maxima of CaSi3Pt and the non-nuclear-maximum counts.
3. The CHGCAR against the free-atom valence reference around Fe in FeNi3.

Output: out/paw.json
"""
import json

import numpy as np

from common import OUT, load, results
from pydemi.descriptors import featurize
from pydemi.descriptors.bonding import promolecule_density
from pydemi.descriptors.registry import augmentation_radii, geometry
from pydemi.descriptors.structural import census
from pydemi.operators.topology import ascent_basins

out = {}

# 1 ------------------------------------------------------------------ dataset-wide
df = results()
iflag = df["rho_min_int__flag"] == 1
out["negative_density"] = {
    "n": len(df), "n_rho_min_negative": int((df["rho_min"] < 0).sum()),
    "rho_min_ratio_median": float(df["rho_min_ratio"].median()),
    "n_rho_min_int_negative": int((df.loc[~iflag, "rho_min_int"] < 0).sum()),
    "rho_min_int_ratio_median": float(df.loc[~iflag, "rho_min_int_ratio"].median()),
    "n_no_voxel_outside_spheres": int(iflag.sum()),
    "n_with_nnm": int((df["n_NNM"] > 0).sum()),
    "n_with_nnm_paw": int((df["n_NNM_paw"] > 0).sum()),
    "def_out_volume_fraction": [float(df["def_out_volume_fraction"].quantile(q)) for q in (0.05, 0.5, 0.95)],
    "def_polarity_out": [float(df["def_polarity_out"].quantile(q)) for q in (0.05, 0.5, 0.95)],
}

# 2 ------------------------------------------------------------------ CaSi3Pt
vd = load("CaSi3Pt_107")
rho = vd.rho.data
c = census(vd)
geo = geometry(vd)
basin = ascent_basins(rho, vd.lattice.matrix, c["lower26"])
q = np.bincount(basin.ravel(), weights=rho.ravel(), minlength=rho.size) * vd.rho.dV
maxima = np.flatnonzero(c["maxima"])
d, own = geo.distance.ravel()[maxima], geo.atom_index.ravel()[maxima]
sp = vd.structure.species
si = [k for k in range(len(maxima)) if sp[own[k]] == "Si"]
kmin = int(np.argmin(rho))
f = featurize(vd, domains=["structural"], extensions=["paw"])
out["CaSi3Pt"] = {
    "grid": list(rho.shape), "rho_min": float(rho.min()),
    "rho_min_distance_A": float(geo.distance.ravel()[kmin]),
    "rho_min_nearest": sp[geo.atom_index.ravel()[kmin]],
    "maxima_at_nuclei_of": sorted({sp[own[k]] for k in range(len(maxima)) if d[k] < 0.1}),
    "si_maxima_distances_A": sorted(round(float(d[k]), 3) for k in si),
    "si_maxima_charge_e": float(sum(q[maxima[k]] for k in si)),
    "paw_radii_A": {e: float(r) for e, r in (vd.paw_radii or {}).items()},
    "n_NNM_count_0.8A": f["n_NNM"] * vd.structure.volume, "Q_NNM": f["Q_NNM"],
    "n_NNM_paw_count": f["n_NNM_paw"] * vd.structure.volume, "Q_NNM_paw": f["Q_NNM_paw"],
}

# 3 ------------------------------------------------------------------ FeNi3
vd = load("FeNi3_221")
rho = vd.rho.data
pro = promolecule_density(vd)
geo = geometry(vd)
dV = vd.rho.dV
fe = vd.structure.species.index("Fe")
table = []
for lo, hi in [(0, 0.4), (0.4, 0.8), (0.8, 1.2), (1.2, 1.5), (1.5, 3.0)]:
    m = (geo.atom_index == fe) & (geo.distance >= lo) & (geo.distance < hi)
    table.append({"shell_A": [lo, hi], "chgcar_e": float(rho[m].sum() * dV),
                  "free_atom_valence_e": float(pro[m].sum() * dV)})
out["FeNi3"] = {"zval": vd.zval, "zval_source": vd.sources.get("zval"),
                "paw_radius_Fe_A": float(augmentation_radii(vd)[0][fe]),
                "Q_tot": float(rho.sum() * dV), "promolecule_charge": float(pro.sum() * dV),
                "fe_radial_table": table}

(OUT / "paw.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
