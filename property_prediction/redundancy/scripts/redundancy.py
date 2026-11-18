"""The descriptors that correlate strongly with another one (Tables S5 and S6 of the paper).

Computed on the TRAINING split only. A descriptor is redundant when |Spearman rho| >= 0.90
with another one: descriptors are grouped by complete linkage on 1 - |rho| (every pair in a
group has |rho| >= 0.90) and each group keeps the member with the largest mean SHAP share
over the 17 properties (55-descriptor LightGBM models, training split).
-> redundancy/results/descriptors.csv, pairs.csv, groups.csv, sets.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydemi
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OUT = HERE / "results"
CUT = 0.90
PROPS = ["bulk_modulus", "shear_modulus", "youngs_modulus", "C11", "C12", "C44", "debye_temperature",
         "sound_velocity_transverse", "sound_velocity_longitudinal", "thermal_conductivity_clarke",
         "thermal_conductivity_cahill", "formation_energy", "cohesive_energy", "band_gap", "is_metal",
         "is_gap_direct", "refractive_index"]


def main() -> None:
    d = pd.read_csv(ROOT / "data/dataset.csv")
    tr = d[d.split == "train"]
    cat = pydemi.catalogue().set_index("name")
    names = list(pydemi.descriptor_names())
    X = tr[names].astype(float)

    info = pd.DataFrame(index=names)
    info["domain"] = cat.loc[names, "domain"]
    top = json.load(open(ROOT / "results/study_v1_top10/top10.json"))
    share = pd.DataFrame({t: dict(top[t]["ranking"]) for t in PROPS}).reindex(names).fillna(0.0)
    info["shap_mean"] = share.mean(axis=1)

    rho = X.corr(method="spearman").abs().fillna(0.0)
    np.fill_diagonal(rho.values, 1.0)
    pairs = [(a, b, rho.at[a, b]) for i, a in enumerate(names) for b in names[i + 1:] if rho.at[a, b] >= CUT]
    pd.DataFrame(pairs, columns=["a", "b", "abs_spearman"]).sort_values("abs_spearman", ascending=False) \
        .to_csv(OUT / "pairs.csv", index=False)
    info["max_rho_descriptor"] = pd.Series({f: rho.loc[f].drop(f).max() for f in names})
    info["most_similar"] = pd.Series({f: rho.loc[f].drop(f).idxmax() for f in names})

    lab = fcluster(linkage(squareform(1 - rho.values, checks=False), "complete"), 1 - CUT, "distance")
    groups, keep = [], set()
    for g in np.unique(lab):
        mem = sorted([f for f, l in zip(names, lab) if l == g], key=lambda f: -info.at[f, "shap_mean"])
        keep.add(mem[0])
        if len(mem) > 1:
            groups.append({"keep": mem[0], "drop": mem[1:],
                           "min_abs_spearman": float(rho.loc[mem, mem].values.min())})
    info["redundant"] = [f not in keep for f in names]
    pd.DataFrame(groups).to_csv(OUT / "groups.csv", index=False)

    sets = {"all55": names, "no_redundant": [f for f in names if not info.at[f, "redundant"]]}
    json.dump(sets, open(OUT / "sets.json", "w"), indent=1)
    info.round(4).to_csv(OUT / "descriptors.csv", index_label="descriptor")

    pd.set_option("display.width", 220, "display.max_rows", 100)
    print(info.sort_values(["redundant", "shap_mean"], ascending=[False, True])[
        ["domain", "redundant", "shap_mean", "max_rho_descriptor", "most_similar"]].round(3).to_string())
    print("\ngroups (|rho| >= %.2f):" % CUT)
    for g in groups:
        print(f"  keep {g['keep']:28s} drop {', '.join(g['drop'])}  (min |rho| {g['min_abs_spearman']:.3f})")
    print({k: len(v) for k, v in sets.items()})


if __name__ == "__main__":
    main()
