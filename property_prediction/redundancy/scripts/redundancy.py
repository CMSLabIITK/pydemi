"""Which of the 56 descriptors are unused, redundant or uninformative?

Computed on the TRAINING split only (val/test are kept for the retraining that follows).
    unused      constant (relative std < 1e-9) or nearly so (one value in >= 95 % of structures)
    redundant   |Spearman rho| >= 0.90 with another descriptor; descriptors are grouped by
                complete linkage on 1 - |rho| (every pair in a group has |rho| >= 0.90) and
                the group keeps the member with the largest total SHAP share
    unimportant never among the 10 most important descriptors of any of the 17 properties
                (SHAP of the 55-descriptor LightGBM models, train split; paper Fig. 5)
also reported: max |Spearman rho| of each descriptor with any of the 17 properties.
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
    info["nan_frac"] = X.isna().mean()
    info["n_unique"] = X.nunique()
    info["mode_frac"] = [X[f].value_counts(normalize=True).iloc[0] if X[f].notna().any() else 1.0 for f in names]
    # constant up to round-off (lap_concentration = 1/2 to 1e-16) or one value in >= 95 %
    info["rel_std"] = X.std() / X.abs().mean().replace(0, np.nan)
    info["unused"] = (info.rel_std.fillna(0) < 1e-9) | (info.mode_frac >= 0.95)

    top = json.load(open(ROOT / "results/study_v1_top10/top10.json"))
    share = pd.DataFrame({t: dict(top[t]["ranking"]) for t in PROPS}).reindex(names).fillna(0.0)
    info["shap_mean"] = share.mean(axis=1)
    info["shap_max"] = share.max(axis=1)
    info["n_top10"] = [sum(f in top[t]["features"] for t in PROPS) for f in names]
    info["unimportant"] = info.n_top10 == 0

    rel = tr[names + PROPS].astype(float).corr(method="spearman").loc[names, PROPS].abs()
    info["max_rho_property"] = rel.max(axis=1)
    info["best_property"] = rel.idxmax(axis=1)

    live = [f for f in names if not info.at[f, "unused"]]
    rho = X[live].corr(method="spearman").abs().fillna(0.0)
    np.fill_diagonal(rho.values, 1.0)
    pairs = [(a, b, rho.at[a, b]) for i, a in enumerate(live) for b in live[i + 1:] if rho.at[a, b] >= CUT]
    pd.DataFrame(pairs, columns=["a", "b", "abs_spearman"]).sort_values("abs_spearman", ascending=False) \
        .to_csv(OUT / "pairs.csv", index=False)
    info["max_rho_descriptor"] = pd.Series({f: rho.loc[f].drop(f).max() for f in live})
    info["most_similar"] = pd.Series({f: rho.loc[f].drop(f).idxmax() for f in live})

    lab = fcluster(linkage(squareform(1 - rho.values, checks=False), "complete"), 1 - CUT, "distance")
    groups, keep = [], set()
    for g in np.unique(lab):
        mem = sorted([f for f, l in zip(live, lab) if l == g], key=lambda f: -info.at[f, "shap_mean"])
        keep.add(mem[0])
        if len(mem) > 1:
            groups.append({"keep": mem[0], "drop": mem[1:],
                           "min_abs_spearman": float(rho.loc[mem, mem].values.min())})
    info["redundant"] = [f in live and f not in keep for f in names]
    pd.DataFrame(groups).to_csv(OUT / "groups.csv", index=False)

    sets = {
        "all55": [f for f in names if not info.at[f, "unused"]],
        "no_redundant": [f for f in names if not (info.at[f, "unused"] or info.at[f, "redundant"])],
        "important34": [f for f in names if not (info.at[f, "unused"] or info.at[f, "unimportant"])],
    }
    sets["lean"] = [f for f in sets["no_redundant"] if f in sets["important34"]]
    json.dump(sets, open(OUT / "sets.json", "w"), indent=1)
    info.round(4).to_csv(OUT / "descriptors.csv", index_label="descriptor")

    pd.set_option("display.width", 220, "display.max_rows", 100)
    print(info.sort_values(["unused", "redundant", "shap_mean"], ascending=[False, False, True])[
        ["domain", "unused", "redundant", "unimportant", "n_top10", "shap_mean", "max_rho_descriptor",
         "most_similar", "max_rho_property", "best_property", "nan_frac", "mode_frac"]].round(3).to_string())
    print("\ngroups (|rho| >= %.2f):" % CUT)
    for g in groups:
        print(f"  keep {g['keep']:28s} drop {', '.join(g['drop'])}  (min |rho| {g['min_abs_spearman']:.3f})")
    print({k: len(v) for k, v in sets.items()})


if __name__ == "__main__":
    main()
