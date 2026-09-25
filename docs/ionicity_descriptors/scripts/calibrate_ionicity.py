"""Reconstruct the legacy calibrated ionicity on the current descriptor table.

grid_ionicity and ionicity_residual are NOT part of the current pydemi (they were
entries 60 and 62 of the superseded PDF-spec build and were dropped when the library was
rebuilt to prompt.md). This script re-implements their legacy definition
(legacy/pydemi_pdf_spec/calibration/__init__.py) outside the library:

    grid_ionicity     = sigmoid(b0 + sum_j b_j z_j),  z_j = (x_j - mean_j) / std_j,
                        features x = (f_int / lnf, V_spread)   [legacy default:
                        fint_over_lnf and the electronic Hartree site-potential spread],
                        b fitted by least squares to Phillips' spectroscopic ionicity f_i
    ionicity_residual = grid_ionicity - (1 - exp(-dchi^2 / 4))   [Pauling ionicity]

Reference compounds: the Phillips table (data/phillips_ionicity.csv, copied from the
legacy build; 'verified' rows match a secondary source, 'recalled' rows are unchecked)
matched to the dataset by reduced formula, taking the polymorph with Phillips' structure
(diamond 227, zincblende 216, wurtzite 186, rocksalt 225) where the dataset has it.
Calibrations: on the verified rows (the primary one) and on all rows; leave-one-out RMSE
for every fit; alternative feature sets for comparison.

Usage:  python calibrate_ionicity.py
        -> ../data/phillips_matched.csv, calibration_fits.csv, ionicity_predictions.csv,
           calibration.json
"""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.optimize import least_squares

DATA = Path(__file__).resolve().parents[1] / "data"
SG = {"diamond": 227, "zincblende": 216, "wurtzite": 186, "rocksalt": 225}
DEFAULT = ("fint_over_lnf", "V_spread")
ALTERNATIVES = {"legacy default: f_int/lnf + V_spread": DEFAULT,
                "V_spread": ("V_spread",), "f_int/lnf": ("fint_over_lnf",),
                "rho_min_int_ratio": ("rho_min_int_ratio",),
                "def_polarity_out": ("def_polarity_out",),
                "rho_min_int_ratio + V_spread": ("rho_min_int_ratio", "V_spread"),
                "rho_min_int_ratio + def_polarity_out": ("rho_min_int_ratio", "def_polarity_out"),
                "Pauling ionicity (compositional baseline)": ("pauling_ionicity",)}


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def fit(Z, y):
    return least_squares(lambda b: sigmoid(b[0] + Z @ b[1:]) - y, np.zeros(Z.shape[1] + 1)).x


def calibrate(ref, feats):
    X = ref[list(feats)].to_numpy(float)
    y = ref["f_i"].to_numpy(float)
    mean, std = X.mean(0), X.std(0)
    std[std == 0] = 1
    Z = (X - mean) / std
    b = fit(Z, y)
    pred = sigmoid(b[0] + Z @ b[1:])
    loo = np.array([sigmoid(fit(np.delete(Z, k, 0), np.delete(y, k))[0]
                            + Z[k] @ fit(np.delete(Z, k, 0), np.delete(y, k))[1:]) for k in range(len(y))])
    return {"features": list(feats), "mean": mean.tolist(), "std": std.tolist(), "coef": b.tolist(),
            "n": int(len(y)), "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            "loo_rmse": float(np.sqrt(np.mean((loo - y) ** 2))),
            "spearman_loo": float(pd.Series(loo).rank().corr(pd.Series(y).rank()))}, pred, loo


def predict(cal, df):
    X = df[cal["features"]].to_numpy(float)
    Z = (X - np.array(cal["mean"])) / np.array(cal["std"])
    b = np.array(cal["coef"])
    return sigmoid(b[0] + Z @ b[1:])


def main():
    d = pd.read_csv(DATA / "ionicity_dataset.csv")
    d["red"] = d["formula"].apply(lambda f: Composition(f).reduced_formula)
    rows = [r for r in csv.DictReader(l for l in open(DATA / "phillips_ionicity.csv") if not l.startswith("#"))]
    ref = []
    for r in rows:
        cand = d[d["red"] == Composition(r["formula"]).reduced_formula]
        if cand.empty:
            continue
        want = SG.get(r["structure"])
        same = cand[cand["spacegroup"] == want]
        pick = (same if not same.empty else cand).iloc[0]
        ref.append({"formula": r["formula"], "f_i": float(r["f_i"]), "status": r["status"],
                    "phillips_structure": r["structure"], "id": pick["id"],
                    "structure_matches": bool(not same.empty), **{k: pick[k] for k in
                    ("chem_class", "fint_over_lnf", "V_spread", "rho_min_int_ratio", "def_polarity_out",
                     "pauling_ionicity", "f_int", "lnf")}})
    ref = pd.DataFrame(ref)
    ref.to_csv(DATA / "phillips_matched.csv", index=False)
    print(len(ref), "Phillips compounds in the dataset;", int(ref["structure_matches"].sum()), "in Phillips' structure;",
          int((ref["status"] == "verified").sum()), "verified")

    fits = []
    cals = {}
    for tset, sub in (("verified", ref[ref["status"] == "verified"]), ("all", ref)):
        for name, feats in ALTERNATIVES.items():
            cal, pred, loo = calibrate(sub, feats)
            fits.append({"targets": tset, "features": name, **{k: cal[k] for k in ("n", "rmse", "loo_rmse", "spearman_loo")}})
            if name.startswith("legacy"):
                cals[tset] = cal
                ref.loc[sub.index, f"grid_ionicity_fit_{tset}"] = pred
                ref.loc[sub.index, f"grid_ionicity_loo_{tset}"] = loo
    F = pd.DataFrame(fits)
    F.to_csv(DATA / "calibration_fits.csv", index=False)
    print(F.round(3).to_string())
    ref.to_csv(DATA / "phillips_matched.csv", index=False)
    (DATA / "calibration.json").write_text(json.dumps(cals, indent=1))

    cal = cals["verified"]
    ok = np.isfinite(d[list(cal["features"])]).all(axis=1)
    d["grid_ionicity"] = np.where(ok, predict(cal, d.fillna({"fint_over_lnf": np.nan})), np.nan)
    d["ionicity_residual"] = d["grid_ionicity"] - d["pauling_ionicity"]
    d["grid_ionicity_all_targets"] = predict(cals["all"], d)
    d[["id", "formula", "chem_class", "crystal_system_relaxed", "n_elements", "pauling_ionicity", "grid_ionicity",
       "ionicity_residual", "grid_ionicity_all_targets"]].to_csv(DATA / "ionicity_predictions.csv", index=False)
    print(d[["grid_ionicity", "ionicity_residual"]].describe().round(3))


if __name__ == "__main__":
    main()
