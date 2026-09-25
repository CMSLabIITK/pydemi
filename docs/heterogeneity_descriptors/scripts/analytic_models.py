"""Model crystals with known site structure, for the site-heterogeneity descriptors.

The cell is a 2 x 2 x 2 block of a simple-cubic lattice (spacing 3 A, box 6 A, 0.1 A grid)
with rock-salt ordering: four A sites and four B sites. Every site is a normalized
Gaussian g(r; alpha) = (alpha/pi)^(3/2) exp(-alpha r^2) holding one electron, built by
pydemi's GaussianSuperposition (periodic images included). Descriptors use the default
nearest-atom partition and shells.

A. Size contrast: alpha_A = 2, alpha_B from 0.8 to 5 (B compact for alpha_B > 2). By
   symmetry all A sites are equivalent, and so are all B sites: within = 0 exactly, and
   between = (X_A - X_B)^2 / 4.
B. A point "defect": all sites alpha = 2 except one A site, alpha' from 1 to 4. The four A
   sites are no longer equivalent, and nor are the B sites (three of them touch the
   defect). Within-element variance appears.
C. Positional disorder: alpha_A = 2, alpha_B = 3, every atom displaced by a Gaussian
   random vector of rms sigma per axis (sigma from 0 to 0.25 A, 5 seeds each).

Every row also records Var_i(X) - within - between (the ANOVA identity, which must be 0
to round-off) and the per-site values.

Usage:  python analytic_models.py   -> ../data/analytic_A.csv, analytic_B.csv, analytic_C.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pydemi.descriptors.heterogeneity import _per_site_f_bond, _per_site_m1, _per_site_zeta
from pydemi.descriptors.registry import REGISTRY, Sentinel, make_options
from pydemi.io.base import Lattice, Structure
from pydemi.validate.analytic import GaussianSuperposition

OUT = Path(__file__).resolve().parents[1] / "data"
BOX, N = 6.0, 60
BASES = {"m1": _per_site_m1, "f_bond": _per_site_f_bond, "zeta": _per_site_zeta}
STATS = ["site_std", "within_element_var", "between_element_var"]


def cell(disp=None):
    frac, sp = [], []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                frac.append([i / 2, j / 2, k / 2])
                sp.append("Na" if (i + j + k) % 2 == 0 else "Cl")
    frac = np.array(frac) + 0.013                      # off the grid points
    if disp is not None:
        frac = frac + disp / BOX
    return Structure(Lattice(np.eye(3) * BOX), sp, frac)


def evaluate(s, alphas):
    vd = GaussianSuperposition(s, list(alphas), [1.0] * len(alphas), tol=1e-30).volumetric((N, N, N))
    v = vd.with_options(make_options())
    row = {}
    for b, per in BASES.items():
        x = per(v)
        for st in STATS:
            r = REGISTRY[f"{b}_{st}"].func(v)
            row[f"{b}_{st}"] = r.value if isinstance(r, Sentinel) else float(r)
        row[f"{b}_identity_residual"] = float(np.var(x) - row[f"{b}_within_element_var"]
                                              - row[f"{b}_between_element_var"])
        for i, xi in enumerate(x):
            row[f"{b}@{i}_{s.species[i]}"] = float(xi)
    return row


def main():
    rows = []
    for aB in [0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        s = cell()
        alphas = [2.0 if e == "Na" else aB for e in s.species]
        rows.append({"alpha_B": aB, **evaluate(s, alphas)})
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "analytic_A.csv", index=False)
    print(A[["alpha_B"] + [f"{b}_{st}" for b in BASES for st in STATS] +
             [f"{b}_identity_residual" for b in BASES]].to_string(float_format=lambda x: f"{x:.3e}"))

    rows = []
    for a1 in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0]:
        s = cell()
        alphas = [2.0] * 8
        alphas[0] = a1                                  # site 0 is an A (Na) site
        rows.append({"alpha_defect": a1, **evaluate(s, alphas)})
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "analytic_B.csv", index=False)
    print(B[["alpha_defect"] + [f"{b}_{st}" for b in BASES for st in STATS]].to_string(
        float_format=lambda x: f"{x:.3e}"))

    rows = []
    for sigma in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25]:
        for seed in range(5 if sigma > 0 else 1):
            disp = np.random.default_rng(seed).normal(0.0, sigma, (8, 3))
            s = cell(disp)
            alphas = [2.0 if e == "Na" else 3.0 for e in s.species]
            rows.append({"sigma_A": sigma, "seed": seed, **evaluate(s, alphas)})
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "analytic_C.csv", index=False)
    print(C.groupby("sigma_A")[[f"{b}_{st}" for b in BASES for st in STATS]].mean().to_string(
        float_format=lambda x: f"{x:.3e}"))


if __name__ == "__main__":
    main()
