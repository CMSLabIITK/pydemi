"""Analytic model densities for the four anisotropy descriptors.

A. One anisotropic Gaussian, rho = exp(-(a x^2 + a y^2 + c z^2)), for c/a from 1/4 to 4.
   Exact anisotropy tensor: T = diag(a, a, c) / (2a + c) (the gradient second moment
   int (d_i rho)^2 dV of a Gaussian is a_i times int rho^2 dV), hence an exact charge_FA.
B. Two spherical Gaussians (alpha = 2 / A^2) at separation d along x: zeta and charge_FA
   as the atoms merge.
C. A single spherical atom: zeta = 0 and T = I/3 exactly.
D. The zeta_ELF of a spherical Gaussian (exact value 0) against density scale and grid.

Usage:  python analytic_examples.py
        -> ../data/analytic_A.csv, analytic_B.csv, analytic_C.csv, analytic_zetaELF_baseline.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pydemi.descriptors.registry import REGISTRY, make_options
from pydemi.io.base import Grid, Lattice, Structure, VolumetricData
from pydemi.validate.analytic import GaussianSuperposition

OUT = Path(__file__).resolve().parents[1] / "data"
NAMES = ["zeta", "zeta_ELF", "T_eigenvalues_t1", "T_eigenvalues_t2", "T_eigenvalues_t3", "charge_FA"]


def evaluate(vd):
    v = vd.with_options(make_options())
    out = {}
    for n in NAMES:
        r = REGISTRY[n].func(v)
        out[n] = float(getattr(r, "value", r))
    return out


def exact_fa(t):
    t = np.asarray(t, float)
    return float(np.sqrt(1.5) * np.linalg.norm(t - 1 / 3) / np.linalg.norm(t))


def anisotropic(a, c, box=16.0, n=112):
    s = Structure(Lattice(np.eye(3) * box), ["H"], [[0.5, 0.5, 0.5]])
    x = (np.arange(n) / n - 0.5) * box
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    rho = np.exp(-(a * X ** 2 + a * Y ** 2 + c * Z ** 2))
    return VolumetricData(s, Grid(rho, s.lattice))


def main():
    rows = []
    for ratio in [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]:
        a, c = 1.0, ratio
        f = evaluate(anisotropic(a, c))
        t = np.sort(np.array([a, a, c]) / (2 * a + c))
        rows.append({"c_over_a": ratio, **f, "t1_exact": t[0], "t2_exact": t[1], "t3_exact": t[2],
                     "charge_FA_exact": exact_fa(t)})
    pd.DataFrame(rows).to_csv(OUT / "analytic_A.csv", index=False)
    print(pd.DataFrame(rows)[["c_over_a", "zeta", "charge_FA", "charge_FA_exact",
                              "T_eigenvalues_t1", "t1_exact"]].round(5).to_string())

    rows = []
    box, n = 12.0, 96
    for d in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]:
        s = Structure(Lattice(np.eye(3) * box), ["H", "H"],
                      [[0.5 - d / (2 * box), 0.5, 0.5], [0.5 + d / (2 * box), 0.5, 0.5]])
        vd = GaussianSuperposition(s, [2.0, 2.0], [1.0, 1.0], tol=1e-30).volumetric((n, n, n))
        rows.append({"separation_A": d, **evaluate(vd)})
    pd.DataFrame(rows).to_csv(OUT / "analytic_B.csv", index=False)
    print(pd.DataFrame(rows)[["separation_A", "zeta", "zeta_ELF", "charge_FA",
                              "T_eigenvalues_t1", "T_eigenvalues_t3"]].round(5).to_string())

    s = Structure(Lattice(np.eye(3) * box), ["H"], [[0.5, 0.5, 0.5]])
    vd = GaussianSuperposition(s, [2.0], [1.0], tol=1e-30).volumetric((n, n, n))
    f = evaluate(vd)
    pd.DataFrame([f]).to_csv(OUT / "analytic_C.csv", index=False)
    print("single spherical atom:", {k: f"{v:.2e}" for k, v in f.items()})

    rows = []
    for alpha in (1.0, 2.0):
        for peak in (0.1, 0.5, 1.0, 5.0):
            for n in (72, 96, 128):
                s = Structure(Lattice(np.eye(3) * box), ["H"], [[0.5, 0.5, 0.5]])
                x = (np.arange(n) / n - 0.5) * box
                X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
                vd = VolumetricData(s, Grid(peak * np.exp(-alpha * (X ** 2 + Y ** 2 + Z ** 2)), s.lattice))
                rows.append({"alpha": alpha, "peak_e_per_A3": peak, "n": n,
                             "zeta_ELF": evaluate(vd)["zeta_ELF"]})
    pd.DataFrame(rows).to_csv(OUT / "analytic_zetaELF_baseline.csv", index=False)
    print(pd.DataFrame(rows).pivot_table(index=["alpha", "peak_e_per_A3"], columns="n",
                                         values="zeta_ELF").round(4).to_string())


if __name__ == "__main__":
    main()
