"""Model densities with known percolation levels and density floors.

Every atom is a normalized Gaussian g(r; alpha) = (alpha/pi)^(3/2) exp(-alpha r^2) holding
Z electrons, summed over periodic images (lattice sums converged to 1e-14 relative).

A. Simple cubic lattice, spacing a (the specification's test case), alpha = 1 / A^2,
   Z = 4, grid spacing a/40 with the atom on a grid point. Along each axis the best path
   runs through the atoms, and its bottleneck is the bond midpoint (a/2, 0, 0), an index-1
   saddle; so rho_perc = rho(a/2, 0, 0) exactly, the same along a, b, c. The density
   minimum is the cube centre (a/2, a/2, a/2); rho_min_ratio = rho_min / (Z / a^3).
   a from 2 to 5 A.
B. Tetragonal distortion: a = b = 3 A, c from 2.4 to 6 A. rho_perc_c = rho(0, 0, c/2),
   rho_perc_a = rho(a/2, 0, 0), perc_anisotropy = (max - min) / mean, exact.
C. A uniform background b added to lattice A (a = 4 A): the levels shift by b exactly,
   rho_min_ratio -> 1 as b grows ("jellium" limit), perc_anisotropy of B -> 0.
D. Grid convergence: lattice A (a = 3.5 A) with the atom OFF the grid points, grid
   spacing 0.05-0.2 A: the grid misses the exact saddle, so rho_perc is biased.

Usage:  python analytic_models.py   -> ../data/analytic_A.csv, analytic_B.csv, analytic_C.csv,
                                       analytic_D.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pydemi.descriptors.registry import REGISTRY, Sentinel, make_options
from pydemi.io.base import Grid, Lattice, Structure, VolumetricData

OUT = Path(__file__).resolve().parents[1] / "data"
NAMES = ["rho_perc_a", "rho_perc_b", "rho_perc_c", "perc_anisotropy", "rho_min_ratio", "rho_min"]
ALPHA, Z = 1.0, 4.0


def lattice_density(points, cell, alpha=ALPHA, z=Z, reps=4):
    """rho at Cartesian points for one atom per orthorhombic cell (diag lengths ``cell``)."""
    pts = np.atleast_2d(points)
    out = np.zeros(len(pts))
    for i in range(-reps, reps + 1):
        for j in range(-reps, reps + 1):
            for k in range(-reps, reps + 1):
                d = pts - np.array([i, j, k]) * cell
                out += z * (alpha / np.pi) ** 1.5 * np.exp(-alpha * np.sum(d * d, axis=1))
    return out


def grid_density(cell, shape, origin, alpha=ALPHA, z=Z, background=0.0):
    fr = [np.arange(n) / n for n in shape]
    F = np.stack(np.meshgrid(*fr, indexing="ij"), axis=-1)
    X = F * cell - origin
    X -= cell * np.rint(X / cell)                     # minimum image, then add neighbours
    flat = X.reshape(-1, 3)
    return lattice_density(flat, cell, alpha, z).reshape(shape) + background


def evaluate(cell, rho):
    s = Structure(Lattice(np.diag(cell)), ["X"], [[0.0, 0.0, 0.0]])
    v = VolumetricData(s, Grid(rho, s.lattice)).with_options(make_options(domains=("structural",)))
    out = {}
    for n in NAMES:
        r = REGISTRY[n].func(v)
        out[n] = r.value if isinstance(r, Sentinel) else float(r)
    return out


def main():
    rows = []
    for a in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        cell = np.array([a, a, a])
        rho = grid_density(cell, (40, 40, 40), np.zeros(3))
        mid = lattice_density([[a / 2, 0, 0]], cell)[0]
        cen = lattice_density([[a / 2, a / 2, a / 2]], cell)[0]
        rows.append({"a": a, **evaluate(cell, rho), "rho_perc_exact": mid, "rho_min_exact": cen,
                     "rho_min_ratio_exact": cen / (Z / a ** 3)})
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "analytic_A.csv", index=False)
    print(A.round(6).to_string())

    rows = []
    for c in [2.4, 3.0, 3.6, 4.2, 4.8, 5.4, 6.0]:
        cell = np.array([3.0, 3.0, c])
        rho = grid_density(cell, (36, 36, int(round(c / (3.0 / 36)))), np.zeros(3))
        pa = lattice_density([[1.5, 0, 0]], cell)[0]
        pc = lattice_density([[0, 0, c / 2]], cell)[0]
        p = np.array([pa, pa, pc])
        rows.append({"c": c, **evaluate(cell, rho), "rho_perc_a_exact": pa, "rho_perc_c_exact": pc,
                     "perc_anisotropy_exact": (p.max() - p.min()) / p.mean()})
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "analytic_B.csv", index=False)
    print(B.round(6).to_string())

    rows = []
    for b in [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        cell = np.array([4.0, 4.0, 4.0])
        rho = grid_density(cell, (40, 40, 40), np.zeros(3), background=b)
        cellB = np.array([3.0, 3.0, 5.4])
        rhoB = grid_density(cellB, (36, 36, 65), np.zeros(3), background=b)
        e = evaluate(cell, rho)
        eB = evaluate(cellB, rhoB)
        rows.append({"background": b, **e, "perc_anisotropy_tetragonal": eB["perc_anisotropy"],
                     "rho_perc_exact": lattice_density([[2, 0, 0]], cell)[0] + b})
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "analytic_C.csv", index=False)
    print(C.round(6).to_string())

    rows = []
    a = 3.5
    cell = np.array([a, a, a])
    exact = lattice_density([[a / 2, 0, 0]], cell)[0]
    for h in [0.2, 0.175, 0.15, 0.125, 0.1, 0.0875, 0.07, 0.05]:
        n = int(round(a / h))
        rho = grid_density(cell, (n, n, n), np.array([0.37, 0.21, 0.13]) * (a / n))
        e = evaluate(cell, rho)
        rows.append({"spacing_A": a / n, "n": n, **e, "rho_perc_exact": exact})
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "analytic_D.csv", index=False)
    print(D.round(6).to_string())


if __name__ == "__main__":
    main()
