"""Model densities with known answers for V_spread, rho_mid_std and lap_concentration.

Atoms are normalized Gaussians q (alpha/pi)^(3/2) exp(-alpha r^2), periodic.

A. V_spread (electronic Hartree potential at the nuclei, eV). Rock-salt 8-site cell, spacing
   3 A (box 6 A, 60^3 grid), alpha = 2 / A^2; A sites carry 1 + delta electrons and B
   sites 1 - delta. The exact Hartree potential at a site follows from the reciprocal-space
   lattice sum of Gaussian charges,
       V(R) = (4 pi k / V) sum_{G != 0} rho(G) exp(i G.R) / G^2,
       rho(G) = sum_j q_j exp(-G^2 / (4 alpha)) exp(-i G.R_j),   k = 14.3996 eV A,
   which is independent of the real-space grid. V_spread = std over sites.
   A2: the same with A and B of equal charge but different widths (alpha_B from 1 to 4):
   a potential spread without any charge transfer.
B. rho_mid_std. Tetragonal cell a = b = 3 A, one Gaussian atom (Z = 4, alpha = 1); c from
   2.7 to 3.6 A. The bond census counts j as a first-shell neighbour of i when
   |R_ij| <= 1.1 d_i: for c < 3.3 A the c bonds are the shortest and a/b bonds enter only
   if a <= 1.1 c; the midpoint densities are exact lattice sums.
C. lap_concentration on arbitrary periodic densities (random Gaussian superpositions):
   identically 1/2; lap_concentration_valence for one Gaussian atom against the exact
   radial integral outside r = 0.8 A.

Usage:  python analytic_models.py   -> ../data/analytic_A.csv, analytic_A2.csv, analytic_B.csv,
                                       analytic_C.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate

from pydemi.descriptors.bonding import site_potentials
from pydemi.descriptors.registry import REGISTRY, Sentinel, make_options
from pydemi.io.base import Grid, Lattice, Structure, VolumetricData
from pydemi.validate.analytic import GaussianSuperposition

OUT = Path(__file__).resolve().parents[1] / "data"
K = 14.399645


def val(r):
    return r.value if isinstance(r, Sentinel) else float(r)


def rocksalt():
    frac, sp = [], []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                frac.append([i / 2, j / 2, k / 2])
                sp.append("Na" if (i + j + k) % 2 == 0 else "Cl")
    return Structure(Lattice(np.eye(3) * 6.0), sp, np.array(frac) + 0.013)


def exact_site_potentials(s, alphas, charges, gmax=12):
    lat = s.lattice.matrix
    B = 2 * np.pi * np.linalg.inv(lat).T
    V = abs(np.linalg.det(lat))
    n = np.arange(-gmax, gmax + 1)
    N = np.stack(np.meshgrid(n, n, n, indexing="ij"), -1).reshape(-1, 3)
    N = N[np.any(N != 0, axis=1)]
    G = N @ B
    G2 = np.sum(G * G, 1)
    R = s.frac_coords @ lat
    phase = np.exp(-1j * G @ R.T)                       # (nG, nat)
    rhoG = (phase * (np.array(charges)[None, :] * np.exp(-G2[:, None] / (4 * np.array(alphas)[None, :])))).sum(1)
    return np.real((4 * np.pi * K / V) * (rhoG[:, None] * np.conj(phase) / G2[:, None]).sum(0))


def main():
    s = rocksalt()
    rows = []
    for delta in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        q = [1 + delta if e == "Na" else 1 - delta for e in s.species]
        al = [2.0] * 8
        vd = GaussianSuperposition(s, al, q, tol=1e-30).volumetric((60, 60, 60))
        v = vd.with_options(make_options(potential_source="hartree"))
        Vs = site_potentials(v)
        Ve = exact_site_potentials(s, al, q)
        rows.append({"delta": delta, "V_spread": val(REGISTRY["V_spread"].func(v)),
                     "V_spread_exact": float(np.std(Ve)), "V_A": float(Vs[0]), "V_A_exact": float(Ve[0]),
                     "V_B": float(Vs[1]), "V_B_exact": float(Ve[1])})
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "analytic_A.csv", index=False)
    print(A.round(5).to_string())

    rows = []
    for aB in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        al = [2.0 if e == "Na" else aB for e in s.species]
        vd = GaussianSuperposition(s, al, [1.0] * 8, tol=1e-30).volumetric((60, 60, 60))
        v = vd.with_options(make_options(potential_source="hartree"))
        Ve = exact_site_potentials(s, al, [1.0] * 8)
        rows.append({"alpha_B": aB, "V_spread": val(REGISTRY["V_spread"].func(v)),
                     "V_spread_exact": float(np.std(Ve))})
    A2 = pd.DataFrame(rows)
    A2.to_csv(OUT / "analytic_A2.csv", index=False)
    print(A2.round(5).to_string())

    rows = []
    for c in [2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6]:
        st = Structure(Lattice(np.diag([3.0, 3.0, c])), ["X"], [[0.013, 0.013, 0.013]])
        vd = GaussianSuperposition(st, [1.0], [4.0], tol=1e-30).volumetric((60, 60, int(round(c / 0.05))))
        v = vd.with_options(make_options())
        g = lambda p: sum(4.0 * (1 / np.pi) ** 1.5 * np.exp(-np.sum((np.array(p) - np.array([i * 3.0, j * 3.0, k * c])) ** 2))
                          for i in range(-4, 5) for j in range(-4, 5) for k in range(-4, 5))
        ma, mc = g([1.5, 0, 0]), g([0, 0, c / 2])
        d = min(3.0, c)
        mids = ([ma] * 4 if 3.0 <= 1.1 * d else []) + ([mc] * 2 if c <= 1.1 * d else [])
        rows.append({"c": c, "rho_mid_mean": val(REGISTRY["rho_mid_mean"].func(v)),
                     "rho_mid_std": val(REGISTRY["rho_mid_std"].func(v)),
                     "rho_mid_std_exact": float(np.std(mids)), "rho_mid_mean_exact": float(np.mean(mids)),
                     "n_bond_types": int((3.0 <= 1.1 * d) + (c <= 1.1 * d))})
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "analytic_B.csv", index=False)
    print(B.round(5).to_string())

    rows = []
    rng = np.random.default_rng(0)
    for trial in range(8):
        n = int(rng.integers(1, 6))
        box = float(rng.uniform(4, 8))
        st = Structure(Lattice(np.eye(3) * box), ["X"] * n, rng.random((n, 3)))
        al = rng.uniform(0.5, 4.0, n)
        vd = GaussianSuperposition(st, list(al), list(rng.uniform(1, 8, n)), tol=1e-30).volumetric((48, 48, 48))
        v = vd.with_options(make_options())
        rows.append({"case": f"random {trial}", "n_atoms": n, "lap_concentration": val(REGISTRY["lap_concentration"].func(v)),
                     "lap_concentration_valence": val(REGISTRY["lap_concentration_valence"].func(v))})
    for a in [0.5, 1.0, 2.0, 4.0]:
        st = Structure(Lattice(np.eye(3) * 12.0), ["X"], [[0.5011, 0.5011, 0.5011]])
        vd = GaussianSuperposition(st, [a], [1.0], tol=1e-30).volumetric((120, 120, 120))
        v = vd.with_options(make_options())
        # lap g = g (4 a^2 r^2 - 6 a); negative for r < sqrt(3/(2a))
        lap = lambda r: (a / np.pi) ** 1.5 * np.exp(-a * r * r) * (4 * a * a * r * r - 6 * a) * 4 * np.pi * r * r
        r0 = np.sqrt(1.5 / a)
        lo, hi = 0.8, 12.0
        if r0 > lo:
            neg = -integrate.quad(lap, lo, r0)[0]
            pos = integrate.quad(lap, r0, hi)[0]
        else:
            neg, pos = 0.0, integrate.quad(lap, lo, hi)[0]
        rows.append({"case": f"one Gaussian, alpha = {a}", "n_atoms": 1,
                     "lap_concentration": val(REGISTRY["lap_concentration"].func(v)),
                     "lap_concentration_valence": val(REGISTRY["lap_concentration_valence"].func(v)),
                     "lap_concentration_valence_exact": neg / (neg + pos)})
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "analytic_C.csv", index=False)
    print(C.round(6).to_string())


if __name__ == "__main__":
    main()
