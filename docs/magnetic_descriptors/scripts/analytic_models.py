"""Model magnetization densities with known answers, for the six magnetic descriptors.

Densities are normalized Gaussians g(r; alpha) = (alpha/pi)^(3/2) exp(-alpha r^2), periodic by
minimum image in a cubic box (0.1 A grid). The charge density rho and the magnetization
m = rho_up - rho_down are built separately and passed to pydemi as a collinear run.

A. One magnetic atom (1 mu_B, spin exponent alpha_s) in a 12 A box, charge 8 e with
   alpha_c = 1 / A^2. Exact values: m1_spin = 2 / sqrt(pi alpha_s), <r^2> = 3 / (2 alpha_s),
   f_bond_spin = P(3/2, alpha_s c2^2) - P(3/2, alpha_s c1^2) (regularized lower incomplete
   gamma), and the voxel Pearson correlation of two Gaussians over the box volume V:
     r = (I_cs/V - Q_c Q_s/V^2) / sqrt((I_cc/V - Q_c^2/V^2)(I_ss/V - Q_s^2/V^2)),
   with I_ab = Q_a Q_b (alpha_a alpha_b / (pi (alpha_a + alpha_b)))^(3/2).
A2. The same atom (alpha_s = 3) in boxes of 7 to 16 A: the correlation depends on the vacuum.
B. Two sublattices: the rock-salt 8-site cell of the heterogeneity report (spacing 3 A),
   A sites carrying +1 mu_B and B sites mu_B in [-1, 1] (compact spin Gaussians, alpha = 6).
   Exact: spin_frustration = 1 - |mu_A + mu_B| / (|mu_A| + |mu_B|), mu_site_std = |mu_A - mu_B| / 2.
B2. Ferromagnet (mu_A = mu_B = 1) plus a diffuse negative interstitial spin polarization of
   -p mu_B per cell (a Gaussian of alpha = 1 at the cube centre of each octant): the
   nearest-atom partition hands it to the atoms, so spin_frustration stays 0 until the
   atoms' net moment changes sign, while f_bond_spin and m1_spin shift.

Usage:  python analytic_models.py   -> ../data/analytic_A.csv, analytic_A2.csv, analytic_B.csv,
                                       analytic_B2.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammainc

from pydemi.descriptors.magnetic import site_moments
from pydemi.descriptors.registry import REGISTRY, Sentinel, make_options
from pydemi.io.base import Grid, Lattice, Structure, VolumetricData

OUT = Path(__file__).resolve().parents[1] / "data"
NAMES = ["m1_spin", "sigma_r2_spin", "f_bond_spin", "mu_site_std", "spin_frustration",
         "spin_charge_correlation", "M_abs_per_atom", "M_net_per_atom"]
C1, C2 = 0.8, 1.5


def g(r, a):
    return (a / np.pi) ** 1.5 * np.exp(-a * r * r)


def gaussians(X, box, centres, alphas, charges):
    out = np.zeros(X.shape[:3])
    for c, a, q in zip(centres, alphas, charges):
        d = X - np.asarray(c)
        d -= box * np.rint(d / box)
        out += q * g(np.linalg.norm(d, axis=-1), a)
    return out


def grid(box, n):
    x = np.arange(n) / n * box
    return np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1)


def evaluate(species, frac, box, rho, m):
    s = Structure(Lattice(np.eye(3) * box), species, np.asarray(frac, float))
    vd = VolumetricData(s, Grid(rho, s.lattice), magnetization=Grid(m, s.lattice))
    v = vd.with_options(make_options())
    row = {}
    for n in NAMES:
        r = REGISTRY[n].func(v)
        row[n] = r.value if isinstance(r, Sentinel) else float(r)
    mu = site_moments(v)
    row["sum_mu_minus_M"] = float(mu.sum() - m.sum() * vd.rho.dV)
    for i, x in enumerate(mu):
        row[f"mu@{i}_{species[i]}"] = float(x)
    return row


def pearson_exact(ac, Qc, as_, Qs, V):
    def I(a, b, qa, qb):
        return qa * qb * (a * b / (np.pi * (a + b))) ** 1.5
    cov = I(ac, as_, Qc, Qs) / V - Qc * Qs / V ** 2
    vc = I(ac, ac, Qc, Qc) / V - Qc ** 2 / V ** 2
    vs = I(as_, as_, Qs, Qs) / V - Qs ** 2 / V ** 2
    return cov / np.sqrt(vc * vs)


def main():
    box, n = 12.0, 120
    X = grid(box, n)
    c0 = np.array([6.013, 6.013, 6.013])
    rows = []
    for a_s in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
        rho = gaussians(X, box, [c0], [1.0], [8.0])
        m = gaussians(X, box, [c0], [a_s], [1.0])
        row = {"alpha_s": a_s, **evaluate(["Fe"], [c0 / box], box, rho, m)}
        row.update(m1_spin_exact=2 / np.sqrt(np.pi * a_s),
                   sigma_r2_spin_exact=1.5 / a_s - 4 / (np.pi * a_s),
                   f_bond_spin_exact=gammainc(1.5, a_s * C2 ** 2) - gammainc(1.5, a_s * C1 ** 2),
                   spin_charge_correlation_exact=pearson_exact(1.0, 8.0, a_s, 1.0, box ** 3))
        rows.append(row)
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "analytic_A.csv", index=False)
    print(A[["alpha_s"] + [c for k in ("m1_spin", "sigma_r2_spin", "f_bond_spin", "spin_charge_correlation")
                            for c in (k, f"{k}_exact")]].round(5).to_string())

    rows = []
    for b in [7.0, 8.0, 10.0, 12.0, 14.0, 16.0]:
        nn = int(round(b / 0.1))
        Xb = grid(b, nn)
        cb = np.array([b / 2 + 0.013] * 3)
        rho = gaussians(Xb, b, [cb], [1.0], [8.0])
        m = gaussians(Xb, b, [cb], [3.0], [1.0])
        row = {"box_A": b, **evaluate(["Fe"], [cb / b], b, rho, m)}
        row["spin_charge_correlation_exact"] = pearson_exact(1.0, 8.0, 3.0, 1.0, b ** 3)
        rows.append(row)
    A2 = pd.DataFrame(rows)
    A2.to_csv(OUT / "analytic_A2.csv", index=False)
    print(A2[["box_A", "spin_charge_correlation", "spin_charge_correlation_exact", "m1_spin"]].round(5).to_string())

    box, n = 6.0, 60
    X = grid(box, n)
    frac, sp = [], []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                frac.append([i / 2, j / 2, k / 2])
                sp.append("Fe" if (i + j + k) % 2 == 0 else "Mn")
    frac = np.array(frac) + 0.013
    cart = frac * box
    rho = gaussians(X, box, cart, [1.5] * 8, [8.0] * 8)
    rows = []
    for muB in [1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0]:
        mus = [1.0 if e == "Fe" else muB for e in sp]
        m = gaussians(X, box, cart, [6.0] * 8, mus)
        row = {"mu_B_sublattice": muB, **evaluate(sp, frac, box, rho, m)}
        row["spin_frustration_exact"] = 1 - abs(1 + muB) / (1 + abs(muB))
        row["mu_site_std_exact"] = abs(1 - muB) / 2
        rows.append(row)
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "analytic_B.csv", index=False)
    print(B[["mu_B_sublattice", "spin_frustration", "spin_frustration_exact", "mu_site_std", "mu_site_std_exact",
             "M_net_per_atom", "sum_mu_minus_M"]].round(5).to_string())

    rows = []
    octants = [np.array([0.25, 0.25, 0.25]) * box + np.array([i, j, k]) * box / 2 + 0.013
               for i in range(2) for j in range(2) for k in range(2)]
    for p in [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 7.5, 8.5, 10.0]:
        m = gaussians(X, box, cart, [6.0] * 8, [1.0] * 8) + gaussians(X, box, octants, [1.0] * 8, [-p / 8] * 8)
        row = {"p_interstitial": p, **evaluate(sp, frac, box, rho, m)}
        rows.append(row)
    B2 = pd.DataFrame(rows)
    B2.to_csv(OUT / "analytic_B2.csv", index=False)
    print(B2[["p_interstitial"] + NAMES].round(4).to_string())


if __name__ == "__main__":
    main()
