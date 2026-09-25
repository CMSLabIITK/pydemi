"""Model densities with a known deformation density, for the seven deformation descriptors.

Every "atom" is a normalized Gaussian g(r; alpha) = (alpha/pi)^(3/2) exp(-alpha r^2) holding
one electron, and the promolecule is built by pydemi itself from a custom reference
directory holding exactly those radial densities (deformation_reference="custom"). So
delta_rho is known in closed form and the descriptors can be checked.

A. Breathing atom. One atom, crystal density g(r; s*alpha0), reference g(r; alpha0),
   alpha0 = 2 / A^2, for s from 0.4 to 2.5 (s > 1: contraction). The exact values come
   from 1-D radial quadrature of delta(r) = g(r; s alpha0) - g(r; alpha0).
B. Bond charge. Two atoms at separation d; the crystal takes q electrons from the two
   atoms (q/2 each) and puts them in a Gaussian (alpha_b = 3 / A^2) at the bond midpoint.
   B1: q = 0.2, d from 1.0 to 4.4 A (the midpoint crosses the 0.8 and 1.5 A shells).
   B2: d = 2.4 A, q from 0 to 0.6.
C. Charge transfer. A cation-like atom (alpha = 2) gives q electrons to an anion-like atom
   (alpha = 1) 2.2 A away; the extra charge sits in a more diffuse Gaussian (alpha = 0.6).
   q from 0 to 0.8.

Usage:  python analytic_models.py
        -> ../data/analytic_A.csv, analytic_B1.csv, analytic_B2.csv, analytic_C.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pydemi.descriptors.registry import REGISTRY, Sentinel, make_options
from pydemi.io.base import Grid, Lattice, Structure, VolumetricData

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data"
REF = OUT / "analytic_reference"           # custom reference directory (radial files)
NAMES = ["m1_def", "m2_def", "sigma_r2_def", "f_bond_def", "f_int_def", "f_bond_dep", "def_polarity"]
BOX, N = 12.0, 120                          # 0.1 A grid spacing
C1, C2 = 0.8, 1.5                           # the default shells (Angstrom)
ALPHA = {"H": 2.0, "Li": 2.0, "F": 1.0}     # reference exponent per label


def g(r, a):
    return (a / np.pi) ** 1.5 * np.exp(-a * r * r)


def write_reference():
    REF.mkdir(parents=True, exist_ok=True)
    r = np.linspace(0.0, 8.0, 16001)
    for el, a in ALPHA.items():
        np.savetxt(REF / f"{el}.dat", np.column_stack([r, g(r, a)]), header="r [A]  rho [e/A^3]")


def grid_points():
    x = (np.arange(N) / N) * BOX
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1)
    return X


def gaussians(X, centres, alphas, charges):
    """Periodic sum by minimum image (the box is large enough for every exponent used)."""
    rho = np.zeros(X.shape[:3])
    for c, a, q in zip(centres, alphas, charges):
        d = X - np.asarray(c)
        d -= BOX * np.rint(d / BOX)
        rho += q * g(np.linalg.norm(d, axis=-1), a)
    return rho


def evaluate(species, frac, rho):
    s = Structure(Lattice(np.eye(3) * BOX), species, np.asarray(frac, float))
    vd = VolumetricData(s, Grid(rho, s.lattice))
    v = vd.with_options(make_options(deformation_reference="custom", custom_reference=str(REF)))
    out = {}
    for n in NAMES:
        r = REGISTRY[n].func(v)
        out[n] = r.value if isinstance(r, Sentinel) else float(r)
    return out


def exact_breathing(s, a0=2.0):
    """The seven descriptors of delta(r) = g(r; s a0) - g(r; a0) by radial quadrature."""
    r = np.linspace(0.0, 8.0, 200001)
    d = g(r, s * a0) - g(r, a0)
    w = 4 * np.pi * r * r
    a = np.abs(d)
    tr = np.trapezoid
    pos, neg = np.clip(d, 0, None), np.clip(-d, 0, None)
    bond = (r > C1) & (r <= C2)
    m1 = tr(w * a * r, r) / tr(w * a, r)
    m2 = tr(w * a * r * r, r) / tr(w * a, r)
    return {"m1_def": m1, "m2_def": m2, "sigma_r2_def": m2 - m1 * m1,
            "f_bond_def": tr(w * pos * bond, r) / tr(w * pos, r),
            "f_int_def": tr(w * pos * (r > C2), r) / tr(w * pos, r),
            "f_bond_dep": tr(w * neg * bond, r) / tr(w * neg, r),
            "def_polarity": tr(w * a, r)}


def main():
    write_reference()
    X = grid_points()
    c0 = np.array([0.5, 0.5, 0.5]) * BOX + 0.0123       # off the grid points (no cusp sampling)
    f0 = c0 / BOX

    rows = []
    for s in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5]:
        rho = gaussians(X, [c0], [2.0 * s], [1.0])
        f = evaluate(["H"], [f0], rho)
        ex = exact_breathing(s)
        rows.append({"s": s, **f, **{f"{k}_exact": v for k, v in ex.items()}})
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "analytic_A.csv", index=False)
    print(A[["s"] + [c for n in NAMES for c in (n, f"{n}_exact")]].round(4).to_string())

    def bond_model(d, q, ab=3.0):
        ca = c0 - np.array([d / 2, 0, 0])
        cb = c0 + np.array([d / 2, 0, 0])
        rho = gaussians(X, [ca, cb, c0], [2.0, 2.0, ab], [1 - q / 2, 1 - q / 2, q])
        return evaluate(["H", "H"], [ca / BOX, cb / BOX], rho)

    B1 = pd.DataFrame([{"d_A": d, "q": 0.2, **bond_model(d, 0.2)}
                       for d in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4,
                                 3.6, 3.8, 4.0, 4.4]])
    B1.to_csv(OUT / "analytic_B1.csv", index=False)
    print(B1.round(4).to_string())
    B2 = pd.DataFrame([{"d_A": 2.4, "q": q, **bond_model(2.4, q)}
                       for q in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]])
    B2.to_csv(OUT / "analytic_B2.csv", index=False)
    print(B2.round(4).to_string())

    rows = []
    for q in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
        ca = c0 - np.array([1.1, 0, 0])
        cb = c0 + np.array([1.1, 0, 0])
        rho = gaussians(X, [ca, cb, cb], [2.0, 1.0, 0.6], [1 - q, 1.0, q])
        rows.append({"q": q, **evaluate(["Li", "F"], [ca / BOX, cb / BOX], rho)})
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "analytic_C.csv", index=False)
    print(C.round(4).to_string())


if __name__ == "__main__":
    main()
