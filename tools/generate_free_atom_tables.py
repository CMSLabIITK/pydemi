"""
Generate ``src/pydemi/data/free_atoms.npz``: spherical free-atom orbital densities
for Z = 1 .. 96 (development-time tool).

Each element is solved with the spherical, non-relativistic, non-spin-polarized
LDA atom solver in ``tools/atomic_solver.py`` (Slater exchange + PW92
correlation; validated against the NIST LDA atomic reference data), using the
ground-state configuration in ``src/pydemi/data/elements.csv``.

Stored per element ``X``:

    X_r         (R,)   radii, Angstrom (log grid, R_MIN .. R_MAX)
    X_orbitals  (K, 4) n, l, occupation, eigenvalue (hartree), sorted by eigenvalue
    X_density   (K, R) density of ONE electron in each orbital, e/Angstrom^3 (float32)

The total free-atom density is sum_k occ_k X_density[k]; the valence part
(ZVAL electrons) takes the highest-energy orbitals first.

Usage:  python tools/generate_free_atom_tables.py [--workers 16]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

from atomic_solver import solve_atom  # noqa: E402
from pydemi.constants import BOHR_ANGSTROM, DENSITY_FROM_AU  # noqa: E402
from pydemi.data import configuration, symbol_of  # noqa: E402

OUT = ROOT / "src" / "pydemi" / "data" / "free_atoms.npz"
Z_MAX = 96
R_MIN, R_MAX, N_R = 1e-5, 30.0, 700


def solve(z: int):
    el = symbol_of(z)
    res = solve_atom(z, configuration(el))
    if not res.converged:
        raise RuntimeError(f"{el}: SCF did not converge (residual {res.extra['residual']:.1e})")
    r_A = res.r * BOHR_ANGSTROM
    grid = np.geomspace(R_MIN, R_MAX, N_R)
    orbitals = np.array([(o[0], o[1], o[2], o[3]) for o in res.orbitals], dtype=np.float64)
    dens = np.array([np.interp(np.log(grid), np.log(r_A), o[4] * DENSITY_FROM_AU, right=0.0)
                     for o in res.orbitals], dtype=np.float32)
    return el, grid, orbitals, dens, res.extra["electrons"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    arrays = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for el, grid, orbitals, dens, n in pool.map(solve, range(1, Z_MAX + 1)):
            arrays[f"{el}_r"] = grid
            arrays[f"{el}_orbitals"] = orbitals
            arrays[f"{el}_density"] = dens
            print(f"{el:>2}: {len(orbitals)} orbitals, {n:.6f} electrons", flush=True)
    arrays["_meta"] = np.array([f"LDA (Slater + PW92), non-relativistic, spherical; "
                                f"Z = 1..{Z_MAX}; r in Angstrom, density in e/Angstrom^3"])
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
