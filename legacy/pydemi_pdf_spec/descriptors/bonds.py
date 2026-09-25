"""
pydemi.descriptors.bonds
------------------------
I2 -- bond-strength descriptors (reference entries 104-105).

Bond census: atom j (any periodic image, including images of i itself in
small cells) is a nearest-neighbour bond partner of i when
|R_j - R_i| <= (1 + tol) d_i, with d_i the distance from i to its closest
neighbour. This per-atom first shell is the explicit convention the
reference asks for in disordered multi-element cells, where "bond length"
is not single-valued; every pair is counted once.

    rho_mid_mean  <rho(midpoint)> over the census   (104), e/Angstrom^3
    rho_mid_std   std over the census                (105)
    n_bonds, bond_length_mean                        census metadata

Midpoint densities use exact trigonometric interpolation of the grid.
The midpoint stands in for the bond critical point; for heteronuclear
bonds the true critical point is shifted toward the smaller atom.
``bond_length_mean`` is the <d> the Cohen and rho-based bulk-modulus
baselines (61, 106; Phase 6) need.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..engine import RHO, Engine
from ..geometry import _image_points
from ..structure import Structure
from .potential import fourier_interpolate

BOND_TOL = 0.1


@dataclass(frozen=True, eq=False)
class BondCensus:
    i: np.ndarray          # (n_bonds,) atom index
    j: np.ndarray          # (n_bonds,) partner atom index
    shift: np.ndarray      # (n_bonds, 3) lattice translation of j's image
    length: np.ndarray     # (n_bonds,) Angstrom

    def midpoints_frac(self, structure: Structure) -> np.ndarray:
        f = structure.frac_coords
        return 0.5 * (f[self.i] + f[self.j] + self.shift)


def bond_census(structure: Structure, tol: float = BOND_TOL) -> BondCensus:
    n = structure.n_atoms
    frac = structure.frac_coords
    # images of every atom around the cell; widen until the first shell of
    # every atom is covered (the shell radius is bounded by the lattice)
    reps = 1
    while True:
        pts, owner = _image_points(structure, (reps,) * 3)
        shifts = np.floor(pts @ structure.inv_lattice - (frac % 1.0)[owner] + 0.5)
        tree = cKDTree(pts)
        centres = (frac % 1.0) @ structure.lattice
        d, _ = tree.query(centres, k=2)
        d_min = d[:, 1]
        radius = (1.0 + tol) * d_min.max()
        h = np.linalg.norm(structure.inv_lattice, axis=0)
        if all(int(np.ceil(radius * hj)) + 1 <= reps for hj in h):
            break
        reps += 1

    pairs = {}
    for i in range(n):
        for k in tree.query_ball_point(centres[i], (1.0 + tol) * d_min[i] + 1e-9):
            j, t = int(owner[k]), shifts[k].astype(int)
            length = float(np.linalg.norm(pts[k] - centres[i]))
            if length < 1e-9:
                continue                               # the atom itself
            # canonical key: (i, j, t) and (j, i, -t) are the same bond
            key = (i, j, tuple(t)) if (i, j, tuple(t)) <= (j, i, tuple(-t)) else (j, i, tuple(-t))
            pairs[key] = length
    if not pairs:
        empty = np.zeros(0, int)
        return BondCensus(empty, empty, np.zeros((0, 3), int), np.zeros(0))
    keys = sorted(pairs)
    # shifts above are relative to the wrapped (frac % 1) positions; convert
    # to the stored coordinates so midpoints use structure.frac_coords
    wrap = np.floor(frac)
    i = np.array([k[0] for k in keys])
    j = np.array([k[1] for k in keys])
    t = np.array([k[2] for k in keys]) - wrap[j] + wrap[i]
    return BondCensus(i, j, t, np.array([pairs[k] for k in keys]))


def bond_family(engine: Engine, field: str = RHO, tol: float = BOND_TOL) -> dict:
    census = bond_census(engine.structure, tol)
    if census.length.size == 0:
        nan = float("nan")
        return {"rho_mid_mean": nan, "rho_mid_std": nan, "n_bonds": 0,
                "bond_length_mean": nan}
    rho_mid = fourier_interpolate(engine[field].values, census.midpoints_frac(engine.structure))
    return {
        "rho_mid_mean": float(rho_mid.mean()),
        "rho_mid_std": float(rho_mid.std()),
        "n_bonds": int(census.length.size),
        "bond_length_mean": float(census.length.mean()),
    }
