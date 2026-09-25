"""
pydemi.io.cube
==============
Gaussian cube files (spec §3), e.g. from Quantum ESPRESSO ``pp.x``
(``output_format = 6``) or ABINIT ``cut3d``.

Cube files are in bohr and atomic units: lengths are converted to Angstrom
and densities from e/bohr^3 to e/Angstrom^3 on read (a negative voxel count
on an axis line means that axis is already in Angstrom). The grid is taken
to be periodic, the spanning vectors times the counts give the lattice, and
the data are z-fastest (C order).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

import numpy as np

from ..constants import BOHR_ANGSTROM, DENSITY_FROM_AU
from ..data import atomic_number, symbol_of
from .base import FloatArray, Grid, Lattice, Structure

PathLike = Union[str, Path]
Quantity = Literal["density", "raw"]


def read_cube(path: PathLike, quantity: Quantity = "density") -> tuple[Structure, Grid]:
    """(Structure, Grid). ``quantity="density"`` converts e/bohr^3 -> e/Angstrom^3;
    ``"raw"`` returns the values unscaled (potentials, ELF)."""
    lines = Path(path).read_text().splitlines()
    tok = lines[2].split()
    n_atoms = int(tok[0])
    counts: list[int] = []
    vecs: list[FloatArray] = []
    in_bohr = True
    for k in range(3):
        t = lines[3 + k].split()
        n = int(t[0])
        counts.append(abs(n))
        in_bohr = n > 0
        vecs.append(np.array([float(x) for x in t[1:4]]) * (BOHR_ANGSTROM if n > 0 else 1.0))
    to_A = BOHR_ANGSTROM if in_bohr else 1.0
    origin = np.array([float(x) for x in tok[1:4]]) * to_A
    lattice = np.array([vecs[k] * counts[k] for k in range(3)])
    idx = 6
    species, cart = [], []
    for i in range(abs(n_atoms)):
        t = lines[idx + i].split()
        species.append(symbol_of(int(float(t[0]))))
        cart.append([float(x) * to_A for x in t[2:5]])
    idx += abs(n_atoms)
    if n_atoms < 0:                     # orbital cube: one line of orbital indices
        idx += 1
    values = np.fromstring(" ".join(lines[idx:]), sep=" ")
    n = counts[0] * counts[1] * counts[2]
    if values.size < n:
        raise ValueError(f"{path}: expected {n} grid values, found {values.size}")
    data = values[:n].reshape(counts)                              # z fastest = C order
    if quantity == "density":
        data = data * DENSITY_FROM_AU
    lat = Lattice(lattice)
    frac = ((np.array(cart) - origin) @ lat.inverse) % 1.0          # grid origin -> fractional 0
    return Structure(lat, species, frac), Grid(data, lat)


def write_cube(path: PathLike, structure: Structure, data: FloatArray,
               quantity: Quantity = "density", comment: str = "written by pydemi") -> None:
    """Inverse of :func:`read_cube` (bohr, e/bohr^3 for densities)."""
    values = np.asarray(data, dtype=np.float64)
    counts = values.shape
    to_bohr = 1.0 / BOHR_ANGSTROM
    out = [comment, "periodic grid, " + ("e/bohr^3" if quantity == "density" else "raw values")]
    out.append(f"{structure.n_atoms:5d} {0.0:12.6f} {0.0:12.6f} {0.0:12.6f}")
    for k in range(3):
        v = structure.lattice.matrix[k] / counts[k] * to_bohr
        out.append(f"{counts[k]:5d} {v[0]:14.8f} {v[1]:14.8f} {v[2]:14.8f}")
    for s, x in zip(structure.species, structure.cart_coords * to_bohr):
        z = atomic_number(s)
        out.append(f"{z:5d} {float(z):12.6f} {x[0]:14.8f} {x[1]:14.8f} {x[2]:14.8f}")
    flat = (values / DENSITY_FROM_AU if quantity == "density" else values).ravel()
    out += [" ".join(f"{v:15.8E}" for v in flat[i:i + 6]) for i in range(0, flat.size, 6)]
    Path(path).write_text("\n".join(out) + "\n")
