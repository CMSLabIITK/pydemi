"""
pydemi.io.xsf
=============
XCrySDen XSF periodic data grids (spec §3), e.g. from Quantum ESPRESSO
``pp.x`` (``output_format = 5``) or ABINIT ``cut3d``.

XSF lengths are in Angstrom. Density values are taken to be in e/bohr^3
(what pp.x and cut3d write) and converted to e/Angstrom^3; pass
``quantity="raw"`` for other data. General grids repeat the first point at
the end of every axis; that duplicate is dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

import numpy as np

from ..constants import DENSITY_FROM_AU
from ..data import atomic_number, symbol_of
from .base import FloatArray, Grid, Lattice, Structure

PathLike = Union[str, Path]
Quantity = Literal["density", "raw"]


def read_xsf(path: PathLike, quantity: Quantity = "density") -> tuple[Structure, Grid]:
    lines = [l.strip() for l in Path(path).read_text().splitlines()]
    lines = [l for l in lines if l and not l.startswith("#")]
    i = lines.index("PRIMVEC") + 1
    lattice = np.array([[float(x) for x in lines[i + k].split()[:3]] for k in range(3)])
    i = lines.index("PRIMCOORD") + 1
    n_atoms = int(lines[i].split()[0])
    species, cart = [], []
    for k in range(n_atoms):
        t = lines[i + 1 + k].split()
        species.append(symbol_of(int(t[0])) if t[0].isdigit() else t[0])
        cart.append([float(x) for x in t[1:4]])
    start = next(k for k, l in enumerate(lines) if l.startswith("BEGIN_DATAGRID_3D"))
    counts = [int(x) for x in lines[start + 1].split()]
    origin = np.array([float(x) for x in lines[start + 2].split()[:3]])
    end = next(k for k in range(start, len(lines)) if lines[k].startswith("END_DATAGRID_3D"))
    values = np.fromstring(" ".join(lines[start + 6:end]), sep=" ")
    n = counts[0] * counts[1] * counts[2]
    if values.size < n:
        raise ValueError(f"{path}: expected {n} grid values, found {values.size}")
    data = values[:n].reshape(counts, order="F")[:-1, :-1, :-1]     # drop the periodic duplicate
    if quantity == "density":
        data = data * DENSITY_FROM_AU
    lat = Lattice(lattice)
    frac = ((np.array(cart) - origin) @ lat.inverse) % 1.0
    return Structure(lat, species, frac), Grid(np.ascontiguousarray(data), lat)


def write_xsf(path: PathLike, structure: Structure, data: FloatArray,
              quantity: Quantity = "density", name: str = "pydemi") -> None:
    values = np.asarray(data, dtype=np.float64)
    if quantity == "density":
        values = values / DENSITY_FROM_AU
    periodic = np.pad(values, [(0, 1)] * 3, mode="wrap")
    out = ["CRYSTAL", "PRIMVEC"]
    out += ["  " + " ".join(f"{x:.12f}" for x in row) for row in structure.lattice.matrix]
    out += ["PRIMCOORD", f"{structure.n_atoms} 1"]
    out += [f"{atomic_number(s)} " + " ".join(f"{x:.12f}" for x in c)
            for s, c in zip(structure.species, structure.cart_coords)]
    out += ["BEGIN_BLOCK_DATAGRID_3D", name, f"BEGIN_DATAGRID_3D_{name}",
            " ".join(str(n) for n in periodic.shape), "0.0 0.0 0.0"]
    out += ["  " + " ".join(f"{x:.12f}" for x in row) for row in structure.lattice.matrix]
    flat = periodic.ravel(order="F")
    out += [" ".join(f"{v:.10E}" for v in flat[i:i + 6]) for i in range(0, flat.size, 6)]
    out += ["END_DATAGRID_3D", "END_BLOCK_DATAGRID_3D"]
    Path(path).write_text("\n".join(out) + "\n")
