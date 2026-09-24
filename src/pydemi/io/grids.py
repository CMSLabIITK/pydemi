"""
pydemi.io.grids
---------------
Code-agnostic volumetric formats: Gaussian cube and XCrySDen XSF, plus
writers for both (e.g. to view ELF_D or a deformation density in VESTA).

These cover the other major plane-wave codes through their standard
post-processors:

  Quantum ESPRESSO   pp.x with plot_num = 0 and output_format = 6 (cube)
                     or 5 (XSF)
  ABINIT             cut3d, cube or XSF output

(QE's native charge-density.hdf5 and ABINIT's binary _DEN files are not
read directly: without sample files to validate against, a direct reader
would be untested.)

Units: both formats are read as periodic grids. Density values are taken
to be in e/bohr^3 (cube's convention, and what pp.x and cut3d write) and
converted to pydemi's e/Angstrom^3; pass ``density_unit="e/A^3"`` if the
file already holds that, or ``density_unit=None`` for non-density data
(potentials, ELF), which is returned unscaled. XSF general grids repeat the
first point at the end of every axis; that duplicate is dropped.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from ..constants import BOHR_ANGSTROM, DENSITY_FROM_AU
from ..elements import atomic_number
from ..structure import Structure
from .vasp import ChargeDensity, VolumetricData

_UNITS = {"e/bohr^3": DENSITY_FROM_AU, "e/A^3": 1.0, None: 1.0}


def _symbol(z: int) -> str:
    from pymatgen.core import Element
    return Element.from_Z(int(z)).symbol


def _scale(unit):
    if unit not in _UNITS:
        raise ValueError(f"density_unit must be one of {list(_UNITS)}, got {unit!r}")
    return _UNITS[unit]


# ---------------------------------------------------------------- cube

def read_cube(path, density_unit: Optional[str] = "e/bohr^3") -> VolumetricData:
    lines = Path(path).read_text().splitlines()
    tok = lines[2].split()
    n_atoms = int(tok[0])
    origin = np.array([float(x) for x in tok[1:4]])
    counts, vecs = [], []
    for k in range(3):
        t = lines[3 + k].split()
        n = int(t[0])
        v = np.array([float(x) for x in t[1:4]])
        # positive count: vectors in bohr; negative: Angstrom
        counts.append(abs(n))
        vecs.append(v * (BOHR_ANGSTROM if n > 0 else 1.0))
    to_A = BOHR_ANGSTROM if int(lines[3].split()[0]) > 0 else 1.0
    origin = origin * to_A
    lattice = np.array([vecs[k] * counts[k] for k in range(3)])
    idx = 6
    species, cart = [], []
    for i in range(abs(n_atoms)):
        t = lines[idx + i].split()
        species.append(_symbol(int(float(t[0]))))
        cart.append([float(x) * to_A for x in t[2:5]])
    idx += abs(n_atoms)
    if n_atoms < 0:              # orbital cube: one line of orbital indices
        idx += 1
    values = np.fromstring(" ".join(lines[idx:]), sep=" ")
    n = counts[0] * counts[1] * counts[2]
    if values.size < n:
        raise ValueError(f"{path}: expected {n} grid values, found {values.size}")
    data = values[:n].reshape(counts) * _scale(density_unit)     # z fastest = C order
    frac = (np.array(cart) - origin) @ np.linalg.inv(lattice)
    return VolumetricData(Structure(lattice, species, frac % 1.0), [data], "cube", str(path))


def write_cube(path, structure: Structure, values: np.ndarray,
               density_unit: Optional[str] = "e/bohr^3", comment: str = "written by pydemi"):
    values = np.asarray(values, float)
    counts = values.shape
    to_bohr = 1.0 / BOHR_ANGSTROM
    out = [comment, "periodic grid, values " + (density_unit or "unscaled")]
    out.append(f"{structure.n_atoms:5d} {0.0:12.6f} {0.0:12.6f} {0.0:12.6f}")
    for k in range(3):
        v = structure.lattice[k] / counts[k] * to_bohr
        out.append(f"{counts[k]:5d} {v[0]:12.6f} {v[1]:12.6f} {v[2]:12.6f}")
    for s, x in zip(structure.species, structure.cart_coords * to_bohr):
        z = atomic_number(s)
        out.append(f"{z:5d} {float(z):12.6f} {x[0]:12.6f} {x[1]:12.6f} {x[2]:12.6f}")
    flat = (values / _scale(density_unit)).ravel()
    for i in range(0, flat.size, 6):
        out.append(" ".join(f"{v:13.5E}" for v in flat[i:i + 6]))
    Path(path).write_text("\n".join(out) + "\n")


# ---------------------------------------------------------------- XSF

def read_xsf(path, density_unit: Optional[str] = "e/bohr^3") -> VolumetricData:
    lines = [l.strip() for l in Path(path).read_text().splitlines()]
    lines = [l for l in lines if l and not l.startswith("#")]

    def block_after(key):
        return lines.index(key) + 1

    i = block_after("PRIMVEC")
    lattice = np.array([[float(x) for x in lines[i + k].split()[:3]] for k in range(3)])
    i = block_after("PRIMCOORD")
    n_atoms = int(lines[i].split()[0])
    species, cart = [], []
    for k in range(n_atoms):
        t = lines[i + 1 + k].split()
        species.append(_symbol(int(t[0])) if t[0].isdigit() else t[0])
        cart.append([float(x) for x in t[1:4]])
    start = next(k for k, l in enumerate(lines) if l.startswith("BEGIN_DATAGRID_3D"))
    counts = [int(x) for x in lines[start + 1].split()]
    # lines: origin, spanning vectors a, b, c, then data until END_DATAGRID_3D
    end = next(k for k in range(start, len(lines)) if lines[k].startswith("END_DATAGRID_3D"))
    values = np.fromstring(" ".join(lines[start + 6:end]), sep=" ")
    n = counts[0] * counts[1] * counts[2]
    if values.size < n:
        raise ValueError(f"{path}: expected {n} grid values, found {values.size}")
    grid = values[:n].reshape(counts, order="F")[:-1, :-1, :-1]   # drop periodic duplicate
    frac = np.array(cart) @ np.linalg.inv(lattice)
    return VolumetricData(Structure(lattice, species, frac % 1.0),
                          [grid * _scale(density_unit)], "xsf", str(path))


def write_xsf(path, structure: Structure, values: np.ndarray,
              density_unit: Optional[str] = "e/bohr^3", name: str = "pydemi"):
    values = np.asarray(values, float) / _scale(density_unit)
    # general grid: repeat the first plane at the end of every axis
    periodic = np.pad(values, [(0, 1)] * 3, mode="wrap")
    out = ["CRYSTAL", "PRIMVEC"]
    out += ["  " + " ".join(f"{x:.10f}" for x in row) for row in structure.lattice]
    out += ["PRIMCOORD", f"{structure.n_atoms} 1"]
    out += [f"{atomic_number(s)} " + " ".join(f"{x:.10f}" for x in c)
            for s, c in zip(structure.species, structure.cart_coords)]
    out += ["BEGIN_BLOCK_DATAGRID_3D", name, f"BEGIN_DATAGRID_3D_{name}",
            " ".join(str(n) for n in periodic.shape), "0.0 0.0 0.0"]
    out += ["  " + " ".join(f"{x:.10f}" for x in row) for row in structure.lattice]
    flat = periodic.ravel(order="F")
    out += [" ".join(f"{v:.8E}" for v in flat[i:i + 6]) for i in range(0, flat.size, 6)]
    out += ["END_DATAGRID_3D", "END_BLOCK_DATAGRID_3D"]
    Path(path).write_text("\n".join(out) + "\n")


def read_density(path, density_unit: Optional[str] = "e/bohr^3") -> ChargeDensity:
    """Charge density from a cube or XSF file (by extension)."""
    suffix = Path(path).suffix.lower()
    if suffix in (".cube", ".cub"):
        vol = read_cube(path, density_unit)
    elif suffix == ".xsf":
        vol = read_xsf(path, density_unit)
    else:
        raise ValueError(f"{path}: unknown grid format {suffix!r} (use .cube or .xsf)")
    return ChargeDensity(vol.structure, vol.blocks[0], source=str(path))
