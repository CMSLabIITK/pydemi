"""
pydemi.io.vasp
==============
The CHGCAR reader (spec §3), pydemi's only input, and a writer for the
same format.

Layout: a POSCAR header, a blank line, the grid dimensions n1 n2 n3, then
the data in Fortran (column-major, fastest index first) order.

* A CHGCAR stores rho * V_cell: every value is divided by the cell volume
  here, so the density comes back in electrons / Angstrom^3.
* Only the first block, the total density, is read. A spin-polarized run
  writes further blocks (the magnetization, after the PAW
  augmentation-occupancy lines); they are not used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from ..io.base import (FloatArray, Grid, Lattice, Structure, VolumetricData, expand_species,
                       species_labels)

PathLike = Union[str, Path]


# ----------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------

def _is_int(token: str) -> bool:
    try:
        int(token)
        return True
    except ValueError:
        return False


def _parse_header(lines: Sequence[str], species: Optional[Sequence[str]]) -> tuple[Structure, int]:
    """POSCAR header -> (Structure, index of the line after the coordinates)."""
    scale_tokens = lines[1].split()
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    if len(scale_tokens) == 3:
        lattice = lattice * np.array([float(x) for x in scale_tokens])[:, None]
        cart_scale = 1.0
    else:
        s = float(scale_tokens[0])
        if s < 0:                                   # negative scale = target volume
            s = (-s / abs(float(np.linalg.det(lattice)))) ** (1.0 / 3.0)
        lattice = lattice * s
        cart_scale = s

    idx = 5
    tokens = lines[idx].split()
    file_species: Optional[list[str]] = None
    if not all(_is_int(t) for t in tokens):
        file_species = tokens
        idx += 1
    counts = [int(x) for x in lines[idx].split()]
    idx += 1

    if species is not None:
        labels = expand_species(species, counts)
    elif file_species is not None:
        labels = species_labels(file_species, counts)
    else:
        raise ValueError("VASP 4 file without a species line: pass species=[...]")

    n_atoms = sum(counts)
    if lines[idx].strip()[:1].lower() == "s":       # selective dynamics
        idx += 1
    cartesian = lines[idx].strip()[:1].lower() in ("c", "k")
    idx += 1
    coords = np.array([[float(x) for x in lines[i].split()[:3]]
                       for i in range(idx, idx + n_atoms)])
    idx += n_atoms
    frac = (coords * cart_scale) @ np.linalg.inv(lattice) if cartesian else coords
    return Structure(Lattice(lattice), labels, frac), idx


def _find_dims(lines: Sequence[str], start: int,
               dims: Optional[tuple[int, int, int]] = None) -> int:
    """Index of the next grid-dimension line at or after ``start`` (-1 if none).

    With ``dims`` given only a line holding exactly those three integers
    counts, which skips augmentation-occupancy and magnetic-moment lines.
    """
    for i in range(start, len(lines)):
        tokens = lines[i].split()
        if len(tokens) != 3 or not all(_is_int(t) for t in tokens):
            continue
        if dims is None or tuple(int(t) for t in tokens) == dims:
            return i
    return -1


def _read_block(lines: Sequence[str], start: int, n_values: int) -> tuple[NDArray[np.float64], int]:
    """``n_values`` floats from line ``start`` on -> (values, index of the next line)."""
    per_line = len(lines[start].split())
    n_lines = -(-n_values // per_line)
    values = np.fromstring(" ".join(lines[start:start + n_lines]), sep=" ")
    if values.size == n_values:
        return values, start + n_lines
    collected, i, count = [], start, 0                  # irregular line lengths
    while count < n_values and i < len(lines):
        row = np.fromstring(lines[i], sep=" ")
        collected.append(row)
        count += row.size
        i += 1
    values = np.concatenate(collected)
    if values.size < n_values:
        raise ValueError(f"data block truncated: expected {n_values} values, got {values.size}")
    return values[:n_values], i


def read_blocks(path: PathLike, scaled_by_volume: bool, species: Optional[Sequence[str]] = None,
                max_blocks: Optional[int] = None) -> tuple[Structure, list[NDArray[np.float64]]]:
    """Every data block of a VASP volumetric file.

    ``scaled_by_volume``: True for a CHGCAR (stored as rho * V).
    """
    lines = Path(path).read_text().splitlines()
    structure, idx = _parse_header(lines, species)
    dim_idx = _find_dims(lines, idx)
    if dim_idx < 0:
        raise ValueError(f"{path}: no grid-dimension line")
    d = [int(t) for t in lines[dim_idx].split()]
    dims = (d[0], d[1], d[2])
    n_values = dims[0] * dims[1] * dims[2]
    norm = structure.volume if scaled_by_volume else 1.0
    blocks: list[NDArray[np.float64]] = []
    while dim_idx >= 0:
        values, end = _read_block(lines, dim_idx + 1, n_values)
        blocks.append(values.reshape(dims, order="F") / norm)
        if max_blocks is not None and len(blocks) >= max_blocks:
            break
        dim_idx = _find_dims(lines, end, dims)
    return structure, blocks


# ----------------------------------------------------------------------
# file readers
# ----------------------------------------------------------------------

def read_chgcar(path: PathLike, species: Optional[Sequence[str]] = None) -> VolumetricData:
    """CHGCAR / CHG: the structure and the total (PAW pseudo-) density.

    ``species`` (one label per element group or per atom) is needed only for
    a VASP 4 file, whose header has no species line.
    """
    structure, blocks = read_blocks(path, True, species, 1)
    return VolumetricData(structure, Grid(blocks[0], structure.lattice), sources={"rho": str(path)})


# ----------------------------------------------------------------------
# writer
# ----------------------------------------------------------------------

def write_volumetric(path: PathLike, structure: Structure, blocks: Sequence[FloatArray],
                     scaled_by_volume: bool = True, per_line: int = 5,
                     comment: str = "written by pydemi") -> None:
    """VASP 5 volumetric file (no augmentation section); inverse of :func:`read_blocks`."""
    elements = structure.elements
    counts = [structure.species.count(e) for e in elements]
    order = np.concatenate([np.flatnonzero(np.array(structure.species) == e) for e in elements])
    out = [comment, "1.0"]
    out += ["  " + " ".join(f"{x:.12f}" for x in row) for row in structure.lattice.matrix]
    out += ["  " + " ".join(elements), "  " + " ".join(str(c) for c in counts), "Direct"]
    out += ["  " + " ".join(f"{x:.12f}" for x in structure.frac_coords[i]) for i in order]
    norm = structure.volume if scaled_by_volume else 1.0
    for block in blocks:
        b = np.asarray(block, dtype=np.float64)
        out += ["", " " + " ".join(f"{n:4d}" for n in b.shape)]
        flat = (b * norm).ravel(order="F")
        out += [" " + " ".join(f"{v:.11E}" for v in flat[i:i + per_line])
                for i in range(0, flat.size, per_line)]
    Path(path).write_text("\n".join(out) + "\n")
