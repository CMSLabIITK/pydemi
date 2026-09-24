"""
pydemi.io.vasp
==============
Readers for VASP volumetric files (spec §3): CHGCAR / CHG, AECCAR0 / AECCAR2,
ELFCAR, LOCPOT, plus ZVAL and RCORE from a POTCAR or OUTCAR.

Layout shared by all of them: a POSCAR header, a blank line, the grid
dimensions n1 n2 n3, then the data in Fortran (column-major, fastest index
first) order.

* CHGCAR and AECCAR store rho * V_cell: every value is divided by the cell
  volume here, so densities come back in electrons / Angstrom^3.
* A spin-polarized CHGCAR has a second block, m = rho_up - rho_down, with the
  same shape and scaling. PAW augmentation-occupancy lines may sit between
  the blocks: the reader skips lines until the next grid-dimension line
  equal to the first.
* A non-collinear run writes four blocks (rho, m_x, m_y, m_z), detected by
  the block count and returned as a vector magnetization -- never as a
  collinear m.
* ELFCAR (dimensionless, [0, 1]) and LOCPOT (eV) are NOT volume-scaled.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from ..constants import BOHR_ANGSTROM
from ..core.grid import fourier_resample, linear_resample
from ..io.base import (FloatArray, Grid, Lattice, Structure, VolumetricData,
                       expand_species, species_labels)

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
        raise ValueError("VASP 4 file without a species line: pass species=[...] "
                         "(or keep the POTCAR next to it for read_vasp)")

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

    ``scaled_by_volume``: True for CHGCAR / AECCAR (stored as rho * V), False
    for ELFCAR / LOCPOT.
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

def read_chgcar(path: PathLike, species: Optional[Sequence[str]] = None,
                read_spin: bool = True) -> VolumetricData:
    """CHGCAR / CHG: the PAW pseudo-density plus the magnetization, if present."""
    structure, blocks = read_blocks(path, True, species, None if read_spin else 1)
    lat = structure.lattice
    rho = Grid(blocks[0], lat)
    if len(blocks) == 1:
        return VolumetricData(structure, rho, sources={"rho": str(path)})
    if len(blocks) == 2:
        return VolumetricData(structure, rho, magnetization=Grid(blocks[1], lat),
                              sources={"rho": str(path), "magnetization": str(path)})
    if len(blocks) == 4:
        mv = (Grid(blocks[1], lat), Grid(blocks[2], lat), Grid(blocks[3], lat))
        return VolumetricData(structure, rho, magnetization_vector=mv,
                              sources={"rho": str(path), "magnetization": str(path)})
    raise ValueError(f"{path}: {len(blocks)} data blocks; expected 1 (non-spin), "
                     "2 (collinear) or 4 (non-collinear). Refusing to guess.")


def read_aeccar(path: PathLike, species: Optional[Sequence[str]] = None) -> tuple[Structure, Grid]:
    """One AECCAR file (volume-scaled like CHGCAR)."""
    structure, blocks = read_blocks(path, True, species, 1)
    return structure, Grid(blocks[0], structure.lattice)


def read_all_electron(aeccar0_path: PathLike, aeccar2_path: PathLike,
                      species: Optional[Sequence[str]] = None) -> VolumetricData:
    """All-electron density rho = AECCAR0 (frozen core) + AECCAR2 (valence).

    ``density_source`` is ``"all_electron"`` and ``core_density`` keeps AECCAR0.
    """
    s0, core = read_aeccar(aeccar0_path, species)
    s2, val = read_aeccar(aeccar2_path, species)
    if core.shape != val.shape:
        raise ValueError(f"AECCAR grids differ: {core.shape} vs {val.shape}")
    if not s0.lattice.allclose(s2.lattice):
        raise ValueError("AECCAR0 and AECCAR2 lattices differ")
    return VolumetricData(s2, Grid(core.data + val.data, s2.lattice), density_source="all_electron",
                          core_density=core,
                          sources={"rho": f"{aeccar0_path} + {aeccar2_path}",
                                   "core_density": str(aeccar0_path)})


def read_elfcar(path: PathLike, species: Optional[Sequence[str]] = None) -> tuple[Structure, Grid]:
    """ELFCAR (not volume-scaled). A spin-polarized ELFCAR holds ELF_up and
    ELF_down; the first (spin-up) block is returned."""
    structure, blocks = read_blocks(path, False, species)
    return structure, Grid(blocks[0], structure.lattice)


def read_locpot(path: PathLike, species: Optional[Sequence[str]] = None) -> tuple[Structure, Grid]:
    """LOCPOT in eV (not volume-scaled). Which potential it holds depends on
    LVTOT / LVHAR; LVHAR gives the Hartree + ionic electrostatic potential."""
    structure, blocks = read_blocks(path, False, species, 1)
    return structure, Grid(blocks[0], structure.lattice)


# ----------------------------------------------------------------------
# POTCAR / OUTCAR
# ----------------------------------------------------------------------

def read_potcar_zval(path: PathLike) -> list[float]:
    """ZVAL of every PAW dataset in a POTCAR or OUTCAR, in file order.

    Reads the ``POMASS = ...; ZVAL = ...`` header line each dataset has, so
    the OUTCAR's later ``ZVAL =`` summary is not counted twice.
    """
    text = Path(path).read_text(errors="replace")
    header = re.findall(r"POMASS\s*=\s*[-\d.]+\s*;\s*ZVAL\s*=\s*([-\d.]+)", text)
    if header:
        return [float(v) for v in header]
    return [float(v) for v in re.findall(r"ZVAL\s*=\s*([-\d.]+)", text)]


def read_potcar_rcore(path: PathLike) -> list[float]:
    """RCORE (outermost PAW cutoff radius, bohr) of every dataset, in file order."""
    text = Path(path).read_text(errors="replace")
    return [float(v) for v in re.findall(r"RCORE\s*=\s*([-\d.]+)", text)]


def read_potcar_elements(path: PathLike) -> list[str]:
    """Element of every PAW dataset (``TITEL = PAW_PBE Fe_pv ...``), in file order."""
    text = Path(path).read_text(errors="replace")
    titles = re.findall(r"TITEL\s*=\s*\S+\s+(\S+)", text)
    return [t.split("_")[0].split("/")[0] for t in titles]


def _find_paw_source(chgcar: PathLike) -> Optional[Path]:
    d = Path(chgcar).resolve().parent
    return next((d / n for n in ("POTCAR", "OUTCAR") if (d / n).exists()), None)


# ----------------------------------------------------------------------
# everything for one run
# ----------------------------------------------------------------------

def read_vasp(chgcar: PathLike, elf: Optional[PathLike] = None, locpot: Optional[PathLike] = None,
              aeccar0: Optional[PathLike] = None, aeccar2: Optional[PathLike] = None,
              potcar: Union[PathLike, str, None] = "auto",
              species: Optional[Sequence[str]] = None) -> VolumetricData:
    """One VASP run as :class:`VolumetricData`.

    ``chgcar`` supplies the structure and, for spin-polarized runs, the
    magnetization. With ``aeccar0`` and ``aeccar2`` the density is the
    all-electron AECCAR0 + AECCAR2 (``density_source="all_electron"``);
    otherwise it is the CHGCAR pseudo-density. An ELFCAR on a coarser grid
    (VASP's NGX rather than NGXF) is resampled trilinearly onto the density
    grid, which keeps it in [0, 1]; a LOCPOT on a different grid is Fourier
    resampled.

    ``potcar="auto"`` looks for a POTCAR, then an OUTCAR, next to the
    CHGCAR and reads the ZVAL and RCORE of each PAW dataset from it
    (``None`` skips this). ZVAL sets the valence electron count of the
    free-atom reference; RCORE the PAW augmentation radius.
    """
    source: Optional[Path] = None
    if potcar == "auto":
        source = _find_paw_source(chgcar)
    elif potcar is not None:
        source = Path(potcar)
    if species is None and source is not None:
        head = Path(chgcar).read_text(errors="replace").splitlines()[:7]
        if len(head) > 5 and all(_is_int(t) for t in head[5].split()):
            species = read_potcar_elements(source) or None       # VASP 4 file

    vd = read_chgcar(chgcar, species)
    lat = vd.structure.lattice
    if aeccar0 is not None and aeccar2 is not None:
        ae = read_all_electron(aeccar0, aeccar2, species)
        if not ae.structure.lattice.allclose(lat) or ae.rho.shape != vd.rho.shape:
            raise ValueError("AECCAR lattice or grid differs from the CHGCAR's")
        vd.rho, vd.core_density, vd.density_source = ae.rho, ae.core_density, "all_electron"
        vd.sources.update(ae.sources)
    elif (aeccar0 is None) != (aeccar2 is None):
        raise ValueError("give both aeccar0 and aeccar2, or neither")

    shape = vd.rho.shape
    if elf is not None:
        s, g = read_elfcar(elf, species)
        _same_cell(s, lat, elf)
        vd.elf = Grid(np.clip(linear_resample(g.data, shape), 0.0, 1.0), lat) if g.shape != shape else g
        vd.sources["elf"] = str(elf)
    if locpot is not None:
        s, g = read_locpot(locpot, species)
        _same_cell(s, lat, locpot)
        vd.potential = Grid(fourier_resample(g.data, shape), lat) if g.shape != shape else g
        vd.sources["potential"] = str(locpot)

    if source is not None:
        elements = vd.structure.elements
        zvals, rcore = read_potcar_zval(source), read_potcar_rcore(source)
        if len(zvals) == len(elements):
            vd.zval = dict(zip(elements, zvals))
            vd.sources["zval"] = str(source)
        else:
            warnings.warn(f"{source}: {len(zvals)} ZVAL entries for {len(elements)} elements; "
                          "ignored", stacklevel=2)
        if len(rcore) == len(elements):
            vd.paw_radii = {e: r * BOHR_ANGSTROM for e, r in zip(elements, rcore)}
            vd.sources["paw_radii"] = str(source)
    vd.__post_init__()                          # re-validate the assembled fields
    return vd


def _same_cell(s: Structure, lattice: Lattice, path: PathLike) -> None:
    if not s.lattice.allclose(lattice):
        raise ValueError(f"{path}: lattice differs from the CHGCAR's")


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
