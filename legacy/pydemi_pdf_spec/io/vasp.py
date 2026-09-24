"""
pydemi.io.vasp
--------------
Readers (and a minimal writer) for VASP volumetric files: CHGCAR / CHG,
AECCAR0 / AECCAR2, ELFCAR, LOCPOT.

All of these share one layout: a POSCAR header, a blank line, the grid
dimensions, the data in Fortran (x-fastest) order, and -- for CHGCAR --
PAW augmentation occupancies. Spin-polarized and non-collinear runs
append further blocks, each introduced by a repeat of the grid-dimension
line. Unlike a reader that stops at the first block, :func:`read_volumetric`
returns every block, so the magnetization density of a spin-polarized
CHGCAR is available to the spin descriptors (Family E).

Units
-----
CHGCAR and AECCAR store rho * V_cell; they are divided by the cell volume
here so densities come back in electrons / Angstrom^3. ELFCAR (dimensionless)
and LOCPOT (eV) are stored unscaled and returned as-is.

PAW caveat
----------
CHGCAR holds the pseudo-valence density plus compensation charge, not an
all-electron density. The all-electron reconstruction is AECCAR0 (frozen
core) + AECCAR2 (valence); use :func:`read_aeccar` for it.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union
import warnings

import numpy as np

from ..structure import Structure

PathLike = Union[str, Path]

SPIN_NONE = "none"
SPIN_COLLINEAR = "collinear"
SPIN_NONCOLLINEAR = "noncollinear"


@dataclass(eq=False)
class VolumetricData:
    """Every data block of one VASP volumetric file."""
    structure: Structure
    blocks: list                      # list of (nx, ny, nz) ndarrays
    kind: str = "volumetric"
    source: Optional[str] = None

    @property
    def shape(self) -> tuple:
        return self.blocks[0].shape


@dataclass(eq=False)
class ChargeDensity:
    """
    Total density and (optionally) magnetization density.

    ``magnetization`` is ``None`` for non-spin-polarized runs, an
    (nx, ny, nz) array m = rho_up - rho_down for collinear runs, and a
    (3, nx, ny, nz) array (m_x, m_y, m_z) for non-collinear runs.
    """
    structure: Structure
    total: np.ndarray
    magnetization: Optional[np.ndarray] = None
    spin_mode: str = SPIN_NONE
    all_electron: bool = False
    source: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple:
        return self.total.shape


# ----------------------------------------------------------------------
# low-level parsing
# ----------------------------------------------------------------------

def _is_int(token: str) -> bool:
    try:
        int(token)
        return True
    except ValueError:
        return False


def _parse_header(lines: Sequence[str], species: Optional[Sequence[str]]):
    """Parse the POSCAR header. Returns (Structure, index of next line)."""
    scale_tokens = lines[1].split()
    lattice = np.array([[float(x) for x in lines[i].split()[:3]]
                        for i in (2, 3, 4)])

    if len(scale_tokens) == 3:
        lattice = lattice * np.array([float(x) for x in scale_tokens])[:, None]
        cart_scale = 1.0
    else:
        s = float(scale_tokens[0])
        if s < 0:  # negative scale = target cell volume
            s = (-s / abs(np.linalg.det(lattice))) ** (1.0 / 3.0)
        lattice = lattice * s
        cart_scale = s

    idx = 5
    tokens = lines[idx].split()
    if all(_is_int(t) for t in tokens):
        file_species = None           # VASP 4: no species line
    else:
        file_species = tokens
        idx += 1
    counts = [int(x) for x in lines[idx].split()]
    idx += 1

    if species is not None:
        labels = list(species)
        if len(labels) == len(counts):
            labels = [s for s, c in zip(labels, counts) for _ in range(c)]
    elif file_species is not None:
        # POTCAR-style labels such as "Fe_pv" or "O/" -> element symbol
        clean = [t.split("_")[0].split("/")[0] for t in file_species]
        labels = [s for s, c in zip(clean, counts) for _ in range(c)]
    else:
        warnings.warn(
            "VASP4-format file has no species line; pass species=[...] "
            "to label atoms. Using placeholders X0, X1, ...",
            stacklevel=3,
        )
        labels = [f"X{t}" for t, c in enumerate(counts) for _ in range(c)]

    n_atoms = sum(counts)
    if lines[idx].strip()[:1].lower() == "s":  # Selective dynamics
        idx += 1
    cartesian = lines[idx].strip()[:1].lower() in ("c", "k")
    idx += 1

    coords = np.array([[float(x) for x in lines[i].split()[:3]]
                       for i in range(idx, idx + n_atoms)])
    idx += n_atoms
    if cartesian:
        frac = (coords * cart_scale) @ np.linalg.inv(lattice)
    else:
        frac = coords

    return Structure(lattice, labels, frac), idx


def _find_dims(lines: Sequence[str], start: int, dims=None) -> int:
    """Index of the next grid-dimension line at or after ``start``, or -1.

    With ``dims`` given, only a line holding exactly those three integers
    counts, which skips augmentation-occupancy and magnetic-moment lines.
    """
    for i in range(start, len(lines)):
        tokens = lines[i].split()
        if len(tokens) != 3 or not all(_is_int(t) for t in tokens):
            continue
        if dims is None or tuple(int(t) for t in tokens) == dims:
            return i
    return -1


def _read_block(lines: Sequence[str], start: int, n_values: int):
    """Read ``n_values`` floats beginning at line ``start``.

    Returns (1-D array, index of the line after the block).
    """
    per_line = len(lines[start].split())
    n_lines = -(-n_values // per_line)
    chunk = " ".join(lines[start:start + n_lines])
    values = np.fromstring(chunk, sep=" ")
    if values.size == n_values:
        return values, start + n_lines

    # irregular line lengths: fall back to accumulating line by line
    collected, i, count = [], start, 0
    while count < n_values and i < len(lines):
        row = np.fromstring(lines[i], sep=" ")
        collected.append(row)
        count += row.size
        i += 1
    values = np.concatenate(collected)
    if values.size < n_values:
        raise ValueError(
            f"data block truncated: expected {n_values} values, got {values.size}"
        )
    return values[:n_values], i


def read_volumetric(path: PathLike,
                    scaled_by_volume: bool = True,
                    species: Optional[Sequence[str]] = None,
                    kind: str = "volumetric",
                    max_blocks: Optional[int] = None) -> VolumetricData:
    """
    Read every data block of a VASP volumetric file.

    Parameters
    ----------
    path : file path
    scaled_by_volume : bool
        True for CHGCAR/AECCAR (stored as rho * V), False for ELFCAR/LOCPOT.
    species : optional element labels, either one per species group or one
        per atom. Required for VASP 4 files that lack a species line.
    max_blocks : stop after this many blocks (e.g. 1 to skip spin data).
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    structure, idx = _parse_header(lines, species)

    dim_idx = _find_dims(lines, idx)
    if dim_idx < 0:
        raise ValueError(f"{path}: no grid-dimension line found")
    dims = tuple(int(t) for t in lines[dim_idx].split())
    n_values = dims[0] * dims[1] * dims[2]
    norm = structure.volume if scaled_by_volume else 1.0

    blocks = []
    while dim_idx >= 0:
        values, end = _read_block(lines, dim_idx + 1, n_values)
        blocks.append(values.reshape(dims, order="F") / norm)
        if max_blocks is not None and len(blocks) >= max_blocks:
            break
        dim_idx = _find_dims(lines, end, dims)

    return VolumetricData(structure, blocks, kind=kind, source=str(path))


# ----------------------------------------------------------------------
# file-specific readers
# ----------------------------------------------------------------------

def _charge_from_blocks(vol: VolumetricData, all_electron=False) -> ChargeDensity:
    blocks = vol.blocks
    if len(blocks) == 1:
        mag, mode = None, SPIN_NONE
    elif len(blocks) == 2:
        mag, mode = blocks[1], SPIN_COLLINEAR
    elif len(blocks) == 4:
        mag, mode = np.stack(blocks[1:4]), SPIN_NONCOLLINEAR
    else:
        raise ValueError(
            f"{vol.source}: {len(blocks)} data blocks; expected 1 (non-spin), "
            "2 (collinear) or 4 (non-collinear). Refusing to guess."
        )
    return ChargeDensity(vol.structure, blocks[0], mag, mode,
                         all_electron=all_electron, source=vol.source)


def read_chgcar(path: PathLike, species=None, read_spin: bool = True) -> ChargeDensity:
    """Read a CHGCAR/CHG: total density plus magnetization if present."""
    vol = read_volumetric(path, scaled_by_volume=True, species=species,
                          kind="CHGCAR", max_blocks=None if read_spin else 1)
    return _charge_from_blocks(vol)


def read_aeccar(aeccar0: PathLike, aeccar2: PathLike, species=None) -> ChargeDensity:
    """All-electron density rho_AE = AECCAR0 (core) + AECCAR2 (valence)."""
    core = read_volumetric(aeccar0, True, species, kind="AECCAR0", max_blocks=1)
    val = read_volumetric(aeccar2, True, species, kind="AECCAR2", max_blocks=1)
    if core.shape != val.shape:
        raise ValueError(f"AECCAR grids differ: {core.shape} vs {val.shape}")
    if not np.allclose(core.structure.lattice, val.structure.lattice):
        raise ValueError("AECCAR0 and AECCAR2 lattices differ")
    return ChargeDensity(val.structure, core.blocks[0] + val.blocks[0],
                         all_electron=True, source=str(aeccar2),
                         extra={"core": core.blocks[0]})


def read_elfcar(path: PathLike, species=None) -> VolumetricData:
    """ELFCAR. One block (non-spin) or two (spin up, spin down).

    Note ELFCAR is usually on the coarse NGX grid, not the CHGCAR NGXF grid.
    """
    return read_volumetric(path, scaled_by_volume=False, species=species,
                           kind="ELFCAR")


def read_locpot(path: PathLike, species=None) -> VolumetricData:
    """LOCPOT (eV). Which potential it holds depends on LVTOT/LVHAR."""
    return read_volumetric(path, scaled_by_volume=False, species=species,
                           kind="LOCPOT")


# ----------------------------------------------------------------------
# writer
# ----------------------------------------------------------------------

def write_volumetric(path: PathLike, structure: Structure, blocks,
                     scaled_by_volume: bool = True, per_line: int = 5,
                     comment: str = "written by pydemi") -> None:
    """Write a VASP5-format volumetric file (no augmentation section)."""
    elements = structure.elements
    counts = [structure.species.count(e) for e in elements]
    order = np.concatenate([np.flatnonzero(np.array(structure.species) == e)
                            for e in elements])

    out = [comment, "1.0"]
    out += ["  " + " ".join(f"{x:.10f}" for x in row) for row in structure.lattice]
    out.append("  " + " ".join(elements))
    out.append("  " + " ".join(str(c) for c in counts))
    out.append("Direct")
    out += ["  " + " ".join(f"{x:.10f}" for x in structure.frac_coords[i])
            for i in order]

    norm = structure.volume if scaled_by_volume else 1.0
    for block in blocks:
        block = np.asarray(block)
        out.append("")
        out.append(" " + " ".join(f"{n:4d}" for n in block.shape))
        flat = (block * norm).ravel(order="F")
        for i in range(0, flat.size, per_line):
            out.append(" " + " ".join(f"{v:.11E}" for v in flat[i:i + per_line]))

    Path(path).write_text("\n".join(out) + "\n")
