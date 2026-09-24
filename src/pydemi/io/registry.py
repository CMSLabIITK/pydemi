"""
pydemi.io.registry
==================
Format sniffing and dispatch: :func:`read` turns any supported density file
into :class:`~pydemi.io.base.VolumetricData`.

Formats are recognized by file name first (CHGCAR, CHG, AECCAR0/2, ELFCAR,
LOCPOT, ``*.cube`` / ``*.cub``, ``*.xsf``), then by content.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Union

from .base import VolumetricData
from .cube import read_cube
from .vasp import read_vasp
from .xsf import read_xsf

PathLike = Union[str, Path]
Format = Literal["chgcar", "aeccar", "elfcar", "locpot", "cube", "xsf"]

_NAMES: dict[str, Format] = {"CHGCAR": "chgcar", "CHG": "chgcar", "AECCAR0": "aeccar",
                             "AECCAR1": "aeccar", "AECCAR2": "aeccar", "ELFCAR": "elfcar",
                             "LOCPOT": "locpot"}


def sniff(path: PathLike) -> Format:
    """Guess the format of a volumetric file."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".cube", ".cub"):
        return "cube"
    if suffix == ".xsf":
        return "xsf"
    stem = p.name.split(".")[0].split("_")[0].upper()
    if stem in _NAMES:
        return _NAMES[stem]
    for key, fmt in _NAMES.items():
        if p.name.upper().startswith(key):
            return fmt
    with open(p, errors="replace") as fh:
        head = [fh.readline() for _ in range(8)]
    text = "".join(head).upper()
    if "PRIMVEC" in text or "CRYSTAL" in head[0].upper():
        return "xsf"
    try:
        int(head[2].split()[0])
        float(head[3].split()[1])
        if len(head[3].split()) == 4 and len(head[2].split()) >= 4:
            return "cube"
    except (ValueError, IndexError):
        pass
    return "chgcar"


def read(path: PathLike, **kwargs: Any) -> VolumetricData:
    """Read a charge-density file of any supported format.

    VASP files go through :func:`pydemi.io.vasp.read_vasp` (keyword arguments
    such as ``elf=``, ``locpot=``, ``aeccar0=``, ``aeccar2=`` are passed on);
    cube and XSF files hold one density and are read with unit conversion.
    """
    fmt = sniff(path)
    if fmt == "chgcar":
        return read_vasp(path, **kwargs)
    if fmt in ("cube", "xsf"):
        reader = read_cube if fmt == "cube" else read_xsf
        structure, grid = reader(path, "density")
        return VolumetricData(structure, grid, sources={"rho": str(path)})
    raise ValueError(f"{path}: a {fmt.upper()} file is not a charge density; read the CHGCAR "
                     f"and pass this file to read_vasp (elf=, locpot=, aeccar0=/aeccar2=)")
