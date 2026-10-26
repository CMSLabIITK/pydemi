"""
pydemi.data
===========
Element data shipped with pydemi (spec §15: element data live only in this
directory and in :mod:`pydemi.constants`).

``elements.csv``   Z, symbol, covalent radius (Angstrom), ground-state configuration

Covalent radii from Cordero et al. (2008) as tabulated by Magpie (BSD licence,
``LICENSE-matminer``), configurations from pymatgen; read here without pymatgen.
pydemi uses the covalent radii (scaled shells, power partition).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _table() -> dict[str, tuple[int, Optional[float], str]]:
    rows: dict[str, tuple[int, Optional[float], str]] = {}
    for line in (DATA_DIR / "elements.csv").read_text().splitlines():
        if not line or line.startswith("#") or line.startswith("Z,"):
            continue
        z, sym, radius, config = line.split(",", 3)
        rows[sym] = (int(z), float(radius) if radius else None, config)
    return rows


def _row(symbol: str) -> tuple[int, Optional[float], str]:
    try:
        return _table()[symbol]
    except KeyError:
        raise KeyError(f"unknown element {symbol!r}") from None


def atomic_number(symbol: str) -> int:
    return _row(symbol)[0]


def symbol_of(z: int) -> str:
    for sym, (zz, _, _) in _table().items():
        if zz == int(z):
            return sym
    raise KeyError(f"no element with Z = {z}")


def covalent_radius(symbol: str) -> float:
    """Covalent radius in Angstrom (Cordero et al. 2008, via Magpie)."""
    r = _row(symbol)[1]
    if r is None:
        raise KeyError(f"no covalent radius for {symbol!r}")
    return r
