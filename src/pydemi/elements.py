"""
pydemi.elements
---------------
Elemental property lookup from the Magpie tables vendored in
``pydemi/data/magpie_elements.csv`` (matminer 0.10.1, BSD licence; Ward et
al., npj Comput. Mater. 2, 16028 (2016)).

Missing entries are imputed with the mean over all elements by default,
matching matminer's ``impute_nan=True``.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parent / "data" / "magpie_elements.csv"


@lru_cache(maxsize=None)
def _tables():
    lines = [l for l in _DATA.read_text().splitlines() if not l.startswith("#")]
    header = lines[0].split(",")
    rows = [l.split(",") for l in lines[1:]]
    symbols = [r[1] for r in rows]
    props = {}
    for j, name in enumerate(header[2:], start=2):
        props[name] = np.array([float(r[j]) for r in rows])
    return {s: i for i, s in enumerate(symbols)}, props


def element_property(symbol: str, prop: str, impute_nan: bool = True) -> float:
    index, props = _tables()
    if symbol not in index:
        raise KeyError(f"no Magpie data for element {symbol!r}")
    column = props[prop]
    value = column[index[symbol]]
    if np.isnan(value) and impute_nan:
        value = float(np.nanmean(column))
    return float(value)


def atomic_number(symbol: str) -> int:
    index, _ = _tables()
    return index[symbol] + 1


def covalent_radii(elements) -> dict:
    """Magpie covalent radii in Angstrom, e.g. for power-diagram weights."""
    return {e: element_property(e, "CovalentRadius") / 100.0 for e in elements}
