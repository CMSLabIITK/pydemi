"""
pydemi.descriptors.composition
------------------------------
Tier 2 compositional baseline (reference entries 16-30).

These reproduce Magpie / matminer ``ElementProperty`` statistics
(L. Ward et al., npj Comput. Mater. 2, 16028 (2016); L. Ward et al.,
Comput. Mater. Sci. 152, 60 (2018)) and claim no novelty -- cite matminer
when reporting them. The element tables are matminer 0.10.1's Magpie
files, vendored verbatim in ``pydemi/data`` (BSD licence), so values agree
with matminer without pulling in its dependency tree (see :mod:`pydemi.elements`).

Missing table entries are imputed with the mean over all elements, which
is matminer's ``impute_nan=True`` default. Pass ``impute_nan=False`` to
get NaN instead.
"""

from typing import Mapping
import warnings

import numpy as np

from ..elements import atomic_number, element_property
from ..structure import Structure

# spglib space-group number -> crystal system, 1 = triclinic ... 7 = cubic
_CRYSTAL_SYSTEMS = ((2, 1), (15, 2), (74, 3), (142, 4), (167, 5), (194, 6), (230, 7))


def composition_features(amounts: Mapping[str, float], impute_nan: bool = True) -> dict:
    """Entries 16-27, 29-30 from element -> amount (any normalization)."""
    elements = list(amounts)
    x = np.array([float(amounts[e]) for e in elements])
    x = x / x.sum()

    def prop(name):
        return np.array([element_property(e, name, impute_nan) for e in elements])

    A, chi, vec = prop("AtomicWeight"), prop("Electronegativity"), prop("NValence")
    R, period, fblock = prop("AtomicRadius"), prop("Row"), prop("IsFBlock")
    elneg_diff = float(chi.max() - chi.min())

    return {
        "mean_mass": float(x @ A),
        "max_mass": float(A.max()),
        "mass_range": float(A.max() - A.min()),
        "mean_elneg": float(x @ chi),
        "elneg_diff": elneg_diff,
        "mean_vec": float(x @ vec),
        "max_vec": float(vec.max()),
        "mean_radius": float(x @ R),
        "radius_diff": float(R.max() - R.min()),
        "n_elements": len(elements),
        "ionicity": float(1.0 - np.exp(-elneg_diff ** 2 / 4.0)),
        "mean_period": float(x @ period),
        "is_f_block": int(bool(np.all(fblock == 1))),
        "has_f_block": int(bool(np.any(fblock == 1))),
    }


def crystal_system(structure: Structure, symprec: float = 0.01):
    """(crystal_system_int, space_group_number) via spglib."""
    import spglib
    numbers = [atomic_number(s) for s in structure.species]
    with warnings.catch_warnings():
        # spglib >= 2.5 warns about its legacy error handling on every call
        warnings.simplefilter("ignore", DeprecationWarning)
        data = spglib.get_symmetry_dataset(
            (structure.lattice, structure.frac_coords, numbers), symprec=symprec)
    if data is None:
        return float("nan"), float("nan")
    # spglib >= 2.5 returns a dataclass, older versions a dict
    sg = int(data.number if hasattr(data, "number") else data["number"])
    cs = next(code for upper, code in _CRYSTAL_SYSTEMS if sg <= upper)
    return cs, sg


def tier2(structure: Structure, impute_nan: bool = True, symprec: float = 0.01) -> dict:
    amounts = {e: structure.species.count(e) for e in structure.elements}
    out = composition_features(amounts, impute_nan)
    cs, sg = crystal_system(structure, symprec)
    out["crystal_system_int"] = cs
    out["space_group_number"] = sg
    return out
