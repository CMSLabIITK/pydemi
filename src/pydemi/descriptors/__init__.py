"""
pydemi.descriptors
------------------
Descriptor families built on the shared grid engine.

    >>> from pydemi import Engine
    >>> from pydemi.descriptors import compute_descriptors
    >>> eng = Engine.from_chgcar("CHGCAR")
    >>> d = compute_descriptors(eng)                     # every implemented family
    >>> d = compute_descriptors(eng, kinds=("descriptor",))  # physical descriptors only

See :mod:`pydemi.descriptors.registry` for each quantity's reference entry,
kind and formula.
"""

from typing import Iterable, Optional

from ..engine import ELF, RHO, Engine
from ..shells import Shells
from .anisotropy import anisotropy_family
from .bonds import bond_census, bond_family
from .topology import morse_census, percolation_levels, topology_family
from .composition import composition_features, crystal_system, tier2
from .deformation import deformation_density, deformation_family, promolecule
from .elf import elf_family, elf_fidelity
from .hessian import ellipticity_family, nci_family
from .information import information_family
from .kinetic import elf_d_field, energy_family
from .potential import potential_family, site_potentials
from .sites import hirshfeld_charges, partition_family, site_charges, site_family
from .spin import site_moments, spin_family
from .registry import (CROSS_TERM, DATASET, DESCRIPTOR, FIELD, METADATA,
                       PREPROCESSING, REGISTRY, SCALAR_KINDS, SITE, VARIANT,
                       DescriptorInfo, describe, names)
from .tier1 import tier1
from .tier3 import tier3

ALL_KINDS = SCALAR_KINDS
FAMILIES = ("tier1", "tier2", "tier3", "B", "F2", "F3", "F4", "F5", "F6", "I1",
            "C", "E", "H", "G", "I2", "A", "D")
# families Family D needs as inputs
_D_INPUTS = ("tier1", "tier2", "F2", "I2")


def _family_b(engine, field, shells):
    out = elf_family(engine, elf_d_field(engine, field).name, "ELFD", shells)
    if ELF in engine:
        out.update(elf_family(engine, ELF, "ELF", shells))
        out.update(elf_fidelity(engine, ELF, elf_d_field(engine, field).name, shells))
    return out


# family -> callable(engine, field, shells) for the independent families
_RUNNERS = {
    "B": _family_b,
    "F2": potential_family,
    "F3": lambda e, f, s: nci_family(e, f),
    "F4": ellipticity_family,
    "F5": energy_family,
    "F6": lambda e, f, s: information_family(e, f),
    "I1": lambda e, f, s: anisotropy_family(e, f),
    "C": site_family,
    "E": spin_family,
    "H": partition_family,
    "G": topology_family,
    "I2": lambda e, f, s: bond_family(e, f),
    # AECCAR when loaded, else NaN: the CHGCAR route must be asked for directly
    "A": lambda e, f, s: deformation_family(e, None, s),
}


def compute_descriptors(engine: Engine, families: Iterable[str] = FAMILIES,
                        kinds: Iterable[str] = ALL_KINDS, field: str = RHO,
                        shells: Optional[Shells] = None,
                        calibrations: Optional[dict] = None) -> dict:
    """Compute the requested families; returns {name: value} in reference order.

    ``kinds`` filters the output by registry kind (descriptor, variant,
    cross_term, preprocessing, metadata). Tier 3 depends on Tier 1 and 2,
    and Family D on Tier 1-2, F2 and I2; those are computed internally
    whenever needed. ``calibrations`` may hold an ``"ionicity"``
    (IonicityCalibration) and a ``"bulk"`` (BulkModulusCalibration) entry
    for Family D; without them the calibrated values are NaN.
    """
    families, kinds = tuple(families), tuple(kinds)
    requested = families
    if "D" in families:
        families = tuple(dict.fromkeys(families + _D_INPUTS))
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown families {sorted(unknown)}; choose from {FAMILIES}")

    values = {}
    if "tier1" in families or "tier3" in families:
        values.update(tier1(engine, field, shells))
    if "tier2" in families or "tier3" in families:
        values.update(tier2(engine.structure))
    if "tier3" in families:
        values.update(tier3(engine, values, values, field, shells))
    for family in families:
        if family in _RUNNERS:
            values.update(_RUNNERS[family](engine, field, shells))
    if "D" in families:
        from ..calibration import calibration_family
        cal = calibrations or {}
        values.update(calibration_family(engine.structure, values,
                                         cal.get("ionicity"), cal.get("bulk")))

    wanted = names(kinds=kinds, opt_in=True)
    return {n: values[n] for n in wanted
            if n in values and REGISTRY[n].family in requested}


__all__ = [
    "compute_descriptors", "tier1", "tier2", "tier3", "composition_features",
    "crystal_system", "elf_family", "elf_d_field", "energy_family",
    "potential_family", "site_potentials", "nci_family", "ellipticity_family",
    "information_family", "anisotropy_family", "site_family", "partition_family",
    "site_charges", "spin_family", "site_moments", "topology_family",
    "morse_census", "percolation_levels", "bond_family", "bond_census",
    "deformation_family", "deformation_density", "promolecule", "elf_fidelity",
    "hirshfeld_charges",
    "describe", "names", "REGISTRY", "DescriptorInfo", "FAMILIES",
    "DESCRIPTOR", "VARIANT", "CROSS_TERM", "PREPROCESSING", "METADATA",
    "FIELD", "SITE", "DATASET",
]
