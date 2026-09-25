"""
pydemi.shells
-------------
Radial shell partition about the nearest nucleus:

    core          r_k <= c1
    bond          c1 < r_k <= c2
    interstitial  r_k > c2

``Shells()`` uses the absolute cutoffs c1 = 0.8, c2 = 1.5 Angstrom.
``Shells.scaled(s1, s2, radii)`` uses per-element cutoffs c = s * R_e,
with R_e a tabulated radius of the nearest atom's element.

Every shell-partitioned descriptor inherits a cutoff sensitivity (the
dQ/dc slope is non-trivial for both forms); report it rather than
assume it away.
"""

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from .constants import SHELL_C1, SHELL_C2
from .geometry import NearestAtom
from .structure import Structure


@dataclass(frozen=True, eq=False)
class ShellMasks:
    core: np.ndarray
    bond: np.ndarray
    interstitial: np.ndarray


@dataclass(frozen=True)
class Shells:
    c1: float = SHELL_C1
    c2: float = SHELL_C2
    radii: Optional[Mapping[str, float]] = None  # set => radius-scaled mode

    def __post_init__(self):
        if not (0 <= self.c1 < self.c2):
            raise ValueError(f"need 0 <= c1 < c2, got c1={self.c1}, c2={self.c2}")

    @classmethod
    def scaled(cls, s1: float, s2: float, radii: Mapping[str, float]) -> "Shells":
        """Per-element cutoffs c1 = s1 * R_e, c2 = s2 * R_e."""
        return cls(c1=s1, c2=s2, radii=dict(radii))

    @property
    def is_scaled(self) -> bool:
        return self.radii is not None

    def atom_cutoffs(self, structure: Structure):
        """(c1_i, c2_i) per atom, Angstrom."""
        if not self.is_scaled:
            ones = np.ones(structure.n_atoms)
            return self.c1 * ones, self.c2 * ones
        missing = [e for e in structure.elements if e not in self.radii]
        if missing:
            raise KeyError(f"no radius for element(s) {missing}")
        R = np.array([self.radii[s] for s in structure.species])
        return self.c1 * R, self.c2 * R

    def masks(self, geometry: NearestAtom, structure: Structure) -> ShellMasks:
        c1, c2 = self.atom_cutoffs(structure)
        r = geometry.distance
        c1_k = c1[geometry.atom_index]
        c2_k = c2[geometry.atom_index]
        return ShellMasks(core=r <= c1_k,
                          bond=(r > c1_k) & (r <= c2_k),
                          interstitial=r > c2_k)

