"""
pydemi.structure
----------------
Minimal periodic structure: lattice, per-atom species, fractional
coordinates. Deliberately dependency-free; convert to and from pymatgen
with :meth:`Structure.from_pymatgen` / :meth:`Structure.to_pymatgen`.

Conventions: ``lattice`` rows are the lattice vectors a1, a2, a3 in
Angstrom, so a Cartesian position is ``x = u @ lattice`` for fractional
row vector ``u``.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True, eq=False)
class Structure:
    lattice: np.ndarray       # (3, 3) Angstrom, rows are lattice vectors
    species: tuple            # (n_atoms,) element symbols
    frac_coords: np.ndarray   # (n_atoms, 3)

    def __post_init__(self):
        lattice = np.asarray(self.lattice, dtype=float).reshape(3, 3)
        frac = np.asarray(self.frac_coords, dtype=float).reshape(-1, 3)
        species = tuple(str(s) for s in self.species)
        if len(species) != frac.shape[0]:
            raise ValueError(
                f"{len(species)} species labels for {frac.shape[0]} atoms"
            )
        if abs(np.linalg.det(lattice)) < 1e-12:
            raise ValueError("lattice is singular")
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "frac_coords", frac)
        object.__setattr__(self, "species", species)

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    @cached_property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    @cached_property
    def cart_coords(self) -> np.ndarray:
        return self.frac_coords @ self.lattice

    @cached_property
    def inv_lattice(self) -> np.ndarray:
        return np.linalg.inv(self.lattice)

    @cached_property
    def reciprocal_lattice(self) -> np.ndarray:
        """Rows b_j with a_i . b_j = 2 pi delta_ij (1/Angstrom)."""
        return 2.0 * np.pi * self.inv_lattice.T

    @cached_property
    def elements(self) -> tuple:
        """Distinct elements in order of first appearance."""
        return tuple(dict.fromkeys(self.species))

    @cached_property
    def element_index(self) -> np.ndarray:
        """(n_atoms,) index of each atom's element into :attr:`elements`."""
        lookup = {e: i for i, e in enumerate(self.elements)}
        return np.array([lookup[s] for s in self.species], dtype=int)

    @classmethod
    def from_pymatgen(cls, structure) -> "Structure":
        return cls(
            lattice=structure.lattice.matrix,
            species=[site.specie.symbol for site in structure],
            frac_coords=structure.frac_coords,
        )

    def to_pymatgen(self):
        from pymatgen.core import Structure as PmgStructure
        return PmgStructure(self.lattice, list(self.species), self.frac_coords)

    def __repr__(self) -> str:
        formula = " ".join(
            f"{e}{self.species.count(e)}" for e in self.elements
        )
        return f"Structure({formula}, V={self.volume:.3f} A^3)"
