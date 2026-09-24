"""
pydemi.io.base
==============
The data model (spec §2): :class:`Lattice`, :class:`Grid`, :class:`Structure`
and :class:`VolumetricData`.

Conventions
-----------
* ``Lattice.matrix`` holds the lattice vectors a1, a2, a3 as ROWS, in
  Angstrom, so a Cartesian position is ``x = u @ A`` for a fractional row
  vector ``u``.
* ``Grid.data[i, j, k]`` is the value at fractional coordinate
  (i/n1, j/n2, k/n3). Grids are ALWAYS periodic: index arithmetic is modular
  everywhere.
* Densities (``rho``, ``magnetization``) are in electrons / Angstrom^3, the
  ELF is dimensionless in [0, 1], potentials are in eV.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]
DensitySource = Literal["pseudo", "all_electron"]
SpinMode = Literal["none", "collinear", "noncollinear"]


@dataclass(frozen=True, eq=False)
class Lattice:
    """Periodic lattice; ``matrix`` rows are a1, a2, a3 in Angstrom."""

    matrix: NDArray[np.float64]

    def __post_init__(self) -> None:
        m = np.array(self.matrix, dtype=np.float64).reshape(3, 3)
        if abs(float(np.linalg.det(m))) < 1e-12:
            raise ValueError("lattice matrix is singular")
        m.setflags(write=False)
        object.__setattr__(self, "matrix", m)

    @cached_property
    def volume(self) -> float:
        """Cell volume |det A| in Angstrom^3."""
        return float(abs(np.linalg.det(self.matrix)))

    @cached_property
    def inverse(self) -> NDArray[np.float64]:
        """A^{-1}; fractional coordinates are u = x @ A^{-1}."""
        return np.asarray(np.linalg.inv(self.matrix), dtype=np.float64)

    @cached_property
    def reciprocal(self) -> NDArray[np.float64]:
        """B = inv(A)^T: rows are b1, b2, b3 WITHOUT the 2 pi factor (a_i . b_j = delta_ij)."""
        return np.ascontiguousarray(self.inverse.T)

    @cached_property
    def metric(self) -> NDArray[np.float64]:
        """Contravariant metric g^{ab} = b_a . b_b, i.e. G = B B^T (1/Angstrom^2)."""
        B = self.reciprocal
        return np.asarray(B @ B.T, dtype=np.float64)

    @cached_property
    def covariant_metric(self) -> NDArray[np.float64]:
        """Covariant metric g_ab = a_a . a_b, i.e. A A^T (Angstrom^2)."""
        A = self.matrix
        return np.asarray(A @ A.T, dtype=np.float64)

    @cached_property
    def lengths(self) -> NDArray[np.float64]:
        """|a1|, |a2|, |a3| in Angstrom."""
        return np.asarray(np.linalg.norm(self.matrix, axis=1), dtype=np.float64)

    @cached_property
    def heights(self) -> NDArray[np.float64]:
        """Distances between opposite cell faces, 1 / |b_a| (Angstrom)."""
        return np.asarray(1.0 / np.linalg.norm(self.reciprocal, axis=1), dtype=np.float64)

    def allclose(self, other: "Lattice", atol: float = 1e-6) -> bool:
        return bool(np.allclose(self.matrix, other.matrix, atol=atol))

    def __repr__(self) -> str:
        return f"Lattice(lengths={np.round(self.lengths, 4).tolist()}, V={self.volume:.4f} A^3)"


@dataclass(eq=False)
class Grid:
    """A scalar field sampled on a periodic grid spanning ``lattice``."""

    data: FloatArray
    lattice: Lattice

    def __post_init__(self) -> None:
        data = np.asarray(self.data)
        if data.ndim != 3 or min(data.shape) < 1:
            raise ValueError(f"grid data must be a 3-D array, got shape {data.shape}")
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float64)
        self.data = data

    @property
    def shape(self) -> Tuple[int, int, int]:
        s = self.data.shape
        return (int(s[0]), int(s[1]), int(s[2]))

    @property
    def n_voxels(self) -> int:
        return int(self.data.size)

    @property
    def dV(self) -> float:
        """Voxel volume V_cell / N (Angstrom^3)."""
        return self.lattice.volume / self.n_voxels

    def astype(self, dtype: Any) -> "Grid":
        return Grid(self.data.astype(dtype, copy=False), self.lattice)


@dataclass(eq=False)
class Structure:
    """Periodic structure: lattice, one element symbol per atom, fractional coordinates."""

    lattice: Lattice
    species: list[str]
    frac_coords: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not isinstance(self.lattice, Lattice):
            self.lattice = Lattice(np.asarray(self.lattice, dtype=np.float64))
        self.species = [str(s) for s in self.species]
        frac = np.array(self.frac_coords, dtype=np.float64).reshape(-1, 3)
        if frac.shape[0] != len(self.species):
            raise ValueError(f"{len(self.species)} species for {frac.shape[0]} atoms")
        self.frac_coords = frac

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    @property
    def volume(self) -> float:
        return self.lattice.volume

    @property
    def cart_coords(self) -> NDArray[np.float64]:
        return np.asarray(self.frac_coords @ self.lattice.matrix, dtype=np.float64)

    @property
    def elements(self) -> list[str]:
        """Distinct elements, in order of first appearance."""
        return list(dict.fromkeys(self.species))

    @property
    def element_index(self) -> NDArray[np.int64]:
        """Index of each atom's element in :attr:`elements`."""
        lookup = {e: i for i, e in enumerate(self.elements)}
        return np.array([lookup[s] for s in self.species], dtype=np.int64)

    def site_counts(self) -> dict[str, int]:
        """Number of sites of each element."""
        return {e: self.species.count(e) for e in self.elements}

    @classmethod
    def from_pymatgen(cls, structure: Any) -> "Structure":
        """Convert a ``pymatgen.core.Structure`` (structure I/O is pymatgen's only role)."""
        return cls(Lattice(np.asarray(structure.lattice.matrix)),
                   [site.specie.symbol for site in structure],
                   np.asarray(structure.frac_coords))

    def to_pymatgen(self) -> Any:
        from pymatgen.core.structure import Structure as PmgStructure
        return PmgStructure(self.lattice.matrix, list(self.species), self.frac_coords)

    def __repr__(self) -> str:
        formula = " ".join(f"{e}{n}" for e, n in self.site_counts().items())
        return f"Structure({formula}, V={self.volume:.3f} A^3)"


def as_structure(obj: Any) -> Structure:
    """Accept a pydemi :class:`Structure` or a pymatgen ``Structure``."""
    if isinstance(obj, Structure):
        return obj
    if hasattr(obj, "lattice") and hasattr(obj, "frac_coords") and hasattr(obj, "sites"):
        return Structure.from_pymatgen(obj)
    raise TypeError(f"expected a pydemi or pymatgen Structure, got {type(obj).__name__}")


@dataclass(eq=False)
class VolumetricData:
    """One structure with its grid fields (spec §2).

    ``rho`` is always present. ``magnetization`` is the collinear m = rho_up -
    rho_down; a non-collinear run sets ``magnetization_vector`` = (m_x, m_y,
    m_z) instead and leaves ``magnetization`` None. ``density_source`` records
    whether ``rho`` is the PAW pseudo-density (``"pseudo"``, CHGCAR) or the
    all-electron reconstruction (``"all_electron"``, AECCAR0 + AECCAR2), which
    changes the interpretation of every core-region descriptor. For the
    all-electron case ``core_density`` keeps AECCAR0.

    ``zval`` (element -> valence electrons of the PAW dataset) and
    ``paw_radii`` (element -> augmentation radius RCORE, Angstrom) come from
    a POTCAR or OUTCAR when one was read.

    ``options`` and ``cache`` are managed by :func:`pydemi.featurize`: the
    geometry pass, derivatives and derived fields are computed once and
    cached here, keyed by the settings they depend on.
    """

    structure: Structure
    rho: Grid
    magnetization: Optional[Grid] = None
    elf: Optional[Grid] = None
    potential: Optional[Grid] = None
    density_source: DensitySource = "pseudo"
    magnetization_vector: Optional[Tuple[Grid, Grid, Grid]] = None
    core_density: Optional[Grid] = None
    zval: Optional[dict[str, float]] = None
    paw_radii: Optional[dict[str, float]] = None
    sources: dict[str, str] = field(default_factory=dict)
    options: Any = None
    cache: dict[Any, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.structure = as_structure(self.structure)
        if self.density_source not in ("pseudo", "all_electron"):
            raise ValueError(f"density_source must be 'pseudo' or 'all_electron', "
                             f"got {self.density_source!r}")
        lat = self.structure.lattice
        for name, g in self._grids():
            if not g.lattice.allclose(lat):
                raise ValueError(f"{name}: grid lattice differs from the structure's")
            if g.shape != self.rho.shape:
                raise ValueError(f"{name}: grid shape {g.shape} differs from rho's "
                                 f"{self.rho.shape}; resample it first (pydemi.core.grid.resample)")
        if self.magnetization is not None and self.magnetization_vector is not None:
            raise ValueError("give magnetization (collinear) or magnetization_vector "
                             "(non-collinear), not both")

    def _grids(self) -> list[tuple[str, Grid]]:
        out = [("rho", self.rho)]
        for name in ("magnetization", "elf", "potential", "core_density"):
            g = getattr(self, name)
            if g is not None:
                out.append((name, g))
        if self.magnetization_vector is not None:
            out += [(f"magnetization_{c}", g) for c, g in zip("xyz", self.magnetization_vector)]
        return out

    @property
    def spin_mode(self) -> SpinMode:
        if self.magnetization_vector is not None:
            return "noncollinear"
        if self.magnetization is not None:
            return "collinear"
        return "none"

    @property
    def lattice(self) -> Lattice:
        return self.structure.lattice

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.rho.shape

    def with_options(self, options: Any) -> "VolumetricData":
        """A view with different ``options``, sharing this object's grids and cache."""
        return dataclasses.replace(self, options=options)

    def astype(self, dtype: Any) -> "VolumetricData":
        """Copy with every grid cast to ``dtype`` (float32 mode, spec §13); fresh cache."""
        def cast(g: Optional[Grid]) -> Optional[Grid]:
            return None if g is None else g.astype(dtype)
        mv = None if self.magnetization_vector is None else (
            self.magnetization_vector[0].astype(dtype), self.magnetization_vector[1].astype(dtype),
            self.magnetization_vector[2].astype(dtype))
        return dataclasses.replace(
            self, rho=self.rho.astype(dtype), magnetization=cast(self.magnetization),
            elf=cast(self.elf), potential=cast(self.potential), core_density=cast(self.core_density),
            magnetization_vector=mv, cache={})


def species_labels(tokens: Sequence[str], counts: Sequence[int]) -> list[str]:
    """Expand per-group element labels ('Fe_pv', 'O/') to one clean symbol per atom."""
    clean = [t.split("_")[0].split("/")[0] for t in tokens]
    return [s for s, c in zip(clean, counts) for _ in range(int(c))]


def expand_species(labels: Sequence[str], counts: Sequence[int]) -> list[str]:
    """``labels`` is one label per group or one per atom; return one per atom."""
    labels = list(labels)
    if len(labels) == len(counts):
        return species_labels(labels, counts)
    if len(labels) == int(sum(counts)):
        return [str(s) for s in labels]
    raise ValueError(f"{len(labels)} species labels for {len(counts)} groups / "
                     f"{int(sum(counts))} atoms")


def check_mapping(m: Optional[Mapping[str, float]], elements: Sequence[str], what: str) -> None:
    if m is None:
        return
    missing = [e for e in elements if e not in m]
    if missing:
        raise KeyError(f"{what}: no value for element(s) {missing}")
