"""
pydemi.engine
-------------
``Engine`` ties one structure to its named grid fields and caches the
shared primitives -- grid, nearest-atom geometry, shell masks, partition
-- per grid shape. Fields read from different VASP files may live on
different grids (ELFCAR is usually on the coarse NGX grid), so every
cache is keyed by shape.

Standard field names
--------------------
``rho``                total (pseudo-valence + augmentation) density, CHGCAR
``magnetization``      collinear m = rho_up - rho_down (CHGCAR block 2)
``magnetization_abs``  |m| (collinear) or |m_vec| (non-collinear)
``rho_ae``             all-electron density, AECCAR0 + AECCAR2
``elf``                ELF (ELFCAR block 1); ``elf_down`` for block 2
``potential``          LOCPOT, eV
"""

from pathlib import Path
from typing import Optional

import numpy as np

from .field import Field
from .elements import covalent_radii
from .geometry import NearestAtom, assign_atoms, nearest_atom
from .grid import Grid
from .partition import BeckePartition, HardPartition, HirshfeldPartition, Partition
from .io.vasp import (SPIN_COLLINEAR, SPIN_NONCOLLINEAR, SPIN_NONE,
                      ChargeDensity, read_aeccar, read_chgcar, read_elfcar,
                      read_locpot)
from .shells import ShellMasks, Shells
from .structure import Structure

RHO = "rho"
MAG = "magnetization"
MAG_ABS = "magnetization_abs"
RHO_AE = "rho_ae"
ELF = "elf"
ELF_DOWN = "elf_down"
POT = "potential"


class Engine:
    def __init__(self, structure: Structure, fields: Optional[dict] = None,
                 method: str = "fd", shells: Optional[Shells] = None,
                 spin_mode: str = SPIN_NONE,
                 magnetization_vector: Optional[np.ndarray] = None,
                 zval: Optional[dict] = None, reference=None):
        self.structure = structure
        # element -> valence electron count of the PAW dataset (POTCAR ZVAL);
        # None falls back to pydemi.atoms.reference.default_zval
        self.zval = dict(zval) if zval else None
        # free-atom reference densities; None -> the shared LDA reference
        self._reference = reference
        # element -> PAW augmentation radius (RCORE, Angstrom) when known; inside
        # it the CHGCAR density is pseudized
        self.paw_radii = None
        self.method = method
        self.shells = shells if shells is not None else Shells()
        self.spin_mode = spin_mode
        self.magnetization_vector = magnetization_vector
        self._fields = {}
        self._grids = {}
        self._geometry = {}
        self._shell_masks = {}
        self._partitions = {}
        # derived quantities shared between descriptor families, keyed by
        # tuples (kind, source_field, ...), e.g. ("kinetic", "rho")
        self.cache = {}
        for name, values in (fields or {}).items():
            self.add_field(name, values)

    def __repr__(self):
        return (f"Engine({self.structure!r}, fields={list(self._fields)}, "
                f"spin={self.spin_mode!r}, method={self.method!r})")

    # ---------------- fields ----------------

    def add_field(self, name: str, values) -> Field:
        values = np.asarray(values, dtype=float)
        # anything derived from a replaced field is stale
        self.cache = {k: v for k, v in self.cache.items() if name not in k}
        field = Field(values, self.grid(values.shape), name=name, method=self.method)
        self._fields[name] = field
        return field

    def __getitem__(self, name: str) -> Field:
        try:
            return self._fields[name]
        except KeyError:
            raise KeyError(f"no field {name!r}; available: {list(self._fields)}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._fields

    @property
    def field_names(self) -> list:
        return list(self._fields)

    # ---------------- shared primitives (cached per shape) ----------------

    def _default_shape(self, shape):
        if shape is not None:
            return tuple(shape)
        if RHO in self._fields:
            return self._fields[RHO].grid.shape
        if self._fields:
            return next(iter(self._fields.values())).grid.shape
        raise ValueError("engine has no fields; pass a grid shape")

    def grid(self, shape=None) -> Grid:
        shape = self._default_shape(shape)
        if shape not in self._grids:
            self._grids[shape] = Grid(self.structure.lattice, shape)
        return self._grids[shape]

    def geometry(self, shape=None) -> NearestAtom:
        shape = self._default_shape(shape)
        if shape not in self._geometry:
            self._geometry[shape] = nearest_atom(self.grid(shape), self.structure)
        return self._geometry[shape]

    def shell_masks(self, shape=None, shells: Optional[Shells] = None) -> ShellMasks:
        shape = self._default_shape(shape)
        shells = shells if shells is not None else self.shells
        key = (shape, shells.c1, shells.c2,
               None if shells.radii is None else tuple(sorted(shells.radii.items())))
        if key not in self._shell_masks:
            self._shell_masks[key] = shells.masks(self.geometry(shape), self.structure)
        return self._shell_masks[key]

    def atom_radii(self, radii=None) -> np.ndarray:
        """Per-atom radii (Angstrom) from an element -> radius map, or Magpie
        covalent radii when ``radii`` is None."""
        table = covalent_radii(self.structure.elements) if radii is None else radii
        missing = [e for e in self.structure.elements if e not in table]
        if missing:
            raise KeyError(f"no radius for element(s) {missing}")
        return np.array([table[s] for s in self.structure.species], float)

    def partition(self, shape=None, scheme: str = "nearest", radii=None,
                  k: int = 60, cells: int = 8, r_cut: float = 6.5,
                  part: str = "total") -> Partition:
        """Voxel-to-atom partition: "nearest", "power", "becke" or "hirshfeld".

        ``radii`` (element -> Angstrom) sets the power-diagram weights (default
        Magpie covalent radii) and, for Becke, enables the heteronuclear size
        adjustment (default off: plain Becke is pure geometry). For Becke,
        ``cells`` nearest images receive weight and each product runs over
        the ``k`` nearest images; see :mod:`pydemi.partition` on why ``k``
        must be reported. Hirshfeld weights use the engine's free-atom
        reference from every atom image within ``r_cut`` Angstrom; ``part``
        ("total" or "valence") must match the electrons of the partitioned
        field, or even a pure free-atom superposition gets nonzero charges
        (the descriptor functions choose it from the field).
        """
        shape = self._default_shape(shape)
        n = self.structure.n_atoms
        if scheme == "nearest":
            key = (shape, scheme)
        elif scheme == "power":
            r = self.atom_radii(radii)
            key = (shape, scheme, tuple(r))
        elif scheme == "becke":
            r = None if radii is None else self.atom_radii(radii)
            key = (shape, scheme, None if r is None else tuple(r), int(k), int(cells))
        elif scheme == "hirshfeld":
            key = (shape, scheme, id(self.reference), float(r_cut), part)
        else:
            raise ValueError(f"scheme must be nearest, power, becke or hirshfeld; got {scheme!r}")
        if key not in self._partitions:
            if scheme == "nearest":
                built = HardPartition(self.geometry(shape), n, "nearest")
            elif scheme == "power":
                built = HardPartition(assign_atoms(self.grid(shape), self.structure, r), n, "power")
            elif scheme == "becke":
                built = BeckePartition(self.grid(shape), self.structure, k=k, cells=cells, radii=r)
            else:
                zval = self.valence()
                radial = {e: self.reference.radial(e, part, zval.get(e))
                          for e in self.structure.elements}
                built = HirshfeldPartition(self.grid(shape), self.structure, radial, r_cut=r_cut)
            self._partitions[key] = built
        return self._partitions[key]

    # ---------------- free-atom reference ----------------

    @property
    def reference(self):
        if self._reference is None:
            from .atoms.reference import default_reference
            self._reference = default_reference()
        return self._reference

    def valence(self) -> dict:
        """Element -> ZVAL, from POTCAR / explicit values or the default rule."""
        from .atoms.reference import resolve_zval
        return resolve_zval(self.structure.elements, self.zval)

    # ---------------- constructors ----------------

    @classmethod
    def from_charge_density(cls, cd: ChargeDensity, **kwargs) -> "Engine":
        eng = cls(cd.structure, spin_mode=cd.spin_mode, **kwargs)
        eng.add_field(RHO_AE if cd.all_electron else RHO, cd.total)
        if cd.spin_mode == SPIN_COLLINEAR:
            eng.add_field(MAG, cd.magnetization)
            eng.add_field(MAG_ABS, np.abs(cd.magnetization))
        elif cd.spin_mode == SPIN_NONCOLLINEAR:
            eng.magnetization_vector = cd.magnetization
            eng.add_field(MAG_ABS, np.linalg.norm(cd.magnetization, axis=0))
        return eng

    @classmethod
    def from_chgcar(cls, path, species=None, read_spin: bool = True, **kwargs) -> "Engine":
        return cls.from_charge_density(read_chgcar(path, species, read_spin), **kwargs)

    @classmethod
    def from_file(cls, path, density_unit: Optional[str] = "e/bohr^3", **kwargs) -> "Engine":
        """Engine from a CHGCAR-style file, or a .cube / .xsf density
        (Quantum ESPRESSO pp.x, ABINIT cut3d; see :mod:`pydemi.io.grids`)."""
        if Path(path).suffix.lower() in (".cube", ".cub", ".xsf"):
            from .io.grids import read_density
            return cls.from_charge_density(read_density(path, density_unit), **kwargs)
        return cls.from_chgcar(path, **kwargs)

    @classmethod
    def from_vasp_dir(cls, directory, species=None, **kwargs) -> "Engine":
        """Load CHGCAR plus whichever of AECCAR0/2, ELFCAR, LOCPOT exist.

        The valence electron counts of the free-atom reference are the ZVAL
        values (one per species group, in file order) of the POTCAR, or of
        the OUTCAR when there is no POTCAR.
        """
        d = Path(directory)
        eng = cls.from_chgcar(d / "CHGCAR", species=species, **kwargs)
        source = next((d / n for n in ("POTCAR", "OUTCAR") if (d / n).exists()), None)
        if source is not None:
            from .atoms.reference import read_potcar_rcore, read_potcar_zval
            from .constants import BOHR_ANGSTROM
            elements = eng.structure.elements
            zvals = read_potcar_zval(source)
            if "zval" not in kwargs:
                if len(zvals) == len(elements):
                    eng.zval = dict(zip(elements, zvals))
                else:
                    import warnings
                    warnings.warn(f"{source.name} has {len(zvals)} ZVAL entries for "
                                  f"{len(elements)} elements; ignoring it", stacklevel=2)
            rcore = read_potcar_rcore(source)
            if len(rcore) == len(elements):
                eng.paw_radii = {e: r * BOHR_ANGSTROM for e, r in zip(elements, rcore)}

        def _check(struct, name):
            if not np.allclose(struct.lattice, eng.structure.lattice, atol=1e-6):
                raise ValueError(f"{d / name}: lattice differs from CHGCAR")

        if (d / "AECCAR0").exists() and (d / "AECCAR2").exists():
            ae = read_aeccar(d / "AECCAR0", d / "AECCAR2", species)
            _check(ae.structure, "AECCAR2")
            eng.add_field(RHO_AE, ae.total)
        if (d / "ELFCAR").exists():
            elf = read_elfcar(d / "ELFCAR", species)
            _check(elf.structure, "ELFCAR")
            eng.add_field(ELF, elf.blocks[0])
            if len(elf.blocks) > 1:
                eng.add_field(ELF_DOWN, elf.blocks[1])
        if (d / "LOCPOT").exists():
            pot = read_locpot(d / "LOCPOT", species)
            _check(pot.structure, "LOCPOT")
            eng.add_field(POT, pot.blocks[0])
        return eng
