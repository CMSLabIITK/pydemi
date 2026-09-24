"""
pydemi.descriptors.registry
===========================
Descriptor registration and metadata (spec §8), the options ``featurize``
takes, and the shared accessors descriptors use.

Every descriptor is a function ``f(vd) -> float`` registered with

    @register(name="zeta", domain="bonding", field="rho",
              requires=["gradient", "geometry"], units="dimensionless",
              range=(0.0, 1.0), intensive=True,
              sentinel_cases={"uniform_density": 0.0}, references=[...])

The docstring carries the formula in the notation of the specification and
the analytic limiting values where known; :func:`catalogue` turns the
registry into a DataFrame, so the SI descriptor tables are generated from
code.

Sentinels
---------
A descriptor that is undefined for a degenerate input (0/0 for a uniform
density, no voxels in a region, ...) returns ``Sentinel(value, case)``: the
documented constant ``value`` for the documented ``case``. ``featurize``
reports the value and sets the companion flag ``<name>__flag`` -- never a
bare NaN. The only NaN allowed is a within-element variance when no element
has two sites, which comes with the per-element site counts (spec §10).

Intensivity
-----------
Every registered descriptor must be intensive: unchanged when the cell is
replaced by a supercell of the same material. Extensive quantities are
normalized (per volume, per atom, or by a total) before registration;
the raw values appear only as metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from ..constants import (DERIVATIVE_BACKEND, FD_ORDER, LAPLACIAN_METHOD, NNM_R_CUT, SHELL_C1,
                         SHELL_C2)
from ..core.geometry import NearestAtom, ShellMasks, Shells, geometry_of, shells_of
from ..fields.density import Derivatives, abs_m, derivatives, rho
from ..io.base import FloatArray, VolumetricData

DOMAINS = ("bonding", "structural", "magnetic", "heterogeneity", "compositional")
EXTENSIONS = ("paw",)
PARTITIONS = ("nearest", "power", "becke", "hirshfeld")
DEFORMATION_REFERENCES = ("auto", "aeccar0", "tabulated", "custom")
ELF_SOURCES = ("auto", "reconstruct", "file")
POTENTIAL_SOURCES = ("auto", "locpot", "hartree", "esp")


# ----------------------------------------------------------------------
# options
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureOptions:
    """Settings of one ``featurize`` call; recorded in the output metadata."""

    domains: tuple[str, ...] = DOMAINS
    partition: str = "nearest"
    shells: Shells = Shells(SHELL_C1, SHELL_C2)
    deformation_reference: str = "auto"
    custom_reference: Optional[str] = None
    elf_source: str = "auto"
    potential_source: str = "auto"
    laplacian_method: str = LAPLACIAN_METHOD
    derivative_backend: str = DERIVATIVE_BACKEND
    fd_order: int = FD_ORDER
    nnm_r_cut: float = NNM_R_CUT
    extensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        def check(value: str, allowed: Sequence[str], what: str) -> None:
            if value not in allowed:
                raise ValueError(f"{what} must be one of {list(allowed)}, got {value!r}")
        for d in self.domains:
            check(d, DOMAINS, "domain")
        for e in self.extensions:
            check(e, EXTENSIONS, "extension")
        check(self.partition, PARTITIONS, "partition")
        check(self.deformation_reference, DEFORMATION_REFERENCES, "deformation_reference")
        check(self.elf_source, ELF_SOURCES, "elf_source")
        check(self.potential_source, POTENTIAL_SOURCES, "potential_source")
        check(self.laplacian_method, ("metric", "diagonal"), "laplacian_method")
        check(self.derivative_backend, ("fft", "fd"), "derivative_backend")
        if self.deformation_reference == "custom" and not self.custom_reference:
            raise ValueError("deformation_reference='custom' needs custom_reference=<directory>")


def make_options(shells: Union[Shells, tuple[float, float], None] = None,
                 **kwargs: Any) -> FeatureOptions:
    if shells is None:
        sh = Shells()
    elif isinstance(shells, Shells):
        sh = shells
    else:
        sh = Shells(float(shells[0]), float(shells[1]))
    if "domains" in kwargs and kwargs["domains"] is not None:
        d = kwargs["domains"]
        kwargs["domains"] = tuple(d.split(",")) if isinstance(d, str) else tuple(d)
    else:
        kwargs.pop("domains", None)
    if "extensions" in kwargs:
        e = kwargs["extensions"] or ()
        kwargs["extensions"] = tuple(e.split(",")) if isinstance(e, str) else tuple(e)
    return FeatureOptions(shells=sh, **kwargs)


def options(vd: VolumetricData) -> FeatureOptions:
    """The options attached to ``vd`` by ``featurize`` (defaults otherwise)."""
    o = vd.options
    return o if isinstance(o, FeatureOptions) else FeatureOptions()


# ----------------------------------------------------------------------
# sentinels and registration
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Sentinel:
    """A documented constant returned for a documented degenerate case."""

    value: float
    case: str


Result = Union[float, Sentinel]
DescriptorFn = Callable[[VolumetricData], Result]


@dataclass(frozen=True)
class DescriptorSpec:
    name: str
    domain: str
    field: str
    requires: tuple[str, ...]
    units: str
    range: tuple[float, float]
    intensive: bool
    sentinel_cases: Mapping[str, float]
    references: tuple[str, ...]
    adopted: bool
    extension: Optional[str]
    func: DescriptorFn = field(compare=False)
    doc: str = ""

    @property
    def formula(self) -> str:
        """First paragraph of the docstring: the formula in spec notation."""
        return self.doc.strip().split("\n\n")[0].replace("\n", " ").strip()


REGISTRY: dict[str, DescriptorSpec] = {}


def register(name: str, domain: str, field: str, requires: Iterable[str], units: str,
             range: tuple[float, float] = (-np.inf, np.inf), intensive: bool = True,
             sentinel_cases: Optional[Mapping[str, float]] = None,
             references: Iterable[str] = (), adopted: bool = False,
             extension: Optional[str] = None, doc: Optional[str] = None
             ) -> Callable[[DescriptorFn], DescriptorFn]:
    """Register a descriptor function (decorator)."""
    if domain not in DOMAINS:
        raise ValueError(f"{name}: unknown domain {domain!r}")
    if not intensive:
        raise ValueError(f"{name}: only intensive quantities may enter a feature vector; "
                         "normalize it (per volume, per atom, or by a total) first")
    if extension is not None and extension not in EXTENSIONS:
        raise ValueError(f"{name}: unknown extension {extension!r}")

    def deco(fn: DescriptorFn) -> DescriptorFn:
        if name in REGISTRY:
            raise ValueError(f"descriptor {name!r} registered twice")
        REGISTRY[name] = DescriptorSpec(
            name=name, domain=domain, field=field, requires=tuple(requires), units=units,
            range=(float(range[0]), float(range[1])), intensive=intensive,
            sentinel_cases=dict(sentinel_cases or {}), references=tuple(references),
            adopted=adopted, extension=extension, func=fn,
            doc=doc if doc is not None else (fn.__doc__ or ""))
        return fn
    return deco


def selected(opts: FeatureOptions) -> list[DescriptorSpec]:
    """Registered descriptors for the requested domains and extensions, in registration order."""
    return [s for s in REGISTRY.values()
            if s.domain in opts.domains and (s.extension is None or s.extension in opts.extensions)]


def finite_or(value: float, sentinel: float, case: str) -> Result:
    """``value`` if finite, else the flagged sentinel."""
    return value if np.isfinite(value) else Sentinel(sentinel, case)


# ----------------------------------------------------------------------
# shared accessors (everything cached on vd)
# ----------------------------------------------------------------------

def geometry(vd: VolumetricData) -> NearestAtom:
    return geometry_of(vd)


def masks(vd: VolumetricData) -> ShellMasks:
    return shells_of(vd, options(vd).shells)


FieldFn = Callable[[VolumetricData], FloatArray]
_FIELDS: dict[str, FieldFn] = {"rho": rho, "abs_m": abs_m}


def register_field(name: str, fn: FieldFn) -> None:
    _FIELDS[name] = fn


def field_values(vd: VolumetricData, name: str) -> FloatArray:
    try:
        return _FIELDS[name](vd)
    except KeyError:
        raise KeyError(f"unknown field {name!r}; known: {sorted(_FIELDS)}") from None


def field_derivatives(vd: VolumetricData, name: str) -> Derivatives:
    o = options(vd)
    backend = "fft" if o.derivative_backend == "fft" else "fd"
    return derivatives(vd, name, _FIELDS[name], backend, o.fd_order)


def laplacian(vd: VolumetricData, name: str = "rho") -> FloatArray:
    method = "metric" if options(vd).laplacian_method == "metric" else "diagonal"
    return field_derivatives(vd, name).laplacian(method)


def is_uniform(values: FloatArray) -> bool:
    """True when max - min <= UNIFORM_TOL max |f| (or the field is zero)."""
    from ..constants import UNIFORM_TOL
    top = float(np.max(np.abs(values)))
    return top == 0.0 or float(np.ptp(values)) <= UNIFORM_TOL * top


# ----------------------------------------------------------------------
# metadata hooks: always evaluated, whatever domains were requested
# ----------------------------------------------------------------------

MetadataFn = Callable[[VolumetricData], dict[str, Any]]
METADATA_HOOKS: list[MetadataFn] = []


def metadata_hook(fn: MetadataFn) -> MetadataFn:
    METADATA_HOOKS.append(fn)
    return fn


# ----------------------------------------------------------------------
# PAW augmentation radii (for the ``paw`` extension)
# ----------------------------------------------------------------------

def augmentation_radii(vd: VolumetricData) -> tuple["np.ndarray", str]:
    """(per-atom PAW augmentation radius R_PAW in Angstrom, source).

    RCORE from the POTCAR / OUTCAR (``source="potcar"``) or, when unknown, the
    covalent radius (``source="covalent"``). Inside R_PAW a VASP CHGCAR is
    pseudized; outside it equals the all-electron valence density.
    """
    from ..data import covalent_radius
    if vd.paw_radii is not None:
        return np.array([vd.paw_radii[s] for s in vd.structure.species]), "potcar"
    return np.array([covalent_radius(s) for s in vd.structure.species]), "covalent"
