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

Stability
---------
``stability="fragile"`` marks a descriptor that is computed exactly as
specified but is not numerically converged on typical VASP grids: its value
changes by more than ~10% between derivative schemes or when the grid is
coarsened to 80% (measured on the 6,059-structure dataset; see
paper/analysis). Fragile descriptors stay in the default ``featurize``
output; ``descriptor_names(..., include_fragile=False)`` gives the
model-ready set without them.

Intensivity
-----------
Every registered descriptor must be intensive: unchanged when the cell is
replaced by a supercell of the same material. Extensive quantities are
normalized (per volume, per atom, or by a total) before registration;
the raw values appear only as metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Sequence, Union

import numpy as np

from ..constants import (DERIVATIVE_BACKEND, FD_ORDER, LAPLACIAN_METHOD, NNM_R_CUT, SHELL_C1,
                         SHELL_C2)
from ..core.geometry import NearestAtom, ShellMasks, Shells, geometry_of, shells_of
from ..fields.density import Derivatives, abs_m, derivatives, rho
from ..io.base import FloatArray, VolumetricData

DOMAINS = ("bonding", "structural", "magnetic", "heterogeneity", "compositional")
EXTENSIONS = ("paw", "robust")
STABILITY = ("robust", "fragile")
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
    func: DescriptorFn = dataclass_field(compare=False)
    doc: str = ""
    stability: str = "robust"

    @property
    def formula(self) -> str:
        """First paragraph of the docstring: the formula in spec notation."""
        return self.doc.strip().split("\n\n")[0].replace("\n", " ").strip()


REGISTRY: dict[str, DescriptorSpec] = {}


def register(name: str, domain: str, field: str, requires: Iterable[str], units: str,
             range: tuple[float, float] = (-np.inf, np.inf), intensive: bool = True,
             sentinel_cases: Optional[Mapping[str, float]] = None,
             references: Iterable[str] = (), adopted: bool = False,
             extension: Optional[str] = None, doc: Optional[str] = None,
             stability: str = "robust") -> Callable[[DescriptorFn], DescriptorFn]:
    """Register a descriptor function (decorator)."""
    if domain not in DOMAINS:
        raise ValueError(f"{name}: unknown domain {domain!r}")
    if not intensive:
        raise ValueError(f"{name}: only intensive quantities may enter a feature vector; "
                         "normalize it (per volume, per atom, or by a total) first")
    if extension is not None and extension not in EXTENSIONS:
        raise ValueError(f"{name}: unknown extension {extension!r}")
    if stability not in STABILITY:
        raise ValueError(f"{name}: stability must be one of {list(STABILITY)}, got {stability!r}")

    def deco(fn: DescriptorFn) -> DescriptorFn:
        if name in REGISTRY:
            raise ValueError(f"descriptor {name!r} registered twice")
        REGISTRY[name] = DescriptorSpec(
            name=name, domain=domain, field=field, requires=tuple(requires), units=units,
            range=(float(range[0]), float(range[1])), intensive=intensive,
            sentinel_cases=dict(sentinel_cases or {}), references=tuple(references),
            adopted=adopted, extension=extension, func=fn,
            doc=doc if doc is not None else (fn.__doc__ or ""), stability=stability)
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


_OPTION_FREE_FIELDS = ("rho", "abs_m", "elf_file")


def field_derivatives(vd: VolumetricData, name: str) -> Derivatives:
    """Cached derivatives of a registered field.

    A derived field (ELF_D, delta rho, ...) depends on the options that
    produced it, so its cache entry is keyed by them too: featurizing one
    object twice with, e.g., a different ``laplacian_method`` never reuses
    stale derivatives.
    """
    o = options(vd)
    backend: Literal["fft", "fd"] = "fft" if o.derivative_backend == "fft" else "fd"
    key = name if name in _OPTION_FREE_FIELDS else (
        f"{name}|{o.derivative_backend}{o.fd_order}|{o.laplacian_method}|{o.deformation_reference}"
        f"|{o.custom_reference}|{o.elf_source}|{o.potential_source}")
    return derivatives(vd, key, _FIELDS[name], backend, o.fd_order)


def laplacian(vd: VolumetricData, name: str = "rho") -> FloatArray:
    method: Literal["metric", "diagonal"] = "metric" if options(vd).laplacian_method == "metric" else "diagonal"
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

def _source_label(vd: VolumetricData, key: str, file_label: str) -> str:
    src = vd.sources.get(key)
    return "table" if src == "table" else file_label if src else "given"


def augmentation_radii(vd: VolumetricData) -> tuple[FloatArray, str]:
    """(per-atom PAW augmentation radius R_PAW in Angstrom, source).

    RCORE from the POTCAR / OUTCAR (``source="potcar"``), a per-element
    table passed to ``read_vasp`` (``"table"``) or radii set on the object
    (``"given"``); elements without one fall
    back to the covalent radius (``"covalent"``, or e.g. ``"table+covalent"``
    when only some are covered). Inside R_PAW a VASP CHGCAR is pseudized;
    outside it equals the all-electron valence density.
    """
    from ..data import covalent_radius
    known = vd.paw_radii or {}
    radii = np.array([known[s] if s in known else covalent_radius(s) for s in vd.structure.species])
    if not known:
        return radii, "covalent"
    label = _source_label(vd, "paw_radii", "potcar")
    return radii, label if set(vd.structure.elements) <= set(known) else label + "+covalent"


def zval_source(vd: VolumetricData) -> str:
    """Where the valence electron counts of a pseudo-density came from.

    "potcar" / "outcar" (the run's own file), "table" (``read_vasp(zval=)``),
    "given" (set on the object), "default" (:func:`pydemi.data.default_zval`),
    with "+default" when only some elements are covered; "not_used" for an
    all-electron density, whose reference counts are atomic numbers.
    """
    from pathlib import Path
    if vd.density_source == "all_electron":
        return "not_used"
    if not vd.zval:
        return "default"
    src = vd.sources.get("zval", "")
    name = Path(src).name.upper() if src and src != "table" else ""
    label = ("table" if src == "table" else "potcar" if name.startswith("POTCAR")
             else "outcar" if name.startswith("OUTCAR") else "given")
    return label if set(vd.structure.elements) <= set(vd.zval) else label + "+default"


@metadata_hook
def _zval_metadata(vd: VolumetricData) -> dict[str, Any]:
    return {"zval_source": zval_source(vd)}


@metadata_hook
def _origin_metadata(vd: VolumetricData) -> dict[str, Any]:
    """Whether rho was computed (DFT) or predicted by a model, and the charge rescaling."""
    return {"density_origin": vd.sources.get("origin", "dft"),
            "density_model": vd.sources.get("rho", "")[len("predicted:"):]
            if vd.sources.get("rho", "").startswith("predicted:") else "",
            "charge_scale": float(vd.sources.get("charge_scale", 1.0))}
