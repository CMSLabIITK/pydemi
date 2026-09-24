"""
pydemi.descriptors
==================
The descriptor registry, :func:`featurize` and :func:`catalogue` (spec §8, §12).

Importing this package registers every descriptor of the five domains
(bonding, structural, magnetic, heterogeneity, compositional) and the
off-by-default PAW extension.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence, Union

import numpy as np

from ..core.geometry import Shells
from ..io.base import VolumetricData
from . import bonding, structural, magnetic, heterogeneity  # noqa: F401  (register descriptors)
from .registry import (METADATA_HOOKS, REGISTRY, DescriptorSpec, FeatureOptions, Sentinel,
                       make_options, selected)

__all__ = ["featurize", "catalogue", "descriptor_names", "REGISTRY", "FeatureOptions"]


def _version() -> str:
    from .. import __version__
    return __version__


def featurize(vd: VolumetricData, domains: Union[Sequence[str], str, None] = None,
              partition: str = "nearest",
              shells: Union[Shells, tuple[float, float], None] = None,
              deformation_reference: str = "auto", elf_source: str = "auto",
              laplacian_method: str = "metric", derivative_backend: str = "fft",
              fd_order: int = 4, potential_source: str = "auto",
              custom_reference: Optional[str] = None,
              extensions: Union[Sequence[str], str] = (), float32: bool = False,
              return_metadata: bool = False) -> Any:
    """Named, fixed-length descriptor vector of one structure.

    Returns ``dict[str, float]``: one entry per registered descriptor of the
    requested ``domains`` (all five by default) and ``extensions`` (none by
    default; ``"paw"`` adds the PAW-aware descriptors). With
    ``return_metadata=True`` returns ``(features, metadata)``; the metadata
    record the settings used, the density source, quality flags such as
    ``euler_consistency``, and the companion flag ``<name>__flag`` of every
    descriptor that fell back to a documented sentinel.

    ``float32=True`` casts every grid to float32 first (memory-constrained
    runs); the default is float64.
    """
    t0 = time.perf_counter()
    opts = make_options(shells, domains=domains, partition=partition,
                        deformation_reference=deformation_reference, elf_source=elf_source,
                        laplacian_method=laplacian_method, derivative_backend=derivative_backend,
                        fd_order=fd_order, potential_source=potential_source,
                        custom_reference=custom_reference, extensions=extensions)
    base = vd.astype(np.float32) if float32 else vd
    v = base.with_options(opts)
    feats: dict[str, float] = {}
    flags: dict[str, str] = {}
    for spec in selected(opts):
        result = spec.func(v)
        if isinstance(result, Sentinel):
            feats[spec.name] = float(result.value)
            flags[spec.name] = result.case
        else:
            feats[spec.name] = float(result)
    if not return_metadata:
        return feats
    meta: dict[str, Any] = {
        "n_atoms": v.structure.n_atoms,
        "volume": v.structure.volume,
        "grid_shape": "x".join(str(n) for n in v.shape),
        "density_source": v.density_source,
        "spin_mode": v.spin_mode,
        "site_counts": ",".join(f"{e}:{n}" for e, n in v.structure.site_counts().items()),
        "partition": opts.partition,
        "shells": f"{opts.shells.c1},{opts.shells.c2}" + (",scaled" if opts.shells.scaled else ""),
        "derivative_backend": opts.derivative_backend
        + (f"{opts.fd_order}" if opts.derivative_backend == "fd" else ""),
        "laplacian_method": opts.laplacian_method,
        "precision": "float32" if float32 else "float64",
    }
    for hook in METADATA_HOOKS:
        meta.update(hook(v))
    for spec in selected(opts):
        if spec.sentinel_cases:
            meta[f"{spec.name}__flag"] = int(spec.name in flags)
    meta["sentinels"] = ";".join(f"{k}:{c}" for k, c in flags.items())
    meta["wall_time_s"] = time.perf_counter() - t0
    meta["pydemi_version"] = _version()
    meta["error"] = ""
    return feats, meta


def descriptor_names(domains: Union[Sequence[str], str, None] = None,
                     extensions: Union[Sequence[str], str] = ()) -> list[str]:
    """Names of the descriptors ``featurize`` returns for these domains / extensions."""
    return [s.name for s in selected(make_options(None, domains=domains, extensions=extensions))]


def catalogue() -> Any:
    """Descriptor metadata as a DataFrame (one row per registered descriptor)."""
    import pandas as pd

    def row(s: DescriptorSpec) -> dict[str, Any]:
        return {"name": s.name, "domain": s.domain, "field": s.field,
                "requires": ",".join(s.requires), "units": s.units,
                "range_min": s.range[0], "range_max": s.range[1], "intensive": s.intensive,
                "sentinel_cases": ";".join(f"{k}={v}" for k, v in s.sentinel_cases.items()),
                "adopted": s.adopted, "extension": s.extension or "",
                "formula": s.formula, "references": "; ".join(s.references)}
    return pd.DataFrame([row(s) for s in REGISTRY.values()])
