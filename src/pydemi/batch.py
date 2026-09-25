"""
pydemi.batch
============
Featurize many structures in parallel (spec §12, §13): one process per
structure via ``concurrent.futures.ProcessPoolExecutor``, never per
descriptor.

``featurize_batch`` returns a tidy DataFrame -- one row per structure,
descriptor columns in registry order, then metadata columns (``n_atoms``,
``volume``, ``grid_shape``, ``density_source``, ``magnetic``,
``euler_consistency``, ``wall_time_s``, ``pydemi_version``, ``error``, ...).
Errors go into the frame (``on_error="record"``); a 6,000-structure run never
crashes on one bad file.
"""

from __future__ import annotations

import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Optional, Union

PathLike = Union[str, Path]
COMPANIONS = {"elf": "ELFCAR", "locpot": "LOCPOT", "aeccar0": "AECCAR0", "aeccar2": "AECCAR2"}


def _read(path: Path, companions: bool, read_options: Optional[dict[str, Any]] = None) -> Any:
    from .io.registry import read, sniff
    if path.is_dir():
        path = path / "CHGCAR"
    is_vasp = sniff(path) == "chgcar"
    kwargs: dict[str, Any] = dict(read_options or {}) if is_vasp else {}
    if companions and is_vasp:
        for key, name in COMPANIONS.items():
            if (path.parent / name).exists():
                kwargs[key] = path.parent / name
        if ("aeccar0" in kwargs) != ("aeccar2" in kwargs):
            kwargs.pop("aeccar0", None)
            kwargs.pop("aeccar2", None)
    return read(path, **kwargs)


def _one(path: str, companions: bool, options: dict[str, Any],
         read_options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    from . import __version__
    from .descriptors import featurize
    t0 = time.perf_counter()
    try:
        vd = _read(Path(path), companions, read_options)
        feats, meta = featurize(vd, return_metadata=True, **options)
        return {"path": path, **feats, **meta}
    except Exception as exc:                                   # noqa: BLE001 -- recorded
        return {"path": path, "error": f"{type(exc).__name__}: {exc}",
                "wall_time_s": time.perf_counter() - t0, "pydemi_version": __version__,
                "traceback": traceback.format_exc(limit=3)}


def featurize_batch(paths: Iterable[PathLike], n_workers: int = 8, on_error: str = "record",
                    progress: bool = True, companions: bool = False,
                    read_options: Optional[dict[str, Any]] = None,
                    **featurize_kwargs: Any) -> Any:
    """Featurize every structure in ``paths`` (CHGCAR / cube / xsf files or run directories).

    ``on_error``: "record" (error message in the ``error`` column, descriptors
    NaN), "raise" (stop at the first failure) or "skip" (drop the row).
    ``companions=True`` also reads ELFCAR, LOCPOT and AECCAR0/2 found next to
    each CHGCAR; the default reads the density only, so every row is computed
    from the same inputs. ``read_options`` go to :func:`pydemi.read_vasp` for
    VASP files -- e.g. ``{"zval": {...}, "paw_radii": {...}}`` tables for runs
    without a POTCAR or OUTCAR. Other keyword arguments go to
    :func:`pydemi.featurize`.
    """
    import pandas as pd

    from .descriptors import descriptor_names
    if on_error not in ("record", "raise", "skip"):
        raise ValueError(f"on_error must be record, raise or skip; got {on_error!r}")
    items = [str(p) for p in paths]
    rows: list[Optional[dict[str, Any]]] = [None] * len(items)
    workers = max(1, int(n_workers))

    def handle(i: int, row: dict[str, Any], done: int) -> None:
        if row.get("error"):
            if on_error == "raise":
                raise RuntimeError(f"{items[i]}: {row['error']}\n{row.get('traceback', '')}")
        rows[i] = None if (row.get("error") and on_error == "skip") else row
        if progress:
            status = "ok" if not row.get("error") else f"error: {row['error']}"
            print(f"[{done}/{len(items)}] {items[i]}: {status} ({row.get('wall_time_s', 0):.1f} s)",
                  file=sys.stderr, flush=True)

    if workers == 1:
        for i, p in enumerate(items):
            handle(i, _one(p, companions, featurize_kwargs, read_options), i + 1)
    else:
        with ProcessPoolExecutor(workers) as pool:
            futures = {pool.submit(_one, p, companions, featurize_kwargs, read_options): i
                       for i, p in enumerate(items)}
            for done, fut in enumerate(as_completed(futures), start=1):
                handle(futures[fut], fut.result(), done)

    names = descriptor_names(featurize_kwargs.get("domains"), featurize_kwargs.get("extensions", ()))
    kept = [r for r in rows if r is not None]
    df = pd.DataFrame(kept)
    for n in names:
        if n not in df.columns:
            df[n] = float("nan")
    if "error" not in df.columns:
        df["error"] = ""
    df["error"] = df["error"].fillna("")
    meta = [c for c in df.columns if c not in names and c not in ("path", "traceback")]
    return df[["path"] + names + meta].reset_index(drop=True)


def find_runs(root: PathLike, glob: str = "*/CHGCAR") -> list[Path]:
    """Density files under ``root`` matching ``glob``, sorted."""
    return sorted(Path(root).glob(glob))
