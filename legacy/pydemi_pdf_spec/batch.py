"""
pydemi.batch
------------
Descriptors for many structures, in parallel, resumable, written to CSV
one row at a time.

Each input is a VASP run directory (CHGCAR plus whichever of AECCAR0/2,
ELFCAR, LOCPOT, POTCAR exist) or a single density file (CHGCAR, .cube,
.xsf). ``material_id`` is the directory name (or file stem). A failure is
recorded in the ``error`` column instead of stopping the batch; with
``resume=True`` IDs already in the CSV are skipped, so an interrupted run
continues where it stopped.
"""

import csv
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .descriptors import FAMILIES, REGISTRY, compute_descriptors, names

DENSITY_FILES = ("CHGCAR",)


def find_runs(root) -> list:
    """Directories under ``root`` (inclusive) that contain a CHGCAR, sorted."""
    root = Path(root)
    return sorted({p.parent for name in DENSITY_FILES for p in root.rglob(name)})


def material_id(path) -> str:
    path = Path(path)
    return path.name if path.is_dir() else path.stem


def _load(path):
    from .engine import Engine
    path = Path(path)
    return Engine.from_vasp_dir(path) if path.is_dir() else Engine.from_file(path)


def _calibrations(ionicity_cal, bulk_cal):
    from .calibration import BulkModulusCalibration, IonicityCalibration
    cal = {}
    if ionicity_cal:
        cal["ionicity"] = IonicityCalibration.load(ionicity_cal)
    if bulk_cal:
        cal["bulk"] = BulkModulusCalibration.load(bulk_cal)
    return cal


def compute_one(path, families: Sequence[str] = FAMILIES,
                ionicity_cal: Optional[str] = None, bulk_cal: Optional[str] = None) -> dict:
    """One CSV row: material_id, path, error, then every descriptor."""
    row = {"material_id": material_id(path), "path": str(path), "error": ""}
    try:
        eng = _load(path)
        row.update(compute_descriptors(eng, families=families,
                                       calibrations=_calibrations(ionicity_cal, bulk_cal)))
    except Exception as exc:   # recorded, the batch goes on
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc(limit=3)
    return row


def columns(families: Sequence[str] = FAMILIES) -> list:
    return ["material_id", "path", "error"] + [n for n in names(opt_in=False)
                                               if REGISTRY[n].family in families]


def done_ids(out_csv) -> set:
    p = Path(out_csv)
    if not p.exists():
        return set()
    with p.open(newline="") as fh:
        return {row["material_id"] for row in csv.DictReader(fh)}


def run_batch(inputs: Iterable, out_csv, families: Sequence[str] = FAMILIES,
              workers: int = 1, resume: bool = True, ionicity_cal: Optional[str] = None,
              bulk_cal: Optional[str] = None, progress: bool = True) -> dict:
    """Compute every input and append rows to ``out_csv``.

    ``inputs``: paths (run directories or density files); a single root
    directory is expanded with :func:`find_runs`. Returns counts of done,
    skipped and failed inputs.
    """
    inputs = [Path(p) for p in inputs]
    if len(inputs) == 1 and inputs[0].is_dir() and not (inputs[0] / "CHGCAR").exists():
        inputs = find_runs(inputs[0])
    families = tuple(families)
    header = columns(families)
    skip = done_ids(out_csv) if resume else set()
    todo = [p for p in inputs if material_id(p) not in skip]
    out = Path(out_csv)
    new_file = not out.exists() or not resume
    counts = {"done": 0, "skipped": len(inputs) - len(todo), "failed": 0}

    with out.open("w" if new_file else "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        if new_file:
            writer.writeheader()

        def record(row):
            writer.writerow(row)
            fh.flush()
            counts["done"] += 1
            counts["failed"] += bool(row["error"])
            if progress:
                status = "FAILED " + row["error"] if row["error"] else "ok"
                print(f"[{counts['done']}/{len(todo)}] {row['material_id']}: {status}", flush=True)

        args = (families, ionicity_cal, bulk_cal)
        if workers <= 1:
            for p in todo:
                record(compute_one(p, *args))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(compute_one, p, *args) for p in todo]
                for fut in as_completed(futures):
                    record(fut.result())
    return counts
