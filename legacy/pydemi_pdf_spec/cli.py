"""
pydemi command line.

    pydemi compute RUN_DIR_OR_ROOT ... -o descriptors.csv [--families tier1 F2 ...] [--workers 4]
    pydemi list [--family F] [--kind K]
    pydemi describe NAME
    pydemi convergence RUN_DIR_OR_FILE [--factors 1 0.8 0.6] [--rtol 0.02]
    pydemi resample IN OUT --spacing 0.1
"""

import argparse
import sys

from .descriptors import FAMILIES, REGISTRY, describe, names
from .descriptors.registry import SCALAR_KINDS


def _cmd_compute(a):
    from .batch import run_batch
    counts = run_batch(a.inputs, a.output, families=a.families, workers=a.workers,
                       resume=not a.no_resume, ionicity_cal=a.ionicity_cal,
                       bulk_cal=a.bulk_cal, progress=not a.quiet)
    print(f"done {counts['done']} (failed {counts['failed']}), skipped {counts['skipped']} "
          f"-> {a.output}")
    return 1 if counts["failed"] else 0


def _cmd_list(a):
    kinds = None if a.all_kinds else (tuple(a.kind) if a.kind else SCALAR_KINDS)
    for n in names(family=a.family, kinds=kinds, opt_in=True):
        i = REGISTRY[n]
        entry = "" if i.entry is None else str(i.entry)
        print(f"{entry:>4} {i.family:6s} {i.kind:13s} {n}")
    return 0


def _cmd_describe(a):
    i = describe(a.name)
    for k in ("name", "entry", "family", "kind", "inputs", "formula", "note", "opt_in"):
        print(f"{k:8s} {getattr(i, k)}")
    return 0


def _load(path):
    from pathlib import Path
    from .engine import Engine
    p = Path(path)
    return Engine.from_vasp_dir(p) if p.is_dir() else Engine.from_file(p)


def _cmd_convergence(a):
    from .convergence import convergence_report, format_report
    rep = convergence_report(_load(a.input), factors=a.factors, rtol=a.rtol)
    print(format_report(rep, limit=a.limit))
    return 0


def _cmd_resample(a):
    from .io.vasp import write_volumetric
    from .resample import resample_engine
    eng = resample_engine(_load(a.input), spacing=a.spacing)
    field = "rho" if "rho" in eng else eng.field_names[0]
    write_volumetric(a.output, eng.structure, [eng[field].values])
    print(f"{field}: {eng[field].grid.shape} -> {a.output}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pydemi", description="Charge-density descriptors.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compute", help="descriptors for run directories or density files")
    c.add_argument("inputs", nargs="+")
    c.add_argument("-o", "--output", default="descriptors.csv")
    c.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    c.add_argument("--workers", type=int, default=1)
    c.add_argument("--no-resume", action="store_true")
    c.add_argument("--ionicity-cal", help="IonicityCalibration JSON for Family D")
    c.add_argument("--bulk-cal", help="BulkModulusCalibration JSON for entry 106")
    c.add_argument("-q", "--quiet", action="store_true")
    c.set_defaults(func=_cmd_compute)

    l = sub.add_parser("list", help="registered descriptors")
    l.add_argument("--family")
    l.add_argument("--kind", nargs="+")
    l.add_argument("--all-kinds", action="store_true", help="include field/site/dataset entries")
    l.set_defaults(func=_cmd_list)

    d = sub.add_parser("describe", help="formula, inputs and caveats of one descriptor")
    d.add_argument("name")
    d.set_defaults(func=_cmd_describe)

    v = sub.add_parser("convergence", help="per-descriptor grid-convergence report")
    v.add_argument("input")
    v.add_argument("--factors", nargs="+", type=float, default=[1.0, 0.8, 0.6])
    v.add_argument("--rtol", type=float, default=0.02)
    v.add_argument("--limit", type=int, default=60)
    v.set_defaults(func=_cmd_convergence)

    r = sub.add_parser("resample", help="Fourier-resample a density to a grid spacing")
    r.add_argument("input")
    r.add_argument("output")
    r.add_argument("--spacing", type=float, required=True, help="Angstrom")
    r.set_defaults(func=_cmd_resample)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
