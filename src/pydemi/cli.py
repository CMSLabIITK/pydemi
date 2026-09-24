"""
pydemi.cli
==========
Command line (spec §12):

    pydemi featurize CHGCAR --out features.json
    pydemi batch ./runs --glob "*/CHGCAR" --out features.csv --workers 8 --domains bonding,magnetic
    pydemi catalogue --out catalogue.csv
    pydemi sweep CHGCAR --param c2 --range 1.0:2.5:0.05 --out sweep.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional, Sequence


def _options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--domains", default=None,
                   help="comma-separated: bonding,structural,magnetic,heterogeneity,compositional")
    p.add_argument("--extensions", default="", help="comma-separated, e.g. paw")
    p.add_argument("--partition", default="nearest", choices=["nearest", "power", "becke", "hirshfeld"])
    p.add_argument("--shells", default=None, help="c1,c2 in Angstrom (default 0.8,1.5)")
    p.add_argument("--deformation-reference", default="auto",
                   choices=["auto", "aeccar0", "tabulated", "custom"])
    p.add_argument("--custom-reference", default=None, help="directory for --deformation-reference custom")
    p.add_argument("--elf-source", default="auto", choices=["auto", "reconstruct", "file"])
    p.add_argument("--potential-source", default="auto", choices=["auto", "locpot", "hartree", "esp"])
    p.add_argument("--laplacian-method", default="metric", choices=["metric", "diagonal"])
    p.add_argument("--derivative-backend", default="fft", choices=["fft", "fd"])
    p.add_argument("--fd-order", type=int, default=4)
    p.add_argument("--float32", action="store_true")


def _kwargs(a: argparse.Namespace) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "domains": a.domains, "extensions": a.extensions, "partition": a.partition,
        "deformation_reference": a.deformation_reference, "custom_reference": a.custom_reference,
        "elf_source": a.elf_source, "potential_source": a.potential_source,
        "laplacian_method": a.laplacian_method, "derivative_backend": a.derivative_backend,
        "fd_order": a.fd_order, "float32": a.float32}
    if a.shells:
        c1, c2 = (float(x) for x in a.shells.split(","))
        kw["shells"] = (c1, c2)
    return kw


def _read_one(a: argparse.Namespace) -> Any:
    from .io.registry import read
    extra = {k: getattr(a, k) for k in ("elf", "locpot", "aeccar0", "aeccar2")
             if getattr(a, k, None)}
    return read(a.path, **extra)


def _json_safe(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if hasattr(x, "item"):
        return _json_safe(x.item())
    return x


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="pydemi", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    f = sub.add_parser("featurize", help="descriptors of one structure")
    f.add_argument("path")
    f.add_argument("--out", required=True, help=".json (or .csv)")
    for k in ("elf", "locpot", "aeccar0", "aeccar2"):
        f.add_argument(f"--{k}", default=None)
    _options(f)

    b = sub.add_parser("batch", help="descriptors of many structures")
    b.add_argument("root")
    b.add_argument("--glob", default="*/CHGCAR")
    b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--on-error", default="record", choices=["record", "raise", "skip"])
    b.add_argument("--companions", action="store_true",
                   help="also read ELFCAR, LOCPOT, AECCAR0/2 next to each CHGCAR")
    b.add_argument("--quiet", action="store_true")
    _options(b)

    c = sub.add_parser("catalogue", help="descriptor metadata")
    c.add_argument("--out", required=True)

    s = sub.add_parser("sweep", help="shell-cutoff sensitivity")
    s.add_argument("path")
    s.add_argument("--param", required=True, choices=["c1", "c2"])
    s.add_argument("--range", required=True, help="start:stop:step (inclusive)")
    s.add_argument("--out", required=True)
    for k in ("elf", "locpot", "aeccar0", "aeccar2"):
        s.add_argument(f"--{k}", default=None)
    _options(s)

    a = parser.parse_args(argv)

    if a.command == "featurize":
        from .descriptors import featurize
        feats, meta = featurize(_read_one(a), return_metadata=True, **_kwargs(a))
        out = Path(a.out)
        if out.suffix == ".csv":
            import pandas as pd
            pd.DataFrame([{"path": a.path, **feats, **meta}]).to_csv(out, index=False)
        else:
            out.write_text(json.dumps({"path": a.path,
                                       "features": {k: _json_safe(v) for k, v in feats.items()},
                                       "metadata": {k: _json_safe(v) for k, v in meta.items()}},
                                      indent=1))
        return 0

    if a.command == "batch":
        from .batch import featurize_batch, find_runs
        paths = find_runs(a.root, a.glob)
        if not paths:
            print(f"no files match {a.glob!r} under {a.root}", file=sys.stderr)
            return 1
        df = featurize_batch(paths, n_workers=a.workers, on_error=a.on_error,
                             progress=not a.quiet, companions=a.companions, **_kwargs(a))
        df.to_csv(a.out, index=False)
        n_err = int((df["error"] != "").sum())
        print(f"wrote {a.out}: {len(df)} structures, {n_err} errors", file=sys.stderr)
        return 0

    if a.command == "catalogue":
        from .descriptors import catalogue
        catalogue().to_csv(a.out, index=False)
        return 0

    if a.command == "sweep":
        import numpy as np

        from .validate.convergence import sensitivity_sweep
        start, stop, step = (float(x) for x in a.range.split(":"))
        values = np.round(np.arange(start, stop + step / 2, step), 10)
        kw = _kwargs(a)
        kw.pop("shells", None)
        vd = _read_one(a)
        if a.param == "c1":
            df = sensitivity_sweep(vd, c1_range=values, **kw)
        else:
            df = sensitivity_sweep(vd, c2_range=values, **kw)
        df.to_csv(a.out, index=False)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
