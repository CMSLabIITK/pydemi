"""
pydemi.cli
==========
Command line (spec §12):

    pydemi featurize CHGCAR --out features.json
    pydemi batch ./runs --glob "*/CHGCAR" --out features.csv --workers 8 --domains bonding
    pydemi catalogue --out catalogue.csv
    pydemi sweep CHGCAR --param c2 --range 1.0:2.5:0.05 --out sweep.csv
    pydemi gui                      # graphical interface in the browser (pip install "pydemi[gui]")
    pydemi predict Si.cif --out-dir predicted   # ChargE3Net density from a structure (pydemi[ml])

The input is a VASP CHGCAR, or for ``predict`` a crystal structure (CIF, POSCAR, ...).
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
                   help="comma-separated: bonding,structural,heterogeneity")
    p.add_argument("--partition", default="nearest", choices=["nearest", "power", "becke"])
    p.add_argument("--shells", default=None, help="c1,c2 in Angstrom (default 0.8,1.5)")
    p.add_argument("--laplacian-method", default="metric", choices=["metric", "diagonal"])
    p.add_argument("--derivative-backend", default="fft", choices=["fft", "fd"])
    p.add_argument("--fd-order", type=int, default=4)
    p.add_argument("--float32", action="store_true")


def _kwargs(a: argparse.Namespace) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "domains": a.domains, "partition": a.partition,
        "laplacian_method": a.laplacian_method, "derivative_backend": a.derivative_backend,
        "fd_order": a.fd_order, "float32": a.float32}
    if a.shells:
        c1, c2 = (float(x) for x in a.shells.split(","))
        kw["shells"] = (c1, c2)
    return kw


def _read_one(a: argparse.Namespace) -> Any:
    from .io.vasp import read_chgcar
    return read_chgcar(a.path)


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
    f.add_argument("path", help="CHGCAR")
    f.add_argument("--out", required=True, help=".json (or .csv)")
    _options(f)

    b = sub.add_parser("batch", help="descriptors of many structures")
    b.add_argument("root")
    b.add_argument("--glob", default="*/CHGCAR")
    b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--on-error", default="record", choices=["record", "raise", "skip"])
    b.add_argument("--quiet", action="store_true")
    _options(b)

    c = sub.add_parser("catalogue", help="descriptor metadata")
    c.add_argument("--out", required=True)

    s = sub.add_parser("sweep", help="shell-cutoff sensitivity")
    s.add_argument("path")
    s.add_argument("--param", required=True, choices=["c1", "c2"])
    s.add_argument("--range", required=True, help="start:stop:step (inclusive)")
    s.add_argument("--out", required=True)
    _options(s)

    g = sub.add_parser("gui", help="graphical interface in the browser (needs pydemi[gui])")
    g.add_argument("--host", default="localhost",
                   help="address to listen on (default localhost: this machine only)")
    g.add_argument("--port", type=int, default=None,
                   help="port (default 8501, or the next free one)")
    g.add_argument("--no-browser", action="store_true", help="do not open a browser")

    p = sub.add_parser("predict", help="predict a CHGCAR from a crystal structure with "
                                       "ChargE3Net (needs pydemi[ml])")
    p.add_argument("structures", nargs="+", help="CIF, POSCAR/CONTCAR or any file ASE reads")
    p.add_argument("--out-dir", required=True, help="writes <name>.CHGCAR here")
    p.add_argument("--featurize", action="store_true",
                   help="also write <name>_descriptors.json from the prediction")
    p.add_argument("--device", default=None, help="cpu, cuda, cuda:1, ... (default: a GPU if any)")
    p.add_argument("--encut", type=float, default=500.0,
                   help="ENCUT (eV) that sets the VASP grid; the training data used 500")
    p.add_argument("--grid", default=None,
                   help="n1,n2,n3 instead of the VASP grid (a coarse grid is a quick preview)")
    p.add_argument("--checkpoint", default=None, help="other ChargE3Net weights (.pt)")

    a = parser.parse_args(argv)

    if a.command == "predict":
        return _predict(a)

    if a.command == "gui":
        from .gui import launch
        return launch(host=a.host, port=a.port, browser=not a.no_browser)

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
                             progress=not a.quiet, **_kwargs(a))
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


def _predict(a: argparse.Namespace) -> int:
    from . import ml
    gone = ml.missing()
    if gone:
        print(f"pydemi predict needs {', '.join(gone)}: {ml.INSTALL_HINT}", file=sys.stderr)
        return 1
    from .io.vasp import write_volumetric
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = ml.load(a.checkpoint, a.device)
    print(f"{ml.MODEL_NAME} on {model.device}. Cite: {ml.CITATION}", file=sys.stderr)
    failed = 0
    for path in a.structures:
        name = Path(path).stem if Path(path).suffix else Path(path).name
        try:
            s = ml.read_structure(path)
            n1, n2, n3 = ((int(n) for n in a.grid.split(",")) if a.grid
                          else ml.vasp_grid_shape(s.lattice, a.encut))
            shape = (n1, n2, n3)

            def show(done: int, total: int) -> None:
                print(f"\r{name}: {100 * done / total:5.1f}% of {total:,} grid points",
                      end="", file=sys.stderr, flush=True)

            vd = ml.predict(s, shape=shape, encut=a.encut, checkpoint=a.checkpoint,
                            device=a.device, progress=show)
        except (ValueError, OSError) as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            failed += 1
            continue
        print(file=sys.stderr)
        target = out / f"{name}.CHGCAR"
        write_volumetric(target, s, [vd.rho.data],
                         comment=f"{name}: predicted by {ml.MODEL_NAME}, not DFT (pydemi)")
        print(f"wrote {target} (grid {'x'.join(map(str, shape))}, "
              f"scale {vd.sources['charge_scale']}, {vd.sources['predict_time_s']} s)",
              file=sys.stderr)
        if a.featurize:
            from .descriptors import featurize
            feats, meta = featurize(vd, return_metadata=True)
            (out / f"{name}_descriptors.json").write_text(json.dumps(
                {"path": str(path), "density": "predicted", **vd.sources,
                 "features": {k: _json_safe(v) for k, v in feats.items()},
                 "metadata": {k: _json_safe(v) for k, v in meta.items()}}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
