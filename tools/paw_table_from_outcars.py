"""
Per-element PAW data (POTCAR title, ZVAL, RCORE) from every OUTCAR under a dataset root.

For runs of the same POTCAR set that lack a POTCAR and an OUTCAR: pass the
table to ``pydemi batch --paw-table`` (the run's own files still win). Stops
if an element appears with two different ZVALs, i.e. the runs mix POTCARs.

Usage:  python tools/paw_table_from_outcars.py ROOT OUT.json [--glob "*/OUTCAR"] [--workers 16]
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from pydemi.io.vasp import read_potcar_rcore, read_potcar_zval


def one(outcar: Path):
    text = outcar.read_text(errors="replace")
    titles = re.findall(r"TITEL\s*=\s*\S+\s+(\S+)", text)
    zv, rc = read_potcar_zval(outcar), read_potcar_rcore(outcar)
    if not (len(titles) == len(zv) == len(rc)) or not titles:
        return None
    return [(t.split("_")[0].split("/")[0], t, z, r) for t, z, r in zip(titles, zv, rc)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--glob", default="*/OUTCAR")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    files = sorted(Path(a.root).glob(a.glob))
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(one, files, chunksize=32))
    z, rc, title = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for res in filter(None, results):
        for e, t, zv, r in res:
            z[e][zv] += 1
            rc[e][r] += 1
            title[e][t] += 1
    mixed = {e: dict(c) for e, c in z.items() if len(c) > 1}
    if mixed:
        print(f"elements with more than one ZVAL (mixed POTCARs): {mixed}", file=sys.stderr)
        return 1
    table = {e: {"potcar": title[e].most_common(1)[0][0], "zval": z[e].most_common(1)[0][0],
                 "rcore_bohr": rc[e].most_common(1)[0][0], "n_runs": sum(z[e].values())}
             for e in sorted(z)}
    Path(a.out).write_text(json.dumps(table, indent=1) + "\n")
    print(f"{sum(r is not None for r in results)} of {len(files)} OUTCARs read; "
          f"{len(table)} elements -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
