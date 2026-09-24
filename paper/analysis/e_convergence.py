"""Grid convergence of every cheap descriptor on random real structures:
relative change between the native grid and an 80% Fourier-coarsened grid."""
import collections
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import DATASET, OUT, sample_ids
from pydemi import Engine
from pydemi.convergence import convergence_report


def one(mid):
    try:
        warnings.simplefilter("ignore")
        eng = Engine.from_vasp_dir(DATASET / mid)
        rep = convergence_report(eng, factors=(1.0, 0.8), rtol=0.02)
        return {"id": mid, "shape": list(eng.grid().shape),
                "rows": [{"name": r["name"], "rel": r["rel_change"]} for r in rep["rows"]]}
    except Exception as exc:
        return {"id": mid, "error": repr(exc)}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    with ProcessPoolExecutor(10) as pool:
        res = list(pool.map(one, sample_ids(n, seed=11)))
    (OUT / "convergence.json").write_text(json.dumps(res, indent=1))
    ok = [r for r in res if "rows" in r]
    print(f"{len(ok)} structures")
    frac = [np.mean([x["rel"] > 0.02 for x in r["rows"]]) for r in ok]
    print(f"share of descriptors changing > 2% at 80% resolution: median {np.median(frac):.3f}, range {min(frac):.3f}-{max(frac):.3f} (of {len(ok[0]['rows'])})")
    per = collections.defaultdict(list)
    for r in ok:
        for x in r["rows"]:
            per[x["name"]].append(x["rel"])
    med = sorted(((np.median(v), k) for k, v in per.items() if np.all(np.isfinite(v))), reverse=True)
    print("most grid-sensitive (median rel. change):")
    for m, k in med[:15]: print(f"   {k:32s} {m:.3f}")
    stable = [k for m, k in med if m < 0.001]
    print(f"{len(stable)} descriptors change < 0.1% (median); e.g. {stable[:12]}")
