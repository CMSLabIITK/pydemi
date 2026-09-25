"""Within-element site variance where it must vanish by symmetry.

If all sites of every element are symmetry-equivalent (spglib), the true
within-element variance is exactly 0; the computed value is numerical noise.
"""
import csv
import json
import warnings

import numpy as np
import spglib

from common import DATASET, OUT, RESULTS
from pydemi.elements import atomic_number
from pydemi.io.vasp import _parse_header

rows = {r["material_id"]: r for r in csv.DictReader(open(RESULTS))}
res = {"all_equivalent": [], "not_equivalent": []}
for mid, r in rows.items():
    with open(DATASET / mid / "CHGCAR") as fh:
        head = [next(fh) for _ in range(8 + 400)] if False else None
    lines = []
    with open(DATASET / mid / "CHGCAR") as fh:
        for line in fh:
            lines.append(line)
            if len(lines) > 7 and not line.strip():
                break
    s, _ = _parse_header(lines, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = spglib.get_symmetry_dataset((s.lattice, s.frac_coords, [atomic_number(x) for x in s.species]), 0.01)
    eq = ds.equivalent_atoms
    sp = np.array(s.species)
    has_repeat = any((sp == e).sum() > 1 for e in set(sp))
    if not has_repeat:
        continue
    all_eq = all(len(set(eq[sp == e])) == 1 for e in set(sp))
    val = r["m1_within_share"]
    if val in ("", "nan"):
        continue
    res["all_equivalent" if all_eq else "not_equivalent"].append(
        {"id": mid, "m1_within_share": float(val), "m1_var_within": float(r["m1_var_within"]),
         "m1_var_between": float(r["m1_var_between"]) if r["m1_var_between"] not in ("", "nan") else None})
(OUT / "site_symmetry.json").write_text(json.dumps(res))
for k, v in res.items():
    w = np.array([x["m1_var_within"] for x in v]); sh = np.array([x["m1_within_share"] for x in v])
    print(f"{k:15s} n={len(v):5d}  m1_var_within median {np.median(w):.2e} (A^2)  within_share median {np.median(sh):.3f}  95th {np.percentile(sh,95):.3f}")
