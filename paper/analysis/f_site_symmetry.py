"""Within-element site variance where it must vanish by symmetry.

If all sites of every element are symmetry-equivalent (spglib, tolerance
0.01 A), the true within-element variance is exactly 0 and the computed value
is numerical noise. Values from the dataset rerun table.

Output: out/site_symmetry.json
"""
import json
import warnings

import numpy as np
import spglib

from common import DATASET, OUT, results
from pydemi.data import atomic_number


def header(path):
    """Lattice, species and fractional coordinates from a VASP 5 CHGCAR header."""
    with open(path) as fh:
        lines = [next(fh) for _ in range(7)]
        scale = float(lines[1].split()[0])
        lat = np.array([[float(x) for x in lines[k].split()] for k in (2, 3, 4)]) * scale
        els, counts = lines[5].split(), [int(x) for x in lines[6].split()]
        mode = next(fh).strip().lower()
        if mode.startswith("s"):
            mode = next(fh).strip().lower()
        pos = np.array([[float(x) for x in next(fh).split()[:3]] for _ in range(sum(counts))])
    if mode.startswith(("c", "k")):
        pos = pos * scale @ np.linalg.inv(lat)
    species = [e.split("_")[0].split("/")[0] for e, c in zip(els, counts) for _ in range(c)]
    return lat, pos, species


df = results()
res = {"all_equivalent": [], "not_equivalent": []}
for mid, r in df.iterrows():
    if not np.isfinite(r["m1_within_element_var"]):
        continue                                   # no element with two sites
    lat, pos, sp = header(DATASET / mid / "CHGCAR")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq = spglib.get_symmetry_dataset((lat, pos, [atomic_number(x) for x in sp]), 0.01).equivalent_atoms
    sp = np.array(sp)
    all_eq = all(len(set(eq[sp == e])) == 1 for e in set(sp))
    res["all_equivalent" if all_eq else "not_equivalent"].append(
        {"id": mid, "m1_within_element_var": float(r["m1_within_element_var"]),
         "m1_between_element_var": float(r["m1_between_element_var"])})
(OUT / "site_symmetry.json").write_text(json.dumps(res))
for k, v in res.items():
    w = np.array([x["m1_within_element_var"] for x in v])
    print(f"{k:15s} n={len(v):5d}  m1_within_element_var median {np.median(w):.2e} A^2, 95th {np.percentile(w, 95):.2e}")
