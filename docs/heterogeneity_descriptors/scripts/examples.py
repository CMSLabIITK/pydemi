"""Per-site values for example materials, under the four partitions.

For each example: the per-site m1^(i), f_bond^(i) and zeta^(i) (nearest, power, becke and
hirshfeld partitions), the site's element, Wyckoff letter and symmetry-equivalence class
(spglib, 0.01 A), its nearest-neighbour distance and coordination number (neighbours
within 1.15 x the shortest distance), and the heterogeneity descriptors themselves.

Usage:  python examples.py   -> ../data/sites_examples.csv, ../data/examples.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import spglib

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "paper" / "analysis"))
from common import load  # noqa: E402

from pydemi.data import atomic_number  # noqa: E402
from pydemi.descriptors.heterogeneity import _per_site_f_bond, _per_site_m1, _per_site_zeta  # noqa: E402
from pydemi.descriptors.registry import REGISTRY, Sentinel, make_options  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data"
EXAMPLES = [("NaCl_225", "NaCl (rock salt)"), ("GaAs_216", "GaAs (zinc blende)"),
            ("Si_227", "Si (diamond)"), ("ZrO2_14", "ZrO2 (baddeleyite)"),
            ("Ba2SnO4_139", "Ba2SnO4 (Ruddlesden-Popper)"), ("Fe3Si_225", "Fe3Si (D0_3)"),
            ("Fe3C_62", "Fe3C (cementite)"), ("B13C2_166", "B13C2 (boron carbide)")]
PARTITIONS = ("nearest", "power", "becke", "hirshfeld")
BASES = {"m1": _per_site_m1, "f_bond": _per_site_f_bond, "zeta": _per_site_zeta}
STATS = ["site_std", "within_element_var", "between_element_var"]


def neighbours(s):
    """Shortest distance and coordination number (within 1.15 x shortest) of every site."""
    lat = s.lattice.matrix
    f = s.frac_coords
    shifts = np.array([[i, j, k] for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2)
                       for k in (-2, -1, 0, 1, 2)])
    dmin, cn = [], []
    for a in range(len(f)):
        d = np.linalg.norm(((f[None, :, :] + shifts[:, None, :]) - f[a]) @ lat, axis=-1).ravel()
        d = d[d > 1e-6]
        dmin.append(d.min())
        cn.append(int((d <= 1.15 * d.min()).sum()))
    return np.array(dmin), np.array(cn)


def main():
    sites, rows = [], []
    for run, label in EXAMPLES:
        vd = load(run)
        s = vd.structure
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = spglib.get_symmetry_dataset((s.lattice.matrix, s.frac_coords,
                                              [atomic_number(x) for x in s.species]), 0.01)
        dmin, cn = neighbours(s)
        per = {}
        row = {"id": run, "label": label, "n_atoms": s.n_atoms, "elements": " ".join(s.elements),
               "n_distinct_sites": len(set(ds.equivalent_atoms))}
        for p in PARTITIONS:
            v = vd.with_options(make_options(partition=p))
            for b, fn in BASES.items():
                per[(p, b)] = fn(v)
                for st in STATS:
                    r = REGISTRY[f"{b}_{st}"].func(v)
                    row[f"{b}_{st}@{p}"] = r.value if isinstance(r, Sentinel) else float(r)
        rows.append(row)
        for i in range(s.n_atoms):
            site = {"id": run, "label": label, "site": i, "element": s.species[i],
                    "wyckoff": ds.wyckoffs[i], "equivalent_to": int(ds.equivalent_atoms[i]),
                    "d_nn": float(dmin[i]), "cn": int(cn[i])}
            for (p, b), x in per.items():
                site[f"{b}@{p}"] = float(x[i])
            sites.append(site)
        print(run, {k: round(v, 5) for k, v in row.items() if isinstance(v, float) and "@nearest" in k})
    pd.DataFrame(sites).to_csv(OUT / "sites_examples.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "examples.csv", index=False)


if __name__ == "__main__":
    main()
