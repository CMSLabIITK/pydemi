"""Sensitivity of the deformation descriptors to the choices they depend on.

1. Shells (f_bond_def, f_int_def, f_bond_dep only; the moments and def_polarity do not use
   them): the default absolute shells (0.8, 1.5) A against (0.6, 1.3), (1.0, 1.8), (0.8, 2.0)
   and radius-scaled shells (0.6, 1.3) x covalent radius of the nearest atom. N random
   structures (seed 31).
2. Valence counts: runs without an OUTCAR that contain one of the nine elements whose
   dataset PAW choice is a semicore one (K, Ca, Sr, Y, Zr, Nb, Ba, Cs, Rb), read with the
   dataset's per-element table (as in the rerun) and with pydemi's rule-based default ZVAL
   (what a user without the table would get).
3. Grid: the paper's convergence analysis (paper/analysis/e_convergence.py: full grid
   against a Fourier-coarsened copy with 80% of the points per axis, 30 structures),
   filtered to the deformation descriptors.

Usage:  python robustness.py [N=60] [M=40]
        -> ../data/robust_shells.csv, robust_zval.csv, robust_grid.csv
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "paper" / "analysis"))
from common import DATASET, OUT as PAPER_OUT, all_ids, load, sample_ids  # noqa: E402

from pydemi.core.geometry import Shells  # noqa: E402
from pydemi.descriptors.registry import REGISTRY, Sentinel, make_options  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data"
DEF = ["m1_def", "m2_def", "sigma_r2_def", "f_bond_def", "f_int_def", "f_bond_dep", "def_polarity"]
SHELLED = ["f_bond_def", "f_int_def", "f_bond_dep"]
SHELLS = {"abs_0.8_1.5": Shells(0.8, 1.5), "abs_0.6_1.3": Shells(0.6, 1.3),
          "abs_1.0_1.8": Shells(1.0, 1.8), "abs_0.8_2.0": Shells(0.8, 2.0),
          "scaled_0.6_1.3": Shells(0.6, 1.3, scaled=True)}
SEMICORE = {"K", "Ca", "Sr", "Y", "Zr", "Nb", "Ba", "Cs", "Rb"}


def value(r):
    return r.value if isinstance(r, Sentinel) else float(r)


def shells_one(mid):
    try:
        vd = load(mid)
        row = {"id": mid}
        for tag, sh in SHELLS.items():
            v = vd.with_options(make_options(shells=sh))
            for n in SHELLED:
                row[f"{n}@{tag}"] = value(REGISTRY[n].func(v))
        return row
    except Exception as exc:                                    # noqa: BLE001
        return {"id": mid, "error": repr(exc)}


def zval_one(mid):
    try:
        row = {"id": mid}
        for tag, kw in (("table", {}), ("default", {"zval": None})):
            vd = load(mid, **kw)
            v = vd.with_options(make_options())
            row[f"Q@{tag}"] = float(vd.rho.data.sum() * vd.rho.dV)
            from pydemi.descriptors.bonding import promolecule_density
            row[f"Qpro@{tag}"] = float(promolecule_density(v).sum() * vd.rho.dV)
            for n in DEF:
                row[f"{n}@{tag}"] = value(REGISTRY[n].func(v))
        return row
    except Exception as exc:                                    # noqa: BLE001
        return {"id": mid, "error": repr(exc)}


def rel(a, b):
    return (a - b).abs() / np.maximum(np.maximum(a.abs(), b.abs()), 1e-12)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    no_outcar = [i for i in all_ids() if not (DATASET / i / "OUTCAR").exists()
                 and {str(e) for e in Composition(i.rsplit("_", 1)[0]).elements} & SEMICORE]
    semi = sorted(np.random.default_rng(5).choice(no_outcar, size=min(m, len(no_outcar)), replace=False))
    with ProcessPoolExecutor(12) as ex:
        sh = pd.DataFrame(list(ex.map(shells_one, sample_ids(n, seed=31), chunksize=1)))
        zv = pd.DataFrame(list(ex.map(zval_one, semi, chunksize=1)))
    sh.to_csv(OUT / "robust_shells.csv", index=False)
    zv.to_csv(OUT / "robust_zval.csv", index=False)

    ok = sh[sh["error"].isna()] if "error" in sh else sh
    rows = []
    for name in SHELLED:
        ref = ok[f"{name}@abs_0.8_1.5"]
        for tag in list(SHELLS)[1:]:
            v = ok[f"{name}@{tag}"]
            rows.append({"descriptor": name, "shells": tag, "median_abs_change": (v - ref).abs().median(),
                         "p90_abs_change": (v - ref).abs().quantile(0.9),
                         "spearman": ref.rank().corr(v.rank())})
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "robust_shells_summary.csv", index=False)
    print(len(ok), "structures (shells)")
    print(s.round(3).to_string())

    ok = zv[zv["error"].isna()] if "error" in zv else zv
    rows = []
    for name in DEF:
        a, b = ok[f"{name}@table"], ok[f"{name}@default"]
        rows.append({"descriptor": name, "median_rel": rel(b, a).median(), "p90_rel": rel(b, a).quantile(0.9),
                     "max_rel": rel(b, a).max(), "spearman": a.rank().corr(b.rank())})
    z = pd.DataFrame(rows)
    z.to_csv(OUT / "robust_zval_summary.csv", index=False)
    print(len(ok), "structures (ZVAL)")
    print("promolecule electrons minus cell electrons, default ZVAL: median",
          float((ok["Qpro@default"] - ok["Q@default"]).median()), "max",
          float((ok["Qpro@default"] - ok["Q@default"]).abs().max()))
    print(z.round(4).to_string())

    conv = pd.read_csv(PAPER_OUT / "convergence.csv")
    conv = conv[conv["descriptor"].isin(DEF + [f"{d}_out" for d in DEF])]
    conv.to_csv(OUT / "robust_grid.csv", index=False)
    print(conv.groupby("descriptor")["rel_change_x0.8"].describe(percentiles=[0.5, 0.9]).round(4).to_string())
