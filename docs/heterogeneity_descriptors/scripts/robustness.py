"""Sensitivity of the site-heterogeneity descriptors to the choices they depend on.

1. Derivatives (only the zeta statistics use the gradient): FFT (default) against central
   differences of order 2, 4 and 8, for N random structures (seed 31).
2. Shells (only the f_bond statistics use them): (0.8, 1.5) A against (0.6, 1.3),
   (1.0, 1.8) and radius-scaled (0.6, 1.3) x covalent radius.
3. Supercell invariance: each statistic is a population statistic over sites, so a
   2 x 1 x 1 supercell (density tiled, atoms duplicated) must give the same value. M
   random structures.
4. Partition and grid: the paper's analyses (paper/analysis/b_partitions.py, 300
   structures; e_convergence.py, 30 structures), filtered to these descriptors.

Usage:  python robustness.py [N=60] [M=12]
        -> ../data/robust_derivatives.csv, robust_shells.csv, robust_supercell.csv,
           robust_partitions.csv, robust_grid.csv (+ *_summary.csv)
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "paper" / "analysis"))
from common import OUT as PAPER_OUT, load, sample_ids  # noqa: E402

from pydemi.core.geometry import Shells  # noqa: E402
from pydemi.descriptors.registry import REGISTRY, Sentinel, make_options  # noqa: E402
from pydemi.io.base import Grid, Lattice, Structure, VolumetricData  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data"
BASES = ["m1", "f_bond", "zeta"]
STATS = ["site_std", "within_element_var", "between_element_var"]
NAMES = [f"{b}_{s}" for b in BASES for s in STATS]
ZETA = [n for n in NAMES if n.startswith("zeta")]
FBOND = [n for n in NAMES if n.startswith("f_bond")]
SCHEMES = {"fft": {}, "fd2": {"derivative_backend": "fd", "fd_order": 2},
           "fd4": {"derivative_backend": "fd", "fd_order": 4},
           "fd8": {"derivative_backend": "fd", "fd_order": 8}}
SHELLS = {"abs_0.8_1.5": Shells(0.8, 1.5), "abs_0.6_1.3": Shells(0.6, 1.3),
          "abs_1.0_1.8": Shells(1.0, 1.8), "scaled_0.6_1.3": Shells(0.6, 1.3, scaled=True)}


def value(r):
    return r.value if isinstance(r, Sentinel) else float(r)


def one(mid):
    try:
        vd = load(mid)
        row = {"id": mid, "n_atoms": vd.structure.n_atoms}
        for tag, kw in SCHEMES.items():
            v = vd.with_options(make_options(**kw))
            for n in ZETA:
                row[f"{n}@{tag}"] = value(REGISTRY[n].func(v))
        for tag, sh in SHELLS.items():
            v = vd.with_options(make_options(shells=sh))
            for n in FBOND:
                row[f"{n}@{tag}"] = value(REGISTRY[n].func(v))
        return row
    except Exception as exc:                                    # noqa: BLE001
        return {"id": mid, "error": repr(exc)}


def supercell(mid):
    try:
        vd = load(mid)
        s = vd.structure
        lat = s.lattice.matrix.copy()
        lat2 = lat.copy()
        lat2[0] *= 2
        f = np.asarray(s.frac_coords) % 1.0
        f2 = np.vstack([f * [0.5, 1, 1], (f + [1, 0, 0]) * [0.5, 1, 1]])
        s2 = Structure(Lattice(lat2), list(s.species) * 2, f2)
        vd2 = VolumetricData(s2, Grid(np.tile(vd.rho.data, (2, 1, 1)), s2.lattice))
        row = {"id": mid, "n_atoms": s.n_atoms}
        for tag, x in (("cell", vd), ("supercell", vd2)):
            v = x.with_options(make_options())
            for n in NAMES:
                row[f"{n}@{tag}"] = value(REGISTRY[n].func(v))
        return row
    except Exception as exc:                                    # noqa: BLE001
        return {"id": mid, "error": repr(exc)}


def rel(a, b):
    return (a - b).abs() / np.maximum(np.maximum(a.abs(), b.abs()), 1e-300)


def summarize(df, names, tags, ref, out):
    ok = df[df["error"].isna()] if "error" in df else df
    rows = []
    for n in names:
        a = ok[f"{n}@{ref}"]
        for t in tags:
            b = ok[f"{n}@{t}"]
            m = a.notna() & b.notna()
            rows.append({"descriptor": n, "variant": t, "n": int(m.sum()),
                         "median_rel": rel(b[m], a[m]).median(), "p90_rel": rel(b[m], a[m]).quantile(0.9),
                         "spearman": a[m].rank().corr(b[m].rank())})
    s = pd.DataFrame(rows)
    s.to_csv(OUT / out, index=False)
    print(s.round(4).to_string())
    return s


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    small = [i for i in sample_ids(400, seed=77) if load(i).structure.n_atoms <= 12][:m]
    with ProcessPoolExecutor(12) as ex:
        df = pd.DataFrame(list(ex.map(one, sample_ids(n, seed=31), chunksize=1)))
        sc = pd.DataFrame(list(ex.map(supercell, small, chunksize=1)))
    df.to_csv(OUT / "robust_derivatives_shells.csv", index=False)
    sc.to_csv(OUT / "robust_supercell.csv", index=False)
    summarize(df, ZETA, ["fd2", "fd4", "fd8"], "fft", "robust_derivatives_summary.csv")
    summarize(df, FBOND, ["abs_0.6_1.3", "abs_1.0_1.8", "scaled_0.6_1.3"], "abs_0.8_1.5",
              "robust_shells_summary.csv")
    summarize(sc, NAMES, ["supercell"], "cell", "robust_supercell_summary.csv")

    part = pd.read_csv(PAPER_OUT / "partitions_summary.csv").drop_duplicates()
    part = part[part["descriptor"].isin(NAMES)]
    part.to_csv(OUT / "robust_partitions.csv", index=False)
    conv = pd.read_csv(PAPER_OUT / "convergence.csv")
    conv = conv[conv["descriptor"].isin(NAMES)]
    conv.to_csv(OUT / "robust_grid.csv", index=False)
    print(conv.groupby("descriptor")["rel_change_x0.8"].describe(percentiles=[0.5, 0.9]).round(4).to_string())
