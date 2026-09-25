"""Where the deformation sums come from, on real VASP densities (PAW pseudo-densities).

For N random structures (ZVAL and RCORE from the run's OUTCAR, else the dataset table):
- the accumulated charge (delta_rho > 0) and depleted charge (delta_rho < 0) split over
  the three shells: core r <= 0.8 A, bond 0.8 < r <= 1.5 A, interstitial r > 1.5 A;
- the share of the accumulated, depleted and absolute deformation inside the PAW
  augmentation spheres (r <= R_PAW of the nearest atom), and the share of the m1_def
  numerator sum |drho| r from inside them;
- the fraction of the bond shell's volume that lies inside the PAW spheres;
- the radial profile of accumulation and depletion against r / R_PAW (0.05 bins, per
  electron), for the median profile in the figure.

Usage:  python region_shares.py [N=100]   -> ../data/region_shares.csv, ../data/radial_paw.csv
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "paper" / "analysis"))
from common import load, sample_ids  # noqa: E402

from pydemi.descriptors.bonding import delta_rho  # noqa: E402
from pydemi.descriptors.registry import augmentation_radii, geometry, make_options, masks  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data"
XB = np.arange(0.0, 3.0001, 0.05)          # r / R_PAW bins


def one(mid):
    try:
        vd = load(mid)
        v = vd.with_options(make_options())
        d = delta_rho(v)
        geo = geometry(v)
        m = masks(v)
        R = augmentation_radii(v)[0]
        Rk = R[geo.atom_index]
        inside = geo.distance <= Rk
        pos, neg = np.clip(d, 0, None), np.clip(-d, 0, None)
        a = np.abs(d)
        P, Nn = pos.sum(), neg.sum()
        dV = vd.rho.dV
        Q = float(vd.rho.data.sum() * dV)
        row = {"id": mid, "Q": Q, "R_paw_mean": float(R.mean()), "R_paw_min": float(R.min()),
               "R_paw_max": float(R.max()),
               "acc_e_per_Q": float(P * dV / Q), "dep_e_per_Q": float(Nn * dV / Q)}
        for name, sh in (("core", m.core), ("bond", m.bond), ("int", m.interstitial)):
            row[f"acc_{name}"] = float(pos[sh].sum() / P)
            row[f"dep_{name}"] = float(neg[sh].sum() / Nn)
            row[f"vol_{name}"] = float(sh.mean())
        row["acc_inside_paw"] = float(pos[inside].sum() / P)
        row["dep_inside_paw"] = float(neg[inside].sum() / Nn)
        row["abs_inside_paw"] = float(a[inside].sum() / a.sum())
        row["m1_numerator_inside_paw"] = float((a * geo.distance)[inside].sum() / (a * geo.distance).sum())
        row["vol_inside_paw"] = float(inside.mean())
        row["bond_shell_inside_paw"] = float((m.bond & inside).sum() / max(m.bond.sum(), 1))
        x = (geo.distance / Rk).ravel()
        k = np.digitize(x, XB) - 1
        ok = (k >= 0) & (k < XB.size - 1)
        acc = np.bincount(k[ok], weights=pos.ravel()[ok], minlength=XB.size - 1) * dV / Q
        dep = np.bincount(k[ok], weights=neg.ravel()[ok], minlength=XB.size - 1) * dV / Q
        prof = pd.DataFrame({"id": mid, "x_lo": XB[:-1], "x_hi": XB[1:], "acc": acc, "dep": dep})
        return row, prof
    except Exception as exc:                                    # noqa: BLE001
        return {"id": mid, "error": repr(exc)}, None


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(one, sample_ids(n, seed=2026), chunksize=1))
    df = pd.DataFrame([r for r, _ in res])
    df.to_csv(OUT / "region_shares.csv", index=False)
    pd.concat([p for _, p in res if p is not None]).to_csv(OUT / "radial_paw.csv", index=False)
    ok = df[df["error"].isna()] if "error" in df else df
    print(len(ok), "structures")
    print(ok.drop(columns="id").describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.round(3).to_string())
