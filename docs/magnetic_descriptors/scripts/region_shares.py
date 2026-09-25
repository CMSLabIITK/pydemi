"""Where the magnetization sits, and how the descriptors respond to options, on real densities.

For N random magnetic structures (seed 2026, from the dataset's magnetic runs):
- share of sum |m| inside the PAW augmentation spheres, in the core / bond / interstitial
  shells, and carried by negative m (m < 0);
- the moment-sum identity sum_i mu_i - sum_k m_k dV (nearest-atom partition; must be 0);
- the element carrying the largest |mu_i|, and the share of sum_i |mu_i| on the d/f atoms;
- the radial profile of |m| against r / R_PAW (0.05 bins, per mu_B of sum |m| dV);
- f_bond_spin with other shells ((0.6, 1.3), (1.0, 1.8) A and scaled (0.6, 1.3) x R_cov);
- mu_site_std and spin_frustration under the four partitions.

Usage:  python region_shares.py [N=150]  -> ../data/region_shares.csv, radial_paw.csv
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "paper" / "analysis"))
from common import load  # noqa: E402

from pydemi.core.geometry import Shells  # noqa: E402
from pydemi.data import atomic_number  # noqa: E402
from pydemi.descriptors.magnetic import site_moments  # noqa: E402
from pydemi.descriptors.registry import (REGISTRY, Sentinel, augmentation_radii, geometry,  # noqa: E402
                                         make_options, masks)

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data"
XB = np.arange(0.0, 3.0001, 0.05)
SHELLS = {"abs_0.6_1.3": Shells(0.6, 1.3), "abs_1.0_1.8": Shells(1.0, 1.8),
          "scaled_0.6_1.3": Shells(0.6, 1.3, scaled=True)}


def value(r):
    return r.value if isinstance(r, Sentinel) else float(r)


def df_block(z):
    return 21 <= z <= 30 or 39 <= z <= 48 or 57 <= z <= 80 or 89 <= z <= 103


def one(mid):
    try:
        vd = load(mid)
        v = vd.with_options(make_options())
        m = vd.magnetization.data
        a = np.abs(m)
        geo = geometry(v)
        sh = masks(v)
        R = augmentation_radii(v)[0]
        Rk = R[geo.atom_index]
        inside = geo.distance <= Rk
        tot = a.sum()
        mu = site_moments(v)
        s = vd.structure
        row = {"id": mid, "n_atoms": s.n_atoms,
               "abs_inside_paw": float(a[inside].sum() / tot), "vol_inside_paw": float(inside.mean()),
               "abs_core": float(a[sh.core].sum() / tot), "abs_bond": float(a[sh.bond].sum() / tot),
               "abs_int": float(a[sh.interstitial].sum() / tot),
               "negative_share": float(np.clip(-m, 0, None).sum() / tot),
               "identity_residual": float(mu.sum() - m.sum() * vd.rho.dV),
               "top_element": s.species[int(np.argmax(np.abs(mu)))],
               "top_mu": float(np.abs(mu).max()),
               "df_share_of_site_moments": float(sum(abs(x) for x, e in zip(mu, s.species)
                                                     if df_block(atomic_number(e))) / np.abs(mu).sum())}
        for n in ("m1_spin", "f_bond_spin", "mu_site_std", "spin_frustration", "spin_charge_correlation"):
            row[n] = value(REGISTRY[n].func(v))
        for tag, shl in SHELLS.items():
            row[f"f_bond_spin@{tag}"] = value(REGISTRY["f_bond_spin"].func(vd.with_options(make_options(shells=shl))))
        for p in ("becke", "power", "hirshfeld"):
            vp = vd.with_options(make_options(partition=p))
            row[f"mu_site_std@{p}"] = value(REGISTRY["mu_site_std"].func(vp))
            row[f"spin_frustration@{p}"] = value(REGISTRY["spin_frustration"].func(vp))
        x = (geo.distance / Rk).ravel()
        k = np.digitize(x, XB) - 1
        ok = (k >= 0) & (k < XB.size - 1)
        prof = pd.DataFrame({"id": mid, "x_lo": XB[:-1], "x_hi": XB[1:],
                             "abs": np.bincount(k[ok], weights=a.ravel()[ok], minlength=XB.size - 1) / tot})
        return row, prof
    except Exception as exc:                                    # noqa: BLE001
        return {"id": mid, "error": repr(exc)}, None


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    d = pd.read_csv(OUT / "magnetic_dataset.csv")
    ids = d.loc[d["magnetic"], "id"].sample(n, random_state=2026).tolist()
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(one, ids, chunksize=1))
    df = pd.DataFrame([r for r, _ in res])
    df.to_csv(OUT / "region_shares.csv", index=False)
    pd.concat([p for _, p in res if p is not None]).to_csv(OUT / "radial_paw.csv", index=False)
    ok = df[df["error"].isna()] if "error" in df else df
    print(len(ok), "structures")
    print(ok.select_dtypes("number").describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.round(4).to_string())
    print(ok["top_element"].value_counts().head(15).to_string())
