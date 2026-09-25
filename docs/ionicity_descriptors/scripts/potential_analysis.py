"""What V_spread measures, on real densities, and per-site values for examples.

For N random structures (seed 2026) and the example materials:
- V_spread from the electronic Hartree potential (the dataset's source: no LOCPOT) and
  from the full electrostatic potential ("esp": Hartree + Gaussian-smeared ions with
  Z_i = ZVAL);
- the site potentials V_i and the per-site valence charge Q_i (nearest-atom partition),
  the net charges q_i = ZVAL_i - Q_i and the Hirshfeld charges;
- the std over sites of ZVAL_i, of q_i and of the Hirshfeld charges, to separate the
  part of V_spread that comes from differing valence counts from charge transfer;
- the bond census midpoint densities (for rho_mid_std) with the bond lengths and pairs.

Usage:  python potential_analysis.py [N=150]
        -> ../data/potential_sample.csv, sites_examples.csv, bonds_examples.csv
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "paper" / "analysis"))
from common import load, sample_ids  # noqa: E402

import pydemi  # noqa: E402
from pydemi.constants import BOND_TOL  # noqa: E402
from pydemi.core.geometry import bond_census_of  # noqa: E402
from pydemi.descriptors.bonding import _rho_mid, site_potentials  # noqa: E402
from pydemi.descriptors.heterogeneity import density_site_sums  # noqa: E402
from pydemi.descriptors.registry import REGISTRY, Sentinel, make_options  # noqa: E402
from pydemi.operators.sitestats import site_charge  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data"
EXAMPLES = ["Si_227", "GaAs_216", "ZnO_186", "NaCl_225", "MgO_225", "LiF_225", "Al_225", "Cu_225",
            "SrTiO3_221", "Al2O3_167"]


def val(r):
    return r.value if isinstance(r, Sentinel) else float(r)


def analyse(mid, sites=False):
    vd = load(mid)
    s = vd.structure
    vh = vd.with_options(make_options(potential_source="hartree"))
    ve = vd.with_options(make_options(potential_source="esp"))
    Vh, Ve = site_potentials(vh), site_potentials(ve)
    Q = site_charge(density_site_sums(vh))
    z = np.array([vd.zval[e] for e in s.species])
    q = z - Q
    h = np.asarray(pydemi.hirshfeld_charges(vd))
    row = {"id": mid, "n_atoms": s.n_atoms, "V_spread_hartree": float(np.std(Vh)), "V_spread_esp": float(np.std(Ve)),
           "zval_std": float(np.std(z)), "net_charge_std": float(np.std(q)), "hirshfeld_std": float(np.std(h)),
           "rho_mid_mean": val(REGISTRY["rho_mid_mean"].func(vh)), "rho_mid_std": val(REGISTRY["rho_mid_std"].func(vh))}
    out = [row]
    if sites:
        site_rows = [{"id": mid, "site": i, "element": e, "zval": float(z[i]), "V_hartree": float(Vh[i]),
                      "V_esp": float(Ve[i]), "Q_nearest": float(Q[i]), "q_net": float(q[i]), "q_hirshfeld": float(h[i])}
                     for i, e in enumerate(s.species)]
        cen = bond_census_of(vh, BOND_TOL)
        mids = _rho_mid(vh)
        bond_rows = [{"id": mid, "i": int(a), "j": int(b), "pair": "-".join(sorted([s.species[a], s.species[b]])),
                      "length": float(L), "rho_mid": float(m)}
                     for a, b, L, m in zip(cen.i, cen.j, cen.length, np.atleast_1d(mids))] \
            if not isinstance(mids, Sentinel) else []
        out += [site_rows, bond_rows]
    return out


def one(mid):
    try:
        return analyse(mid)[0]
    except Exception as exc:                                    # noqa: BLE001
        return {"id": mid, "error": repr(exc)}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    with ProcessPoolExecutor(12) as ex:
        rows = list(ex.map(one, sample_ids(n, seed=2026), chunksize=1))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "potential_sample.csv", index=False)
    ok = df[df["error"].isna()] if "error" in df else df
    print(len(ok), "structures")
    c = ok.drop(columns=["id"]).rank().corr()
    print(c.loc[["V_spread_hartree", "V_spread_esp"], ["zval_std", "net_charge_std", "hirshfeld_std",
                                                       "V_spread_hartree", "V_spread_esp"]].round(3))
    ex_rows, sites, bonds = [], [], []
    for m in EXAMPLES:
        r, sr, br = analyse(m, sites=True)
        ex_rows.append(r)
        sites += sr
        bonds += br
        print(m, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)})
    pd.DataFrame(ex_rows).to_csv(OUT / "examples.csv", index=False)
    pd.DataFrame(sites).to_csv(OUT / "sites_examples.csv", index=False)
    pd.DataFrame(bonds).to_csv(OUT / "bonds_examples.csv", index=False)
