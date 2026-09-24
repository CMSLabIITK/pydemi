"""Is the CHGCAR route of the deformation density usable on real VASP data?

For random structures with an OUTCAR (ZVAL, RCORE): the share of |delta rho|
inside vs outside the PAW spheres, delta-rho statistics outside them, and
whether CHGCAR Hirshfeld charges follow electronegativity in binaries.
"""
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import DATASET, OUT, sample_ids
from pydemi import Engine
from pydemi.elements import element_property


def one(mid):
    try:
        warnings.simplefilter("ignore")
        eng = Engine.from_vasp_dir(DATASET / mid)
        if eng.paw_radii is None or eng.zval is None:
            return {"id": mid, "skip": "no OUTCAR"}
        from pydemi.descriptors.deformation import promolecule
        from pydemi.descriptors.sites import hirshfeld_charges
        rho = eng["rho"].values
        pro = promolecule(eng)
        drho = rho - pro
        geo = eng.geometry()
        R = np.array([eng.paw_radii[s] for s in eng.structure.species])
        inside = geo.distance <= R[geo.atom_index]
        a = np.abs(drho)
        dV = eng.grid().dV
        out = {"id": mid, "n_atoms": eng.structure.n_atoms,
               "Q_tot": float(rho.sum() * dV), "Q_pro": float(pro.sum() * dV),
               "abs_drho_total": float(a.sum() * dV),
               "abs_drho_inside_paw": float(a[inside].sum() * dV),
               "volume_fraction_inside_paw": float(inside.mean()),
               "charge_inside_paw_chgcar": float(rho[inside].sum() * dV),
               "charge_inside_paw_promolecule": float(pro[inside].sum() * dV)}
        els = eng.structure.elements
        if len(els) == 2:
            q = hirshfeld_charges(eng, "rho")
            chi = {e: element_property(e, "Electronegativity") for e in els}
            qe = {e: float(np.mean([qi for qi, s in zip(q, eng.structure.species) if s == e])) for e in els}
            lo, hi = sorted(els, key=lambda e: chi[e])
            out.update(binary=True, dchi=chi[hi] - chi[lo], q_electropositive=qe[lo], q_electronegative=qe[hi])
        return out
    except Exception as exc:
        return {"id": mid, "error": repr(exc)}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    with ProcessPoolExecutor(16) as pool:
        rows = list(pool.map(one, sample_ids(n, seed=7)))
    (OUT / "deformation_chgcar_route.json").write_text(json.dumps(rows, indent=1))
    ok = [r for r in rows if "abs_drho_total" in r]
    print(f"{len(ok)} structures with OUTCAR (skipped {sum('skip' in r for r in rows)}, errors {sum('error' in r for r in rows)})")
    share = np.array([r["abs_drho_inside_paw"] / r["abs_drho_total"] for r in ok])
    vol = np.array([r["volume_fraction_inside_paw"] for r in ok])
    pol = np.array([r["abs_drho_total"] / r["Q_tot"] for r in ok])
    cin = np.array([(r["charge_inside_paw_chgcar"] - r["charge_inside_paw_promolecule"]) / r["charge_inside_paw_promolecule"] for r in ok])
    print(f"share of |drho| inside PAW spheres: median {np.median(share):.3f} IQR [{np.percentile(share,25):.3f}, {np.percentile(share,75):.3f}]  (volume share median {np.median(vol):.3f})")
    print(f"def_polarity (int|drho|/Q): median {np.median(pol):.3f} IQR [{np.percentile(pol,25):.3f}, {np.percentile(pol,75):.3f}]")
    print(f"relative charge difference inside PAW spheres (CHGCAR vs promolecule): median {np.median(cin):+.3f} IQR [{np.percentile(cin,25):+.3f}, {np.percentile(cin,75):+.3f}]")
    b = [r for r in ok if r.get("binary")]
    right = [r for r in b if r["q_electropositive"] > r["q_electronegative"]]
    strong = [r for r in b if r["dchi"] > 0.4]
    print(f"binaries: {len(b)}; Hirshfeld charge order follows electronegativity in {len(right)} ({len(right)/max(len(b),1):.0%}); for dchi > 0.4: {sum(r in right for r in strong)}/{len(strong)}")
