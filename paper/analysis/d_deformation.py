"""The deformation density of a CHGCAR, and Hirshfeld charges from it.

For random structures with an OUTCAR (ZVAL, RCORE from the run itself): the
share of int |delta rho| inside vs outside the PAW spheres, the charge inside
the spheres, and -- for binaries -- whether the Hirshfeld charges (Z_i = ZVAL)
follow the Pauling electronegativity.

Usage:  python d_deformation.py [N=150]
Output: out/deformation.json
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pymatgen.core import Element

from common import OUT, WORKERS, load, sample_ids
from pydemi import hirshfeld_charges
from pydemi.descriptors.bonding import promolecule_density
from pydemi.descriptors.registry import augmentation_radii, geometry


def one(mid):
    try:
        vd = load(mid, zval=None, paw_radii=None)
        if vd.zval is None or vd.paw_radii is None:
            return {"id": mid, "skip": "no OUTCAR"}
        rho = vd.rho.data
        pro = promolecule_density(vd)
        a = np.abs(rho - pro)
        geo = geometry(vd)
        R = augmentation_radii(vd)[0]
        inside = geo.distance <= R[geo.atom_index]
        dV = vd.rho.dV
        out = {"id": mid, "n_atoms": vd.structure.n_atoms,
               "Q_tot": float(rho.sum() * dV), "Q_pro": float(pro.sum() * dV),
               "abs_drho_total": float(a.sum() * dV), "abs_drho_inside_paw": float(a[inside].sum() * dV),
               "volume_fraction_inside_paw": float(inside.mean()),
               "charge_inside_paw_chgcar": float(rho[inside].sum() * dV),
               "charge_inside_paw_promolecule": float(pro[inside].sum() * dV)}
        els = vd.structure.elements
        if len(els) == 2:
            q = hirshfeld_charges(vd)
            chi = {e: Element(e).X for e in els}
            sp = vd.structure.species
            qe = {e: float(np.mean([qi for qi, s in zip(q, sp) if s == e])) for e in els}
            lo, hi = sorted(els, key=lambda e: chi[e])
            out.update(binary=True, dchi=chi[hi] - chi[lo], q_electropositive=qe[lo], q_electronegative=qe[hi],
                       pair=f"{lo}-{hi}")
        return out
    except Exception as exc:                                    # noqa: BLE001
        return {"id": mid, "error": repr(exc)}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    with ProcessPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(one, sample_ids(n, seed=7), chunksize=1))
    (OUT / "deformation.json").write_text(json.dumps(rows, indent=1))
    ok = [r for r in rows if "abs_drho_total" in r]
    print(f"{len(ok)} structures with OUTCAR (skipped {sum('skip' in r for r in rows)}, "
          f"errors {sum('error' in r for r in rows)})")
    q = lambda x: f"median {np.median(x):.3f} IQR [{np.percentile(x, 25):.3f}, {np.percentile(x, 75):.3f}]"
    share = np.array([r["abs_drho_inside_paw"] / r["abs_drho_total"] for r in ok])
    vol = np.array([r["volume_fraction_inside_paw"] for r in ok])
    pol = np.array([r["abs_drho_total"] / r["Q_tot"] for r in ok])
    cin = np.array([(r["charge_inside_paw_chgcar"] - r["charge_inside_paw_promolecule"])
                    / r["charge_inside_paw_promolecule"] for r in ok])
    print("share of int|drho| inside PAW spheres:", q(share), "| volume share", q(vol))
    print("int|drho| / Q over the cell:", q(pol))
    print("relative charge difference inside the spheres (CHGCAR vs promolecule):", q(cin))
    b = [r for r in ok if r.get("binary")]
    right = [r for r in b if r["q_electropositive"] > r["q_electronegative"]]
    strong = [r for r in b if r["dchi"] > 0.4]
    print(f"binaries: {len(b)}; Hirshfeld charge order follows electronegativity in {len(right)} "
          f"({len(right) / max(len(b), 1):.0%}); for dchi > 0.4: {sum(r in right for r in strong)}/{len(strong)}")
