"""Where the anisotropy sums come from, on real VASP densities.

For N random structures with their own OUTCAR (RCORE known):
- share of sum |grad rho| (zeta's denominator) and of sum |grad rho|^2 (T's trace) from
  voxels inside the PAW augmentation spheres;
- zeta and charge_FA recomputed over the voxels outside the spheres only;
- share of sum |grad ELF| from voxels with rho < 0.01 e/A^3, and zeta_ELF over rho >= 0.01.

Usage:  python region_shares.py [N=100]   -> ../data/region_shares.csv
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "paper" / "analysis"))
from common import load, sample_ids  # noqa: E402

from pydemi.descriptors.bonding import elf_field_name  # noqa: E402
from pydemi.descriptors.registry import (REGISTRY, augmentation_radii, field_derivatives,  # noqa: E402
                                         geometry, make_options)
from pydemi.operators.anisotropy import (anisotropy_tensor, fractional_anisotropy,  # noqa: E402
                                         gradient_anisotropy)

OUT = Path(__file__).resolve().parents[1] / "data"
LOW = 0.01            # e/A^3


def one(mid):
    try:
        vd = load(mid, zval=None, paw_radii=None)
        if vd.paw_radii is None:
            return None
        v = vd.with_options(make_options())
        g = field_derivatives(v, "rho").gradient
        geo = geometry(v)
        R = augmentation_radii(v)[0]
        inside = geo.distance <= R[geo.atom_index]
        gn = np.linalg.norm(g, axis=-1)
        ge = field_derivatives(v, elf_field_name(v)).gradient
        gen = np.linalg.norm(ge, axis=-1)
        low = vd.rho.data < LOW
        out_T = anisotropy_tensor(g[~inside])
        return {"id": mid, "zeta": REGISTRY["zeta"].func(v), "charge_FA": REGISTRY["charge_FA"].func(v),
                "zeta_ELF": REGISTRY["zeta_ELF"].func(v),
                "volume_inside": float(inside.mean()),
                "grad_share_inside": float(gn[inside].sum() / gn.sum()),
                "grad2_share_inside": float((gn[inside] ** 2).sum() / (gn ** 2).sum()),
                "zeta_outside": gradient_anisotropy(g, geo.direction, ~inside),
                "charge_FA_outside": fractional_anisotropy(out_T),
                "volume_low_density": float(low.mean()),
                "elf_grad_share_low_density": float(gen[low].sum() / gen.sum()),
                "zeta_ELF_high_density": gradient_anisotropy(ge, geo.direction, ~low)}
    except Exception as exc:                                    # noqa: BLE001
        return {"id": mid, "error": repr(exc)}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    with ProcessPoolExecutor(8) as ex:
        rows = [r for r in ex.map(one, sample_ids(n, seed=2026), chunksize=1) if r]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "region_shares.csv", index=False)
    ok = df[df.get("error", pd.Series(index=df.index, dtype=object)).isna()] if "error" in df else df
    print(len(ok), "structures")
    print(ok.describe().round(3).to_string())
    for a, b in (("zeta", "zeta_outside"), ("charge_FA", "charge_FA_outside"), ("zeta_ELF", "zeta_ELF_high_density")):
        print(f"Spearman {a} vs {b}: {ok[a].rank().corr(ok[b].rank()):.3f}")
