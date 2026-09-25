"""Percolation levels of example materials, with planar cuts through their bottlenecks.

For each example: the three levels, perc_anisotropy, rho_min_ratio and rho_min_int_ratio;
the bottleneck voxel of each axis; and one grid plane through the bottleneck of the
lowest-level axis, spanned by that axis and a second lattice vector, with rho and the
in-plane atoms (for the figure: the super-level sets {rho >= rho_perc_alpha}).

Usage:  python examples.py   -> ../data/slices.npz, ../data/examples.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "paper" / "analysis"))
from common import load  # noqa: E402

from pydemi.descriptors.registry import REGISTRY, Sentinel, augmentation_radii, geometry, make_options  # noqa: E402
from pydemi.operators.topology import percolation_levels  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data"
EXAMPLES = [("Na_229", "Na (bcc, free-electron metal)"), ("Al_225", "Al (fcc metal)"),
            ("Si_227", "Si (covalent)"), ("NaCl_225", "NaCl (ionic)"),
            ("C_194", "C (P6_3/mmc sp3 polytype)"), ("NbSe2_194", "NbSe2 (layered metal)"),
            ("CO2_205", "CO2 (molecular solid)"), ("Ca2N_166", "Ca2N (layered electride)")]
NAMES = ["rho_perc_a", "rho_perc_b", "rho_perc_c", "perc_anisotropy", "rho_min", "rho_min_ratio",
         "rho_min_int", "rho_min_int_ratio", "rho_int_mean"]


def value(r):
    return r.value if isinstance(r, Sentinel) else float(r)


def main():
    store, rows = {}, []
    for run, label in EXAMPLES:
        vd = load(run)
        v = vd.with_options(make_options(extensions=("paw",)))
        rho = vd.rho.data
        s = vd.structure
        geo = geometry(v)
        R = augmentation_radii(v)[0]
        lev = percolation_levels(rho)
        row = {"id": run, "label": label, "mean_rho": float(rho.mean()),
               "len_a": float(np.linalg.norm(s.lattice.matrix[0])),
               "len_c": float(np.linalg.norm(s.lattice.matrix[2]))}
        for n in NAMES:
            row[n] = value(REGISTRY[n].func(v))
        bott = []
        for ax, lab in enumerate("abc"):
            k = tuple(np.argwhere(rho == lev[ax])[0])
            bott.append(k)
            row[f"bottleneck_r_{lab}"] = float(geo.distance[k])
            row[f"bottleneck_r_over_Rpaw_{lab}"] = float(geo.distance[k] / R[geo.atom_index[k]])
        rows.append(row)
        low = int(np.argmin(lev))
        i = low
        j = [a for a in range(3) if a != low][0] if low != 2 else 0
        kax = 3 - i - j
        idx = bott[low][kax]
        sl = [slice(None)] * 3
        sl[kax] = idx
        sl = tuple(sl)
        n = rho.shape
        fi, fj = np.meshgrid(np.arange(n[i]) / n[i], np.arange(n[j]) / n[j], indexing="ij")
        A = s.lattice.matrix
        e1 = A[i] / np.linalg.norm(A[i])
        e2 = A[j] - (A[j] @ e1) * e1
        e2 /= np.linalg.norm(e2)
        xy = fi[..., None] * A[i] + fj[..., None] * A[j]
        bk = np.array([bott[low][i] / n[i], bott[low][j] / n[j]])
        bxy = bk[0] * A[i] + bk[1] * A[j]
        fz = idx / n[kax]
        atoms, spec = [], []
        for t, (f, el) in enumerate(zip(s.frac_coords, s.species)):
            dz = ((f[kax] - fz + 0.5) % 1 - 0.5) * np.linalg.norm(A[kax])
            if abs(dz) > 0.6:
                continue
            for s1 in (-1, 0, 1):
                for s2 in (-1, 0, 1):
                    p = ((f[i] % 1) + s1) * A[i] + ((f[j] % 1) + s2) * A[j]
                    atoms.append([p @ e1, p @ e2, abs(dz)])
                    spec.append(el)
        store.update({f"{run}__X": xy @ e1, f"{run}__Y": xy @ e2, f"{run}__rho": rho[sl], f"{run}__levels": lev,
                      f"{run}__bottleneck": np.array([bxy @ e1, bxy @ e2]), f"{run}__atoms": np.array(atoms),
                      f"{run}__species": np.array(spec),
                      f"{run}__meta": np.array([label, "abc"[i] + "abc"[j], "abc"[low]])})
        print(run, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()})
    np.savez_compressed(OUT / "slices.npz", **store)
    pd.DataFrame(rows).to_csv(OUT / "examples.csv", index=False)


if __name__ == "__main__":
    main()
