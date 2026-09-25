"""Deformation density of example materials: planar cuts and radial profiles.

For each example: one grid plane through an atom, spanned by two lattice vectors, with
rho, the promolecule and delta_rho = rho - promolecule; the in-plane atom positions and
their PAW augmentation radii (for the circles in the figure); and the radial profiles
of the accumulated (delta_rho > 0) and depleted (delta_rho < 0) charge against the
distance r to the nearest nucleus (sum over voxels in 0.05 A bins, times dV).
The seven descriptors, whole cell and outside the PAW spheres, are printed and stored.

Usage:  python examples.py   -> ../data/slices.npz, ../data/radial_examples.csv,
                                ../data/examples.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "paper" / "analysis"))
from common import load  # noqa: E402

from pydemi.descriptors.bonding import delta_rho, promolecule_density  # noqa: E402
from pydemi.descriptors.registry import (REGISTRY, Sentinel, augmentation_radii,  # noqa: E402
                                         geometry, make_options)

OUT = Path(__file__).resolve().parents[1] / "data"
DEF = ["m1_def", "m2_def", "sigma_r2_def", "f_bond_def", "f_int_def", "f_bond_dep", "def_polarity"]
# run, label, the two lattice axes spanning the plane (the third is fixed at an atom)
EXAMPLES = [("Si_227", "Si (covalent)", (0, 1)), ("GaAs_216", "GaAs (polar covalent)", (0, 1)),
            ("NaCl_225", "NaCl (ionic)", (0, 1)), ("MgO_225", "MgO (ionic oxide)", (0, 1)),
            ("Al_225", "Al (sp metal)", (0, 1)), ("Cu_225", "Cu (d metal)", (0, 1)),
            ("BN_194", "h-BN (layered)", (0, 2)), ("Ca2N_166", "Ca2N (electride)", (0, 2))]
BINS = np.arange(0.0, 4.0001, 0.05)


def value(r):
    return r.value if isinstance(r, Sentinel) else float(r)


def main():
    store, radial, rows = {}, [], []
    for run, label, (i, j) in EXAMPLES:
        vd = load(run)
        v = vd.with_options(make_options(extensions=("paw",)))
        d = delta_rho(v)
        pro = promolecule_density(v)
        geo = geometry(v)
        R = augmentation_radii(v)[0]
        dV = vd.rho.dV
        Q = float(vd.rho.data.sum() * dV)
        # radial profiles (per electron of the cell)
        r = geo.distance.ravel()
        dd = d.ravel()
        k = np.digitize(r, BINS) - 1
        ok = (k >= 0) & (k < BINS.size - 1)
        acc = np.bincount(k[ok], weights=np.clip(dd[ok], 0, None), minlength=BINS.size - 1) * dV / Q
        dep = np.bincount(k[ok], weights=np.clip(-dd[ok], 0, None), minlength=BINS.size - 1) * dV / Q
        for a, b, x, y in zip(BINS[:-1], BINS[1:], acc, dep):
            radial.append({"id": run, "r_lo": a, "r_hi": b, "accumulated_per_e": x, "depleted_per_e": y})
        # descriptors
        row = {"id": run, "label": label, "Q": Q, "R_paw_min": float(R.min()), "R_paw_max": float(R.max()),
               "elements": " ".join(vd.structure.elements)}
        for n in DEF:
            row[n] = value(REGISTRY[n].func(v))
            row[n + "_out"] = value(REGISTRY[n + "_out"].func(v))
        row["def_out_volume_fraction"] = value(REGISTRY["def_out_volume_fraction"].func(v))
        inside = geo.distance <= R[geo.atom_index]
        a = np.abs(d)
        row["abs_share_inside_paw"] = float(a[inside].sum() / a.sum())
        rows.append(row)
        # slice
        kax = 3 - i - j
        atom = int(np.argmin(np.abs(((vd.structure.frac_coords[:, kax] + 0.5) % 1) - 0.5)))
        idx = int(round(vd.structure.frac_coords[atom, kax] * vd.shape[kax])) % vd.shape[kax]
        sl = [slice(None)] * 3
        sl[kax] = idx
        sl = tuple(sl)
        n = vd.shape
        fi, fj = np.meshgrid(np.arange(n[i]) / n[i], np.arange(n[j]) / n[j], indexing="ij")
        A = vd.structure.lattice.matrix
        e1 = A[i] / np.linalg.norm(A[i])
        e2 = A[j] - (A[j] @ e1) * e1
        e2 /= np.linalg.norm(e2)
        xy = fi[..., None] * A[i] + fj[..., None] * A[j]
        # atoms whose plane coordinate matches the cut, with their images inside the plotted cell
        fz = idx / n[kax]
        atoms = []
        for t, (f, el) in enumerate(zip(vd.structure.frac_coords, vd.structure.species)):
            dz = ((f[kax] - fz + 0.5) % 1 - 0.5) * np.linalg.norm(A[kax])
            if abs(dz) > 0.05:
                continue
            for s1 in (-1, 0, 1):
                for s2 in (-1, 0, 1):
                    p = ((f[i] % 1) + s1) * A[i] + ((f[j] % 1) + s2) * A[j]
                    atoms.append([p @ e1, p @ e2, R[t], t])
        key = run
        store.update({f"{key}__X": xy @ e1, f"{key}__Y": xy @ e2, f"{key}__rho": vd.rho.data[sl],
                      f"{key}__pro": pro[sl], f"{key}__drho": d[sl], f"{key}__atoms": np.array(atoms),
                      f"{key}__species": np.array([vd.structure.species[int(x[3])] for x in atoms]),
                      f"{key}__meta": np.array([label, "abc"[i] + "abc"[j]])})
        print(run, {k_: round(v_, 4) for k_, v_ in row.items() if isinstance(v_, float)})
    np.savez_compressed(OUT / "slices.npz", **store)
    pd.DataFrame(radial).to_csv(OUT / "radial_examples.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "examples.csv", index=False)


if __name__ == "__main__":
    main()
