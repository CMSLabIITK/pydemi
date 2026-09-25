"""Magnetization density of example materials: planar cuts, radial profiles and site moments.

For each example: one grid plane through a magnetic atom (spanned by two lattice vectors)
with m = rho_up - rho_down and rho; the in-plane atoms and their PAW radii; radial
profiles of the positive and negative magnetization against the distance to the nearest
nucleus (0.05 A bins, per mu_B of sum |m| dV); the site moments mu_i (nearest-atom
partition) with a check that sum_i mu_i equals sum_k m_k dV; and the descriptors.

Usage:  python examples.py   -> ../data/slices.npz, radial_examples.csv, sites_examples.csv,
                                examples.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "paper" / "analysis"))
from common import load  # noqa: E402

from pydemi.descriptors.magnetic import site_moments  # noqa: E402
from pydemi.descriptors.registry import (REGISTRY, Sentinel, augmentation_radii,  # noqa: E402
                                         geometry, make_options)

OUT = Path(__file__).resolve().parents[1] / "data"
NAMES = ["m1_spin", "sigma_r2_spin", "f_bond_spin", "mu_site_std", "spin_frustration",
         "spin_charge_correlation", "M_abs_per_atom", "M_net_per_atom"]
EXAMPLES = [("Fe_229", "bcc Fe", (0, 1)), ("Ni_225", "fcc Ni", (0, 1)), ("FeNi3_221", "FeNi3 (L1_2)", (0, 1)),
            ("Fe3Si_225", "Fe3Si (D0_3)", (0, 1)), ("NiO_225", "NiO (FM cell)", (0, 1)),
            ("ErFe2_227", "ErFe2 (Laves, ferrimagnet)", (0, 1)), ("UN_225", "UN (5f)", (0, 1)),
            ("NdHoIn2_225", "NdHoIn2 (compensated 4f)", (0, 1))]
BINS = np.arange(0.0, 4.0001, 0.05)


def value(r):
    return r.value if isinstance(r, Sentinel) else float(r)


def main():
    store, radial, sites, rows = {}, [], [], []
    for run, label, (i, j) in EXAMPLES:
        vd = load(run)
        v = vd.with_options(make_options())
        m = vd.magnetization.data
        dV = vd.rho.dV
        geo = geometry(v)
        R = augmentation_radii(v)[0]
        mu = site_moments(v)
        s = vd.structure
        a = np.abs(m)
        inside = geo.distance <= R[geo.atom_index]
        row = {"id": run, "label": label, "n_atoms": s.n_atoms, "elements": " ".join(s.elements),
               "sum_mu": float(mu.sum()), "sum_m_dV": float(m.sum() * dV),
               "abs_share_inside_paw": float(a[inside].sum() / a.sum()),
               "negative_share": float(np.clip(-m, 0, None).sum() / a.sum())}
        for n in NAMES:
            row[n] = value(REGISTRY[n].func(v))
        rows.append(row)
        for t in range(s.n_atoms):
            sites.append({"id": run, "label": label, "site": t, "element": s.species[t], "mu": float(mu[t])})
        r = geo.distance.ravel()
        k = np.digitize(r, BINS) - 1
        ok = (k >= 0) & (k < BINS.size - 1)
        tot = a.sum()
        pos = np.bincount(k[ok], weights=np.clip(m.ravel()[ok], 0, None), minlength=BINS.size - 1) / tot
        neg = np.bincount(k[ok], weights=np.clip(-m.ravel()[ok], 0, None), minlength=BINS.size - 1) / tot
        for lo, hi, x, y in zip(BINS[:-1], BINS[1:], pos, neg):
            radial.append({"id": run, "r_lo": lo, "r_hi": hi, "positive": x, "negative": y})
        # slice through the atom with the largest |mu|
        kax = 3 - i - j
        atom = int(np.argmax(np.abs(mu)))
        idx = int(round(s.frac_coords[atom, kax] * vd.shape[kax])) % vd.shape[kax]
        sl = [slice(None)] * 3
        sl[kax] = idx
        sl = tuple(sl)
        n = vd.shape
        fi, fj = np.meshgrid(np.arange(n[i]) / n[i], np.arange(n[j]) / n[j], indexing="ij")
        A = s.lattice.matrix
        e1 = A[i] / np.linalg.norm(A[i])
        e2 = A[j] - (A[j] @ e1) * e1
        e2 /= np.linalg.norm(e2)
        xy = fi[..., None] * A[i] + fj[..., None] * A[j]
        fz = idx / n[kax]
        atoms, spec = [], []
        for t, (f, el) in enumerate(zip(s.frac_coords, s.species)):
            dz = ((f[kax] - fz + 0.5) % 1 - 0.5) * np.linalg.norm(A[kax])
            if abs(dz) > 0.05:
                continue
            for s1 in (-1, 0, 1):
                for s2 in (-1, 0, 1):
                    p = ((f[i] % 1) + s1) * A[i] + ((f[j] % 1) + s2) * A[j]
                    atoms.append([p @ e1, p @ e2, R[t], mu[t]])
                    spec.append(el)
        store.update({f"{run}__X": xy @ e1, f"{run}__Y": xy @ e2, f"{run}__m": m[sl], f"{run}__rho": vd.rho.data[sl],
                      f"{run}__atoms": np.array(atoms), f"{run}__species": np.array(spec),
                      f"{run}__meta": np.array([label, "abc"[i] + "abc"[j]])})
        print(run, {k_: (round(v_, 4) if isinstance(v_, float) else v_) for k_, v_ in row.items()})
    np.savez_compressed(OUT / "slices.npz", **store)
    pd.DataFrame(radial).to_csv(OUT / "radial_examples.csv", index=False)
    pd.DataFrame(sites).to_csv(OUT / "sites_examples.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "examples.csv", index=False)


if __name__ == "__main__":
    main()
