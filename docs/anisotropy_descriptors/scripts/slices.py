"""Planar cuts through real VASP densities for the figure of local anisotropy maps.

For each example structure: one grid plane through an atom, spanned by two lattice
vectors, with rho, |grad rho|, the local non-radial fraction w = 1 - |grad rho . u| / |grad rho|
(the integrand of zeta, weighted by |grad rho| in the sum), ELF_D and the same fraction for
grad ELF. Cartesian coordinates of the plane's grid points are stored for plotting.

Usage:  python slices.py   -> ../data/slices.npz
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "paper" / "analysis"))
from common import load  # noqa: E402

from pydemi.descriptors.bonding import elf_field_name  # noqa: E402
from pydemi.descriptors.registry import REGISTRY, field_derivatives, field_values, geometry, make_options  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data"
# run, label, the two lattice axes spanning the plane (the third is fixed at an atom)
EXAMPLES = [("Si_227", "Si (covalent)", (0, 1)), ("NaCl_225", "NaCl (ionic)", (0, 1)),
            ("Cu_225", "Cu (d metal)", (0, 1)), ("Al_225", "Al (sp metal)", (0, 1)),
            ("BN_194", "h-BN (layered)", (0, 2)), ("Ca2N_166", "Ca2N (electride)", (0, 2))]


def nonradial(g, u):
    gn = np.linalg.norm(g, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = 1.0 - np.abs(np.einsum("...i,...i->...", g, u)) / gn
    return np.where(gn > 0, w, np.nan), gn


def main():
    store = {}
    for run, label, (i, j) in EXAMPLES:
        vd = load(run)
        v = vd.with_options(make_options())
        k = 3 - i - j
        atom = int(np.argmin(np.abs(((vd.structure.frac_coords[:, k] + 0.5) % 1) - 0.5)))
        idx = int(round(vd.structure.frac_coords[atom, k] * vd.shape[k])) % vd.shape[k]
        sl = [slice(None)] * 3
        sl[k] = idx
        sl = tuple(sl)
        g = field_derivatives(v, "rho").gradient[sl]
        u = geometry(v).direction[sl]
        w, gn = nonradial(g, u)
        elf = field_values(v, elf_field_name(v))[sl]
        we, gen = nonradial(field_derivatives(v, elf_field_name(v)).gradient[sl], u)
        n = vd.shape
        fi, fj = np.meshgrid(np.arange(n[i]) / n[i], np.arange(n[j]) / n[j], indexing="ij")
        A = vd.structure.lattice.matrix
        xy2 = fi[..., None] * A[i] + fj[..., None] * A[j]          # Cartesian in-plane positions
        e1 = A[i] / np.linalg.norm(A[i])
        e2 = A[j] - (A[j] @ e1) * e1
        e2 /= np.linalg.norm(e2)
        X, Y = xy2 @ e1, xy2 @ e2
        f = {n_: REGISTRY[n_].func(v) for n_ in ("zeta", "zeta_ELF", "charge_FA")}
        key = run
        store.update({f"{key}__X": X, f"{key}__Y": Y, f"{key}__rho": vd.rho.data[sl], f"{key}__gradnorm": gn,
                      f"{key}__w": w, f"{key}__elf": elf, f"{key}__w_elf": we, f"{key}__gradnorm_elf": gen,
                      f"{key}__meta": np.array([label, str(f["zeta"]), str(f["zeta_ELF"]), str(f["charge_FA"]),
                                                "abc"[i] + "abc"[j]])})
        print(run, {k_: round(float(getattr(x, 'value', x)), 4) for k_, x in f.items()})
    np.savez_compressed(OUT / "slices.npz", **store)


if __name__ == "__main__":
    main()
