"""Laplacian and gradient discretizations on real VASP densities.

For N random structures: lnf (fraction of voxels with lap rho < 0) and
zeta from four discretizations, relative to the spectral reference.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import DATASET, OUT, sample_ids
from pydemi import Engine
from pydemi.descriptors.primitives import gradient_anisotropy


def lap_np_gradient_twice(rho, grid):
    """np.gradient applied twice: double-width stencil, one-sided at the boundary planes."""
    n = np.array(rho.shape, float)
    M = grid.metric * np.outer(n, n)
    d = [np.gradient(rho, axis=a) for a in range(3)]
    lap = np.zeros_like(rho)
    for m in range(3):
        lap += M[m, m] * np.gradient(d[m], axis=m)
        for q in range(m + 1, 3):
            lap += M[m, q] * (np.gradient(d[m], axis=q) + np.gradient(d[q], axis=m))
    return lap


def lap_wide_periodic(rho, grid):
    """Double-width stencil, periodic."""
    n = np.array(rho.shape, float)
    M = grid.metric * np.outer(n, n)
    c = lambda f, a: 0.5 * (np.roll(f, -1, a) - np.roll(f, 1, a))
    d = [c(rho, a) for a in range(3)]
    lap = np.zeros_like(rho)
    for m in range(3):
        for q in range(3):
            lap += M[m, q] * c(d[m], q)
    return lap


def grad_np_gradient(rho, grid):
    n = np.array(rho.shape, float)
    d_u = [n[m] * np.gradient(rho, axis=m) for m in range(3)]
    A = grid.inv_lattice
    return np.stack([sum(A[a, m] * d_u[m] for m in range(3)) for a in range(3)], axis=-1)


def one(mid):
    try:
        eng = Engine.from_chgcar(DATASET / mid / "CHGCAR", read_spin=False)
        rho, grid = eng["rho"].values, eng.grid()
        geo = eng.geometry()
        laps = {"np_gradient_twice": lap_np_gradient_twice(rho, grid),
                "wide_periodic": lap_wide_periodic(rho, grid),
                "fd": grid.laplacian(rho, "fd"),
                "spectral": grid.laplacian(rho, "spectral")}
        out = {"id": mid, "shape": list(rho.shape), "n_atoms": eng.structure.n_atoms}
        out.update({f"lnf_{k}": float(np.mean(v < 0)) for k, v in laps.items()})
        out.update({f"lnf_rho_{k}": float(rho[v < 0].sum() / rho.sum()) for k, v in laps.items()})
        out["zeta_np_gradient"] = gradient_anisotropy(grad_np_gradient(rho, grid), geo.direction)
        out["zeta_fd"] = gradient_anisotropy(grid.gradient(rho, "fd"), geo.direction)
        out["zeta_spectral"] = gradient_anisotropy(grid.gradient(rho, "spectral"), geo.direction)
        out["lap_integral_fd"] = float(laps["fd"].sum() * grid.dV)
        return out
    except Exception as exc:
        return {"id": mid, "error": repr(exc)}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    ids = sample_ids(n)
    with ProcessPoolExecutor(16) as pool:
        rows = list(pool.map(one, ids))
    (OUT / "laplacian_stencils.json").write_text(json.dumps(rows, indent=1))
    ok = [r for r in rows if "error" not in r]
    print(f"{len(ok)} of {len(rows)} structures")
    ref = np.array([r["lnf_spectral"] for r in ok])
    for k in ("np_gradient_twice", "wide_periodic", "fd"):
        v = np.array([r[f"lnf_{k}"] for r in ok])
        rel = (v - ref) / ref
        print(f"lnf {k:18s} rel. to spectral: median {np.median(rel):+.3f}  IQR [{np.percentile(rel,25):+.3f}, {np.percentile(rel,75):+.3f}]")
    for k in ("np_gradient", "fd"):
        v = np.array([r[f"zeta_{k}"] for r in ok]); z = np.array([r["zeta_spectral"] for r in ok])
        rel = (v - z) / z
        print(f"zeta {k:12s} rel. to spectral: median {np.median(rel):+.3f}  IQR [{np.percentile(rel,25):+.3f}, {np.percentile(rel,75):+.3f}]")
