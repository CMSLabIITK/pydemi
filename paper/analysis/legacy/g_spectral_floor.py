"""Spectral round-off floor: voxel-count vs charge-weighted Laplacian fraction
for an isolated Gaussian (alpha = 2 / A^2) in a 7 A box, 80^3 grid."""
import json

import numpy as np

from common import OUT
from pydemi import Engine, Grid, Structure
from pydemi.descriptors import tier1
from pydemi.testing import GaussianSuperposition, gaussian_fraction_within

alpha, L, n = 2.0, 7.0, 80
s = Structure(np.eye(3) * L, ["X"], [[0.5, 0.5, 0.5]])
rho, _, _ = GaussianSuperposition(s, alpha, 1.0, tol=1e-30).on_grid(Grid(s.lattice, (n,) * 3), False)
r0 = np.sqrt(1.5 / alpha)
out = {"exact_lnf": 4 / 3 * np.pi * r0 ** 3 / L ** 3, "exact_lnf_rho": float(gaussian_fraction_within(r0, alpha))}
for m in ("fd", "spectral"):
    t = tier1(Engine(s, {"rho": rho}, method=m))
    out[f"lnf_{m}"], out[f"lnf_rho_{m}"] = t["lnf"], t["lnf_rho"]
(OUT / "spectral_floor.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
