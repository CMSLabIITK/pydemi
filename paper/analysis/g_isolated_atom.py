"""FFT round-off in near-empty space: an isolated Gaussian (alpha = 2 / A^2, 1 e) in a
7 A box on an 80^3 grid, bonding descriptors with FFT, FD4 and FD2 derivatives.

Output: out/isolated_atom.json
"""
import json

import pydemi
from common import OUT
from pydemi.validate.analytic import GaussianSuperposition, cubic_cell

s = cubic_cell(7.0, ["H"], [[0.5, 0.5, 0.5]])
vd = GaussianSuperposition(s, [2.0], [1.0], tol=1e-30).volumetric((80, 80, 80))
keys = ["f_H_negative", "lap_concentration_valence", "G_over_rho", "f_NCI", "H_bond_mean",
        "ELF_bond_avg", "f_ELF_localized", "zeta"]
out = {}
for tag, kw in {"fft": {}, "fd4": {"derivative_backend": "fd", "fd_order": 4},
                "fd2": {"derivative_backend": "fd", "fd_order": 2}}.items():
    f = pydemi.featurize(vd, domains=["bonding"], **kw)
    out[tag] = {k: f[k] for k in keys}
(OUT / "isolated_atom.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
