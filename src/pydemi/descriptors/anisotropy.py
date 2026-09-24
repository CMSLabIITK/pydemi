"""
pydemi.descriptors.anisotropy
-----------------------------
I1 -- charge anisotropy tensor (reference entries 102-103). Instead of
collapsing gradient directionality into the scalar zeta, keep the tensor

    T_ab = sum_k d_a rho_k d_b rho_k / sum_k |grad rho_k|^2,   tr T = 1

    T_eig_1 >= T_eig_2 >= T_eig_3   eigenvalues (102); all 1/3 when isotropic
    charge_FA = sqrt(3/2) ||T - I/3||_F / ||T||_F      fractional anisotropy (103)

FA is 0 for an isotropic gradient distribution and 1 when every gradient
points along one axis (a density that varies in one direction only).
"""

import numpy as np

from ..engine import RHO, Engine


def anisotropy_tensor(engine: Engine, field: str = RHO) -> np.ndarray:
    g = engine[field].gradient.reshape(-1, 3)
    T = g.T @ g
    trace = np.trace(T)
    return T / trace if trace > 0 else np.full((3, 3), np.nan)


def anisotropy_family(engine: Engine, field: str = RHO) -> dict:
    T = anisotropy_tensor(engine, field)
    if np.isnan(T).any():
        nan = float("nan")
        return {"T_eig_1": nan, "T_eig_2": nan, "T_eig_3": nan, "charge_FA": nan}
    t = np.linalg.eigvalsh(T)[::-1]
    fa = np.sqrt(1.5) * np.linalg.norm(T - np.eye(3) / 3.0) / np.linalg.norm(T)
    return {"T_eig_1": float(t[0]), "T_eig_2": float(t[1]), "T_eig_3": float(t[2]),
            "charge_FA": float(fa)}
