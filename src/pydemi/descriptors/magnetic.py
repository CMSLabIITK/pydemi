"""
pydemi.descriptors.magnetic
===========================
Magnetic domain (spec §8.3), from the magnetization density m = rho_up -
rho_down (electrons / Angstrom^3; integrated, mu_B).

Every entry returns 0.0 for a non-magnetic structure -- never NaN -- with
``magnetic = False`` in the metadata. A spin-polarized run counts as
magnetic when sum_k |m_k| dV exceeds MAGNETIC_TOL mu_B per atom; below
that the ratios would be ratios of numerical noise.

Non-collinear runs are handled as vectors: |m| is the vector norm, mu_i and
M_net are vector sums, and spin_frustration uses vector norms.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..constants import MAGNETIC_TOL
from ..core.partition import partition_of
from ..fields.density import abs_m
from ..io.base import VolumetricData
from .registry import metadata_hook, options

F64 = NDArray[np.float64]


def magnetization_components(vd: VolumetricData) -> list[F64]:
    """[m] (collinear), [m_x, m_y, m_z] (non-collinear) or [] (not spin-polarized)."""
    if vd.magnetization_vector is not None:
        return [np.asarray(g.data, dtype=np.float64) for g in vd.magnetization_vector]
    if vd.magnetization is not None:
        return [np.asarray(vd.magnetization.data, dtype=np.float64)]
    return []


def total_moments(vd: VolumetricData) -> tuple[float, float]:
    """(M_abs, M_net) = (sum |m| dV, |sum m dV|) in mu_B (extensive: metadata only)."""
    comps = magnetization_components(vd)
    if not comps:
        return 0.0, 0.0
    dV = vd.rho.dV
    M_abs = float(np.sum(abs_m(vd), dtype=np.float64) * dV)
    M_net = float(np.linalg.norm([np.sum(c, dtype=np.float64) * dV for c in comps]))
    return M_abs, M_net


def is_magnetic(vd: VolumetricData) -> bool:
    M_abs, _ = total_moments(vd)
    return M_abs > MAGNETIC_TOL * vd.structure.n_atoms


def site_moments(vd: VolumetricData) -> F64:
    """mu_i = sum_k w_i(k) m_k dV under the requested partition: (n_atoms,) collinear,
    (n_atoms, 3) non-collinear, zeros when not spin-polarized."""
    o = options(vd)
    key = ("site_moments", o.partition)
    if key not in vd.cache:
        comps = magnetization_components(vd)
        n = vd.structure.n_atoms
        if not comps:
            vd.cache[key] = np.zeros(n)
        else:
            part = partition_of(vd, o.partition)
            mu = np.stack([part.site_sum(c) * vd.rho.dV for c in comps], axis=-1)
            vd.cache[key] = mu[:, 0] if mu.shape[1] == 1 else mu
    out: F64 = vd.cache[key]
    return out


def site_moment_magnitudes(vd: VolumetricData) -> F64:
    """Signed mu_i (collinear) or |mu_i| (non-collinear)."""
    mu = site_moments(vd)
    return mu if mu.ndim == 1 else np.asarray(np.linalg.norm(mu, axis=1), dtype=np.float64)


@metadata_hook
def _magnetic_metadata(vd: VolumetricData) -> dict[str, Any]:
    M_abs, M_net = total_moments(vd)
    return {"magnetic": is_magnetic(vd), "M_abs": M_abs, "M_net": M_net}
