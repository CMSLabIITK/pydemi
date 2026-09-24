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

from typing import Any, Callable

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


# ----------------------------------------------------------------------
# descriptors
# ----------------------------------------------------------------------

from ..operators.fractions import shell_fraction  # noqa: E402
from ..operators.moments import radial_moment  # noqa: E402
from .registry import Result, Sentinel, geometry, masks, register  # noqa: E402

_NM = {"non_magnetic": 0.0}


def _magnetic(fn: "Callable[[VolumetricData], Result]") -> "Callable[[VolumetricData], Result]":
    def wrapped(vd: VolumetricData) -> Result:
        return fn(vd) if is_magnetic(vd) else Sentinel(0.0, "non_magnetic")
    wrapped.__doc__ = fn.__doc__
    return wrapped


def _reg(name: str, units: str, requires: list[str], rng: tuple[float, float] = (0.0, np.inf)
         ) -> "Callable[[Callable[[VolumetricData], Result]], Callable[[VolumetricData], Result]]":
    def deco(fn: "Callable[[VolumetricData], Result]") -> "Callable[[VolumetricData], Result]":
        register(name=name, domain="magnetic", field="magnetization", requires=requires,
                 units=units, range=rng, sentinel_cases=_NM, doc=fn.__doc__)(_magnetic(fn))
        return fn
    return deco


@_reg("M_abs_per_atom", "mu_B/atom", ["magnetization"])
def M_abs_per_atom(vd: VolumetricData) -> Result:
    """M_abs_per_atom = sum_k |m_k| dV / n_atoms

    The intensive version of M_abs = sum_k |m_k| dV (which is in the metadata).
    """
    return total_moments(vd)[0] / vd.structure.n_atoms


@_reg("M_net_per_atom", "mu_B/atom", ["magnetization"])
def M_net_per_atom(vd: VolumetricData) -> Result:
    """M_net_per_atom = |sum_k m_k dV| / n_atoms

    The intensive version of M_net = |sum_k m_k dV| (in the metadata); vector
    norm for non-collinear runs.
    """
    return total_moments(vd)[1] / vd.structure.n_atoms


@_reg("m1_spin", "Angstrom", ["magnetization", "geometry"])
def m1_spin(vd: VolumetricData) -> Result:
    """m1_spin = sum_k |m_k| r_k / sum_k |m_k|"""
    return radial_moment(abs_m(vd), geometry(vd).distance, 1, "abs")


@_reg("sigma_r2_spin", "Angstrom^2", ["magnetization", "geometry"])
def sigma_r2_spin(vd: VolumetricData) -> Result:
    """sigma_r2_spin = sum_k |m_k| r_k^2 / sum_k |m_k| - m1_spin^2"""
    r = geometry(vd).distance
    a = radial_moment(abs_m(vd), r, 1, "abs")
    return radial_moment(abs_m(vd), r, 2, "abs") - a * a


@_reg("f_bond_spin", "dimensionless", ["magnetization", "geometry", "shells"], (0.0, 1.0))
def f_bond_spin(vd: VolumetricData) -> Result:
    """f_bond_spin = sum_{k in bond} |m_k| / sum_k |m_k|"""
    return shell_fraction(abs_m(vd), masks(vd).bond)


@_reg("mu_site_std", "mu_B", ["magnetization", "partition"])
def mu_site_std(vd: VolumetricData) -> Result:
    """mu_site_std = std over i of mu_i,  mu_i = sum_k w_i(k) m_k dV

    Non-collinear: sqrt(mean_i |mu_i - mean mu|^2) with vector moments.
    """
    mu = site_moments(vd).reshape(vd.structure.n_atoms, -1)
    return float(np.sqrt(np.mean(np.sum((mu - mu.mean(axis=0)) ** 2, axis=1))))


@_reg("spin_frustration", "dimensionless", ["magnetization", "partition"], (0.0, 1.0))
def spin_frustration(vd: VolumetricData) -> Result:
    """spin_frustration = 1 - |sum_i mu_i| / sum_i |mu_i|

    0 for a ferromagnet, 1 for a perfectly compensated antiferromagnet;
    vector norms for non-collinear runs.
    """
    mu = site_moments(vd).reshape(vd.structure.n_atoms, -1)
    denom = float(np.linalg.norm(mu, axis=1).sum())
    if denom == 0.0:
        return Sentinel(0.0, "non_magnetic")
    return 1.0 - float(np.linalg.norm(mu.sum(axis=0))) / denom


@_reg("spin_charge_correlation", "dimensionless", ["magnetization"], (-1.0, 1.0))
def spin_charge_correlation(vd: VolumetricData) -> Result:
    """spin_charge_correlation = Pearson r(rho_k, |m_k|) over voxels"""
    a, b = vd.rho.data.ravel(), abs_m(vd).ravel()
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return Sentinel(0.0, "non_magnetic")
    return float(np.corrcoef(a, b)[0, 1])
