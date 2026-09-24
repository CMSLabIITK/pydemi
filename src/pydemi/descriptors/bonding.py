"""
pydemi.descriptors.bonding
==========================
Bonding-domain descriptors (spec §8.1).

Distances r_k and unit vectors u_k are those of the geometry pass (nearest
nucleus, minimum image); shells are core r <= c1, bond c1 < r <= c2,
interstitial r > c2.
"""

from __future__ import annotations

import numpy as np

from ..io.base import VolumetricData
from ..operators.anisotropy import gradient_anisotropy
from ..operators.fractions import shell_fraction
from ..operators.laplacian import concentration, negative_fraction, weighted_negative_fraction
from ..operators.moments import radial_moment
from .registry import (Result, Sentinel, field_derivatives, finite_or, geometry, is_uniform,
                       laplacian, masks, register)

_ZERO = {"zero_density": 0.0}


# ----------------------------------------------------------------------
# Tier 1: gradient anisotropy, radial moments, shell fractions
# ----------------------------------------------------------------------

@register(name="zeta", domain="bonding", field="rho", requires=["gradient", "geometry"],
          units="dimensionless", range=(0.0, 1.0),
          sentinel_cases={"uniform_density": 0.0},
          references=["Tier-1 charge-density descriptor set"])
def zeta(vd: VolumetricData) -> Result:
    """zeta = 1 - sum_k |grad rho_k . u_k| / sum_k |grad rho_k|

    0 exactly for a single spherical atom (the gradient is radial); grows as
    density concentrates off the radial directions, e.g. in bonds. Uniform
    density (0/0): 0.0, flagged.
    """
    if is_uniform(vd.rho.data):
        return Sentinel(0.0, "uniform_density")
    g = field_derivatives(vd, "rho").gradient
    return finite_or(gradient_anisotropy(g, geometry(vd).direction), 0.0, "uniform_density")


@register(name="m1", domain="bonding", field="rho", requires=["geometry"], units="Angstrom",
          range=(0.0, np.inf), sentinel_cases=_ZERO)
def m1(vd: VolumetricData) -> Result:
    """m1 = sum_k rho_k r_k / sum_k rho_k

    Single Slater 1s atom, rho ~ exp(-2 zeta r): m1 = 3/(2 zeta).
    """
    return finite_or(radial_moment(vd.rho.data, geometry(vd).distance, 1, "signed"), 0.0,
                     "zero_density")


@register(name="m2", domain="bonding", field="rho", requires=["geometry"], units="Angstrom^2",
          range=(0.0, np.inf), sentinel_cases=_ZERO)
def m2(vd: VolumetricData) -> Result:
    """m2 = sum_k rho_k r_k^2 / sum_k rho_k

    Single Slater 1s atom: m2 = 3/zeta^2.
    """
    return finite_or(radial_moment(vd.rho.data, geometry(vd).distance, 2, "signed"), 0.0,
                     "zero_density")


@register(name="sigma_r2", domain="bonding", field="rho", requires=["geometry"],
          units="Angstrom^2", range=(0.0, np.inf), sentinel_cases=_ZERO)
def sigma_r2(vd: VolumetricData) -> Result:
    """sigma_r2 = m2 - m1^2

    Single Slater 1s atom: sigma_r2 = 3/(4 zeta^2).
    """
    r = geometry(vd).distance
    a = radial_moment(vd.rho.data, r, 1, "signed")
    b = radial_moment(vd.rho.data, r, 2, "signed")
    return finite_or(b - a * a, 0.0, "zero_density")


def _shell(which: str) -> None:
    label = {"core": "r <= c1", "bond": "c1 < r <= c2", "int": "r > c2"}[which]

    def fn(vd: VolumetricData) -> Result:
        m = masks(vd)
        shell = {"core": m.core, "bond": m.bond, "int": m.interstitial}[which]
        return finite_or(shell_fraction(vd.rho.data, shell), 0.0, "zero_density")

    register(name=f"f_{which}", domain="bonding", field="rho", requires=["geometry", "shells"],
             units="dimensionless", range=(0.0, 1.0), sentinel_cases=_ZERO,
             doc=f"f_{which} = sum_{{k: {label}}} rho_k / sum_k rho_k\n\n"
                 "Shell fraction of the density about the nearest nucleus.")(fn)


for _w in ("core", "bond", "int"):
    _shell(_w)


# ----------------------------------------------------------------------
# Tier 1: Laplacian
# ----------------------------------------------------------------------

@register(name="lnf", domain="bonding", field="rho", requires=["laplacian"],
          units="dimensionless", range=(0.0, 1.0), sentinel_cases={"uniform_density": 0.0})
def lnf(vd: VolumetricData) -> Result:
    """lnf = (1/N) sum_k 1(lap rho_k < 0)

    Volume fraction of charge concentration. Single Slater 1s atom in a box:
    the volume fraction with r < 1/zeta. Uniform density: 0.0 (lap rho = 0), flagged.
    """
    if is_uniform(vd.rho.data):
        return Sentinel(0.0, "uniform_density")
    return negative_fraction(laplacian(vd))


@register(name="lnf_charge_weighted", domain="bonding", field="rho", requires=["laplacian"],
          units="dimensionless", range=(0.0, 1.0),
          sentinel_cases={"uniform_density": 0.0, "zero_density": 0.0})
def lnf_charge_weighted(vd: VolumetricData) -> Result:
    """lnf_charge_weighted = sum_{lap rho_k < 0} rho_k / sum_k rho_k

    Charge fraction in charge-concentration voxels; not interchangeable with
    lnf, and far less sensitive to the discretization (voxels at low density
    carry little weight). Uniform density: 0.0, flagged.
    """
    if is_uniform(vd.rho.data):
        return Sentinel(0.0, "uniform_density")
    return finite_or(weighted_negative_fraction(laplacian(vd), vd.rho.data), 0.0, "zero_density")


@register(name="lap_concentration", domain="bonding", field="rho", requires=["laplacian"],
          units="dimensionless", range=(0.0, 1.0), sentinel_cases={"uniform_density": 0.5})
def lap_concentration(vd: VolumetricData) -> Result:
    """lap_concentration = sum_{lap rho < 0} |lap rho_k| / sum_k |lap rho_k|

    Identically 1/2 for every periodic density: int lap rho dV = 0 exactly on
    a periodic grid (documented correction to the specification; kept for
    completeness, see ``lap_concentration_valence``). Uniform density: 0.5, flagged.
    """
    if is_uniform(vd.rho.data):
        return Sentinel(0.5, "uniform_density")
    return finite_or(concentration(laplacian(vd)), 0.5, "uniform_density")


@register(name="lap_concentration_valence", domain="bonding", field="rho",
          requires=["laplacian", "geometry", "shells"], units="dimensionless", range=(0.0, 1.0),
          sentinel_cases={"uniform_density": 0.5, "empty_region": 0.5})
def lap_concentration_valence(vd: VolumetricData) -> Result:
    """lap_concentration_valence = sum_{r > c1, lap rho < 0} |lap rho_k| / sum_{r > c1} |lap rho_k|

    The Laplacian concentration outside the core shell, where it is not
    fixed at 1/2 by the periodic identity (documented correction). Uniform
    density or no voxel beyond c1: 0.5, flagged.
    """
    if is_uniform(vd.rho.data):
        return Sentinel(0.5, "uniform_density")
    outside = ~masks(vd).core
    if not outside.any():
        return Sentinel(0.5, "empty_region")
    return finite_or(concentration(laplacian(vd), outside), 0.5, "uniform_density")

