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
from numpy.typing import NDArray

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



# ----------------------------------------------------------------------
# derived fields: the potential (cached on vd)
# ----------------------------------------------------------------------

from ..constants import (DENSITY_TO_AU, GRADIENT_TO_AU, NCI_RHO_MAX_AU, NCI_S_MAX,  # noqa: E402
                         RHO_FLOOR_AU, S_PREFACTOR)
from ..core.grid import fourier_interpolate  # noqa: E402
from ..fields.potential import hartree_potential  # noqa: E402
from ..io.base import FloatArray  # noqa: E402
from .registry import options, register_field  # noqa: E402


def potential(vd: VolumetricData) -> FloatArray:
    """The electronic Hartree potential of rho (eV, cell average zero), cached."""
    key = ("field", "potential", "hartree")
    if key not in vd.cache:
        vd.cache[key] = hartree_potential(vd.rho.data, vd.lattice)
    out: FloatArray = vd.cache[key]
    return out


register_field("potential", potential)


def site_potentials(vd: VolumetricData) -> FloatArray:
    """V_site: the potential at each nucleus (eV), by exact Fourier interpolation."""
    return fourier_interpolate(potential(vd), vd.structure.frac_coords)


# ----------------------------------------------------------------------
# Hessian: ellipticity
# ----------------------------------------------------------------------

def _eigenvalues(vd: VolumetricData) -> FloatArray:
    return field_derivatives(vd, "rho").hessian_eigenvalues


@register(name="ellip_bond_bounded_avg", domain="bonding", field="rho",
          requires=["hessian", "shells"], units="dimensionless", range=(0.0, 1.0),
          sentinel_cases={"no_bond_voxels": 0.0})
def ellip_bond_bounded_avg(vd: VolumetricData) -> Result:
    """ellip_bond_bounded_avg = mean of 1 - lambda2/lambda1 over {k in bond, lambda2 < 0}

    lambda1 <= lambda2 <= lambda3 are the Hessian eigenvalues of rho. Per
    voxel 1 - lambda2/lambda1 = e / (1 + e) with e = lambda1/lambda2 - 1 the
    usual ellipticity, a monotone map of e onto [0, 1): 0 for cylindrical
    symmetry (lambda1 = lambda2), -> 1 as lambda2 -> 0-. Unlike the mean of e
    (dropped: it diverges as lambda2 -> 0- and changed by a median 22%
    between FFT and FD4 derivatives on the 6,059-structure dataset), it
    cannot diverge; it changed by a median 0.2% between FFT and FD4, 1.1%
    with FD2 and 0.2% on an 80% grid (48 structures). No such voxel: 0.0,
    flagged.
    """
    ev = _eigenvalues(vd)
    sel = masks(vd).bond & (ev[..., 1] < 0)
    if not sel.any():
        return Sentinel(0.0, "no_bond_voxels")
    return float(np.mean(1.0 - ev[..., 1][sel] / ev[..., 0][sel]))


# ----------------------------------------------------------------------
# non-covalent interactions (atomic units)
# ----------------------------------------------------------------------

def _rho_au(vd: VolumetricData) -> tuple[FloatArray, FloatArray]:
    """(rho in e/bohr^3, clipped at 0; reduced gradient s, inf below RHO_FLOOR_AU), cached."""
    o = options(vd)
    key = ("rho_au", o.derivative_backend, o.fd_order)
    if key not in vd.cache:
        rho = np.maximum(vd.rho.data, 0.0) * DENSITY_TO_AU
        grad = field_derivatives(vd, "rho").gradient_norm * GRADIENT_TO_AU
        with np.errstate(divide="ignore", invalid="ignore"):
            s = grad / (S_PREFACTOR * np.maximum(rho, 1e-300) ** (4.0 / 3.0))
        vd.cache[key] = (rho, np.asarray(np.where(rho > RHO_FLOOR_AU, s, np.inf)))
    out: tuple[FloatArray, FloatArray] = vd.cache[key]
    return out


def _nci(vd: VolumetricData) -> "NDArray[np.bool_] | Sentinel":
    if is_uniform(vd.rho.data):
        return Sentinel(0.0, "uniform_density")
    rho, s = _rho_au(vd)
    return np.asarray((s < NCI_S_MAX) & (rho < NCI_RHO_MAX_AU))


@register(name="f_NCI", domain="bonding", field="rho", requires=["gradient"],
          units="dimensionless", range=(0.0, 1.0), sentinel_cases={"uniform_density": 0.0})
def f_NCI(vd: VolumetricData) -> Result:
    """f_NCI = (1/N) sum_k 1(s_k < 0.5 and rho_k < 0.05 a.u.),  s = |grad rho| / (2 (3 pi^2)^(1/3) rho^(4/3))

    Uniform density (s = 0 everywhere, no non-covalent region): 0.0, flagged.
    """
    sel = _nci(vd)
    return sel if isinstance(sel, Sentinel) else float(np.mean(sel))


@register(name="NCI_attractive", domain="bonding", field="rho", requires=["gradient", "hessian"],
          units="dimensionless", range=(0.0, 1.0),
          sentinel_cases={"uniform_density": 0.0, "no_nci_voxels": 0.0})
def NCI_attractive(vd: VolumetricData) -> Result:
    """NCI_attractive = fraction of NCI voxels with lambda2 < 0"""
    sel = _nci(vd)
    if isinstance(sel, Sentinel):
        return sel
    if not sel.any():
        return Sentinel(0.0, "no_nci_voxels")
    return float(np.mean(_eigenvalues(vd)[..., 1][sel] < 0))


@register(name="sign_lambda2_rho_mean", domain="bonding", field="rho",
          requires=["gradient", "hessian"], units="e/bohr^3",
          sentinel_cases={"uniform_density": 0.0, "no_nci_voxels": 0.0})
def sign_lambda2_rho_mean(vd: VolumetricData) -> Result:
    """sign_lambda2_rho_mean = mean of sign(lambda2) rho_k over NCI voxels (rho in a.u.)"""
    sel = _nci(vd)
    if isinstance(sel, Sentinel):
        return sel
    if not sel.any():
        return Sentinel(0.0, "no_nci_voxels")
    lam2 = _eigenvalues(vd)[..., 1][sel]
    return float(np.mean(np.sign(lam2) * _rho_au(vd)[0][sel]))


# ----------------------------------------------------------------------
# electrostatic potential
# ----------------------------------------------------------------------

@register(name="V_spread", domain="bonding", field="potential", requires=["potential"],
          units="eV", range=(0.0, np.inf))
def V_spread(vd: VolumetricData) -> Result:
    """V_spread = std over sites i of V(R_i)

    V is the electronic Hartree potential of rho; its value at each nucleus by
    exact Fourier interpolation.
    """
    return float(np.std(site_potentials(vd)))


@register(name="V_int_min", domain="bonding", field="potential", requires=["potential", "shells"],
          units="eV", sentinel_cases={"empty_region": 0.0})
def V_int_min(vd: VolumetricData) -> Result:
    """V_int_min = min over interstitial voxels of V_k (cell average of V set to 0)"""
    inter = masks(vd).interstitial
    if not inter.any():
        return Sentinel(0.0, "empty_region")
    return float(potential(vd)[inter].min())


# ----------------------------------------------------------------------
# bond midpoints
# ----------------------------------------------------------------------

from ..constants import BOND_TOL  # noqa: E402
from ..core.geometry import bond_census_of  # noqa: E402


def _rho_mid(vd: VolumetricData) -> "FloatArray | Sentinel":
    census = bond_census_of(vd, BOND_TOL)
    if census.length.size == 0:
        return Sentinel(0.0, "no_bonds")
    key = ("rho_mid", BOND_TOL)
    if key not in vd.cache:
        vd.cache[key] = fourier_interpolate(vd.rho.data, census.midpoints_frac(vd.structure))
    out: FloatArray = vd.cache[key]
    return out


@register(name="rho_mid_mean", domain="bonding", field="rho", requires=["bond_census"],
          units="e/Angstrom^3", range=(-np.inf, np.inf), sentinel_cases={"no_bonds": 0.0})
def rho_mid_mean(vd: VolumetricData) -> Result:
    """rho_mid_mean = mean of rho at nearest-neighbour bond midpoints

    Neighbours: the first shell of each atom, |R_ij| <= (1 + BOND_TOL) d_i;
    rho at the midpoints by exact Fourier interpolation.
    """
    x = _rho_mid(vd)
    return x if isinstance(x, Sentinel) else float(np.mean(x))


@register(name="rho_mid_std", domain="bonding", field="rho", requires=["bond_census"],
          units="e/Angstrom^3", range=(0.0, np.inf), sentinel_cases={"no_bonds": 0.0})
def rho_mid_std(vd: VolumetricData) -> Result:
    """rho_mid_std = std of rho at nearest-neighbour bond midpoints"""
    x = _rho_mid(vd)
    return x if isinstance(x, Sentinel) else float(np.std(x))
