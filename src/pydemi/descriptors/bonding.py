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
# derived fields: kinetic terms, ELF, potential (cached on vd)
# ----------------------------------------------------------------------

from typing import Any  # noqa: E402

from ..constants import ELF_LOCALIZED, NCI_RHO_MAX_AU, NCI_S_MAX, S_PREFACTOR  # noqa: E402
from ..core.grid import fourier_interpolate  # noqa: E402
from ..fields.elf import KineticTerms, elf_d, energy_densities, kinetic_terms  # noqa: E402
from ..fields.potential import (centred, electrostatic_potential, hartree_potential,  # noqa: E402
                                ion_charges)
from ..io.base import FloatArray  # noqa: E402
from .registry import field_values, metadata_hook, options, register_field  # noqa: E402


def kinetic(vd: VolumetricData) -> KineticTerms:
    """Kirzhnits kinetic terms of rho (atomic units), shared by ELF_D, g, v, H and s."""
    o = options(vd)
    key = ("kinetic", o.derivative_backend, o.fd_order, o.laplacian_method)
    if key not in vd.cache:
        d = field_derivatives(vd, "rho")
        vd.cache[key] = kinetic_terms(vd.rho.data, d.gradient_norm, laplacian(vd))
    out: KineticTerms = vd.cache[key]
    return out


def elf_source(vd: VolumetricData) -> str:
    """'file' or 'reconstruct', resolving elf_source='auto'."""
    src = options(vd).elf_source
    if src == "auto":
        return "file" if vd.elf is not None else "reconstruct"
    if src == "file" and vd.elf is None:
        raise ValueError("elf_source='file' but no ELFCAR was read (pass elf= to read_vasp)")
    return src


def _elf_file(vd: VolumetricData) -> FloatArray:
    assert vd.elf is not None
    return vd.elf.data


def _elf_reconstructed(vd: VolumetricData) -> FloatArray:
    key = ("field", "elf_d", options(vd).derivative_backend, options(vd).fd_order,
           options(vd).laplacian_method)
    if key not in vd.cache:
        vd.cache[key] = elf_d(kinetic(vd))
    out: FloatArray = vd.cache[key]
    return out


register_field("elf_file", _elf_file)
register_field("elf_d", _elf_reconstructed)


def elf_field_name(vd: VolumetricData) -> str:
    return "elf_file" if elf_source(vd) == "file" else "elf_d"


def potential_source(vd: VolumetricData) -> str:
    """'locpot', 'hartree' or 'esp', resolving potential_source='auto' (LOCPOT when read)."""
    src = options(vd).potential_source
    if src == "auto":
        return "locpot" if vd.potential is not None else "hartree"
    if src == "locpot" and vd.potential is None:
        raise ValueError("potential_source='locpot' but no LOCPOT was read (pass locpot=)")
    return src


def potential(vd: VolumetricData) -> FloatArray:
    """The potential (eV, cell average zero) from the requested source."""
    src = potential_source(vd)
    key = ("field", "potential", src)
    if key not in vd.cache:
        if src == "locpot":
            assert vd.potential is not None
            vd.cache[key] = centred(vd.potential.data)
        elif src == "hartree":
            vd.cache[key] = hartree_potential(vd.rho.data, vd.lattice)
        else:
            q = ion_charges(vd.structure.species, vd.density_source == "all_electron", vd.zval)
            vd.cache[key] = electrostatic_potential(vd.rho.data, vd.structure, q)
    out: FloatArray = vd.cache[key]
    return out


register_field("potential", potential)


def site_potentials(vd: VolumetricData) -> FloatArray:
    """V_site: the potential at each nucleus (eV), by exact Fourier interpolation."""
    return fourier_interpolate(potential(vd), vd.structure.frac_coords)


@metadata_hook
def _field_metadata(vd: VolumetricData) -> dict[str, Any]:
    o = options(vd)
    out: dict[str, Any] = {}
    if "bonding" in o.domains:
        out["elf_source"] = elf_source(vd)
        out["potential_source"] = potential_source(vd)
    return out


# ----------------------------------------------------------------------
# Hessian: ellipticity
# ----------------------------------------------------------------------

def _eigenvalues(vd: VolumetricData) -> FloatArray:
    return field_derivatives(vd, "rho").hessian_eigenvalues


def _ellipticity(vd: VolumetricData) -> "FloatArray | Sentinel":
    ev = _eigenvalues(vd)
    sel = masks(vd).bond & (ev[..., 1] < 0)
    if not sel.any():
        return Sentinel(0.0, "no_bond_voxels")
    return np.asarray(ev[..., 0][sel] / ev[..., 1][sel] - 1.0)


@register(name="ellip_bond_avg", domain="bonding", field="rho", requires=["hessian", "shells"],
          units="dimensionless", range=(0.0, np.inf), sentinel_cases={"no_bond_voxels": 0.0})
def ellip_bond_avg(vd: VolumetricData) -> Result:
    """ellip_bond_avg = mean of lambda1/lambda2 - 1 over {k in bond, lambda2 < 0}

    lambda1 <= lambda2 <= lambda3 are the Hessian eigenvalues of rho. Near a
    bond path with lambda2 -> 0- the ratio diverges, so the mean is dominated
    by few voxels. No such voxel: 0.0, flagged.
    """
    e = _ellipticity(vd)
    return e if isinstance(e, Sentinel) else float(np.mean(e))


@register(name="ellip_bond_std", domain="bonding", field="rho", requires=["hessian", "shells"],
          units="dimensionless", range=(0.0, np.inf), sentinel_cases={"no_bond_voxels": 0.0})
def ellip_bond_std(vd: VolumetricData) -> Result:
    """ellip_bond_std = std of lambda1/lambda2 - 1 over {k in bond, lambda2 < 0}"""
    e = _ellipticity(vd)
    return e if isinstance(e, Sentinel) else float(np.std(e))


# ----------------------------------------------------------------------
# local energy densities (Abramov), atomic units
# ----------------------------------------------------------------------

def _bond_valid(vd: VolumetricData) -> "NDArray[np.bool_] | Sentinel":
    sel = masks(vd).bond & kinetic(vd).valid
    return sel if sel.any() else Sentinel(0.0, "empty_region")


@register(name="f_H_negative", domain="bonding", field="rho", requires=["laplacian", "shells"],
          units="dimensionless", range=(0.0, 1.0), sentinel_cases={"empty_region": 0.0})
def f_H_negative(vd: VolumetricData) -> Result:
    """f_H_negative = (1/N_bond) sum_{k in bond} 1(H_k < 0),  H = g + v

    g = C_F rho^(5/3) + (1/72)|grad rho|^2/rho + (1/6) lap rho, v = (1/4) lap rho - 2 g
    (atomic units). Voxels below the density floor are excluded.
    """
    sel = _bond_valid(vd)
    if isinstance(sel, Sentinel):
        return sel
    return float(np.mean(energy_densities(kinetic(vd))[2][sel] < 0.0))


@register(name="H_bond_mean", domain="bonding", field="rho", requires=["laplacian", "shells"],
          units="hartree/bohr^3", sentinel_cases={"empty_region": 0.0})
def H_bond_mean(vd: VolumetricData) -> Result:
    """H_bond_mean = <H_k> over the bond shell,  H = g + v = (1/4) lap rho - g (a.u.)"""
    sel = _bond_valid(vd)
    if isinstance(sel, Sentinel):
        return sel
    return float(np.mean(energy_densities(kinetic(vd))[2][sel]))


@register(name="G_over_rho", domain="bonding", field="rho", requires=["laplacian", "shells"],
          units="hartree/electron", range=(-np.inf, np.inf), sentinel_cases={"empty_region": 0.0})
def G_over_rho(vd: VolumetricData) -> Result:
    """G_over_rho = <g_k / rho_k> over the bond shell (a.u.)"""
    sel = _bond_valid(vd)
    if isinstance(sel, Sentinel):
        return sel
    kt = kinetic(vd)
    return float(np.mean(kt.g[sel] / kt.rho[sel]))


# ----------------------------------------------------------------------
# ELF
# ----------------------------------------------------------------------

def _elf(vd: VolumetricData) -> FloatArray:
    return field_values(vd, elf_field_name(vd))


@register(name="f_ELF_localized", domain="bonding", field="elf", requires=["elf", "shells"],
          units="dimensionless", range=(0.0, 1.0), sentinel_cases={"empty_region": 0.0})
def f_ELF_localized(vd: VolumetricData) -> Result:
    """f_ELF_localized = (1/N_bond) sum_{k in bond} 1(ELF_k > 0.5)

    ELF from the ELFCAR or the reconstruction ELF_D (``elf_source``, recorded
    in the metadata).
    """
    bond = masks(vd).bond
    if not bond.any():
        return Sentinel(0.0, "empty_region")
    return float(np.mean(_elf(vd)[bond] > ELF_LOCALIZED))


@register(name="ELF_bond_avg", domain="bonding", field="elf", requires=["elf", "shells"],
          units="dimensionless", range=(0.0, 1.0), sentinel_cases={"empty_region": 0.0})
def ELF_bond_avg(vd: VolumetricData) -> Result:
    """ELF_bond_avg = <ELF_k> over the bond shell"""
    bond = masks(vd).bond
    if not bond.any():
        return Sentinel(0.0, "empty_region")
    return float(np.mean(_elf(vd)[bond]))


@register(name="ELF_core_valence_contrast", domain="bonding", field="elf",
          requires=["elf", "shells"], units="dimensionless", range=(0.0, np.inf),
          sentinel_cases={"empty_region": 0.0, "zero_denominator": 0.0})
def ELF_core_valence_contrast(vd: VolumetricData) -> Result:
    """ELF_core_valence_contrast = <ELF>_core / <ELF>_bond"""
    m = masks(vd)
    if not (m.core.any() and m.bond.any()):
        return Sentinel(0.0, "empty_region")
    e = _elf(vd)
    b = float(np.mean(e[m.bond]))
    if b == 0.0:
        return Sentinel(0.0, "zero_denominator")
    return float(np.mean(e[m.core])) / b


@register(name="zeta_ELF", domain="bonding", field="elf", requires=["elf", "gradient", "geometry"],
          units="dimensionless", range=(0.0, 1.0), sentinel_cases={"uniform_density": 0.0})
def zeta_ELF(vd: VolumetricData) -> Result:
    """zeta_ELF = 1 - sum_k |grad ELF_k . u_k| / sum_k |grad ELF_k|

    The gradient-anisotropy operator applied to the ELF field. Uniform ELF: 0.0, flagged.
    """
    if is_uniform(_elf(vd)):
        return Sentinel(0.0, "uniform_density")
    g = field_derivatives(vd, elf_field_name(vd)).gradient
    return finite_or(gradient_anisotropy(g, geometry(vd).direction), 0.0, "uniform_density")


# ----------------------------------------------------------------------
# non-covalent interactions
# ----------------------------------------------------------------------

def reduced_gradient(kt: KineticTerms) -> FloatArray:
    """s = |grad rho| / (2 (3 pi^2)^(1/3) rho^(4/3)) (a.u., dimensionless); inf where invalid."""
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.sqrt(kt.grad2) / (S_PREFACTOR * np.maximum(kt.rho, 1e-300) ** (4.0 / 3.0))
    return np.asarray(np.where(kt.valid, s, np.inf))


def _nci(vd: VolumetricData) -> "NDArray[np.bool_] | Sentinel":
    if is_uniform(vd.rho.data):
        return Sentinel(0.0, "uniform_density")
    kt = kinetic(vd)
    return np.asarray((reduced_gradient(kt) < NCI_S_MAX) & (kt.rho < NCI_RHO_MAX_AU))


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
    return float(np.mean(np.sign(lam2) * kinetic(vd).rho[sel]))


# ----------------------------------------------------------------------
# electrostatic potential
# ----------------------------------------------------------------------

@register(name="V_spread", domain="bonding", field="potential", requires=["potential"],
          units="eV", range=(0.0, np.inf))
def V_spread(vd: VolumetricData) -> Result:
    """V_spread = std over sites i of V(R_i)

    The potential at each nucleus by exact Fourier interpolation; the source
    (LOCPOT, Hartree or Hartree + ionic) is recorded in the metadata.
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


# ----------------------------------------------------------------------
# deformation density
# ----------------------------------------------------------------------

from ..core.geometry import pair_regions  # noqa: E402
from ..fields.deformation import promolecule as _promolecule  # noqa: E402
from ..fields.deformation import reference_tables  # noqa: E402


def deformation_reference(vd: VolumetricData) -> str:
    """The reference actually used: 'aeccar0', 'tabulated' or 'custom' (resolves 'auto')."""
    ref = options(vd).deformation_reference
    if ref == "auto":
        return "aeccar0" if vd.core_density is not None else "tabulated"
    if ref == "aeccar0" and vd.core_density is None:
        raise ValueError("deformation_reference='aeccar0' needs AECCAR0 + AECCAR2 "
                         "(read_vasp(..., aeccar0=, aeccar2=))")
    return ref


def promolecule_density(vd: VolumetricData) -> FloatArray:
    """The promolecule on the density grid for the reference in use (cached)."""
    ref = deformation_reference(vd)
    key = ("promolecule", ref, options(vd).custom_reference)
    if key not in vd.cache:
        ae = vd.density_source == "all_electron"
        if ref == "aeccar0":
            assert vd.core_density is not None
            tables = reference_tables(vd.structure, "tabulated", "valence", vd.zval)
            vd.cache[key] = vd.core_density.data + _promolecule(vd.shape, vd.structure, tables)
        else:
            tables = reference_tables(vd.structure, ref, "total" if ae else "valence", vd.zval,
                                      options(vd).custom_reference)
            vd.cache[key] = _promolecule(vd.shape, vd.structure, tables)
    out: FloatArray = vd.cache[key]
    return out


def delta_rho(vd: VolumetricData) -> FloatArray:
    """delta_rho = rho - promolecule."""
    return np.asarray(vd.rho.data - promolecule_density(vd))


register_field("delta_rho", delta_rho)


@metadata_hook
def _deformation_metadata(vd: VolumetricData) -> dict[str, Any]:
    if "bonding" not in options(vd).domains:
        return {}
    d = delta_rho(vd)
    return {"deformation_reference": deformation_reference(vd),
            "def_charge_mismatch": float(np.sum(d) * vd.rho.dV)}


def _drho_parts(vd: VolumetricData) -> tuple[FloatArray, FloatArray, FloatArray]:
    d = delta_rho(vd)
    return d, np.abs(d), geometry(vd).distance


@register(name="m1_def", domain="bonding", field="delta_rho", requires=["promolecule", "geometry"],
          units="Angstrom", range=(0.0, np.inf), sentinel_cases={"zero_deformation": 0.0})
def m1_def(vd: VolumetricData) -> Result:
    """m1_def = sum_k |drho_k| r_k / sum_k |drho_k|,  drho = rho - promolecule"""
    d, a, r = _drho_parts(vd)
    return finite_or(radial_moment(d, r, 1, "abs"), 0.0, "zero_deformation")


@register(name="m2_def", domain="bonding", field="delta_rho", requires=["promolecule", "geometry"],
          units="Angstrom^2", range=(0.0, np.inf), sentinel_cases={"zero_deformation": 0.0})
def m2_def(vd: VolumetricData) -> Result:
    """m2_def = sum_k |drho_k| r_k^2 / sum_k |drho_k|"""
    d, a, r = _drho_parts(vd)
    return finite_or(radial_moment(d, r, 2, "abs"), 0.0, "zero_deformation")


@register(name="sigma_r2_def", domain="bonding", field="delta_rho",
          requires=["promolecule", "geometry"], units="Angstrom^2", range=(0.0, np.inf),
          sentinel_cases={"zero_deformation": 0.0})
def sigma_r2_def(vd: VolumetricData) -> Result:
    """sigma_r2_def = m2_def - m1_def^2"""
    d, a, r = _drho_parts(vd)
    m1v = radial_moment(d, r, 1, "abs")
    return finite_or(radial_moment(d, r, 2, "abs") - m1v * m1v, 0.0, "zero_deformation")


def _accumulated_share(vd: VolumetricData, shell: "NDArray[np.bool_]") -> Result:
    d = delta_rho(vd)
    pos = d > 0
    if not pos.any():
        return Sentinel(0.0, "no_accumulation")
    return float(np.sum(d[shell & pos]) / np.sum(d[pos]))


@register(name="f_bond_def", domain="bonding", field="delta_rho", requires=["promolecule", "shells"],
          units="dimensionless", range=(0.0, 1.0), sentinel_cases={"no_accumulation": 0.0})
def f_bond_def(vd: VolumetricData) -> Result:
    """f_bond_def = sum_{k in bond, drho > 0} drho_k / sum_{drho > 0} drho_k"""
    return _accumulated_share(vd, masks(vd).bond)


@register(name="f_int_def", domain="bonding", field="delta_rho", requires=["promolecule", "shells"],
          units="dimensionless", range=(0.0, 1.0), sentinel_cases={"no_accumulation": 0.0})
def f_int_def(vd: VolumetricData) -> Result:
    """f_int_def = sum_{k in int, drho > 0} drho_k / sum_{drho > 0} drho_k"""
    return _accumulated_share(vd, masks(vd).interstitial)


@register(name="f_bond_dep", domain="bonding", field="delta_rho", requires=["promolecule", "shells"],
          units="dimensionless", range=(0.0, 1.0), sentinel_cases={"no_depletion": 0.0})
def f_bond_dep(vd: VolumetricData) -> Result:
    """f_bond_dep = sum_{k in bond, drho < 0} |drho_k| / sum_{drho < 0} |drho_k|"""
    d = delta_rho(vd)
    neg = d < 0
    if not neg.any():
        return Sentinel(0.0, "no_depletion")
    return float(np.sum(-d[masks(vd).bond & neg]) / np.sum(-d[neg]))


@register(name="def_polarity", domain="bonding", field="delta_rho", requires=["promolecule"],
          units="dimensionless", range=(0.0, np.inf), sentinel_cases={"zero_density": 0.0})
def def_polarity(vd: VolumetricData) -> Result:
    """def_polarity = sum_k |drho_k| dV / Q_tot,  Q_tot = sum_k rho_k dV"""
    Q = float(np.sum(vd.rho.data))
    if Q == 0.0:
        return Sentinel(0.0, "zero_density")
    return float(np.sum(np.abs(delta_rho(vd))) / Q)


def pair_charge_transfer(vd: VolumetricData) -> "FloatArray | Sentinel":
    """int drho dV over the region between each nearest-neighbour pair (electrons).

    The region of pair (i, j) is the set of voxels whose two nearest nuclei
    are i and j (second-order Voronoi cell); pairs from the bond census.
    """
    census = bond_census_of(vd, BOND_TOL)
    if census.length.size == 0:
        return Sentinel(0.0, "no_bonds")
    key = ("pair_regions",)
    if key not in vd.cache:
        vd.cache[key] = pair_regions(vd.shape, vd.structure)
    a, b, t = vd.cache[key]
    d = delta_rho(vd).ravel() * vd.rho.dV
    n = vd.structure.n_atoms
    def code(i: "NDArray[np.int64]", j: "NDArray[np.int64]", s: "NDArray[np.int64]"
             ) -> "NDArray[np.int64]":
        s = s + 50
        return np.asarray(((i * n + j) * 101 + s[:, 0]) * 101 * 101 + s[:, 1] * 101 + s[:, 2])
    voxel_code = code(a, b, t)
    pair_code = code(census.i, census.j, census.shift_wrapped)
    uniq, inv = np.unique(voxel_code, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    pos = np.searchsorted(uniq, pair_code)
    pos = np.minimum(pos, uniq.size - 1)
    found = uniq[pos] == pair_code
    return np.where(found, sums[pos], 0.0)


@register(name="bond_charge_transfer_pair_mean", domain="bonding", field="delta_rho",
          requires=["promolecule", "bond_census"], units="electrons",
          sentinel_cases={"no_bonds": 0.0})
def bond_charge_transfer_pair_mean(vd: VolumetricData) -> Result:
    """bond_charge_transfer_pair_mean = mean over nearest-neighbour pairs of int_{region(i,j)} drho dV

    region(i, j): voxels whose two nearest nuclei are i and j.
    """
    x = pair_charge_transfer(vd)
    return x if isinstance(x, Sentinel) else float(np.mean(x))


@register(name="bond_charge_transfer_pair_std", domain="bonding", field="delta_rho",
          requires=["promolecule", "bond_census"], units="electrons", range=(0.0, np.inf),
          sentinel_cases={"no_bonds": 0.0})
def bond_charge_transfer_pair_std(vd: VolumetricData) -> Result:
    """bond_charge_transfer_pair_std = std over nearest-neighbour pairs of int_{region(i,j)} drho dV"""
    x = pair_charge_transfer(vd)
    return x if isinstance(x, Sentinel) else float(np.std(x))
