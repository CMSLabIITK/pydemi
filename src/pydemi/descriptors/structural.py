"""
pydemi.descriptors.structural
=============================
Structural-domain descriptors (spec §8.2): percolation, the extremum census,
non-nuclear maxima, the density floor, the charge-anisotropy tensor and the
information-theoretic measures.

Intensivity: every count is per unit volume (1/Angstrom^3); the non-nuclear
charge is a fraction of Q_tot; the Shannon entropy and the disequilibrium of
the shape function rho~ = rho / N_e are normalized by the cell volume, which
makes them unchanged under a supercell (the unnormalized S shifts by ln 8
and D divides by 8 for a 2x2x2 supercell).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..constants import ANGSTROM_BOHR, DENSITY_TO_AU, GRADIENT_TO_AU, RHO_FLOOR_AU
from ..io.base import FloatArray, VolumetricData
from ..operators.anisotropy import anisotropy_tensor, fractional_anisotropy
from ..operators.topology import ascent_basins, extremum_census, percolation_levels
from .registry import (Result, Sentinel, field_derivatives, geometry, is_uniform, masks,
                       metadata_hook, options, register)


# ----------------------------------------------------------------------
# percolation
# ----------------------------------------------------------------------

def _percolation(vd: VolumetricData) -> FloatArray:
    key = ("percolation",)
    if key not in vd.cache:
        vd.cache[key] = percolation_levels(vd.rho.data)
    out: FloatArray = vd.cache[key]
    return out


def _perc(axis: int, label: str) -> None:
    def fn(vd: VolumetricData) -> Result:
        return float(_percolation(vd)[axis])
    register(name=f"rho_perc_{label}", domain="structural", field="rho", requires=["percolation"],
             units="e/Angstrom^3",
             doc=f"rho_perc_{label} = max {{c : the super-level set {{rho > c}} spans a{axis + 1} "
                 "under PBC}\n\nThe density at the bottleneck of the best connecting path along "
                 f"a{axis + 1}: high for metals, near zero for ionic and molecular solids. "
                 "(Documented correction: the lowest spanning level is always min rho.)")(fn)


for _a, _l in enumerate("abc"):
    _perc(_a, _l)


@register(name="perc_anisotropy", domain="structural", field="rho", requires=["percolation"],
          units="dimensionless", range=(0.0, np.inf), sentinel_cases={"zero_levels": 0.0})
def perc_anisotropy(vd: VolumetricData) -> Result:
    """perc_anisotropy = (max_alpha - min_alpha) / mean_alpha of rho_perc"""
    p = _percolation(vd)
    m = float(p.mean())
    if m == 0.0:
        return Sentinel(0.0, "zero_levels")
    return float((p.max() - p.min()) / m)


# ----------------------------------------------------------------------
# extremum census
# ----------------------------------------------------------------------

def census(vd: VolumetricData) -> dict[str, Any]:
    key = ("census",)
    if key not in vd.cache:
        vd.cache[key] = extremum_census(vd.rho.data)
    out: dict[str, Any] = vd.cache[key]
    return out


def _count(which: str, text: str) -> None:
    def fn(vd: VolumetricData) -> Result:
        return float(census(vd)[which]) / vd.structure.volume
    register(name=which, domain="structural", field="rho", requires=["census"],
             units="1/Angstrom^3", range=(0.0, np.inf),
             doc=f"{which} = (number of {text}) / V_cell\n\nMaxima and minima by 26-neighbour "
                 "comparison with modular wrapping; saddles from the lower / upper link of the "
                 "Freudenthal triangulation (see pydemi.operators.topology).")(fn)


for _w, _t in (("n_max", "local maxima"), ("n_min", "local minima"),
               ("n_saddle1", "index-1 saddles"), ("n_saddle2", "index-2 saddles")):
    _count(_w, _t)


@metadata_hook
def _census_metadata(vd: VolumetricData) -> dict[str, Any]:
    """euler_consistency for every structure: nonzero = the grid does not resolve the topology."""
    return {"euler_consistency": int(census(vd)["euler_consistency"])}


def _nnm(vd: VolumetricData, r_cut: FloatArray) -> tuple[np.ndarray, np.ndarray]:
    """(flat indices of the non-nuclear maxima, their basins' charge / Q_tot)."""
    c = census(vd)
    maxima = np.flatnonzero(c["maxima"])
    geo = geometry(vd)
    far = geo.distance.ravel()[maxima] > r_cut[geo.atom_index.ravel()[maxima]]
    nnm = maxima[far]
    key = ("basins",)
    if key not in vd.cache:
        vd.cache[key] = ascent_basins(vd.rho.data, vd.lattice.matrix, c["lower26"])
    basin = vd.cache[key].ravel()
    q = float(np.sum(vd.rho.data.ravel()[np.isin(basin, nnm)]))
    total = float(np.sum(vd.rho.data))
    return nnm, np.array([q / total if total != 0.0 else 0.0])


@register(name="n_NNM", domain="structural", field="rho", requires=["census", "geometry"],
          units="1/Angstrom^3", range=(0.0, np.inf))
def n_NNM(vd: VolumetricData) -> Result:
    """n_NNM = (number of local maxima with min_i |r - R_i| > r_cut) / V_cell

    26-neighbour maxima; r_cut = NNM_R_CUT (0.8 Angstrom) by default.
    """
    r_cut = np.full(vd.structure.n_atoms, options(vd).nnm_r_cut)
    return float(_nnm(vd, r_cut)[0].size) / vd.structure.volume


@register(name="Q_NNM", domain="structural", field="rho", requires=["census", "geometry"],
          units="dimensionless", range=(0.0, 1.0))
def Q_NNM(vd: VolumetricData) -> Result:
    """Q_NNM = (charge in the steepest-ascent basins of the non-nuclear maxima) / Q_tot"""
    r_cut = np.full(vd.structure.n_atoms, options(vd).nnm_r_cut)
    return float(_nnm(vd, r_cut)[1][0])


# ----------------------------------------------------------------------
# density floor
# ----------------------------------------------------------------------

@register(name="rho_min", domain="structural", field="rho", requires=[], units="e/Angstrom^3")
def rho_min(vd: VolumetricData) -> Result:
    """rho_min = min_k rho_k

    For a PAW pseudo-density this is usually negative and near a nucleus
    (see the ``paw`` extension's rho_min_int).
    """
    return float(np.min(vd.rho.data))


@register(name="rho_min_ratio", domain="structural", field="rho", requires=[],
          units="dimensionless", sentinel_cases={"zero_density": 0.0})
def rho_min_ratio(vd: VolumetricData) -> Result:
    """rho_min_ratio = rho_min / <rho>_V"""
    m = float(np.mean(vd.rho.data))
    if m == 0.0:
        return Sentinel(0.0, "zero_density")
    return float(np.min(vd.rho.data)) / m


@register(name="rho_int_mean", domain="structural", field="rho", requires=["shells"],
          units="e/Angstrom^3", sentinel_cases={"empty_region": 0.0})
def rho_int_mean(vd: VolumetricData) -> Result:
    """rho_int_mean = mean of rho over the interstitial shell (r > c2)"""
    inter = masks(vd).interstitial
    if not inter.any():
        return Sentinel(0.0, "empty_region")
    return float(np.mean(vd.rho.data[inter]))


# ----------------------------------------------------------------------
# charge-anisotropy tensor
# ----------------------------------------------------------------------

def _tensor(vd: VolumetricData) -> "tuple[FloatArray, float] | Sentinel":
    if is_uniform(vd.rho.data):
        return Sentinel(0.0, "uniform_density")
    T = anisotropy_tensor(field_derivatives(vd, "rho").gradient)
    if not np.isfinite(T).all():
        return Sentinel(0.0, "uniform_density")
    return np.linalg.eigvalsh(T), fractional_anisotropy(T)


def _t(k: int) -> None:
    def fn(vd: VolumetricData) -> Result:
        r = _tensor(vd)
        if isinstance(r, Sentinel):
            return Sentinel(1.0 / 3.0, r.case)
        return float(r[0][k])
    register(name=f"T_eigenvalues_t{k + 1}", domain="structural", field="rho",
             requires=["gradient"], units="dimensionless", range=(0.0, 1.0),
             sentinel_cases={"uniform_density": 1.0 / 3.0},
             doc=f"T_eigenvalues_t{k + 1} = eigenvalue {k + 1} (ascending, t1 <= t2 <= t3) of "
                 "T_ab = sum_k d_a rho_k d_b rho_k / sum_k |grad rho_k|^2 (trace 1)\n\n"
                 "All 1/3 for an isotropic gradient distribution. Rotation-invariant as a "
                 "sorted triple. Uniform density: 1/3, flagged.")(fn)


for _k in range(3):
    _t(_k)


@register(name="charge_FA", domain="structural", field="rho", requires=["gradient"],
          units="dimensionless", range=(0.0, 1.0), sentinel_cases={"uniform_density": 0.0})
def charge_FA(vd: VolumetricData) -> Result:
    """charge_FA = sqrt(3/2) ||T - (1/3) I||_F / ||T||_F

    0 for an isotropic gradient distribution, 1 when every gradient is parallel.
    """
    r = _tensor(vd)
    return r if isinstance(r, Sentinel) else float(r[1])


# ----------------------------------------------------------------------
# information-theoretic measures (atomic units)
# ----------------------------------------------------------------------

def _shape_function(vd: VolumetricData) -> "dict[str, Any] | Sentinel":
    key = ("information", options(vd).derivative_backend, options(vd).fd_order)
    if key not in vd.cache:
        dV = vd.rho.dV * ANGSTROM_BOHR ** 3                          # bohr^3
        V = vd.structure.volume * ANGSTROM_BOHR ** 3
        rho = np.maximum(vd.rho.data, 0.0) * DENSITY_TO_AU
        N = float(np.sum(rho) * dV)
        if N <= 0.0:
            vd.cache[key] = Sentinel(0.0, "zero_density")
        else:
            p = rho / N
            pos = p > 0
            S = -float(np.sum(p[pos] * np.log(p[pos])) * dV)
            D = float(np.sum(p * p) * dV)
            valid = rho > RHO_FLOOR_AU
            g2 = (field_derivatives(vd, "rho").gradient_norm * GRADIENT_TO_AU / N) ** 2
            fisher = float(np.sum(g2[valid] / p[valid]) * dV)
            vd.cache[key] = {"S": S - np.log(V), "D": D * V, "I": fisher}
    out: "dict[str, Any] | Sentinel" = vd.cache[key]
    return out


@register(name="shannon_entropy", domain="structural", field="rho", requires=[],
          units="dimensionless", range=(-np.inf, 0.0), sentinel_cases={"zero_density": 0.0})
def shannon_entropy(vd: VolumetricData) -> Result:
    """shannon_entropy = -int rho~ ln rho~ dV - ln V,  rho~ = rho / N_e

    Normalized by the cell volume (intensive; unit-independent; 0 for a
    uniform density, negative otherwise -- minus the Kullback-Leibler
    divergence of rho~ from the uniform distribution).
    """
    r = _shape_function(vd)
    return r if isinstance(r, Sentinel) else float(r["S"])


@register(name="fisher_information", domain="structural", field="rho", requires=["gradient"],
          units="1/bohr^2", range=(0.0, np.inf), sentinel_cases={"zero_density": 0.0})
def fisher_information(vd: VolumetricData) -> Result:
    """fisher_information = int |grad rho~|^2 / rho~ dV  (a.u.; voxels above the density floor)"""
    r = _shape_function(vd)
    return r if isinstance(r, Sentinel) else float(r["I"])


@register(name="disequilibrium", domain="structural", field="rho", requires=[],
          units="dimensionless", range=(0.0, np.inf), sentinel_cases={"zero_density": 0.0})
def disequilibrium(vd: VolumetricData) -> Result:
    """disequilibrium = V int rho~^2 dV

    Normalized by the cell volume (intensive; 1 for a uniform density).
    """
    r = _shape_function(vd)
    return r if isinstance(r, Sentinel) else float(r["D"])


@register(name="LMC_complexity", domain="structural", field="rho", requires=[],
          units="dimensionless", range=(0.0, np.inf), sentinel_cases={"zero_density": 0.0})
def LMC_complexity(vd: VolumetricData) -> Result:
    """LMC_complexity = D e^S  (D, S the unnormalized disequilibrium and entropy; = 1 when uniform)"""
    r = _shape_function(vd)
    return r if isinstance(r, Sentinel) else float(r["D"] * np.exp(r["S"]))
