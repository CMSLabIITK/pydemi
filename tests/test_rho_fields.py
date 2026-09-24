"""
Phase 2 (fields derivable from rho alone). Everything the reference defines
in atomic units is checked against an independent a.u. evaluation, so a
wrong Angstrom <-> bohr power fails loudly.
"""

import numpy as np
import pytest

from pydemi import Engine, Grid, Structure
from pydemi.constants import BOHR_ANGSTROM as A0, C_F, COULOMB_EV_ANGSTROM
from pydemi.descriptors import (FAMILIES, REGISTRY, compute_descriptors, names)
from pydemi.descriptors.anisotropy import anisotropy_family
from pydemi.descriptors.dataset import correlation, zeta_ellip_agreement
from pydemi.descriptors.elf import elf_family, sweep_inflection, threshold_sweep
from pydemi.descriptors.hessian import ellipticity_family, nci_family
from pydemi.descriptors.information import information_family
from pydemi.descriptors.kinetic import (elf_d_field, energy_densities, energy_family,
                                        kinetic_terms)
from pydemi.descriptors.potential import (fourier_interpolate, hartree_potential,
                                          site_potentials)
from pydemi.testing import GaussianSuperposition


def _gaussian_engine(alpha_au, N=2.0, L_bohr=14.0, n=64, method="spectral", frac=(0.5,) * 3):
    """One Gaussian atom specified in atomic units, built in Angstrom."""
    s = Structure(np.eye(3) * L_bohr * A0, ["X"], [frac])
    dens = GaussianSuperposition(s, alpha_au / A0 ** 2, N, tol=1e-30)
    grid = Grid(s.lattice, (n, n, n))
    rho, _, _ = dens.on_grid(grid, derivatives=False)
    return Engine(s, {"rho": rho}, method=method), grid


def _au_reference(alpha_au, N, eng):
    """Analytic rho, |grad rho|^2, lap rho in a.u. at every voxel."""
    r = eng.geometry().distance / A0
    rho = N * (alpha_au / np.pi) ** 1.5 * np.exp(-alpha_au * r ** 2)
    grad2 = (2 * alpha_au * r * rho) ** 2
    lap = (4 * alpha_au ** 2 * r ** 2 - 6 * alpha_au) * rho
    return rho, grad2, lap


# ---------------------------------------------------------------- F1 / F5

def test_kinetic_terms_are_in_atomic_units():
    alpha, N = 0.6, 2.0
    eng, _ = _gaussian_engine(alpha, N)
    kt = kinetic_terms(eng)
    rho, grad2, lap = _au_reference(alpha, N, eng)
    np.testing.assert_allclose(kt.rho, rho, atol=1e-10 * rho.max())
    np.testing.assert_allclose(kt.grad2, grad2, atol=1e-8 * grad2.max())
    np.testing.assert_allclose(kt.lap, lap, atol=1e-8 * np.abs(lap).max())


def test_elf_d_and_energy_densities_match_au_formulas():
    alpha, N = 0.6, 2.0
    eng, _ = _gaussian_engine(alpha, N)
    rho, grad2, lap = _au_reference(alpha, N, eng)
    pick = (rho > 1e-4) & (eng.geometry().distance > 0.2)
    g = C_F * rho ** (5 / 3) + grad2 / (72 * rho) + lap / 6
    elf = 1 / (1 + ((g - grad2 / (8 * rho)) / (C_F * rho ** (5 / 3))) ** 2)
    np.testing.assert_allclose(elf_d_field(eng).values[pick], elf[pick], rtol=1e-6)
    H = energy_densities(eng)["H"]
    np.testing.assert_allclose(H[pick], (lap / 4 - g)[pick], rtol=1e-6, atol=1e-12)


def test_uniform_electron_gas():
    # grad rho = lap rho = 0: ELF_D = 1/2, g = C_F rho^{5/3}, H = -g
    rho_au = 0.03
    s = Structure(np.eye(3) * 4.0, ["Na"], [[0, 0, 0]])
    eng = Engine(s, {"rho": np.full((12, 12, 12), rho_au / A0 ** 3)})
    np.testing.assert_allclose(elf_d_field(eng).values, 0.5, rtol=1e-12)
    e = energy_family(eng)
    assert e["f_H_negative"] == 1.0
    assert e["H_bond_mean"] == pytest.approx(-C_F * rho_au ** (5 / 3), rel=1e-10)
    assert e["G_over_rho"] == pytest.approx(C_F * rho_au ** (2 / 3), rel=1e-10)


def test_low_density_voxels_are_excluded():
    s = Structure(np.eye(3) * 4.0, ["Na"], [[0, 0, 0]])
    eng = Engine(s, {"rho": np.full((8, 8, 8), -1e-3)})   # unphysical, all below floor
    assert np.all(elf_d_field(eng).values == 0.0)
    assert np.isnan(energy_family(eng)["H_bond_mean"])


def test_kinetic_cache_survives_elf_d_registration():
    eng, _ = _gaussian_engine(0.6, n=24)
    kt = kinetic_terms(eng)
    elf_d_field(eng)
    assert kinetic_terms(eng) is kt
    eng.add_field("rho", eng["rho"].values * 2)            # replacing rho invalidates it
    assert kinetic_terms(eng) is not kt


# ---------------------------------------------------------------- Family B

def test_elf_family_on_constant_field():
    s = Structure(np.eye(3) * 6.0, ["Si"], [[0, 0, 0]])
    eng = Engine(s, {"rho": np.ones((20, 20, 20)), "elf": np.full((20, 20, 20), 0.7)})
    out = elf_family(eng, "elf", "ELF")
    assert out["f_ELF_localized"] == 1.0
    assert out["ELF_bond_avg"] == pytest.approx(0.7)
    assert out["ELF_threshold_sweep_025"] == 1.0 and out["ELF_threshold_sweep_075"] == 0.0
    assert out["ELF_threshold_sweep_inflection"] == pytest.approx(0.7, abs=0.02)
    assert out["ELF_core_valence_contrast"] == pytest.approx(1.0)
    assert np.isnan(out["zeta_ELF"])                        # no gradient anywhere


def test_threshold_sweep_is_survival_function():
    v = np.linspace(0, 1, 1001)
    t, f = threshold_sweep(v, [0.0, 0.3, 1.0])
    np.testing.assert_allclose(f, [1.0, 0.7, 0.0], atol=2e-3)
    assert sweep_inflection(np.random.default_rng(0).normal(0.62, 0.05, 5000)) == \
        pytest.approx(0.62, abs=0.03)


def test_true_elf_names_only_with_elfcar():
    eng, _ = _gaussian_engine(0.6, n=24)
    d = compute_descriptors(eng, families=("B",))
    assert "f_ELFD_localized" in d and "f_ELF_localized" not in d
    eng.add_field("elf", np.full((12, 12, 12), 0.5))       # coarse grid, like ELFCAR
    d = compute_descriptors(eng, families=("B",))
    assert d["f_ELF_localized"] == 0.0 and "f_ELFD_localized" in d


# ---------------------------------------------------------------- F2

def test_hartree_potential_solves_poisson():
    eng, grid = _gaussian_engine(0.6, n=48)
    V = hartree_potential(eng).values
    rho = eng["rho"].values
    lap = grid.laplacian(V, "spectral")
    np.testing.assert_allclose(lap, -4 * np.pi * COULOMB_EV_ANGSTROM * (rho - rho.mean()),
                               atol=1e-8 * np.abs(lap).max())
    assert abs(V.mean()) < 1e-9 * np.abs(V).max()


def test_hartree_potential_of_gaussian_near_its_centre():
    # isolated Gaussian + neutralizing background: V(0) - V(r) =
    # k N [2 sqrt(a/pi) - erf(sqrt(a) r)/r] - (2 pi / 3) k (N / V_cell) r^2
    from scipy.special import erf
    alpha_A, N, L = 2.0, 1.0, 16.0
    s = Structure(np.eye(3) * L, ["X"], [[0.5, 0.5, 0.5]])
    rho, _, _ = GaussianSuperposition(s, alpha_A, N).on_grid(Grid(s.lattice, (64,) * 3), False)
    eng = Engine(s, {"rho": rho})
    V = hartree_potential(eng).values
    r = eng.geometry().distance
    k = COULOMB_EV_ANGSTROM
    centre = V[32, 32, 32]
    for idx in ((36, 32, 32), (40, 32, 32), (44, 32, 32)):
        rr = r[idx]
        expected = k * N * (2 * np.sqrt(alpha_A / np.pi) - erf(np.sqrt(alpha_A) * rr) / rr) \
            - 2 * np.pi / 3 * k * N / L ** 3 * rr ** 2
        assert centre - V[idx] == pytest.approx(expected, rel=2e-3)


def test_fourier_interpolation():
    rng = np.random.default_rng(0)
    f = rng.random((6, 7, 8))
    idx = np.array([[1, 2, 3], [5, 0, 7]])
    np.testing.assert_allclose(fourier_interpolate(f, idx / np.array(f.shape)),
                               f[tuple(idx.T)], rtol=1e-12)
    # band-limited field: exact anywhere
    grid = Grid(np.eye(3) * 3.0, (10, 12, 14))
    u = grid.frac_coords()
    g = np.cos(2 * np.pi * (2 * u[..., 0] - u[..., 2])) + 0.5 * np.sin(2 * np.pi * u[..., 1])
    p = np.array([[0.123, 0.456, 0.789]])
    exact = np.cos(2 * np.pi * (2 * 0.123 - 0.789)) + 0.5 * np.sin(2 * np.pi * 0.456)
    assert fourier_interpolate(g, p)[0] == pytest.approx(exact, rel=1e-12)


def test_site_potentials_and_locpot_names():
    s = Structure(np.eye(3) * 5.0, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    grid = Grid(s.lattice, (20, 20, 20))
    rho, _, _ = GaussianSuperposition(s, [1.2, 0.8], [1.0, 7.0]).on_grid(grid, False)
    eng = Engine(s, {"rho": rho})
    vh = site_potentials(eng)
    assert vh.shape == (2,) and vh[1] > vh[0]      # more charge on Cl -> higher V_H
    d = compute_descriptors(eng, families=("F2",))
    assert set(d) == {"VH_spread", "VH_int_min"}
    assert d["VH_spread"] == pytest.approx(np.std(vh))
    u = grid.frac_coords()
    eng.add_field("potential", np.cos(2 * np.pi * u[..., 0]))
    np.testing.assert_allclose(site_potentials(eng, "locpot"), [1.0, -1.0], atol=1e-12)
    d = compute_descriptors(eng, families=("F2",))
    assert d["V_spread"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        site_potentials(eng, "nope")


# ---------------------------------------------------------------- F3 / F4

def test_nci_between_separated_fragments():
    # two diffuse atoms 3.4 A apart: the low-density, low-gradient region
    # between them is the NCI signature, with lambda2 < 0 across the gap
    s = Structure(np.diag([10.0, 8.0, 8.0]), ["Ar", "Ar"], [[0.33, 0.5, 0.5], [0.67, 0.5, 0.5]])
    rho, _, _ = GaussianSuperposition(s, 1.0, 8.0).on_grid(Grid(s.lattice, (60, 48, 48)), False)
    out = nci_family(Engine(s, {"rho": rho}))
    assert 0.0 < out["f_NCI"] < 0.2
    assert out["NCI_attractive"] > 0.5
    assert out["sign_lambda2_rho_mean"] < 0


def test_nci_empty_for_dense_uniform_density():
    s = Structure(np.eye(3) * 4.0, ["Na"], [[0, 0, 0]])
    out = nci_family(Engine(s, {"rho": np.full((10, 10, 10), 0.5 / A0 ** 3)}))
    assert out["f_NCI"] == 0.0 and np.isnan(out["NCI_attractive"])


def _cylinder_engine(ax, ay):
    """Density constant along z, Gaussian in x and y with exponents ax, ay."""
    s = Structure(np.eye(3) * 6.0, ["C"], [[0.5, 0.5, 0.5]])
    grid = Grid(s.lattice, (48, 48, 24))
    x = grid.cart_coords() - 3.0
    rho = np.exp(-ax * x[..., 0] ** 2 - ay * x[..., 1] ** 2)
    return Engine(s, {"rho": rho}, method="spectral")


def test_ellipticity_orders_circular_below_elliptical():
    circ = ellipticity_family(_cylinder_engine(1.0, 1.0))
    ell = ellipticity_family(_cylinder_engine(1.0, 2.5))
    for out in (circ, ell):
        assert out["ellip_bond_median"] >= 0 and out["ellip_bond_avg"] >= 0
    assert ell["ellip_bond_median"] > circ["ellip_bond_median"]
    # on the axis the eigenvalues are -2 ax rho and -2 ay rho exactly
    ev = _cylinder_engine(1.0, 2.5)["rho"].hessian_eigenvalues[24, 24, 0]
    assert ev[0] / ev[1] - 1 == pytest.approx(1.5, rel=1e-4)  # box edge not fully periodic


# ---------------------------------------------------------------- F6

def test_information_measures_closed_form():
    alpha = 0.8                                             # a.u.
    eng, _ = _gaussian_engine(alpha, N=3.0, L_bohr=16.0, n=72)
    out = information_family(eng)
    S = 1.5 * (1 + np.log(np.pi / alpha))
    D = (alpha / (2 * np.pi)) ** 1.5
    assert out["shannon_entropy"] == pytest.approx(S, rel=1e-8)
    # voxels below the density floor are left out of the Fisher integral
    assert out["fisher_information"] == pytest.approx(6 * alpha, rel=1e-5)
    assert out["disequilibrium"] == pytest.approx(D, rel=1e-8)
    assert out["LMC_complexity"] == pytest.approx(D * np.exp(S), rel=1e-8)


def test_lmc_complexity_is_unit_free_and_normalization_free():
    a = information_family(_gaussian_engine(0.8, N=3.0, L_bohr=16.0, n=48)[0])
    b = information_family(_gaussian_engine(0.8, N=7.0, L_bohr=16.0, n=48)[0])
    assert a["LMC_complexity"] == pytest.approx(b["LMC_complexity"], rel=1e-10)


# ---------------------------------------------------------------- I1

def test_anisotropy_isotropic_and_layered():
    iso = anisotropy_family(_gaussian_engine(0.6, n=32)[0])
    np.testing.assert_allclose([iso["T_eig_1"], iso["T_eig_2"], iso["T_eig_3"]], 1 / 3, atol=1e-10)
    assert iso["charge_FA"] == pytest.approx(0.0, abs=1e-8)

    # varies along one reciprocal direction only, in a triclinic cell
    lat = np.array([[4.0, 0, 0], [1.0, 5.0, 0], [0.5, 0.7, 6.0]])
    s = Structure(lat, ["X"], [[0, 0, 0]])
    u = Grid(lat, (16, 18, 20)).frac_coords()
    layered = Engine(s, {"rho": 2 + np.cos(2 * np.pi * u[..., 2])})
    out = anisotropy_family(layered)
    assert out["charge_FA"] == pytest.approx(1.0, abs=1e-10)
    assert out["T_eig_1"] == pytest.approx(1.0, abs=1e-10)


def test_anisotropy_of_constant_field_is_nan():
    s = Structure(np.eye(3) * 3.0, ["X"], [[0, 0, 0]])
    assert np.isnan(anisotropy_family(Engine(s, {"rho": np.ones((6, 6, 6))}))["charge_FA"])


# ---------------------------------------------------------------- assembly

def test_all_families_and_dataset_helpers():
    s = Structure(np.eye(3) * 5.0, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    grid = Grid(s.lattice, (24, 24, 24))
    rho, _, _ = GaussianSuperposition(s, [1.2, 0.8], [1.0, 7.0]).on_grid(grid, False)
    eng = Engine(s, {"rho": rho, "elf": np.full((12, 12, 12), 0.6),
                     "potential": np.cos(2 * np.pi * grid.frac_coords()[..., 0])})
    d = compute_descriptors(eng)
    assert set(d) == set(names())                          # every scalar entry
    assert {REGISTRY[k].family for k in d} == set(FAMILIES)

    rows = [dict(zeta=z, ellip_bond_avg=2 * z + 1, ellip_bond_median=-z) for z in range(5)]
    agree = zeta_ellip_agreement(rows)
    assert agree["mean"]["pearson"] == pytest.approx(1.0)
    assert agree["median"]["spearman"] == pytest.approx(-1.0)
    assert correlation([1, 2], [3, 4])["n"] == 2
