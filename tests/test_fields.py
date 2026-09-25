"""Milestone 8: derived fields -- ELF_D, energy densities, potentials, NCI (spec §6, §8.1)."""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erf

import pydemi
from pydemi.constants import C_F, COULOMB_EV_ANGSTROM, DENSITY_TO_AU
from pydemi.descriptors.bonding import kinetic, potential
from pydemi.descriptors.registry import FeatureOptions
from pydemi.fields.elf import elf_d
from pydemi.fields.potential import electrostatic_potential, hartree_potential
from pydemi.io.base import Grid, Lattice, Structure
from pydemi.validate import elf_fidelity
from pydemi.validate.analytic import GaussianSuperposition, SlaterSuperposition, cubic_cell, uniform

REAL = Path("/data/sai/new_charge/6000_data_aug13/FeNi3_221/CHGCAR")


def _crystal(shape=(30, 30, 32)):
    s = Structure(Lattice(np.array([[4.1, 0, 0], [0.7, 4.3, 0], [0.4, 0.5, 4.5]])), ["Fe", "O"],
                  [[0.1, 0.1, 0.1], [0.6, 0.55, 0.6]])
    return SlaterSuperposition(s, [1.4, 2.0], [8.0, 6.0], tol=1e-20).volumetric(shape)


# ---------------------------------------------------------------- ELF_D

def test_elf_d_is_in_unit_interval_on_analytic_data():
    vd = _crystal().with_options(FeatureOptions())
    e = elf_d(kinetic(vd))
    assert e.min() >= 0.0 and e.max() <= 1.0


@pytest.mark.skipif(not REAL.exists(), reason="dataset CHGCAR not available")
def test_elf_d_is_in_unit_interval_on_real_data():
    """Spec §6.2: ELF_D in [0, 1] everywhere on real (PAW, partly negative) data."""
    vd = pydemi.read_vasp(REAL).with_options(FeatureOptions())
    e = elf_d(kinetic(vd))
    assert np.isfinite(e).all() and e.min() >= 0.0 and e.max() <= 1.0


def test_uniform_electron_gas_limits():
    """grad rho = lap rho = 0: ELF_D = 1/2, g = C_F rho^(5/3), H = -g < 0."""
    rho = 0.1
    vd = uniform(cubic_cell(4.0), (12, 12, 12), rho)
    kt = kinetic(vd.with_options(FeatureOptions()))
    np.testing.assert_allclose(elf_d(kt), 0.5, atol=1e-12)
    g_exact = C_F * (rho * DENSITY_TO_AU) ** (5.0 / 3.0)
    np.testing.assert_allclose(kt.g, g_exact, rtol=1e-9)
    f = pydemi.featurize(vd, domains=["bonding"])
    assert f["f_H_negative"] == 1.0
    assert f["H_bond_mean"] == pytest.approx(-g_exact, rel=1e-9)
    assert f["G_over_rho"] == pytest.approx(g_exact / (rho * DENSITY_TO_AU), rel=1e-9)
    assert f["ELF_bond_avg"] == pytest.approx(0.5, abs=1e-12)


def test_elf_source_selection(tmp_path):
    vd = _crystal()
    _, meta = pydemi.featurize(vd, domains=["bonding"], return_metadata=True)
    assert meta["elf_source"] == "reconstruct"
    vd.elf = Grid(np.full(vd.shape, 0.7), vd.lattice)
    f, meta = pydemi.featurize(vd, domains=["bonding"], return_metadata=True)
    assert meta["elf_source"] == "file" and f["ELF_bond_avg"] == pytest.approx(0.7)
    f2 = pydemi.featurize(vd, domains=["bonding"], elf_source="reconstruct")
    assert f2["ELF_bond_avg"] != pytest.approx(0.7)
    vd.elf = None
    with pytest.raises(ValueError, match="ELFCAR"):
        pydemi.featurize(vd, domains=["bonding"], elf_source="file")


def test_elf_fidelity():
    a = np.linspace(0, 1, 1000).reshape(10, 10, 10)
    m = elf_fidelity(a, a)
    assert m["pearson_r"] == pytest.approx(1.0) and m["mae"] == 0.0 and m["rmse"] == 0.0
    assert elf_fidelity(a, a + 0.1)["mae"] == pytest.approx(0.1)


# ---------------------------------------------------------------- potentials

def test_hartree_potential_of_a_gaussian():
    """V(r) - V(0) = k N [erf(sqrt(a) r)/r - 2 sqrt(a/pi)] + (2 pi k N / 3V) r^2 in a cubic box
    (the neutralizing background of the G = 0 convention adds the r^2 term)."""
    a, alpha, N, n = 10.0, 3.0, 2.0, 64
    s = cubic_cell(a, ["H"], [[0.5, 0.5, 0.5]])
    rho, _, _ = GaussianSuperposition(s, [alpha], [N], tol=1e-30).on_grid((n, n, n))
    V = hartree_potential(rho, s.lattice)
    c = n // 2
    k = COULOMB_EV_ANGSTROM
    for step in (4, 8, 12):
        r = step * a / n
        exact = k * N * (erf(math.sqrt(alpha) * r) / r - 2 * math.sqrt(alpha / math.pi)) \
            + 2 * math.pi * k * N / (3 * a ** 3) * r * r
        assert V[c + step, c, c] - V[c, c, c] == pytest.approx(exact, rel=2e-3)


def test_electrostatic_potential_of_a_neutral_cell_vanishes():
    """Electron Gaussians with the ion width and charge cancel the ionic term exactly."""
    s = Structure(Lattice(np.eye(3) * 6.0), ["H", "H"], [[0.2, 0.3, 0.4], [0.7, 0.6, 0.8]])
    sigma = 0.5
    rho, _, _ = GaussianSuperposition(s, [1 / (2 * sigma ** 2)] * 2, [1.0, 1.0],
                                      tol=1e-30).on_grid((48, 48, 48))
    V = electrostatic_potential(rho, s, np.array([1.0, 1.0]), sigma)
    assert np.abs(V).max() < 1e-6 * np.abs(hartree_potential(rho, s.lattice)).max()


def test_potential_sources():
    vd = _crystal()
    _, meta = pydemi.featurize(vd, domains=["bonding"], return_metadata=True)
    assert meta["potential_source"] == "hartree"
    esp = pydemi.featurize(vd, domains=["bonding"], potential_source="esp")
    hart = pydemi.featurize(vd, domains=["bonding"])
    assert esp["V_spread"] != pytest.approx(hart["V_spread"])
    vd.potential = Grid(np.full(vd.shape, 5.0) + vd.rho.data, vd.lattice)
    f, meta = pydemi.featurize(vd, domains=["bonding"], return_metadata=True)
    assert meta["potential_source"] == "locpot"
    v = potential(vd.with_options(FeatureOptions()))
    assert abs(v.mean()) < 1e-12                        # referenced to the cell average
    vd.potential = None
    with pytest.raises(ValueError, match="LOCPOT"):
        pydemi.featurize(vd, domains=["bonding"], potential_source="locpot")


# ---------------------------------------------------------------- NCI, midpoints

def test_nci_region_between_two_distant_atoms():
    s = Structure(Lattice(np.eye(3) * 8.0), ["H", "H"], [[0.35, 0.5, 0.5], [0.65, 0.5, 0.5]])
    vd = GaussianSuperposition(s, [1.0, 1.0], [1.0, 1.0], tol=1e-30).volumetric((40, 40, 40))
    f = pydemi.featurize(vd, domains=["bonding"])
    assert 0.0 < f["f_NCI"] < 1.0
    assert 0.0 <= f["NCI_attractive"] <= 1.0


def test_uniform_nci_sentinel():
    f, meta = pydemi.featurize(uniform(cubic_cell(4.0), (12, 12, 12)), domains=["bonding"],
                               return_metadata=True)
    assert f["f_NCI"] == 0.0 and meta["f_NCI__flag"] == 1
    assert not any(np.isnan(v) for v in f.values())


def test_bond_midpoint_density():
    """Two Gaussians 2 A apart in a big box: rho(mid) = 2 N (a/pi)^(3/2) exp(-a d^2 / 4)."""
    a, d = 1.5, 2.0
    s = Structure(Lattice(np.eye(3) * 12.0), ["H", "H"],
                  [[0.5 - d / 24, 0.5, 0.5], [0.5 + d / 24, 0.5, 0.5]])
    vd = GaussianSuperposition(s, [a, a], [1.0, 1.0], tol=1e-30).volumetric((64, 64, 64))
    f = pydemi.featurize(vd, domains=["bonding"])
    exact = 2 * (a / math.pi) ** 1.5 * math.exp(-a * d * d / 4)
    assert f["rho_mid_mean"] == pytest.approx(exact, rel=1e-6)
    assert f["rho_mid_std"] == pytest.approx(0.0, abs=1e-12)
