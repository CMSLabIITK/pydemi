"""Milestone 8: derived fields -- the Hartree potential, NCI (spec §6, §8.1)."""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erf

import pydemi
from pydemi.constants import COULOMB_EV_ANGSTROM
from pydemi.descriptors.bonding import _rho_au, potential
from pydemi.descriptors.registry import FeatureOptions
from pydemi.fields.potential import hartree_potential
from pydemi.io.base import Lattice, Structure
from pydemi.validate.analytic import GaussianSuperposition, SlaterSuperposition, cubic_cell, uniform

REAL = Path("/data/sai/new_charge/6000_data_aug13/FeNi3_221/CHGCAR")


def _crystal(shape=(30, 30, 32)):
    s = Structure(Lattice(np.array([[4.1, 0, 0], [0.7, 4.3, 0], [0.4, 0.5, 4.5]])), ["Fe", "O"],
                  [[0.1, 0.1, 0.1], [0.6, 0.55, 0.6]])
    return SlaterSuperposition(s, [1.4, 2.0], [8.0, 6.0], tol=1e-20).volumetric(shape)


# ---------------------------------------------------------------- real data

@pytest.mark.skipif(not REAL.exists(), reason="dataset CHGCAR not available")
def test_nci_fields_are_finite_on_real_data():
    """Real PAW pseudo-densities are partly negative: rho is clipped and s stays defined."""
    rho, s = _rho_au(pydemi.read_chgcar(REAL).with_options(FeatureOptions()))
    assert (rho >= 0).all() and not np.isnan(s).any() and np.isfinite(s).any()


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


def test_potential_is_referenced_to_the_cell_average():
    v = potential(_crystal().with_options(FeatureOptions()))
    assert abs(v.mean()) < 1e-12


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
