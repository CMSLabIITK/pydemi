"""Milestone 7: magnetic domain (spec §8.3)."""

import numpy as np
import pytest

import pydemi
from pydemi.descriptors.magnetic import site_moments, total_moments
from pydemi.descriptors.registry import FeatureOptions
from pydemi.io.base import Grid, Lattice, Structure
from pydemi.validate.analytic import GaussianSuperposition

LAT = Lattice(np.array([[4.0, 0.0, 0.0], [0.5, 4.2, 0.0], [0.2, 0.3, 4.4]]))
FRAC = [[0.1, 0.1, 0.1], [0.6, 0.55, 0.6], [0.35, 0.8, 0.3]]
SHAPE = (36, 36, 38)
MAG_NAMES = ["M_abs_per_atom", "M_net_per_atom", "m1_spin", "sigma_r2_spin", "f_bond_spin",
             "mu_site_std", "spin_frustration", "spin_charge_correlation"]


def _vd(moments, noncollinear=None):
    s = Structure(LAT, ["Fe", "Fe", "O"], FRAC)
    rho = GaussianSuperposition(s, [1.6, 1.6, 2.2], [8.0, 8.0, 6.0], tol=1e-18)
    # narrow spin densities, so the moments of neighbouring sites do not overlap
    mag = GaussianSuperposition(s, [9.0, 9.0, 9.0], moments, tol=1e-18)
    vd = rho.volumetric(SHAPE, magnetization=mag)
    if noncollinear is not None:
        vd.magnetization = None
        vd.magnetization_vector = tuple(
            Grid(GaussianSuperposition(s, [9.0, 9.0, 9.0], [d[0][c], d[1][c], 0.0],
                                       tol=1e-18).on_grid(SHAPE)[0], s.lattice)
            for c, d in zip(range(3), [noncollinear] * 3))
    return vd


@pytest.mark.parametrize("partition", ["nearest", "power"])
def test_site_moments_sum_to_the_net_moment(partition):
    """sum_i mu_i == M_net exactly: the partition tiles space (unlike RWIGS spheres)."""
    vd = _vd([2.0, -0.7, 0.05]).with_options(FeatureOptions(partition=partition))
    mu = site_moments(vd)
    _, M_net = total_moments(vd)
    assert abs(mu.sum()) == pytest.approx(M_net, rel=1e-12)
    assert mu[0] > 0 > mu[1]


def test_ferro_and_antiferro():
    ferro = pydemi.featurize(_vd([2.0, 2.0, 0.0]), domains=["magnetic"])
    afm = pydemi.featurize(_vd([2.0, -2.0, 0.0]), domains=["magnetic"])
    assert ferro["spin_frustration"] == pytest.approx(0.0, abs=1e-12)
    assert afm["spin_frustration"] == pytest.approx(1.0, abs=1e-9)
    assert afm["M_net_per_atom"] == pytest.approx(0.0, abs=1e-9)
    assert afm["M_abs_per_atom"] == pytest.approx(ferro["M_abs_per_atom"], rel=1e-9)
    assert list(ferro) == MAG_NAMES


def test_noncollinear_vector_generalization():
    """Two equal moments at 90 degrees: frustration = 1 - sqrt(2)/2."""
    vd = _vd([0.0, 0.0, 0.0], noncollinear=[(2.0, 0.0, 0.0), (0.0, 2.0, 0.0)])
    assert vd.spin_mode == "noncollinear"
    f = pydemi.featurize(vd, domains=["magnetic"])
    assert f["spin_frustration"] == pytest.approx(1.0 - np.sqrt(0.5), abs=1e-4)
    mu = site_moments(vd.with_options(FeatureOptions()))
    assert mu.shape == (3, 3)
    assert np.linalg.norm(mu.sum(axis=0)) == pytest.approx(total_moments(vd)[1], rel=1e-12)


@pytest.mark.parametrize("moments", [None, [1e-6, -1e-6, 0.0]], ids=["unpolarized", "below_tol"])
def test_non_magnetic_is_zero_never_nan(moments):
    s = Structure(LAT, ["Fe", "Fe", "O"], FRAC)
    rho = GaussianSuperposition(s, [1.6, 1.6, 2.2], [8.0, 8.0, 6.0], tol=1e-18)
    mag = None if moments is None else GaussianSuperposition(s, [2.2, 2.2, 3.0], moments, tol=1e-18)
    f, meta = pydemi.featurize(rho.volumetric(SHAPE, magnetization=mag), domains=["magnetic"],
                               return_metadata=True)
    assert meta["magnetic"] is False
    for name in MAG_NAMES:
        assert f[name] == 0.0 and meta[f"{name}__flag"] == 1
