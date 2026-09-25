"""Milestone 4: Tier-1 descriptors against analytic densities (spec §8.1, §11)."""

import functools

import numpy as np
import pytest

import pydemi
from pydemi.descriptors.registry import REGISTRY, register
from pydemi.io.base import Lattice, Structure
from pydemi.validate.analytic import (SlaterSuperposition, cubic_cell, slater_lnf, slater_moment,
                                      uniform)

TIER1 = ["zeta", "m1", "m2", "sigma_r2", "f_core", "f_bond", "f_int", "lnf",
         "lnf_charge_weighted", "lap_concentration", "lap_concentration_valence"]
Z, A = 3.0, 4.0


@functools.lru_cache(maxsize=None)
def _slater(n):
    return SlaterSuperposition(cubic_cell(A), [Z], tol=1e-30).volumetric((n, n, n))


def _f(vd, **kw):
    return pydemi.featurize(vd, domains=["bonding"], **kw)


@pytest.mark.parametrize("backend", ["fft", "fd"])
def test_slater_closed_forms(backend):
    """Single Slater 1s: int rho = 1, m1 = 3/(2 zeta), m2 = 3/zeta^2, sigma_r2 = 3/(4 zeta^2), zeta = 0."""
    vd = _slater(96)
    assert np.isclose(vd.rho.data.sum() * vd.rho.dV, 1.0, atol=1e-4)
    f = _f(vd, derivative_backend=backend)
    assert f["m1"] == pytest.approx(slater_moment(1, Z), rel=5e-4)
    # m2 / sigma_r2 carry a small bias: distances are to the nearest image in a 4 A box
    assert f["m2"] == pytest.approx(slater_moment(2, Z), rel=1.5e-3)
    assert f["sigma_r2"] == pytest.approx(slater_moment(2, Z) - slater_moment(1, Z) ** 2, rel=1e-2)
    assert 0.0 <= f["zeta"] < 5e-5


def test_slater_moments_converge_with_the_grid():
    errs = [abs(_f(_slater(n))["m1"] - slater_moment(1, Z)) for n in (48, 64, 96)]
    assert errs[0] > errs[1] > errs[2]


def test_slater_lnf_is_the_sphere_volume_fraction():
    """lnf -> volume fraction with r < 1/zeta. With finite differences; the FFT
    Laplacian rings on the cusp (spec §4), which biases the sign count."""
    exact = slater_lnf(Z, A ** 3)
    assert _f(_slater(96), derivative_backend="fd")["lnf"] == pytest.approx(exact, rel=0.05)
    assert _f(_slater(96), derivative_backend="fft")["lnf"] > 1.1 * exact


def test_two_atoms_give_nonzero_anisotropy():
    s = Structure(Lattice(np.eye(3) * 5.0), ["H", "H"], [[0.4, 0.5, 0.5], [0.6, 0.5, 0.5]])
    vd = SlaterSuperposition(s, [1.5, 1.5], tol=1e-20).volumetric((48, 48, 48))
    f = _f(vd)
    assert f["zeta"] > 1e-2
    assert f["lnf_charge_weighted"] != pytest.approx(f["lnf"])


@pytest.mark.parametrize("backend", ["fft", "fd"])
def test_lap_concentration_is_identically_one_half(backend):
    """Documented correction: int lap rho dV = 0 on a periodic grid, so the ratio is 1/2."""
    s = Structure(Lattice(np.array([[4.0, 0, 0], [1.1, 4.3, 0], [0.5, 0.7, 4.6]])), ["H", "O"],
                  [[0.1, 0.2, 0.3], [0.55, 0.6, 0.7]])
    vd = SlaterSuperposition(s, [1.2, 2.0], [1.0, 6.0], tol=1e-20).volumetric((40, 42, 44))
    f = _f(vd, derivative_backend=backend)
    assert f["lap_concentration"] == pytest.approx(0.5, abs=1e-10)
    assert f["lap_concentration_valence"] != pytest.approx(0.5, abs=1e-3)


def test_uniform_density_sentinels():
    feats, meta = pydemi.featurize(uniform(cubic_cell(4.0), (16, 16, 16)), domains=["bonding"],
                                   return_metadata=True)
    assert not any(np.isnan(v) for v in feats.values())
    for name, value in (("zeta", 0.0), ("lnf", 0.0), ("lnf_charge_weighted", 0.0),
                        ("lap_concentration", 0.5), ("lap_concentration_valence", 0.5)):
        assert feats[name] == value
        assert meta[f"{name}__flag"] == 1
        assert f"{name}:uniform_density" in meta["sentinels"]
    assert meta["m1__flag"] == 0                      # moments are defined for a uniform density


def test_featurize_names_and_metadata():
    vd = _slater(48)
    feats, meta = pydemi.featurize(vd, return_metadata=True)
    assert list(feats)[:len(TIER1)] == TIER1
    assert pydemi.descriptor_names(["bonding"])[:len(TIER1)] == TIER1
    for key in ("n_atoms", "volume", "grid_shape", "density_source", "wall_time_s",
                "pydemi_version", "error"):
        assert key in meta
    assert meta["grid_shape"] == "48x48x48" and meta["density_source"] == "pseudo"


def test_catalogue_is_generated_from_the_registry():
    cat = pydemi.catalogue()
    for col in ("name", "domain", "field", "requires", "units", "range_min", "range_max",
                "intensive", "sentinel_cases", "adopted", "formula", "references"):
        assert col in cat.columns
    assert set(TIER1) <= set(cat["name"])
    assert cat["intensive"].all()
    assert all(isinstance(f, str) and "=" in f for f in cat["formula"])


def test_register_refuses_extensive_quantities():
    with pytest.raises(ValueError, match="intensive"):
        register(name="Q_tot_test", domain="bonding", field="rho", requires=[], units="e",
                 intensive=False)
    assert "Q_tot_test" not in REGISTRY


def test_options_are_validated():
    vd = _slater(48)
    with pytest.raises(ValueError, match="partition"):
        pydemi.featurize(vd, partition="bader")
    with pytest.raises(ValueError, match="domain"):
        pydemi.featurize(vd, domains=["optical"])
