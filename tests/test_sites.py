"""Milestone 6: site aggregation and the heterogeneity meta-operator (spec §8.4)."""

import numpy as np
import pytest

import pydemi
from pydemi.core.partition import partition_of
from pydemi.descriptors.heterogeneity import density_site_sums
from pydemi.descriptors.registry import FeatureOptions
from pydemi.io.base import Lattice, Structure
from pydemi.operators.sitestats import (site_charge, site_m1, site_statistics,
                                        variance_decomposition)
from pydemi.validate.analytic import GaussianSuperposition

LAT = Lattice(np.array([[4.2, 0.0, 0.0], [0.6, 4.4, 0.0], [0.3, 0.4, 4.6]]))
FRAC = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.1], [0.25, 0.6, 0.55], [0.75, 0.2, 0.7]]


def _vd(species=("Fe", "Fe", "O", "O"), shape=(28, 28, 30), magnetic=False):
    s = Structure(LAT, list(species), FRAC)
    rho = GaussianSuperposition(s, [1.5, 1.7, 2.3, 2.1], [8.0, 8.0, 6.0, 6.0], tol=1e-18)
    mag = GaussianSuperposition(s, [2.0, 2.0, 3.0, 3.0], [2.0, -1.0, 0.1, 0.0], tol=1e-18)
    return rho.volumetric(shape, magnetization=mag if magnetic else None)


def test_anova_identity_holds_exactly():
    rng = np.random.default_rng(3)
    x = rng.normal(size=11)
    groups = list("AABBBCCCCDE")
    within, between = variance_decomposition(x, groups)
    assert within + between == pytest.approx(x.var(), rel=1e-12)


def test_anova_identity_on_computed_site_values():
    vd = _vd().with_options(FeatureOptions())
    x = site_m1(density_site_sums(vd))
    st = site_statistics(x, vd.structure.species)
    assert st.within + st.between == pytest.approx(x.var(), rel=1e-12)
    assert st.std == pytest.approx(x.std()) and st.range == pytest.approx(np.ptp(x))


def test_nearest_partition_tiles_space():
    """Site sums add up to the whole cell: charges, and the Tier-1 m1."""
    vd = _vd().with_options(FeatureOptions())
    sums = density_site_sums(vd)
    total = vd.rho.data.sum() * vd.rho.dV
    assert site_charge(sums).sum() == pytest.approx(total, rel=1e-12)
    f = pydemi.featurize(vd, domains=["bonding"])
    assert sums.r1.sum() / sums.total.sum() == pytest.approx(f["m1"], rel=1e-12)


def test_heterogeneity_names_and_values():
    feats = pydemi.featurize(_vd(magnetic=True), domains=["heterogeneity"])
    for base in ("m1", "f_bond", "zeta", "mu"):
        for suffix in ("site_range", "site_max", "site_min", "within_element_var",
                       "between_element_var"):
            assert f"{base}_{suffix}" in feats
    assert "mu_site_std" not in feats                      # magnetic domain (spec §8.3)
    assert feats["m1_site_max"] >= feats["m1_site_min"]
    assert feats["m1_site_range"] == pytest.approx(feats["m1_site_max"] - feats["m1_site_min"])
    assert np.isclose(feats["m1_site_std"] ** 2,
                      feats["m1_within_element_var"] + feats["m1_between_element_var"])


def test_heterogeneity_sentinels():
    # one site per element: within-element variance NaN (with site counts); single element: between 0
    f, meta = pydemi.featurize(_vd(species=("Fe", "Ni", "O", "S")), domains=["heterogeneity"],
                               return_metadata=True)
    assert np.isnan(f["m1_within_element_var"]) and meta["m1_within_element_var__flag"] == 1
    assert meta["site_counts"] == "Fe:1,Ni:1,O:1,S:1" and meta["max_sites_per_element"] == 1
    f, meta = pydemi.featurize(_vd(species=("Fe",) * 4), domains=["heterogeneity"],
                               return_metadata=True)
    assert f["m1_between_element_var"] == 0.0 and meta["m1_between_element_var__flag"] == 1
    # non-magnetic: every mu statistic is 0.0, flagged, never NaN
    assert meta["magnetic"] is False
    for k in ("mu_site_range", "mu_site_max", "mu_within_element_var", "mu_between_element_var"):
        assert f[k] == 0.0 and meta[f"{k}__flag"] == 1


def test_power_partition_is_a_different_tiling():
    vd = _vd()
    near = pydemi.featurize(vd, domains=["heterogeneity"])
    power = pydemi.featurize(vd, domains=["heterogeneity"], partition="power")
    assert near["m1_site_std"] != pytest.approx(power["m1_site_std"])
    p = partition_of(vd.with_options(FeatureOptions(partition="power")), "power")
    q = p.site_sum(vd.rho.data) * vd.rho.dV
    assert q.sum() == pytest.approx(vd.rho.data.sum() * vd.rho.dV, rel=1e-12)
