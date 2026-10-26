"""Descriptor plots: one per descriptor, and the plotted data reproduce the descriptor."""

import pickle

import numpy as np
import pytest

pytest.importorskip("plotly")

from pydemi.gui import core, plots  # noqa: E402
import pydemi
from pydemi.io.base import Lattice, Structure
from pydemi.validate.analytic import GaussianSuperposition


@pytest.fixture(scope="module", params=["nearest", "becke"])
def run(request):
    s = Structure(Lattice(np.array([[4.0, 0.0, 0.0], [0.5, 4.1, 0.0], [0.3, 0.4, 4.3]])),
                  ["Na", "Cl", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.45, 0.5], [0.2, 0.7, 0.35]])
    vd = GaussianSuperposition(s, [1.4, 2.2, 2.2], [1.0, 8.0, 8.0], tol=1e-20).volumetric((20, 20, 22))
    feats, _ = core.compute(vd, request.param)
    return plots.plot_data(vd, request.param), feats


def _mean(h):
    c = 0.5 * (h["edges"][1:] + h["edges"][:-1])
    return float(np.sum(c * h["counts"]) / np.sum(h["counts"]))


def test_every_descriptor_has_a_figure(run):
    data, feats = run
    assert set(plots.FIGURES) == set(pydemi.descriptor_names())
    for name in pydemi.descriptor_names():
        fig, caption = plots.figure(name, data, feats)
        assert fig.layout.title.text.startswith(f"{name} = ") and caption
        fig.to_json()                              # serializable, as the browser needs it


def test_data_is_small_and_picklable(run):
    data, _ = run
    assert len(pickle.dumps(data)) < 2_000_000


def test_radial_profiles_reproduce_the_bonding_descriptors(run):
    d, f = run
    rad = d["radial"]
    hi = rad["edges"][1:]
    q = rad["charge"]
    core_, bond = hi <= d["c1"] + 1e-9, (hi > d["c1"] + 1e-9) & (hi <= d["c2"] + 1e-9)
    assert q[core_].sum() / q.sum() == pytest.approx(f["f_core"], rel=1e-9)
    assert q[bond].sum() / q.sum() == pytest.approx(f["f_bond"], rel=1e-9)
    assert rad["vol_neg"].sum() / rad["volume"].sum() == pytest.approx(f["lnf"], rel=1e-12)
    assert rad["charge_neg"].sum() / q.sum() == pytest.approx(f["lnf_charge_weighted"], rel=1e-9)
    lap = rad["lap_neg"].sum() / (rad["lap_neg"].sum() + rad["lap_pos"].sum())
    assert lap == pytest.approx(f["lap_concentration"], rel=1e-9)
    r = 0.5 * (rad["edges"][1:] + rad["edges"][:-1])
    assert np.sum(r * q) / q.sum() == pytest.approx(f["m1"], abs=plots.RADIAL_BIN)
    fisher = d["info"]["fisher_radial"].sum()
    assert fisher == pytest.approx(f["fisher_information"], rel=1e-9)


def test_histograms_are_centred_on_their_descriptors(run):
    d, f = run
    w = 1.0 / plots.UNIT_BINS
    assert 1.0 - _mean(d["zeta_cos"]) == pytest.approx(f["zeta"], abs=w)
    for k in range(3):
        assert _mean(d["tensor"]["cos2"][k]) == pytest.approx(f[f"T_eigenvalues_t{k + 1}"], abs=w)
    np.testing.assert_allclose(d["tensor"]["eigenvalues"],
                               [f[f"T_eigenvalues_t{k}"] for k in (1, 2, 3)], rtol=1e-12)
    if d["ellip"] is not None:
        assert _mean(d["ellip"]) == pytest.approx(f["ellip_bond_bounded_avg"], abs=w)
    assert -_mean(d["info"]["ln_ratio"]) == pytest.approx(f["shannon_entropy"], abs=0.2)


def test_points_and_sites_match_the_descriptors(run):
    d, f = run
    V = 4.0 * 4.1 * 4.3
    assert d["maxima"]["total"] / V == pytest.approx(f["n_max"], rel=1e-12)
    assert d["minima"]["total"] / V == pytest.approx(f["n_min"], rel=1e-12)
    assert d["nnm"]["q"].sum() == pytest.approx(f["Q_NNM"], abs=1e-12)
    assert np.std(d["potential"]["sites"]) == pytest.approx(f["V_spread"], rel=1e-12)
    assert np.mean(d["bonds"]["rho"]) == pytest.approx(f["rho_mid_mean"], rel=1e-12)
    np.testing.assert_allclose(d["percolation"]["levels"],
                               [f["rho_perc_a"], f["rho_perc_b"], f["rho_perc_c"]])
    for ax in range(3):        # a spanning path crosses every slab
        assert d["percolation"]["levels"][ax] <= d["percolation"]["profiles"][ax]["max"].min()
    for base in ("m1", "f_bond", "zeta"):
        x = d["sites"][base]
        assert np.std(x) == pytest.approx(f[f"{base}_site_std"], rel=1e-9)
        assert np.ptp(x) == pytest.approx(f[f"{base}_site_range"], rel=1e-9)
