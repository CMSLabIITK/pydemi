"""Milestone 10: structural domain -- percolation, census, NNM, floor, tensor, information (spec §8.2)."""

import math

import numpy as np
import pytest

import pydemi
from pydemi.io.base import Grid, Lattice, Structure, VolumetricData
from pydemi.operators.topology import extremum_census, percolation_levels, spans
from pydemi.validate.analytic import GaussianSuperposition, cubic_cell, uniform
from pydemi.validate.invariance import translate


def _periodic(n, a=4.0):
    x = np.arange(n) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    f = (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y) + np.cos(2 * np.pi * Z)
         + 0.1 * np.sin(2 * np.pi * (X + 2 * Y + Z)))
    return f + 3.5


@pytest.mark.parametrize("n", [12, 24, 48])
def test_census_of_a_periodic_function_with_known_critical_points(n):
    """cos 2pi x + cos 2pi y + cos 2pi z (+ a small periodic tilt): 1 max, 1 min,
    3 + 3 saddles per cell, and euler_consistency 0 at every resolution."""
    c = extremum_census(_periodic(n))
    assert (c["n_max"], c["n_min"], c["n_saddle1"], c["n_saddle2"]) == (1, 1, 3, 3)
    assert c["euler_consistency"] == 0


def test_census_descriptors_are_per_volume():
    vd = VolumetricData(cubic_cell(4.0), Grid(_periodic(24), Lattice(np.eye(3) * 4.0)))
    f, meta = pydemi.featurize(vd, domains=["structural"], return_metadata=True)
    assert f["n_max"] == pytest.approx(1 / 64.0) and f["n_saddle1"] == pytest.approx(3 / 64.0)
    assert meta["euler_consistency"] == 0


def test_simple_cubic_gaussians_percolate_at_the_midpoint_density():
    """Spec §8.2 test: the bottleneck of a simple-cubic lattice is the bond midpoint,
    so rho_perc = rho(a/2, 0, 0) in all three directions."""
    a, alpha, n = 3.0, 1.2, 30
    s = cubic_cell(a, ["H"], [[0.0, 0.0, 0.0]])
    g = GaussianSuperposition(s, [alpha], [1.0], tol=1e-30)
    vd = g.volumetric((n, n, n))
    mid = g.evaluate(np.array([[a / 2, 0.0, 0.0]]))[0][0]
    f = pydemi.featurize(vd, domains=["structural"])
    for axis in "abc":
        assert f[f"rho_perc_{axis}"] == pytest.approx(mid, rel=1e-12)
    assert f["perc_anisotropy"] == pytest.approx(0.0, abs=1e-12)


def test_percolation_does_not_depend_on_the_cell_origin():
    """An atom at the cell corner: its blob crosses every face without spanning."""
    s = cubic_cell(6.0, ["H"], [[0.0, 0.0, 0.0]])
    vd = GaussianSuperposition(s, [2.0], [1.0], tol=1e-30).volumetric((24, 24, 24))
    a = pydemi.featurize(vd, domains=["structural"])
    b = pydemi.featurize(translate(vd, (12, 12, 12)), domains=["structural"])
    for axis in "abc":
        assert a[f"rho_perc_{axis}"] == pytest.approx(b[f"rho_perc_{axis}"], rel=1e-12)
        assert a[f"rho_perc_{axis}"] < 0.5 * vd.rho.data.max()


def test_spans_needs_a_winding_cluster():
    blob = np.zeros((10, 10, 10), bool)
    blob[[0, 1, 9], 4:6, 4:6] = True          # crosses the a-face but is finite
    assert not spans(blob).any()
    rod = np.zeros((10, 10, 10), bool)
    rod[:, 4, 4] = True
    assert spans(rod).tolist() == [True, False, False]


def test_percolation_level_is_the_highest_spanning_level():
    """Documented correction: the LOWEST spanning level is always min(rho)."""
    s = cubic_cell(3.0, ["H"], [[0.0, 0.0, 0.0]])
    rho = GaussianSuperposition(s, [1.2], [1.0], tol=1e-30).on_grid((20, 20, 20))[0]
    lv = percolation_levels(rho)
    assert spans(rho > lv[0] - 1e-12)[0] and not spans(rho > lv[0])[0]
    assert lv[0] > rho.min() * 1.5


def test_a_non_nuclear_maximum_is_counted_with_its_basin_charge():
    s = Structure(Lattice(np.eye(3) * 6.0), ["H", "H"], [[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]])
    atoms = GaussianSuperposition(s, [2.0, 2.0], [1.0, 1.0], tol=1e-30)
    ghost = GaussianSuperposition(cubic_cell(6.0, ["X"], [[0.5, 0.5, 0.5]]), [3.0], [0.5], tol=1e-30)
    shape = (36, 36, 36)
    rho = atoms.on_grid(shape)[0] + ghost.on_grid(shape)[0]
    f = pydemi.featurize(VolumetricData(s, Grid(rho, s.lattice)), domains=["structural"])
    assert f["n_NNM"] == pytest.approx(1 / 216.0)
    assert 0.15 < f["Q_NNM"] < 0.25                     # the ghost holds 0.5 of 2.5 electrons


def test_uniform_limits_and_sentinels():
    f, meta = pydemi.featurize(uniform(cubic_cell(4.0), (12, 12, 12), 0.2),
                               domains=["structural"], return_metadata=True)
    assert f["shannon_entropy"] == pytest.approx(0.0, abs=1e-12)
    assert f["disequilibrium"] == pytest.approx(1.0, rel=1e-12)
    assert f["LMC_complexity"] == pytest.approx(1.0, rel=1e-12)
    for k in ("T_eigenvalues_t1", "T_eigenvalues_t2", "T_eigenvalues_t3"):
        assert f[k] == pytest.approx(1.0 / 3.0) and meta[f"{k}__flag"] == 1
    assert f["charge_FA"] == 0.0 and meta["charge_FA__flag"] == 1
    assert not any(math.isnan(v) for v in f.values())


def test_anisotropy_tensor_of_a_layered_density():
    """A density varying along a3 only: t3 = 1, FA = 1."""
    x = np.arange(24) / 24
    rho = np.broadcast_to(1.0 + 0.5 * np.cos(2 * np.pi * x)[None, None, :], (24, 24, 24)).copy()
    vd = VolumetricData(cubic_cell(4.0), Grid(rho, Lattice(np.eye(3) * 4.0)))
    f = pydemi.featurize(vd, domains=["structural"])
    assert f["T_eigenvalues_t3"] == pytest.approx(1.0) and f["T_eigenvalues_t1"] == pytest.approx(0.0, abs=1e-12)
    assert f["charge_FA"] == pytest.approx(1.0)
