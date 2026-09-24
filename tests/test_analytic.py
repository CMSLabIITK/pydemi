"""
Validation against closed-form Slater and Gaussian densities: the grid
primitives (charge, charge-weighted radial moments, shell fractions)
must reproduce the isolated-atom analytic values when atoms are well
separated, and converge as the grid is refined.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from pydemi import Engine, Grid, Structure
from pydemi.testing import (GaussianSuperposition, SlaterSuperposition,
                            gaussian_fraction_within, gaussian_moment,
                            slater_fraction_within, slater_moment)


def _engine_for(density, structure, n):
    grid = Grid(structure.lattice, (n, n, n))
    rho, _, _ = density.on_grid(grid, derivatives=False)
    return Engine(structure, {"rho": rho})


def _moments(eng):
    rho = eng["rho"].values
    r = eng.geometry().distance
    return rho.sum() * eng.grid().dV, (rho * r).sum() / rho.sum(), (rho * r ** 2).sum() / rho.sum()


def test_closed_forms_against_quadrature():
    zeta, alpha, R = 1.7, 2.3, 0.9
    f_s = lambda r: 4 * np.pi * r ** 2 * zeta ** 3 / np.pi * np.exp(-2 * zeta * r)
    f_g = lambda r: 4 * np.pi * r ** 2 * (alpha / np.pi) ** 1.5 * np.exp(-alpha * r ** 2)
    for n in (0, 1, 2, 3):
        assert slater_moment(n, zeta) == pytest.approx(
            quad(lambda r: f_s(r) * r ** n, 0, np.inf)[0], rel=1e-10)
        assert gaussian_moment(n, alpha) == pytest.approx(
            quad(lambda r: f_g(r) * r ** n, 0, np.inf)[0], rel=1e-10)
    assert slater_fraction_within(R, zeta) == pytest.approx(quad(f_s, 0, R)[0], rel=1e-10)
    assert gaussian_fraction_within(R, alpha) == pytest.approx(quad(f_g, 0, R)[0], rel=1e-10)


def test_slater_moments_single_atom():
    zeta = 2.0
    s = Structure(np.eye(3) * 10.0, ["X"], [[0.3, 0.4, 0.5]])
    errs = []
    for n in (60, 120):
        Q, m1, m2 = _moments(_engine_for(SlaterSuperposition(s, zeta, 1.0, tol=1e-10), s, n))
        errs.append(abs(m1 / slater_moment(1, zeta) - 1))
        assert Q == pytest.approx(1.0, rel=2e-2)
        assert m2 == pytest.approx(slater_moment(2, zeta), rel=2e-2)
    assert errs[1] < 1e-2 and errs[1] < errs[0]   # accurate and converging


def test_slater_shell_fractions_two_atoms():
    # two well-separated atoms of different exponent in a triclinic cell
    lat = np.array([[12.0, 0, 0], [2.0, 11.0, 0], [1.0, 1.5, 12.0]])
    s = Structure(lat, ["A", "B"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    zetas, N = np.array([1.8, 2.6]), np.array([3.0, 5.0])
    eng = _engine_for(SlaterSuperposition(s, zetas, N, tol=1e-10), s, 110)
    rho, m = eng["rho"].values, eng.shell_masks()
    total = rho.sum()
    expected_core = np.sum(N * slater_fraction_within(0.8, zetas)) / N.sum()
    expected_int = np.sum(N * (1 - slater_fraction_within(1.5, zetas))) / N.sum()
    assert rho[m.core].sum() / total == pytest.approx(expected_core, abs=5e-3)
    assert rho[m.interstitial].sum() / total == pytest.approx(expected_int, abs=5e-3)
    # the partition recovers each atom's charge
    site_q = eng.partition().site_sum(rho) * eng.grid().dV
    np.testing.assert_allclose(site_q, N, rtol=2e-2)


def test_gaussian_moments():
    # rho and rho r^2 are smooth, so their grid sums are spectrally exact;
    # rho r has a kink at the nucleus (|r|), so m1 is only algebraically accurate
    alpha = 3.0
    s = Structure(np.eye(3) * 7.0, ["X"], [[0.5, 0.5, 0.5]])
    Q, m1, m2 = _moments(_engine_for(GaussianSuperposition(s, alpha, 2.0), s, 56))
    assert Q == pytest.approx(2.0, rel=1e-8)
    assert m2 == pytest.approx(gaussian_moment(2, alpha), rel=1e-8)
    assert m1 == pytest.approx(gaussian_moment(1, alpha), rel=5e-4)


def test_gaussian_laplacian_sign_structure():
    # the Laplacian of a Gaussian is negative exactly inside r = sqrt(3/(2 alpha))
    alpha = 2.0
    s = Structure(np.eye(3) * 7.0, ["X"], [[0.0, 0.0, 0.0]])
    grid = Grid(s.lattice, (64, 64, 64))
    rho, _, hess = GaussianSuperposition(s, alpha, 1.0).on_grid(grid)
    exact = hess[..., :3].sum(-1)
    eng = Engine(s, {"rho": rho})
    eng_spec = Engine(s, {"rho": rho}, method="spectral")
    r = eng.geometry().distance
    r0 = np.sqrt(1.5 / alpha)
    # skip the sign change itself and the far tail, where |lap| is at noise level
    away = (np.abs(r - r0) > 0.05) & (np.abs(exact) > 1e-8 * np.abs(exact).max())
    for e in (eng, eng_spec):
        neg = e["rho"].laplacian < 0
        np.testing.assert_array_equal(neg[away], (r < r0)[away])


def test_gaussian_hessian_eigenvalues():
    # spherical density: radial eigenvalue f'' and two degenerate f'/r
    alpha = 2.0
    s = Structure(np.eye(3) * 7.0, ["X"], [[0.0, 0.0, 0.0]])
    g = GaussianSuperposition(s, alpha, 1.0)
    grid = Grid(s.lattice, (48, 48, 48))
    rho, _, _ = g.on_grid(grid, derivatives=False)
    eng = Engine(s, {"rho": rho}, method="spectral")
    r = eng.geometry().distance
    ev = eng["rho"].hessian_eigenvalues
    pick = (r > 0.4) & (r < 1.4)
    f = rho[pick]
    tangential = -2 * alpha * f
    radial = (4 * alpha ** 2 * r[pick] ** 2 - 2 * alpha) * f
    exp = np.sort(np.stack([radial, tangential, tangential], -1), -1)
    np.testing.assert_allclose(ev[pick], exp, atol=1e-8 * np.abs(exp).max())
