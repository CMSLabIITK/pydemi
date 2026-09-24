import numpy as np
import pytest

from pydemi import Field, Grid, hessian_eigenvalues, unpack_hessian
from pydemi.testing import GaussianSuperposition

from conftest import TRICLINIC


def _rel_err(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


@pytest.fixture
def gaussian_on_grid(triclinic_two_atoms):
    def make(n):
        grid = Grid(TRICLINIC, (n, n + 2, n + 4))
        g = GaussianSuperposition(triclinic_two_atoms, [2.0, 3.0], [4.0, 6.0])
        return (grid,) + g.on_grid(grid)
    return make


def test_spectral_exact_on_triclinic(gaussian_on_grid):
    grid, rho, grad, hess = gaussian_on_grid(40)
    assert _rel_err(grid.gradient(rho, "spectral"), grad) < 1e-9
    assert _rel_err(grid.hessian(rho, "spectral"), hess) < 1e-9
    lap = hess[..., :3].sum(-1)
    assert _rel_err(grid.laplacian(rho, "spectral"), lap) < 1e-9


def test_fd_second_order_convergence(gaussian_on_grid):
    errs = {}
    for n in (24, 48):
        grid, rho, grad, hess = gaussian_on_grid(n)
        lap = hess[..., :3].sum(-1)
        errs[n] = (_rel_err(grid.gradient(rho), grad),
                   _rel_err(grid.hessian(rho), hess),
                   _rel_err(grid.laplacian(rho), lap))
    for coarse, fine in zip(errs[24], errs[48]):
        assert 3.0 < coarse / fine < 5.0   # ~4x per halving of h


@pytest.mark.parametrize("method", ["fd", "spectral"])
def test_laplacian_is_hessian_trace(gaussian_on_grid, method):
    grid, rho, _, _ = gaussian_on_grid(20)
    H = grid.hessian(rho, method)
    np.testing.assert_allclose(grid.laplacian(rho, method), H[..., :3].sum(-1),
                               atol=1e-10 * np.abs(H).max())


def test_boundary_is_periodic():
    # a density centred on the cell corner has its structure on the grid
    # boundary; one-sided differences there would give a large error
    lat = np.eye(3) * 6.0
    grid = Grid(lat, (40, 40, 40))
    x = grid.cart_coords()
    d = x - np.round(x / 6.0) * 6.0
    r2 = (d ** 2).sum(-1)
    rho = np.exp(-r2)
    exact = (-2.0 * d) * rho[..., None]
    err = np.abs(grid.gradient(rho) - exact)
    assert err[0].max() < 3 * err[5:35].max()  # boundary plane no worse than interior


def test_hessian_eigenvalues_sorted_and_correct():
    rng = np.random.default_rng(1)
    full = rng.normal(size=(7, 3, 3))
    full = full + full.transpose(0, 2, 1)
    packed = np.stack([full[:, a, b] for a, b in
                       ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], axis=-1)
    np.testing.assert_allclose(unpack_hessian(packed), full)
    ev = hessian_eigenvalues(packed, chunk=3)
    np.testing.assert_allclose(ev, np.linalg.eigvalsh(full))
    assert np.all(np.diff(ev, axis=-1) >= 0)


def test_field_caches_and_laplacian_from_hessian(gaussian_on_grid):
    grid, rho, _, _ = gaussian_on_grid(16)
    f = Field(rho, grid, "rho")
    assert f.gradient is f.gradient
    _ = f.hessian
    np.testing.assert_allclose(f.laplacian, grid.laplacian(rho), atol=1e-10)
    assert f.hessian_eigenvalues.shape == grid.shape + (3,)
    f.clear_cache()
    assert "gradient" not in f.__dict__


def test_bad_inputs():
    grid = Grid(np.eye(3), (4, 4, 4))
    with pytest.raises(ValueError):
        grid.gradient(np.zeros((4, 4, 5)))
    with pytest.raises(ValueError):
        grid.laplacian(np.zeros((4, 4, 4)), method="nope")


def test_integrate_charge(gaussian_on_grid):
    grid, rho, _, _ = gaussian_on_grid(32)
    assert grid.integrate(rho) == pytest.approx(10.0, rel=1e-9)
