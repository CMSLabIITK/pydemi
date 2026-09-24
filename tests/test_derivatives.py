"""Milestone 2: derivatives on periodic, non-orthogonal grids (spec §4)."""

import functools
import itertools

import numpy as np
import pytest
from scipy.spatial import cKDTree

from pydemi.core import derivatives as D
from pydemi.core.grid import cart_coords
from pydemi.io.base import Lattice, Structure
from pydemi.validate.analytic import GaussianSuperposition, SlaterSuperposition

TRICLINIC = Lattice(np.array([[4.0, 0.0, 0.0], [1.3, 4.2, 0.0], [0.7, 0.9, 4.5]]))
ORTHO = Lattice(np.diag([4.0, 4.4, 4.8]))
FRAC = [[0.1, 0.2, 0.3], [0.6, 0.55, 0.8]]


def _gauss(lat, shape):
    g = GaussianSuperposition(Structure(lat, ["H", "H"], FRAC), [2.0, 3.0], [1.0, 2.0])
    return g.on_grid(shape)


def _rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


@pytest.mark.parametrize("lat", [ORTHO, TRICLINIC], ids=["orthorhombic", "triclinic"])
def test_fft_exact_for_band_limited(lat):
    rho, grad, H = _gauss(lat, (32, 34, 36))
    lap = np.trace(H, axis1=-2, axis2=-1)
    assert _rel(D.gradient(rho, lat, "fft"), grad) < 1e-10
    assert _rel(D.laplacian(rho, lat, "metric", "fft"), lap) < 1e-10
    assert _rel(D.hessian(rho, lat, "fft"), H) < 1e-10


@pytest.mark.parametrize("lat", [ORTHO, TRICLINIC], ids=["orthorhombic", "triclinic"])
@pytest.mark.parametrize("order", [2, 4, 6])
def test_fd_convergence_order(lat, order):
    errs = []
    for n in (32, 64):
        rho, grad, H = _gauss(lat, (n, n, n))
        lap = np.trace(H, axis1=-2, axis2=-1)
        errs.append((_rel(D.gradient(rho, lat, "fd", order), grad),
                     _rel(D.laplacian(rho, lat, "metric", "fd", order), lap),
                     _rel(D.hessian(rho, lat, "fd", order), H)))
    for coarse, fine in zip(*errs):
        rate = np.log2(coarse / fine)
        assert rate > order - 0.6, (order, rate)


def test_metric_laplacian_needs_off_diagonal_terms():
    """Diagonal-only is exact for orthorhombic cells and materially wrong otherwise."""
    rho, _, H = _gauss(ORTHO, (32, 32, 32))
    lap = np.trace(H, axis1=-2, axis2=-1)
    np.testing.assert_allclose(D.laplacian(rho, ORTHO, "diagonal"), D.laplacian(rho, ORTHO, "metric"))
    rho, _, H = _gauss(TRICLINIC, (32, 32, 32))
    lap = np.trace(H, axis1=-2, axis2=-1)
    assert _rel(D.laplacian(rho, TRICLINIC, "metric"), lap) < 1e-10
    assert _rel(D.laplacian(rho, TRICLINIC, "diagonal"), lap) > 0.05


@pytest.mark.parametrize("backend", ["fft", "fd"])
def test_hessian_symmetric_trace_and_eigenvalues(backend):
    rho, _, H = _gauss(TRICLINIC, (32, 32, 32))
    Hn = D.hessian(rho, TRICLINIC, backend)
    np.testing.assert_array_equal(Hn, np.swapaxes(Hn, -1, -2))
    np.testing.assert_allclose(np.trace(Hn, axis1=-2, axis2=-1),
                               D.laplacian(rho, TRICLINIC, "metric", backend), atol=1e-9)
    ev = D.hessian_eigenvalues(rho, TRICLINIC, backend)
    assert np.all(np.diff(ev, axis=-1) >= 0)                              # ascending
    tol = 1e-9 if backend == "fft" else 5e-3
    assert _rel(ev, np.linalg.eigvalsh(H)) < tol
    small = D.eigenvalues_packed(D.cartesian_hessian_packed(
        D.frac_hessian(rho, backend), TRICLINIC), chunk=1000)
    np.testing.assert_allclose(small, ev)                                 # chunking is exact


@functools.lru_cache(maxsize=None)
def _slater(lat, n):
    s = Structure(lat, ["H", "H"], FRAC)
    rho, grad, H = SlaterSuperposition(s, [2.0, 2.5], tol=1e-12).on_grid((n, n, n))
    x = cart_coords((n, n, n), lat).reshape(-1, 3)
    shifts = np.array(list(itertools.product(range(-1, 2), repeat=3))) @ lat.matrix
    images = (s.cart_coords[:, None, :] + shifts[None, :, :]).reshape(-1, 3)
    d, _ = cKDTree(images).query(x)
    return rho, grad, np.trace(H, axis1=-2, axis2=-1), d.reshape(rho.shape)


@pytest.mark.parametrize("lat", [ORTHO, TRICLINIC], ids=["orthorhombic", "triclinic"])
def test_slater_fd_converges_away_from_the_cusp(lat):
    errs = []
    for n in (48, 96):
        rho, grad, lap, d = _slater(lat, n)
        far = d > 0.5
        errs.append((np.abs(D.gradient(rho, lat, "fd") - grad)[far].max() / np.abs(grad[far]).max(),
                     np.abs(D.laplacian(rho, lat, "metric", "fd") - lap)[far].max()
                     / np.abs(lap[far]).max()))
    assert errs[1][0] < 1e-3 and errs[1][1] < 5e-3
    assert all(c / f > 8 for c, f in zip(*errs))


def test_fft_rings_on_a_cusp():
    """Documented limitation (spec §4): spectral derivatives of a cusped density
    ring across the cell and do not converge with grid refinement."""
    for n in (48, 96):
        rho, grad, lap, d = _slater(TRICLINIC, n)
        far = d > 0.5
        err = np.abs(D.laplacian(rho, TRICLINIC, "metric", "fft") - lap)[far].max()
        assert err / np.abs(lap[far]).max() > 0.5


def test_g_squared_matches_spectral_laplacian():
    rho, _, _ = _gauss(TRICLINIC, (20, 22, 24))
    from scipy import fft as sfft
    F = sfft.rfftn(rho)
    lap = sfft.irfftn(-D.g_squared(TRICLINIC, rho.shape) * F, s=rho.shape)
    np.testing.assert_allclose(lap, D.laplacian(rho, TRICLINIC, "metric", "fft"), atol=1e-10)


def test_invalid_arguments():
    rho = np.zeros((4, 4, 4))
    with pytest.raises(ValueError, match="backend"):
        D.gradient(rho, ORTHO, "spectral")
    with pytest.raises(ValueError, match="order"):
        D.gradient(rho, ORTHO, "fd", 3)
    with pytest.raises(ValueError, match="method"):
        D.laplacian(rho, ORTHO, "full")
