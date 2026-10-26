"""Structure and density viewer: strided grid, slices and figures."""

import pickle

import numpy as np
import pytest

pytest.importorskip("plotly")

from pydemi.gui import viewer  # noqa: E402
from pydemi.io.base import Lattice, Structure
from pydemi.validate.analytic import GaussianSuperposition


def _vd(shape):
    s = Structure(Lattice(np.array([[4.0, 0.0, 0.0], [0.5, 4.1, 0.0], [0.3, 0.4, 4.3]])),
                  ["Na", "Cl", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.45, 0.5], [0.2, 0.7, 0.35]])
    return GaussianSuperposition(s, [1.4, 2.2, 2.2], [1.0, 8.0, 8.0], tol=1e-20).volumetric(shape)


@pytest.fixture(scope="module")
def small():
    vd = _vd((20, 20, 22))
    return vd, viewer.preview(vd)


def test_small_grid_is_kept_whole(small):
    vd, p = small
    assert p["step"] == (1, 1, 1)
    np.testing.assert_allclose(p["grid"], vd.rho.data, rtol=1e-6)
    assert [len(a) for a in p["axes"]] == list(vd.shape)


def test_large_grid_is_strided_to_real_grid_values():
    vd = _vd((200, 30, 97))
    p = viewer.preview(vd)
    assert p["step"] == (3, 1, 2)
    assert all(n <= viewer.MAX_GRID for n in p["grid"].shape)
    np.testing.assert_allclose(p["grid"], vd.rho.data[::3, ::1, ::2], rtol=1e-6)
    np.testing.assert_allclose(p["axes"][0], np.arange(0, 200, 3) / 200)
    assert len(pickle.dumps(p)) < 5_000_000


def test_slice_picks_the_nearest_plane_periodically(small):
    _, p = small
    plane, at = viewer.slice_plane(p, 2, 0.5)
    assert at == pytest.approx(11 / 22)
    np.testing.assert_array_equal(plane, p["grid"][:, :, 11])
    _, at = viewer.slice_plane(p, 0, 0.99)          # nearer to 1 = 0 than to 19/20
    assert at == 0.0
    plane, _ = viewer.slice_plane(p, 1, 0.25)
    assert plane.shape == (20, 22)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("log", [True, False])
def test_figures_serialise(small, axis, log):
    _, p = small
    fig, caption = viewer.slice_figure(p, axis, 0.0, log)
    fig.to_json()
    assert caption and fig.layout.title.text.startswith("ρ on the")
    heat = fig.data[0]
    u, v = (k for k in range(3) if k != axis)       # x runs along u, y along v
    assert np.asarray(heat.z).shape == (len(p["axes"][v]) + 1, len(p["axes"][u]) + 1)  # periodic edge
    assert heat.x[-1] == pytest.approx(np.linalg.norm(p["cell"][u]))
    viewer.structure_figure(p).to_json()


def test_atoms_on_the_plane_are_marked(small):
    _, p = small
    fig, _ = viewer.slice_figure(p, 2, 0.0)        # Na sits at c = 0
    marked = [t for d in fig.data[1:] for t in d.text]
    assert "Na1" in marked and "Cl1" not in marked
    assert marked.count("Na1") == 4               # at the origin: drawn at all four corners


def test_ordinal():
    assert [viewer._ordinal(n) for n in (2, 3, 4, 11, 12, 21, 22)] == \
        ["2nd", "3rd", "4th", "11th", "12th", "21st", "22nd"]
