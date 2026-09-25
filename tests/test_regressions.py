"""Regressions found in the code-guide review of 2026-09-25."""

import numpy as np
import pytest

import pydemi
from pydemi.core.geometry import Shells
from pydemi.descriptors.registry import geometry
from pydemi.io.base import Lattice, Structure
from pydemi.validate.analytic import GaussianSuperposition

LAT = Lattice(np.array([[4.0, 0.0, 0.0], [0.9, 4.1, 0.0], [0.6, 0.7, 4.3]]))      # triclinic
S = Structure(LAT, ["Fe", "O", "O"], [[0.0, 0.0, 0.0], [0.5, 0.45, 0.5], [0.2, 0.7, 0.35]])


def _vd():
    return GaussianSuperposition(S, [1.4, 2.2, 2.2], [8.0, 6.0, 6.0], tol=1e-20).volumetric((20, 20, 22))


@pytest.mark.parametrize("first, second", [({"laplacian_method": "metric"}, {"laplacian_method": "diagonal"}),
                                           ({"derivative_backend": "fd", "fd_order": 2},
                                            {"derivative_backend": "fd", "fd_order": 6})])
def test_reusing_an_object_with_other_options_gives_fresh_values(first, second):
    """Derived-field derivatives and site statistics are cached per option set."""
    fresh = pydemi.featurize(_vd(), **second)
    vd = _vd()
    pydemi.featurize(vd, **first)
    reused = pydemi.featurize(vd, **second)
    for k in fresh:
        assert reused[k] == pytest.approx(fresh[k], rel=1e-12, abs=1e-15, nan_ok=True), k


def test_rho_min_int_with_radius_scaled_shells():
    """In scaled mode c2 is a multiple of the covalent radius (O: 0.66 A), not 1.0 A: a dip
    0.7-0.95 A from an O nucleus lies outside max(c2_O, R_PAW) and must set the floor."""
    vd = _vd()
    vd.paw_radii = {"Fe": 0.5, "O": 0.4}
    geo = geometry(vd)
    o = np.array([s == "O" for s in vd.structure.species])[geo.atom_index]
    k = np.flatnonzero((o & (geo.distance > 0.7) & (geo.distance < 0.95)).ravel())[0]
    vd.rho.data.ravel()[k] = -1.0
    sh = Shells(0.5, 1.0, scaled=True)
    f = pydemi.featurize(vd, domains=["structural"], extensions=["paw"], shells=sh)
    assert f["rho_min_int"] == -1.0
    f_abs = pydemi.featurize(vd, domains=["structural"], extensions=["paw"], shells=Shells(0.5, 1.0))
    assert f_abs["rho_min_int"] > -1.0                 # absolute 1.0 A excludes that voxel
