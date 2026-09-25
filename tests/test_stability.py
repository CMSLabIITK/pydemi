"""Stability tags and the robust ellipticity (decision of 2026-09-25: E2, E3, C2)."""

import numpy as np
import pytest

import pydemi
from pydemi.descriptors.registry import REGISTRY, make_options, masks, register
from pydemi.io.base import Grid, Lattice, Structure, VolumetricData

FRAGILE = {"ellip_bond_avg", "ellip_bond_std", "n_saddle1", "n_saddle2"}


def test_fragile_set_is_the_measured_one():
    """The tags follow paper/analysis (derivative-scheme and 80%-grid sensitivity > ~10%)."""
    assert {n for n, s in REGISTRY.items() if s.stability == "fragile"} == FRAGILE
    cat = pydemi.catalogue()
    assert set(cat.loc[cat.stability == "fragile", "name"]) == FRAGILE
    assert set(cat.stability) == {"robust", "fragile"}


def test_fragile_descriptors_stay_in_the_default_output():
    everything = pydemi.descriptor_names()
    model_ready = pydemi.descriptor_names(include_fragile=False)
    assert FRAGILE <= set(everything)
    assert set(everything) - set(model_ready) == FRAGILE
    assert [n for n in everything if n not in FRAGILE] == model_ready          # order kept


def test_robust_extension_is_off_by_default():
    assert "ellip_bond_bounded_avg" not in pydemi.descriptor_names()
    assert "ellip_bond_bounded_avg" in pydemi.descriptor_names(extensions=["robust"])
    assert REGISTRY["ellip_bond_bounded_avg"].stability == "robust"


def test_register_validates_stability():
    with pytest.raises(ValueError, match="stability"):
        register(name="_bad", domain="bonding", field="rho", requires=[], units="1",
                 stability="shaky")(lambda vd: 0.0)


def _gaussian(a, b, c, n=96, box=10.0):
    """rho = exp(-(a x^2 + b y^2 + c z^2)) centred in a cubic box, and its exact Hessian."""
    s = Structure(Lattice(np.eye(3) * box), ["H"], [[0.5, 0.5, 0.5]])
    x = (np.arange(n) / n - 0.5) * box
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    rho = np.exp(-(a * X ** 2 + b * Y ** 2 + c * Z ** 2))
    g = np.stack([-2 * a * X, -2 * b * Y, -2 * c * Z], axis=-1)
    H = rho[..., None, None] * (g[..., :, None] * g[..., None, :] + np.diag([-2 * a, -2 * b, -2 * c]))
    return VolumetricData(s, Grid(rho, s.lattice)), H


@pytest.mark.parametrize("abc", [(1.0, 1.5, 2.0), (0.8, 0.8, 1.6)])
def test_bounded_ellipticity_matches_the_exact_hessian(abc):
    vd, H = _gaussian(*abc)
    got = pydemi.featurize(vd, domains=["bonding"], extensions=["robust"])["ellip_bond_bounded_avg"]
    ev = np.linalg.eigvalsh(H)
    sel = masks(vd.with_options(make_options())).bond & (ev[..., 1] < 0)
    exact = float(np.mean(1.0 - ev[..., 1][sel] / ev[..., 0][sel]))
    assert 0.0 < got < 1.0
    assert got == pytest.approx(exact, rel=1e-3)


def test_bounded_ellipticity_is_zero_for_a_spherical_atom():
    vd, _ = _gaussian(1.2, 1.2, 1.2)
    f = pydemi.featurize(vd, domains=["bonding"], extensions=["robust"])
    assert abs(f["ellip_bond_bounded_avg"]) < 1e-6
    assert abs(f["ellip_bond_avg"]) < 1e-6          # the spec form agrees where it is well defined


def test_bounded_ellipticity_is_a_monotone_map_of_the_ellipticity():
    """Per voxel 1 - l2/l1 = e / (1 + e); here e is uniform in the shell, so the means agree too."""
    e = np.array([0.0, 0.1, 1.0, 10.0, 1e6])
    l2 = -1.0
    l1 = (1 + e) * l2
    np.testing.assert_allclose(1 - l2 / l1, e / (1 + e))
    assert np.all(np.diff(e / (1 + e)) > 0) and np.all(e / (1 + e) < 1)
