"""Milestone 11: power and Becke partitions (spec §9)."""

import functools

import numpy as np
import pytest

import pydemi
from pydemi.core.partition import BeckePartition, _becke_step, partition_of
from pydemi.descriptors.registry import FeatureOptions
from pydemi.io.base import Lattice, Structure
from pydemi.validate.analytic import GaussianSuperposition
from pydemi.validate.invariance import compare, rotate, rotation, supercell, translate

LAT = Lattice(np.array([[4.0, 0.0, 0.0], [0.5, 4.1, 0.0], [0.3, 0.4, 4.3]]))


def _vd(species=("Fe", "O", "O"), frac=((0.0, 0.0, 0.0), (0.5, 0.45, 0.5), (0.2, 0.7, 0.35)),
        shape=(20, 20, 22)):
    s = Structure(LAT, list(species), [list(f) for f in frac])
    return GaussianSuperposition(s, [1.4, 2.2, 2.2], [8.0, 6.0, 6.0], tol=1e-20).volumetric(shape)


def _weight_sums(part, n):
    total = np.zeros(n)
    for p in part.pairs():
        np.add.at(total, p.voxel, p.weight)
    return total


@pytest.mark.parametrize("scheme", ["nearest", "power", "becke"])
def test_weights_sum_to_one_and_charge_is_conserved(scheme):
    vd = _vd().with_options(FeatureOptions(partition=scheme))
    part = partition_of(vd, scheme)
    np.testing.assert_allclose(_weight_sums(part, vd.rho.n_voxels), 1.0, atol=1e-12)
    q = part.site_sum(vd.rho.data) * vd.rho.dV
    assert q.sum() == pytest.approx(vd.rho.data.sum() * vd.rho.dV, rel=1e-12)


def test_symmetric_atoms_get_equal_charges():
    s = Structure(Lattice(np.eye(3) * 5.0), ["Na", "Na"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    vd = GaussianSuperposition(s, [1.5, 1.5], [1.0, 1.0], tol=1e-20).volumetric((20, 20, 20))
    q = partition_of(vd.with_options(FeatureOptions()), "becke").site_sum(vd.rho.data)
    # Becke products are truncated to the BECKE_K nearest images, and the truncation
    # breaks exact ties between equidistant images (documented): symmetric only to ~1e-5
    assert q[0] == pytest.approx(q[1], rel=1e-5)


def test_becke_two_atom_weight_is_becke_s():
    """With two atoms (all images far), w_A(r) = s(mu_AB); 1/2 on the bisecting plane."""
    mu = np.linspace(-1, 1, 11)
    s = _becke_step(mu.copy())
    np.testing.assert_allclose(s + _becke_step(-mu.copy()), 1.0)
    assert s[5] == pytest.approx(0.5) and s[0] == pytest.approx(1.0) and s[-1] == pytest.approx(0.0)


def test_site_descriptors_follow_the_partition():
    vd = _vd()
    near = pydemi.featurize(vd, domains=["heterogeneity"])
    becke, bmeta = pydemi.featurize(vd, domains=["heterogeneity"], partition="becke",
                                    return_metadata=True)
    assert bmeta["partition"].startswith("becke(k=")
    assert near["m1_site_std"] != pytest.approx(becke["m1_site_std"])


@functools.lru_cache(maxsize=None)
def _feats(scheme, transform):
    vd = _vd(shape=(16, 16, 18))
    vd = {"base": vd, "supercell": supercell(vd), "translation": translate(vd, (3, 5, 7)),
          "rotation": rotate(vd, rotation((1, 2, 0.5), 37.0))}[transform]
    return pydemi.featurize(vd, domains=["heterogeneity"], partition=scheme)


@pytest.mark.parametrize("transform", ["supercell", "translation", "rotation"])
@pytest.mark.parametrize("scheme", ["power", "becke"])
def test_partitioned_descriptors_are_invariant(scheme, transform):
    bad = compare(_feats(scheme, "base"), _feats(scheme, transform), rtol=1e-6, atol=1e-10)
    assert not bad, bad
