"""Milestone 11: power, Becke and Hirshfeld partitions (spec §9)."""

import functools

import numpy as np
import pytest

import pydemi
from pydemi.core.partition import BeckePartition, _becke_step, partition_of
from pydemi.descriptors.registry import FeatureOptions
from pydemi.fields.deformation import promolecule, reference_tables
from pydemi.io.base import Grid, Lattice, Structure, VolumetricData
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


@pytest.mark.parametrize("scheme", ["nearest", "power", "becke", "hirshfeld"])
def test_weights_sum_to_one_and_charge_is_conserved(scheme):
    vd = _vd().with_options(FeatureOptions(partition=scheme))
    part = partition_of(vd, scheme)
    np.testing.assert_allclose(_weight_sums(part, vd.rho.n_voxels), 1.0, atol=1e-12)
    q = part.site_sum(vd.rho.data) * vd.rho.dV
    assert q.sum() == pytest.approx(vd.rho.data.sum() * vd.rho.dV, rel=1e-12)


@pytest.mark.parametrize("scheme", ["becke", "hirshfeld"])
def test_symmetric_atoms_get_equal_charges(scheme):
    s = Structure(Lattice(np.eye(3) * 5.0), ["Na", "Na"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    vd = GaussianSuperposition(s, [1.5, 1.5], [1.0, 1.0], tol=1e-20).volumetric((20, 20, 20))
    q = partition_of(vd.with_options(FeatureOptions()), scheme).site_sum(vd.rho.data)
    # Becke products are truncated to the BECKE_K nearest images, and the truncation
    # breaks exact ties between equidistant images (documented): symmetric only to ~1e-5
    assert q[0] == pytest.approx(q[1], rel=1e-10 if scheme == "hirshfeld" else 1e-5)


def test_becke_two_atom_weight_is_becke_s():
    """With two atoms (all images far), w_A(r) = s(mu_AB); 1/2 on the bisecting plane."""
    mu = np.linspace(-1, 1, 11)
    s = _becke_step(mu.copy())
    np.testing.assert_allclose(s + _becke_step(-mu.copy()), 1.0)
    assert s[5] == pytest.approx(0.5) and s[0] == pytest.approx(1.0) and s[-1] == pytest.approx(0.0)


def test_hirshfeld_charges_of_a_promolecule():
    """Density = the free-atom superposition itself: w_i rho is atom i's own free density, so
    q_i = ZVAL_i - int rho_free_i dV exactly -- zero up to point sampling. Z_i is the valence
    count for a pseudo-density (documented correction)."""
    zval = {"Na": 1.0, "Cl": 7.0}
    lat = Lattice(np.eye(3) * 5.2)
    frac = [[0.013, 0.021, 0.017], [0.5137, 0.4871, 0.5213]]
    s = Structure(lat, ["Na", "Cl"], frac)
    shape = (52, 52, 52)
    dV = lat.volume / np.prod(shape)
    rho = promolecule(shape, s, reference_tables(s, "tabulated", "valence", zval))
    q = pydemi.hirshfeld_charges(VolumetricData(s, Grid(rho, lat), zval=zval))
    for i, el in enumerate(["Na", "Cl"]):
        one = Structure(lat, [el], [frac[i]])
        n_i = promolecule(shape, one, reference_tables(one, "tabulated", "valence", zval)).sum() * dV
        assert q[i] == pytest.approx(zval[el] - n_i, abs=1e-10)
    np.testing.assert_allclose(q, 0.0, atol=5e-3)


def test_site_descriptors_follow_the_partition():
    vd = _vd()
    near = pydemi.featurize(vd, domains=["heterogeneity"])
    hirsh, meta = pydemi.featurize(vd, domains=["heterogeneity"], partition="hirshfeld",
                                   return_metadata=True)
    becke, bmeta = pydemi.featurize(vd, domains=["heterogeneity"], partition="becke",
                                    return_metadata=True)
    assert meta["partition"] == "hirshfeld" and bmeta["partition"].startswith("becke(k=")
    assert near["m1_site_std"] != pytest.approx(hirsh["m1_site_std"])
    assert near["m1_site_std"] != pytest.approx(becke["m1_site_std"])


@functools.lru_cache(maxsize=None)
def _feats(scheme, transform):
    vd = _vd(shape=(16, 16, 18))
    vd = {"base": vd, "supercell": supercell(vd), "translation": translate(vd, (3, 5, 7)),
          "rotation": rotate(vd, rotation((1, 2, 0.5), 37.0))}[transform]
    return pydemi.featurize(vd, domains=["heterogeneity", "magnetic"], partition=scheme)


@pytest.mark.parametrize("transform", ["supercell", "translation", "rotation"])
@pytest.mark.parametrize("scheme", ["power", "hirshfeld", "becke"])
def test_partitioned_descriptors_are_invariant(scheme, transform):
    bad = compare(_feats(scheme, "base"), _feats(scheme, transform), rtol=1e-6, atol=1e-10)
    assert not bad, bad
