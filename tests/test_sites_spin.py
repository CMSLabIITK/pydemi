"""Phase 3: partitions (H), site-resolved statistics (C), spin density (E)."""

import itertools

import numpy as np
import pytest

from pydemi import Engine, Grid, Structure, assign_atoms
from pydemi.descriptors import compute_descriptors, names, REGISTRY, tier1
from pydemi.descriptors.sites import (partition_family, site_charges, site_family,
                                      site_statistics, site_sums, site_values,
                                      variance_decomposition, whole_cell_values)
from pydemi.descriptors.spin import site_moments, spin_family
from pydemi.io.vasp import (SPIN_COLLINEAR, SPIN_NONCOLLINEAR, ChargeDensity)
from pydemi.partition import BeckePartition, _becke_step
from pydemi.testing import (GaussianSuperposition, SlaterSuperposition,
                            gaussian_moment, slater_fraction_within, slater_moment)

from conftest import TRICLINIC


def _density(structure, alphas, N, n):
    grid = Grid(structure.lattice, n if isinstance(n, tuple) else (n, n, n))
    rho, _, _ = GaussianSuperposition(structure, alphas, N).on_grid(grid, False)
    return rho


@pytest.fixture(scope="module")
def nacl():
    s = Structure(np.eye(3) * 5.0, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    return Engine(s, {"rho": _density(s, [1.2, 0.8], [1.0, 7.0], 28)})


# ---------------------------------------------------------------- partitions

def test_nearest_partition_reproduces_tier1(nacl):
    cell = whole_cell_values(site_sums(nacl))
    t1 = tier1(nacl)
    for k, v in cell.items():
        assert v == pytest.approx(t1[k], rel=1e-12, abs=1e-15), k


def test_power_with_equal_radii_is_nearest(triclinic_two_atoms):
    grid = Grid(TRICLINIC, (9, 10, 11))
    near = assign_atoms(grid, triclinic_two_atoms)
    power = assign_atoms(grid, triclinic_two_atoms, power_radii=[1.3, 1.3])
    np.testing.assert_array_equal(near.atom_index, power.atom_index)
    np.testing.assert_allclose(near.distance, power.distance)


def test_power_matches_brute_force(skewed_cell):
    grid = Grid(skewed_cell.lattice, (8, 9, 10))
    radii = np.array([1.4, 0.7, 1.0])
    got = assign_atoms(grid, skewed_cell, power_radii=radii)
    x = grid.cart_coords().reshape(-1, 3)
    best = np.full(x.shape[0], np.inf)
    dist = np.zeros(x.shape[0])
    for t in itertools.product(range(-4, 5), repeat=3):
        shift = np.array(t) @ skewed_cell.lattice
        for i, R in enumerate(skewed_cell.cart_coords):
            d2 = np.sum((x - R - shift) ** 2, axis=1)
            p = d2 - radii[i] ** 2
            better = p < best
            best[better], dist[better] = p[better], np.sqrt(d2[better])
    np.testing.assert_allclose(got.distance.ravel(), dist, atol=1e-10)


def test_power_gives_larger_atoms_more_volume(nacl):
    near = nacl.partition(scheme="nearest").site_count()
    power = nacl.partition(scheme="power", radii={"Na": 0.6, "Cl": 1.6}).site_count()
    # symmetric positions: equal up to tie-breaking on the cell boundary
    assert near[0] == pytest.approx(near[1], rel=3e-2)
    assert power[1] / power[0] > 1.5
    # default radii come from the Magpie covalent table
    np.testing.assert_allclose(nacl.atom_radii(), [1.66, 1.02])


def test_becke_weights_are_a_partition_of_unity(nacl):
    part = nacl.partition(scheme="becke", k=20)
    counts = part.site_count()
    assert counts.sum() == pytest.approx(28 ** 3, rel=1e-10)
    for p in part.pairs():
        assert np.all((p.weight > 0) & (p.weight <= 1 + 1e-12))
    np.testing.assert_allclose(site_charges(nacl, "becke", k=20).sum(),
                               nacl["rho"].integral(), rtol=1e-10)


def test_becke_two_atom_closed_form():
    # isolated homonuclear pair: w_1(r) = s(mu), mu = (r_1 - r_2) / R
    L = 24.0
    s = Structure(np.eye(3) * L, ["H", "H"], [[0.45, 0.5, 0.5], [0.55, 0.5, 0.5]])
    grid = Grid(s.lattice, (40, 8, 8))
    part = BeckePartition(grid, s, k=2, cells=2)
    x = grid.cart_coords().reshape(-1, 3)
    R = 0.1 * L
    w1 = np.zeros(x.shape[0])
    for p in part.pairs():
        np.add.at(w1, p.voxel[p.atom == 0], p.weight[p.atom == 0])
    d1 = np.linalg.norm(x - s.cart_coords[0], axis=1)
    d2 = np.linalg.norm(x - s.cart_coords[1], axis=1)
    near = (d1 < 4) & (d2 < 4)              # periodic images irrelevant here
    np.testing.assert_allclose(w1[near], _becke_step((d1 - d2)[near] / R), atol=1e-12)


def test_becke_size_adjustment_favours_the_larger_atom(nacl):
    plain = site_charges(nacl, "becke", k=20)
    sized = site_charges(nacl, "becke", k=20, radii={"Na": 0.6, "Cl": 1.6})
    assert sized[1] > plain[1]
    with pytest.raises(ValueError):
        BeckePartition(nacl.grid(), nacl.structure, k=4, cells=8)


def test_becke_is_opt_in(nacl):
    d = compute_descriptors(nacl, families=("H",))
    assert "m1_power" in d and "m1_becke" not in d
    assert "m1_becke" not in names() and "m1_becke" in names(opt_in=True)
    both = partition_family(nacl, schemes=("power", "becke"), k=20)
    assert "m1_becke" in both and "zeta_site_std_becke" in both


# ---------------------------------------------------------------- Family C

def test_variance_decomposition_is_exact():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    g = rng.choice(["Fe", "Co", "Ni"], size=40)
    within, between, share = variance_decomposition(x, g)
    assert within + between == pytest.approx(np.var(x), rel=1e-12)
    assert 0 < share < 1


def test_variance_decomposition_singletons_are_undefined():
    within, between, share = variance_decomposition([1.0, 2.0, 4.0], ["A", "B", "C"])
    assert np.isnan(within) and np.isnan(share)
    assert between == pytest.approx(np.var([1.0, 2.0, 4.0]))


def test_site_values_against_slater_closed_forms():
    lat = np.array([[12.0, 0, 0], [2.0, 11.0, 0], [1.0, 1.5, 12.0]])
    s = Structure(lat, ["A", "B"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    zetas = np.array([1.8, 2.6])
    rho, _, _ = SlaterSuperposition(s, zetas, [3.0, 5.0], tol=1e-10).on_grid(
        Grid(lat, (110, 110, 110)), False)
    v = site_values(site_sums(Engine(s, {"rho": rho})))
    np.testing.assert_allclose(v["m1"], [slater_moment(1, z) for z in zetas], rtol=1e-2)
    bond = slater_fraction_within(1.5, zetas) - slater_fraction_within(0.8, zetas)
    np.testing.assert_allclose(v["f_bond"], bond, atol=5e-3)
    np.testing.assert_allclose(v["charge"], [3.0, 5.0], rtol=2e-2)


def test_site_family_sees_chemical_differentiation():
    # two elements, two sites each; same-element sites identical by symmetry
    s = Structure(np.eye(3) * 7.0, ["Na", "Na", "Cl", "Cl"],
                  [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    alphas = {"Na": 1.4, "Cl": 0.7}
    eng = Engine(s, {"rho": _density(s, alphas, {"Na": 1.0, "Cl": 7.0}, 36)})
    c = site_family(eng)
    assert c["n_atoms"] == 4
    # same-element sites differ only by boundary tie-breaking
    assert c["m1_within_share"] < 0.02
    assert c["m1_var_within"] + c["m1_var_between"] == pytest.approx(c["m1_site_std"] ** 2,
                                                                     rel=1e-10)
    assert c["m1_site_range"] == pytest.approx(c["m1_site_max"] - c["m1_site_min"])


def test_site_statistics_ignore_empty_sites():
    stats = site_statistics({"m1": np.array([1.0, np.nan, 3.0])}, ["A", "A", "B"],
                            quantities=("m1",))
    assert stats["m1_site_std"] == pytest.approx(1.0)


# ---------------------------------------------------------------- Family E

def _spin_engine(moments, alpha=1.5, L=8.0, noncollinear=False):
    """Two well-separated atoms with Gaussian charge and magnetization."""
    s = Structure(np.eye(3) * L, ["Fe", "Fe"], [[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
    grid = Grid(s.lattice, (40, 40, 40))
    # one periodic, unit-charge Gaussian per atom
    blobs = [GaussianSuperposition(Structure(s.lattice, ["Fe"], [f]), alpha, 1.0)
             .on_grid(grid, False)[0] for f in s.frac_coords]
    rho = 8.0 * (blobs[0] + blobs[1])
    if noncollinear:
        m = sum(np.asarray(mom)[:, None, None, None] * b for mom, b in zip(moments, blobs))
        cd = ChargeDensity(s, rho, m, SPIN_NONCOLLINEAR)
    else:
        m = moments[0] * blobs[0] + moments[1] * blobs[1]
        cd = ChargeDensity(s, rho, m, SPIN_COLLINEAR)
    return Engine.from_charge_density(cd)


def test_ferromagnet():
    e = spin_family(_spin_engine([2.0, 2.0]))
    assert e["M_abs"] == pytest.approx(4.0, rel=1e-6)
    assert e["M_net"] == pytest.approx(4.0, rel=1e-6)
    assert e["spin_frustration"] == pytest.approx(0.0, abs=1e-9)
    assert e["mu_site_std"] == pytest.approx(0.0, abs=5e-3)   # boundary tie-breaking
    assert e["m1_spin"] == pytest.approx(gaussian_moment(1, 1.5), rel=1e-3)
    assert e["spin_charge_correlation"] == pytest.approx(1.0, abs=1e-9)
    assert (e["is_spin_polarized"], e["is_magnetic"]) == (1, 1)


def test_antiferromagnet():
    eng = _spin_engine([2.0, -2.0])
    e = spin_family(eng)
    # opposite moments partially cancel where the blobs overlap: M_abs < 4
    assert 3.99 < e["M_abs"] < 4.0
    assert e["M_net"] == pytest.approx(0.0, abs=1e-9)
    assert e["spin_frustration"] == pytest.approx(1.0, abs=1e-9)
    assert e["mu_site_std"] == pytest.approx(2.0, rel=2e-3)
    np.testing.assert_allclose(site_moments(eng), [2.0, -2.0], rtol=2e-3)


def test_noncollinear_uses_vector_sums():
    # perpendicular moments: collinear bookkeeping would call this either
    # fully aligned or fully cancelled; the vector answer is 1 - 1/sqrt(2)
    eng = _spin_engine([[2.0, 0, 0], [0, 2.0, 0]], noncollinear=True)
    e = spin_family(eng)
    assert e["M_net"] == pytest.approx(2.0 * np.sqrt(2), rel=1e-6)
    assert 3.99 < e["M_abs"] < 4.0            # |m_vec| < |m_1| + |m_2| on the overlap
    # each site also picks up the tail of its neighbour's moment
    assert e["spin_frustration"] == pytest.approx(1 - 1 / np.sqrt(2), rel=3e-3)
    assert site_moments(eng).shape == (2, 3)
    assert e["mu_site_std"] == pytest.approx(np.sqrt(2.0), rel=2e-3)


def test_non_spin_polarized_sentinel(nacl):
    e = spin_family(nacl)
    assert all(e[k] == 0 for k in e)
    assert np.all(site_moments(nacl) == 0)


def test_spin_polarized_but_non_magnetic():
    e = spin_family(_spin_engine([1e-5, 1e-5]))
    assert e["is_spin_polarized"] == 1 and e["is_magnetic"] == 0
    assert e["M_abs"] > 0
    assert e["m1_spin"] == 0.0 and e["spin_frustration"] == 0.0


# ---------------------------------------------------------------- assembly

def test_every_default_scalar_is_computed():
    eng = _spin_engine([2.0, -1.0])
    shape = eng.grid().shape
    eng.add_field("elf", np.full((20, 20, 20), 0.6))
    eng.add_field("potential", np.cos(2 * np.pi * eng.grid().frac_coords()[..., 0]))
    d = compute_descriptors(eng)
    assert set(d) == set(names())
    assert all(REGISTRY[k].kind in ("descriptor", "variant", "cross_term",
                                    "preprocessing", "metadata") for k in d)
    assert shape == (40, 40, 40)
