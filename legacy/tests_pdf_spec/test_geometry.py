import itertools

import numpy as np
import pytest

from pydemi import Engine, Grid, Shells, nearest_atom


def _brute_force(grid, structure, reps=4):
    x = grid.cart_coords().reshape(-1, 3)
    best = np.full(x.shape[0], np.inf)
    owner = np.zeros(x.shape[0], dtype=int)
    for t in itertools.product(range(-reps, reps + 1), repeat=3):
        shift = np.array(t) @ structure.lattice
        for i, R in enumerate(structure.cart_coords):
            d = np.linalg.norm(x - (R + shift), axis=1)
            closer = d < best
            best[closer], owner[closer] = d[closer], i
    return best.reshape(grid.shape), owner.reshape(grid.shape)


@pytest.mark.parametrize("fixture", ["skewed_cell", "triclinic_two_atoms", "cubic_one_atom"])
def test_matches_brute_force(request, fixture):
    s = request.getfixturevalue(fixture)
    grid = Grid(s.lattice, (9, 10, 11))
    geo = nearest_atom(grid, s)
    dist, owner = _brute_force(grid, s)
    np.testing.assert_allclose(geo.distance, dist, atol=1e-10)
    # ownership may legitimately differ only on exact ties
    tie_free = np.abs(dist - geo.distance) < 1e-10
    assert np.all((geo.atom_index == owner) | ~tie_free)


def test_fractional_rounding_is_not_enough(skewed_cell):
    # documents why the KD-tree search exists: the naive round() minimum
    # image overestimates distances somewhere in this cell
    grid = Grid(skewed_cell.lattice, (9, 10, 11))
    x = grid.cart_coords().reshape(-1, 3)
    inv = np.linalg.inv(skewed_cell.lattice)
    naive = np.full(x.shape[0], np.inf)
    for R in skewed_cell.cart_coords:
        f = (x - R) @ inv
        f -= np.round(f)
        naive = np.minimum(naive, np.linalg.norm(f @ skewed_cell.lattice, axis=1))
    exact = nearest_atom(grid, skewed_cell).distance.ravel()
    assert np.max(naive - exact) > 0.1


def test_direction_is_unit_and_points_away(triclinic_two_atoms):
    grid = Grid(triclinic_two_atoms.lattice, (8, 8, 8))
    geo = nearest_atom(grid, triclinic_two_atoms)
    norms = np.linalg.norm(geo.direction, axis=-1)
    assert np.allclose(norms[geo.distance > 1e-9], 1.0)
    # voxel = nucleus image + r * r_hat
    x = grid.cart_coords()
    R = triclinic_two_atoms.cart_coords[geo.atom_index]
    back = x - geo.distance[..., None] * geo.direction - R
    frac = back @ np.linalg.inv(triclinic_two_atoms.lattice)
    np.testing.assert_allclose(frac, np.round(frac), atol=1e-9)


def test_voxel_on_nucleus_has_zero_direction(cubic_one_atom):
    geo = nearest_atom(Grid(cubic_one_atom.lattice, (4, 4, 4)), cubic_one_atom)
    assert geo.distance[0, 0, 0] == pytest.approx(0.0)
    assert np.all(geo.direction[0, 0, 0] == 0.0)


def test_partition_site_sums(triclinic_two_atoms):
    eng = Engine(triclinic_two_atoms, {"rho": np.ones((6, 7, 8))})
    part = eng.partition()
    assert part.site_count().sum() == 6 * 7 * 8
    np.testing.assert_allclose(part.site_sum(eng["rho"].values), part.site_count())


def test_shells_partition_every_voxel(triclinic_two_atoms):
    eng = Engine(triclinic_two_atoms, {"rho": np.ones((10, 10, 10))})
    m = eng.shell_masks()
    total = m.core.astype(int) + m.bond + m.interstitial
    assert np.all(total == 1)
    r = eng.geometry().distance
    assert np.all(r[m.core] <= 0.8) and np.all(r[m.interstitial] > 1.5)


def test_scaled_shells(triclinic_two_atoms):
    eng = Engine(triclinic_two_atoms, {"rho": np.ones((10, 10, 10))})
    shells = Shells.scaled(0.5, 1.0, {"Si": 1.1, "O": 0.66})
    m = eng.shell_masks(shells=shells)
    geo = eng.geometry()
    c2 = np.where(geo.atom_index == 0, 1.1, 0.66)
    np.testing.assert_array_equal(m.interstitial, geo.distance > c2)
    with pytest.raises(KeyError):
        Shells.scaled(0.5, 1.0, {"Si": 1.1}).atom_cutoffs(triclinic_two_atoms)
    with pytest.raises(ValueError):
        Shells(c1=1.5, c2=0.8)


def test_geometry_cached_per_shape(triclinic_two_atoms):
    eng = Engine(triclinic_two_atoms, {"rho": np.ones((6, 6, 6)), "elf": np.ones((3, 3, 3))})
    assert eng.geometry() is eng.geometry()
    assert eng.geometry((3, 3, 3)).shape == (3, 3, 3)
    assert eng["elf"].grid is eng.grid((3, 3, 3))
