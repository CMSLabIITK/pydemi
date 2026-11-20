"""Geometry pass, shells and image machinery."""

import itertools

import numpy as np
import pytest

from pydemi.core.geometry import (Shells, geometry_of, nearest_atom, shell_masks,
                                  shells_of, sphere_assignment)
from pydemi.core.grid import cart_coords
from pydemi.data import covalent_radius
from pydemi.io.base import Grid, Lattice, Structure, VolumetricData

# strongly sheared, low-symmetry cell: a fixed 3x3x3 supercell can miss the nearest image
SHEARED = Lattice(np.array([[5.0, 0.0, 0.0], [4.1, 1.9, 0.0], [3.3, 1.2, 1.7]]))
FRAC = [[0.03, 0.11, 0.07], [0.52, 0.47, 0.61], [0.81, 0.23, 0.39]]


def _brute(shape, s, reach=5):
    x = cart_coords(shape, s.lattice).reshape(-1, 3)
    shifts = np.array(list(itertools.product(range(-reach, reach + 1), repeat=3)))
    frac = s.frac_coords % 1.0
    pts = ((frac[None] + shifts[:, None]).reshape(-1, 3)) @ s.lattice.matrix
    owner = np.tile(np.arange(s.n_atoms), len(shifts))
    d2 = ((x[:, None, :] - pts[None]) ** 2).sum(-1)
    k = d2.argmin(1)
    return np.linalg.norm(x - pts[k], axis=1), owner[k], x - pts[k]


def test_nearest_atom_matches_brute_force_in_a_sheared_cell():
    s = Structure(SHEARED, ["Fe", "O", "Ni"], FRAC)
    shape = (9, 10, 11)
    geo = nearest_atom(shape, s)
    d, owner, diff = _brute(shape, s)
    np.testing.assert_allclose(geo.distance.ravel(), d, atol=1e-12)
    np.testing.assert_array_equal(geo.atom_index.ravel(), owner)
    u = geo.direction.reshape(-1, 3)
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1.0)
    np.testing.assert_allclose(u * d[:, None], diff, atol=1e-12)


def test_extreme_shear_where_a_3x3x3_supercell_fails():
    """Why the image range is widened adaptively:
    in this cell the nearest image of some voxels lies outside the 3x3x3 supercell."""
    lat = Lattice(np.array([[6.0, 0.0, 0.0], [5.7, 0.8, 0.0], [5.5, 0.6, 0.7]]))
    s = Structure(lat, ["Fe", "O", "Ni"], FRAC)
    shape = (9, 10, 11)
    d_true, owner, _ = _brute(shape, s)
    d_fixed, _, _ = _brute(shape, s, reach=1)
    assert (d_fixed - d_true).max() > 0.01
    geo = nearest_atom(shape, s)
    np.testing.assert_allclose(geo.distance.ravel(), d_true, atol=1e-12)
    np.testing.assert_array_equal(geo.atom_index.ravel(), owner)


def test_atom_on_a_grid_point_has_zero_direction():
    s = Structure(Lattice(np.eye(3) * 4.0), ["H"], [[0.0, 0.0, 0.0]])
    geo = nearest_atom((8, 8, 8), s)
    assert geo.distance[0, 0, 0] == 0.0
    np.testing.assert_array_equal(geo.direction[0, 0, 0], 0.0)


def test_regions_partition_the_cell_and_scale_with_radii():
    s = Structure(SHEARED, ["Fe", "O", "Ni"], FRAC)
    geo = sphere_assignment((10, 10, 10), s)
    m = shell_masks(geo, s, Shells())
    total = m.core.astype(int) + m.bond.astype(int) + m.interstitial.astype(int)
    np.testing.assert_array_equal(total, 1)
    R = np.array([covalent_radius(e) for e in s.species])[geo.atom_index]
    np.testing.assert_array_equal(m.core, geo.distance <= 0.79 * R + 1e-9)
    small = shell_masks(geo, s, Shells(0.5, 1.2))
    np.testing.assert_array_equal(small.core, geo.distance <= 0.5 * R + 1e-9)
    with pytest.raises(ValueError):
        Shells(1.14, 0.79)


def test_geometry_is_computed_once_per_structure():
    s = Structure(SHEARED, ["Fe", "O", "Ni"], FRAC)
    vd = VolumetricData(s, Grid(np.ones((6, 6, 6)), s.lattice))
    assert geometry_of(vd) is geometry_of(vd)
    assert shells_of(vd, Shells()) is shells_of(vd, Shells())
    assert geometry_of(vd.with_options("other")) is geometry_of(vd)      # shared cache


def _brute_spheres(shape, s, reach=5):
    """s = min over all images of r_i / R_i, and the atom that attains it."""
    x = cart_coords(shape, s.lattice).reshape(-1, 3)
    shifts = np.array(list(itertools.product(range(-reach, reach + 1), repeat=3)))
    pts = ((s.frac_coords % 1.0)[None] + shifts[:, None]).reshape(-1, 3) @ s.lattice.matrix
    owner = np.tile(np.arange(s.n_atoms), len(shifts))
    R = np.array([covalent_radius(e) for e in s.species])[owner]
    ratio = np.linalg.norm(x[:, None, :] - pts[None], axis=-1) / R[None]
    k = ratio.argmin(1)
    return ratio[np.arange(len(x)), k], owner[k]


def test_sphere_rule_matches_the_spheres_about_every_nucleus():
    """Core inside any core sphere a R_i, interstitial outside every outer sphere b R_i."""
    s = Structure(SHEARED, ["Cs", "O", "Ni"], FRAC)
    shape = (9, 10, 11)
    a, b = 0.3, 0.6          # this dense cell has no interstitial at 0.79 / 1.14
    geo = sphere_assignment(shape, s)
    ratio, owner = _brute_spheres(shape, s)
    np.testing.assert_array_equal(geo.atom_index.ravel(), owner)
    R = np.array([covalent_radius(e) for e in s.species])
    np.testing.assert_allclose(geo.distance.ravel() / R[owner], ratio, atol=1e-12)
    m = shell_masks(geo, s, Shells(a, b))
    np.testing.assert_array_equal(m.core.ravel(), ratio <= a)
    np.testing.assert_array_equal(m.interstitial.ravel(), ratio > b)
    assert m.core.any() and m.bond.any() and m.interstitial.any()
    # some voxels go to a farther but larger atom than the nearest one
    assert (geo.atom_index != nearest_atom(shape, s).atom_index).any()


def test_sphere_rule_does_not_depend_on_atom_order():
    s = Structure(SHEARED, ["Cs", "O", "Ni"], FRAC)
    perm = [2, 0, 1]
    t = Structure(SHEARED, [s.species[i] for i in perm], [FRAC[i] for i in perm])
    sh = Shells()
    m1 = shell_masks(sphere_assignment((8, 9, 10), s), s, sh)
    m2 = shell_masks(sphere_assignment((8, 9, 10), t), t, sh)
    np.testing.assert_array_equal(m1.core, m2.core)
    np.testing.assert_array_equal(m1.interstitial, m2.interstitial)


def test_sphere_rule_with_one_element_is_the_nearest_atom_rule():
    s = Structure(SHEARED, ["Si", "Si", "Si"], FRAC)
    vd = VolumetricData(s, Grid(np.ones((8, 9, 10)), s.lattice))
    m = shells_of(vd, Shells())
    R = covalent_radius("Si")
    r = nearest_atom(vd.shape, s).distance
    np.testing.assert_array_equal(m.core, r <= 0.79 * R + 1e-9)
    np.testing.assert_array_equal(m.interstitial, r > 1.14 * R + 1e-9)
