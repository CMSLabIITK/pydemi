"""Milestone 3: geometry pass, shells and image machinery (spec §5, §16)."""

import itertools

import numpy as np
import pytest

from pydemi.core.geometry import (Shells, assign_atoms, geometry_of, image_blocks, nearest_atom,
                                  shell_masks, shells_of)
from pydemi.core.grid import cart_coords
from pydemi.data import covalent_radius
from pydemi.io.base import Grid, Lattice, Structure, VolumetricData

# strongly sheared, low-symmetry cell: a fixed 3x3x3 supercell can miss the nearest image
SHEARED = Lattice(np.array([[5.0, 0.0, 0.0], [4.1, 1.9, 0.0], [3.3, 1.2, 1.7]]))
FRAC = [[0.03, 0.11, 0.07], [0.52, 0.47, 0.61], [0.81, 0.23, 0.39]]


def _brute(shape, s, weights=None, reach=5):
    x = cart_coords(shape, s.lattice).reshape(-1, 3)
    shifts = np.array(list(itertools.product(range(-reach, reach + 1), repeat=3)))
    frac = s.frac_coords % 1.0
    pts = ((frac[None] + shifts[:, None]).reshape(-1, 3)) @ s.lattice.matrix
    owner = np.tile(np.arange(s.n_atoms), len(shifts))
    d2 = ((x[:, None, :] - pts[None]) ** 2).sum(-1)
    if weights is not None:
        d2 = d2 - (np.asarray(weights) ** 2)[owner][None]
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
    """Why the image range is widened adaptively (spec §16, minimum-image bugs):
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


def test_power_diagram_matches_brute_force():
    s = Structure(SHEARED, ["Fe", "O", "Ni"], FRAC)
    shape = (8, 9, 10)
    radii = np.array([1.3, 0.7, 1.1])
    got = assign_atoms(shape, s, radii)
    d, owner, _ = _brute(shape, s, weights=radii)
    np.testing.assert_array_equal(got.atom_index.ravel(), owner)
    np.testing.assert_allclose(got.distance.ravel(), d, atol=1e-12)


def test_atom_on_a_grid_point_has_zero_direction():
    s = Structure(Lattice(np.eye(3) * 4.0), ["H"], [[0.0, 0.0, 0.0]])
    geo = nearest_atom((8, 8, 8), s)
    assert geo.distance[0, 0, 0] == 0.0
    np.testing.assert_array_equal(geo.direction[0, 0, 0], 0.0)


def test_shells_partition_the_cell_and_scale_with_radii():
    s = Structure(SHEARED, ["Fe", "O", "Ni"], FRAC)
    geo = nearest_atom((10, 10, 10), s)
    m = shell_masks(geo, s, Shells())
    total = m.core.astype(int) + m.bond.astype(int) + m.interstitial.astype(int)
    np.testing.assert_array_equal(total, 1)
    np.testing.assert_array_equal(m.core, geo.distance <= 0.8)
    scaled = shell_masks(geo, s, Shells(0.5, 1.2, scaled=True))
    c1 = 0.5 * np.array([covalent_radius(e) for e in s.species])[geo.atom_index]
    np.testing.assert_array_equal(scaled.core, geo.distance <= c1)
    with pytest.raises(ValueError):
        Shells(1.5, 0.8)


def test_geometry_is_computed_once_per_structure():
    s = Structure(SHEARED, ["Fe", "O", "Ni"], FRAC)
    vd = VolumetricData(s, Grid(np.ones((6, 6, 6)), s.lattice))
    assert geometry_of(vd) is geometry_of(vd)
    assert shells_of(vd, Shells()) is shells_of(vd, Shells())
    assert geometry_of(vd.with_options("other")) is geometry_of(vd)      # shared cache


def test_image_blocks_cover_every_image_within_the_cutoff():
    s = Structure(SHEARED, ["Fe", "O", "Ni"], FRAC)
    shape, cutoff = (7, 8, 9), 4.0
    x = cart_coords(shape, s.lattice).reshape(-1, 3)
    shifts = np.array(list(itertools.product(range(-6, 7), repeat=3)))
    pts = ((s.frac_coords % 1.0)[None] + shifts[:, None]).reshape(-1, 3) @ s.lattice.matrix
    owner = np.tile(np.arange(3), len(shifts))
    d = np.linalg.norm(x[:, None] - pts[None], axis=-1)
    f = np.where(d < cutoff, np.exp(-d), 0.0)
    expect = np.stack([f[:, owner == a].sum(1) for a in range(3)], axis=1)
    got = np.zeros_like(expect)
    for b in image_blocks(shape, s, cutoff, block=4):
        w = np.where(b.distance < cutoff, np.exp(-b.distance), 0.0)
        for a in range(3):
            got[b.voxel, a] += w[:, b.owner == a].sum(1)
    np.testing.assert_allclose(got, expect, rtol=1e-12)
