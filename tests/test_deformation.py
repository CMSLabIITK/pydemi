"""Milestone 9: promolecule and deformation-density descriptors (spec §6.1, §8.1, §11)."""

import numpy as np
import pytest

import pydemi
from pydemi.core.geometry import bond_census, pair_regions
from pydemi.data import atomic_number, default_zval
from pydemi.descriptors.bonding import delta_rho, pair_charge_transfer, promolecule_density
from pydemi.descriptors.registry import FeatureOptions
from pydemi.fields.deformation import (promolecule, radial_table, reference_tables,
                                       tabulated_radial)
from pydemi.io.base import Grid, Lattice, Structure
from pydemi.validate.analytic import SlaterSuperposition

DEF_NAMES = ["m1_def", "m2_def", "sigma_r2_def", "f_bond_def", "f_int_def", "f_bond_dep",
             "def_polarity", "bond_charge_transfer_pair_mean", "bond_charge_transfer_pair_std"]


def _electrons(r, rho):
    return np.trapezoid(4 * np.pi * r * r * rho, r)


@pytest.mark.parametrize("element", ["H", "O", "Si", "Fe", "Ni", "Zn", "Yb", "U"])
def test_tables_hold_the_right_electron_counts(element):
    r, rho = tabulated_radial(element, "total")
    assert _electrons(r, rho) == pytest.approx(atomic_number(element), rel=2e-4)
    r, rho = tabulated_radial(element, "valence")
    assert _electrons(r, rho) == pytest.approx(default_zval(element), rel=2e-4)
    r, rho = tabulated_radial(element, "valence", zval=1.0)
    assert _electrons(r, rho) == pytest.approx(1.0, rel=2e-4)


def test_radial_table_cutoff_and_lookup():
    r = np.geomspace(1e-4, 20.0, 800)
    rho = np.exp(-2.0 * r)
    t = radial_table(r, rho, tol=1e-6)
    assert t.r_max == pytest.approx(-np.log(1e-6) / 2.0, rel=0.02)
    d = np.array([0.0, 0.5, 1.0, 3.0, t.r_max + 1.0])
    np.testing.assert_allclose(t(d)[:4], np.exp(-2.0 * d[:4]), rtol=1e-3)
    assert t(d)[4] == 0.0


def test_promolecule_of_one_atom_integrates_to_its_electrons():
    """Real-space point sampling: exact for a nucleus off the grid points; a nucleus ON
    a grid point over-counts the (all-electron-shaped) valence cusp, which converges
    away with the grid (documented; reported per structure as def_charge_mismatch)."""
    def total(frac, n):
        s = Structure(Lattice(np.eye(3) * 9.0), ["O"], [frac])
        p = promolecule((n, n, n), s, reference_tables(s, "tabulated", "valence"))
        return p.sum() * (9.0 ** 3 / n ** 3)
    assert total([0.3037, 0.4071, 0.5013], 90) == pytest.approx(6.0, rel=2e-3)
    on_grid = [abs(total([0.3, 0.4, 0.5], n) - 6.0) for n in (60, 90, 150)]
    assert on_grid[0] > on_grid[1] > on_grid[2] and on_grid[2] < 0.012


def _custom_files(tmp_path, zetas, scale=1.0):
    r = np.geomspace(1e-5, 30.0, 4000)
    for el, z in zetas.items():
        np.savetxt(tmp_path / f"{el}.dat", np.column_stack([r, scale * z ** 3 / np.pi *
                                                             np.exp(-2 * z * r)]))
    return str(tmp_path)


def _two_slater():
    s = Structure(Lattice(np.eye(3) * 7.0), ["H", "He"], [[0.4, 0.5, 0.5], [0.62, 0.5, 0.5]])
    return SlaterSuperposition(s, {"H": 1.6, "He": 2.0}, tol=1e-30).volumetric((56, 56, 56))


def test_custom_reference_equal_to_the_crystal_atoms_gives_zero(tmp_path):
    """Superposition of the reference atoms -> delta_rho ~ 0 (analytically known integral 0)."""
    vd = _two_slater()
    ref = _custom_files(tmp_path, {"H": 1.6, "He": 2.0})
    f, meta = pydemi.featurize(vd, domains=["bonding"], deformation_reference="custom",
                               custom_reference=ref, return_metadata=True)
    assert meta["deformation_reference"] == "custom"
    assert abs(meta["def_charge_mismatch"]) < 1e-3
    assert f["def_polarity"] < 2e-3


def test_custom_reference_with_half_the_charge(tmp_path):
    """Reference atoms holding half the electrons: int delta_rho dV = N/2 = 1."""
    vd = _two_slater()
    ref = _custom_files(tmp_path, {"H": 1.6, "He": 2.0}, scale=0.5)
    _, meta = pydemi.featurize(vd, domains=["bonding"], deformation_reference="custom",
                               custom_reference=ref, return_metadata=True)
    assert meta["def_charge_mismatch"] == pytest.approx(1.0, abs=2e-3)


def test_aeccar0_reference_is_core_plus_tabulated_valence():
    """Documented correction: promolecule = AECCAR0 + sum of free-atom valence densities."""
    s = Structure(Lattice(np.eye(3) * 5.0), ["Si", "Si"], [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    shape = (40, 40, 40)
    core = SlaterSuperposition(s, [8.0, 8.0], [10.0, 10.0], tol=1e-30).on_grid(shape)[0]
    valence = promolecule(shape, s, reference_tables(s, "tabulated", "valence"))
    vd = pydemi.VolumetricData(s, Grid(core + valence, s.lattice), density_source="all_electron",
                               core_density=Grid(core, s.lattice))
    f, meta = pydemi.featurize(vd, domains=["bonding"], return_metadata=True)
    assert meta["deformation_reference"] == "aeccar0"               # 'auto' prefers it
    np.testing.assert_allclose(delta_rho(vd.with_options(FeatureOptions())), 0.0, atol=1e-12)
    assert f["def_polarity"] == pytest.approx(0.0, abs=1e-12)
    tab = pydemi.featurize(vd, domains=["bonding"], deformation_reference="tabulated")
    assert tab["def_polarity"] > 0.01                               # total free atoms differ


def test_aeccar0_needs_the_core_density():
    with pytest.raises(ValueError, match="AECCAR0"):
        pydemi.featurize(_two_slater(), domains=["bonding"], deformation_reference="aeccar0")


def test_pair_regions_tile_the_cell_and_match_the_census():
    s = Structure(Lattice(np.array([[4.0, 0, 0], [0.6, 4.1, 0], [0.3, 0.4, 4.3]])),
                  ["Fe", "O", "O"], [[0.0, 0.0, 0.0], [0.5, 0.45, 0.5], [0.2, 0.7, 0.35]])
    shape = (18, 18, 20)
    i, j, t = pair_regions(shape, s)
    assert i.size == np.prod(shape)
    assert np.all((i < j) | ((i == j) & np.any(t != 0, axis=1)))
    census = bond_census(s, 0.1)
    regions = {(a, b, tuple(c)) for a, b, c in zip(i, j, t)}
    for a, b, c in zip(census.i, census.j, census.shift_wrapped):
        assert (a, b, tuple(c)) in regions                  # every bond has a region
    vd = SlaterSuperposition(s, [1.4, 2.0, 2.0], [8.0, 6.0, 6.0], tol=1e-20).volumetric(shape)
    v = vd.with_options(FeatureOptions())
    q = pair_charge_transfer(v)
    assert q.shape == census.length.shape and np.all(np.isfinite(q))


def test_deformation_names_and_no_nan():
    f, meta = pydemi.featurize(_two_slater(), domains=["bonding"], return_metadata=True)
    for name in DEF_NAMES:
        assert name in f and np.isfinite(f[name])
    assert meta["deformation_reference"] == "tabulated"
