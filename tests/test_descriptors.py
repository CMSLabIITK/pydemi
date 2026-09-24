import numpy as np
import pytest

from pydemi import Engine, Grid, Structure
from pydemi.descriptors import (DESCRIPTOR, REGISTRY, VARIANT, composition_features,
                                compute_descriptors, crystal_system, names, tier1)
from pydemi.descriptors.composition import element_property
from pydemi.testing import (GaussianSuperposition, SlaterSuperposition,
                            gaussian_fraction_within, slater_fraction_within,
                            slater_moment)


def _engine(density, structure, n, method="fd"):
    rho, _, _ = density.on_grid(Grid(structure.lattice, (n, n, n)), derivatives=False)
    return Engine(structure, {"rho": rho}, method=method)


# ---------------------------------------------------------------- Tier 1

def test_tier1_against_slater_closed_forms():
    lat = np.array([[12.0, 0, 0], [2.0, 11.0, 0], [1.0, 1.5, 12.0]])
    s = Structure(lat, ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    zetas, N = np.array([1.8, 2.6]), np.array([3.0, 5.0])
    t = tier1(_engine(SlaterSuperposition(s, zetas, N, tol=1e-10), s, 110))
    w = N / N.sum()
    m1 = sum(w[i] * slater_moment(1, zetas[i]) for i in range(2))
    m2 = sum(w[i] * slater_moment(2, zetas[i]) for i in range(2))
    assert t["m1"] == pytest.approx(m1, rel=1e-2)
    assert t["m2"] == pytest.approx(m2, rel=2e-2)
    assert t["sigma_r2"] == pytest.approx(m2 - m1 ** 2, rel=5e-2)
    assert t["f_core"] == pytest.approx(w @ slater_fraction_within(0.8, zetas), abs=5e-3)
    f_int = w @ (1 - slater_fraction_within(1.5, zetas))
    assert t["f_int"] == pytest.approx(f_int, abs=5e-3)
    assert t["f_core"] + t["f_bond"] + t["f_int"] == pytest.approx(1.0, abs=1e-12)
    assert t["Q_tot"] == pytest.approx(8.0, rel=2e-2)


def test_zeta_vanishes_for_isolated_spherical_atom():
    s = Structure(np.eye(3) * 9.0, ["Ar"], [[0.5, 0.5, 0.5]])
    t = tier1(_engine(GaussianSuperposition(s, 2.0, 1.0), s, 72, "spectral"))
    assert abs(t["zeta"]) < 1e-3


def test_zeta_positive_for_overlapping_atoms():
    s = Structure(np.eye(3) * 6.0, ["Si", "Si"], [[0.4, 0.5, 0.5], [0.6, 0.5, 0.5]])
    t = tier1(_engine(GaussianSuperposition(s, 1.5, 4.0), s, 48))
    assert t["zeta"] > 0.02


def test_lnf_closed_form_for_gaussian():
    # lap < 0 exactly inside r0 = sqrt(3 / (2 alpha))
    alpha, L = 2.0, 7.0
    s = Structure(np.eye(3) * L, ["X"], [[0.5, 0.5, 0.5]])
    dens = GaussianSuperposition(s, alpha, 1.0, tol=1e-30)
    r0 = np.sqrt(1.5 / alpha)
    t_fd = tier1(_engine(dens, s, 80, "fd"))
    t_sp = tier1(_engine(dens, s, 80, "spectral"))
    assert t_fd["lnf"] == pytest.approx(4 / 3 * np.pi * r0 ** 3 / L ** 3, rel=2e-2)
    for t in (t_fd, t_sp):
        assert t["lnf_rho"] == pytest.approx(gaussian_fraction_within(r0, alpha), rel=2e-2)
    # the spectral Laplacian has an absolute round-off floor, so its sign in
    # near-empty voxels is noise: voxel-count lnf is badly off, lnf_rho is not
    assert t_sp["lnf"] > 5 * t_fd["lnf"]


@pytest.mark.parametrize("method", ["fd", "spectral"])
def test_lap_concentration_is_identically_half(method):
    # documents why the specified entry 14 carries no information: it is
    # 1/2 for any periodic field, even noise
    rng = np.random.default_rng(0)
    s = Structure(np.array([[4.0, 0, 0], [1.0, 5.0, 0], [0.5, 0.7, 6.0]]),
                  ["X", "Y"], [[0.1, 0.2, 0.3], [0.6, 0.5, 0.4]])
    eng = Engine(s, {"rho": rng.random((16, 18, 20))}, method=method)
    assert tier1(eng)["lap_concentration"] == pytest.approx(0.5, abs=1e-12)
    # for a physical density the core-excluded share is informative
    atoms = _engine(GaussianSuperposition(s, [1.5, 2.5], [4.0, 6.0]), s, 40, method)
    t = tier1(atoms)
    assert t["lap_concentration"] == pytest.approx(0.5, abs=1e-12)
    assert abs(t["lap_concentration_valence"] - 0.5) > 0.1


def test_scale_behaviour_of_ratio_variants():
    """Scale the whole system by s: lengths x s, densities / s^3.

    moment_ratio (m2/m1) scales like a length, moment_ratio_scale_free is
    invariant; zeta_over_rvar ~ 1/s^2 while zeta_over_sigma_r ~ 1/s.
    """
    out = {}
    for scale in (1.0, 1.5):
        s = Structure(np.eye(3) * 6.0 * scale, ["A", "B"],
                      [[0.3, 0.5, 0.5], [0.62, 0.5, 0.5]])
        dens = GaussianSuperposition(s, np.array([1.6, 2.4]) / scale ** 2, [3.0, 5.0])
        out[scale] = tier1(_engine(dens, s, 48, "spectral"))
    a, b = out[1.0], out[1.5]
    assert a["zeta"] == pytest.approx(b["zeta"], rel=1e-6)
    assert a["moment_ratio_scale_free"] == pytest.approx(b["moment_ratio_scale_free"], rel=1e-6)
    assert b["moment_ratio"] / a["moment_ratio"] == pytest.approx(1.5, rel=1e-6)
    assert a["zeta_over_sigma_r"] / b["zeta_over_sigma_r"] == pytest.approx(1.5, rel=1e-6)
    assert a["zeta_over_rvar"] / b["zeta_over_rvar"] == pytest.approx(2.25, rel=1e-6)
    # voxel-count and charge-weighted lnf are both scale-free
    # (tail voxels whose Laplacian sits at round-off level may flip sign)
    assert a["lnf"] == pytest.approx(b["lnf"], abs=1e-3)
    assert a["lnf_rho"] == pytest.approx(b["lnf_rho"], rel=1e-9)


def test_zero_denominator_gives_nan():
    # everything inside the core shell -> f_int = 0 -> bond_int_ratio undefined
    s = Structure(np.eye(3) * 2.0, ["X"], [[0.5, 0.5, 0.5]])
    t = tier1(_engine(GaussianSuperposition(s, 20.0, 1.0), s, 24))
    assert t["f_int"] == 0.0
    assert np.isnan(t["bond_int_ratio"])


# ---------------------------------------------------------------- Tier 2

def test_composition_values():
    c = composition_features({"Fe": 1, "Co": 1, "Ni": 1, "Cr": 1})
    assert c["mean_vec"] == pytest.approx((8 + 9 + 10 + 6) / 4)
    assert c["max_vec"] == 10
    assert c["n_elements"] == 4
    nacl = composition_features({"Na": 1, "Cl": 1})
    assert nacl["elneg_diff"] == pytest.approx(3.16 - 0.93)
    assert nacl["ionicity"] == pytest.approx(1 - np.exp(-(2.23 ** 2) / 4))
    assert nacl["mean_period"] == pytest.approx(3.0)
    # amounts need not be normalized
    assert composition_features({"Na": 4, "Cl": 4}) == nacl


def test_f_block_flags():
    assert composition_features({"La": 1, "Ce": 1})["is_f_block"] == 1
    lao = composition_features({"La": 1, "O": 1})
    assert (lao["is_f_block"], lao["has_f_block"]) == (0, 1)
    assert composition_features({"Fe": 1})["has_f_block"] == 0


def test_missing_values_imputed_like_matminer():
    # He has no Pauling electronegativity in Magpie
    assert np.isnan(element_property("He", "Electronegativity", impute_nan=False))
    imputed = element_property("He", "Electronegativity")
    assert 1.0 < imputed < 2.5


def test_matches_matminer_when_available():
    mm = pytest.importorskip("matminer.featurizers.composition")
    from pymatgen.core import Composition
    ep = mm.ElementProperty("magpie", ["AtomicWeight", "NValence", "Row"],
                            ["mean", "maximum", "range"])
    vals = dict(zip(ep.feature_labels(), ep.featurize(Composition("Fe2O3"))))
    ours = composition_features({"Fe": 2, "O": 3})
    assert ours["mean_mass"] == pytest.approx(vals["MagpieData mean AtomicWeight"])
    assert ours["max_vec"] == pytest.approx(vals["MagpieData maximum NValence"])
    assert ours["mean_period"] == pytest.approx(vals["MagpieData mean Row"])


@pytest.mark.parametrize("lattice,species,frac,expected", [
    (np.eye(3) * 5.64, ["Na"] * 4 + ["Cl"] * 4,
     [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
      [.5, .5, .5], [.5, 0, 0], [0, .5, 0], [0, 0, .5]], (7, 225)),
    (np.array([[3.0, 0, 0], [-1.5, 2.598076211, 0], [0, 0, 4.0]]), ["Mg"],
     [[0, 0, 0]], (6, 191)),
    (np.array([[4.0, 0, 0], [0.7, 4.3, 0], [0.4, 0.9, 5.1]]), ["Si", "O"],
     [[0.0, 0.0, 0.0], [0.31, 0.17, 0.43]], (1, 1)),
])
def test_crystal_system(lattice, species, frac, expected):
    assert crystal_system(Structure(lattice, species, frac)) == expected


# ---------------------------------------------------------------- assembly

@pytest.fixture(scope="module")
def full_result():
    s = Structure(np.eye(3) * 5.0, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    eng = _engine(GaussianSuperposition(s, [1.2, 0.8], [1.0, 7.0]), s, 32)
    return eng, compute_descriptors(eng, families=TIERS)


TIERS = ("tier1", "tier2", "tier3")


def _tier_names(**kw):
    return [n for n in names(**kw) if REGISTRY[n].family in TIERS]


def test_every_registered_quantity_is_computed(full_result):
    _, d = full_result
    assert list(d) == _tier_names()     # all of them, in reference order
    assert all(np.isfinite(v) for v in d.values())


def test_kind_and_family_filters(full_result):
    eng, _ = full_result
    only = compute_descriptors(eng, kinds=(DESCRIPTOR,))
    assert all(REGISTRY[k].kind == DESCRIPTOR for k in only)
    assert "lnf_rho" not in only and "sqrt_zeta" not in only
    t2 = compute_descriptors(eng, families=("tier2",))
    assert all(REGISTRY[k].family == "tier2" for k in t2)
    with pytest.raises(ValueError):
        compute_descriptors(eng, families=("tier9",))


def test_honest_descriptor_count():
    # entries 1-28 and 32 are physical descriptors: 15 + 13 + 1 = 29;
    # 29-31 are metadata, 33-37 cross terms, 38-39 preprocessing
    physical = _tier_names(kinds=(DESCRIPTOR,))
    assert len(physical) == 29
    assert {REGISTRY[n].entry for n in physical} == set(range(1, 29)) | {32}
    assert all(REGISTRY[n].entry is not None for n in names(kinds=(VARIANT,)))


def test_tier1_on_all_electron_field():
    # the AECCAR route: same machinery, different field
    s = Structure(np.eye(3) * 6.0, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    rho, _, _ = GaussianSuperposition(s, [1.2, 0.8], [1.0, 7.0]).on_grid(
        Grid(s.lattice, (24, 24, 24)), derivatives=False)
    eng = Engine(s, {"rho": rho, "rho_ae": 2.0 * rho})
    ae = compute_descriptors(eng, families=("tier1",), field="rho_ae")
    ps = compute_descriptors(eng, families=("tier1",))
    assert ae["Q_tot"] == pytest.approx(2 * ps["Q_tot"])
    assert ae["m1"] == pytest.approx(ps["m1"])     # moments are normalized
