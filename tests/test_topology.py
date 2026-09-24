"""Phase 4: topology (G) and bond-midpoint density (I2)."""

import numpy as np
import pytest

from pydemi import Engine, Grid, Structure
from pydemi.descriptors import compute_descriptors, names, REGISTRY
from pydemi.descriptors.bonds import bond_census, bond_family
from pydemi.descriptors.topology import (OFFSETS, _link_components, _wraps, ascent_basins,
                                         maxima_persistence, morse_census, percolation_levels,
                                         topology_family)
from pydemi.testing import GaussianSuperposition

from conftest import TRICLINIC


def _bits(*offsets):
    rows = [tuple(r) for r in OFFSETS]
    return sum(1 << rows.index(tuple(o)) for o in offsets)


# ---------------------------------------------------------------- link table

def test_link_components():
    comps = _link_components()
    assert comps[0] == 0 and comps[(1 << 14) - 1] == 1
    assert comps[_bits((1, 0, 0))] == 1
    # antipodal link vertices are not adjacent (2 e1 is not an edge)
    assert comps[_bits((1, 0, 0), (-1, 0, 0))] == 2
    # e1 and e1+e2 are adjacent (difference e2 is an edge)
    assert comps[_bits((1, 0, 0), (1, 1, 0))] == 1


# ---------------------------------------------------------------- census

@pytest.mark.parametrize("shape", [(3, 3, 3), (6, 7, 8), (16, 12, 10)])
def test_euler_identity_holds_for_any_field(shape):
    rng = np.random.default_rng(sum(shape))
    for values in (rng.random(shape), np.round(rng.random(shape), 1)):   # with ties too
        c = morse_census(values)
        assert c["euler_consistency"] == 0
        assert c["n_max"] >= 1 and c["n_min"] >= 1


def test_census_of_generic_trigonometric_field():
    # sum of three shifted cosines: 1 max, 1 min, 3 saddles of each index
    grid = Grid(np.eye(3) * 4.0, (24, 26, 28))
    u = grid.frac_coords()
    f = (np.cos(2 * np.pi * (u[..., 0] + 0.013)) + 0.9 * np.cos(2 * np.pi * (u[..., 1] + 0.021))
         + 0.8 * np.cos(2 * np.pi * (u[..., 2] + 0.037)))
    c = morse_census(f)
    assert (c["n_max"], c["n_min"], c["n_saddle1"], c["n_saddle2"]) == (1, 1, 3, 3)


def test_basins_end_at_maxima_and_split_charge():
    s = Structure(np.eye(3) * 8.0, ["X", "X"], [[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
    grid = Grid(s.lattice, (32, 24, 24))
    rho, _, _ = GaussianSuperposition(s, 1.0, 3.0).on_grid(grid, False)
    c = morse_census(rho)
    basins = ascent_basins(rho, s.lattice, c["lower_mask"])
    roots = np.unique(basins)
    assert set(roots) <= set(np.flatnonzero(c["maxima"]))
    assert len(roots) == 2
    q = [rho.ravel()[basins.ravel() == r].sum() * grid.dV for r in roots]
    np.testing.assert_allclose(q, [3.0, 3.0], rtol=2e-2)


def test_non_nuclear_maximum_is_found():
    # an atom at the origin plus an electron blob at the empty body centre
    s = Structure(np.eye(3) * 7.0, ["Na"], [[0.0, 0.0, 0.0]])
    grid = Grid(s.lattice, (28, 28, 28))
    atom, _, _ = GaussianSuperposition(s, 2.0, 1.0).on_grid(grid, False)
    blob_s = Structure(s.lattice, ["X"], [[0.5, 0.5, 0.5]])
    blob, _, _ = GaussianSuperposition(blob_s, 0.8, 0.5).on_grid(grid, False)
    d = topology_family(Engine(s, {"rho": atom + blob}))
    assert d["n_NNM"] == 1
    assert d["Q_NNM"] == pytest.approx(0.5, rel=0.05)
    assert d["euler_consistency"] == 0
    assert d["n_NNM_significant"] == 1
    only_atom = topology_family(Engine(s, {"rho": atom}))
    assert only_atom["n_NNM"] == 0 and only_atom["Q_NNM"] == 0.0


def test_ripple_inflates_raw_nnm_count_but_not_the_robust_ones():
    s = Structure(np.eye(3) * 7.0, ["Na"], [[0.0, 0.0, 0.0]])
    grid = Grid(s.lattice, (28, 28, 28))
    atom, _, _ = GaussianSuperposition(s, 2.0, 1.0).on_grid(grid, False)
    blob, _, _ = GaussianSuperposition(Structure(s.lattice, ["X"], [[0.5, 0.5, 0.5]]),
                                       0.8, 0.5).on_grid(grid, False)
    noisy = atom + blob + 1e-5 * np.random.default_rng(0).random(grid.shape)
    d = topology_family(Engine(s, {"rho": noisy}))
    assert d["n_NNM"] > 50                        # ripple maxima everywhere
    assert d["n_NNM_significant"] == 1
    assert d["Q_NNM"] == pytest.approx(0.5, rel=0.05)
    assert d["euler_consistency"] == 0


def test_pseudized_shell_lobes_are_not_non_nuclear():
    # a PAW-like atom: no maximum at the nucleus, a shell of charge 0.85 A out
    s = Structure(np.eye(3) * 6.0, ["Si"], [[0.5, 0.5, 0.5]])
    grid = Grid(s.lattice, (40, 40, 40))
    x = grid.cart_coords() - 3.0
    r = np.linalg.norm(x, axis=-1)
    lobes = 1 + 0.3 * np.cos(4 * np.arctan2(x[..., 1], x[..., 0]))   # break the sphere
    # (no constant background: a perfectly flat region would produce a
    # tie-broken maximum of its own)
    rho = np.exp(-((r - 0.85) / 0.25) ** 2) * lobes
    eng = Engine(s, {"rho": rho})
    eng.paw_radii = {"Si": 0.60}                    # too small: lobes counted
    assert topology_family(eng)["n_NNM"] > 0
    eng.paw_radii = {"Si": 1.005}                   # Si RCORE = 1.9 bohr
    d = topology_family(eng)
    assert d["n_NNM"] == 0 and d["paw_radii_known"] == 1
    eng.paw_radii = None                            # falls back to the covalent radius (1.11 A)
    d = topology_family(eng)
    assert d["n_NNM"] == 0 and d["paw_radii_known"] == 0


def _brute_force_persistence(values):
    """Voxel-level superlevel-set union-find on the Freudenthal graph (elder rule)."""
    shape = values.shape
    flat = values.ravel()
    parent = -np.ones(flat.size, dtype=int)
    top = np.arange(flat.size)
    merge = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k in np.argsort(-flat, kind="stable"):
        parent[k] = k
        idx = np.array(np.unravel_index(k, shape))
        for off in OFFSETS:
            n = np.ravel_multi_index(tuple((idx + off) % shape), shape)
            if parent[n] < 0:
                continue
            ra, rb = find(k), find(n)
            if ra == rb:
                continue
            ta, tb = top[ra], top[rb]
            young, old = (ta, tb) if flat[ta] < flat[tb] else (tb, ta)
            if young != k:              # k itself joining is not the death of a maximum
                merge[young] = flat[k]
            parent[ra] = rb
            top[rb] = old
    return merge


def test_persistence_matches_voxel_union_find():
    rng = np.random.default_rng(3)
    for shape in ((6, 7, 5), (8, 8, 8)):
        f = rng.random(shape)
        c = morse_census(f)
        basin = ascent_basins(f, np.eye(3) * 5.0, c["lower_mask"])
        peaks, merge, absorber = maxima_persistence(f, basin)
        brute = _brute_force_persistence(f)
        assert set(peaks) == set(np.flatnonzero(c["maxima"]))
        got = {p: m for p, m in zip(peaks, merge) if np.isfinite(m)}
        assert set(got) == set(brute)
        for p, m in got.items():
            assert m == pytest.approx(brute[p])
        assert np.sum(~np.isfinite(merge)) == 1           # only the global maximum survives


def test_persistence_removes_ripple_and_conserves_charge():
    # an atom, a genuine interstitial blob, and ripple everywhere
    s = Structure(np.eye(3) * 7.0, ["Na"], [[0.0, 0.0, 0.0]])
    grid = Grid(s.lattice, (28, 28, 28))
    atom, _, _ = GaussianSuperposition(s, 2.0, 1.0).on_grid(grid, False)
    blob, _, _ = GaussianSuperposition(Structure(s.lattice, ["X"], [[0.5, 0.5, 0.5]]),
                                       0.8, 0.5).on_grid(grid, False)
    sea = 0.02 * (1 + 0.005 * np.random.default_rng(0).random(grid.shape))  # flat electron sea
    d = topology_family(Engine(s, {"rho": atom + blob + sea}))
    assert d["n_NNM"] > 50 and d["n_NNM_significant"] > 1   # ripple basins hold > 0.01 e
    assert d["n_NNM_persistent"] == 1
    # the blob's own charge plus the sea it absorbed; never more than the non-nuclear charge
    assert 0.5 < d["Q_NNM_persistent"] <= d["Q_NNM"] + 1e-9


def test_persistent_maxima_need_charge():
    # a well-separated but nearly empty interstitial bump is persistent yet
    # holds far below 0.01 e, so it is not counted; a real blob is
    s = Structure(np.eye(3) * 7.0, ["Na"], [[0.0, 0.0, 0.0]])
    grid = Grid(s.lattice, (28, 28, 28))
    atom, _, _ = GaussianSuperposition(s, 2.0, 1.0).on_grid(grid, False)
    site = Structure(s.lattice, ["X"], [[0.5, 0.5, 0.5]])
    for q, expected in ((1e-4, 0), (0.5, 1)):
        blob, _, _ = GaussianSuperposition(site, 0.8, q).on_grid(grid, False)
        d = topology_family(Engine(s, {"rho": atom + blob}))
        assert d["n_NNM"] == 1
        assert d["n_NNM_persistent"] == expected


def test_interstitial_floor_ignores_paw_core_dips():
    s = Structure(np.eye(3) * 6.0, ["Si"], [[0.5, 0.5, 0.5]])
    grid = Grid(s.lattice, (40, 40, 40))
    r = np.linalg.norm(grid.cart_coords() - 3.0, axis=-1)
    rho = 0.05 + np.exp(-((r - 0.9) / 0.3) ** 2) - 0.2 * np.exp(-(r / 0.2) ** 2)   # negative at the nucleus
    eng = Engine(s, {"rho": rho})
    eng.paw_radii = {"Si": 1.005}
    d = topology_family(eng)
    assert d["rho_min"] < 0 and d["rho_min_ratio"] < 0
    assert d["rho_min_int"] == pytest.approx(rho[r > 1.5].min())
    assert d["rho_min_int_ratio"] > 0
    # 1.6 A cube: no voxel is farther than 1.39 A from the atom
    tiny = Engine(Structure(np.eye(3) * 1.6, ["Si"], [[0, 0, 0]]), {"rho": np.ones((8, 8, 8))})
    assert np.isnan(topology_family(tiny)["rho_min_int"])      # no voxel beyond max(c2, R)


# ---------------------------------------------------------------- percolation

def test_wraps_detects_windings():
    m = np.zeros((6, 6, 4), bool)
    m[:, 2, 1] = True                          # straight tube along a
    assert list(_wraps(m)) == [True, False, False]
    m = np.zeros((6, 6, 4), bool)
    for i in range(6):                         # face-connected staircase along [1 1 0]
        m[i, i, 2] = m[(i + 1) % 6, i, 2] = True
    assert list(_wraps(m)) == [True, True, False]
    m = np.zeros((6, 6, 4), bool)
    m[4:, 1, 1] = m[:2, 1, 1] = True           # a finite rod crossing the boundary
    assert list(_wraps(m)) == [False, False, False]
    assert list(_wraps(np.ones((3, 3, 3), bool))) == [True, True, True]
    assert list(_wraps(np.zeros((3, 3, 3), bool))) == [False, False, False]


def test_percolation_of_layered_density():
    # varies along c only: connected in-plane at any level below the max,
    # across layers only below the min
    grid = Grid(TRICLINIC, (10, 12, 16))
    u = grid.frac_coords()
    rho = 1.0 + np.cos(2 * np.pi * u[..., 2])
    levels = percolation_levels(rho)
    assert levels[0] == pytest.approx(rho.max()) and levels[1] == pytest.approx(rho.max())
    assert levels[2] == pytest.approx(rho.min())


def test_percolation_level_is_the_bottleneck():
    # a tube along a whose density dips to 0.3 at one point
    rho = np.zeros((12, 5, 5))
    rho[:, 2, 2] = 1.0
    rho[7, 2, 2] = 0.3
    levels = percolation_levels(rho)
    assert levels[0] == pytest.approx(0.3)
    assert levels[1] == pytest.approx(0.0) and levels[2] == pytest.approx(0.0)


def test_metal_like_percolates_high_ionic_like_low():
    s = Structure(np.eye(3) * 4.0, ["X"], [[0, 0, 0]])
    grid = Grid(s.lattice, (20, 20, 20))
    dense, _, _ = GaussianSuperposition(s, 0.3, 4.0).on_grid(grid, False)    # overlapping
    sparse, _, _ = GaussianSuperposition(s, 3.0, 4.0).on_grid(grid, False)   # isolated
    hi = topology_family(Engine(s, {"rho": dense}))
    lo = topology_family(Engine(s, {"rho": sparse}))
    assert hi["rho_perc_a"] / dense.mean() > 10 * lo["rho_perc_a"] / sparse.mean()
    assert hi["rho_min_ratio"] > 10 * lo["rho_min_ratio"]
    assert hi["perc_anisotropy"] == pytest.approx(0.0, abs=1e-12)   # cubic


# ---------------------------------------------------------------- bonds

def test_bond_census_fcc_and_simple_cubic():
    a = 3.6
    fcc = Structure(np.eye(3) * a, ["Cu"] * 4,
                    [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    b = bond_census(fcc)
    assert len(b.length) == 24
    np.testing.assert_allclose(b.length, a / np.sqrt(2))
    sc = Structure(np.eye(3) * 2.9, ["Fe"], [[0, 0, 0]])       # simple cubic, 1 atom
    b = bond_census(sc)
    assert len(b.length) == 3                                   # 6 neighbours / 2
    np.testing.assert_allclose(b.length, 2.9)


def test_bond_census_handles_unwrapped_coordinates():
    lat = TRICLINIC
    s1 = Structure(lat, ["Si", "O"], [[0.1, 0.2, 0.3], [0.6, 0.55, 0.7]])
    s2 = Structure(lat, ["Si", "O"], [[1.1, -0.8, 0.3], [0.6, 2.55, -0.3]])
    b1, b2 = bond_census(s1), bond_census(s2)
    np.testing.assert_allclose(np.sort(b1.length), np.sort(b2.length))
    # midpoints agree modulo the lattice
    m1 = np.sort(b1.midpoints_frac(s1) % 1.0, axis=0)
    m2 = np.sort(b2.midpoints_frac(s2) % 1.0, axis=0)
    np.testing.assert_allclose(m1, m2, atol=1e-12)


def test_bond_midpoint_density():
    # two Gaussians 2 A apart in a big box: rho(mid) is known in closed form
    L, alpha, N = 12.0, 1.2, 2.0
    s = Structure(np.eye(3) * L, ["H", "H"], [[0.5 - 1 / L, 0.5, 0.5], [0.5 + 1 / L, 0.5, 0.5]])
    rho, _, _ = GaussianSuperposition(s, alpha, N).on_grid(Grid(s.lattice, (60, 60, 60)), False)
    d = bond_family(Engine(s, {"rho": rho}))
    expected = 2 * N * (alpha / np.pi) ** 1.5 * np.exp(-alpha * 1.0)
    assert d["n_bonds"] == 1
    assert d["bond_length_mean"] == pytest.approx(2.0)
    assert d["rho_mid_mean"] == pytest.approx(expected, rel=1e-6)
    assert d["rho_mid_std"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- assembly

def test_g_and_i2_registered_and_computed():
    s = Structure(np.eye(3) * 5.0, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    rho, _, _ = GaussianSuperposition(s, [1.2, 0.8], [1.0, 7.0]).on_grid(
        Grid(s.lattice, (24, 24, 24)), False)
    d = compute_descriptors(Engine(s, {"rho": rho}), families=("G", "I2"))
    assert set(d) == set(names(family="G")) | set(names(family="I2"))
    assert d["euler_consistency"] == 0
    assert REGISTRY["euler_consistency"].kind == "metadata"
