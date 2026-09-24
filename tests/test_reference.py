"""Phase 5: free-atom solver, reference densities, deformation density (A),
Hirshfeld partition (101) and ELF_D fidelity (73)."""

import numpy as np
import pytest

from pydemi import Engine, Grid, Structure
from pydemi.atoms.reference import (AtomicLDAReference, IsolatedAtomReference,
                                    default_zval, read_potcar_zval)
from pydemi.atoms.solver import RadialGrid, _solve_l, solve_atom
from pydemi.descriptors import compute_descriptors, names
from pydemi.descriptors.deformation import deformation_family, form_factor, promolecule
from pydemi.descriptors.elf import elf_fidelity, resample
from pydemi.descriptors.sites import hirshfeld_charges, site_charges
from pydemi.io.vasp import write_volumetric
from pydemi.testing import GaussianSuperposition


class GaussianReference:
    """Analytic free atoms: N (a/pi)^{3/2} exp(-a r^2), same for 'total' and 'valence'."""

    pseudized = True

    def __init__(self, alpha, electrons):
        self.alpha, self.N = dict(alpha), dict(electrons)

    def radial(self, element, part="total", zval=None):
        r = np.geomspace(1e-5, 12.0, 3000)
        a = self.alpha[element]
        return r, self.N[element] * (a / np.pi) ** 1.5 * np.exp(-a * r * r)

    def electrons(self, element, part="total", zval=None):
        return self.N[element]


NACL = Structure(np.eye(3) * 5.6, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
GREF = GaussianReference({"Na": 1.1, "Cl": 0.7}, {"Na": 1.0, "Cl": 7.0})


def _gauss_rho(structure, n=36):
    alphas = [GREF.alpha[s] for s in structure.species]
    N = [GREF.N[s] for s in structure.species]
    rho, _, _ = GaussianSuperposition(structure, alphas, N).on_grid(
        Grid(structure.lattice, (n, n, n)), False)
    return rho


# ---------------------------------------------------------------- solver

@pytest.mark.parametrize("Z,l", [(1, 0), (1, 1), (26, 0), (26, 2)])
def test_hydrogenic_eigenvalues(Z, l):
    g = RadialGrid()
    w, psi = _solve_l(-Z / g.r, l, g, 2, Z)
    exact = np.array([-Z ** 2 / (2 * (l + 1 + k) ** 2) for k in range(2)])
    np.testing.assert_allclose(w, exact, rtol=2e-5)
    np.testing.assert_allclose(np.sum(psi ** 2, axis=0) * g.h, 1.0, rtol=1e-12)


def test_lda_atoms_match_nist_reference_data():
    # Kotochigova et al., Phys. Rev. A 55, 191 (1997), LDA (VWN), non-relativistic
    he = solve_atom(2, [(1, "s", 2)], correlation="vwn")
    assert he.energy == pytest.approx(-2.834836, abs=2e-5)
    assert he.orbitals[0][3] == pytest.approx(-0.570425, abs=2e-5)
    ne = solve_atom(10, [(1, "s", 2), (2, "s", 2), (2, "p", 6)], correlation="vwn")
    eps = {(o[0], o[1]): o[3] for o in ne.orbitals}
    assert ne.energy == pytest.approx(-128.233481, abs=1e-4)
    assert eps[(1, 0)] == pytest.approx(-30.305855, abs=5e-5)
    assert eps[(2, 1)] == pytest.approx(-0.498034, abs=2e-5)
    assert ne.converged and ne.extra["electrons"] == pytest.approx(10.0, abs=1e-8)


def test_default_zval_matches_standard_potcars():
    expect = {"O": 6, "Fe": 8, "Co": 9, "Ni": 10, "Cr": 6, "Cu": 11, "Hf": 4,
              "Ta": 5, "W": 6, "Au": 11}
    assert {e: default_zval(e) for e in expect} == expect


def test_reference_electron_counts(tmp_path):
    ref = AtomicLDAReference(cache_dir=str(tmp_path))
    for part, zval, expected in (("total", None, 26.0), ("valence", 8, 8.0), ("valence", 14, 14.0)):
        r, n = ref.radial("Fe", part, zval)
        assert np.trapezoid(4 * np.pi * r ** 2 * n, r) == pytest.approx(expected, rel=1e-4)
    with pytest.raises(ValueError):
        ref.radial("Fe", "valence", 40)
    # disk cache round trip
    assert list(tmp_path.glob("atoms/Fe_*.npz"))
    again = AtomicLDAReference(cache_dir=str(tmp_path)).radial("Fe", "valence", 8)
    np.testing.assert_allclose(again[1], ref.radial("Fe", "valence", 8)[1])


def test_read_potcar_zval(tmp_path):
    p = tmp_path / "POTCAR"
    p.write_text("  PAW_PBE Fe_pv 02Aug2007\n   POMASS =   55.847; ZVAL   =   14.000    mass and valenz\n"
                 " End of Dataset\n  PAW_PBE O 08Apr2002\n   POMASS =   16.000; ZVAL   =    6.000\n")
    assert read_potcar_zval(p) == [14.0, 6.0]


# ---------------------------------------------------------------- promolecule / Family A

def test_form_factor_of_gaussian():
    r, n = GREF.radial("Cl")
    G = np.array([0.0, 1.0, 3.0])
    a, N = 0.7, 7.0
    np.testing.assert_allclose(form_factor(r, n, G), N * np.exp(-G ** 2 / (4 * a)), rtol=1e-5)


def test_promolecule_matches_analytic_superposition():
    eng = Engine(NACL, {"rho": _gauss_rho(NACL)}, reference=GREF)
    pro = promolecule(eng)
    # form-factor table interpolation limits agreement to ~1e-5
    np.testing.assert_allclose(pro, eng["rho"].values, rtol=1e-4, atol=1e-6 * pro.max())
    assert pro.sum() * eng.grid().dV == pytest.approx(8.0, rel=1e-10)


def test_deformation_vanishes_for_the_promolecule_itself():
    eng = Engine(NACL, {"rho": _gauss_rho(NACL), "rho_ae": _gauss_rho(NACL)}, reference=GREF)
    d = deformation_family(eng)
    assert d["def_all_electron"] == 1
    assert d["def_polarity"] < 1e-5 and abs(d["def_charge_mismatch"]) < 1e-8
    assert abs(d["bond_charge_transfer"]) < 1e-5


def test_deformation_sees_bond_charge():
    # add a charge-neutral redistribution: +q at the bond midpoint region, -q on the atoms
    # atoms 2.4 A apart, so the midpoint lies in the bonding shell (1.2 A from each)
    s = Structure(np.eye(3) * 6.0, ["Na", "Na"], [[0.3, 0.5, 0.5], [0.7, 0.5, 0.5]])
    ref = GaussianReference({"Na": 1.5}, {"Na": 2.0})
    grid = Grid(s.lattice, (40, 40, 40))
    atoms, _, _ = GaussianSuperposition(s, 1.5, 2.0).on_grid(grid, False)
    mid, _, _ = GaussianSuperposition(Structure(s.lattice, ["X"], [[0.5, 0.5, 0.5]]),
                                      3.0, 0.2).on_grid(grid, False)
    rho = atoms * (1 - 0.1 / 2.0) + mid        # 0.2 e moved into the bond
    eng = Engine(s, {"rho_ae": rho}, reference=ref)
    d = deformation_family(eng)
    assert d["def_charge_mismatch"] == pytest.approx(0.0, abs=1e-8)
    assert d["bond_charge_transfer"] > 0.05
    assert d["f_bond_def"] > 0.5
    assert 0 < d["def_polarity"] < 0.2


def test_chgcar_route_is_opt_in_and_warns_with_all_electron_reference(tmp_path):
    eng = Engine(NACL, {"rho": _gauss_rho(NACL)},
                 reference=AtomicLDAReference(cache_dir=str(tmp_path)))
    d = compute_descriptors(eng, families=("A",))
    assert np.isnan(d["m1_def"]) and d["def_all_electron"] == 0
    with pytest.warns(UserWarning, match="PAW pseudization"):
        deformation_family(eng, field="rho")


def test_mismatch_warning_for_wrong_electron_count():
    eng = Engine(NACL, {"rho_ae": 1.2 * _gauss_rho(NACL)}, reference=GREF)
    with pytest.warns(UserWarning, match="integrates to"):
        d = deformation_family(eng)
    assert d["def_charge_mismatch"] == pytest.approx(1.6, rel=1e-6)


def test_isolated_atom_reference_recovers_radial_profile(tmp_path):
    L = 10.0
    s = Structure(np.eye(3) * L, ["Na"], [[0.5, 0.5, 0.5]])
    rho, _, _ = GaussianSuperposition(s, 1.1, 1.0).on_grid(Grid(s.lattice, (60, 60, 60)), False)
    write_volumetric(tmp_path / "CHGCAR_Na", s, [rho])
    ref = IsolatedAtomReference({"Na": tmp_path / "CHGCAR_Na"}, bins=80)
    r, n = ref.radial("Na", "valence")
    exact = (1.1 / np.pi) ** 1.5 * np.exp(-1.1 * r * r)
    inner = (r > 0.3) & (r < 3.0)
    np.testing.assert_allclose(n[inner], exact[inner], rtol=0.05, atol=1e-4)
    assert ref.pseudized
    with pytest.raises(KeyError):
        ref.radial("Na", "total")


# ---------------------------------------------------------------- Hirshfeld

def test_hirshfeld_is_a_partition_of_unity_and_neutral_for_the_promolecule():
    eng = Engine(NACL, {"rho_ae": _gauss_rho(NACL)}, reference=GREF)
    part = eng.partition(eng["rho_ae"].grid.shape, "hirshfeld")
    # weights below min_weight (1e-10) are dropped
    assert part.site_count().sum() == pytest.approx(36 ** 3, rel=1e-8)
    for p in part.pairs():
        assert np.all((p.weight > 0) & (p.weight <= 1 + 1e-12))
    # rho = promolecule -> every atom holds exactly its free-atom charge
    Q = site_charges(eng, "hirshfeld", "rho_ae")
    np.testing.assert_allclose(Q, [1.0, 7.0], rtol=1e-4)
    q = hirshfeld_charges(eng, "rho_ae")
    np.testing.assert_allclose(q, [11 - 1.0, 17 - 7.0], rtol=1e-4)   # Z - Q with Z = 11, 17


def test_hirshfeld_detects_charge_transfer():
    # move 0.3 e from Na's free-atom shape onto Cl's
    rho = _gauss_rho(NACL)
    na, _, _ = GaussianSuperposition(Structure(NACL.lattice, ["Na"], [[0, 0, 0]]), 1.1, 0.3).on_grid(
        Grid(NACL.lattice, (36,) * 3), False)
    cl, _, _ = GaussianSuperposition(Structure(NACL.lattice, ["Cl"], [[.5, .5, .5]]), 0.7, 0.3).on_grid(
        Grid(NACL.lattice, (36,) * 3), False)
    eng = Engine(NACL, {"rho": rho - na + cl}, reference=GREF, zval={"Na": 1, "Cl": 7})
    q = hirshfeld_charges(eng, "rho")
    assert q[0] > 0.05 and q[1] < -0.05                 # Na positive, Cl negative
    assert q.sum() == pytest.approx(0.0, abs=1e-8)


def test_hirshfeld_r_cut_convergence():
    eng = Engine(NACL, {"rho_ae": _gauss_rho(NACL)}, reference=GREF)
    a = site_charges(eng, "hirshfeld", "rho_ae", r_cut=5.0)
    b = site_charges(eng, "hirshfeld", "rho_ae", r_cut=8.0)
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_hirshfeld_weights_follow_the_field_electrons(tmp_path):
    # valence field -> valence weights: the valence promolecule itself is neutral
    ref = AtomicLDAReference(cache_dir=str(tmp_path))
    # needs a production-like spacing (~0.08 A): on coarse grids the band-limited
    # promolecule and the real-space weights resolve the 3d peak differently
    s = Structure(np.eye(3) * 4.0, ["Fe", "Ni"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    base = Engine(s, {"rho": np.ones((48, 48, 48))}, reference=ref)
    pro = promolecule(base)
    eng = Engine(s, {"rho": pro}, reference=ref)
    with pytest.warns(UserWarning, match="pseudization"):
        q = hirshfeld_charges(eng)
    np.testing.assert_allclose(q, 0.0, atol=2e-3)


def test_hirshfeld_is_default_in_family_h():
    eng = Engine(NACL, {"rho": _gauss_rho(NACL, 24)}, reference=GREF)
    d = compute_descriptors(eng, families=("H",))
    assert "m1_hirshfeld" in d and "m1_power" in d and "m1_becke" not in d
    assert set(d) == set(names(family="H"))


# ---------------------------------------------------------------- ELF_D fidelity (73)

def test_resample_exact_and_general():
    rng = np.random.default_rng(0)
    f = rng.random((12, 12, 16))
    np.testing.assert_array_equal(resample(f, (6, 6, 8)), f[::2, ::2, ::2])
    g = resample(f, (5, 7, 9))
    assert g.shape == (5, 7, 9) and g[0, 0, 0] == pytest.approx(f[0, 0, 0])


def test_elf_fidelity_perfect_and_degraded():
    rho = _gauss_rho(NACL)
    eng = Engine(NACL, {"rho": rho}, reference=GREF)
    from pydemi.descriptors.kinetic import elf_d_field
    elfd = elf_d_field(eng).values
    eng.add_field("elf", elfd[::2, ::2, ::2])                      # ELFCAR on NGX = NGXF / 2
    perfect = elf_fidelity(eng)
    assert perfect["ELFD_fidelity_r"] == pytest.approx(1.0) and perfect["ELFD_fidelity_mae"] == 0
    noisy = elfd[::2, ::2, ::2] + 0.1 * np.random.default_rng(1).standard_normal((18,) * 3)
    eng.add_field("elf", noisy)
    worse = elf_fidelity(eng)
    assert worse["ELFD_fidelity_r"] < 0.99 and worse["ELFD_fidelity_mae"] > 0.05
    d = compute_descriptors(eng, families=("B",))
    assert "ELFD_fidelity_r_bond" in d and "f_ELF_localized" in d


# ---------------------------------------------------------------- engine

def test_read_zval_from_outcar_ignores_summary_line(tmp_path):
    p = tmp_path / "OUTCAR"
    p.write_text("   POMASS =  227.028; ZVAL   =   11.000    mass and valenz\n"
                 "   POMASS =   72.610; ZVAL   =    4.000    mass and valenz\n"
                 "  Ionic Valenz\n   ZVAL   =  11.00  4.00\n")
    assert read_potcar_zval(p) == [11.0, 4.0]


def test_from_vasp_dir_reads_zval_from_outcar(tmp_path):
    write_volumetric(tmp_path / "CHGCAR", NACL, [_gauss_rho(NACL, 12)])
    (tmp_path / "OUTCAR").write_text("POMASS = 22.990; ZVAL = 7.000\nPOMASS = 35.453; ZVAL = 7.000\n"
                                     "   ZVAL   =   7.00  7.00\n")
    assert Engine.from_vasp_dir(tmp_path).valence() == {"Na": 7.0, "Cl": 7.0}


def test_from_vasp_dir_reads_potcar_zval(tmp_path):
    write_volumetric(tmp_path / "CHGCAR", NACL, [_gauss_rho(NACL, 12)])
    (tmp_path / "POTCAR").write_text("ZVAL   =    7.000\nZVAL   =    7.000\n")
    eng = Engine.from_vasp_dir(tmp_path)
    assert eng.valence() == {"Na": 7.0, "Cl": 7.0}
    (tmp_path / "POTCAR").write_text("ZVAL   =    7.000\n")
    with pytest.warns(UserWarning, match="POTCAR"):
        eng = Engine.from_vasp_dir(tmp_path)
    assert eng.valence() == {"Na": 1.0, "Cl": 7.0}                # default rule
