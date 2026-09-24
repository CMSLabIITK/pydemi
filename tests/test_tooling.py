"""Phase 6: calibration (D, 106), strain response (107), resampling,
convergence report, cube / XSF readers, batch runner and CLI."""

import csv

import numpy as np
import pytest

from pydemi import Engine, Grid, Structure
from pydemi.calibration import (BulkModulusCalibration, IonicityCalibration,
                                calibration_family, cohen_bulk_modulus, cohen_lambda,
                                fit_bulk_modulus, fit_ionicity, phillips_table,
                                reduced_formula)
from pydemi.cli import main as cli
from pydemi.convergence import convergence_report, format_report
from pydemi.descriptors import compute_descriptors, names
from pydemi.io.grids import read_cube, read_density, read_xsf, write_cube, write_xsf
from pydemi.io.vasp import write_volumetric
from pydemi.batch import done_ids, find_runs, run_batch
from pydemi.resample import fft_friendly, fourier_resample, resample_engine, shape_for_spacing
from pydemi.strain import strain_response
from pydemi.testing import GaussianSuperposition

from conftest import TRICLINIC

NACL = Structure(np.eye(3) * 5.6, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _rho(structure, n=24, alphas=(1.1, 0.7), N=(1.0, 7.0)):
    rho, _, _ = GaussianSuperposition(structure, list(alphas)[:structure.n_atoms],
                                      list(N)[:structure.n_atoms]).on_grid(
        Grid(structure.lattice, (n, n, n) if np.isscalar(n) else n), False)
    return rho


# ---------------------------------------------------------------- literature / Cohen

def test_phillips_table():
    t = phillips_table()
    assert len(t) == 67
    assert t["GaAs"]["f_i"] == 0.310 and t["NaCl"]["f_i"] == 0.935
    assert {r["status"] for r in t.values()} == {"verified", "recalled"}
    v = phillips_table(verified_only=True)
    assert "GaAs" in v and "NaCl" not in v          # NaCl not yet checked against a source
    assert all(r["source"] for r in t.values())


@pytest.mark.parametrize("species,lam", [
    (["Si"] * 8, 0), (["Si", "C"], 0), (["Ga", "As"], 1), (["Zn", "S"], 2),
    (["Mg", "O"], 2), (["Na", "Cl"], None), (["Fe", "Co", "Ni", "Cr"], None),
    (["Ga", "Ga", "O", "O", "O"], None)])
def test_cohen_lambda(species, lam):
    assert cohen_lambda(species) == lam


def test_cohen_formula_reproduces_known_moduli():
    # Si d = 2.35 A, B0 ~ 98 GPa; GaAs d = 2.45 A, B0 ~ 75 GPa
    assert cohen_bulk_modulus(2.35, 0) == pytest.approx(98, rel=0.03)
    assert cohen_bulk_modulus(2.45, 1) == pytest.approx(75.5, rel=0.03)
    assert reduced_formula(["As", "Ga"]) == "GaAs"


# ---------------------------------------------------------------- ionicity calibration

def _synthetic_rows(noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = {}
    for formula, r in phillips_table().items():
        f = r["f_i"]
        logit = np.log((f + 0.01) / (1.01 - f))
        rows[formula] = {"fint_over_lnf": 0.5 + 0.4 * logit + noise * rng.normal(),
                         "VH_spread": 2.0 + 1.5 * logit + noise * rng.normal()}
    return rows


def test_fit_ionicity_recovers_a_monotone_relation(tmp_path):
    cal = fit_ionicity(_synthetic_rows())
    assert cal.n == 67 and cal.n_tetrahedral > 30
    assert cal.rmse < 0.01 and cal.loocv_rmse < 0.02
    rows = _synthetic_rows()
    assert cal.predict(rows["GaAs"]) == pytest.approx(0.310, abs=0.02)
    cal.save(tmp_path / "ion.json")
    again = IonicityCalibration.load(tmp_path / "ion.json")
    assert again.predict(rows["NaCl"]) == pytest.approx(cal.predict(rows["NaCl"]))
    assert np.isnan(cal.predict({"fint_over_lnf": 1.0}))       # missing feature
    noisy = fit_ionicity(_synthetic_rows(noise=0.3))
    assert noisy.loocv_rmse > cal.loocv_rmse


def test_fit_ionicity_needs_enough_compounds():
    rows = {k: v for k, v in list(_synthetic_rows().items())[:3]}
    with pytest.raises(ValueError):
        fit_ionicity(rows)


def test_fit_bulk_modulus_power_law(tmp_path):
    rows = [{"rho_mid_mean": r, "bond_length_mean": d} for r, d in
            [(0.3, 2.3), (0.5, 2.5), (0.2, 2.9), (0.8, 2.1), (0.4, 2.7)]]
    x = np.array([r["rho_mid_mean"] / r["bond_length_mean"] ** 3 for r in rows])
    B0 = 3000.0 * x ** 0.9
    cal = fit_bulk_modulus(rows, B0)
    assert cal.a == pytest.approx(3000.0, rel=1e-8) and cal.b == pytest.approx(0.9, rel=1e-8)
    cal.save(tmp_path / "b.json")
    assert BulkModulusCalibration.load(tmp_path / "b.json").predict(rows[0]) == pytest.approx(B0[0])


def test_family_d_in_compute_descriptors():
    eng = Engine(NACL, {"rho": _rho(NACL)})
    d = compute_descriptors(eng, families=("D",))
    assert set(d) == set(names(family="D"))                    # inputs computed, not returned
    assert np.isnan(d["grid_ionicity"]) and np.isnan(d["B0_rho_proxy"])
    assert d["cohen_in_scope"] == 0 and np.isfinite(d["cohen_B0_predicted"])
    cal = fit_ionicity(_synthetic_rows())
    full = compute_descriptors(eng, families=("tier2", "D"), calibrations={"ionicity": cal})
    assert 0 < full["grid_ionicity"] < 1
    assert full["ionicity_residual"] == pytest.approx(full["grid_ionicity"] - full["ionicity"])


# ---------------------------------------------------------------- strain (107)

def test_strain_response_ignores_uniform_dilation():
    rho = _rho(NACL, 20)
    for eps in (0.01,):
        s_m = Structure(NACL.lattice * (1 - eps), NACL.species, NACL.frac_coords)
        s_p = Structure(NACL.lattice * (1 + eps), NACL.species, NACL.frac_coords)
        # same charge per voxel, density rescaled by the volume change
        em = Engine(s_m, {"rho": rho / (1 - eps) ** 3})
        ep = Engine(s_p, {"rho": rho / (1 + eps) ** 3})
        out = strain_response(em, ep, eps)
        assert out["drho_deps"] == pytest.approx(0.0, abs=1e-12)
        assert out["charge_drift"] == pytest.approx(0.0, abs=1e-12)


def test_strain_response_measures_redistribution():
    rho = _rho(NACL, 20)
    shift = np.zeros_like(rho)
    shift[0, 0, 0], shift[10, 10, 10] = 1.0, -1.0              # move charge between two voxels
    V = NACL.volume
    em = Engine(NACL, {"rho": rho - 0.01 * shift})
    ep = Engine(NACL, {"rho": rho + 0.01 * shift})
    out = strain_response(em, ep, 0.01)
    moved = 2 * 0.02 * V / rho.size                             # electrons |n+ - n-| summed
    assert out["drho_deps"] == pytest.approx(moved / (2 * 0.01 * 8.0), rel=1e-6)
    with pytest.raises(ValueError):
        strain_response(em, Engine(Structure(NACL.lattice, ["K", "Cl"], NACL.frac_coords),
                                   {"rho": rho}), 0.01)


# ---------------------------------------------------------------- resampling

def test_fft_friendly_and_spacing():
    assert fft_friendly(97) == 98 and fft_friendly(128) == 128 and fft_friendly(11) == 12
    assert shape_for_spacing(np.diag([5.0, 7.3, 10.0]), 0.1) == (50, 75, 100)


def test_fourier_resample_is_exact_for_band_limited_fields():
    grid = Grid(TRICLINIC, (12, 14, 16))
    u = grid.frac_coords()
    f = 1 + np.cos(2 * np.pi * u[..., 0]) + 0.3 * np.sin(2 * np.pi * (u[..., 1] - 2 * u[..., 2]))
    up = fourier_resample(f, (24, 21, 32))
    back = fourier_resample(up, f.shape)
    np.testing.assert_allclose(back, f, atol=1e-12)
    assert up.mean() == pytest.approx(f.mean())
    u2 = Grid(TRICLINIC, (24, 21, 32)).frac_coords()
    exact = 1 + np.cos(2 * np.pi * u2[..., 0]) + 0.3 * np.sin(2 * np.pi * (u2[..., 1] - 2 * u2[..., 2]))
    np.testing.assert_allclose(up, exact, atol=1e-12)


def test_resample_engine_keeps_grid_ratios():
    eng = Engine(NACL, {"rho": _rho(NACL, 24), "elf": np.full((12, 12, 12), 0.5)})
    eng.add_field("elf_d", np.zeros((24, 24, 24)))              # derived: dropped
    new = resample_engine(eng, scale=0.5)
    assert new["rho"].grid.shape == (12, 12, 12) and new["elf"].grid.shape == (6, 6, 6)
    assert "elf_d" not in new
    assert new["rho"].integral() == pytest.approx(eng["rho"].integral(), rel=1e-10)
    with pytest.raises(ValueError):
        resample_engine(eng, spacing=0.1, scale=0.5)


# ---------------------------------------------------------------- convergence report

def test_convergence_report_on_smooth_density():
    eng = Engine(NACL, {"rho": _rho(NACL, 40)})
    rep = convergence_report(eng, factors=(1.0, 0.8), families=("tier1",), rtol=0.02)
    assert rep["shapes"] == [(40, 40, 40), (32, 32, 32)]
    rows = {r["name"]: r for r in rep["rows"]}
    assert rows["m1"]["converged"] and rows["Q_tot"]["rel_change"] < 1e-10
    changes = [r["rel_change"] for r in rep["rows"]]
    assert changes == sorted(changes, reverse=True)
    assert "descriptors change by more than" in format_report(rep)


# ---------------------------------------------------------------- cube / XSF

@pytest.mark.parametrize("ext,writer,reader", [(".cube", write_cube, read_cube),
                                               (".xsf", write_xsf, read_xsf)])
def test_grid_format_round_trip(tmp_path, ext, writer, reader):
    s = Structure(TRICLINIC, ["Si", "O"], [[0.1, 0.2, 0.3], [0.6, 0.55, 0.7]])
    rho = np.random.default_rng(0).random((6, 7, 8))
    p = tmp_path / f"rho{ext}"
    writer(p, s, rho)
    vol = reader(p)
    # cube's conventional %13.5E keeps 6 significant digits
    np.testing.assert_allclose(vol.blocks[0], rho, rtol=1e-5)
    np.testing.assert_allclose(vol.structure.lattice, s.lattice, atol=1e-5)
    np.testing.assert_allclose(vol.structure.frac_coords, s.frac_coords, atol=1e-5)
    assert vol.structure.species == ("Si", "O")
    eng = Engine.from_file(p)
    np.testing.assert_allclose(eng["rho"].values, rho, rtol=1e-5)


def test_cube_units(tmp_path):
    s = Structure(np.eye(3) * 4.0, ["Na"], [[0, 0, 0]])
    rho = np.full((4, 4, 4), 0.5)                               # e/A^3
    write_cube(tmp_path / "a.cube", s, rho)                     # stored as e/bohr^3
    raw = read_cube(tmp_path / "a.cube", density_unit=None).blocks[0]
    np.testing.assert_allclose(raw, 0.5 * 0.529177210903 ** 3, rtol=1e-5)
    np.testing.assert_allclose(read_density(tmp_path / "a.cube").total, 0.5, rtol=1e-5)
    with pytest.raises(ValueError):
        read_cube(tmp_path / "a.cube", density_unit="furlongs")


# ---------------------------------------------------------------- batch / CLI

def _runs(tmp_path):
    for name, s in (("NaCl_225", NACL), ("KCl_225", Structure(NACL.lattice, ["K", "Cl"],
                                                              NACL.frac_coords))):
        d = tmp_path / "data" / name
        d.mkdir(parents=True)
        write_volumetric(d / "CHGCAR", s, [_rho(s, 16)])
    bad = tmp_path / "data" / "broken"
    bad.mkdir()
    (bad / "CHGCAR").write_text("not a chgcar\n")
    return tmp_path / "data"


def test_batch_runs_records_errors_and_resumes(tmp_path):
    root = _runs(tmp_path)
    assert [p.name for p in find_runs(root)] == ["KCl_225", "NaCl_225", "broken"]
    out = tmp_path / "d.csv"
    counts = run_batch([root], out, families=("tier1", "tier2"), progress=False)
    assert counts == {"done": 3, "skipped": 0, "failed": 1}
    rows = list(csv.DictReader(out.open()))
    by_id = {r["material_id"]: r for r in rows}
    assert by_id["broken"]["error"] and not by_id["NaCl_225"]["error"]
    assert float(by_id["NaCl_225"]["m1"]) > 0
    assert set(rows[0]) >= {"material_id", "path", "error", "zeta", "mean_mass"}
    again = run_batch([root], out, families=("tier1", "tier2"), progress=False)
    assert again["skipped"] == 3 and again["done"] == 0
    assert done_ids(out) == {"NaCl_225", "KCl_225", "broken"}


def test_batch_parallel(tmp_path):
    root = _runs(tmp_path)
    counts = run_batch(find_runs(root)[:2], tmp_path / "p.csv", families=("tier1",),
                       workers=2, progress=False)
    assert counts["done"] == 2 and counts["failed"] == 0


def test_cli(tmp_path, capsys):
    root = _runs(tmp_path)
    assert cli(["list", "--family", "D"]) == 0
    assert "grid_ionicity" in capsys.readouterr().out
    assert cli(["describe", "lap_concentration"]) == 0
    assert "identically 1/2" in capsys.readouterr().out
    out = tmp_path / "cli.csv"
    code = cli(["compute", str(root / "NaCl_225"), "-o", str(out), "--families", "tier1", "-q"])
    assert code == 0 and "NaCl_225" in out.read_text()
    assert cli(["compute", str(root / "broken"), "-o", str(out), "-q"]) == 1
    assert cli(["convergence", str(root / "NaCl_225"), "--factors", "1", "0.75"]) == 0
    assert "rel.change" in capsys.readouterr().out
    assert cli(["resample", str(root / "NaCl_225" / "CHGCAR"), str(tmp_path / "R"),
                "--spacing", "0.5"]) == 0
    assert Engine.from_chgcar(tmp_path / "R").grid().shape == (12, 12, 12)
