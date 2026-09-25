"""Milestone 12: batch, CLI, sweep, convergence, float32, compositional, PAW extension (spec §5, §8.5, §11-13)."""

import json
import math

import numpy as np
import pytest

import pydemi
from pydemi.cli import main
from pydemi.io.base import Grid, Lattice, Structure, VolumetricData
from pydemi.io.vasp import write_volumetric
from pydemi.validate.analytic import GaussianSuperposition, SlaterSuperposition, cubic_cell, uniform
from pydemi.validate.convergence import (analytic_convergence, grid_convergence,
                                         recommended_spacing, sensitivity_sweep)

LAT = Lattice(np.array([[4.0, 0.0, 0.0], [0.5, 4.1, 0.0], [0.3, 0.4, 4.3]]))
S = Structure(LAT, ["Fe", "O", "O"], [[0.0, 0.0, 0.0], [0.5, 0.45, 0.5], [0.2, 0.7, 0.35]])


def _vd(shape=(20, 20, 22)):
    return GaussianSuperposition(S, [1.4, 2.2, 2.2], [8.0, 6.0, 6.0], tol=1e-20).volumetric(shape)


def _runs(tmp_path, n=3):
    for k in range(n):
        d = tmp_path / f"run{k}"
        d.mkdir()
        write_volumetric(d / "CHGCAR", S, [_vd((16 + 2 * k, 16, 18)).rho.data])
    bad = tmp_path / "broken"
    bad.mkdir()
    (bad / "CHGCAR").write_text("not a CHGCAR\n")
    return sorted(tmp_path.glob("*/CHGCAR"))


# ---------------------------------------------------------------- batch

@pytest.mark.parametrize("workers", [1, 2])
def test_batch_records_errors_and_keeps_a_fixed_schema(tmp_path, workers):
    paths = _runs(tmp_path)
    df = pydemi.featurize_batch(paths, n_workers=workers, progress=False,
                                domains=["bonding", "structural"])
    assert len(df) == 4
    names = pydemi.descriptor_names(["bonding", "structural"])
    assert list(df.columns[1:1 + len(names)]) == names
    for col in ("n_atoms", "volume", "grid_shape", "density_source", "magnetic",
                "euler_consistency", "wall_time_s", "pydemi_version", "error"):
        assert col in df.columns
    broken = df[df["path"].str.contains("broken")]
    assert broken["error"].iloc[0] != "" and math.isnan(broken["m1"].iloc[0])
    assert (df[~df["path"].str.contains("broken")]["error"] == "").all()


def test_batch_skip_and_raise(tmp_path):
    paths = _runs(tmp_path, 1)
    assert len(pydemi.featurize_batch(paths, n_workers=1, on_error="skip", progress=False,
                                      domains=["bonding"])) == 1
    with pytest.raises(RuntimeError, match="broken"):
        pydemi.featurize_batch(paths, n_workers=1, on_error="raise", progress=False,
                               domains=["bonding"])


def test_batch_passes_read_options(tmp_path):
    paths = _runs(tmp_path, 1)
    df = pydemi.featurize_batch(paths, n_workers=1, progress=False, domains=["magnetic"],
                                read_options={"zval": {"Fe": 16.0, "O": 6.0}})
    ok = df[df["error"] == ""]
    assert len(ok) == 1 and ok["zval_source"].iloc[0] == "table"


# ---------------------------------------------------------------- CLI

def test_cli_commands(tmp_path):
    paths = _runs(tmp_path, 1)
    good = str(paths[1])
    assert main(["featurize", good, "--out", str(tmp_path / "f.json"), "--domains", "bonding"]) == 0
    d = json.loads((tmp_path / "f.json").read_text())
    assert "zeta" in d["features"] and d["metadata"]["density_source"] == "pseudo"
    assert main(["catalogue", "--out", str(tmp_path / "cat.csv")]) == 0
    assert (tmp_path / "cat.csv").read_text().startswith("name,domain")
    assert main(["batch", str(tmp_path), "--glob", "*/CHGCAR", "--out", str(tmp_path / "b.csv"),
                 "--workers", "1", "--domains", "bonding", "--quiet"]) == 0
    assert (tmp_path / "b.csv").read_text().count("\n") == 3          # header + 2 rows
    (tmp_path / "paw.json").write_text(json.dumps({"Fe": {"zval": 16, "rcore_bohr": 2.2},
                                                   "O": {"zval": 6}}))
    assert main(["featurize", good, "--out", str(tmp_path / "g.json"), "--domains", "magnetic",
                 "--extensions", "paw", "--paw-table", str(tmp_path / "paw.json")]) == 0
    meta = json.loads((tmp_path / "g.json").read_text())["metadata"]
    assert meta["zval_source"] == "table" and meta["paw_radii_source"] == "table+covalent"
    assert main(["sweep", good, "--param", "c2", "--range", "1.2:1.6:0.2",
                 "--out", str(tmp_path / "s.csv"), "--domains", "bonding"]) == 0
    assert (tmp_path / "s.csv").read_text().count("\n") == 4


# ---------------------------------------------------------------- sweep and convergence

def test_sensitivity_sweep():
    df = sensitivity_sweep(_vd(), c1_range=[0.6, 0.8, 1.6], c2_range=[1.2, 1.5],
                           domains=["bonding"])
    assert len(df) == 4                                        # (1.6, *) skipped: c1 >= c2
    assert {"c1", "c2", "f_core", "f_bond", "f_int"} <= set(df.columns)
    np.testing.assert_allclose(df["f_core"] + df["f_bond"] + df["f_int"], 1.0)


def test_grid_convergence_report():
    df = grid_convergence(_vd((24, 24, 26)), scales=(0.8,), domains=["bonding"])
    assert {"descriptor", "full", "x0.8", "rel_change_x0.8"} <= set(df.columns)
    assert df.set_index("descriptor").loc["m1", "rel_change_x0.8"] < 1e-3


def test_recommended_mesh_comes_from_the_analytic_tests():
    """Spec §11: the minimum mesh is read off the Slater closed forms, not guessed."""
    table = analytic_convergence(spacings=(0.1, 0.08, 0.06))
    print(table.to_string())
    errs = table["integral_rel_error"].to_numpy()
    assert errs[0] > errs[1] > errs[2]
    # m1, m2 reach a ~5e-4 floor set by the nearest-image distance in the 4 A box
    assert recommended_spacing(table, tol=0.02) == pytest.approx(0.08)


# ---------------------------------------------------------------- float32

def test_float32_agrees_with_float64():
    """Spec §13: float32 mode, with the discrepancy reported."""
    vd = _vd((24, 24, 26))
    a = pydemi.featurize(vd, domains=["bonding", "structural", "heterogeneity"])
    b = pydemi.featurize(vd, domains=["bonding", "structural", "heterogeneity"], float32=True)
    rel = {k: abs(a[k] - b[k]) / max(abs(a[k]), 1e-12) for k in a
           if not (math.isnan(a[k]) and math.isnan(b[k]))}
    worst = sorted(rel.items(), key=lambda kv: -kv[1])[:5]
    print("float32 vs float64, largest relative discrepancies:", worst)
    assert np.median(list(rel.values())) < 1e-5
    assert sum(v > 1e-3 for v in rel.values()) <= 0.05 * len(rel)


# ---------------------------------------------------------------- compositional

def test_compositional_is_matminer_magpie():
    matminer = pytest.importorskip("matminer.featurizers.composition")
    from pymatgen.core import Composition
    f = pydemi.featurize(_vd(), domains=["compositional"])
    ref = matminer.ElementProperty.from_preset("magpie").featurize(Composition("FeO2"))
    assert len(f) == len(ref) == 132
    np.testing.assert_allclose(list(f.values()), ref)
    cat = pydemi.catalogue()
    comp = cat[cat.domain == "compositional"]
    assert comp["adopted"].all() and not cat[cat.domain != "compositional"]["adopted"].any()


# ---------------------------------------------------------------- PAW extension

def test_paw_extension_is_off_by_default_and_uses_the_paw_radii():
    vd = _vd()
    assert "rho_min_int" not in pydemi.featurize(vd)
    f, meta = pydemi.featurize(vd, extensions=["paw"], return_metadata=True)
    assert meta["paw_radii_source"] == "covalent"
    vd.paw_radii = {"Fe": 1.2, "O": 0.8}
    _, meta = pydemi.featurize(vd, extensions=["paw"], return_metadata=True)
    assert meta["paw_radii_source"] == "potcar"


def test_rho_min_int_ignores_negative_pseudo_density_inside_the_spheres():
    s = cubic_cell(6.0, ["Si"], [[0.5, 0.5, 0.5]])
    shape = (40, 40, 40)
    rho = SlaterSuperposition(s, [1.0], [4.0], tol=1e-30).on_grid(shape)[0]
    dip = SlaterSuperposition(s, [6.0], [0.5], tol=1e-30).on_grid(shape)[0]
    vd = VolumetricData(s, Grid(rho - dip, s.lattice), paw_radii={"Si": 1.0})
    f = pydemi.featurize(vd, domains=["structural"], extensions=["paw"])
    assert f["rho_min"] < 0.0 < f["rho_min_int"]


def test_n_nnm_paw_ignores_lobes_inside_the_augmentation_sphere():
    """A pseudized atom: no maximum at the nucleus, lobes on a shell at 0.9 A."""
    s = cubic_cell(6.0, ["Si"], [[0.5, 0.5, 0.5]])
    shape = (48, 48, 48)
    lobes = Structure(s.lattice, ["X"] * 6, [[0.5 + dx, 0.5 + dy, 0.5 + dz] for dx, dy, dz in
                                             [(0.15, 0, 0), (-0.15, 0, 0), (0, 0.15, 0),
                                              (0, -0.15, 0), (0, 0, 0.15), (0, 0, -0.15)]])
    rho = GaussianSuperposition(lobes, [4.0] * 6, [1.0] * 6, tol=1e-30).on_grid(shape)[0]
    # a faint background peaked at the nucleus, so the far field has no exactly flat plateau
    x = np.arange(48) / 48 - 0.5
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    rho = rho + 1e-3 * (3.0 + np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y) + np.cos(2 * np.pi * Z))
    vd = VolumetricData(s, Grid(rho, s.lattice), paw_radii={"Si": 1.1})
    f = pydemi.featurize(vd, domains=["structural"], extensions=["paw"])
    assert f["n_NNM"] > 0.0                      # 0.8 A cutoff counts the lobes as non-nuclear
    assert f["n_NNM_paw"] == 0.0                 # max(r_cut, R_PAW) does not


# ---------------------------------------------------------------- sentinels across domains

def test_uniform_density_no_bare_nan_in_any_domain():
    """Spec §10: every degenerate case gives a documented constant, never a bare NaN; the only
    NaN allowed is a within-element variance, which comes with its flag and the site counts."""
    feats, meta = pydemi.featurize(uniform(Structure(LAT, ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
                                           (12, 12, 12)), extensions=["paw"], return_metadata=True)
    for k, v in feats.items():
        if math.isnan(v):
            assert k.endswith("within_element_var") or k.startswith("magpie_"), k
            assert meta[f"{k}__flag"] == 1
