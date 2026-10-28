"""The Compare view's tables (pydemi.gui.compare): DFT and predicted densities of one structure."""

import math

import numpy as np
import pytest

import pydemi
from pydemi.gui import compare
from pydemi.io.base import Grid, Lattice, Structure, VolumetricData

NAMES = list(pydemi.descriptor_names())


def test_relative_difference():
    assert compare.relative(1.0, 1.0) == 0.0 and compare.relative(0.0, 0.0) == 0.0
    assert compare.relative(2.0, 1.0) == compare.relative(1.0, 2.0) == 0.5
    assert compare.relative(1.0, -1.0) == 2.0 and math.isnan(compare.relative(1.0, math.nan))


def test_the_transfer_table_covers_every_descriptor():
    t = compare.transfer()
    assert set(t.index) == set(NAMES)
    assert t["spearman"].isna().sum() == 1 and math.isnan(t.at["lap_concentration", "spearman"])
    assert (t["spearman"].dropna() <= 1).all() and (t["scale"] >= 0).all()
    assert compare.grade(0.995) == "excellent" and compare.grade(0.96) == "good"
    assert compare.grade(0.9) == "fair" and compare.grade(0.5) == "poor"


def test_descriptor_table_marks_differences_noise_and_the_constant():
    scale = compare.transfer()["scale"]
    dft = {n: float(scale[n]) for n in NAMES}
    pred = dict(dft)
    pred["m1"] *= 1.10                                       # 9 % apart
    pred["f_int"] *= 1.005                                   # within 1 %
    dft["rho_mid_std"], pred["rho_mid_std"] = 1e-6 * scale["rho_mid_std"], 3e-6 * scale["rho_mid_std"]
    t = compare.descriptors(dft, pred).set_index("descriptor")
    assert len(t) == 56
    assert t.at["m1", "difference %"] == pytest.approx(100 * 0.1 / 1.1) and t.at["m1", "note"] == "differs"
    assert t.at["f_int", "difference %"] < 1 and t.at["f_int", "note"] == ""
    assert math.isnan(t.at["rho_mid_std", "difference %"]) and t.at["rho_mid_std", "note"] == "≈ 0 in both"
    assert t.at["lap_concentration", "note"] == "constant"
    s = compare.descriptor_summary(t.reset_index())
    assert s == {"median": 0.0, "close": 54, "n": 55, "far": 1}


def test_density_error_needs_the_same_grid():
    lat = Lattice(np.eye(3) * 3.0)
    s = Structure(lat, ["Si"], [[0, 0, 0]])
    a = VolumetricData(s, Grid(np.full((4, 4, 4), 2.0), lat))
    b = VolumetricData(s, Grid(np.full((4, 4, 4), 2.1), lat))
    assert compare.density_error(a, b) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="grids differ"):
        compare.density_error(a, VolumetricData(s, Grid(np.ones((4, 4, 5)), lat)))


def test_property_table_agrees_within_the_typical_error(property_bundle):
    from pydemi import props
    path, X, counts = property_bundle
    b = props.load(path)
    feats = dict(zip(NAMES, X[0]))
    p = props.predict(feats, b, counts=counts[0])
    q = dict(p)
    name = next(n for n, t in b.targets.items() if t.task == "regression" and t.error)
    q[name] = p[name] + 3 * b.targets[name].error             # far beyond the typical error
    cls = next(n for n, t in b.targets.items() if t.task == "classification")
    q[cls] = 1 - p[cls] if abs(p[cls] - 0.5) > 0.01 else p[cls]
    t = compare.properties(p, q, b).set_index("name")
    assert not t.at[name, "agree"]
    from pydemi.gui.properties import GAPPED_ONLY
    flipped = (p[cls] >= 0.5) != (q[cls] >= 0.5)
    rest = [n for n in b.targets if n not in (name, cls) and not (flipped and n in GAPPED_ONLY)]
    assert t.loc[rest, "agree"].all()
    if flipped and cls == "is_metal":                         # a gapped-only property on one side only
        assert not t.loc[[n for n in b.targets if n in GAPPED_ONLY], "agree"].any()
    assert t.at[cls, "ML − DFT"] == ("class differs" if abs(p[cls] - 0.5) > 0.01 else "same class")
    assert list(compare.properties(p, p, b)["agree"]) == [True] * len(b.targets)
    e = compare.export(compare.descriptors(feats, feats), t.reset_index(), feats, feats, p, q)
    assert len(e) == 56 + len(b.targets) and set(e["kind"]) == {"descriptor", "property"}
