"""Property models applied to the descriptors (pydemi.props)."""

import math

import numpy as np
import pytest

import pydemi
from pydemi import props

FEATS = list(pydemi.descriptor_names())


def _row(X, i):
    return dict(zip(FEATS, X[i]))


def test_model_path_comes_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv(props.ENV, raising=False)
    assert props.model_path() == (props.BUNDLED if props.BUNDLED.exists() else None)
    monkeypatch.setenv(props.ENV, str(tmp_path / "m.json.gz"))
    assert props.model_path() == tmp_path / "m.json.gz"


def test_load_reads_every_target(property_bundle):
    path, *_ = property_bundle
    b = props.load(path)
    assert list(b.targets) == ["bulk_modulus", "dielectric_total", "is_metal", "band_gap"]
    assert b.features == FEATS and b.groups == ["Mechanical", "Dielectric", "Electronic"]
    t = b.targets["bulk_modulus"]
    assert t.error == 2.5 and t.score == ("R²", 0.8)
    assert t.features == FEATS and t.inputs == "descriptors"    # older bundles: the defaults
    assert b.targets["is_metal"].score == ("ROC-AUC", 0.9) and b.targets["is_metal"].error is None
    g = b.targets["band_gap"]
    assert b.needs_composition and g.inputs == "descriptors + composition"
    assert len(g.features) == len(FEATS) + 3 + 2 * 6 and g.seed_metrics["r2"] == (0.79, 0.01)


def test_predict_matches_the_stored_test_predictions(property_bundle):
    """The bundle's test predictions were made by the same boosters, so predict() must give
    them back, including the log10 transform and the composition features."""
    path, X, counts = property_bundle
    b = props.load(path)
    for i in range(20):
        p = props.predict(_row(X, i), b, counts=counts[i])
        for name, t in b.targets.items():
            assert p[name] == pytest.approx(t.test_p[i], rel=1e-9)


def test_predict_needs_the_composition(property_bundle):
    path, X, _ = property_bundle
    with pytest.raises(ValueError, match="composition"):
        props.predict(_row(X, 0), props.load(path))


def test_predictions_are_in_range(property_bundle):
    path, X, counts = property_bundle
    b = props.load(path)
    p = props.predict(_row(X, 5), b, counts=counts[5])  # row 5 has a NaN descriptor
    assert p["dielectric_total"] >= 1.0 and 0.0 <= p["is_metal"] <= 1.0
    assert all(math.isfinite(v) for v in p.values())
    low = props.predict({f: -100.0 for f in FEATS}, b, counts={"Mg": 1})  # far outside: clipped
    assert low["bulk_modulus"] >= 0.0 and low["dielectric_total"] >= 1.0 and low["band_gap"] >= 0.0


def test_composition_features_follow_matminer():
    """Fraction-weighted statistics as in matminer's ElementProperty; Al2O3 by hand."""
    from conftest import MAGPIE
    f = props.composition_features({"Al": 4, "O": 6}, MAGPIE)  # the cell, not the formula unit
    assert f["0-norm"] == 2 and f["2-norm"] == pytest.approx(math.sqrt(0.4 ** 2 + 0.6 ** 2))
    assert f["3-norm"] == pytest.approx((0.4 ** 3 + 0.6 ** 3) ** (1 / 3))
    mean = 0.4 * 1.61 + 0.6 * 3.44
    assert f["MagpieData mean Electronegativity"] == pytest.approx(mean)
    assert f["MagpieData avg_dev Electronegativity"] == pytest.approx(
        0.4 * abs(1.61 - mean) + 0.6 * abs(3.44 - mean))
    assert f["MagpieData range Number"] == 5 and f["MagpieData minimum Number"] == 8
    assert f["MagpieData mode Number"] == 8                     # O, the larger fraction
    assert props.composition_features({"Fe": 1, "Pt": 1}, MAGPIE)["MagpieData mode Number"] == 26  # tie: smaller
    assert math.isnan(props.composition_features({"Si": 1, "O": 2}, MAGPIE)["MagpieData mean Electronegativity"])
    assert math.isnan(props.composition_features({"Xx": 1}, MAGPIE)["MagpieData mean Number"])


def test_match_finds_a_dataset_material(property_bundle):
    path, X, _ = property_bundle
    b = props.load(path)
    m = b.match(_row(X, 199))
    assert m is not None and m.material_id == "mp-199" and m.split == "test"
    assert m.values["bulk_modulus"] == pytest.approx(X[199, 0] * 50 + 100)
    assert b.match(_row(X, 5)).material_id == "mp-5"   # NaN equals NaN
    other = _row(X, 3)
    other[FEATS[7]] *= 1.01
    assert b.match(other) is None


def test_outside_range(property_bundle):
    path, X, _ = property_bundle
    b = props.load(path)
    x = {f: float(np.median(X[:, j])) for j, f in enumerate(FEATS)}
    assert props.outside_range(x, b) == []
    x[FEATS[4]] = 10.0
    assert props.outside_range(x, b) == [FEATS[4]]


def test_load_without_a_model(monkeypatch):
    monkeypatch.delenv(props.ENV, raising=False)
    monkeypatch.setattr(props, "BUNDLED", props.BUNDLED.with_name("absent.json.gz"))
    props.load.cache_clear()
    with pytest.raises(FileNotFoundError, match=props.ENV):
        props.load()


def test_gui_table_and_texts(property_bundle):
    pytest.importorskip("plotly")
    from pydemi.gui import properties as P
    path, X, counts = property_bundle
    b = props.load(path)
    pred = props.predict(_row(X, 0), b, counts=counts[0])
    table = P.table(pred, b)
    assert list(table.columns) == ["property", "group", "prediction", "typical error", "units",
                                   "test score", "name"]
    metal = pred["is_metal"] >= 0.5
    assert (table.set_index("name").at["dielectric_total", "prediction"] == "n/a (metal)") == metal
    with_dft = P.table(pred, b, b.match(_row(X, 0)))
    assert "DFT" in with_dft.columns
    t = b.targets["bulk_modulus"]
    assert P.decimals(t) == 1 and P.error_text(t) == "± 2.5"
    assert P.value_text(b.targets["is_metal"], 0.999) == "Metal (> 99 %)"
    assert P.value_text(b.targets["is_metal"], 0.2) == "Not metal (80 %)"
    for name, t in b.targets.items():                   # every figure builds
        fig = P.probabilities(t, pred[name]) if t.task == "classification" else P.parity(t, pred[name])
        assert fig.data and P.importance(t).data
    assert "test set" in P.match_text(b.match(_row(X, 199)))
    assert P.counts({"site_counts": "Na:1,Cl:2"}) == {"Na": 1, "Cl": 2} and P.counts({}) == {}
    assert P.inputs_text(b) == f"the {len(FEATS)} descriptors and the composition (Magpie)"
    assert P.feature_text("MagpieData avg_dev Electronegativity") == "Electronegativity (avg dev)"
    assert P.feature_text("2-norm") == "stoichiometry 2-norm" and P.feature_text("zeta") == "zeta"
    gap = b.targets["band_gap"]
    kinds = {tr.name for tr in P.importance(gap).data}
    assert kinds <= {"descriptor", "composition"} and kinds
    rows = dict(P.model_rows(gap, pred, b))
    assert rows["Inputs"] == f"descriptors + composition ({len(gap.features)})"
    assert rows["R², 5 seeds"] == "0.790 ± 0.010"
    assert "inputs" in P.export(pred, b).columns
    from dataclasses import replace
    bulk = b.targets["bulk_modulus"]
    low = replace(bulk, metrics={**bulk.metrics, "r2": 0.3})
    assert P.weak(low) and not P.weak(bulk) and not P.weak(b.targets["is_metal"])
    assert P.weak(replace(gap, seed_metrics={"r2": (0.45, 0.02)}))        # the seed mean decides
    assert not table["test score"].str.contains("⚠").any()
    assert "rough indication" in P.note(low, pred) and P.export(pred, b)["weak_model"].dtype == bool


def _counts(formula: str) -> dict[str, float]:
    """Element counts of a formula such as 'Rb(ThSe3)2'."""
    import re
    stack: list[dict[str, float]] = [{}]
    for el, n, close, m in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)|(\))(\d*)|\(", formula):
        if el:
            stack[-1][el] = stack[-1].get(el, 0.0) + float(n or 1)
        elif close:
            inner = stack.pop()
            for e, k in inner.items():
                stack[-1][e] = stack[-1].get(e, 0.0) + k * float(m or 1)
        else:
            stack.append({})
    return stack[0]


def test_shipped_bundle_reproduces_its_test_predictions():
    """The bundle in the package (trimmed models) gives back the test-set predictions made by
    the full models when it was built, from the stored descriptors and the formula."""
    pytest.importorskip("lightgbm")
    assert _counts("Rb(ThSe3)2") == {"Rb": 1, "Th": 2, "Se": 6}
    b = props.load(props.BUNDLED)
    assert len(b.targets) == 22 and b.dataset is not None and b.composition is not None
    ds = b.dataset
    test = [i for i, s in enumerate(ds["split"]) if s == "test"]
    # each target's test predictions cover the test materials that have a value for it
    rows = {name: [i for i in test if not np.isnan(ds["y"][name][i])] for name in b.targets}
    assert all(len(rows[n]) == len(t.test_p) for n, t in b.targets.items())
    for k in test[::30]:
        feats = dict(zip(b.features, ds["X"][k]))
        p = props.predict(feats, b, counts=_counts(ds["formula"][k]))
        for name, t in b.targets.items():
            if k in rows[name]:
                assert p[name] == pytest.approx(t.test_p[rows[name].index(k)], rel=1e-9, abs=1e-12), \
                    (name, ds["formula"][k])
