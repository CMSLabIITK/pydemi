"""Shared fixtures."""

import gzip
import json

import numpy as np
import pytest

import pydemi

# a two-property element table in the format of a bundle's "composition" block
MAGPIE = {"p": [0, 2, 3], "properties": ["Number", "Electronegativity"],
          "elements": {"Na": [11, 0.93], "Cl": [17, 3.16], "Mg": [12, 1.31], "O": [8, 3.44],
                       "Al": [13, 1.61], "Fe": [26, 1.83], "Pt": [78, 2.28], "Si": [14, None]}}
COMPOSITIONS = ({"Mg": 1, "O": 1}, {"Al": 2, "O": 3}, {"Fe": 1, "Pt": 1}, {"Na": 1, "Cl": 2})


def _bundle(path, X: np.ndarray, counts: list[dict], rng: np.random.Generator) -> None:
    """A small property-model bundle in the format of property_prediction/scripts/build_bundle.py:
    two regressions (one on log10) and one classification on random descriptors, in the older
    format without per-target inputs, and one regression on descriptors + composition."""
    import lightgbm as lgb
    from pydemi.props import composition_features
    feats = list(pydemi.descriptor_names())
    comp = [composition_features(c, MAGPIE) for c in counts]
    cnames = list(comp[0])
    XC = np.hstack([X, np.array([[r[k] for k in cnames] for r in comp])])
    n = len(X)
    k = X[:, 0] * 50 + 100                              # depends on the first descriptor
    eps = 10 ** (X[:, 1] + 0.5)                         # log-normal, >= 1
    metal = (X[:, 2] > 0.5).astype(float)
    gap = X[:, 3] + XC[:, len(feats) + cnames.index("MagpieData mean Electronegativity")]
    targets = {}
    for name, y, z, task, how, floor, label, group, units, inputs in (
            ("bulk_modulus", k, k, "regression", None, 0.0, "Bulk modulus", "Mechanical", "GPa", None),
            ("dielectric_total", eps, np.log10(eps), "regression", "log10", 1.0,
             "Dielectric constant, total", "Dielectric", "1", None),
            ("is_metal", metal, metal, "classification", None, None, "Metallic", "Electronic", "0/1", None),
            ("band_gap", gap, gap, "regression", None, 0.0, "Band gap (PBE)", "Electronic", "eV",
             "descriptors + composition")):
        names, A = (feats + cnames, XC) if inputs else (feats, X)
        params = {"objective": "binary" if task == "classification" else "regression",
                  "num_leaves": 7, "min_child_samples": 5, "verbose": -1, "num_threads": 1, "seed": 0}
        model = lgb.train(params, lgb.Dataset(A, z, feature_name=names), num_boost_round=30)
        p = model.predict(A[:20])
        p = p if how is None else 10 ** p
        metrics = ({"roc_auc": 0.9, "accuracy": 0.9, "f1": 0.9, "positive_rate": 0.5}
                   if task == "classification" else {"mae": 2.5, "rmse": 3.0, "r2": 0.8, "median_ae": 2.0})
        gain = model.feature_importance("gain")
        t = {"label": label, "symbol": name[:1].upper(), "group": group, "task": task, "units": units,
             "description": f"test {name}", "transform": how, "floor": floor, "metrics": metrics,
             "n_train": n, "n_val": 0, "n_test": 20, "best_iteration": 30,
             "importance": {f: float(g / gain.sum()) for f, g in
                            sorted(zip(names, gain), key=lambda kv: -kv[1])},
             "test": {"y": y[:20].tolist(), "p": p.tolist(), "formula": [f"X{i}" for i in range(20)]},
             "booster": model.model_to_string()}
        if inputs:
            t.update(features=names, inputs=inputs, importance_kind="mean |SHAP| on the test set",
                     metrics_seeds={"r2": [0.79, 0.01], "mae": [2.6, 0.1]})
        targets[name] = t
    raw = {"name": "Test bundle", "created": "2026-10-19", "split": "train 200",
           "data": "random", "features": feats,
           "feature_range": {f: [float(np.quantile(X[:, j], 0.01)), float(np.quantile(X[:, j], 0.99))]
                             for j, f in enumerate(feats)},
           "composition": MAGPIE, "targets": targets,
           "dataset": {"material_id": [f"mp-{i}" for i in range(n)], "formula": [f"X{i}" for i in range(n)],
                       "split": ["train"] * (n - 1) + ["test"],
                       "X": [[None if np.isnan(v) else float(v) for v in row] for row in X],
                       "y": {"bulk_modulus": k.tolist(), "dielectric_total": eps.tolist(),
                             "is_metal": metal.tolist(), "band_gap": gap.tolist()}}}
    with gzip.open(path, "wt") as fh:
        json.dump(raw, fh)


@pytest.fixture(scope="session")
def property_bundle(tmp_path_factory):
    """Path, descriptor matrix and site counts of a small property-model bundle (needs lightgbm)."""
    from pydemi import props
    if props.load_error():
        pytest.skip(props.load_error())
    rng = np.random.default_rng(0)
    X = rng.random((200, len(pydemi.descriptor_names())))
    X[5, 10] = np.nan                                   # NaN descriptors are allowed
    counts = [COMPOSITIONS[i] for i in rng.integers(0, len(COMPOSITIONS), len(X))]
    path = tmp_path_factory.mktemp("props") / "bundle.json.gz"
    _bundle(path, X, counts, rng)
    return path, X, counts
