"""Material properties predicted from the descriptors.

A *model bundle* (JSON, xz- or gzip-compressed) holds one LightGBM model per property, the descriptor
names it was trained on, their training range, and each model's test-set scores and test
predictions. ``load`` reads a bundle, ``predict`` turns a descriptor dict into one row per
property.

pydemi ships one bundle, ``properties_v2.json.xz``: LightGBM models of 17 properties trained
on the descriptors alone, on the CMS Lab dataset with targets from the Materials Project
(written by ``property_prediction/scripts/build_bundle.py``). ``PYDEMI_PROPERTY_MODEL`` points
at another.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import lzma
import math
import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Union

REQUIRES = ("lightgbm",)
INSTALL_HINT = 'pip install "pydemi[props]"'
ENV = "PYDEMI_PROPERTY_MODEL"
BUNDLED = Path(__file__).with_name("properties_v2.json.xz")

if sys.platform == "darwin":
    # PyTorch's macOS wheels bring their own OpenMP runtime and LightGBM uses Homebrew's libomp;
    # with the property models and ChargE3Net in one process (the GUI), the second runtime
    # would abort the process unless duplicates are allowed. Respects a value set by the user.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def missing() -> list[str]:
    """The packages ``predict`` needs that are not installed, or installed but not loadable."""
    return [m for m in REQUIRES if importlib.util.find_spec(m) is None or load_error() is not None]


@lru_cache(maxsize=1)
def load_error() -> Optional[str]:
    """Why LightGBM cannot be imported (None if it can). On macOS its wheel needs the OpenMP
    runtime, so the import fails with an OSError until ``brew install libomp``."""
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return f"LightGBM is not installed: {INSTALL_HINT}"
    except OSError as e:
        hint = " On macOS, install OpenMP with: brew install libomp" if sys.platform == "darwin" else ""
        return f"LightGBM is installed but cannot load ({e}).{hint}"
    return None


def model_path() -> Optional[Path]:
    """The bundle to use: ``$PYDEMI_PROPERTY_MODEL`` if set, else the shipped one, else None."""
    env = os.environ.get(ENV)
    if env:
        return Path(env).expanduser()
    return BUNDLED if BUNDLED.exists() else None


@dataclass
class Target:
    """One property model and how well it did on the test set."""

    name: str
    label: str
    symbol: str
    group: str
    task: str                       # "regression" or "classification"
    units: str
    description: str
    transform: Optional[str]        # the model predicts log10(y) or log1p(y)
    floor: Optional[float]          # predictions are clipped from below at this value
    metrics: dict[str, float]
    n_train: int
    n_test: int
    importance: dict[str, float]    # share of the importance (gain or mean |SHAP|), largest first
    test_y: list[float] = field(repr=False)
    test_p: list[float] = field(repr=False)
    test_formula: list[str] = field(repr=False)
    booster: Any = field(repr=False)
    features: list[str] = field(default_factory=list, repr=False)   # the model's inputs, in order
    importance_kind: str = "share of split gain"
    seed_metrics: dict[str, tuple[float, float]] = field(default_factory=dict)  # mean, std over seeds

    @property
    def error(self) -> Optional[float]:
        """Typical error of a regression: its test-set MAE."""
        return self.metrics.get("mae")

    @property
    def score(self) -> tuple[str, float]:
        """The headline test score: R² for a regression, ROC-AUC for a classification."""
        return ("ROC-AUC", self.metrics["roc_auc"]) if self.task == "classification" else ("R²", self.metrics["r2"])


@dataclass
class Match:
    """A dataset material whose descriptors equal the input's."""

    material_id: str
    formula: str
    split: str                      # "train", "val" or "test"
    values: dict[str, float]        # its DFT values (NaN where it has none)


@dataclass
class Bundle:
    name: str
    path: Path
    created: str
    split: str
    data: str
    features: list[str]
    feature_range: dict[str, tuple[float, float]]
    targets: dict[str, Target]
    dataset: Optional[dict[str, Any]] = field(default=None, repr=False)

    @property
    def groups(self) -> list[str]:
        return list(dict.fromkeys(t.group for t in self.targets.values()))

    def match(self, feats: Mapping[str, float], rtol: float = 1e-5) -> Optional[Match]:
        """The dataset material with the same descriptors, if the input is one of them."""
        import numpy as np
        if not self.dataset:
            return None
        X = self.dataset["X"]
        x = np.array([float(feats.get(f, math.nan)) for f in self.features])
        same = (np.isclose(X, x, rtol=rtol, atol=1e-12) | (np.isnan(X) & np.isnan(x))).all(axis=1)
        hits = np.flatnonzero(same)
        if len(hits) != 1:
            return None
        i = int(hits[0])
        ds = self.dataset
        return Match(ds["material_id"][i], ds["formula"][i], ds["split"][i],
                     {k: float(v[i]) for k, v in ds["y"].items()})


@lru_cache(maxsize=2)
def load(path: Union[str, Path, None] = None) -> Bundle:
    """Read a model bundle (cached)."""
    p = Path(path) if path is not None else model_path()
    if p is None:
        raise FileNotFoundError(f"no property model; set {ENV} to a model bundle")
    import lightgbm as lgb
    opener = lzma.open if p.suffix == ".xz" else gzip.open
    with opener(p, "rt") as fh:
        raw = json.load(fh)
    targets = {}
    for name, t in raw["targets"].items():
        targets[name] = Target(
            name=name, label=t["label"], symbol=t["symbol"], group=t["group"], task=t["task"],
            units=t["units"], description=t["description"], transform=t["transform"],
            floor=t["floor"], metrics=t["metrics"], n_train=t["n_train"], n_test=t["n_test"],
            importance=t["importance"], test_y=t["test"]["y"], test_p=t["test"]["p"],
            test_formula=t["test"]["formula"], booster=lgb.Booster(model_str=t["booster"]),
            features=t.get("features", raw["features"]),
            importance_kind=t.get("importance_kind", "share of split gain"),
            seed_metrics={k: (v[0], v[1]) for k, v in t.get("metrics_seeds", {}).items()})
    dataset = raw.get("dataset")
    if dataset:
        import numpy as np
        dataset = dict(dataset, X=np.array(dataset["X"], dtype=float),
                       y={k: np.array(v, dtype=float) for k, v in dataset["y"].items()})
    return Bundle(name=raw["name"], path=p, created=raw["created"], split=raw["split"],
                  data=raw["data"], features=raw["features"],
                  feature_range={k: (v[0], v[1]) for k, v in raw["feature_range"].items()},
                  targets=targets, dataset=dataset)


def _value(t: Target, z: float) -> float:
    if t.task == "classification":
        return z
    y = 10.0 ** z if t.transform == "log10" else math.expm1(z) if t.transform == "log1p" else z
    return max(y, t.floor) if t.floor is not None else y


def predict(feats: Mapping[str, float], bundle: Optional[Bundle] = None) -> dict[str, float]:
    """Each property's prediction for one material: the value in the target's units, or for a
    classification the probability of the positive class. NaN descriptors are allowed."""
    import numpy as np
    b = bundle or load()
    pred = {}
    for name, t in b.targets.items():
        x = np.array([[float(feats.get(f, math.nan)) for f in (t.features or b.features)]])
        pred[name] = _value(t, float(t.booster.predict(x)[0]))
    return pred


def outside_range(feats: Mapping[str, float], bundle: Optional[Bundle] = None) -> list[str]:
    """Descriptors outside the 1st-99th percentile of the training materials."""
    b = bundle or load()
    out = []
    for f, (lo, hi) in b.feature_range.items():
        v = float(feats.get(f, math.nan))
        if math.isfinite(v) and lo < hi and not lo <= v <= hi:
            out.append(f)
    return out
