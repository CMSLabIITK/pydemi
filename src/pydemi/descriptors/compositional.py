"""
pydemi.descriptors.compositional
================================
Compositional baseline (spec §8.5): a thin wrapper over

    matminer.featurizers.composition.ElementProperty.from_preset("magpie")

Nothing is reimplemented. Every feature is registered with
``domain="compositional"`` and ``adopted=True`` so it can be kept out of
novelty claims and used as an explicit baseline. Names are matminer's labels
with the "MagpieData " prefix replaced by "magpie_" and spaces by "_"
(e.g. ``magpie_mean_Electronegativity``).

The labels are read from ``data/magpie_labels.txt`` (generated from matminer)
so that the catalogue does not need matminer; computing the features does
(``pip install pydemi[full]``). A feature matminer returns as NaN (element
data missing from the Magpie tables) is reported as NaN with its companion
flag set.
"""

from __future__ import annotations

import numpy as np

from ..data import DATA_DIR
from ..io.base import VolumetricData
from .registry import Result, Sentinel, register

_REFERENCE = ("L. Ward, A. Agrawal, A. Choudhary, C. Wolverton, npj Comput. Mater. 2, 16028 "
              "(2016); matminer: L. Ward et al., Comput. Mater. Sci. 152, 60 (2018)")


def _labels() -> list[str]:
    return [l for l in (DATA_DIR / "magpie_labels.txt").read_text().splitlines()
            if l and not l.startswith("#")]


def _name(label: str) -> str:
    return "magpie_" + label.replace("MagpieData ", "").replace(" ", "_")


LABELS = _labels()


def magpie_features(vd: VolumetricData) -> dict[str, float]:
    """All Magpie features of ``vd``'s composition (cached)."""
    key = ("magpie",)
    if key not in vd.cache:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from matminer.featurizers.composition import ElementProperty  # type: ignore[import-untyped]
                from pymatgen.core.composition import Composition
        except ImportError as exc:
            raise ImportError("the compositional domain needs matminer and pymatgen: "
                              "pip install 'pydemi[full]'") from exc
        ep = ElementProperty.from_preset("magpie")
        if list(ep.feature_labels()) != LABELS:
            raise RuntimeError("installed matminer's Magpie labels differ from "
                               "pydemi/data/magpie_labels.txt; regenerate the label file")
        counts = vd.structure.site_counts()
        values = ep.featurize(Composition(counts))
        vd.cache[key] = {lab: float(v) for lab, v in zip(LABELS, values)}
    out: dict[str, float] = vd.cache[key]
    return out


def _register(label: str) -> None:
    def fn(vd: VolumetricData) -> Result:
        v = magpie_features(vd)[label]
        return v if np.isfinite(v) else Sentinel(float("nan"), "missing_element_data")
    stat, _, prop = label.replace("MagpieData ", "").partition(" ")
    register(name=_name(label), domain="compositional", field="composition",
             requires=["composition"], units="(Magpie)", adopted=True,
             sentinel_cases={"missing_element_data": float("nan")}, references=[_REFERENCE],
             doc=f"{_name(label)} = {stat} over the composition of the elemental {prop}\n\n"
                 "matminer ElementProperty.from_preset('magpie') (adopted baseline, not "
                 "reimplemented).")(fn)


for _label in LABELS:
    _register(_label)


def magpie_names() -> list[str]:
    return [_name(l) for l in LABELS]


__all__ = ["magpie_features", "magpie_names", "LABELS"]
