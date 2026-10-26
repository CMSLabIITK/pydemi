"""The Magpie element table that pydemi.props.composition_features reads (from matminer), and
a check that it reproduces data/composition.csv for every material.

    python scripts/magpie_table.py     # writes data/magpie_table.json, prints the largest difference
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition, Element

from pydemi.props import composition_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
P = (0, 2, 3, 5, 7, 10)


def table() -> dict:
    ep = ElementProperty.from_preset("magpie")
    elements = {}
    for z in range(1, 104):
        e = Element.from_Z(z)
        vals = [ep.data_source.get_elemental_property(e, prop) for prop in ep.features]
        elements[e.symbol] = [None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
                              for v in vals]
    return {"source": "matminer MagpieData (Ward et al., npj Comput. Mater. 2, 16028, 2016)",
            "p": list(P), "properties": list(ep.features), "elements": elements}


def main() -> None:
    warnings.simplefilter("ignore")
    t = table()
    (DATA / "magpie_table.json").write_text(json.dumps(t))
    d = pd.read_csv(DATA / "dataset.csv", usecols=["structure", "formula"])
    c = pd.read_csv(DATA / "composition.csv").set_index("structure")
    worst, where = 0.0, None
    for s, f in zip(d.structure, d.formula):
        counts = Composition(f).get_el_amt_dict()
        mine = composition_features(counts, t)
        ref = c.loc[s]
        for k, v in mine.items():
            r = float(ref[k])
            diff = 0.0 if (math.isnan(v) and math.isnan(r)) else abs(v - r) / max(1.0, abs(r))
            if not diff <= worst:
                worst, where = diff, (s, k, v, r)
    print(f"{len(d)} materials, {len(c.columns) - 2} features: largest relative difference {worst:.2e}", where)


if __name__ == "__main__":
    main()
