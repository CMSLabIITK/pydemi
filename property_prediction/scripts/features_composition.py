"""Composition-only features for the baseline: Magpie element statistics (132) and the
stoichiometry p-norms (6) of matminer, from each material's formula.

Writes data/composition.csv (structure, material_id, split + 138 columns) and
data/composition_columns.csv.

    nice -n 10 python scripts/features_composition.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matminer
import pandas as pd
from matminer.featurizers.composition import ElementProperty, Stoichiometry
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.base import MultipleFeaturizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    warnings.simplefilter("ignore")
    d = pd.read_csv(DATA / "dataset.csv", usecols=["structure", "material_id", "formula", "split"])
    c = StrToComposition(target_col_id="composition").featurize_dataframe(d, "formula", pbar=False)
    feat = MultipleFeaturizer([Stoichiometry(), ElementProperty.from_preset("magpie")])
    feat.set_n_jobs(1)
    c = feat.featurize_dataframe(c, "composition", pbar=False)
    labels = feat.feature_labels()
    out = c[["structure", "material_id", "split"] + labels]
    out.to_csv(DATA / "composition.csv", index=False)
    pd.DataFrame({"column": labels,
                  "source": [f"matminer {matminer.__version__} "
                             + ("Stoichiometry" if i < 6 else "ElementProperty magpie")
                             for i in range(len(labels))]}).to_csv(DATA / "composition_columns.csv", index=False)
    print(f"{len(out)} materials x {len(labels)} composition features;",
          f"NaN cells: {int(out[labels].isna().sum().sum())}")


if __name__ == "__main__":
    main()
