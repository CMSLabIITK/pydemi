"""Markdown tables of the documentation from docs/_generated (written by make_figures.py).

Called by conf.py; writes docs/_generated/md/*.md, which the pages include. Every number in
them comes from a real run of pydemi on a VASP CHGCAR of the CMS Lab dataset.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

GEN = Path(__file__).resolve().parents[1] / "_generated"
MD = GEN / "md"


def _g(x: float, n: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{n}g}"


def property_models() -> str:
    p = pd.read_csv(GEN / "property_models.csv")
    rows = ["| property | symbol | units | group | train / test | test score | MAE |",
            "|---|---|---|---|---|---|---|"]
    for r in p.itertuples():
        label = r.label if r.label != "Elastic constant" else f"Elastic constant {r.symbol}"
        mae = "–" if pd.isna(r.mae) else _g(r.mae, 3)
        rows.append(f"| {label} | {r.symbol} | {r.units} | {r.group} | {r.n_train} / {r.n_test} | "
                    f"{r.metric} {r.score:.2f} | {mae} |")
    return "\n".join(rows) + "\n"


def gaas_properties() -> str:
    ex = json.loads((GEN / "examples.json").read_text())["GaAs_216"]
    p = pd.read_csv(GEN / "property_models.csv").set_index("name")
    dft = ex.get("dataset", {}).get("dft", {})
    rows = ["| property | estimate | DFT value | units | typical error (test MAE) |",
            "|---|---|---|---|---|"]
    for k, v in ex["properties"].items():
        t = p.loc[k]
        lab = t.label if t.label != "Elastic constant" else f"Elastic constant {t.symbol}"
        if t.task == "classification":
            rows.append(f"| {lab} | probability {v:.2f} | {int(dft[k]) if k in dft and not math.isnan(dft[k]) else '–'} "
                        f"| 0/1 | ROC-AUC {t.score:.2f} |")
        else:
            rows.append(f"| {lab} | {_g(v)} | {_g(dft.get(k))} | {t.units} | {_g(t.mae, 3)} |")
    return "\n".join(rows) + "\n"


def features(struct: str, domain: str | None = None) -> str:
    ex = json.loads((GEN / "examples.json").read_text())[struct]
    cat = pd.read_csv(GEN / "catalogue.csv").set_index("name")
    rows = ["| descriptor | value | units |", "|---|---|---|"]
    for k, v in ex["features"].items():
        if domain and cat.loc[k, "domain"] != domain:
            continue
        rows.append(f"| {{ref}}`{k} <d-{k.replace('_', '-').lower()}>` | {_g(v, 5)} | {cat.loc[k, 'units']} |")
    return "\n".join(rows) + "\n"


def cli_help() -> str:
    import os
    import subprocess
    import sys
    env = {**os.environ, "COLUMNS": "100"}
    out = []
    for cmd in ["", "featurize", "batch", "catalogue", "sweep", "predict", "gui"]:
        args = [sys.executable, "-m", "pydemi.cli"] + ([cmd] if cmd else []) + ["--help"]
        text = subprocess.run(args, capture_output=True, text=True, env=env).stdout.strip()
        out.append(f"### `pydemi{' ' + cmd if cmd else ''}`\n\n```text\n{text}\n```\n")
    return "\n".join(out)


def build() -> None:
    MD.mkdir(parents=True, exist_ok=True)
    (MD / "cli.md").write_text(cli_help())
    (MD / "property_models.md").write_text(property_models())
    (MD / "gaas_properties.md").write_text(gaas_properties())
    (MD / "gaas_features.md").write_text(features("GaAs_216"))
    (MD / "sbo2_heterogeneity.md").write_text(features("SbO2_15", "heterogeneity"))
    ex = json.loads((GEN / "examples.json").read_text())
    if "ml" in ex:
        (MD / "gaas_ml.md").write_text(ml_table(ex["ml"]))
        (MD / "gaas_ml_props.md").write_text(ml_props(ex["ml"]))


def ml_table(ml: dict) -> str:
    rows = ["| descriptor | DFT density | ChargE3Net density | difference | transfer (605 test structures) |",
            "|---|---|---|---|---|"]
    for r in ml["descriptors"]:
        diff = r["note"] if r["diff"] is None else f"{r['diff']:.2g} %" + (" (differs)" if r["note"] else "")
        rows.append(f"| {{ref}}`{r['name']} <d-{r['name'].replace('_', '-').lower()}>` | {_g(r['dft'], 4)} | "
                    f"{_g(r['ml'], 4)} | {diff} | {r['transfer']} |")
    return "\n".join(rows) + "\n"


def ml_props(ml: dict) -> str:
    ex = json.loads((GEN / "examples.json").read_text())["GaAs_216"]
    dft = ex.get("dataset", {}).get("dft", {})
    p = pd.read_csv(GEN / "property_models.csv").set_index("name")
    rows = ["| property | from the DFT density | from the predicted density | DFT value | units |",
            "|---|---|---|---|---|"]
    for k, a in ml["props_dft"].items():
        t = p.loc[k]
        if t.task == "classification":
            continue
        lab = t.label if t.label != "Elastic constant" else f"Elastic constant {t.symbol}"
        rows.append(f"| {lab} | {_g(a)} | {_g(ml['props_ml'][k])} | {_g(dft.get(k))} | {t.units} |")
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    build()
