"""The Compare view's tables: the DFT CHGCAR of a structure against the density ChargE3Net
predicts for the same structure on the same grid (descriptors, and the properties
predicted from each)."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import pydemi
from pydemi import props
from pydemi.gui import properties as P
from pydemi.io.base import VolumetricData

TRANSFER = Path(__file__).with_name("transfer.csv")
NOISE = 1e-3     # below this fraction of a descriptor's typical size in both densities: noise
CLOSE = 0.01     # relative difference counted as agreement
FAR = 0.05       # relative difference marked in the table
INPUT_TEXT = {"descriptors + composition": "D + C", "descriptors": "D", "composition": "C"}


@lru_cache(maxsize=1)
def transfer() -> pd.DataFrame:
    """Per descriptor: Spearman correlation between DFT and predicted values over the 605
    held-out structures of the density model, and the typical size of its values."""
    return pd.read_csv(TRANSFER, comment="#").set_index("descriptor")


def grade(rho: float) -> str:
    if not math.isfinite(rho):
        return "constant"
    return "excellent" if rho >= 0.99 else "good" if rho >= 0.95 else "fair" if rho >= 0.8 else "poor"


def relative(a: float, b: float) -> float:
    """|a - b| / max(|a|, |b|): 0 to 1, symmetric, as in the descriptor stability analysis."""
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.nan
    d = max(abs(a), abs(b))
    return abs(a - b) / d if d > 0 else 0.0


def density_error(dft: VolumetricData, pred: VolumetricData) -> float:
    """NMAPE (%) of the predicted density against the DFT one on the same grid."""
    a, b = dft.rho.data, pred.rho.data
    if a.shape != b.shape:
        raise ValueError(f"grids differ: {a.shape} and {b.shape}")
    return float(100 * np.abs(b - a).sum() / np.abs(a).sum())


def descriptors(dft: dict, pred: dict) -> pd.DataFrame:
    """One row per descriptor: both values, their relative difference and how well the
    descriptor transfers to predicted densities in general."""
    cat = pydemi.catalogue().set_index("name")
    tr = transfer()
    rows = []
    for name in pydemi.descriptor_names():
        a, b = float(dft[name]), float(pred[name])
        rho = float(tr.at[name, "spearman"]) if name in tr.index else math.nan
        scale = float(tr.at[name, "scale"]) if name in tr.index else math.nan
        g = grade(rho)
        if g == "constant":
            diff, note = math.nan, "constant"
        elif scale > 0 and max(abs(a), abs(b)) < NOISE * scale:
            diff, note = math.nan, "≈ 0 in both"
        else:
            diff = relative(a, b)
            note = "" if not math.isfinite(diff) else "differs" if diff > FAR else ""
        rows.append({"descriptor": name, "domain": cat.at[name, "domain"], "DFT": a, "ML": b,
                     "difference %": 100 * diff if math.isfinite(diff) else math.nan, "note": note,
                     "transfer": g if g == "constant" else f"{g} (ρs {rho:.3f})"})
    return pd.DataFrame(rows)


def descriptor_summary(table: pd.DataFrame) -> dict[str, float]:
    compared = table[table["note"] != "constant"]
    d = compared["difference %"]
    close = int((d <= 100 * CLOSE).sum() + (compared["note"] == "≈ 0 in both").sum())
    return {"median": float(d.median()), "close": close, "n": len(compared),
            "far": int((d > 100 * FAR).sum())}


def properties(dft: dict[str, float], pred: dict[str, float], bundle: props.Bundle,
               match: Optional[props.Match] = None) -> pd.DataFrame:
    """One row per property: the prediction from each density and whether they agree within
    the model's typical error (the same class, for the yes/no properties)."""
    rows = []
    for name, t in bundle.targets.items():
        a, b = dft[name], pred[name]
        ok_a, ok_b = P.applies(t, dft), P.applies(t, pred)
        if t.task == "classification":
            same = (a >= 0.5) == (b >= 0.5)
            diff, agree = ("same class" if same else "class differs"), same
        elif ok_a and ok_b:
            d = b - a
            diff = f"{d:+,.{P.decimals(t)}f}"
            if float(diff.replace(",", "")) == 0:
                diff = "0"
            agree = t.error is None or abs(d) <= t.error
        else:
            diff, agree = "–", ok_a == ok_b
        row = {"property": P.title(t, bundle) + (" ⚠" if P.weak(t) else ""), "group": t.group}
        if match is not None:
            row["DFT value"] = P.dft_text(t, match.values.get(name, math.nan))
        rows.append({**row,
                     "from DFT CHGCAR": P.value_text(t, a) if ok_a else "n/a (metal)",
                     "from ML CHGCAR": P.value_text(t, b) if ok_b else "n/a (metal)",
                     "ML − DFT": diff, "agree": agree, "typical error": P.error_text(t),
                     "units": P.units(t), "inputs": INPUT_TEXT.get(t.inputs, t.inputs),
                     "weak": P.weak(t), "name": name})
    return pd.DataFrame(rows)


def export(desc: pd.DataFrame, prop: Optional[pd.DataFrame], dft: dict, pred: dict,
           pdft: Optional[dict], ppred: Optional[dict]) -> pd.DataFrame:
    """Plain numbers of both inputs for the CSV download."""
    rows = [{"kind": "descriptor", "name": n, "dft": dft[n], "ml": pred[n], "relative_difference": d / 100}
            for n, d in zip(desc["descriptor"], desc["difference %"])]
    if prop is not None and pdft is not None and ppred is not None:
        rows += [{"kind": "property", "name": n, "dft": pdft[n], "ml": ppred[n],
                  "relative_difference": relative(pdft[n], ppred[n])} for n in prop["name"]]
    return pd.DataFrame(rows)
