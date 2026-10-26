"""The Properties view's table and figures: property models (``pydemi.props``) applied to
the descriptors of the open density."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pydemi import props
from pydemi.gui.plots import ACCENT, ELEMENT_COLORS, MUTED

UNIT_TEXT = {"1": "", "0/1": "", "W/(m K)": "W/(m·K)"}
CLASS_WORDS = {"is_metal": ("Metal", "Not metal"), "is_gap_direct": ("Direct", "Indirect")}
# trained on materials with a band gap only: not meaningful for a material predicted metallic
GAPPED_ONLY = ("is_gap_direct", "dielectric_total", "dielectric_electronic", "dielectric_ionic",
               "refractive_index")


def counts(meta: dict) -> dict[str, int]:
    """Sites of each element, from the descriptors' metadata ("Mg:4,O:4")."""
    pairs = (x.split(":") for x in str(meta.get("site_counts", "")).split(",") if ":" in x)
    return {e: int(n) for e, n in pairs}


def inputs_text(bundle: props.Bundle) -> str:
    """What the models read, for the caption under the table."""
    kinds = {t.inputs for t in bundle.targets.values()}
    if kinds == {"descriptors"}:
        return f"the {len(bundle.features)} descriptors"
    return f"the {len(bundle.features)} descriptors and the composition (Magpie)"


def feature_text(f: str) -> str:
    """A short name for a model input: Magpie columns become "Electronegativity (mean)"."""
    if f.startswith("MagpieData "):
        _, stat, prop = f.split(" ", 2)
        return f"{prop} ({stat.replace('_', ' ')})"
    if f.endswith("-norm"):
        return f"stoichiometry {f}"
    return f


def is_composition(f: str) -> bool:
    return f.startswith("MagpieData ") or f.endswith("-norm")


def metallic(pred: dict[str, float]) -> bool:
    return pred.get("is_metal", 0.0) >= 0.5


def applies(t: props.Target, pred: dict[str, float]) -> bool:
    return not (t.name in GAPPED_ONLY and metallic(pred))


def units(t: props.Target) -> str:
    return UNIT_TEXT.get(t.units, t.units)


def decimals(t: props.Target) -> int:
    """Digits after the point: one past the leading digit of the test MAE."""
    e = t.error
    if not e or not math.isfinite(e):
        return 3
    return max(0, 1 - math.floor(math.log10(e)))


def value_text(t: props.Target, v: float) -> str:
    if t.task == "classification":
        yes, no = CLASS_WORDS.get(t.name, ("Yes", "No"))
        sure = 100 * max(v, 1 - v)
        return f"{yes if v >= 0.5 else no} ({'> 99' if sure > 99 else f'{sure:.0f}'} %)"
    return f"{v:,.{decimals(t)}f}"


def error_text(t: props.Target) -> str:
    return "" if t.error is None else f"± {t.error:,.{decimals(t)}f}"


def title(t: props.Target, bundle: props.Bundle) -> str:
    """The property's name, with its symbol when other properties share the label."""
    shared = sum(o.label == t.label for o in bundle.targets.values()) > 1
    return f"{t.label} {t.symbol}" if shared else t.label


def dft_text(t: props.Target, v: float) -> str:
    if not math.isfinite(v):
        return "–"
    if t.task == "classification":
        return CLASS_WORDS.get(t.name, ("Yes", "No"))[0 if v >= 0.5 else 1]
    return f"{v:,.{decimals(t)}f}"


WEAK_R2 = 0.5
"""Regression models with a test R² below this are marked weak: their values are rough indications."""


def weak(t: props.Target) -> bool:
    """A regression model whose test R² (mean over the seeds when known) is below WEAK_R2."""
    if t.task != "regression":
        return False
    r2 = t.seed_metrics["r2"][0] if "r2" in t.seed_metrics else t.metrics.get("r2", 1.0)
    return r2 < WEAK_R2


def table(pred: dict[str, float], bundle: props.Bundle,
          match: Optional[props.Match] = None) -> pd.DataFrame:
    """One row per property, in the bundle's order; with a DFT column for a dataset material."""
    rows = []
    for name, t in bundle.targets.items():
        score, s = t.score
        row = {"property": title(t, bundle), "group": t.group,
               "prediction": value_text(t, pred[name]) if applies(t, pred) else "n/a (metal)"}
        if match is not None:
            row["DFT"] = dft_text(t, match.values.get(name, math.nan))
        rows.append({**row, "typical error": error_text(t), "units": units(t),
                     "test score": f"{'AUC' if score == 'ROC-AUC' else score} {s:.2f}"
                                   + (" ⚠" if weak(t) else ""), "name": name})
    return pd.DataFrame(rows)


SPLIT_TEXT = {"train": "training set: the model has seen it, so the agreement is not a test",
              "val": "validation set: used to stop training, not to fit",
              "test": "test set: the model has not seen it"}


def match_text(m: props.Match) -> str:
    return f"In the dataset as {m.material_id} ({m.formula}), {SPLIT_TEXT.get(m.split, m.split)}."


def export(pred: dict[str, float], bundle: props.Bundle) -> pd.DataFrame:
    """Plain numbers for the CSV download."""
    return pd.DataFrame([{"property": name, "value": pred[name],
                          "kind": "probability" if t.task == "classification" else "value",
                          "units": t.units, "test_mae": t.error,
                          "test_r2": t.metrics.get("r2"), "test_roc_auc": t.metrics.get("roc_auc"),
                          "inputs": t.inputs, "weak_model": weak(t), "n_train": t.n_train,
                          "model": bundle.name}
                         for name, t in bundle.targets.items()])


def _layout(fig: go.Figure, title: str, x: str, y: str, height: int = 360) -> go.Figure:
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, height=height,
                      margin=dict(l=60, r=20, t=50, b=50), showlegend=False)
    return fig


def parity(t: props.Target, value: float) -> go.Figure:
    """Test-set predictions against the DFT values, with this material's prediction marked."""
    y, p = np.asarray(t.test_y), np.asarray(t.test_p)
    log = t.transform == "log10"
    lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
    if log:
        lo = max(lo, 1e-3)
    fig = go.Figure()
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=MUTED, dash="dot"),
                    hoverinfo="skip")
    fig.add_scatter(x=y, y=p, mode="markers", marker=dict(size=5, color=ELEMENT_COLORS[0], opacity=0.55),
                    text=t.test_formula, hovertemplate="%{text}<br>DFT %{x:.4g}<br>model %{y:.4g}<extra></extra>")
    u = units(t)
    # on a log axis plotly takes shapes in data units but annotations in log10 units
    fig.add_hline(y=value, line_dash="dash", line_color=ACCENT)
    fig.add_annotation(x=0, xref="paper", y=math.log10(max(value, 1e-12)) if log else value,
                       text=f"this material: {value_text(t, value)} {u}".strip(), showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(color=ACCENT))
    suffix = f" ({u})" if u else ""
    _layout(fig, f"Test set: {t.n_test} materials, MAE {error_text(t)[2:]}{' ' + u if u else ''}, "
                 f"R² {t.metrics['r2']:.3f}", f"DFT{suffix}", f"Model{suffix}")
    if log:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    return fig


def probabilities(t: props.Target, value: float) -> go.Figure:
    """Test-set probabilities by true class, with this material's probability marked."""
    y, p = np.asarray(t.test_y), np.asarray(t.test_p)
    yes, no = CLASS_WORDS.get(t.name, ("Yes", "No"))
    fig = go.Figure()
    for cls, label, color in ((0, no, ELEMENT_COLORS[0]), (1, yes, ELEMENT_COLORS[1])):
        fig.add_histogram(x=p[y == cls], name=f"{label} (DFT)", marker_color=color, opacity=0.7,
                          xbins=dict(start=0, end=1, size=0.05))
    fig.add_vline(x=value, line_dash="dash", line_color=ACCENT,
                  annotation_text=f"this material: {100 * value:.0f} %",
                  annotation_position="top left" if value > 0.5 else "top right",
                  annotation_font_color=ACCENT)
    _layout(fig, f"Test set: {t.n_test} materials, ROC-AUC {t.metrics['roc_auc']:.3f}, "
                 f"accuracy {100 * t.metrics['accuracy']:.0f} %",
            f"Predicted probability ({yes.lower()})", "Materials")
    fig.update_layout(barmode="overlay", showlegend=True,
                      legend=dict(orientation="h", yanchor="top", y=0.97, x=0.5, xanchor="center",
                                  bgcolor="rgba(0,0,0,0)"))
    return fig


def importance(t: props.Target, top: int = 10) -> go.Figure:
    """The inputs the model leans on most, descriptors and composition features told apart."""
    items = list(t.importance.items())[:top][::-1]
    fig = go.Figure()
    for kind, color, comp in (("descriptor", ELEMENT_COLORS[2], False), ("composition", ELEMENT_COLORS[0], True)):
        sel = [(k, v) for k, v in items if is_composition(k) == comp]
        if sel:
            fig.add_bar(x=[100 * v for _, v in sel], y=[feature_text(k) for k, _ in sel], orientation="h",
                        marker_color=color, name=kind, hovertemplate="%{y}: %{x:.1f} %<extra></extra>")
    fig.update_yaxes(categoryorder="array", categoryarray=[feature_text(k) for k, _ in items])
    what = "mean |SHAP|" if "SHAP" in t.importance_kind else "split gain"
    _layout(fig, f"Top {len(items)} inputs", f"Share of {what} (%)", "", height=40 + 26 * len(items) + 60)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.25, x=0),
                      height=fig.layout.height + 30)
    return fig


def model_rows(t: props.Target, pred: dict[str, float], bundle: props.Bundle,
               match: Optional[props.Match] = None) -> list[tuple[str, str]]:
    u, value = units(t), pred[t.name]
    rows = [("Property", title(t, bundle)), ("Symbol", t.symbol)]
    if applies(t, pred):
        rows.append(("Prediction", f"{value_text(t, value)} {u}".strip()))
    else:
        rows += [("Prediction", "n/a (predicted metallic)"),
                 ("Model output", f"{value_text(t, value)} {u}".strip())]
    if match is not None:
        rows.append((f"DFT ({match.material_id})",
                     f"{dft_text(t, match.values.get(t.name, math.nan))} {u}".strip()))
    if t.task == "regression":
        rows += [("Typical error", f"{error_text(t)} {u} (test MAE)".strip()),
                 ("Median error", f"{t.metrics['median_ae']:,.{decimals(t)}f} {u}".strip()),
                 ("Test R²", f"{t.metrics['r2']:.3f}")]
    else:
        rows += [("Test ROC-AUC", f"{t.metrics['roc_auc']:.3f}"),
                 ("Test accuracy", f"{100 * t.metrics['accuracy']:.1f} %"),
                 ("Test F1", f"{t.metrics['f1']:.3f}")]
    key = "r2" if t.task == "regression" else "roc_auc"
    if key in t.seed_metrics:
        m, sd = t.seed_metrics[key]
        rows.append((f"{'R²' if key == 'r2' else 'ROC-AUC'}, 5 seeds",
                     f"{m:.3f} ± {sd:.3f}" if sd >= 5e-4 else f"{m:.3f} (spread < 0.001)"))
    n_in = len(t.features) if t.features else len(bundle.features)
    rows += [("Inputs", f"{t.inputs} ({n_in})"),
             ("Trained on", f"{t.n_train:,} materials"), ("Tested on", f"{t.n_test:,} materials"),
             ("Model", bundle.name)]
    return rows


def note(t: props.Target, pred: dict[str, float]) -> Optional[str]:
    """A caveat that applies to this prediction, if any."""
    if not applies(t, pred):
        return ("Predicted metallic. This model was trained on materials with a band gap only, "
                "so its value does not apply here.")
    if t.name == "band_gap" and metallic(pred):
        return "Predicted metallic; the gap model is trained on metals too (gap 0)."
    if t.name == "cohesive_energy":
        return "Less reliable for lanthanide and actinide compounds."
    if weak(t):
        return (f"Weak model (test R² below {WEAK_R2}): read the value as a rough indication, "
                "not a prediction to rely on.")
    return None
