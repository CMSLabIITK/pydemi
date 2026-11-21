"""Figures and example outputs of the documentation, from real VASP CHGCARs.

Run once, with the dataset at hand (the files are committed; the docs build does not run it):

    python docs/tools/make_figures.py /path/to/6000_data_aug13

Writes docs/_generated/: one interactive plotly fragment per descriptor (the same figure the
graphical interface shows), the structure and a density slice of the main example, and the
descriptor values and property estimates quoted in the tutorials.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import plotly.io as pio

import pydemi
from pydemi import props
from pydemi.gui import plots, viewer

OUT = Path(__file__).resolve().parents[1] / "_generated"
MAIN = "GaAs_216"       # bonding and structural descriptors
HET = "SbO2_15"         # heterogeneity descriptors: two inequivalent Sb sites (Sb3+ and Sb5+)


def fragment(fig, path: Path) -> None:
    fig.update_layout(autosize=True, height=fig.layout.height or 430, template="plotly_white")
    html = pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"displaylogo": False, "responsive": True})
    # plotly.js is loaded by the page; draw once it is there
    html = html.replace('<script>',
                        '<script>window.addEventListener("load", function () {')
    path.write_text(html.replace("</script>", "});</script>"))


def run(root: Path, name: str) -> tuple[dict, dict]:
    vd = pydemi.read_chgcar(root / name / "CHGCAR")
    t = time.perf_counter()
    feats, meta = pydemi.featurize(vd, return_metadata=True)
    sec = time.perf_counter() - t
    data = plots.plot_data(vd)
    info = {"structure": name,
            "grid": list(vd.shape), "atoms": len(vd.structure.species), "seconds": round(sec, 2)}
    return {"vd": vd, "feats": feats, "data": data, "meta": meta}, info


def main(root: Path) -> None:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    het = set(pydemi.descriptor_names("heterogeneity"))
    summary = {}
    for name, which in [(MAIN, [n for n in pydemi.descriptor_names() if n not in het]),
                        (HET, sorted(het))]:
        r, info = run(root, name)
        for d in which:
            fig, cap = plots.figure(d, r["data"], r["feats"])
            fragment(fig, fig_dir / f"{d}.html")
            (fig_dir / f"{d}.txt").write_text(f"{cap}\n")
        summary[name] = {**info, "features": {k: float(v) for k, v in r["feats"].items()},
                         "flags": {k: str(v) for k, v in r["meta"].items() if k != "options"}}
        if props.missing() == []:
            summary[name]["properties"] = {k: float(v) for k, v in props.predict(r["feats"]).items()}
            m = props.load().match(r["feats"])
            if m is not None:
                summary[name]["dataset"] = {"material_id": m.material_id, "split": m.split,
                                            "dft": m.values}
        if name == MAIN:
            p = viewer.preview(r["vd"])
            fragment(viewer.structure_figure(p), fig_dir / "_structure.html")
            fig, _ = viewer.slice_figure(p, 2, 0.0, True)
            fragment(fig, fig_dir / "_slice.html")
    (OUT / "examples.json").write_text(json.dumps(summary, indent=1, default=str))
    if props.missing() == []:
        rows = []
        for t in props.load().targets.values():
            kind, _ = t.score       # the score of the paper: mean over the five seeds
            score = t.seed_metrics["roc_auc" if t.task == "classification" else "r2"][0]
            rows.append({"name": t.name, "label": t.label, "symbol": t.symbol, "group": t.group,
                         "task": t.task, "units": t.units, "n_train": t.n_train, "n_test": t.n_test,
                         "metric": kind, "score": score,
                         "mae": None if t.task == "classification" else round(t.seed_metrics["mae"][0], 4)})
        import pandas as pd
        pd.DataFrame(rows).to_csv(OUT / "property_models.csv", index=False)
    cat = pydemi.catalogue().drop(columns="references")
    cat.to_csv(OUT / "catalogue.csv", index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
