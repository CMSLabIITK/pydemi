"""The structure-only ML path against the DFT-grid predictions of Section 7.

For test structures written out as CIF files, ChargE3Net predicts the density
on the grid chosen by pydemi.vasp_grid_shape (no DFT input;
~/charge3net/charge3net/scripts/predict_from_cif.sh), and pydemi reads the
result with read_predicted. This script compares, per structure:
  - the grid with the DFT grid;
  - the density with the fine-tuned model's full-grid test prediction on the
    DFT grid (same model, same points if the grids agree);
  - the descriptors (renormalized, as n_ml_descriptors.py --renorm) with those
    of the DFT-grid prediction and of the DFT density.

Usage:  python p_cif_path.py NPZ_DIR   -> out/cif_path.csv, out/cif_path_summary.txt, ../si/cif_path_table.tex
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pydemi

from common import OUT, READ_OPTIONS

PRED_GRID = Path("/home/shubham/charge3net/charge3net/data/test_set_predictions_finetune/cubes")
DOMAINS = ["bonding", "structural", "heterogeneity"]
EXCLUDE = {"lap_concentration", "def_out_volume_fraction"}      # not scored (o_ml_analysis.py)


def main(npz_dir: Path) -> None:
    ref_ml = pd.read_csv(OUT / "ml" / "desc_finetune_renorm.csv").set_index("name")
    ref_dft = pd.read_csv(OUT / "ml" / "desc_dft.csv").set_index("name")
    scores = pd.read_csv(OUT / "ml_scores_finetune.csv").set_index("descriptor")
    rows, lines = [], []
    for p in sorted(npz_dir.glob("*.npz")):
        name = p.stem
        vd = pydemi.read_predicted(p, **READ_OPTIONS)
        old = np.load(PRED_GRID / f"{name}.npy")
        same_grid = tuple(old.shape) == vd.rho.shape
        raw = vd.rho.data / float(vd.sources["charge_scale"])
        d_rho = float(np.abs(raw - old).max() / np.abs(old).max()) if same_grid else np.nan
        feats = pydemi.featurize(vd, domains=DOMAINS, extensions=["paw"])
        names = [n for n in feats if n in ref_ml.columns and n in scores.index and n not in EXCLUDE]
        a = pd.Series({n: feats[n] for n in names})
        b, c = ref_ml.loc[name, names].astype(float), ref_dft.loc[name, names].astype(float)

        def rel(x, y):
            return ((x - y).abs() / np.maximum(np.maximum(x.abs(), y.abs()), 1e-12)).fillna(0.0)
        transferable = [n for n in names if scores.loc[n, "spearman"] >= 0.95]
        r_ml, r_dft = rel(a, b), rel(a, c)
        tr = r_ml[transferable]
        rows.append({"name": name, "grid": "x".join(map(str, vd.rho.shape)), "same_grid_as_dft": same_grid,
                     "median_rel_desc_diff_vs_dft_grid_prediction_transferable": tr.median(),
                     "p90_rel_desc_diff_vs_dft_grid_prediction_transferable": tr.quantile(0.9),
                     "max_rel_density_diff_vs_dft_grid_prediction": d_rho,
                     "charge_scale": float(vd.sources["charge_scale"]),
                     "max_rel_desc_diff_vs_dft_grid_prediction": r_ml.max(),
                     "median_rel_desc_err_vs_dft_transferable": r_dft[transferable].median(),
                     "median_rel_desc_err_vs_dft_grid_prediction_transferable": rel(b, c)[transferable].median()})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cif_path.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        lines.append(df.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    lines.append(f"\n{len(df)} structures; grids identical to DFT: {int(df.same_grid_as_dft.sum())}; "
                 f"largest density difference vs the DFT-grid prediction: "
                 f"{df.max_rel_density_diff_vs_dft_grid_prediction.max():.2e} (relative to max rho); "
                 f"largest descriptor difference: {df.max_rel_desc_diff_vs_dft_grid_prediction.max():.2e}\n"
                 f"over the descriptors with Spearman >= 0.95 (fine-tuned): median difference between the two "
                 f"paths {df.median_rel_desc_diff_vs_dft_grid_prediction_transferable.median():.2e} (largest "
                 f"per-structure median {df.median_rel_desc_diff_vs_dft_grid_prediction_transferable.max():.2e}), "
                 f"against a median error vs DFT of {df.median_rel_desc_err_vs_dft_transferable.median():.2e}")
    sci = lambda v: f"{v:.1e}".replace("e-0", "e-").replace("e-", r"\times10^{-") + "}"
    tex = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{The path from a structure alone for eight test structures written out as CIF files: "
           r"grid chosen by \texttt{vasp\_grid\_shape} (identical to the DFT grid in every case), charge "
           r"rescaling factor, largest density difference from the fine-tuned model's prediction on the DFT "
           r"grid (relative to the density maximum), median and 90th-percentile relative difference of the "
           r"%d descriptors with $\rho_s\geq0.95$ between the two paths, and their median relative error "
           r"against DFT (\texttt{paper/analysis/p\_cif\_path.py}).}" % len(transferable),
           r"\label{tab:si-cif}", r"\small", r"\begin{tabular}{llrrrrr}", r"\toprule",
           r"Structure & Grid & Scale & $\max|\Delta\rho|$ & Median & p90 & Error vs DFT \\", r"\midrule"]
    for _, r in df.iterrows():
        nm = r["name"].rsplit("_", 1)
        formula = "".join(f"$_{{{c}}}$" if c.isdigit() else c for c in nm[0])
        tex.append(r"%s (%s) & %s & %.4f & $%s$ & $%s$ & $%s$ & %.2f\%% \\" % (
            formula, nm[1], r["grid"].replace("x", r"$\times$"), r["charge_scale"],
            sci(r["max_rel_density_diff_vs_dft_grid_prediction"]),
            sci(r["median_rel_desc_diff_vs_dft_grid_prediction_transferable"]),
            sci(r["p90_rel_desc_diff_vs_dft_grid_prediction_transferable"]),
            100 * r["median_rel_desc_err_vs_dft_transferable"]))
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (Path(__file__).resolve().parents[1] / "si" / "cif_path_table.tex").write_text("\n".join(tex) + "\n")
    text = "\n".join(lines)
    (OUT / "cif_path_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
