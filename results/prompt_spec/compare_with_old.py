"""
Compare the prompt.md rerun with the earlier (PDF-spec) run, descriptor by descriptor.

For each descriptor name present in both tables: rows matched by run
directory, Spearman rank correlation, median |relative difference| and the
fraction of rows that agree to 1e-6. Definitions changed for several names
(per-volume counts, volume-normalized information measures, percolation
threshold, FFT derivatives, ZVAL table), so disagreement is expected there;
the table shows which names kept their meaning.

Usage:  python results/prompt_spec/compare_with_old.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "descriptors_6000_data_aug13.csv"
NEW = HERE / "descriptors_6000_data_aug13.csv"
SKIP = {"path", "error", "material_id", "n_atoms", "n_elements"}


def main() -> None:
    old = pd.read_csv(OLD, low_memory=False)
    new = pd.read_csv(NEW, low_memory=False)
    old["run"] = old["path"].astype(str).str.rstrip("/").str.split("/").str[-1]     # run directory
    new["run"] = new["path"].astype(str).str.split("/").str[-2]                      # .../run/CHGCAR
    m = old.merge(new, on="run", suffixes=("_old", "_new"))
    rows = []
    for c in sorted((set(old.columns) & set(new.columns)) - SKIP - {"run"}):
        a = pd.to_numeric(m[f"{c}_old"], errors="coerce")
        b = pd.to_numeric(m[f"{c}_new"], errors="coerce")
        ok = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 10:
            continue
        a, b = a[ok], b[ok]
        rel = (a - b).abs() / np.maximum(np.maximum(a.abs(), b.abs()), 1e-12)
        rows.append({"descriptor": c, "n": int(ok.sum()),
                     "spearman": a.rank().corr(b.rank()),
                     "median_rel_diff": float(rel.median()),
                     "frac_equal_1e-6": float((rel < 1e-6).mean())})
    out = pd.DataFrame(rows).sort_values("spearman")
    out.to_csv(HERE / "comparison_with_old.csv", index=False)
    print(f"{len(m)} runs matched; {len(out)} shared descriptors")
    with pd.option_context("display.width", 140, "display.max_rows", 200):
        print(out.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
