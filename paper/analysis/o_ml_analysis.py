"""DFT versus ChargE3Net-predicted densities on the 605 test structures (main-text
Section 7, Fig. 8; Supporting Information, Section S6).

Inputs (out/ml/): the descriptors of n_ml_descriptors.py for the DFT density and the
predictions of the from-scratch and the fine-tuned model, raw and renormalized to
N = sum ZVAL; the per-structure grid NMAPE and timings of the ChargE3Net full-grid
tests; the probe NMAPE of the three models (1,000 probes per structure).

Per descriptor and model: median and 90th-percentile relative error over structures,
|a - b| / max(|a|, |b|), Spearman rank correlation and R^2 across structures.
Left out of the statistics, figure and table (kept in the score files):
lap_concentration (identically 1/2 on a periodic grid) and def_out_volume_fraction
(set by the structure and the PAW radii alone).

Usage:  python o_ml_analysis.py  -> out/ml_scores_<model>.csv, out/ml_summary.txt,
                                    ../figures/fig_ml.pdf (+ .png), ../si/ml_table.tex
"""
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUT, REPO  # noqa: E402

ML = OUT / "ml"
FIG = REPO / "paper" / "figures"
SI = REPO / "paper" / "si"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "savefig.bbox": "tight"})
MODELS = {"scratch": "from scratch", "finetune": "fine-tuned"}
SKIP = {"error", "wall_time_s", "N", "N_ref", "euler_consistency", "def_charge_mismatch"}
EXCLUDE = ["lap_concentration", "def_out_volume_fraction"]
F_ELEMENTS = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
              "Ac", "Th", "Pa", "U", "Np", "Pu"}
DOMAIN_COLOR = {"bonding": "C0", "structural": "C2", "heterogeneity": "C4"}
PARITY = ["f_bond", "m1_def", "rho_perc_a", "zeta_site_std", "n_min", "zeta_ELF"]
LOG_PARITY = {"n_min"}

lines = []


def say(s=""):
    lines.append(s)
    print(s)


def rel(a, b):
    return (a - b).abs() / np.maximum(np.maximum(a.abs(), b.abs()), 1e-12)


def score(dft, ml):
    ml = ml.loc[dft.index]
    rows = []
    for c in dft.columns:
        if c in SKIP or c.endswith("__flag") or dft[c].dtype == object:
            continue
        a, b = dft[c].astype(float), ml[c].astype(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 20 or a[ok].std() == 0:
            continue
        a, b = a[ok], b[ok]
        r = rel(a, b)
        rows.append({"descriptor": c, "n": int(ok.sum()), "median_rel": r.median(), "p90_rel": r.quantile(0.9),
                     "spearman": a.rank().corr(b.rank()),
                     "r2": 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()})
    return pd.DataFrame(rows).set_index("descriptor").sort_values("spearman")


def load():
    dft = pd.read_csv(ML / "desc_dft.csv").set_index("name")
    desc = {m + suf: pd.read_csv(ML / f"desc_{m}{suf}.csv").set_index("name")
            for m in MODELS for suf in ("", "_renorm")}
    return dft, desc


def summary(dft, desc, scores):
    grid = {m: pd.read_csv(ML / f"nmape_grid_{m}.csv").set_index("filename")["nmape"] for m in MODELS}
    t = {m: pd.read_csv(ML / f"time_grid_{m}.csv").set_index("filename")["time"] for m in MODELS}
    probes = pd.read_csv(ML / "nmape_probes.csv").set_index("filename")
    say(f"test structures: {len(dft)}")
    for m, lab in MODELS.items():
        g = grid[m]
        say(f"grid NMAPE {lab}: mean {g.mean():.3f}%, median {g.median():.3f}%, p90 {g.quantile(0.9):.3f}%, "
            f"max {g.max():.2f}% ({g.idxmax()}); time/structure median {t[m].median():.1f} s, mean {t[m].mean():.1f} s")
    for c in probes.columns:
        p = probes[c]
        say(f"probe NMAPE {c}: mean {p.mean():.2f}%, median {p.median():.2f}%, max {p.max():.1f}% ({p.idxmax()})")
    dt = dft_time(grid["finetune"].index)
    ch = dt["elapsed_s"] * dt["cores"] / 3600
    say(f"DFT single point (OUTCAR, {len(dt)} test structures): elapsed median {dt.elapsed_s.median():.0f} s on "
        f"median {dt.cores.median():.0f} cores, {ch.median():.2f} core-hours (p90 {ch.quantile(.9):.2f})")
    pt = pd.read_csv(ML / "desc_finetune.csv")["wall_time_s"]
    say(f"pydemi on a predicted density (bonding+structural+heterogeneity, paw): median {pt.median():.1f} s, "
        f"p90 {pt.quantile(.9):.1f} s, one core")
    better = (grid["finetune"] < grid["scratch"]).sum()
    say(f"fine-tuned better than scratch on the grid: {better}/{len(grid['scratch'])}")
    fel = pd.Series({n: any(e in F_ELEMENTS for e in _species(n)) for n in grid["finetune"].index})
    for m, lab in MODELS.items():
        g = grid[m]
        say(f"  {lab}: f-element structures {fel.sum()}: median NMAPE {g[fel].median():.3f}% vs "
            f"{g[~fel].median():.3f}% others; of the 20 worst, {fel[g.nlargest(20).index].sum()} contain f elements")
    for k, d in desc.items():
        nq = (d["N"] - d["N_ref"]) / d["N_ref"]
        if k.endswith("_renorm"):
            continue
        say(f"electron count {k}: median |dN|/N {100 * nq.abs().median():.3f}%, p90 {100 * nq.abs().quantile(.9):.3f}%, "
            f"max {100 * nq.abs().max():.2f}% ({nq.abs().idxmax()}); signed median {100 * nq.median():+.3f}%")
    errs = {k: (d["error"].fillna("") != "").sum() for k, d in desc.items()}
    say(f"pydemi errors on predicted densities: {errs}")
    say()
    for k, s in scores.items():
        say(f"{k}: {len(s)} descriptors; Spearman >= 0.99: {(s.spearman >= 0.99).sum()}, >= 0.95: "
            f"{(s.spearman >= 0.95).sum()}, < 0.8: {(s.spearman < 0.8).sum()}; median over descriptors of "
            f"median rel. error {s.median_rel.median():.2%}; of p90 {s.p90_rel.median():.2%}")
    sc, ft = scores["scratch"], scores["finetune"]
    both = sc.index.intersection(ft.index)
    say(f"fine-tuned Spearman higher than scratch for {(ft.spearman[both] > sc.spearman[both]).sum()}/{len(both)}; "
        f"median rel. error lower for {(ft.median_rel[both] < sc.median_rel[both]).sum()}/{len(both)}")
    rn = scores["finetune_renorm"]
    d = (rn.spearman - ft.spearman).reindex(both)
    say(f"renormalization (fine-tuned): Spearman change median {d.median():+.4f}, largest gain "
        f"{d.max():+.4f} ({d.idxmax()}), largest loss {d.min():+.4f} ({d.idxmin()})")
    say("\nlowest Spearman, fine-tuned:")
    say(ft.head(15).round(4).to_string())
    return grid, fel


def dft_time(names):
    """Elapsed time and core count of the DFT single point, from the end of each OUTCAR (cached)."""
    cache = ML / "dft_time_test.csv"
    if cache.exists():
        return pd.read_csv(cache).set_index("name")
    from common import DATASET
    rows = []
    for n in names:
        f = DATASET / n / "OUTCAR"
        if not f.exists():
            continue
        with open(f, "rb") as h:
            head = h.read(20000).decode(errors="ignore")
            h.seek(0, 2)
            h.seek(max(0, h.tell() - 20000))
            tail = h.read().decode(errors="ignore")
        el, co = re.search(r"Elapsed time \(sec\):\s*([\d.]+)", tail), re.search(r"running on\s+(\d+) total cores", head)
        if el and co:
            rows.append({"name": n, "elapsed_s": float(el.group(1)), "cores": int(co.group(1))})
    df = pd.DataFrame(rows).set_index("name")
    df.to_csv(cache)
    return df


def _species(name):
    return set(re.findall(r"[A-Z][a-z]?", name.split("_")[0]))


def per_structure(dft, pred, names):
    """Median relative error over the scored descriptors, per structure."""
    r = pd.DataFrame({c: rel(dft[c].astype(float), pred.loc[dft.index, c].astype(float)) for c in names})
    return r.median(axis=1)


def figure(dft, desc, scores, grid, fel):
    cat = pd.read_csv(REPO / "docs" / "catalogue.csv").set_index("name")
    fig = plt.figure(figsize=(7.2, 4.6))
    outer = fig.add_gridspec(2, 1, hspace=0.32, height_ratios=[1.45, 1])
    top = outer[0].subgridspec(1, 3, wspace=0.55)
    bottom = outer[1].subgridspec(1, 6, wspace=0.6)
    # (a) grid NMAPE per structure
    ax = fig.add_subplot(top[0])
    s, f = grid["scratch"], grid["finetune"].reindex(grid["scratch"].index)
    ax.scatter(s[~fel], f[~fel], s=4, alpha=0.5, lw=0, color="0.35", label="others")
    ax.scatter(s[fel], f[fel], s=5, alpha=0.8, lw=0, color="C3", label="with f elements")
    lim = (0.1, 15)
    ax.plot(lim, lim, "k-", lw=0.6)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="NMAPE, from scratch (%)",
           ylabel="NMAPE, fine-tuned (%)", title="(a) density error per structure")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.2)
    # (b) descriptor Spearman, scratch vs fine-tuned
    ax = fig.add_subplot(top[1])
    sc, ft = scores["scratch"], scores["finetune"]
    both = sc.index.intersection(ft.index)
    x, y = (1 - sc.spearman[both]).clip(1e-5), (1 - ft.spearman[both]).clip(1e-5)
    col = [DOMAIN_COLOR.get(cat.loc[n, "domain"], "k") if n in cat.index else "k" for n in both]
    ax.scatter(x, y, s=9, c=col, lw=0)
    lim = (1e-5, 2)
    ax.plot(lim, lim, "k-", lw=0.6)
    for v in (0.01, 0.05):
        ax.axhline(v, color="0.7", lw=0.5, ls=":")
        ax.axvline(v, color="0.7", lw=0.5, ls=":")
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel=r"$1-\rho_s$, from scratch",
           ylabel=r"$1-\rho_s$, fine-tuned", title="(b) rank agreement per descriptor")
    ax.legend(handles=[plt.Line2D([], [], ls="", marker="o", ms=3, color=c, label=d)
                       for d, c in DOMAIN_COLOR.items()], loc="upper left", frameon=False, handletextpad=0.1)
    # (c) per-structure descriptor error vs density error
    ax = fig.add_subplot(top[2])
    names = [n for n in ft.index if ft.loc[n, "spearman"] >= 0.95]
    e = 100 * per_structure(dft, desc["finetune"], names)
    g = grid["finetune"].reindex(e.index)
    ax.scatter(g[~fel.reindex(e.index)], e[~fel.reindex(e.index)], s=4, alpha=0.5, lw=0, color="0.35")
    ax.scatter(g[fel.reindex(e.index)], e[fel.reindex(e.index)], s=5, alpha=0.8, lw=0, color="C3")
    rs = g.rank().corr(e.rank())
    ax.set(xscale="log", yscale="log", xlabel="NMAPE, fine-tuned (%)",
           ylabel="median descriptor error (%)",
           title=f"(c) per structure ({len(names)} descr.)")
    ax.text(0.04, 0.95, rf"$\rho_s$ = {rs:.2f}", transform=ax.transAxes, va="top", fontsize=6.5)
    say(f"\nper-structure median descriptor error (fine-tuned, {len(names)} descriptors with Spearman >= 0.95): "
        f"median {e.median():.3f}%, p90 {e.quantile(.9):.3f}%; Spearman with grid NMAPE {rs:.3f}")
    # (d-i) parity plots
    for k, n in enumerate(PARITY):
        ax = fig.add_subplot(bottom[k])
        a, b = dft[n].astype(float), desc["finetune"].loc[dft.index, n].astype(float)
        ok = np.isfinite(a) & np.isfinite(b)
        ax.scatter(a[ok], b[ok], s=2, alpha=0.5, lw=0, color="C0")
        if n in LOG_PARITY:
            ok &= (a > 0) & (b > 0)
            ax.set(xscale="log", yscale="log")
        lo, hi = np.nanmin([a[ok].min(), b[ok].min()]), np.nanmax([a[ok].max(), b[ok].max()])
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.5)
        ax.set(xlim=(lo, hi), ylim=(lo, hi))
        ax.set_box_aspect(1)
        ax.set_title(f"({'defghi'[k]}) {n}", fontsize=6.5)
        ax.text(0.05, 0.95, rf"$\rho_s$={ft.loc[n, 'spearman']:.3f}", transform=ax.transAxes, va="top", fontsize=5.5)
        ax.tick_params(labelsize=5, length=2)
        if k == 0:
            ax.set_ylabel("fine-tuned ML")
        ax.set_xlabel("DFT", fontsize=6.5)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_ml.{ext}", dpi=250)
    plt.close(fig)
    print("wrote fig_ml")


def figure_classes(dft, desc, scores, grid):
    """SI: density and descriptor error by chemical class (classes of the anisotropy report)."""
    cls = pd.read_csv(REPO / "docs" / "anisotropy_descriptors/data/anisotropy_dataset.csv")
    cls = cls.set_index("id")["chem_class"]
    order = ["elemental", "intermetallic", "boride/carbide", "hydride", "pnictide", "chalcogenide", "oxide", "halide"]
    names = [n for n in scores["finetune"].index if scores["finetune"].loc[n, "spearman"] >= 0.95]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.3})
    say("\nby chemical class (n; median grid NMAPE scratch / fine-tuned; median descriptor error fine-tuned):")
    for m, off, colr in (("scratch", -0.2, "0.6"), ("finetune", 0.2, "C0")):
        g = grid[m]
        e = 100 * per_structure(dft, desc[m], names)
        c = cls.reindex(g.index)
        for ax, v in zip(axes, (g, e.reindex(g.index))):
            data = [v[c == k].dropna() for k in order]
            ax.boxplot(data, positions=np.arange(len(order)) + off, widths=0.35, showfliers=False,
                       patch_artist=True, boxprops={"facecolor": colr, "alpha": 0.6, "lw": 0.6},
                       medianprops={"color": "k", "lw": 0.8}, whiskerprops={"lw": 0.6}, capprops={"lw": 0.6})
    for k in order:
        sel = cls.reindex(grid["finetune"].index) == k
        e = 100 * per_structure(dft, desc["finetune"], names).reindex(grid["finetune"].index)
        say(f"  {k}: {int(sel.sum())}; {grid['scratch'][sel].median():.3f} / {grid['finetune'][sel].median():.3f}%; "
            f"{e[sel].median():.3f}%")
    for ax, lab in zip(axes, ("grid NMAPE (%)", f"median error over {len(names)} descriptors (%)")):
        ax.set_xticks(range(len(order)), [f"{k}\n({int((cls.reindex(grid['finetune'].index) == k).sum())})"
                                          for k in order], rotation=45, ha="right", fontsize=6.5)
        ax.set_ylabel(lab)
        ax.set_yscale("log")
        ax.set_yticks([0.2, 0.5, 1, 2, 5], ["0.2", "0.5", "1", "2", "5"])
        ax.minorticks_off()
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.6) for c in ("0.6", "C0")],
                   labels=["from scratch", "fine-tuned"], frameon=False, loc="upper left")
    axes[0].set_title("(a) density error")
    axes[1].set_title("(b) descriptor error")
    for ext in ("pdf", "png"):
        fig.savefig(FIG / "si" / f"fig_si_ml_classes.{ext}", dpi=250)
    plt.close(fig)
    print("wrote si/fig_si_ml_classes")


def esc(s):
    return str(s).replace("_", r"\_")


def table(scores):
    sc, ft, rn = scores["scratch"], scores["finetune"], scores["finetune_renorm"]
    order = ft.sort_values("spearman", ascending=False).index
    rows = []
    pc = lambda v: f"{100 * v:.0f}" if 100 * v >= 10 else f"{100 * v:.2f}"
    for n in order:
        r2 = ft.loc[n, "r2"]
        rows.append(r"\texttt{%s} & %s & %.4f & %s & %s & %.4f & %s & %.4f \\" % (
            esc(n), pc(sc.loc[n, "median_rel"]), sc.loc[n, "spearman"], pc(ft.loc[n, "median_rel"]),
            pc(ft.loc[n, "p90_rel"]), ft.loc[n, "spearman"], f"{r2:.3f}" if r2 > -10 else f"{r2:.0f}",
            rn.loc[n, "spearman"]))
    per = (len(rows) + 1) // 2
    out = []
    for k in range(2):
        out += [r"\begin{table}[p]", r"\centering",
                r"\caption{Descriptors of the predicted densities against those of the DFT densities of the "
                r"605 test structures (part %d of 2), ordered by the fine-tuned model's Spearman rank "
                r"correlation $\rho_s$ across structures: median (and, for the fine-tuned model, "
                r"90th-percentile) relative error $|a-b|/\max(|a|,|b|)$ in per cent, $\rho_s$, $R^2$ and $\rho_s$ "
                r"after renormalizing the prediction to $N=\sum Z_{\mathrm{val}}$ "
                r"(\texttt{paper/analysis/o\_ml\_analysis.py}).}" % (k + 1)]
        if k == 0:
            out.append(r"\label{tab:si-ml}")
        out += [r"\scriptsize", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
                r" & \multicolumn{2}{c}{from scratch} & \multicolumn{5}{c}{fine-tuned} \\",
                r"Descriptor & med.\ (\%) & $\rho_s$ & med.\ (\%) & p90 (\%) & $\rho_s$ & $R^2$ & $\rho_s$ (renorm.) \\",
                r"\midrule"] + rows[k * per:(k + 1) * per] + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (SI / "ml_table.tex").write_text("\n".join(out) + "\n")
    print("wrote si/ml_table.tex")


def main():
    dft, desc = load()
    scores = {k: score(dft, d) for k, d in desc.items()}
    for k, s in scores.items():
        s.to_csv(OUT / f"ml_scores_{k}.csv")
    scores = {k: s.drop(index=EXCLUDE, errors="ignore") for k, s in scores.items()}
    grid, fel = summary(dft, desc, scores)
    figure(dft, desc, scores, grid, fel)
    figure_classes(dft, desc, scores, grid)
    table(scores)
    (OUT / "ml_summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
