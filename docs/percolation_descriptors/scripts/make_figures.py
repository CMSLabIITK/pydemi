"""All figures of the percolation and density-floor report (PNG, 200 dpi, and vector PDF).

Inputs (../data/): analytic_*.csv (analytic_models.py), percolation_dataset.csv
(dataset_table.py), slices.npz, examples.csv (examples.py), bottlenecks.csv, cell_tests.csv
(bottlenecks.py), ml_scores_*.csv (ChargE3Net test-set scores); from the repository,
paper/analysis/out/convergence.csv. Figures whose inputs are missing are skipped.

Usage:  python make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE.parent / "data", HERE.parent / "figures"
PAPER = HERE.parents[2] / "paper" / "analysis" / "out"
CLASSES = ["elemental", "intermetallic", "boride/carbide", "hydride", "pnictide", "chalcogenide",
           "oxide", "halide"]
CLS_COLOR = dict(zip(CLASSES, plt.cm.Dark2(np.arange(8))))
SYSTEMS = ["cubic", "hexagonal", "trigonal", "tetragonal", "orthorhombic", "monoclinic", "triclinic"]
NAMES = ["rho_perc_a", "rho_perc_b", "rho_perc_c", "perc_anisotropy", "rho_min_ratio"]

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "figure.dpi": 100, "savefig.bbox": "tight"})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print("wrote", name)


def dataset():
    return pd.read_csv(DATA / "percolation_dataset.csv")


def short(c):
    return c.replace("boride/carbide", "B/C")


# ------------------------------------------------------------------ 1 models
def fig_models():
    A, B, C, D = (pd.read_csv(DATA / f"analytic_{k}.csv") for k in "ABCD")
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.3))
    ax[0].plot(A["a"], A["rho_perc_exact"], "k-", lw=1, label=r"$\rho(a/2,0,0)$ exact")
    ax[0].plot(A["a"], A["rho_perc_a"], "o", color="C0", label=r"$\rho_{\mathrm{perc}}$ (a = b = c)")
    ax[0].plot(A["a"], A["rho_min_exact"], "k--", lw=1, label=r"$\rho(a/2,a/2,a/2)$ exact")
    ax[0].plot(A["a"], A["rho_min"], "s", color="C3", label=r"$\rho_{\min}$")
    ax[0].set(yscale="log", xlabel="lattice spacing a (Å)", ylabel="e/Å$^3$",
              title="(a) simple cubic lattice of Gaussians")
    ax[0].legend(frameon=False, fontsize=6.5)
    ax[1].plot(B["c"], B["rho_perc_a_exact"], "k-", lw=1)
    ax[1].plot(B["c"], B["rho_perc_a"], "o", color="C0", label=r"$\rho_{\mathrm{perc},a}=\rho_{\mathrm{perc},b}$")
    ax[1].plot(B["c"], B["rho_perc_c_exact"], "k-", lw=1)
    ax[1].plot(B["c"], B["rho_perc_c"], "s", color="C1", label=r"$\rho_{\mathrm{perc},c}$")
    ax1 = ax[1].twinx()
    ax1.plot(B["c"], B["perc_anisotropy_exact"], "k:", lw=1)
    ax1.plot(B["c"], B["perc_anisotropy"], "^", color="C3", label="perc_anisotropy (right)")
    ax1.set_ylabel("perc_anisotropy", color="C3")
    ax[1].set(yscale="log", xlabel="c (Å), a = b = 3 Å", ylabel="e/Å$^3$", title="(b) tetragonal stretch")
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax1.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.5, loc="lower left")
    b = C["background"]
    ax[2].plot(b, C["rho_min_ratio"], "o-", color="C3", label="rho_min_ratio (cubic)")
    ax[2].plot(b, C["perc_anisotropy_tetragonal"], "s-", color="C2", label="perc_anisotropy (c/a = 1.8)")
    ax[2].plot(b, C["rho_perc_a"] - b, "^-", color="C0", label=r"$\rho_{\mathrm{perc}}-b$ (e/Å$^3$)")
    ax[2].set(xscale="symlog", xlabel="uniform background b (e/Å$^3$)", ylabel="value",
              title="(c) adding a uniform electron gas")
    ax[2].set_xscale("symlog", linthresh=0.005)
    ax[2].legend(frameon=False, fontsize=6.5)
    h = D["spacing_A"]
    for lab, c in (("a", "C0"), ("b", "C1"), ("c", "C2")):
        ax[3].plot(h, (D[f"rho_perc_{lab}"] - D["rho_perc_exact"]) / D["rho_perc_exact"], "o-", color=c, ms=3.5,
                   label=f"rel. error, axis {lab}")
    ax[3].plot(h, D["perc_anisotropy"], "k^-", ms=3.5, label="perc_anisotropy (exact 0)")
    ax[3].axhline(0, color="0.6", lw=0.8)
    ax[3].set(xlabel="grid spacing (Å), atom off the grid", ylabel="value", title="(d) grid registration")
    ax[3].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig01_models")


# ------------------------------------------------------------------ 2 slices
def fig_slices():
    z = np.load(DATA / "slices.npz", allow_pickle=False)
    runs = list(dict.fromkeys(k.split("__")[0] for k in z.files))
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7))
    for ax, run in zip(axes.flat, runs):
        X, Y, R = z[f"{run}__X"], z[f"{run}__Y"], z[f"{run}__rho"]
        if R.shape != X.shape:                      # plane axes stored in array order
            R = R.T
        lev = z[f"{run}__levels"]
        meta = z[f"{run}__meta"]
        Rp = np.clip(R, 1e-4, None)
        ax.pcolormesh(X, Y, Rp, cmap="Greys", norm=LogNorm(vmin=1e-3, vmax=max(Rp.max(), 1.0)), shading="auto",
                      rasterized=True)
        for k, (lab, c) in enumerate(zip("abc", ("C0", "C1", "C3"))):
            ax.contour(X, Y, R, levels=[lev[k]], colors=[c], linewidths=1.0 if lab != meta[2] else 1.6,
                       linestyles="-" if lab != meta[2] else "--")
        bx, by = z[f"{run}__bottleneck"]
        ax.plot(bx, by, "x", color="m", ms=8, mew=2)
        xmin, xmax, ymin, ymax = X.min(), X.max(), Y.min(), Y.max()
        for (x, y, dz), e in zip(z[f"{run}__atoms"], z[f"{run}__species"]):
            if xmin <= x <= xmax and ymin <= y <= ymax and dz < 0.3:
                ax.text(x, y, e, fontsize=6, ha="center", va="center", color="y")
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xticks=[], yticks=[])
        ax.set_aspect("equal")
        ax.set_title(f"{meta[0]}\n({meta[1]} plane; levels a/b/c = {lev[0]:.3f}/{lev[1]:.3f}/{lev[2]:.3f})",
                     fontsize=7.5)
    fig.suptitle(r"$\rho$ (grey, log) with the iso-lines $\rho=\rho_{\mathrm{perc},\alpha}$ (blue a, orange b, "
                 r"red c; dashed = lowest axis) and its bottleneck voxel (magenta ×), in a plane through that "
                 "bottleneck", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig02_slices")


# ------------------------------------------------------------------ 3 bottlenecks
def fig_bottlenecks():
    B = pd.read_csv(DATA / "bottlenecks.csv")
    B = B[B["error"].isna()] if "error" in B else B
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.3))
    r = pd.concat([B[f"r_over_Rpaw_{l}"] for l in "abc"])
    m = pd.concat([B[f"midpointness_{l}"] for l in "abc"])
    p = pd.concat([B[f"pair_over_dmin_{l}"] for l in "abc"])
    ax[0].hist(r, bins=np.linspace(0, 3, 61), color="C0")
    ax[0].axvline(1, color="k", lw=0.8)
    ax[0].set(xlabel=r"bottleneck $r/R_{\mathrm{PAW}}$", ylabel="bottlenecks (3 per structure)",
              title=f"(a) bottlenecks vs PAW spheres ({len(B)} structures)")
    ax[1].hist(m, bins=np.linspace(0, 1, 51), color="C1")
    ax[1].set(xlabel=r"$|d_1-d_2|/(d_1+d_2)$ (0 = pair midpoint)", ylabel="bottlenecks",
              title="(b) position between the two nearest nuclei")
    ax[2].hist(p, bins=np.linspace(0.9, 3, 43), color="C2")
    ax[2].set(xlabel="pair distance / shortest interatomic distance", ylabel="bottlenecks",
              title="(c) which pair of atoms")
    ax[3].hist(B["rmin_over_Rpaw"], bins=np.linspace(0, 3, 61), color="C3")
    ax[3].axvline(1, color="k", lw=0.8)
    ax[3].set(xlabel=r"$r/R_{\mathrm{PAW}}$ of the $\rho_{\min}$ voxel", ylabel="structures",
              title=f"(d) where rho_min is ({(B['rho_min'] < 0).mean():.0%} negative)")
    fig.tight_layout()
    save(fig, "fig03_bottlenecks")


# ------------------------------------------------------------------ 4 distributions
def fig_distributions(d):
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
    for lab, c in zip("abc", ("C0", "C1", "C3")):
        ax[0].hist(d[f"rho_perc_{lab}"], bins=np.logspace(-2.6, 0.3, 60), histtype="step", color=c, label=lab)
    ax[0].set(xscale="log", xlabel=r"$\rho_{\mathrm{perc},\alpha}$ (e/Å$^3$)", ylabel="structures",
              title="(a) percolation levels")
    ax[0].legend(frameon=False)
    ax[1].hist(np.clip(d["perc_anisotropy"], 1e-5, None), bins=np.logspace(-5, 0.5, 60), color="C2")
    ax[1].set(xscale="log", xlabel="perc_anisotropy (values < 1e-5 at 1e-5)", ylabel="structures",
              title="(b) perc_anisotropy")
    ax[2].hist(d["rho_min_ratio"], bins=np.linspace(-4, 1, 80), color="C3", alpha=0.7, label="rho_min_ratio")
    ax[2].hist(d["rho_min_int_ratio"], bins=np.linspace(-4, 1, 80), color="C0", alpha=0.7,
               label="rho_min_int_ratio (paw)")
    ax[2].axvline(0, color="k", lw=0.8)
    ax[2].set(xlabel=r"$\rho_{\min}/\langle\rho\rangle$", ylabel="structures", title="(c) density floor")
    ax[2].legend(frameon=False, fontsize=6.5)
    ax[3].scatter(d["rho_min_int_ratio"], d["rho_min_ratio"], s=2, alpha=0.3, rasterized=True)
    ax[3].plot([-0.2, 1], [-0.2, 1], "k-", lw=0.6)
    r = d["rho_min_int_ratio"].rank().corr(d["rho_min_ratio"].rank())
    ax[3].set(xlabel="rho_min_int_ratio (outside spheres)", ylabel="rho_min_ratio (whole cell)",
              title=f"(d) whole cell vs interstitial (Spearman {r:.2f})", ylim=(-4, 1))
    fig.tight_layout()
    save(fig, "fig04_distributions")


def _box(ax, d, n, key, order, colors, log=False):
    gs = [g for g in order if (d[key] == g).sum() >= 5]
    bp = ax.boxplot([d.loc[d[key] == g, n].dropna() for g in gs], showfliers=False, widths=0.6, patch_artist=True)
    for pch, g in zip(bp["boxes"], gs):
        pch.set_facecolor(colors[g])
        pch.set_alpha(0.75)
    ax.set_xticks(range(1, len(gs) + 1), [short(g) for g in gs], rotation=45, ha="right", fontsize=7)
    if log:
        ax.set_yscale("log")


def fig_by_class(d):
    d = d.assign(perc_anisotropy_floor=np.clip(d["perc_anisotropy"], 1e-5, None))
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))
    cols = [("rho_perc_mean", True, r"mean $\rho_{\mathrm{perc}}$ (e/Å$^3$)"),
            ("rho_perc_min_over_mean_rho", True, r"min $\rho_{\mathrm{perc}}$ / $\langle\rho\rangle$"),
            ("perc_anisotropy_floor", True, "perc_anisotropy"),
            ("rho_min_int_ratio", False, "rho_min_int_ratio (paw)")]
    scol = dict(zip(SYSTEMS, plt.cm.tab10(np.arange(7))))
    for k, (n, log, lab) in enumerate(cols):
        _box(axes[0, k], d, n, "chem_class", CLASSES, CLS_COLOR, log)
        axes[0, k].set_title(lab, fontsize=8)
        _box(axes[1, k], d, n, "crystal_system_relaxed", SYSTEMS, scol, log)
        axes[1, k].set_title(lab + " by crystal system", fontsize=8)
    fig.suptitle("By chemical class (top) and relaxed crystal system (bottom)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig05_by_class_and_system")


def fig_relations(d):
    fig, ax = plt.subplots(1, 4, figsize=(14.5, 3.5))
    pairs = [("rho_mid_mean", "rho_perc_min", "mean ρ at bond midpoints (e/Å$^3$)"),
             ("rho_int_mean", "rho_perc_min", "mean interstitial ρ (e/Å$^3$)"),
             ("ELF_bond_avg", "rho_perc_min_over_mean_rho", "mean ELF in the bond shell"),
             ("charge_FA", "perc_anisotropy", "charge_FA (gradient anisotropy)")]
    for a, (xn, yn, xl) in zip(ax, pairs):
        a.scatter(d[xn], d[yn], s=2, c=[CLS_COLOR[c] for c in d["chem_class"]], alpha=0.35, rasterized=True)
        r = d[xn].rank().corr(d[yn].rank())
        a.set(xlabel=xl, ylabel=yn, title=f"Spearman {r:+.2f}")
        if yn != "rho_perc_min_over_mean_rho":
            a.set_yscale("log")
    ax[3].set_xscale("log")
    handles = [plt.Line2D([], [], ls="", marker="o", color=CLS_COLOR[k], label=k) for k in CLASSES]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, "fig06_relations")


def fig_correlations(d):
    cols = ["rho_perc_min", "rho_perc_mean", "rho_perc_min_over_mean_rho", "perc_anisotropy", "rho_min_ratio",
            "rho_min_int_ratio", "rho_min", "rho_min_int", "rho_int_mean", "rho_mid_mean", "rho_mid_std", "f_int",
            "f_bond", "m1", "ELF_bond_avg", "f_ELF_localized", "zeta", "charge_FA", "T_eigenvalues_t1",
            "def_polarity_out", "shannon_entropy", "delta_chi", "magpie_mean_NValence", "magpie_mean_GSbandgap"]
    c = d[cols].rank().corr()
    c.to_csv(DATA / "spearman_correlations.csv")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols)), cols, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{c.values[i, j]:.1f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center",
                    fontsize=5, color="w" if abs(c.values[i, j]) > 0.6 else "k")
    ax.axhline(5.5, color="k", lw=0.8)
    ax.axvline(5.5, color="k", lw=0.8)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman rank correlation")
    ax.set_title(f"Rank correlations over the {len(d):,} structures")
    save(fig, "fig07_correlations")


def fig_robustness():
    conv = pd.read_csv(PAPER / "convergence.csv")
    names = ["rho_perc_a", "rho_perc_b", "rho_perc_c", "perc_anisotropy", "rho_min", "rho_min_ratio"]
    conv = conv[conv["descriptor"].isin(names) & np.isfinite(conv["rel_change_x0.8"])]
    ct = pd.read_csv(DATA / "cell_tests.csv")
    ct = ct[ct["error"].isna()] if "error" in ct else ct
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    conv = conv.assign(change=np.where(conv["descriptor"] == "perc_anisotropy",
                                       (conv["x0.8"] - conv["full"]).abs(), conv["rel_change_x0.8"].abs()))
    g = conv.groupby("descriptor")["change"]
    x = np.arange(len(names))
    ax[0].bar(x - 0.2, [g.get_group(n).median() for n in names], 0.4, label="median")
    ax[0].bar(x + 0.2, [g.get_group(n).quantile(0.9) for n in names], 0.4, label="90th pct")
    ax[0].set_xticks(x, [n if n != "perc_anisotropy" else "perc_anisotropy\n(absolute change)" for n in names],
                     rotation=45, ha="right", fontsize=7)
    ax[0].set(yscale="log", ylabel="relative change (perc_anisotropy: absolute)", title=f"(a) grid 80% ({conv['id'].nunique()} structures)")
    ax[0].legend(frameon=False)
    sh = ct.dropna(subset=["sheared_a"])
    ax[1].scatter(sh["base_a"], sh["sheared_a"], s=12, color="C3", label=r"along $a_1' = a_1+a_2$")
    ax[1].scatter(sh["base_b"], sh["sheared_b"], s=12, color="C0", marker="s", label=r"along $a_2' = a_2$")
    ax[1].scatter(sh["base_c"], sh["sheared_c"], s=12, color="C2", marker="^", label=r"along $a_3$")
    lim = [sh[["base_a", "sheared_a"]].min().min() * 0.8, sh[["base_a", "sheared_a"]].max().max() * 1.2]
    ax[1].plot(lim, lim, "k-", lw=0.6)
    ax[1].set(xscale="log", yscale="log", xlabel="level in the original cell", ylabel="level in the sheared cell",
              title=f"(b) choice of cell ({len(sh)} structures)")
    ax[1].legend(frameon=False, fontsize=6.5)
    files = [("scratch", DATA / "ml_scores_scratch.csv"), ("fine-tuned", DATA / "ml_scores_finetune.csv")]
    files = [(k, pd.read_csv(p)) for k, p in files if p.exists()]
    mlnames = ["rho_perc_a", "rho_perc_b", "rho_perc_c", "perc_anisotropy", "rho_min_ratio", "rho_min_int_ratio"]
    xx = np.arange(len(mlnames))
    w = 0.8 / max(len(files), 1)
    for k, (tag, s) in enumerate(files):
        s = s.set_index("descriptor").reindex(mlnames)
        ax[2].bar(xx + (k - (len(files) - 1) / 2) * w, s["median_rel"], w, label=f"ChargE3Net {tag}")
    ax[2].set_xticks(xx, mlnames, rotation=45, ha="right", fontsize=7)
    ax[2].set(yscale="log", ylabel="median |relative error|", title="(c) from predicted densities (605 test)")
    ax[2].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig08_robustness_and_ml")


def fig_examples():
    E = pd.read_csv(DATA / "examples.csv")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    x = np.arange(len(E))
    labels = [l.split(" (")[0] for l in E["label"]]
    for k, (lab, c) in enumerate(zip("abc", ("C0", "C1", "C3"))):
        axes[0].bar(x + (k - 1) * 0.27, E[f"rho_perc_{lab}"], 0.27, color=c, label=lab)
    axes[0].set(yscale="log", ylabel="e/Å$^3$", title=r"$\rho_{\mathrm{perc},\alpha}$")
    axes[0].legend(frameon=False)
    axes[1].bar(x, np.clip(E["perc_anisotropy"], 1e-5, None), color="C2")
    axes[1].set(yscale="log", title="perc_anisotropy")
    axes[2].bar(x - 0.2, E["rho_min_ratio"], 0.4, color="C3", label="whole cell")
    axes[2].bar(x + 0.2, E["rho_min_int_ratio"], 0.4, color="C0", label="interstitial (paw)")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set(title=r"$\rho_{\min}/\langle\rho\rangle$")
    axes[2].legend(frameon=False, fontsize=6.5)
    axes[3].bar(x, E[["rho_perc_a", "rho_perc_b", "rho_perc_c"]].min(axis=1) / E["mean_rho"], color="C4")
    axes[3].set(title=r"min $\rho_{\mathrm{perc}}$ / $\langle\rho\rangle$")
    for a in axes:
        a.set_xticks(x, labels, rotation=60, fontsize=7)
    fig.tight_layout()
    save(fig, "fig09_examples")


def main():
    FIG.mkdir(exist_ok=True)
    d = dataset()
    for f in [fig_models, fig_slices, fig_bottlenecks, lambda: fig_distributions(d), lambda: fig_by_class(d),
              lambda: fig_relations(d), lambda: fig_correlations(d), fig_robustness, fig_examples]:
        try:
            f()
        except FileNotFoundError as exc:
            print("skipped:", exc)


if __name__ == "__main__":
    main()
