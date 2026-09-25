"""All figures of the site-heterogeneity report (PNG, 200 dpi, and vector PDF).

Inputs (../data/): analytic_*.csv (analytic_models.py), heterogeneity_dataset.csv
(dataset_table.py), sites_examples.csv, examples.csv (examples.py), robust_*.csv
(robustness.py), ml_scores_*.csv (ChargE3Net test-set scores, copied from the ML
evaluation). Figures whose inputs are missing are skipped with a note.

Usage:  python make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE.parent / "data", HERE.parent / "figures"
BASES = ["m1", "f_bond", "zeta"]
BLAB = {"m1": "$m_1$", "f_bond": r"$f_{\mathrm{bond}}$", "zeta": r"$\zeta$"}
STATS = ["site_std", "within_element_var", "between_element_var"]
SLAB = {"site_std": "site std", "within_element_var": "within-element var",
        "between_element_var": "between-element var"}
NAMES = [f"{b}_{s}" for b in BASES for s in STATS]
CLASSES = ["elemental", "intermetallic", "boride/carbide", "hydride", "pnictide", "chalcogenide",
           "oxide", "halide"]
CLS_COLOR = dict(zip(CLASSES, plt.cm.Dark2(np.arange(8))))
FLOOR = 1e-12

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "figure.dpi": 100, "savefig.bbox": "tight"})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print("wrote", name)


def short(n):
    for b in sorted(BASES, key=len, reverse=True):
        if n.startswith(b + "_"):
            return f"{BLAB[b]} {SLAB[n[len(b) + 1:]]}"
    return n


def dataset():
    return pd.read_csv(DATA / "heterogeneity_dataset.csv")


# ------------------------------------------------------------------ 1 models
def fig_models(d):
    A, B, C = (pd.read_csv(DATA / f"analytic_{k}.csv") for k in "ABC")
    fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.2))
    a = A["alpha_B"]
    ax[0].plot(a, A["m1@0_Na"], "o-", color="C0", label="$m_1^{(i)}$, A sites")
    ax[0].plot(a, A["m1@1_Cl"], "s-", color="C3", label="$m_1^{(i)}$, B sites")
    ax[0].plot(a, 1.1284 / np.sqrt(a), "k:", lw=1, label=r"isolated atom $2/\sqrt{\pi\alpha}$")
    ax0 = ax[0].twinx()
    ax0.plot(a, A["m1_between_element_var"], "^-", color="C2", ms=4, label="between var (right)")
    ax0.plot(a, (A["m1@0_Na"] - A["m1@1_Cl"]) ** 2 / 4, "k+", ms=7, label=r"$(\bar X_A-\bar X_B)^2/4$")
    ax0.set_ylabel("between-element variance (Å$^2$)", color="C2")
    ax[0].set(xlabel=r"$\alpha_B$ (Å$^{-2}$), $\alpha_A$ = 2", ylabel="per-site $m_1$ (Å)",
              title="(a) size contrast: within = 0 by symmetry")
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = ax0.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.5, loc="upper center")
    x = B["alpha_defect"]
    for b, c in (("m1", "C0"), ("f_bond", "C1")):
        ax[1].plot(x, B[f"{b}_within_element_var"], "o-", color=c, label=f"{BLAB[b]} within")
        ax[1].plot(x, B[f"{b}_between_element_var"], "s--", color=c, mfc="none", label=f"{BLAB[b]} between")
    ax[1].set(yscale="log", xlabel=r"$\alpha'$ of the one defect site", ylabel="variance",
              title="(b) one defect site: within appears", ylim=(1e-8, 1e-1))
    ax[1].text(2.05, 2e-8, "exactly 0 at α' = 2", fontsize=6.5)
    ax[1].axvline(2, color="0.7", lw=0.8)
    ax[1].legend(frameon=False, fontsize=6.5)
    g = C.groupby("sigma_A")
    for b, c in (("m1", "C0"), ("f_bond", "C1"), ("zeta", "C4")):
        m, s = g[f"{b}_within_element_var"].mean(), g[f"{b}_within_element_var"].std().fillna(0)
        ax[2].errorbar(m.index, np.maximum(m, FLOOR), yerr=s, fmt="o-", color=c, ms=3.5, capsize=2,
                       label=f"{BLAB[b]} within")
    ax[2].set(yscale="log", xlabel="rms displacement σ per axis (Å)", ylabel="within-element variance",
              title="(c) positional disorder (5 seeds)", ylim=(1e-8, None))
    ax[2].legend(frameon=False)
    res = []
    for b in BASES:
        w = d[f"{b}_within_element_var"].fillna(0.0)
        r = d[f"{b}_site_std"] ** 2 - w - d[f"{b}_between_element_var"]
        res.append(np.abs(r) / np.maximum(d[f"{b}_site_std"] ** 2, 1e-300))
    bins = np.logspace(-18, -10, 50)
    for r, b, c in zip(res, BASES, ("C0", "C1", "C4")):
        ax[3].hist(np.clip(r[d[f"{b}_site_std"] > 0], 1e-18, None), bins=bins, histtype="step", color=c,
                   label=BLAB[b])
    ax[3].set(xscale="log", xlabel=r"$|\,\mathrm{std}^2-\mathrm{within}-\mathrm{between}\,|/\mathrm{std}^2$",
              ylabel="structures", title=f"(d) ANOVA identity, {len(d):,} structures")
    ax[3].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig01_models")


# ------------------------------------------------------------------ 2 per-site values
def fig_sites():
    S = pd.read_csv(DATA / "sites_examples.csv")
    runs = list(dict.fromkeys(S["id"]))
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    ticks, labels, x0 = [], [], 0
    rng = np.random.default_rng(0)
    for run in runs:
        s = S[S["id"] == run]
        for e in dict.fromkeys(s["element"]):
            se = s[s["element"] == e]
            classes = list(dict.fromkeys(se["equivalent_to"]))
            for ax, b in zip(axes, BASES):
                for k, cl in enumerate(classes):
                    sc = se[se["equivalent_to"] == cl]
                    ax.scatter(x0 + rng.uniform(-0.25, 0.25, len(sc)), sc[f"{b}@nearest"], s=14,
                               color=plt.cm.tab10(k % 10), edgecolor="k", lw=0.3, zorder=3)
            ticks.append(x0)
            labels.append(e)
            x0 += 1
        for ax in axes:
            ax.axvline(x0 - 0.5, color="0.8", lw=0.8)
        axes[0].text(x0 - 0.5 - 0.5 * len(dict.fromkeys(s["element"])), 1.03,
                     s["label"].iloc[0].replace(" (", "\n("), transform=axes[0].get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=7)
        x0 += 0.5
    for ax, b in zip(axes, BASES):
        ax.set_ylabel(f"per-site {BLAB[b]}")
    axes[-1].set_xticks(ticks, labels)
    fig.suptitle("Per-site values under the nearest-atom partition; colour = symmetry-equivalence class "
                 "within the element (spglib, 0.01 Å)", fontsize=9, y=1.04)
    fig.tight_layout()
    save(fig, "fig02_per_site_values")


def fig_site_environment():
    S = pd.read_csv(DATA / "sites_examples.csv")
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
    runs = list(dict.fromkeys(S["id"]))
    for k, run in enumerate(runs):
        s = S[S["id"] == run]
        for a, b in zip(ax, BASES):
            a.scatter(s["d_nn"], s[f"{b}@nearest"], s=16, color=plt.cm.tab10(k), label=s["label"].iloc[0],
                      edgecolor="k", lw=0.3)
    for a, b in zip(ax, BASES):
        a.set(xlabel="nearest-neighbour distance of the site (Å)", ylabel=f"per-site {BLAB[b]}")
    ax[0].legend(frameon=False, fontsize=6.3)
    fig.suptitle("Per-site values against the site's nearest-neighbour distance (all sites of the eight "
                 "examples)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig03_site_environment")


def fig_partitions_examples():
    S = pd.read_csv(DATA / "sites_examples.csv")
    runs = ["ZrO2_14", "Fe3C_62", "Ba2SnO4_139"]
    parts = ["nearest", "becke", "power", "hirshfeld"]
    fig, axes = plt.subplots(len(runs), 3, figsize=(13, 7.5))
    for row, run in zip(axes, runs):
        s = S[S["id"] == run].sort_values(["element", "equivalent_to"])
        reps = s.drop_duplicates("equivalent_to")
        x = np.arange(len(reps))
        for a, b in zip(row, BASES):
            for k, p in enumerate(parts):
                a.bar(x + (k - 1.5) * 0.2, reps[f"{b}@{p}"], 0.2, label=p)
            a.set_xticks(x, [f"{e} {w}" for e, w in zip(reps["element"], reps["wyckoff"])], fontsize=7)
            a.set_title(f"{reps['label'].iloc[0]}: per-site {BLAB[b]}", fontsize=8)
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle("Per-site values of the symmetry-distinct sites (element + Wyckoff letter) under the "
                 "four partitions", fontsize=9)
    fig.tight_layout()
    save(fig, "fig04_partitions_examples")


# ------------------------------------------------------------------ 5 distributions and symmetry floor
def fig_distributions(d):
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 8))
    eq = d["all_sites_equivalent_per_element"]
    for row, b in zip(axes, BASES):
        x = d[f"{b}_site_std"]
        row[0].hist(x, bins=60, color="C0")
        row[0].set(xlabel=f"{BLAB[b]} site std", ylabel="structures")
        for a, st in zip(row[1:], ["within_element_var", "between_element_var"]):
            v = d[f"{b}_{st}"]
            bins = np.logspace(-14, 0, 70)
            if st == "within_element_var":
                a.hist(np.clip(v[eq & v.notna()], 1e-14, None), bins=bins, color="0.6", alpha=0.8,
                       label="all sites of each element equivalent")
                a.hist(np.clip(v[~eq & v.notna()], 1e-14, None), bins=bins, color="C3", alpha=0.7,
                       label="inequivalent sites present")
            else:
                single = d["n_elements"] == 1
                a.hist(np.clip(v[~single], 1e-14, None), bins=bins, color="C2", label="compounds")
            a.set(xscale="log", xlabel=f"{BLAB[b]} {SLAB[st]}", ylabel="structures")
            a.legend(frameon=False, fontsize=6.5)
    fig.suptitle(f"Distributions over the {len(d):,} structures (variances on a log axis; values below "
                 "1e-14 drawn at 1e-14)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig05_distributions")


# ------------------------------------------------------------------ 6 by class
def fig_by_class(d):
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    groups = CLASSES
    for row, b in zip(axes, BASES):
        for a, st in zip(row, STATS):
            n = f"{b}_{st}"
            sub = d
            if st == "within_element_var":
                sub = d[~d["all_sites_equivalent_per_element"] & d[n].notna()]
            if st == "between_element_var":
                sub = d[d["n_elements"] > 1]
            gs = [c for c in groups if (sub["chem_class"] == c).sum() > 0]
            data = [sub.loc[sub["chem_class"] == c, n].dropna() for c in gs]
            bp = a.boxplot(data, showfliers=False, widths=0.6, patch_artist=True)
            for p, c in zip(bp["boxes"], gs):
                p.set_facecolor(CLS_COLOR[c])
                p.set_alpha(0.75)
            a.set_xticks(range(1, len(gs) + 1), [c.replace("boride/carbide", "B/C") for c in gs],
                         rotation=45, ha="right", fontsize=7)
            if st != "site_std":
                a.set_yscale("log")
            extra = " (inequivalent only)" if st == "within_element_var" else \
                " (compounds)" if st == "between_element_var" else ""
            a.set_title(f"{BLAB[b]} {SLAB[st]}{extra}", fontsize=8)
    fig.suptitle("By chemical class (boxes: quartiles; whiskers: 1.5 IQR)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig06_by_class")


# ------------------------------------------------------------------ 7 composition and structure
def fig_composition(d):
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
    c = d[d["n_elements"] > 1]
    pairs = [("magpie_range_CovalentRadius", "m1_between_element_var", "range of covalent radius (pm)"),
             ("magpie_range_Electronegativity", "f_bond_between_element_var", r"$\Delta\chi$ (range)"),
             ("zeta", "zeta_site_std", r"cell $\zeta$")]
    for a, (xn, yn, xl) in zip(ax[:3], pairs):
        a.scatter(c[xn], c[yn], s=2, c=[CLS_COLOR[k] for k in c["chem_class"]], alpha=0.35, rasterized=True)
        r = c[xn].rank().corr(c[yn].rank())
        a.set(xlabel=xl, ylabel=short(yn), title=f"Spearman {r:+.2f}")
    ne = d[~d["all_sites_equivalent_per_element"] & d["m1_within_element_var"].notna()]
    for k, b in enumerate(BASES):
        frac = ne[f"{b}_within_element_var"] / (ne[f"{b}_site_std"] ** 2)
        ax[3].hist(frac.clip(0, 1), bins=40, histtype="step", color=f"C{[0, 1, 4][k]}", label=BLAB[b])
    ax[3].set(xlabel="within / (within + between)", ylabel="structures",
              title=f"(d) within share, {len(ne):,} with inequivalent sites")
    ax[3].legend(frameon=False)
    handles = [plt.Line2D([], [], ls="", marker="o", color=CLS_COLOR[k], label=k) for k in CLASSES]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.4, -0.08))
    fig.tight_layout()
    save(fig, "fig07_composition_and_decomposition")


def fig_size(d):
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
    ne = d[d["m1_within_element_var"].notna()]
    for a, b in zip(ax, BASES):
        for flag, c, lab in ((True, "0.5", "all equivalent"), (False, "C3", "inequivalent sites")):
            s = ne[ne["all_sites_equivalent_per_element"] == flag]
            a.scatter(s["n_atoms"], np.maximum(s[f"{b}_within_element_var"], 1e-14), s=3, color=c, alpha=0.4,
                      label=lab, rasterized=True)
        a.set(xscale="log", yscale="log", xlabel="atoms in the cell", ylabel=f"{BLAB[b]} within-element var")
    ax[0].legend(frameon=False, markerscale=3)
    fig.suptitle("The within-element variance against cell size: a symmetry noise floor (grey) and genuine "
                 "site differences (red)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig08_within_vs_size")


# ------------------------------------------------------------------ 9 correlations
def fig_correlations(d):
    cols = NAMES + ["m1_site_range", "mu_site_std", "m1", "f_bond", "zeta", "charge_FA", "n_atoms",
                    "n_distinct_sites", "magpie_range_CovalentRadius", "magpie_range_Electronegativity",
                    "magpie_avg_dev_CovalentRadius"]
    c = d[cols].rank().corr()
    c.to_csv(DATA / "spearman_correlations.csv")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols)), cols, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{c.values[i, j]:.1f}".replace("0.", ".").replace("-.", "−."), ha="center",
                    va="center", fontsize=5, color="w" if abs(c.values[i, j]) > 0.6 else "k")
    ax.axhline(8.5, color="k", lw=0.8)
    ax.axvline(8.5, color="k", lw=0.8)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman rank correlation")
    ax.set_title(f"Rank correlations over the {len(d):,} structures (pairwise complete)")
    save(fig, "fig09_correlations")


# ------------------------------------------------------------------ 10 robustness
def fig_robustness():
    P = pd.read_csv(DATA / "robust_partitions.csv")
    D = pd.read_csv(DATA / "robust_derivatives_summary.csv")
    Sh = pd.read_csv(DATA / "robust_shells_summary.csv")
    G = pd.read_csv(DATA / "robust_grid.csv")
    Sc = pd.read_csv(DATA / "robust_supercell_summary.csv")
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
    parts = ["becke", "power", "hirshfeld"]
    x = np.arange(len(NAMES))
    for k, p in enumerate(parts):
        s = P[P["partition"] == p].set_index("descriptor").reindex(NAMES)
        ax[0].bar(x + (k - 1) * 0.27, s["spearman"], 0.27, label=p)
    ax[0].set_xticks(x, NAMES, rotation=90, fontsize=6.5)
    ax[0].set(ylabel="Spearman with nearest-atom value", title="(a) partition (300 structures)", ylim=(0, 1.3))
    ax[0].legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center")
    for k, t in enumerate(["fd2", "fd4", "fd8"]):
        s = D[D["variant"] == t]
        ax[1].bar(np.arange(len(s)) + (k - 1) * 0.27, s["median_rel"], 0.27, label=t)
    ax[1].set_xticks(range(3), ["site std", "within var", "between var"], fontsize=7.5)
    ax[1].set(yscale="log", ylabel="median relative change vs FFT", title="(b) derivatives (ζ only)")
    ax[1].legend(frameon=False, fontsize=6.5)
    for k, t in enumerate(Sh["variant"].unique()):
        s = Sh[Sh["variant"] == t]
        ax[2].bar(np.arange(len(s)) + (k - 1) * 0.27, s["spearman"], 0.27, label=t.replace("abs_", "").replace("_", ", "))
    ax[2].set_xticks(range(3), ["site std", "within var", "between var"], fontsize=7.5)
    ax[2].set(ylabel="Spearman with default shells", title=r"(c) shells ($f_{\mathrm{bond}}$ only)",
              ylim=(0, 1.3))
    ax[2].legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center")
    g = G.groupby("descriptor")["rel_change_x0.8"]
    med = [g.get_group(n).abs().median() if n in g.groups else np.nan for n in NAMES]
    p90 = [g.get_group(n).abs().quantile(0.9) if n in g.groups else np.nan for n in NAMES]
    sc = Sc.set_index("descriptor").reindex(NAMES)["median_rel"]
    ax[3].bar(x - 0.27, med, 0.27, label="grid 80%: median")
    ax[3].bar(x, p90, 0.27, label="grid 80%: 90th pct")
    ax[3].bar(x + 0.27, np.maximum(sc, 1e-16), 0.27, label="2×1×1 supercell: median")
    ax[3].set_xticks(x, NAMES, rotation=90, fontsize=6.5)
    ax[3].set(yscale="log", ylabel="relative change", title="(d) grid and supercell", ylim=(1e-15, 1e6))
    ax[3].legend(frameon=False, fontsize=6.5, ncol=2, loc="upper center")
    fig.tight_layout()
    save(fig, "fig10_robustness")


# ------------------------------------------------------------------ 11 ML
def fig_ml():
    files = [("scratch", DATA / "ml_scores_scratch.csv"), ("fine-tuned", DATA / "ml_scores_finetune.csv")]
    files = [(k, pd.read_csv(p)) for k, p in files if p.exists()]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    x = np.arange(len(NAMES))
    wdt = 0.8 / len(files)
    for k, (tag, s) in enumerate(files):
        s = s.set_index("descriptor").reindex(NAMES)
        ax[0].bar(x + (k - (len(files) - 1) / 2) * wdt, s["median_rel"], wdt, label=f"ChargE3Net {tag}")
        ax[1].bar(x + (k - (len(files) - 1) / 2) * wdt, 1 - s["spearman"], wdt, label=f"ChargE3Net {tag}")
    for a in ax:
        a.set_xticks(x, NAMES, rotation=90, fontsize=7)
        a.set_yscale("log")
        a.legend(frameon=False)
    ax[0].set(ylabel="median |relative error|", title="(a) descriptor from predicted vs DFT density")
    ax[1].set(ylabel="1 − Spearman", title="(b) rank agreement across the test structures")
    fig.tight_layout()
    save(fig, "fig11_ml_predictability")


# ------------------------------------------------------------------ 12 examples
def fig_examples():
    E = pd.read_csv(DATA / "examples.csv")
    fig, axes = plt.subplots(3, 3, figsize=(13, 8))
    x = np.arange(len(E))
    for row, b in zip(axes, BASES):
        for a, st in zip(row, STATS):
            v = E[f"{b}_{st}@nearest"]
            a.bar(x, np.maximum(v.fillna(0), 1e-14) if st != "site_std" else v, color="C0")
            if st != "site_std":
                a.set_yscale("log")
            a.set_xticks(x, [l.split(" (")[0] for l in E["label"]], rotation=60, fontsize=7)
            a.set_title(f"{BLAB[b]} {SLAB[st]}", fontsize=8)
    fig.suptitle("The eight examples (nearest-atom partition); NaN within-element values (one site per "
                 "element) and exact zeros drawn at 1e-14", fontsize=9)
    fig.tight_layout()
    save(fig, "fig12_examples")


def main():
    FIG.mkdir(exist_ok=True)
    d = dataset()
    steps = [lambda: fig_models(d), fig_sites, fig_site_environment, fig_partitions_examples,
             lambda: fig_distributions(d), lambda: fig_by_class(d), lambda: fig_composition(d),
             lambda: fig_size(d), lambda: fig_correlations(d), fig_robustness, fig_ml, fig_examples]
    for f in steps:
        try:
            f()
        except FileNotFoundError as exc:
            print("skipped:", exc)


if __name__ == "__main__":
    main()
