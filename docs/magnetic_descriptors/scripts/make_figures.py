"""All figures of the magnetic-descriptor report (PNG, 200 dpi, and vector PDF).

Inputs (../data/): analytic_*.csv (analytic_models.py), magnetic_dataset.csv
(dataset_table.py), slices.npz, radial_examples.csv, sites_examples.csv, examples.csv
(examples.py), region_shares.csv, radial_paw.csv (region_shares.py); and, from the
repository, paper/analysis/out/convergence.csv and partitions_summary.csv. Figures whose
inputs are missing are skipped with a note.

Usage:  python make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import SymLogNorm  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE.parent / "data", HERE.parent / "figures"
PAPER = HERE.parents[2] / "paper" / "analysis" / "out"
NAMES = ["m1_spin", "sigma_r2_spin", "f_bond_spin", "mu_site_std", "spin_frustration",
         "spin_charge_correlation"]
SHORT = {"m1_spin": r"$m_1^{s}$ (Å)", "sigma_r2_spin": r"$\sigma^2_{r,s}$ (Å$^2$)",
         "f_bond_spin": r"$f_{\mathrm{bond}}^{s}$", "mu_site_std": r"std$_i\,\mu_i$ ($\mu_B$)",
         "spin_frustration": r"$F_s$ (spin_frustration)", "spin_charge_correlation": r"$r(\rho,|m|)$",
         "M_abs_per_atom": r"$M_{\mathrm{abs}}$/atom ($\mu_B$)", "M_net_per_atom": r"$M_{\mathrm{net}}$/atom ($\mu_B$)"}
GROUPS = ["3d", "4d/5d", "4f", "5f"]
GCOL = dict(zip(GROUPS + ["sp"], ["C0", "C2", "C3", "C4", "0.5"]))
CLASSES = ["elemental", "intermetallic", "boride/carbide", "hydride", "pnictide", "chalcogenide",
           "oxide", "halide"]
CLS_COLOR = dict(zip(CLASSES, plt.cm.Dark2(np.arange(8))))
C1, C2 = 0.8, 1.5

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "figure.dpi": 100, "savefig.bbox": "tight"})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print("wrote", name)


def magnetic():
    d = pd.read_csv(DATA / "magnetic_dataset.csv")
    return d, d[d["magnetic"]]


# ------------------------------------------------------------------ 1 models
def fig_models():
    A, B, B2 = (pd.read_csv(DATA / f"analytic_{k}.csv") for k in ("A", "B", "B2"))
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.3))
    a = A["alpha_s"]
    for n, c in (("m1_spin", "C0"), ("sigma_r2_spin", "C1"), ("f_bond_spin", "C2")):
        ax[0].plot(a, A[f"{n}_exact"], "-", color=c, lw=1)
        ax[0].plot(a, A[n], "o", ms=4, color=c, label=SHORT[n])
    ax[0].set(xscale="log", xlabel=r"spin exponent $\alpha_s$ (Å$^{-2}$)", ylabel="value",
              title="(a) one magnetic atom: lines exact")
    ax[0].legend(frameon=False)
    ax[1].plot(a, A["spin_charge_correlation_exact"], "k-", lw=1, label="exact (box integrals)")
    ax[1].plot(a, A["spin_charge_correlation"], "o", color="C4", label="pydemi")
    ax[1].axvline(1.0, color="0.7", lw=0.8)
    ax[1].text(1.05, 0.52, r"$\alpha_s=\alpha_c$: $m\propto\rho$", fontsize=7)
    ax[1].set(xscale="log", xlabel=r"$\alpha_s$ (charge $\alpha_c$ = 1 Å$^{-2}$)", ylabel=r"$r(\rho,|m|)$",
              title="(b) spin-charge correlation")
    ax[1].legend(frameon=False)
    mb = B["mu_B_sublattice"]
    ax[2].plot(mb, B["spin_frustration_exact"], "k-", lw=1)
    ax[2].plot(mb, B["spin_frustration"], "o", color="C3", label="spin_frustration")
    ax[2].plot(mb, B["mu_site_std_exact"], "k--", lw=1)
    ax[2].plot(mb, B["mu_site_std"], "s", color="C1", label=r"mu_site_std ($\mu_B$)")
    ax[2].plot(mb, B["M_net_per_atom"], "^", color="C0", label=r"$M_{\mathrm{net}}$/atom")
    ax[2].set(xlabel=r"$\mu_B$ of the B sublattice ($\mu_A$ = +1)", ylabel="value",
              title="(b) two sublattices: FM → ferri → AFM")
    ax[2].legend(frameon=False, fontsize=6.5)
    p = B2["p_interstitial"]
    ax[3].plot(p, B2["M_net_per_atom"], "^-", color="C0", label=r"$M_{\mathrm{net}}$/atom")
    ax[3].plot(p, B2["M_abs_per_atom"], "v-", color="C9", label=r"$M_{\mathrm{abs}}$/atom")
    ax[3].plot(p, B2["spin_frustration"], "o-", color="C3", label="spin_frustration")
    ax[3].plot(p, B2["m1_spin"], "s-", color="C1", label="m1_spin (Å)")
    ax[3].set(xlabel=r"negative interstitial polarization per cell ($\mu_B$)", ylabel="value",
              title="(d) FM + antiparallel interstitial spin")
    ax[3].legend(frameon=False, fontsize=6.5)
    ax[2].set_title("(c) two sublattices: FM → ferri → AFM")
    fig.tight_layout()
    save(fig, "fig01_models")


# ------------------------------------------------------------------ 2 slices
def fig_slices():
    z = np.load(DATA / "slices.npz", allow_pickle=False)
    runs = list(dict.fromkeys(k.split("__")[0] for k in z.files))
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))
    for ax, run in zip(axes.flat, runs):
        X, Y, M = z[f"{run}__X"], z[f"{run}__Y"], z[f"{run}__m"]
        vmax = float(np.abs(M).max())
        norm = SymLogNorm(linthresh=max(vmax * 1e-3, 1e-5), vmin=-vmax, vmax=vmax, base=10)
        im = ax.pcolormesh(X, Y, M, cmap="PuOr_r", norm=norm, shading="auto", rasterized=True)
        ax.contour(X, Y, M, levels=[0], colors="k", linewidths=0.3)
        xmin, xmax, ymin, ymax = X.min(), X.max(), Y.min(), Y.max()
        for (x, y, R, mu), e in zip(z[f"{run}__atoms"], z[f"{run}__species"]):
            if xmin - R <= x <= xmax + R and ymin - R <= y <= ymax + R:
                ax.add_patch(plt.Circle((x, y), R, fill=False, color="k", lw=0.6))
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    ax.text(x, y, f"{e}\n{mu:+.2f}", fontsize=5.5, ha="center", va="center")
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xticks=[], yticks=[])
        ax.set_aspect("equal")
        meta = z[f"{run}__meta"]
        ax.set_title(f"{meta[0]} ({meta[1]} plane)", fontsize=8)
        cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cb.ax.tick_params(labelsize=5.5)
    fig.suptitle(r"Magnetization $m=\rho_\uparrow-\rho_\downarrow$ (e/Å$^3$, symmetric log; orange > 0, "
                 r"purple < 0) in a plane through the atom with the largest $|\mu_i|$. Circles: PAW spheres; "
                 r"labels: element and $\mu_i$ ($\mu_B$)", fontsize=8.5)
    save(fig, "fig02_slices")


# ------------------------------------------------------------------ 3 radial and sites
def fig_radial_and_sites():
    R = pd.read_csv(DATA / "radial_examples.csv")
    E = pd.read_csv(DATA / "examples.csv").set_index("id")
    S = pd.read_csv(DATA / "sites_examples.csv")
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True)
    for ax, (run, e) in zip(axes.flat, E.iterrows()):
        r = R[R["id"] == run]
        x = 0.5 * (r["r_lo"] + r["r_hi"])
        w = r["r_hi"] - r["r_lo"]
        ax.bar(x, r["positive"] / w, width=w, color="C1", alpha=0.85, label="m > 0")
        ax.bar(x, -r["negative"] / w, width=w, color="C4", alpha=0.85, label="m < 0")
        for v in (C1, C2):
            ax.axvline(v, color="0.5", ls=":", lw=0.9)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlim(0, 3)
        ax.set_title(f"{e['label']}\n$m_1^s$={e['m1_spin']:.2f} Å, $f^s_{{bond}}$={e['f_bond_spin']:.2f}, "
                     f"$r$={e['spin_charge_correlation']:.2f}", fontsize=7.5)
    for ax in axes[1]:
        ax.set_xlabel("distance to nearest nucleus r (Å)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"per Å of r, per $\mu_B$ of $\sum|m|dV$")
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    fig.suptitle("Radial profile of the magnetization (dotted: shells 0.8 and 1.5 Å)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig03_radial_examples")

    fig, axes = plt.subplots(2, 4, figsize=(13, 5.4))
    for ax, (run, e) in zip(axes.flat, E.iterrows()):
        s = S[S["id"] == run].sort_values(["element", "site"])
        cols = [plt.cm.tab10(list(dict.fromkeys(s["element"])).index(el)) for el in s["element"]]
        ax.bar(range(len(s)), s["mu"], color=cols)
        ax.axhline(0, color="k", lw=0.5)
        els = list(dict.fromkeys(s["element"]))
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=plt.cm.tab10(k)) for k in range(len(els))],
                  labels=els, frameon=False, fontsize=6.5)
        ax.set_xticks([])
        ax.set_title(f"{e['label']}: std {e['mu_site_std']:.2f} $\\mu_B$, $F_s$ {e['spin_frustration']:.3f}",
                     fontsize=7.5)
        ax.set_ylabel(r"$\mu_i$ ($\mu_B$)")
    fig.suptitle("Site moments (nearest-atom partition), one bar per site", fontsize=9)
    fig.tight_layout()
    save(fig, "fig04_site_moments")


# ------------------------------------------------------------------ 5 PAW and shells
def fig_where():
    S = pd.read_csv(DATA / "region_shares.csv")
    S = S[S["error"].isna()] if "error" in S else S
    S = S.assign(minority_share=np.minimum(S["negative_share"], 1 - S["negative_share"]))
    P = pd.read_csv(DATA / "radial_paw.csv")
    P = P[P["id"].isin(S["id"])].assign(x=lambda t: 0.5 * (t["x_lo"] + t["x_hi"]))
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
    q = P.groupby("x")["abs"].quantile([0.25, 0.5, 0.75]).unstack() / 0.05
    ax[0].plot(q.index, q[0.5], color="C1", label="median")
    ax[0].fill_between(q.index, q[0.25], q[0.75], color="C1", alpha=0.3, lw=0, label="IQR")
    ax[0].axvline(1, color="k", lw=0.8)
    ax[0].set(xlabel=r"$r/R_{\mathrm{PAW}}$ of the nearest atom", ylabel=r"$|m|$ per unit $r/R$, per $\mu_B$",
              title=f"(a) radial profile, {len(S)} magnetic structures")
    ax[0].legend(frameon=False)
    cols = ["abs_inside_paw", "vol_inside_paw", "abs_core", "abs_bond", "abs_int", "minority_share",
            "df_share_of_site_moments"]
    labs = ["|m| in\nPAW", "volume\nin PAW", "|m| core", "|m| bond", "|m| int", "minority-\nsign share",
            "d/f share\nof Σ|μ_i|"]
    ax[1].boxplot([S[c] for c in cols], showfliers=False, widths=0.6)
    ax[1].set_xticks(range(1, len(cols) + 1), labs, fontsize=7)
    ax[1].set(ylabel="share", ylim=(0, 1.02), title="(b) where the magnetization is")
    top = S["top_element"].value_counts().head(12)
    ax[2].bar(range(len(top)), top.values, color="C0")
    ax[2].set_xticks(range(len(top)), top.index)
    ax[2].set(ylabel="structures", title="(c) element carrying the largest |μ_i|")
    fig.tight_layout()
    save(fig, "fig05_where")


# ------------------------------------------------------------------ 6 distributions
def fig_distributions(d, m):
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    for ax, n in zip(axes.flat, NAMES + ["M_abs_per_atom", "M_net_per_atom"]):
        for g in GROUPS:
            s = m.loc[m["magnetic_group"] == g, n]
            ax.hist(s, bins=np.linspace(m[n].min(), m[n].max(), 45), histtype="step", color=GCOL[g],
                    label=f"{g} ({len(s)})")
        ax.set(xlabel=SHORT.get(n, n), ylabel="structures")
        ax.set_title(f"median {m[n].median():.3f}", fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=6.5, title="magnetic group", title_fontsize=6.5)
    fig.suptitle(f"Distributions over the {len(m):,} magnetic structures (of {len(d):,}), by magnetic-element group",
                 fontsize=9)
    fig.tight_layout()
    save(fig, "fig06_distributions")


def fig_by_group(m):
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.8))
    for k, n in enumerate(NAMES):
        for row, (key, order, colors) in enumerate((("magnetic_group", GROUPS, GCOL),
                                                    ("chem_class", CLASSES, CLS_COLOR))):
            ax = axes[row, k]
            gs = [g for g in order if (m[key] == g).sum() >= 5]
            bp = ax.boxplot([m.loc[m[key] == g, n] for g in gs], showfliers=False, widths=0.6, patch_artist=True)
            for p, g in zip(bp["boxes"], gs):
                p.set_facecolor(colors[g])
                p.set_alpha(0.7)
            ax.set_xticks(range(1, len(gs) + 1), [g.replace("boride/carbide", "B/C") for g in gs], rotation=45,
                          ha="right", fontsize=7)
            ax.set_title(SHORT[n], fontsize=8)
    fig.suptitle("Magnetic structures by magnetic-element group (top) and chemical class (bottom; classes with "
                 "≥ 5 magnetic structures)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig07_by_group_and_class")


# ------------------------------------------------------------------ 8 frustration and compensation
def fig_frustration(m):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for g in GROUPS:
        s = m[m["magnetic_group"] == g]
        ax[0].scatter(s["M_abs_per_atom"], s["spin_frustration"], s=4, color=GCOL[g], alpha=0.5, label=g,
                      rasterized=True)
        ax[1].scatter(1 - s["M_net_per_atom"] / s["M_abs_per_atom"], s["spin_frustration"], s=4, color=GCOL[g],
                      alpha=0.5, rasterized=True)
        ax[2].scatter(s["m1_spin"], s["spin_charge_correlation"], s=4, color=GCOL[g], alpha=0.5, rasterized=True)
    ax[0].set(xscale="log", xlabel=r"$M_{\mathrm{abs}}$ per atom ($\mu_B$)", ylabel="spin_frustration",
              title="(a) compensation against moment size")
    ax[0].legend(frameon=False, markerscale=3)
    ax[1].plot([0, 1], [0, 1], "k-", lw=0.6)
    ax[1].set(xlabel=r"$1-M_{\mathrm{net}}/M_{\mathrm{abs}}$ (voxel-level compensation)",
              ylabel="spin_frustration (site level)", title="(b) site-level vs voxel-level")
    ax[2].set(xlabel="m1_spin (Å)", ylabel=r"$r(\rho,|m|)$", title="(c) spin-charge correlation vs spin extent")
    fig.tight_layout()
    save(fig, "fig08_frustration_and_correlation")


def fig_correlations(m):
    cols = NAMES + ["M_abs_per_atom", "M_net_per_atom", "mu_site_max", "mu_site_min", "mu_between_element_var",
                    "m1", "f_bond", "zeta", "m1_site_std", "magpie_mean_NdUnfilled", "magpie_mean_NfUnfilled",
                    "magpie_mean_GSmagmom", "n_atoms"]
    c = m[cols].rank().corr()
    c.to_csv(DATA / "spearman_correlations.csv")
    fig, ax = plt.subplots(figsize=(8.5, 7))
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
    ax.set_title(f"Rank correlations over the {len(m):,} magnetic structures")
    save(fig, "fig09_correlations")


# ------------------------------------------------------------------ 10 robustness
def fig_robustness():
    S = pd.read_csv(DATA / "region_shares.csv")
    S = S[S["error"].isna()] if "error" in S else S
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    for t, c in zip(["abs_0.6_1.3", "abs_1.0_1.8", "scaled_0.6_1.3"], ("C0", "C1", "C3")):
        ax[0].scatter(S["f_bond_spin"], S[f"f_bond_spin@{t}"], s=8, color=c, alpha=0.6,
                      label=t.replace("abs_", "(").replace("scaled_", "scaled (").replace("_", ", ") + ")")
    ax[0].plot([0, 0.8], [0, 0.8], "k-", lw=0.6)
    ax[0].set(xlabel="f_bond_spin, default shells", ylabel="other shells", title="(a) shells")
    ax[0].legend(frameon=False, fontsize=6.5)
    for p, c in zip(["becke", "power", "hirshfeld"], ("C0", "C1", "C3")):
        ax[1].scatter(S["mu_site_std"], S[f"mu_site_std@{p}"], s=8, color=c, alpha=0.6, label=p)
        ax[2].scatter(S["spin_frustration"], S[f"spin_frustration@{p}"], s=8, color=c, alpha=0.6, label=p)
    for a, t in ((ax[1], "mu_site_std"), (ax[2], "spin_frustration")):
        lim = max(S[t].max(), 0.01)
        a.plot([0, lim], [0, lim], "k-", lw=0.6)
        a.set(xlabel=f"{t}, nearest-atom partition", ylabel="other partition", title=f"(b) partition: {t}")
        a.legend(frameon=False, fontsize=6.5)
    ax[2].set_title("(c) partition: spin_frustration")
    ax[2].set(xscale="symlog", yscale="symlog")
    ax[2].set_xscale("symlog", linthresh=1e-3)
    ax[2].set_yscale("symlog", linthresh=1e-3)
    conv = pd.read_csv(PAPER / "convergence.csv")
    conv = conv[conv["descriptor"].isin(NAMES) & (conv["full"] != 0)]
    g = conv.groupby("descriptor")["rel_change_x0.8"]
    names = [n for n in NAMES if n in g.groups]
    x = np.arange(len(names))
    ax[3].bar(x - 0.2, [g.get_group(n).abs().median() for n in names], 0.4, label="median")
    ax[3].bar(x + 0.2, [g.get_group(n).abs().quantile(0.9) for n in names], 0.4, label="90th pct")
    ax[3].set_xticks(x, names, rotation=60, fontsize=7)
    ax[3].set(yscale="log", ylabel="relative change",
              title=f"(d) grid 80% ({conv['id'].nunique()} magnetic of 30)")
    ax[3].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig10_robustness")


def fig_examples():
    E = pd.read_csv(DATA / "examples.csv")
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.4))
    x = np.arange(len(E))
    for ax, n in zip(axes.flat, NAMES + ["M_abs_per_atom", "abs_share_inside_paw"]):
        ax.bar(x, E[n], color="C0")
        ax.set_xticks(x, [l.split(" (")[0] for l in E["label"]], rotation=60, fontsize=7)
        ax.set_title(SHORT.get(n, "share of |m| inside PAW spheres"), fontsize=8)
    fig.tight_layout()
    save(fig, "fig11_examples")


def main():
    FIG.mkdir(exist_ok=True)
    d, m = magnetic()
    for f in [fig_models, fig_slices, fig_radial_and_sites, fig_where, lambda: fig_distributions(d, m),
              lambda: fig_by_group(m), lambda: fig_frustration(m), lambda: fig_correlations(m), fig_robustness,
              fig_examples]:
        try:
            f()
        except FileNotFoundError as exc:
            print("skipped:", exc)


if __name__ == "__main__":
    main()
