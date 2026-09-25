"""All figures of the ionicity / V_spread / rho_mid_std / lap_concentration report.

Inputs (../data/): analytic_*.csv, ionicity_dataset.csv, phillips_matched.csv,
calibration_fits.csv, ionicity_predictions.csv, potential_sample.csv, examples.csv,
sites_examples.csv, bonds_examples.csv, ml_scores_scratch.csv, ml_ionicity_scratch.csv;
paper/analysis/out/convergence.csv and derivatives_summary.csv.

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
PAPER = HERE.parents[2] / "paper" / "analysis" / "out"
CLASSES = ["elemental", "intermetallic", "boride/carbide", "hydride", "pnictide", "chalcogenide",
           "oxide", "halide"]
CLS_COLOR = dict(zip(CLASSES, plt.cm.Dark2(np.arange(8))))

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "figure.dpi": 100, "savefig.bbox": "tight"})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print("wrote", name)


def short(c):
    return c.replace("boride/carbide", "B/C")


def boxes(ax, d, n, log=False, order=CLASSES):
    gs = [g for g in order if (d["chem_class"] == g).sum() >= 5]
    bp = ax.boxplot([d.loc[d["chem_class"] == g, n].dropna() for g in gs], showfliers=False, widths=0.6,
                    patch_artist=True)
    for p, g in zip(bp["boxes"], gs):
        p.set_facecolor(CLS_COLOR[g])
        p.set_alpha(0.75)
    ax.set_xticks(range(1, len(gs) + 1), [short(g) for g in gs], rotation=45, ha="right", fontsize=7)
    if log:
        ax.set_yscale("log")


def fig_models():
    A, A2, B, C = (pd.read_csv(DATA / f"analytic_{k}.csv") for k in ("A", "A2", "B", "C"))
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.3))
    ax[0].plot(A["delta"], A["V_spread_exact"], "k-", lw=1, label="exact (reciprocal lattice sum)")
    ax[0].plot(A["delta"], A["V_spread"], "o", color="C0", label="pydemi")
    ax[0].set(xlabel=r"charge transfer $\delta$ (A: 1+$\delta$, B: 1$-\delta$ e)", ylabel="V_spread (eV)",
              title="(a) V_spread: charge transfer")
    ax[0].legend(frameon=False)
    ax[1].plot(A2["alpha_B"], A2["V_spread_exact"], "k-", lw=1, label="exact")
    ax[1].plot(A2["alpha_B"], A2["V_spread"], "s", color="C1", label="pydemi")
    ax[1].set(xlabel=r"$\alpha_B$ (Å$^{-2}$), $\alpha_A$ = 2, equal charges", ylabel="V_spread (eV)",
              title="(b) V_spread without charge transfer")
    ax[1].legend(frameon=False)
    ax[2].plot(B["c"], B["rho_mid_std_exact"], "k-", lw=1, drawstyle="default", label="exact")
    ax[2].plot(B["c"], B["rho_mid_std"], "o", color="C2", label="rho_mid_std")
    ax2 = ax[2].twinx()
    ax2.plot(B["c"], B["n_bond_types"], "d:", color="0.5", label="bond types in the census (right)")
    ax2.set_ylabel("bond types counted", color="0.5")
    ax2.set_ylim(0, 3)
    for v in (2.727, 3.3):
        ax[2].axvline(v, color="C3", ls="--", lw=0.8)
    ax[2].set(xlabel="c (Å), a = b = 3 Å", ylabel="rho_mid_std (e/Å$^3$)", title="(c) rho_mid_std and the census cutoff")
    h1, l1 = ax[2].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[2].legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.5, loc="upper center")
    g = C[C["case"].str.startswith("one")]
    r = C[C["case"].str.startswith("random")]
    x = np.arange(len(C))
    ax[3].bar(x, C["lap_concentration"], color="C0", label="lap_concentration")
    ax[3].bar(x, C["lap_concentration_valence"], color="C1", alpha=0.8, width=0.5, label="lap_concentration_valence")
    ax[3].plot(np.arange(len(r), len(C)), g["lap_concentration_valence_exact"], "k_", ms=14, mew=2,
               label="valence, exact (Gaussian atom)")
    ax[3].set_xticks(x, [c.replace("one Gaussian, alpha = ", "α=").replace("random ", "rand ") for c in C["case"]],
                     rotation=60, fontsize=6.5)
    ax[3].set(ylabel="value", title="(d) lap_concentration ≡ 1/2", ylim=(0, 0.6))
    ax[3].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig01_models")


def fig_vspread(d):
    S = pd.read_csv(DATA / "potential_sample.csv")
    S = S[S["error"].isna()] if "error" in S else S
    fig, ax = plt.subplots(1, 4, figsize=(14.5, 3.5))
    ax[0].scatter(d["zval_site_std"], d["V_spread"], s=2, c=[CLS_COLOR[c] for c in d["chem_class"]], alpha=0.35,
                  rasterized=True)
    r = d["zval_site_std"].rank().corr(d["V_spread"].rank())
    ax[0].set(xlabel="std over sites of ZVAL_i (e)", ylabel="V_spread (eV, Hartree)",
              title=f"(a) V_spread vs valence-count spread (Spearman {r:.2f})")
    for a, x, lab in ((ax[1], "net_charge_std", r"std of net charges $q_i$ (nearest atom)"),
                      (ax[2], "hirshfeld_std", "std of Hirshfeld charges")):
        a.scatter(S[x], S["V_spread_hartree"], s=10, color="C0", label="Hartree (dataset)")
        a.scatter(S[x], S["V_spread_esp"], s=10, color="C3", marker="^", label="Hartree + ions (esp)")
        rh = S[x].rank().corr(S["V_spread_hartree"].rank())
        re = S[x].rank().corr(S["V_spread_esp"].rank())
        a.set(xlabel=lab, ylabel="V_spread (eV)", title=f"Spearman: Hartree {rh:+.2f}, esp {re:+.2f}")
        a.legend(frameon=False, fontsize=6.5)
    ax[1].set_title(f"(b) {ax[1].get_title()}")
    ax[2].set_title(f"(c) {ax[2].get_title()}")
    boxes(ax[3], d, "V_spread")
    ax[3].set(ylabel="V_spread (eV)", title="(d) by chemical class")
    handles = [plt.Line2D([], [], ls="", marker="o", color=CLS_COLOR[k], label=k) for k in CLASSES]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.3, -0.1))
    fig.tight_layout()
    save(fig, "fig02_V_spread")


def fig_sites():
    S = pd.read_csv(DATA / "sites_examples.csv")
    Bd = pd.read_csv(DATA / "bonds_examples.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for k, (run, s) in enumerate(S.groupby("id", sort=False)):
        ax[0].scatter(s["zval"], s["V_hartree"], s=18, color=plt.cm.tab10(k % 10), label=run.split("_")[0])
        ax[1].scatter(s["q_hirshfeld"], s["V_esp"], s=18, color=plt.cm.tab10(k % 10))
    ax[0].set(xlabel="ZVAL of the site (e)", ylabel="Hartree potential at the nucleus (eV)",
              title="(a) site potentials follow the valence count")
    ax[0].legend(frameon=False, fontsize=6.3, ncol=2)
    ax[1].set(xlabel="Hirshfeld charge of the site (e)", ylabel="Hartree + ions at the nucleus (eV)",
              title="(b) full electrostatic site potential")
    runs = list(dict.fromkeys(Bd["id"]))
    for k, run in enumerate(runs):
        b = Bd[Bd["id"] == run]
        for j, (pair, bb) in enumerate(b.groupby("pair")):
            ax[2].scatter(np.full(len(bb), k) + 0.12 * (j - 0.5), bb["rho_mid"], s=14, color=plt.cm.tab10(j),
                          edgecolor="k", lw=0.3)
            ax[2].text(k + 0.12 * (j - 0.5) + 0.05, bb["rho_mid"].mean(), pair, fontsize=5.5)
    ax[2].set_xticks(range(len(runs)), [r.split("_")[0] for r in runs], rotation=60, fontsize=7)
    ax[2].set(ylabel=r"$\rho$ at the bond midpoint (e/Å$^3$)", title="(c) midpoint densities of the census bonds")
    fig.tight_layout()
    save(fig, "fig03_sites_and_bonds")


def fig_rho_mid_lap(d):
    fig, ax = plt.subplots(1, 4, figsize=(14.5, 3.5))
    boxes(ax[0], d, "rho_mid_std")
    ax[0].set(ylabel="rho_mid_std (e/Å$^3$)", title="(a) rho_mid_std by class")
    boxes(ax[1], d, "rho_mid_cv")
    ax[1].set(ylabel="rho_mid_std / rho_mid_mean", title="(b) relative spread")
    ax[2].hist(np.clip(np.abs(d["lap_concentration"] - 0.5), 1e-18, None), bins=np.logspace(-18, -14, 40), color="C0")
    ax[2].set(xscale="log", xlabel="|lap_concentration − 1/2| (exact zeros at 1e-18)", ylabel="structures",
              title=f"(c) lap_concentration, {len(d):,} structures")
    boxes(ax[3], d, "lap_concentration_valence")
    ax[3].set(ylabel="lap_concentration_valence", title="(d) the valence variant by class")
    fig.tight_layout()
    save(fig, "fig04_rho_mid_and_laplacian")


def fig_calibration():
    M = pd.read_csv(DATA / "phillips_matched.csv")
    F = pd.read_csv(DATA / "calibration_fits.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4))
    for a, tset in zip(ax[:2], ("verified", "all")):
        sub = M.dropna(subset=[f"grid_ionicity_loo_{tset}"])
        for st, mk in (("verified", "o"), ("recalled", "s")):
            s = sub[sub["status"] == st]
            a.scatter(s["f_i"], s[f"grid_ionicity_loo_{tset}"], marker=mk, s=22, color="C0" if st == "verified" else "C1",
                      label=f"{st} target", edgecolor="k", lw=0.3)
        for _, r in sub.iterrows():
            a.text(r["f_i"] + 0.005, r[f"grid_ionicity_loo_{tset}"], r["formula"], fontsize=5)
        a.plot([0, 1], [0, 1], "k-", lw=0.6)
        rm = F[(F["targets"] == tset) & F["features"].str.startswith("legacy")].iloc[0]
        a.set(xlabel="Phillips $f_i$", ylabel="grid_ionicity (leave-one-out)", xlim=(-0.05, 1.02), ylim=(-0.05, 1.05),
              title=f"({'ab'[tset == 'all']}) fitted on {tset} targets: LOO RMSE {rm['loo_rmse']:.3f}")
        a.legend(frameon=False, fontsize=6.5, loc="upper left")
    x = np.arange(F["features"].nunique())
    names = list(dict.fromkeys(F["features"]))
    for k, tset in enumerate(("verified", "all")):
        s = F[F["targets"] == tset].set_index("features").reindex(names)
        ax[2].barh(x + (k - 0.5) * 0.4, s["loo_rmse"], 0.4, label=f"{tset} targets")
    ax[2].set_yticks(x, names, fontsize=7)
    ax[2].invert_yaxis()
    ax[2].set(xlabel="leave-one-out RMSE", title="(c) feature sets for the calibration")
    ax[2].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig05_calibration")


def fig_ionicity_dataset():
    P = pd.read_csv(DATA / "ionicity_predictions.csv")
    fig, ax = plt.subplots(1, 4, figsize=(14.5, 3.5))
    for a, n, lab in ((ax[0], "pauling_ionicity", "Pauling ionicity"), (ax[1], "grid_ionicity", "grid_ionicity"),
                      (ax[2], "ionicity_residual", "ionicity_residual")):
        boxes(a, P, n)
        a.set(ylabel=lab, title=lab + " by class")
    ax[2].axhline(0, color="k", lw=0.6)
    ax[3].scatter(P["pauling_ionicity"], P["grid_ionicity"], s=2, c=[CLS_COLOR[c] for c in P["chem_class"]], alpha=0.35,
                  rasterized=True)
    ax[3].plot([0, 1], [0, 1], "k-", lw=0.6)
    r = P["pauling_ionicity"].rank().corr(P["grid_ionicity"].rank())
    ax[3].set(xlabel="Pauling ionicity", ylabel="grid_ionicity", title=f"(d) all 6,059 (Spearman {r:.2f})")
    fig.suptitle("The reconstructed calibrated ionicity applied to the whole dataset (verified-target calibration)",
                 fontsize=9)
    fig.tight_layout()
    save(fig, "fig06_ionicity_dataset")


def fig_correlations(d):
    P = pd.read_csv(DATA / "ionicity_predictions.csv")[["id", "grid_ionicity", "ionicity_residual"]]
    m = d.merge(P, on="id")
    cols = ["grid_ionicity", "ionicity_residual", "pauling_ionicity", "V_spread", "zval_site_std", "rho_mid_std",
            "rho_mid_mean", "rho_mid_cv", "lap_concentration_valence", "fint_over_lnf", "f_int", "lnf",
            "def_polarity_out", "rho_min_int_ratio", "bond_charge_transfer_pair_std", "ELF_bond_avg", "rho_perc_a",
            "magpie_mean_NValence", "n_atoms"]
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
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman rank correlation")
    ax.set_title(f"Rank correlations over the {len(m):,} structures")
    save(fig, "fig07_correlations")


def fig_robustness():
    conv = pd.read_csv(PAPER / "convergence.csv")
    names = ["V_spread", "rho_mid_mean", "rho_mid_std", "lap_concentration_valence", "lnf", "f_int"]
    conv = conv[conv["descriptor"].isin(names) & np.isfinite(conv["rel_change_x0.8"])]
    der = pd.read_csv(PAPER / "derivatives_summary.csv")
    der = der[der["descriptor"].isin(["lnf", "lap_concentration_valence"])]
    ml = pd.read_csv(DATA / "ml_scores_scratch.csv").set_index("descriptor")
    mi = pd.read_csv(DATA / "ml_ionicity_scratch.csv")
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    g = conv.groupby("descriptor")["rel_change_x0.8"]
    x = np.arange(len(names))
    ax[0].bar(x - 0.2, [max(g.get_group(n).abs().median(), 1e-7) for n in names], 0.4, label="median")
    ax[0].bar(x + 0.2, [max(g.get_group(n).abs().quantile(0.9), 1e-7) for n in names], 0.4, label="90th pct")
    ax[0].set_xticks(x, names, rotation=45, ha="right", fontsize=7)
    ax[0].set(yscale="log", ylabel="relative change", title="(a) grid 80% (30 structures)")
    ax[0].legend(frameon=False)
    for k, n in enumerate(["lnf", "lap_concentration_valence"]):
        s = der[der["descriptor"] == n]
        ax[1].bar(np.arange(len(s)) + (k - 0.5) * 0.4, s["median"], 0.4, label=n)
        ax[1].set_xticks(np.arange(len(s)), s["scheme"])
    ax[1].set(yscale="log", ylabel="median relative change vs FFT", title="(b) derivative scheme (200 structures)")
    ax[1].legend(frameon=False, fontsize=6.5)
    mn = ["V_spread", "rho_mid_mean", "rho_mid_std", "lap_concentration_valence", "f_int", "lnf"]
    ax[2].bar(np.arange(len(mn)), ml.loc[mn, "median_rel"], color="C2")
    ax[2].set_xticks(np.arange(len(mn)), mn, rotation=45, ha="right", fontsize=7)
    ax[2].set(yscale="log", ylabel="median |relative error|", title="(c) from ChargE3Net densities (605)")
    ax[3].scatter(mi["grid_ionicity_dft"], mi["grid_ionicity_ml"], s=6, alpha=0.5)
    ax[3].plot([0, 1], [0, 1], "k-", lw=0.6)
    ax[3].set(xlabel="grid_ionicity from the DFT density", ylabel="from the predicted density",
              title="(d) grid_ionicity, DFT vs ML")
    fig.tight_layout()
    save(fig, "fig08_robustness_and_ml")


def main():
    FIG.mkdir(exist_ok=True)
    d = pd.read_csv(DATA / "ionicity_dataset.csv")
    for f in [fig_models, lambda: fig_vspread(d), fig_sites, lambda: fig_rho_mid_lap(d), fig_calibration,
              fig_ionicity_dataset, lambda: fig_correlations(d), fig_robustness]:
        try:
            f()
        except FileNotFoundError as exc:
            print("skipped:", exc)


if __name__ == "__main__":
    main()
