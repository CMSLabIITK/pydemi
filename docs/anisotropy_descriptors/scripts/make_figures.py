"""All figures of the anisotropy-descriptor report (PNG, 200 dpi, and vector PDF).

Inputs (../data/): anisotropy_dataset.csv (classes.py), analytic_*.csv (analytic_examples.py),
slices.npz (slices.py), region_shares.csv (region_shares.py); and, from the repository,
paper/analysis/out/convergence.csv; derivative_sensitivity.csv (derivative_sensitivity.py) and
ml_scores_scratch.csv (ChargE3Net test-set scores, copied from the ML evaluation). Figures whose inputs are missing are skipped with a note.

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
REPO = HERE.parents[2]
CONV = REPO / "paper" / "analysis" / "out" / "convergence.csv"
ML_SCORE = DATA / "ml_scores_scratch.csv"
NAMES = ["zeta", "zeta_ELF", "T_eigenvalues_t1", "T_eigenvalues_t2", "T_eigenvalues_t3", "charge_FA"]
SHORT = {"zeta": r"$\zeta$", "zeta_ELF": r"$\zeta_{\mathrm{ELF}}$", "T_eigenvalues_t1": "$t_1$",
         "T_eigenvalues_t2": "$t_2$", "T_eigenvalues_t3": "$t_3$", "charge_FA": "FA"}
CLASSES = ["elemental", "intermetallic", "boride/carbide", "hydride", "pnictide", "chalcogenide",
           "oxide", "halide"]
SYSTEMS = ["cubic", "hexagonal", "trigonal", "tetragonal", "orthorhombic", "monoclinic", "triclinic"]
SYS_COLOR = dict(zip(SYSTEMS, plt.cm.tab10(np.arange(7))))
CLS_COLOR = dict(zip(CLASSES, plt.cm.Dark2(np.arange(8))))
FLOOR = 1e-9                       # for log axes: values below are drawn at the floor

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "figure.dpi": 100, "savefig.bbox": "tight"})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print("wrote", name)


def dataset():
    return pd.read_csv(DATA / "anisotropy_dataset.csv")


# ------------------------------------------------------------------ 1 analytic
def fig_analytic():
    A, B = pd.read_csv(DATA / "analytic_A.csv"), pd.read_csv(DATA / "analytic_B.csv")
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
    r = A["c_over_a"]
    ax[0].plot(r, A["charge_FA_exact"], "k-", lw=1, label="FA exact")
    ax[0].plot(r, A["charge_FA"], "o", color="C3", label="FA pydemi")
    for k, c in (("1", "C0"), ("3", "C2")):
        ax[0].plot(r, A[f"t{k}_exact"], "-", color=c, lw=1)
        ax[0].plot(r, A[f"T_eigenvalues_t{k}"], "s", ms=4, color=c, label=f"$t_{k}$ (line exact)")
    ax[0].set(xscale="log", xlabel="$c/a$ of $\\rho=e^{-(a x^2+a y^2+c z^2)}$", ylabel="value",
              title="(a) anisotropic Gaussian: tensor")
    ax[0].axvline(1, color="0.7", lw=0.8)
    ax[0].legend(frameon=False)
    ax[1].plot(r, A["zeta"], "o-", color="C1", label=r"$\zeta$")
    ax[1].plot(r, A["zeta_ELF"], "^-", color="C4", label=r"$\zeta_{\mathrm{ELF}}$")
    ax[1].set(xscale="log", xlabel="$c/a$", ylabel="value", title="(b) anisotropic Gaussian: $\\zeta$")
    ax[1].axvline(1, color="0.7", lw=0.8)
    ax[1].legend(frameon=False)
    for a in ax[:2]:
        a.set_xticks([0.25, 0.5, 1, 2, 4], ["1/4", "1/2", "1", "2", "4"])
        a.minorticks_off()
    d = B["separation_A"]
    ax[2].plot(d, B["zeta"], "o-", color="C1", label=r"$\zeta$")
    ax[2].plot(d, B["charge_FA"], "s-", color="C3", label="FA")
    ax[2].plot(d, B["zeta_ELF"], "^-", color="C4", label=r"$\zeta_{\mathrm{ELF}}$")
    ax[2].set(xlabel="separation of two Gaussian atoms (Å)", ylabel="value", title="(c) two atoms merging")
    ax[2].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig01_analytic")


# ------------------------------------------------------------------ 13 zeta_ELF baseline
def fig_zeta_elf_baseline():
    b = pd.read_csv(DATA / "analytic_zetaELF_baseline.csv")
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.0), sharey=True)
    for a, alpha in zip(ax, (1.0, 2.0)):
        sub = b[b.alpha == alpha]
        for k, (peak, g) in enumerate(sub.groupby("peak_e_per_A3")):
            a.plot(12.0 / g["n"], g["zeta_ELF"], "o-", color=plt.cm.viridis(k / 3),
                   label=f"peak {peak:g} e/Å$^3$")
        a.set(xlabel="grid spacing (Å)", title=f"spherical Gaussian, $\\alpha$ = {alpha:g} Å$^{{-2}}$")
        a.axhline(0, color="k", lw=0.6)
    ax[0].set_ylabel(r"$\zeta_{\mathrm{ELF}}$ (exact value 0)")
    ax[1].axhline(0.094, color="C3", ls="--", lw=0.8)
    ax[1].text(0.17, 0.1, "dataset median", color="C3", fontsize=7)
    ax[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig13_zetaELF_baseline")


# ------------------------------------------------------------------ 2, 3 slices
def fig_slices():
    z = np.load(DATA / "slices.npz", allow_pickle=False)
    runs = sorted({k.split("__")[0] for k in z.files}, key=lambda r: list(z.files).index(f"{r}__X"))
    for field, name in (("rho", "fig02_slices_density"), ("elf", "fig03_slices_elf")):
        fig, ax = plt.subplots(2, len(runs), figsize=(2.2 * len(runs), 4.6))
        for c, run in enumerate(runs):
            X, Y = z[f"{run}__X"], z[f"{run}__Y"]
            meta = z[f"{run}__meta"]
            if field == "rho":
                top = np.log10(np.clip(z[f"{run}__rho"], 1e-4, None))
                w, gn = z[f"{run}__w"], z[f"{run}__gradnorm"]
                lab, val = r"$\zeta$", float(meta[1])
                cmap, tl = "viridis", r"log$_{10}\rho$ (e/Å$^3$)"
            else:
                top = z[f"{run}__elf"]
                w, gn = z[f"{run}__w_elf"], z[f"{run}__gradnorm_elf"]
                lab, val = r"$\zeta_{\mathrm{ELF}}$", float(meta[2])
                cmap, tl = "magma", "ELF$_D$"
            m0 = ax[0, c].pcolormesh(X, Y, top, shading="auto", cmap=cmap)
            ax[0, c].set_title(f"{meta[0]}\n{lab} = {val:.3f}", fontsize=8)
            weight = gn / np.nanmax(gn)
            m1 = ax[1, c].pcolormesh(X, Y, np.where(weight > 1e-3, w, np.nan), shading="auto",
                                     cmap="coolwarm", vmin=0, vmax=1)
            for a in ax[:, c]:
                a.set_aspect("equal")
                a.set_xticks([])
                a.set_yticks([])
            ax[1, c].set_xlabel(f"plane ({meta[4]})", fontsize=7)
        fig.colorbar(m0, ax=ax[0, :], shrink=0.8, label=tl if field == "rho" else "ELF$_D$ (last panel scale)")
        fig.colorbar(m1, ax=ax[1, :], shrink=0.8, label=r"$1-|\nabla f\cdot\hat u|/|\nabla f|$")
        save(fig, name)


# ------------------------------------------------------------------ 4 distributions
def fig_distributions(d):
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.8))
    ax[0].hist(d["zeta"], bins=80, color="C1")
    ax[0].set(xlabel=r"$\zeta$", ylabel="structures", title=f"(a) median {d.zeta.median():.3f}")
    ax[1].hist(d["zeta_ELF"], bins=80, color="C4")
    ax[1].set(xlabel=r"$\zeta_{\mathrm{ELF}}$", title=f"(b) median {d.zeta_ELF.median():.3f}")
    ax[2].hist(np.log10(np.clip(d["charge_FA"], FLOOR, None)), bins=80, color="C3")
    ax[2].set(xlabel=r"log$_{10}$ FA (clipped at $10^{-9}$)", title="(c) charge_FA")
    spread = d["T_eigenvalues_t3"] - d["T_eigenvalues_t1"]
    ax[3].hist(np.log10(np.clip(spread, FLOOR, None)), bins=80, color="C0")
    ax[3].set(xlabel=r"log$_{10}(t_3-t_1)$", title="(d) eigenvalue spread")
    fig.tight_layout()
    save(fig, "fig04_distributions")


# ------------------------------------------------------------------ 5 by chemical class
def fig_by_class(d):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    for a, col, log in zip(ax, ["zeta", "zeta_ELF", "charge_FA"], [False, False, True]):
        vals = [np.clip(d.loc[d.chem_class == c, col], FLOOR, None) for c in CLASSES]
        bp = a.boxplot(vals, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], CLASSES):
            patch.set_facecolor(CLS_COLOR[c])
            patch.set_alpha(0.6)
        a.set_xticks(range(1, len(CLASSES) + 1),
                     [f"{c}\n({(d.chem_class == c).sum()})" for c in CLASSES], rotation=60, fontsize=7)
        a.set_title(SHORT[col] + ("  (log scale)" if log else ""))
        if log:
            a.set_yscale("log")
    fig.suptitle("Distributions by chemical class (box: quartiles, whiskers: 5th-95th percentile)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig05_by_class")


# ------------------------------------------------------------------ 6 by crystal system
def fig_by_system(d):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [2.2, 1]})
    rng = np.random.default_rng(0)
    for i, s in enumerate(SYSTEMS):
        sub = d[d.crystal_system_relaxed == s]
        for mag, mk in ((False, "o"), (True, "^")):
            y = np.clip(sub.loc[sub.magnetic == mag, "charge_FA"], FLOOR, None)
            x = i + (0.18 if mag else -0.18) + rng.uniform(-0.12, 0.12, len(y))
            ax[0].scatter(x, y, s=3, marker=mk, color=SYS_COLOR[s], alpha=0.5 if not mag else 0.8, lw=0)
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(len(SYSTEMS)),
                     [f"{s}\n({(d.crystal_system_relaxed == s).sum()})" for s in SYSTEMS], fontsize=7)
    ax[0].set_ylabel("charge_FA (clipped at $10^{-9}$)")
    ax[0].set_title("(a) by crystal system of the relaxed structure (left: non-magnetic o, right: magnetic ^)")
    ax[0].axhline(1e-4, color="k", lw=0.6, ls="--")
    ax[0].text(6.4, 1.3e-4, "cubic non-magnetic max", fontsize=6.5, ha="right")
    cub = d[d.crystal_system_relaxed == "cubic"]
    ax[1].scatter(cub.loc[cub.magnetic, "M_abs_per_atom"], np.clip(cub.loc[cub.magnetic, "charge_FA"], FLOOR, None),
                  s=6, color="C3", label="magnetic")
    ax[1].scatter(np.full((~cub.magnetic).sum(), 1e-3), np.clip(cub.loc[~cub.magnetic, "charge_FA"], FLOOR, None),
                  s=4, color="0.5", label="non-magnetic (x = 0)")
    ax[1].set(xscale="log", yscale="log", xlabel=r"$|M|$ per atom ($\mu_B$)", ylabel="charge_FA",
              title="(b) cubic structures only")
    ax[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig06_by_crystal_system")


# ------------------------------------------------------------------ 7 T shape
def fig_T_shape(d):
    t1, t2, t3 = d["T_eigenvalues_t1"], d["T_eigenvalues_t2"], d["T_eigenvalues_t3"]
    spread = t3 - t1
    ok = spread > 1e-6
    eta = ((t2 - t1) / spread)[ok]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for s in SYSTEMS[1:]:
        m = ok & (d.crystal_system_relaxed == s)
        ax[0].scatter(eta[m[ok]], d.loc[m, "charge_FA"], s=4, color=SYS_COLOR[s], label=s, alpha=0.7, lw=0)
    ax[0].set(yscale="log", xlabel=r"$\eta=(t_2-t_1)/(t_3-t_1)$", ylabel="charge_FA",
              title="(a) tensor shape (structures with $t_3-t_1>10^{-6}$)")
    ax[0].legend(frameon=False, markerscale=3, ncol=2)
    for s in ("hexagonal", "trigonal", "tetragonal", "orthorhombic", "monoclinic"):
        m = ok & (d.crystal_system_relaxed == s)
        ax[1].hist(eta[m[ok]], bins=40, histtype="step", color=SYS_COLOR[s], label=s, density=True)
    ax[1].set(xlabel=r"$\eta$", ylabel="density", title=r"(b) uniaxial systems sit at $\eta=0$ or $1$")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig07_T_shape")


# ------------------------------------------------------------------ 8 zeta vs zeta_ELF
def fig_zeta_pair(d):
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for c in CLASSES:
        m = d.chem_class == c
        ax.scatter(d.loc[m, "zeta"], d.loc[m, "zeta_ELF"], s=4, color=CLS_COLOR[c], label=c, alpha=0.6, lw=0)
    for name in ("Si_227", "NaCl_225", "Cu_225", "Al_225", "Ca2N_166", "BN_194", "Hf_194", "MgO_225"):
        r = d[d.id == name]
        if len(r):
            ax.annotate(name.split("_")[0], (r.zeta.iloc[0], r.zeta_ELF.iloc[0]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    rs = d["zeta"].rank().corr(d["zeta_ELF"].rank())
    ax.set(xscale="log", xlabel=r"$\zeta$", ylabel=r"$\zeta_{\mathrm{ELF}}$",
           title=f"$\\zeta$ vs $\\zeta_{{\\mathrm{{ELF}}}}$ (Spearman {rs:.2f})")
    ax.legend(frameon=False, markerscale=3, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig08_zeta_vs_zetaELF")


# ------------------------------------------------------------------ 9 correlations
def fig_correlations(d):
    others = ["m1", "f_core", "f_bond", "f_int", "lnf_charge_weighted", "ELF_bond_avg", "f_ELF_localized",
              "rho_min_int_ratio", "rho_int_mean", "shannon_entropy", "fisher_information", "perc_anisotropy",
              "def_polarity_out", "M_abs_per_atom", "zeta_site_std"]
    cols = NAMES + others
    C = d[cols].rank().corr().loc[NAMES, cols]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), [SHORT.get(c, c) for c in cols], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(NAMES)), [SHORT[c] for c in NAMES])
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="w" if abs(C.values[i, j]) > 0.6 else "k")
    fig.colorbar(im, ax=ax, label="Spearman", shrink=0.9)
    ax.set_title("Rank correlations over the 6,059 structures")
    fig.tight_layout()
    save(fig, "fig09_correlations")
    C.to_csv(DATA / "spearman_correlations.csv")


# ------------------------------------------------------------------ 10 robustness
def robustness_table():
    rows = {n: {} for n in NAMES}
    if CONV.exists():
        c = pd.read_csv(CONV)
        c = c[np.isfinite(c["rel_change_x0.8"])]
        med = c.groupby("descriptor")["rel_change_x0.8"].median()
        for n in NAMES:
            rows[n]["grid 80% (median rel. change)"] = med.get(n, np.nan)
    sens = DATA / "derivative_sensitivity.csv"
    if sens.exists():
        df = pd.read_csv(sens)
        df = df[df["error"].isna()] if "error" in df else df
        for n in NAMES:
            ref = df[f"{n}@fft"]
            for tag, lab in (("fd2", "FFT vs FD2"), ("fd4", "FFT vs FD4"), ("fd8", "FFT vs FD8"),
                             ("fft_diagonal", "diagonal Laplacian")):
                v = df[f"{n}@{tag}"]
                rel = (v - ref).abs() / np.maximum(np.maximum(v.abs(), ref.abs()), 1e-12)
                rows[n][f"{lab} (median rel. change)"] = rel.median()
    if ML_SCORE.exists():
        s = pd.read_csv(ML_SCORE).set_index("descriptor")
        for n in NAMES:
            if n in s.index:
                rows[n]["ML vs DFT Spearman"] = s.loc[n, "spearman"]
                rows[n]["ML vs DFT (median rel. error)"] = s.loc[n, "median_rel"]
    t = pd.DataFrame(rows).T
    t.to_csv(DATA / "robustness.csv")
    return t


def fig_robustness():
    t = robustness_table()
    change = [c for c in t.columns if "rel. change" in c]
    if not change:
        print("skip fig10: no robustness inputs yet")
        return
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.2), gridspec_kw={"width_ratios": [2, 1]})
    x = np.arange(len(NAMES))
    wdt = 0.8 / len(change)
    for k, c in enumerate(change):
        ax[0].bar(x + (k - (len(change) - 1) / 2) * wdt, np.clip(t[c].astype(float), 1e-9, None), wdt,
                  label=c.replace(" (median rel. change)", ""))
    ax[0].set(yscale="log", ylabel="median relative change", title="(a) numerical sensitivity")
    ax[0].set_xticks(x, [SHORT[n] for n in NAMES])
    ax[0].axhline(0.01, color="k", lw=0.6, ls="--")
    ax[0].legend(frameon=False, fontsize=7)
    if "ML vs DFT Spearman" in t:
        ax[1].bar(x, t["ML vs DFT Spearman"].astype(float), color="C5")
        ax[1].set(ylim=(0, 1.02), ylabel="Spearman", title="(b) ML density (ChargE3Net) vs DFT, 605 tests")
        ax[1].set_xticks(x, [SHORT[n] for n in NAMES])
    fig.tight_layout()
    save(fig, "fig10_robustness")


# ------------------------------------------------------------------ 11 regions
def fig_regions():
    p = DATA / "region_shares.csv"
    if not p.exists():
        print("skip fig11: region_shares.csv missing")
        return
    r = pd.read_csv(p)
    r = r[r["error"].isna()] if "error" in r else r
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].boxplot([r.volume_inside, r.grad_share_inside, r.grad2_share_inside], whis=(5, 95), showfliers=False)
    ax[0].set_xticks([1, 2, 3], ["volume", r"$\sum|\nabla\rho|$", r"$\sum|\nabla\rho|^2$"])
    ax[0].set(ylabel="share inside PAW spheres", title=f"(a) PAW spheres ({len(r)} structures)")
    for a, (x, y, lab) in zip(ax[1:], (("zeta", "zeta_outside", r"$\zeta$"),
                                      ("zeta_ELF", "zeta_ELF_high_density", r"$\zeta_{\mathrm{ELF}}$"))):
        a.scatter(r[x], r[y], s=8, color="C1" if x == "zeta" else "C4")
        lim = [0, max(r[x].max(), r[y].max()) * 1.05]
        a.plot(lim, lim, "k-", lw=0.6)
        rs = r[x].rank().corr(r[y].rank())
        a.set(xlabel=f"{lab} (whole cell)",
              ylabel=f"{lab} outside PAW spheres" if x == "zeta" else f"{lab} over " + r"$\rho\geq0.01$ e/Å$^3$",
              title=f"({'b' if x == 'zeta' else 'c'}) Spearman {rs:.2f}")
    fig.tight_layout()
    save(fig, "fig11_region_shares")


# ------------------------------------------------------------------ 12 examples
EXAMPLES = ["Cu_225", "CaF2_225", "KCl_225", "NaCl_225", "MgO_225", "ZnO_186", "GaN_186", "Ca2N_166",
            "Al_225", "Mg_194", "GaAs_216", "SiC_216", "Si_227", "Te_152", "Bi2Te3_166", "BN_194", "C_166"]


def fig_examples(d):
    e = d.set_index("id").loc[[x for x in EXAMPLES if x in set(d.id)]]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    y = np.arange(len(e))
    for a, col, c in zip(ax, ["zeta", "zeta_ELF", "charge_FA"], ["C1", "C4", "C3"]):
        a.barh(y, e[col], color=c)
        a.set_title(SHORT[col])
        for yi, v in zip(y, e[col]):
            a.text(v, yi, f" {v:.3f}", va="center", fontsize=6.5)
        a.set_xlim(0, e[col].max() * 1.3)
    ax[0].set_yticks(y, [f"{i.split('_')[0]} ({e.loc[i, 'crystal_system_relaxed']})" for i in e.index], fontsize=7)
    ax[0].invert_yaxis()
    fig.tight_layout()
    save(fig, "fig12_examples")
    e[["chem_class", "crystal_system_relaxed", "magnetic"] + NAMES].to_csv(DATA / "examples.csv")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    d = dataset()
    fig_analytic()
    fig_zeta_elf_baseline()
    fig_slices()
    fig_distributions(d)
    fig_by_class(d)
    fig_by_system(d)
    fig_T_shape(d)
    fig_zeta_pair(d)
    fig_correlations(d)
    fig_robustness()
    fig_regions()
    fig_examples(d)
