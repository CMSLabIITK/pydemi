"""All figures of the deformation-descriptor report (PNG, 200 dpi, and vector PDF).

Inputs (../data/): analytic_*.csv (analytic_models.py), deformation_dataset.csv
(dataset_table.py), slices.npz, radial_examples.csv, examples.csv (examples.py),
region_shares.csv, radial_paw.csv (region_shares.py), robust_*.csv (robustness.py),
ml_scores_*.csv (ChargE3Net test-set scores, copied from the ML evaluation). Figures whose
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
DEF = ["m1_def", "m2_def", "sigma_r2_def", "f_bond_def", "f_int_def", "f_bond_dep", "def_polarity"]
SHORT = {"m1_def": r"$m_1^{\Delta}$", "m2_def": r"$m_2^{\Delta}$", "sigma_r2_def": r"$\sigma_{r,\Delta}^2$",
         "f_bond_def": r"$f_{\mathrm{bond}}^{+}$", "f_int_def": r"$f_{\mathrm{int}}^{+}$",
         "f_bond_dep": r"$f_{\mathrm{bond}}^{-}$", "def_polarity": r"$P_{\Delta}$"}
UNITS = {"m1_def": "Å", "m2_def": "Å$^2$", "sigma_r2_def": "Å$^2$"}
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


def label(n):
    return SHORT[n] + (f" ({UNITS[n]})" if n in UNITS else "")


def g(r, a):
    return (a / np.pi) ** 1.5 * np.exp(-a * r * r)


# ------------------------------------------------------------------ 1 breathing atom
def fig_breathing():
    A = pd.read_csv(DATA / "analytic_A.csv")
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.1))
    r = np.linspace(0, 3, 600)
    for s, c in ((0.6, "C0"), (1.5, "C3")):
        ax[0].plot(r, 4 * np.pi * r * r * (g(r, 2 * s) - g(r, 2.0)), color=c,
                   label=f"s = {s} ({'expanded' if s < 1 else 'contracted'})")
    ax[0].axhline(0, color="k", lw=0.6)
    for x in (C1, C2):
        ax[0].axvline(x, color="0.6", ls=":", lw=0.8)
    ax[0].text(0.35, 0.02, "core", transform=ax[0].get_xaxis_transform(), ha="center", fontsize=7)
    ax[0].text(1.15, 0.02, "bond", transform=ax[0].get_xaxis_transform(), ha="center", fontsize=7)
    ax[0].text(2.2, 0.02, "interstitial", transform=ax[0].get_xaxis_transform(), ha="center", fontsize=7)
    ax[0].set(xlabel="r (Å)", ylabel=r"$4\pi r^2\,\Delta\rho(r)$ (e/Å)", title="(a) radial deformation")
    ax[0].legend(frameon=False)
    s = A["s"]
    for n, c in (("m1_def", "C0"), ("m2_def", "C1"), ("sigma_r2_def", "C2")):
        ax[1].plot(s, A[f"{n}_exact"], "-", color=c, lw=1)
        ax[1].plot(s, A[n], "o", ms=4, color=c, label=SHORT[n])
    ax[1].set(xscale="log", xlabel="scale s of the exponent", ylabel="Å or Å$^2$",
              title="(b) radial moments")
    for n, c in (("f_bond_def", "C3"), ("f_int_def", "C4"), ("f_bond_dep", "C5")):
        ax[2].plot(s, A[f"{n}_exact"], "-", color=c, lw=1)
        ax[2].plot(s, A[n], "o", ms=4, color=c, label=SHORT[n])
    ax[2].set(xscale="log", xlabel="scale s", ylabel="share", title="(c) shell shares")
    ax[3].plot(s, A["def_polarity_exact"], "k-", lw=1, label="exact (quadrature)")
    ax[3].plot(s, A["def_polarity"], "o", ms=4, color="C6", label="pydemi")
    ax[3].set(xscale="log", xlabel="scale s", ylabel=SHORT["def_polarity"], title="(d) deformation polarity")
    for a in ax[1:]:
        a.axvline(1, color="0.7", lw=0.8)
        a.set_xticks([0.4, 0.5, 0.7, 1, 1.5, 2, 2.5], ["0.4", "0.5", "0.7", "1", "1.5", "2", "2.5"])
        a.minorticks_off()
        a.legend(frameon=False)
    fig.suptitle("Breathing atom: crystal $g(r; 2s)$ against reference $g(r; 2)$ "
                 "(lines: exact radial quadrature; markers: pydemi on a 0.1 Å grid)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig01_breathing_atom")


# ------------------------------------------------------------------ 2 bond charge / transfer
def fig_bond_charge():
    B1, B2, C = (pd.read_csv(DATA / f"analytic_{k}.csv") for k in ("B1", "B2", "C"))
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.1))
    x = np.linspace(-2.5, 2.5, 500)
    d, q = 2.4, 0.2
    ga, gb, gm = g(np.abs(x + d / 2), 2.0), g(np.abs(x - d / 2), 2.0), g(np.abs(x), 3.0)
    ax[0].plot(x, (1 - q / 2) * (ga + gb) + q * gm, color="k", lw=1, label=r"$\rho$ (crystal)")
    ax[0].plot(x, ga + gb, color="0.5", ls="--", lw=1, label="promolecule")
    ax[0].plot(x, 5 * ((1 - q / 2) * (ga + gb) + q * gm - ga - gb), color="C3", label=r"$5\times\Delta\rho$")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set(xlabel="position along the bond (Å)", ylabel="e/Å$^3$",
              title=f"(a) bond-charge model, d = {d} Å, q = {q}")
    ax[0].legend(frameon=False, loc="upper right")
    dd = B1["d_A"]
    for n, c in (("f_bond_def", "C3"), ("f_int_def", "C4"), ("f_bond_dep", "C5")):
        ax[1].plot(dd, B1[n], "o-", ms=3.5, color=c, label=SHORT[n])
    for v in (2 * C1, 2 * C2):
        ax[1].axvline(v, color="0.6", ls=":", lw=0.9)
    ax[1].text(2 * C1 + 0.05, 0.93, "midpoint\nat 0.8 Å", fontsize=6.5, ha="left", transform=ax[1].get_xaxis_transform())
    ax[1].text(2 * C2 + 0.05, 0.93, "midpoint\nat 1.5 Å", fontsize=6.5, ha="left", transform=ax[1].get_xaxis_transform())
    ax[1].set(xlabel="separation d (Å)", ylabel="share", title="(b) shell shares vs bond length (q = 0.2)")
    ax[1].legend(frameon=False)
    ax2 = ax[2]
    ax2.plot(dd, B1["m1_def"], "o-", ms=3.5, color="C0", label=SHORT["m1_def"] + " (Å)")
    ax2.plot(dd, B1["sigma_r2_def"], "s-", ms=3.5, color="C2", label=SHORT["sigma_r2_def"] + " (Å$^2$)")
    ax2.plot(dd, B1["def_polarity"], "^-", ms=3.5, color="C6", label=SHORT["def_polarity"])
    ax2.set(xlabel="separation d (Å)", ylabel="value", title="(c) moments and polarity vs bond length")
    ax2.legend(frameon=False)
    b = B2[B2["q"] > 0]
    ax[3].plot(b["q"], b["def_polarity"], "o-", color="C6", label=SHORT["def_polarity"] + ", bond charge")
    c = C[C["q"] > 0]
    ax[3].plot(c["q"], c["def_polarity"], "s-", color="C8", label=SHORT["def_polarity"] + ", charge transfer")
    ax[3].plot(b["q"], b["f_bond_def"], "o--", color="C3", mfc="none", label=SHORT["f_bond_def"] + ", bond charge")
    ax[3].plot(b["q"], b["m1_def"], "o--", color="C0", mfc="none", label=SHORT["m1_def"] + ", bond charge")
    ax[3].plot(c["q"], c["m1_def"], "s--", color="C9", mfc="none", label=SHORT["m1_def"] + ", transfer")
    ax[3].set(xlabel="electrons moved q", ylabel="value", title="(d) only the polarity scales with q")
    ax[3].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig02_bond_charge_models")


# ------------------------------------------------------------------ 3 slices
def fig_slices():
    z = np.load(DATA / "slices.npz", allow_pickle=False)
    runs = sorted({k.split("__")[0] for k in z.files}, key=lambda r: list(z.files).index(f"{r}__X"))
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
    norm = SymLogNorm(linthresh=0.005, vmin=-0.5, vmax=0.5, base=10)
    for ax, run in zip(axes.flat, runs):
        X, Y, D = z[f"{run}__X"], z[f"{run}__Y"], z[f"{run}__drho"]
        meta = z[f"{run}__meta"]
        m = ax.pcolormesh(X, Y, D, cmap="RdBu_r", norm=norm, shading="auto", rasterized=True)
        ax.contour(X, Y, D, levels=[0], colors="k", linewidths=0.3)
        atoms, sp = z[f"{run}__atoms"], z[f"{run}__species"]
        xmin, xmax, ymin, ymax = X.min(), X.max(), Y.min(), Y.max()
        for (x, y, R, _), e in zip(atoms, sp):
            if xmin - R <= x <= xmax + R and ymin - R <= y <= ymax + R:
                ax.add_patch(plt.Circle((x, y), R, fill=False, color="k", lw=0.7))
                ax.add_patch(plt.Circle((x, y), C2, fill=False, color="0.35", lw=0.5, ls=":"))
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    ax.text(x, y, e, fontsize=6, ha="center", va="center", color="k")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title(f"{meta[0]} ({meta[1]} plane)")
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(m, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label(r"$\Delta\rho = \rho_{\mathrm{CHGCAR}} - \rho_{\mathrm{pro}}$ (e/Å$^3$, symmetric log)")
    fig.suptitle("Deformation density in a lattice plane through an atom. Solid circles: PAW augmentation "
                 "spheres; dotted: the 1.5 Å outer edge of the bond shell; black line: Δρ = 0", fontsize=9)
    save(fig, "fig03_slices")


# ------------------------------------------------------------------ 4 radial profiles of examples
def fig_radial_examples():
    R = pd.read_csv(DATA / "radial_examples.csv")
    E = pd.read_csv(DATA / "examples.csv").set_index("id")
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True)
    for ax, (run, e) in zip(axes.flat, E.iterrows()):
        r = R[R["id"] == run]
        x = 0.5 * (r["r_lo"] + r["r_hi"])
        w = r["r_hi"] - r["r_lo"]
        ax.bar(x, r["accumulated_per_e"] / w, width=w, color="C3", alpha=0.8, label=r"accumulated ($\Delta\rho>0$)")
        ax.bar(x, -r["depleted_per_e"] / w, width=w, color="C0", alpha=0.8, label=r"depleted ($\Delta\rho<0$)")
        ax.axhline(0, color="k", lw=0.5)
        for v in (C1, C2):
            ax.axvline(v, color="0.5", ls=":", lw=0.9)
        ax.axvspan(e["R_paw_min"], e["R_paw_max"], color="0.85", zorder=0)
        ax.axvline(e["R_paw_min"], color="k", lw=0.8)
        ax.set_title(f"{e['label']}\n$P_\\Delta$={e['def_polarity']:.3f}, in PAW spheres "
                     f"{e['abs_share_inside_paw']:.0%} of $|\\Delta\\rho|$", fontsize=8)
        ax.set_xlim(0, 3.5)
    for ax in axes[1]:
        ax.set_xlabel("distance to nearest nucleus r (Å)")
    for ax in axes[:, 0]:
        ax.set_ylabel("charge per Å of r, per electron")
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    fig.suptitle("Where the deformation sits. Dotted: shell boundaries 0.8 and 1.5 Å; "
                 "black line / grey band: PAW radii of the elements present", fontsize=9)
    fig.tight_layout()
    save(fig, "fig04_radial_examples")


# ------------------------------------------------------------------ 5 PAW dominance
def fig_paw():
    S = pd.read_csv(DATA / "region_shares.csv")
    S = S[S["error"].isna()] if "error" in S else S
    P = pd.read_csv(DATA / "radial_paw.csv")
    P = P[P["id"].isin(S["id"])]
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.3))
    x = 0.5 * (P["x_lo"] + P["x_hi"])
    P = P.assign(x=x)
    w = 0.05
    for col, c, sign, lab in (("acc", "C3", 1, "accumulated"), ("dep", "C0", -1, "depleted")):
        q = P.groupby("x")[col].quantile([0.25, 0.5, 0.75]).unstack() / w
        ax[0].plot(q.index, sign * q[0.5], color=c, label=f"{lab} (median)")
        ax[0].fill_between(q.index, sign * q[0.25], sign * q[0.75], color=c, alpha=0.25, lw=0)
    ax[0].axvline(1, color="k", lw=0.8)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set(xlabel=r"$r / R_{\mathrm{PAW}}$ of the nearest atom", ylabel="charge per unit r/R, per electron",
              title=f"(a) radial profile, {len(S)} random structures")
    ax[0].legend(frameon=False)
    cols = ["abs_inside_paw", "acc_inside_paw", "dep_inside_paw", "m1_numerator_inside_paw", "vol_inside_paw",
            "bond_shell_inside_paw"]
    labs = [r"$|\Delta\rho|$", r"$\Delta\rho^{+}$", r"$\Delta\rho^{-}$", r"$\sum|\Delta\rho|\,r$", "volume",
            "bond-shell\nvolume"]
    ax[1].boxplot([S[c] for c in cols], showfliers=False, widths=0.6)
    ax[1].set_xticks(range(1, len(cols) + 1), labs, fontsize=7.5)
    ax[1].set(ylabel="share inside the PAW spheres", title="(b) share inside the augmentation spheres", ylim=(0, 1))
    shares = pd.DataFrame({"core": [S["acc_core"].median(), S["dep_core"].median(), S["vol_core"].median()],
                           "bond": [S["acc_bond"].median(), S["dep_bond"].median(), S["vol_bond"].median()],
                           "interstitial": [S["acc_int"].median(), S["dep_int"].median(), S["vol_int"].median()]},
                          index=["accumulated", "depleted", "volume"])
    shares = shares.div(shares.sum(axis=1), axis=0)
    left = np.zeros(3)
    for k, c in zip(shares.columns, ("0.35", "C1", "C2")):
        ax[2].barh(shares.index, shares[k], left=left, color=c, label=k)
        for yy, (l, v) in enumerate(zip(left, shares[k])):
            if v > 0.06:
                ax[2].text(l + v / 2, yy, f"{v:.2f}", ha="center", va="center", fontsize=7, color="w")
        left += shares[k].values
    ax[2].set(xlabel="share (medians, renormalized)", title="(c) split over the three shells")
    ax[2].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=7)
    fig.tight_layout()
    save(fig, "fig05_paw_dominance")


# ------------------------------------------------------------------ 6 distributions
def fig_distributions(d):
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.4))
    for ax, n in zip(axes.flat, DEF):
        lo = min(d[n].quantile(0.001), d[n + "_out"].quantile(0.001))
        hi = max(d[n].quantile(0.999), d[n + "_out"].quantile(0.999))
        bins = np.linspace(lo, hi, 60)
        ax.hist(d[n], bins=bins, color="C0", alpha=0.7, label="whole cell")
        ax.hist(d[n + "_out"], bins=bins, color="C1", alpha=0.6, label="outside PAW spheres (_out)")
        ax.set(xlabel=label(n), ylabel="structures")
        ax.set_title(f"{n}: median {d[n].median():.3f} / _out {d[n + '_out'].median():.3f}", fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    ax = axes.flat[-1]
    ax.hist(d["def_out_volume_fraction"], bins=50, color="C2")
    ax.set(xlabel="def_out_volume_fraction", ylabel="structures",
           title="volume outside the PAW spheres")
    fig.suptitle(f"Distributions over the {len(d):,} structures", fontsize=9)
    fig.tight_layout()
    save(fig, "fig06_distributions")


def _boxes(d, cols, name, title):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    groups = [c for c in CLASSES if (d["chem_class"] == c).any()]
    for ax, n in zip(axes.flat, cols):
        data = [d.loc[d["chem_class"] == c, n].dropna() for c in groups]
        bp = ax.boxplot(data, showfliers=False, widths=0.6, patch_artist=True)
        for p, c in zip(bp["boxes"], groups):
            p.set_facecolor(CLS_COLOR[c])
            p.set_alpha(0.75)
        ax.set_xticks(range(1, len(groups) + 1), [c.replace("boride/carbide", "B/C") for c in groups],
                      rotation=45, ha="right", fontsize=7)
        base = n.replace("_out", "")
        ax.set_ylabel(label(base) + (" (_out)" if n.endswith("_out") else ""))
    ax = axes.flat[-1]
    cnt = d["chem_class"].value_counts().reindex(groups)
    ax.bar(range(len(groups)), cnt.values, color=[CLS_COLOR[c] for c in groups])
    ax.set_xticks(range(len(groups)), [c.replace("boride/carbide", "B/C") for c in groups], rotation=45,
                  ha="right", fontsize=7)
    ax.set_ylabel("structures")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    save(fig, name)


def fig_by_class(d):
    _boxes(d, DEF, "fig07_by_class", "Whole-cell deformation descriptors by chemical class "
           "(boxes: quartiles; whiskers: 1.5 IQR)")
    _boxes(d, [n + "_out" for n in DEF], "fig08_by_class_out",
           "The same descriptors over the voxels outside the PAW spheres (paw extension)")


# ------------------------------------------------------------------ 9 ionicity and block
def fig_ionicity(d):
    fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.3))
    dd = d[d["n_elements"] > 1]
    bins = np.arange(0, 3.41, 0.2)
    dd = dd.assign(chi_bin=pd.cut(dd["delta_chi"], bins))
    for a, n in zip(ax[:3], ["def_polarity", "def_polarity_out", "f_bond_dep"]):
        a.scatter(dd["delta_chi"], dd[n], s=2, c=[CLS_COLOR[c] for c in dd["chem_class"]], alpha=0.35,
                  rasterized=True)
        med = dd.groupby("chi_bin", observed=True)[n].median()
        a.plot([iv.mid for iv in med.index], med.values, "k-o", ms=3, lw=1.2, label="binned median")
        rho_s = dd["delta_chi"].rank().corr(dd[n].rank())
        a.set(xlabel=r"$\Delta\chi$ (Pauling, max − min)", ylabel=n,
              title=f"{n} (Spearman {rho_s:+.2f})")
        a.legend(frameon=False)
    groups = ["sp", "d", "f"]
    for k, n in enumerate(["def_polarity", "def_polarity_out"]):
        data = [d.loc[d["heaviest_block"] == b, n] for b in groups]
        pos = np.arange(3) + (k - 0.5) * 0.35
        bp = ax[3].boxplot(data, positions=pos, widths=0.3, showfliers=False, patch_artist=True)
        for p in bp["boxes"]:
            p.set_facecolor(f"C{k}")
            p.set_alpha(0.6)
    ax[3].set_xticks(range(3), ["sp only", "has d (no f)", "has f"])
    ax[3].set(ylabel="value", yscale="log", title="polarity by heaviest block (blue whole, orange _out)")
    handles = [plt.Line2D([], [], ls="", marker="o", color=CLS_COLOR[c], label=c) for c in CLASSES]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "fig09_ionicity_and_block")


# ------------------------------------------------------------------ 10 correlations
def fig_correlations(d):
    cols = DEF + [n + "_out" for n in DEF] + ["m1", "sigma_r2", "f_bond", "f_int", "ELF_bond_avg",
                                              "rho_mid_mean", "bond_charge_transfer_pair_std", "zeta",
                                              "delta_chi", "magpie_mean_NValence"]
    c = d[cols].rank().corr()
    c.to_csv(DATA / "spearman_correlations.csv")
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols)), cols, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{c.values[i, j]:.1f}".replace("0.", ".").replace("-.", "−."), ha="center",
                    va="center", fontsize=5, color="w" if abs(c.values[i, j]) > 0.6 else "k")
    for v in (6.5, 13.5):
        ax.axhline(v, color="k", lw=0.8)
        ax.axvline(v, color="k", lw=0.8)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman rank correlation")
    ax.set_title(f"Rank correlations over the {len(d):,} structures")
    save(fig, "fig10_correlations")


# ------------------------------------------------------------------ 11 descriptor maps
def fig_maps(d):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for cls in CLASSES:
        s = d[d["chem_class"] == cls]
        kw = dict(s=3, color=CLS_COLOR[cls], alpha=0.45, label=cls, rasterized=True)
        ax[0].scatter(s["m1_def"], s["sigma_r2_def"], **kw)
        ax[1].scatter(s["f_bond_def"], s["f_bond_dep"], **kw)
        ax[2].scatter(s["m1_def_out"], s["def_polarity_out"], **kw)
    ax[0].set(xlabel=label("m1_def"), ylabel=label("sigma_r2_def"), title="(a) where, and how spread (whole cell)")
    ax[1].set(xlabel=label("f_bond_def"), ylabel=label("f_bond_dep"), title="(b) bond-shell shares (whole cell)")
    ax[1].plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
    ax[2].set(xlabel=r"$m_1^{\Delta}$ outside PAW (Å)", ylabel=r"$P_\Delta$ outside PAW", yscale="log",
              title="(c) outside the spheres")
    ax[0].legend(frameon=False, markerscale=3, fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig11_descriptor_maps")


# ------------------------------------------------------------------ 12 robustness
def fig_robustness():
    sh = pd.read_csv(DATA / "robust_shells.csv")
    zs = pd.read_csv(DATA / "robust_zval_summary.csv")
    gr = pd.read_csv(DATA / "robust_grid.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    tags = ["abs_0.6_1.3", "abs_1.0_1.8", "abs_0.8_2.0", "scaled_0.6_1.3"]
    for t, c in zip(tags, ("C0", "C1", "C2", "C3")):
        ax[0].scatter(sh["f_bond_def@abs_0.8_1.5"], sh[f"f_bond_def@{t}"], s=9, color=c, alpha=0.7,
                      label=t.replace("abs_", "(").replace("scaled_", "scaled (").replace("_", ", ") + ")")
    ax[0].plot([0, 1], [0, 1], "k-", lw=0.6)
    ax[0].set(xlabel=r"$f_{\mathrm{bond}}^{+}$, default shells (0.8, 1.5) Å", ylabel="other shells",
              title=f"(a) shell choice ({len(sh)} structures)")
    ax[0].legend(frameon=False, fontsize=6.5)
    x = np.arange(len(DEF))
    ax[1].bar(x - 0.2, zs["median_rel"], 0.4, label="median", color="C0")
    ax[1].bar(x + 0.2, zs["p90_rel"], 0.4, label="90th percentile", color="C1")
    ax[1].set_xticks(x, [SHORT[n] for n in zs["descriptor"]])
    ax[1].set(yscale="log", ylabel="relative change", title="(b) rule-based ZVAL instead of the PAW table")
    ax[1].legend(frameon=False)
    g_ = gr.groupby("descriptor")["rel_change_x0.8"]
    names = [n for n in DEF + [f"{n}_out" for n in DEF] if n in g_.groups]
    med = [g_.get_group(n).abs().median() for n in names]
    p90 = [g_.get_group(n).abs().quantile(0.9) for n in names]
    xx = np.arange(len(names))
    ax[2].bar(xx - 0.2, med, 0.4, color="C0", label="median")
    ax[2].bar(xx + 0.2, p90, 0.4, color="C1", label="90th percentile")
    ax[2].set_xticks(xx, names, rotation=90, fontsize=6.5)
    ax[2].set(yscale="log", ylabel="relative change", title="(c) grid: 80% of the points per axis")
    ax[2].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig12_robustness")


# ------------------------------------------------------------------ 13 ML
def fig_ml():
    files = [("scratch", DATA / "ml_scores_scratch.csv"), ("fine-tuned", DATA / "ml_scores_finetune.csv")]
    files = [(k, pd.read_csv(p)) for k, p in files if p.exists()]
    names = DEF + [f"{n}_out" for n in DEF]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    x = np.arange(len(names))
    wdt = 0.8 / len(files)
    for k, (tag, s) in enumerate(files):
        s = s.set_index("descriptor").reindex(names)
        ax[0].bar(x + (k - (len(files) - 1) / 2) * wdt, s["median_rel"], wdt, label=f"ChargE3Net {tag}")
        ax[1].bar(x + (k - (len(files) - 1) / 2) * wdt, 1 - s["spearman"], wdt, label=f"ChargE3Net {tag}")
    for a in ax:
        a.set_xticks(x, names, rotation=90, fontsize=7)
        a.set_yscale("log")
        a.legend(frameon=False)
    ax[0].set(ylabel="median |relative error|", title="(a) descriptor from predicted vs DFT density")
    ax[1].set(ylabel="1 − Spearman", title="(b) rank agreement across the 605 test structures")
    fig.tight_layout()
    save(fig, "fig13_ml_predictability")


# ------------------------------------------------------------------ 14 examples
def fig_examples():
    E = pd.read_csv(DATA / "examples.csv")
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.4))
    x = np.arange(len(E))
    for ax, n in zip(axes.flat, DEF):
        ax.bar(x - 0.2, E[n], 0.4, color="C0", label="whole cell")
        ax.bar(x + 0.2, E[n + "_out"], 0.4, color="C1", label="outside PAW")
        ax.set_xticks(x, [l.split(" (")[0] for l in E["label"]], rotation=60, fontsize=7)
        ax.set_title(label(n))
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    ax = axes.flat[-1]
    ax.bar(x, E["abs_share_inside_paw"], color="0.4")
    ax.set_xticks(x, [l.split(" (")[0] for l in E["label"]], rotation=60, fontsize=7)
    ax.set(title=r"share of $\int|\Delta\rho|$ inside the PAW spheres", ylim=(0, 1))
    fig.tight_layout()
    save(fig, "fig14_examples")


def main():
    FIG.mkdir(exist_ok=True)
    d = pd.read_csv(DATA / "deformation_dataset.csv")
    steps = [fig_breathing, fig_bond_charge, fig_slices, fig_radial_examples, fig_paw,
             lambda: fig_distributions(d), lambda: fig_by_class(d), lambda: fig_ionicity(d),
             lambda: fig_correlations(d), lambda: fig_maps(d), fig_robustness, fig_ml, fig_examples]
    for f in steps:
        try:
            f()
        except FileNotFoundError as exc:
            print("skipped:", exc)


if __name__ == "__main__":
    main()
