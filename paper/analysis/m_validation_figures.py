"""Main-text figures on validation (Fig. 5) and on what the descriptors add (Fig. 7).

Fig. 5: every analytic model of the per-family reports (docs/*/data/analytic_*.csv),
pydemi against the exact value: a parity plot and the relative error per check.
Fig. 7: (a-d) class trends of four robust, chemically interpretable descriptors;
(e) for each non-compositional descriptor, the largest |Spearman| with any of the 132
Magpie composition features over the 6,059 structures.

Usage:  python m_validation_figures.py   -> ../figures/fig_validation.pdf, fig_classes.pdf,
                                            out/analytic_checks.csv, out/magpie_overlap.csv
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUT, REPO, results  # noqa: E402

FIG = REPO / "paper" / "figures"
D = REPO / "docs"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "savefig.bbox": "tight"})
CLASSES = ["elemental", "intermetallic", "boride/carbide", "hydride", "pnictide", "chalcogenide", "oxide", "halide"]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=250)
    plt.close(fig)
    print("wrote", name)


def checks():
    """(family, descriptor, model, pydemi, exact) rows from the reports' analytic models."""
    rows = []

    def add(fam, model, df, pairs):
        for got, ref, lab in pairs:
            for a, b in zip(df[got], df[ref]):
                rows.append((fam, lab, model, float(a), float(b)))
    a = pd.read_csv(D / "anisotropy_descriptors/data/analytic_A.csv")
    add("anisotropy", "anisotropic Gaussian", a, [("T_eigenvalues_t1", "t1_exact", "t1"), ("T_eigenvalues_t3", "t3_exact", "t3"),
                                               ("charge_FA", "charge_FA_exact", "charge_FA")])
    d = pd.read_csv(D / "deformation_descriptors/data/analytic_A.csv")
    add("deformation", "breathing atom", d, [(n, f"{n}_exact", n) for n in
        ("m1_def", "m2_def", "sigma_r2_def", "f_bond_def", "f_int_def", "f_bond_dep", "def_polarity")])
    m = pd.read_csv(D / "magnetic_descriptors/data/analytic_A.csv")
    add("magnetic", "Gaussian spin density", m, [(n, f"{n}_exact", n) for n in
        ("m1_spin", "sigma_r2_spin", "f_bond_spin", "spin_charge_correlation")])
    mb = pd.read_csv(D / "magnetic_descriptors/data/analytic_B.csv")
    add("magnetic", "two sublattices", mb, [("spin_frustration", "spin_frustration_exact", "spin_frustration"),
                                           ("mu_site_std", "mu_site_std_exact", "mu_site_std")])
    p = pd.read_csv(D / "percolation_descriptors/data/analytic_A.csv")
    add("structural", "simple cubic lattice", p, [("rho_perc_a", "rho_perc_exact", "rho_perc_a"),
                                                 ("rho_min", "rho_min_exact", "rho_min"),
                                                 ("rho_min_ratio", "rho_min_ratio_exact", "rho_min_ratio")])
    pb = pd.read_csv(D / "percolation_descriptors/data/analytic_B.csv")
    add("structural", "tetragonal lattice", pb, [("perc_anisotropy", "perc_anisotropy_exact", "perc_anisotropy")])
    i = pd.read_csv(D / "ionicity_descriptors/data/analytic_A.csv")
    add("bonding", "rock-salt charge transfer", i, [("V_spread", "V_spread_exact", "V_spread"),
                                                   ("V_A", "V_A_exact", "V at a nucleus")])
    ib = pd.read_csv(D / "ionicity_descriptors/data/analytic_B.csv")
    add("bonding", "tetragonal lattice", ib, [("rho_mid_mean", "rho_mid_mean_exact", "rho_mid_mean"),
                                             ("rho_mid_std", "rho_mid_std_exact", "rho_mid_std")])
    ic = pd.read_csv(D / "ionicity_descriptors/data/analytic_C.csv").dropna(subset=["lap_concentration_valence_exact"])
    add("bonding", "Gaussian atom", ic, [("lap_concentration_valence", "lap_concentration_valence_exact",
                                          "lap_concentration_valence")])
    df = pd.DataFrame(rows, columns=["family", "descriptor", "model", "pydemi", "exact"])
    df["abs_err"] = (df["pydemi"] - df["exact"]).abs()
    df["rel_err"] = df["abs_err"] / df["exact"].abs().where(df["exact"].abs() > 1e-8)
    df.to_csv(OUT / "analytic_checks.csv", index=False)
    return df


def fig_validation():
    df = checks()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1.5]})
    fams = list(dict.fromkeys(df["family"]))
    for k, f in enumerate(fams):
        s = df[(df["family"] == f) & (df["exact"].abs() > 1e-8)]
        ax[0].scatter(s["exact"].abs(), s["pydemi"].abs(), s=8, color=f"C{k}", label=f)
    lim = [1e-5, 30]
    ax[0].plot(lim, lim, "k-", lw=0.6)
    ax[0].set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="|exact value|", ylabel="|pydemi|",
              title=f"(a) {len(df)} analytic checks")
    ax[0].legend(frameon=False, loc="upper left")
    g = df.groupby(["descriptor", "family"], sort=False)["abs_err"].agg(["median", "max"]).reset_index()
    y = np.arange(len(g))
    for k, f in enumerate(fams):
        sel = g["family"] == f
        ax[1].barh(y[sel], np.maximum(g.loc[sel, "max"], 1e-17), color=f"C{k}", alpha=0.45)
        ax[1].plot(np.maximum(g.loc[sel, "median"], 1e-17), y[sel], "o", color=f"C{k}", ms=3)
    ax[1].set_yticks(y, g["descriptor"], fontsize=5.8)
    ax[1].invert_yaxis()
    ax[1].set(xscale="log", xlim=(1e-17, 1), xlabel="absolute error (bar: max, dot: median)",
              title="(b) error per descriptor")
    fig.tight_layout()
    save(fig, "fig_validation")


def fig_classes():
    df = results().reset_index()
    cls = pd.read_csv(D / "anisotropy_descriptors/data/anisotropy_dataset.csv")[["id", "chem_class"]]
    df = df.merge(cls, on="id")
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
    panels = [("f_bond_def_out", "(a) bond-shell share of the\ndeformation charge (outside PAW)", False),
              ("rho_min_int_ratio", "(b) interstitial density floor\n" r"$\rho_{\min,\mathrm{int}}/\langle\rho\rangle$", False),
              ("charge_FA", "(c) fractional anisotropy of\nthe gradient tensor", True),
              ("rho_perc_a", "(d) percolation level\nalong $a_1$ (e/Å$^3$)", True)]
    for k, (n, title, log) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        data = [df.loc[df["chem_class"] == c, n].dropna() for c in CLASSES]
        if log:
            data = [np.clip(x, 1e-8, None) for x in data]
        bp = ax.boxplot(data, showfliers=False, widths=0.6, patch_artist=True)
        for p, c in zip(bp["boxes"], plt.cm.Dark2(np.arange(8))):
            p.set_facecolor(c)
            p.set_alpha(0.75)
        ax.set_xticks(range(1, 9), [c.replace("boride/carbide", "B/C") for c in CLASSES], rotation=60, ha="right",
                      fontsize=6)
        ax.set_title(title, fontsize=7)
        if log:
            ax.set_yscale("log")
    mag = [c for c in df.columns if c.startswith("magpie_")]
    cat = pd.read_csv(REPO / "docs" / "catalogue.csv")
    own = cat[(cat["domain"] != "compositional") & cat["extension"].isna()]["name"].tolist()
    R = df[own + mag].rank()
    rows = []
    for n in own:
        if R[n].nunique() < 3 or n == "lap_concentration":          # constant 1/2 (round-off only)
            continue
        c = R[mag].corrwith(R[n]).abs()
        rows.append({"descriptor": n, "domain": cat.set_index("name").loc[n, "domain"], "max_abs_spearman": c.max(),
                     "best_magpie": c.idxmax()})
    ov = pd.DataFrame(rows).sort_values("max_abs_spearman")
    ov.to_csv(OUT / "magpie_overlap.csv", index=False)
    ax = fig.add_subplot(gs[1, :])
    colors = {"bonding": "C0", "structural": "C2", "magnetic": "C3", "heterogeneity": "C4"}
    ax.bar(range(len(ov)), ov["max_abs_spearman"], color=[colors[x] for x in ov["domain"]])
    ax.axhline(0.8, color="k", lw=0.6, ls="--")
    ax.set(xlim=(-1, len(ov)), ylim=(0, 1), ylabel="max |Spearman| with\nany Magpie feature",
           title=f"(e) overlap with composition: {int((ov['max_abs_spearman'] < 0.8).sum())} of {len(ov)} descriptors "
                 "below 0.8")
    ax.set_xticks([])
    for d_, c in colors.items():
        ax.bar([0], [0], color=c, label=d_)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_classes")
    print(ov.tail(8).to_string(), "\n", ov["max_abs_spearman"].describe())


if __name__ == "__main__":
    fig_validation()
    fig_classes()
