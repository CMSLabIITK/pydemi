"""Main-text figures that need no ML results: architecture (Fig. 1), PAW pitfalls
(Fig. 3), stability map (Fig. 4) and dataset overview (Fig. 6).

Inputs: out/convergence.csv, out/derivatives_summary.csv, out/partitions_summary.csv
(analyses a, b, e), out/ml_scores_scratch.csv (the ML evaluation's DFT-vs-ChargE3Net
scores), docs/catalogue.csv, the rerun table, and the region-share / bottleneck tables
of the per-family reports in docs/.

Usage:  python l_main_figures.py   -> ../figures/fig_architecture.pdf, fig_paw.pdf,
                                      fig_stability.pdf, fig_dataset.pdf (+ .png)
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from common import OUT, REPO, results  # noqa: E402

FIG = REPO / "paper" / "figures"
DOCS = REPO / "docs"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "savefig.bbox": "tight"})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=250)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ Fig. 1
def box(ax, x, y, w, h, title, lines, color, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=color, ec="k", lw=0.8,
                                ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h - 0.025, title, ha="center", va="top", fontsize=8, weight="bold")
    ax.text(x + w / 2, y + h - 0.065, "\n".join(lines), ha="center", va="top", fontsize=6.6, linespacing=1.35)


def arrow(ax, a, b, dashed=False):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, lw=0.9, color="k",
                                 ls="--" if dashed else "-"))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set(xlim=(-0.02, 1.03), ylim=(-0.03, 1.0))
    ax.axis("off")
    box(ax, 0.00, 0.63, 0.23, 0.33, "Input 1: DFT", ["VASP CHGCAR (1/2/4 blocks)", "AECCAR0 + AECCAR2", "ELFCAR, LOCPOT",
        "cube, XSF", "ZVAL, R_PAW from POTCAR/", "OUTCAR or a PAW table"], "#dbe9f6")
    box(ax, 0.00, 0.20, 0.23, 0.33, "Input 2: ML", ["crystal structure (CIF)", "density-prediction model", "(ChargE3Net, trained on", "the 6,059 densities)",
        "predicted rho on a grid"], "#f6e8db", dashed=True)
    box(ax, 0.30, 0.40, 0.19, 0.40, "VolumetricData", ["Structure, Lattice", "(B = A^-T, G = B B^T)", "Grid: rho, m, ELF, V",
        "density_source,", "zval, paw_radii", "per-object cache", "shared by option views"], "#e8f3e0")
    box(ax, 0.56, 0.62, 0.20, 0.34, "Derived fields", ["derivatives: FFT or FD 2-8", "(exact metric)", "ELF_D, energy densities",
        "Hartree / ESP", "promolecule, delta rho", "|m|"], "#fff2cc")
    box(ax, 0.56, 0.20, 0.20, 0.34, "Geometry pass", ["nearest nucleus: r, u, i", "(periodic images,", "geometric tie-break)",
        "shells core/bond/int", "partitions: nearest, power,", "Becke, Hirshfeld"], "#fff2cc")
    box(ax, 0.81, 0.40, 0.19, 0.40, "Operators", ["radial moments", "shell fractions", "anisotropy tensor", "site aggregation +",
        "variance decomposition", "critical-point census,", "percolation, basins"], "#f3e0ef")
    box(ax, 0.30, -0.02, 0.70, 0.15, "Registry and output", ["@register: domain, field, units, range, sentinels, references, "
        "extension, stability", "featurize / featurize_batch / CLI  ->  descriptor table with __flag and metadata columns"],
        "#eeeeee")
    arrow(ax, (0.23, 0.79), (0.30, 0.66))
    arrow(ax, (0.23, 0.37), (0.30, 0.54), dashed=True)
    arrow(ax, (0.49, 0.66), (0.56, 0.78))
    arrow(ax, (0.49, 0.54), (0.56, 0.38))
    arrow(ax, (0.76, 0.78), (0.81, 0.66))
    arrow(ax, (0.76, 0.38), (0.81, 0.54))
    arrow(ax, (0.905, 0.40), (0.905, 0.13))
    ax.text(0.115, 0.165, "in progress", ha="center", fontsize=6.5, style="italic")
    save(fig, "fig_architecture")


# ------------------------------------------------------------------ Fig. 3
def fig_paw():
    an = pd.read_csv(DOCS / "anisotropy_descriptors/data/region_shares.csv")
    de = pd.read_csv(DOCS / "deformation_descriptors/data/region_shares.csv")
    mg = pd.read_csv(DOCS / "magnetic_descriptors/data/region_shares.csv")
    bt = pd.read_csv(DOCS / "percolation_descriptors/data/bottlenecks.csv")
    pc = pd.read_csv(DOCS / "percolation_descriptors/data/percolation_dataset.csv")
    for t in (an, de, mg, bt):
        if "error" in t:
            t.drop(t.index[t["error"].notna()], inplace=True)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    data = [de["vol_inside_paw"], an["grad_share_inside"], de["abs_inside_paw"], mg["abs_inside_paw"]]
    labs = ["volume", r"$\sum|\nabla\rho|$", r"$\int|\Delta\rho|$", r"$\int|m|$"]
    ax[0].boxplot(data, showfliers=False, widths=0.55)
    ax[0].set_xticks(range(1, 5), labs)
    ax[0].set(ylabel="share inside the PAW spheres", ylim=(0, 1.02), title="(a) what lies inside the spheres")
    r = pd.concat([bt[f"r_over_Rpaw_{l}"] for l in "abc"])
    bins = np.linspace(0, 3, 46)
    ax[1].hist(r, bins=bins, color="C0", alpha=0.75, density=True, label="percolation bottlenecks")
    ax[1].hist(bt["rmin_over_Rpaw"], bins=bins, color="C3", alpha=0.6, density=True, label=r"$\rho_{\min}$")
    ax[1].axvline(1, color="k", lw=0.8)
    ax[1].set(xlabel=r"distance to the nearest nucleus / $R_{\mathrm{PAW}}$", ylabel="density of cases",
              title="(b) where bottlenecks and minima lie")
    ax[1].legend(frameon=False)
    ax[2].scatter(pc["rho_min_int_ratio"], pc["rho_min_ratio"], s=1.5, alpha=0.3, color="C2", rasterized=True)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].plot([-0.1, 1], [-0.1, 1], "k-", lw=0.5)
    ax[2].set(xlabel=r"$\rho_{\min}/\langle\rho\rangle$ outside the spheres", ylabel=r"$\rho_{\min}/\langle\rho\rangle$, whole cell",
              ylim=(-4, 1), title="(c) density floor, 6,059 structures")
    fig.tight_layout()
    save(fig, "fig_paw")


# ------------------------------------------------------------------ Fig. 4
def fig_stability():
    cat = pd.read_csv(REPO / "docs" / "catalogue.csv")
    core = cat[(cat["domain"] != "compositional") & cat["extension"].isna()]
    names = core["name"].tolist()
    conv = pd.read_csv(OUT / "convergence.csv")
    conv = conv[np.isfinite(conv["rel_change_x0.8"]) & ~((conv["full"] == 0) & (conv["x0.8"] == 0))]
    grid = conv.groupby("descriptor")["rel_change_x0.8"].apply(lambda s: s.abs().median())
    der = pd.read_csv(OUT / "derivatives_summary.csv")
    fd4 = der[der["scheme"] == "fd4"].set_index("descriptor")["median"]
    fd2 = der[der["scheme"] == "fd2"].set_index("descriptor")["median"]
    het = pd.read_csv(DOCS / "heterogeneity_descriptors/data/robust_derivatives_summary.csv")
    for tag, target in (("fd4", fd4), ("fd2", fd2)):
        for _, r in het[het["variant"] == tag].iterrows():
            target[r["descriptor"]] = r["median_rel"]
    part = pd.read_csv(OUT / "partitions_summary.csv").drop_duplicates()
    becke = part[part["partition"] == "becke"].set_index("descriptor")["median"]
    hirsh = part[part["partition"] == "hirshfeld"].set_index("descriptor")["median"]
    ml = pd.read_csv(OUT / "ml_scores_scratch.csv").set_index("descriptor")["median_rel"]
    deriv_based = set(fd4.index) | {n for n in names if n.startswith("zeta_")}
    part_based = set(part["descriptor"])
    cols = ["grid 80%", "FD4 vs FFT", "FD2 vs FFT", "Becke", "Hirshfeld", "ML (from scratch)"]
    M = np.full((len(names), len(cols)), np.nan)
    NA = np.zeros_like(M, dtype=bool)
    for i, n in enumerate(names):
        M[i, 0] = grid.get(n, np.nan)
        if n in deriv_based:
            M[i, 1], M[i, 2] = fd4.get(n, np.nan), fd2.get(n, np.nan)
        else:
            NA[i, 1] = NA[i, 2] = True
        if n in part_based:
            M[i, 3], M[i, 4] = becke.get(n, np.nan), hirsh.get(n, np.nan)
        else:
            NA[i, 3] = NA[i, 4] = True
        M[i, 5] = ml.get(n, np.nan)
    stab = core.set_index("name")["stability"].reindex(names)
    dom = core.set_index("name")["domain"].reindex(names)
    half = (len(names) + 1) // 2
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 8.6), gridspec_kw={"wspace": 0.95})
    norm = LogNorm(vmin=1e-5, vmax=1.0)
    for ax, sl in zip(axes, (slice(0, half), slice(half, None))):
        sub = np.clip(M[sl], 1e-5, 1.0)
        im = ax.imshow(sub, aspect="auto", cmap="viridis_r", norm=norm, interpolation="nearest")
        na = NA[sl]
        for (i, j), v in np.ndenumerate(na):
            if v:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="white", ec="0.85", lw=0.3))
                ax.text(j, i, "·", ha="center", va="center", fontsize=6, color="0.6")
        miss = np.isnan(M[sl]) & ~na
        for (i, j), v in np.ndenumerate(miss):
            if v:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="0.8"))
        lab = [f"{n}{'  (f)' if stab[n] == 'fragile' else ''}" for n in names[sl]]
        ax.set_yticks(range(len(lab)), lab, fontsize=5.3)
        for t, n in zip(ax.get_yticklabels(), names[sl]):
            t.set_color({"bonding": "C0", "structural": "C2", "magnetic": "C3", "heterogeneity": "C4"}[dom[n]])
        ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=6.5)
        ax.tick_params(length=0)
    handles = [plt.Line2D([], [], ls="", marker="s", color=c, label=d) for d, c in
               (("bonding", "C0"), ("structural", "C2"), ("magnetic", "C3"), ("heterogeneity", "C4"))]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02),
               title="label colour: domain;  (f) = tagged fragile")
    cb = fig.colorbar(im, ax=axes, shrink=0.35, pad=0.02, location="top")
    cb.set_label("median relative change (clipped to [1e-5, 1]); · = does not depend on this choice; grey = not tested")
    save(fig, "fig_stability")
    pd.DataFrame(M, index=names, columns=cols).to_csv(OUT / "stability_map.csv")


# ------------------------------------------------------------------ Fig. 6
def fig_dataset():
    from pymatgen.core import Composition, Element
    df = results().reset_index()
    cls = pd.read_csv(DOCS / "anisotropy_descriptors/data/anisotropy_dataset.csv")[["id", "chem_class", "formula"]]
    df = df.merge(cls, on="id")
    counts = {}
    for f in df["formula"]:
        for e in Composition(f).elements:
            counts[str(e)] = counts.get(str(e), 0) + 1
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
    ax0 = fig.add_subplot(gs[0, :])
    grid = np.full((10, 18), np.nan)
    labels = {}
    for sym, n in counts.items():
        el = Element(sym)
        row, col = el.row, el.group
        if 57 <= el.Z <= 71:
            row, col = 8, el.Z - 57 + 3
        elif 89 <= el.Z <= 103:
            row, col = 9, el.Z - 89 + 3
        grid[row - 1, col - 1] = n
        labels[(row - 1, col - 1)] = sym
    im = ax0.imshow(grid, cmap="YlOrRd", norm=LogNorm(vmin=1, vmax=np.nanmax(grid)))
    for (r, c), sym in labels.items():
        ax0.text(c, r, f"{sym}\n{int(grid[r, c])}", ha="center", va="center", fontsize=4.6)
    ax0.set(xticks=[], yticks=[], title=f"(a) structures containing each element ({len(counts)} elements, "
                                        f"{len(df):,} structures)")
    for s in ax0.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax0, shrink=0.7, pad=0.01, label="structures")
    order = ["elemental", "intermetallic", "boride/carbide", "hydride", "pnictide", "chalcogenide", "oxide", "halide"]
    ax1 = fig.add_subplot(gs[1, 0])
    vc = df["chem_class"].value_counts().reindex(order)
    ax1.barh(range(len(order)), vc.values, color=plt.cm.Dark2(np.arange(8)))
    ax1.set_yticks(range(len(order)), [o.replace("boride/carbide", "B/C") for o in order])
    ax1.invert_yaxis()
    ax1.set(xlabel="structures", title="(b) chemical classes")
    ax2 = fig.add_subplot(gs[1, 1])
    npts = df["grid_shape"].str.split("x").apply(lambda v: np.prod([int(x) for x in v]))
    ax2.scatter(df["n_atoms"], npts / 1e6, s=2, alpha=0.3, rasterized=True)
    ax2.set(xscale="log", yscale="log", xlabel="atoms per cell", ylabel=r"grid points ($10^6$)", title="(c) cell and grid sizes")
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.scatter(npts / 1e6, df["wall_time_s"], s=2, alpha=0.3, color="C1", rasterized=True)
    ax3.set(xscale="log", yscale="log", xlabel=r"grid points ($10^6$)", ylabel="wall time per structure (s)",
            title=f"(d) cost: median {df['wall_time_s'].median():.0f} s")
    fig.tight_layout()
    save(fig, "fig_dataset")


if __name__ == "__main__":
    fig_architecture()
    fig_paw()
    fig_stability()
    fig_dataset()
