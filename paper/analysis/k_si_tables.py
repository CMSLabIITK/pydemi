"""LaTeX tables for the Supporting Information, generated from repository data.

- si/catalogue_tables.tex: every non-compositional descriptor of the registry
  (docs/catalogue.csv, written by tools/generate_docs.py), with units, formula,
  degenerate-case values and stability tag, split into page-sized floats; the 132
  adopted Magpie descriptors summarized in one table.
- si/dft_settings_table.tex: the DFT settings parsed from the OUTCARs
  (out/dft_settings.csv, j_dft_settings.py).

Usage:  python k_si_tables.py
"""
from collections import Counter
from pathlib import Path

import pandas as pd

from common import OUT, REPO

SI = REPO / "paper" / "si"
ROWS = 34


def esc(s) -> str:
    s = "" if pd.isna(s) else str(s)
    out = []
    for ch in s:
        out.append({"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "^": r"\^{}",
                    "~": r"\~{}", "{": r"\{", "}": r"\}", "$": r"\$"}.get(ch, ch))
    return "".join(out)


def units(u) -> str:
    u = "" if pd.isna(u) else str(u)
    if u == "dimensionless":
        return "--"
    for a, b in (("1/Angstrom^3", r"\AA$^{-3}$"), ("(Angstrom)^2", r"\AA$^2$"), ("1/bohr^2", r"bohr$^{-2}$"),
                 ("e/bohr^3", r"e\,bohr$^{-3}$"),("e/Angstrom^3", r"e\,\AA$^{-3}$"), ("Angstrom^2", r"\AA$^2$"),
                 ("Angstrom", r"\AA"), ("hartree/bohr^3", r"Ha\,bohr$^{-3}$"), ("hartree/electron", "Ha/e"),
                 ("(mu_B)^2", r"$\mu_B^2$"), ("mu_B/atom", r"$\mu_B$/atom"), ("mu_B", r"$\mu_B$"),
                 ("(eV)^2", r"eV$^2$")):
        if u == a:
            return b
    return esc(u)


def formula(row) -> str:
    f = str(row["formula"])
    if "=" in f:
        f = f.split("=", 1)[1].strip()
    return esc(f)


def catalogue():
    c = pd.read_csv(REPO / "docs" / "catalogue.csv")
    core = c[c["domain"] != "compositional"].copy()
    core["ext"] = core["extension"].fillna("")
    order = {"bonding": 0, "structural": 1, "magnetic": 2, "heterogeneity": 3}
    core = core.sort_values(by=["ext", "domain"], key=lambda s: s.map(order) if s.name == "domain" else s,
                            kind="stable")
    blocks, out = [core.iloc[i:i + ROWS] for i in range(0, len(core), ROWS)], []
    for k, b in enumerate(blocks):
        out.append(r"\begin{table}[p]")
        out.append(r"\centering")
        cap = (r"Descriptor catalogue (part %d of %d): every non-compositional descriptor of the registry, "
               r"generated from \texttt{docs/catalogue.csv}. Domain: b bonding, s structural, m magnetic, "
               r"h heterogeneity; extension: paw or robust (off by default). Sentinels are the documented "
               r"values of degenerate cases; stability: r robust, f fragile." % (k + 1, len(blocks)))
        out.append(r"\caption{%s}" % cap)
        if k == 0:
            out.append(r"\label{tab:si-catalogue}")
        out.append(r"\scriptsize")
        out.append(r"\begin{tabular}{p{3.3cm}cp{1.4cm}p{6.6cm}p{2.3cm}c}")
        out.append(r"\toprule")
        out.append(r"Name & Dom. & Units & Formula & Sentinels & Stab. \\")
        out.append(r"\midrule")
        for _, r in b.iterrows():
            dom = r["domain"][0] + (f" ({r['ext']})" if r["ext"] else "")
            sent = esc(str(r["sentinel_cases"]).replace(";", "; ")) if pd.notna(r["sentinel_cases"]) else "--"
            out.append(r"\texttt{%s} & %s & %s & \texttt{%s} & %s & %s \\" % (
                esc(r["name"]), dom, units(r["units"]), formula(r), sent,
                str(r["stability"])[0]))
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        out.append(r"\end{table}")
        out.append("")
    comp = c[c["domain"] == "compositional"]
    stats = Counter(n.split("_")[1] for n in comp["name"])
    props = sorted({n.split("_", 2)[2] for n in comp["name"]})
    out += [r"\begin{table}[htbp]", r"\centering",
            r"\caption{The %d adopted compositional descriptors: matminer's Magpie preset, %d elemental "
            r"properties times %d statistics (\texttt{magpie\_<statistic>\_<property>}), tagged "
            r"\texttt{adopted=True} in the registry.}" % (len(comp), len(props), len(stats)),
            r"\label{tab:si-magpie}", r"\small", r"\begin{tabular}{p{3cm}p{11cm}}", r"\toprule",
            r"Statistics & %s \\" % esc(", ".join(sorted(stats))),
            r"Properties & %s \\" % esc(", ".join(props)), r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (SI / "catalogue_tables.tex").write_text("\n".join(out) + "\n")
    print(len(core), "descriptors in", len(blocks), "tables;", len(comp), "Magpie")


def settings():
    s = pd.read_csv(OUT / "dft_settings.csv")
    n = len(s)

    def share(col, val):
        return int((s[col].astype(str) == str(val)).sum())
    ks = s["kspacing_max"]
    rows = [
        ("VASP version", f"5.3.3 ({share('vasp', '5.3.3')}), 5.4.4 ({n - share('vasp', '5.3.3')})"),
        ("Exchange-correlation", "PBE (\\texttt{LEXCH = PE}) in all runs; no meta-GGA, no $+U$, no van der Waals correction"),
        ("PAW datasets", f"{s['potcars'].str.split('PAW').explode().str.strip().replace('', None).dropna().nunique()} "
                         "PAW\\_PBE datasets, one per element throughout"),
        ("Plane-wave cutoff", "\\texttt{ENCUT} = 500 eV, \\texttt{PREC = Accurate} (all runs)"),
        ("Spin", "collinear spin-polarized (\\texttt{ISPIN = 2}) in all runs; initial moments not recorded in the OUTCAR"),
        ("Electronic convergence", f"\\texttt{{EDIFF}} = $10^{{-6}}$ eV, \\texttt{{NELM}} = 200; "
                                   f"{int((~s['converged']).sum())} runs ({100 * (~s['converged']).mean():.1f}\\%) "
                                   "stopped before reaching \\texttt{EDIFF}"),
        ("Ionic steps", "none (\\texttt{NSW = 0}, \\texttt{IBRION = -1}): single-point calculations"),
        ("Brillouin-zone integration", f"tetrahedron method with Bl\\\"ochl corrections (\\texttt{{ISMEAR = -5}}) in "
                                       f"{share('ISMEAR', -5)} runs; Gaussian, 0.03 eV in {share('ISMEAR', 0)}"),
        ("$k$-points", f"automatic mesh; largest spacing $|\\mathbf b_i|/N_i$ median {ks.median():.3f}\\,\\AA$^{{-1}}$ "
                       f"(5th--95th percentile {ks.quantile(0.05):.3f}--{ks.quantile(0.95):.3f}, $2\\pi$ included)"),
        ("Projection", f"reciprocal space (\\texttt{{LREAL = F}}) in {share('LREAL', 'F')} runs, "
                       f"\\texttt{{Auto}} in {share('LREAL', 'Auto')}"),
        ("Output", "CHGCAR (\\texttt{LCHARG = T}); ELF written (\\texttt{LELF = T}), but the ELFCAR is "
                   "distributed for only 86 runs; no LOCPOT (\\texttt{LVTOT = F}), no AECCAR (\\texttt{LAECHG = F})"),
    ]
    out = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{DFT settings of the dataset, parsed from the %d runs that include their OUTCAR "
           r"(\texttt{paper/analysis/j\_dft\_settings.py}). The other %d runs have only the CHGCAR.}" % (n, 6059 - n),
           r"\label{tab:si-dft}", r"\small", r"\begin{tabular}{p{4cm}p{10.5cm}}", r"\toprule"]
    out += [f"{a} & {b} \\\\" for a, b in rows]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (SI / "dft_settings_table.tex").write_text("\n".join(out) + "\n")
    print("settings table written")


def analytic_checks():
    a = pd.read_csv(OUT / "analytic_checks.csv")
    g = a.groupby(["family", "descriptor", "model"], sort=False).agg(n=("abs_err", "size"), med=("abs_err", "median"),
                                                                   mx=("abs_err", "max")).reset_index()
    fmt = lambda v: "0" if v == 0 else f"{v:.1e}".replace("e-0", "e-").replace("e-", r"\times10^{-") + "}"
    out = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{The %d analytic checks of the descriptor families (main text, Fig.~4): model density, "
           r"number of cases, median and maximum absolute error against the exact value "
           r"(\texttt{paper/analysis/m\_validation\_figures.py}).}" % len(a),
           r"\label{tab:si-analytic}", r"\small", r"\begin{tabular}{llp{4.2cm}rrr}", r"\toprule",
           r"Family & Descriptor & Model & $n$ & Median & Max \\", r"\midrule"]
    for _, r in g.iterrows():
        out.append(r"%s & \texttt{%s} & %s & %d & $%s$ & $%s$ \\" % (r["family"], esc(r["descriptor"]), esc(r["model"]),
                                                                  r["n"], fmt(r["med"]), fmt(r["mx"])))
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (SI / "analytic_checks_table.tex").write_text("\n".join(out) + "\n")
    print("analytic checks table written")


if __name__ == "__main__":
    catalogue()
    settings()
    analytic_checks()
