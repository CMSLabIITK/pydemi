"""Descriptor reference pages of the website from docs/descriptors.tex (one source for both).

Called by conf.py on every build. Writes docs/descriptors/*.md (not committed):

- theory.md        what a CHGCAR contains, notation, regions, derivatives (Sec. 1)
- bonding.md, structural.md, heterogeneity.md: one entry per descriptor (Secs. 2-4), each
  followed by its interactive figure from docs/_generated/figures (made by make_figures.py)
- index.md         all 54 at a glance, with links, and what is not included
- references.md    the bibliography of the reference

The LaTeX macros of the reference (\\desc, \\Def, ...) are rewritten to plain LaTeX, pandoc turns
that into Markdown, and the result is split into pages and given MyST labels and figures.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pypandoc

DOCS = Path(__file__).resolve().parents[1]
TEX = DOCS / "descriptors.tex"
OUT = DOCS / "descriptors"
GEN = DOCS / "_generated"

PAGES = {  # section title prefix -> (file, page title)
    "What a CHGCAR contains": ("theory", "Theory: the density and the common machinery"),
    "Bonding descriptors": ("bonding", "Bonding descriptors (18)"),
    "Structural descriptors": ("structural", "Structural descriptors (18)"),
    "Heterogeneity descriptors": ("heterogeneity", "Heterogeneity descriptors (18)"),
}
EXAMPLE = {"bonding": ("GaAs_216", "GaAs (zinc blende)"),
           "structural": ("GaAs_216", "GaAs (zinc blende)"),
           "heterogeneity": ("SbO2_15", "SbO₂ (Sb₂O₄, two inequivalent Sb sites: Sb³⁺ and Sb⁵⁺)")}


def _arg(s: str, i: int) -> tuple[str, int]:
    """The brace group starting at s[i] == '{' and the index after it."""
    assert s[i] == "{", s[i:i + 30]
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1:j], j + 1
    raise ValueError("unbalanced braces")


def _desc(s: str) -> str:
    """\\desc{label}{names}{units} -> a marked heading pandoc keeps."""
    out, i = [], 0
    while (j := s.find("\\desc{", i)) >= 0:
        out.append(s[i:j])
        label, k = _arg(s, j + 5)
        names, k = _arg(s, k)
        units, k = _arg(s, k)
        units = units.replace("\\quad", " ;; ")
        out.append(f"\n\n\\subsubsection{{DESC@{label}@{names}}}\n\nUNITS@ {units}\n\n")
        i = k
    return "".join(out) + s[i:]


def preprocess(tex: str) -> str:
    body = tex[tex.index("\\section{What a CHGCAR"):tex.index("\\begin{thebibliography}")]
    body = re.sub(r"(?<!\\)%.*", "", body)
    body = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "TIKZFIGURE", body, flags=re.S)
    body = re.sub(r"\{\\small Core and outer spheres.*?nucleus\.\}", "", body, flags=re.S)
    body = re.sub(r"\\(begin|end)\{minipage\}(\[t\]\{[^}]*\})?", "", body)
    body = body.replace("\\vspace{0pt}", "").replace("\\hfill", "").replace("\\centering", "")
    body = re.sub(r"\\begin\{itemize\}\[[^\]]*\]", r"\\begin{itemize}", body)
    body = _desc(body)
    for mac, lab in [("Def", "Definition"), ("Frm", "Formula"), ("Phy", "Physics"),
                     ("Lim", "Limits"), ("Cav", "Caveat")]:
        body = re.sub(rf"\\{mac}(?![a-zA-Z])\s*", rf"\n\nLABEL@{lab}@ ", body)
    body = re.sub(r"\\dn\{([^}]*)\}", lambda m: "\\texttt{" + m.group(1).replace("_", "\\_") + "}", body)
    body = body.replace("\\rhot", "\\tilde\\rho").replace("\\sumk", "\\sum_{k}")
    body = body.replace("\\medskip", "").replace("\\clearpage", "")
    body = re.sub(r"\\label\{[^}]*\}", "", body)
    body = re.sub(r"\(p\.~\\pageref\{[^}]*\}\)", "", body)
    body = re.sub(r"\(Sec\.~\\ref\{sec:dropped\}\)", "(see NOTINCLUDED)", body)
    # the at-a-glance table is rebuilt from the catalogue
    body = body[:body.index("\\section{All 54 descriptors at a glance}")] + \
        body[body.index("\\subsection*{Not included, and why}"):]
    return body


def bibliography(tex: str) -> list[tuple[str, str]]:
    bib = tex[tex.index("\\begin{thebibliography}"):tex.index("\\end{thebibliography}")]
    items = re.findall(r"\\bibitem\{([^}]*)\}(.*?)(?=\\bibitem|\Z)", bib, flags=re.S)
    out = []
    for key, text in items:
        md = pypandoc.convert_text(text.strip(), "commonmark_x", format="latex",
                                   extra_args=["--wrap=none"]).strip()
        out.append((key, md))
    return out


def to_markdown(body: str) -> str:
    md = pypandoc.convert_text(body, "markdown-raw_html-native_divs-native_spans-bracketed_spans"
                               "-fenced_divs-header_attributes-link_attributes-raw_attribute"
                               "-smart",
                               format="latex", extra_args=["--wrap=none"])
    return md


def cites(md: str, order: dict[str, int], here: str) -> str:
    def rep(m: re.Match) -> str:
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        return "".join(f"{{ref}}`[{order[k]}] <ref-{k}>`" for k in keys)
    return re.sub(r"\[(@[^\]]+)\]", rep, md)


def figure_block(names: list[str], page: str) -> str:
    fig_dir = GEN / "figures"
    names = [n for n in names if (fig_dir / f"{n}.html").exists()]
    if not names:
        return ""
    struct, label = EXAMPLE[page]

    def one(n: str) -> str:
        cap = (fig_dir / f"{n}.txt").read_text().strip()
        return (f"```{{raw}} html\n:file: ../_generated/figures/{n}.html\n```\n\n"
                f"```{{container}} figcaption\n{cap}\n```\n\n")

    head = (f"``````{{admonition}} Example: {label}\n:class: example-figure\n\n"
            f"The quantity behind the descriptor for `{struct}`, as the *Plots* view of the "
            f"graphical interface draws it (drag to zoom, hover for values).\n\n")
    if len(names) == 1:
        return head + one(names[0]) + "``````\n"
    tabs = "".join(f"````{{tab-item}} {n}\n{one(n)}````\n" for n in names)
    return head + f"`````{{tab-set}}\n{tabs}`````\n" + "``````\n"


def descriptor_names_of(names: str) -> list[str]:
    names = names.replace("\\_", "_").replace("\\texttt{", "").replace("}", "")
    return [n.strip() for n in names.split(",")]


def entries(md: str, page: str) -> str:
    """Headings of the descriptor entries: label, name, units line, figure after the entry."""
    parts = re.split(r"^(#{3} DESC@.*)$", md, flags=re.M)
    out = [parts[0]]
    for head, rest in zip(parts[1::2], parts[2::2]):
        _, label, names = head.split("@", 2)
        label = label.replace("\\_", "_")
        nl = descriptor_names_of(names)
        title = ", ".join(f"`{n}`" for n in nl)
        um = re.match(r"\s*UNITS@ (.*?)\n", rest)
        units = um.group(1).strip() if um else ""
        rest = rest[um.end():] if um else rest
        u = [x.strip() for x in units.split(";;")]
        badge = f"**Units:** {u[0]}" + (f" · **Range:** {u[1]}" if len(u) > 1 and u[1] else "")
        targets = "".join(f"({t})=\n" for t in dict.fromkeys(_lab(n) for n in [label] + nl))
        # the next lower-level heading ends the entry
        m = re.search(r"^#{1,3} ", rest, flags=re.M)
        entry, tail = (rest[:m.start()], rest[m.start():]) if m else (rest, "")
        fig = figure_block(nl if label != "rho_perc" else nl, page)
        out.append(f"{targets}### {title}\n\n{badge}\n{entry.rstrip()}\n\n{fig}\n{tail}")
    return "".join(out)


def _lab(n: str) -> str:
    return "d-" + n.replace("_", "-").lower()


def labels(md: str) -> str:
    md = re.sub(r"^LABEL@(\w+)@\s*", r"**\1.** ", md, flags=re.M)
    md = re.sub(r"\\?LABEL@(\w+)@\s*", r"\n\n**\1.** ", md)
    return md


UNIT = {"Angstrom": "Å", "Angstrom^2": "Å²", "(Angstrom)^2": "Å²", "e/Angstrom^3": "e/Å³",
        "1/Angstrom^3": "1/Å³", "e/bohr^3": "e/bohr³", "dimensionless": "–"}


def captures(tex: str, names: list[str]) -> dict[str, str]:
    """name -> 'what it captures', from the at-a-glance table of the reference."""
    tab = tex[tex.index("\\section{All 54 descriptors at a glance}"):tex.index("\\end{longtable}")]
    out: dict[str, str] = {}
    for line in tab.splitlines():
        m = re.match(r"\\dn\{([^}]*)\}.*?&[^&]*&\s*(.*?)\s*&\s*\\pageref", line.strip())
        if not m:
            continue
        key, text = m.group(1), m.group(2)
        text = pypandoc.convert_text(text, "commonmark_x", format="latex", extra_args=["--wrap=none"]).strip().replace("--", "–")
        if key.startswith("*"):
            hits = [n for n in names if n.endswith(key[1:])]
        else:
            stem = key.rstrip("abc").rstrip("_t1").rstrip("_") if key.endswith(("_a", "_t1")) else key
            hits = [n for n in names if n == key or (stem != key and n.startswith(stem + "_"))]
        for n in hits:
            out.setdefault(n, text)
    return out


def glance(tex: str) -> str:
    import pandas as pd
    if (GEN / "catalogue.csv").exists():
        cat = pd.read_csv(GEN / "catalogue.csv")
    else:
        import pydemi
        cat = pydemi.catalogue()
    what = captures(tex, list(cat.name))
    ex = json.loads((GEN / "examples.json").read_text()) if (GEN / "examples.json").exists() else {}
    rows = []
    for dom in ("bonding", "structural", "heterogeneity"):
        c = cat[cat.domain == dom]
        struct = EXAMPLE[dom][0]
        vals = ex.get(struct, {}).get("features", {})
        rows.append(f"\n## {dom.capitalize()} ({len(c)})\n\n"
                    f"The last column is the value for `{struct}` with the defaults.\n\n"
                    "| descriptor | what it captures | units | range | example |\n|---|---|---|---|---|")
        for r in c.itertuples():
            rng = f"[{_num(r.range_min)}, {_num(r.range_max)}]"
            v = vals.get(r.name)
            rows.append(f"| {{ref}}`{r.name} <{_lab(r.name)}>` | {what.get(r.name, '')} | "
                        f"{UNIT.get(r.units, r.units)} | {rng} | {'' if v is None else f'{v:.4g}'} |")
    return "\n".join(rows) + "\n"


def _num(x: float) -> str:
    return {float("inf"): "∞", float("-inf"): "−∞"}.get(float(x), f"{float(x):g}")


def build() -> None:
    tex = TEX.read_text()
    bib = bibliography(tex)
    order = {k: i + 1 for i, (k, _) in enumerate(bib)}
    md = to_markdown(preprocess(tex))
    md = cites(md, order, "")
    md = labels(md)
    md = md.replace("TIKZFIGURE", "\n```{figure} ../_static/regions.svg\n:width: 300px\n:align: center\n\n"
                    "Core and outer spheres of a large atom $i$ and a small atom $j$; $r_k$ and "
                    "$\\hat u_k$ are measured from the nearest nucleus.\n```\n")
    md = md.replace("(see NOTINCLUDED)", "(see {ref}`not-included`)")
    md = re.sub(r"\{#[^}]*\}", "", md)
    md = re.sub(r"\s*\$\$(.+?)\$\$[ \t]*", lambda m: "\n\n$$\n" + m.group(1).strip() + "\n$$\n\n", md, flags=re.S)
    OUT.mkdir(exist_ok=True)
    sections = re.split(r"^# (.*)$", md, flags=re.M)
    not_included = ""
    for title, text in zip(sections[1::2], sections[2::2]):
        key = next((v for k, v in PAGES.items() if title.startswith(k)), None)
        if "Not included" in text:
            i = text.index("## Not included")
            text, not_included = text[:i], text[i:].split("\n", 1)[1]
        if key is None:
            continue
        name, page_title = key
        text = entries(text, name) if name != "theory" else text
        note = ("```{note}\nThis page is generated from the descriptor reference "
                "([PDF](../_static/descriptors.pdf)), so the two always agree.\n```\n\n")
        (OUT / f"{name}.md").write_text(f"({name}-page)=\n# {page_title}\n\n{note}{text.strip()}\n")
    (OUT / "index.md").write_text(
        "(descriptors-index)=\n# All 54 descriptors\n\n"
        "Every descriptor is one number from the CHGCAR alone, intensive and unchanged by "
        "translations and rotations of the crystal. The names are those of "
        "`pydemi.featurize`; the formulas are those of `pydemi.catalogue()`. Click a name for "
        "its full entry, with its meaning, limits and an interactive figure.\n\n"
        "```{toctree}\n:maxdepth: 2\n:hidden:\n\ntheory\nbonding\nstructural\nheterogeneity\n```\n"
        "**Defaults:** sphere factors $a = 0.79$, $b = 1.14$; FFT derivatives; nearest-atom "
        "partition; `featurize` returns all 54.\n" + glance(tex) +
        "\n(not-included)=\n## Not included, and why\n\n" + not_included.strip() + "\n")
    refs = "\n\n".join(f"({'ref-' + k})=\n**[{order[k]}]** {t}" for k, t in bib)
    (OUT / "references.md").write_text("# References of the descriptor reference\n\n" + refs + "\n")


if __name__ == "__main__":
    build()
