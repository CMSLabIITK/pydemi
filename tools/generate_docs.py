"""
Regenerate the descriptor tables in README.md and docs/catalogue.csv from the registry.

The README holds hand-written text plus tables between
``<!-- catalogue:<domain>:start -->`` and ``<!-- catalogue:<domain>:end -->``
markers; this script replaces the text between each pair.

Usage:  python tools/generate_docs.py
"""

import re
from pathlib import Path

import pydemi

ROOT = Path(__file__).resolve().parents[1]


def esc(s: str) -> str:
    return s.replace("|", "\\|")


def table(cat, domain: str, extension: str = "") -> str:
    rows = cat[(cat.domain == domain) & (cat.extension == extension)]
    out = ["| Name | Units | Definition | Sentinel cases |", "|---|---|---|---|"]
    for _, r in rows.iterrows():
        out.append(f"| `{r['name']}` | {esc(r['units'])} | {esc(r['formula'])} | "
                   f"{esc(r['sentinel_cases']) or '-'} |")
    return "\n".join(out)


def main() -> None:
    cat = pydemi.catalogue()
    (ROOT / "docs").mkdir(exist_ok=True)
    cat.to_csv(ROOT / "docs" / "catalogue.csv", index=False)
    readme = ROOT / "README.md"
    text = readme.read_text()
    blocks = {f"{d}": table(cat, d) for d in ("bonding", "structural", "magnetic", "heterogeneity")}
    blocks["paw"] = "\n\n".join(table(cat, d, "paw") for d in ("bonding", "structural"))
    for key, body in blocks.items():
        text = re.sub(rf"(<!-- catalogue:{key}:start -->).*?(<!-- catalogue:{key}:end -->)",
                      lambda m: m.group(1) + "\n" + body + "\n" + m.group(2), text, flags=re.S)
    counts = cat[cat.extension == ""].groupby("domain").size().to_dict()
    total = int(sum(counts.values()))
    n_paw = int((cat.extension == "paw").sum())
    text = re.sub(r"<!-- counts -->.*?<!-- /counts -->",
                  f"<!-- counts -->{total} descriptors by default ("
                  + ", ".join(f"{counts[d]} {d}" for d in pydemi.descriptors.registry.DOMAINS)
                  + f"), plus {n_paw} in the off-by-default PAW extension<!-- /counts -->", text)
    readme.write_text(text)
    print(f"README tables and docs/catalogue.csv regenerated ({total} + {n_paw} descriptors)")


if __name__ == "__main__":
    main()
