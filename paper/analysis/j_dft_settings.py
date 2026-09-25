"""DFT settings of the dataset, read from the runs' own OUTCAR files (read-only).

The run folders hold only CHGCAR and, for 4,901 of 6,059 runs, OUTCAR (no INCAR,
KPOINTS or POTCAR). Every OUTCAR echoes the INCAR parameters VASP used; this script
parses them, with the POTCAR titles, the k-point mesh, the grids, the VASP version and
the electronic convergence, and summarizes them for the paper's methods section.

Usage:  python j_dft_settings.py
Output: out/dft_settings.csv (one row per run), out/dft_settings_summary.txt
"""
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import DATASET, OUT, WORKERS, all_ids

sys.path.insert(0, str(OUT.parents[2] / "docs" / "anisotropy_descriptors" / "scripts"))
from classes import header  # noqa: E402

NUM = r"([-+0-9.Ee]+)"
PARAMS = {  # name -> regex on the parameter block (first match)
    "PREC": r"PREC\s*=\s*(\w+)", "ENCUT": rf"ENCUT\s*=\s*{NUM}", "ENAUG": rf"ENAUG\s*=\s*{NUM}",
    "EDIFF": rf"EDIFF\s*=\s*{NUM}", "EDIFFG": rf"EDIFFG\s*=\s*{NUM}", "NSW": r"NSW\s*=\s*(\d+)",
    "IBRION": r"IBRION\s*=\s*(-?\d+)", "ISIF": r"ISIF\s*=\s*(\d+)", "ISPIN": r"ISPIN\s*=\s*(\d+)",
    "ISMEAR": r"ISMEAR\s*=\s*(-?\d+)", "SIGMA": rf"SIGMA\s*=\s*{NUM}", "LREAL": r"LREAL\s*=\s*(\w+)",
    "LASPH": r"LASPH\s*=\s*(\w)", "IALGO": r"IALGO\s*=\s*(\d+)", "NELM": r"NELM\s*=\s*(\d+)",
    "ISYM": r"ISYM\s*=\s*(-?\d+)", "LORBIT": r"LORBIT\s*=\s*(\d+)", "LELF": r"LELF\s*=\s*(\w)",
    "LVTOT": r"LVTOT\s*=\s*(\w)", "LCHARG": r"LCHARG\s*=\s*(\w)", "GGA": r"\n\s*GGA\s*=\s*(\S+)",
    "METAGGA": r"METAGGA\s*=\s*(\w)", "NBANDS": r"NBANDS\s*=\s*(\d+)", "NKPTS": r"NKPTS\s*=\s*(\d+)",
    "NGXF": r"NGXF\s*=\s*(\d+)\s*NGYF\s*=\s*(\d+)\s*NGZF\s*=\s*(\d+)",
}


def parse(run):
    try:
        with open(DATASET / run / "OUTCAR", errors="replace") as fh:
            head = fh.read(600_000)
        row = {"id": run}
        m = re.search(r"vasp\.(\S+)", head)
        row["vasp"] = m.group(1) if m else None
        row["potcars"] = " ".join(sorted(set(re.findall(r"POTCAR:\s+(PAW\S*\s+\S+)", head))))
        row["lexch"] = " ".join(sorted(set(re.findall(r"LEXCH\s*=\s*(\S+)", head))))
        for k, pat in PARAMS.items():
            m = re.search(pat, head)
            row[k] = ("x".join(m.groups()) if k == "NGXF" else m.group(1)) if m else None
        row["LDAU"] = bool(re.search(r"LDAUTYPE|LDAUU", head))
        row["IVDW"] = (re.search(r"IVDW\s*=\s*(\d+)", head) or [None, None])[1] if "IVDW" in head else "0"
        row["LAECHG"] = bool(re.search(r"LAECHG\s*=\s*T", head))
        row["noncollinear"] = bool(re.search(r"LNONCOLLINEAR\s*=\s*T", head))
        row["LSORBIT"] = bool(re.search(r"LSORBIT\s*=\s*T", head))
        m = re.search(r"generate k-points for:\s+(\d+)\s+(\d+)\s+(\d+)", head)
        row["kmesh"] = "x".join(m.groups()) if m else None
        row["kmesh_auto"] = "Automatic generation of k-mesh" in head
        row["kmesh_gamma"] = bool(re.search(r"Gamma-centered|gamma-centered|Gamma centered", head))
        # tail: convergence and final magnetization
        with open(DATASET / run / "OUTCAR", "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 400_000))
            tail = fh.read().decode(errors="replace")
        row["converged"] = "aborting loop because EDIFF is reached" in tail
        row["finished"] = "General timing and accounting" in tail
        mags = re.findall(r"number of electron\s+[-0-9.]+\s+magnetization\s+([-0-9.]+)", tail)
        row["final_magnetization"] = float(mags[-1]) if mags else np.nan
        lat, _, _ = header(DATASET / run / "CHGCAR")
        if row["kmesh"]:
            n = np.array([int(x) for x in row["kmesh"].split("x")])
            b = 2 * np.pi * np.linalg.inv(lat).T
            row["kspacing_max"] = float(np.max(np.linalg.norm(b, axis=1) / n))   # 1/A, incl. 2 pi
        return row
    except Exception as exc:                                    # noqa: BLE001
        return {"id": run, "error": repr(exc)}


def main():
    runs = [r for r in all_ids() if (DATASET / r / "OUTCAR").exists()]
    with ProcessPoolExecutor(WORKERS) as ex:
        df = pd.DataFrame(list(ex.map(parse, runs, chunksize=20)))
    df.to_csv(OUT / "dft_settings.csv", index=False)
    lines = [f"{len(df)} OUTCARs parsed ({int(df.get('error', pd.Series(dtype=object)).notna().sum())} errors)"]
    for c in ["vasp", "lexch", "GGA", "METAGGA", "PREC", "ENCUT", "ENAUG", "EDIFF", "ISPIN", "ISMEAR", "SIGMA",
              "LREAL", "LASPH", "IALGO", "NSW", "IBRION", "ISIF", "ISYM", "LDAU", "IVDW", "LAECHG", "noncollinear",
              "LSORBIT", "LELF", "LVTOT", "LCHARG", "LORBIT", "kmesh_auto", "kmesh_gamma", "converged", "finished"]:
        lines.append(f"{c}: " + ", ".join(f"{k} ({v})" for k, v in Counter(df[c].astype(str)).most_common(6)))
    ks = df["kspacing_max"].dropna()
    lines.append(f"k-spacing max over axes (1/A, incl. 2pi): median {ks.median():.3f}, 5-95% "
                 f"{ks.quantile(0.05):.3f}-{ks.quantile(0.95):.3f}")
    pots = Counter(p for s in df["potcars"].dropna() for p in re.findall(r"PAW\S*\s+\S+", s))
    lines.append(f"{len(pots)} distinct POTCARs; families: " +
                 ", ".join(f"{k} ({v})" for k, v in Counter(p.split()[0] for p in pots).most_common()))
    (OUT / "dft_settings_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
