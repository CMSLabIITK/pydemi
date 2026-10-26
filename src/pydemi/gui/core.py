"""UI-free helpers for the GUI: read an uploaded CHGCAR and featurize it.

Kept separate from ``app.py`` so it can be tested without Streamlit. A ChargE3Net
prediction (``pydemi.gui.predict``) produces the same ``VolumetricData`` and reuses
everything below. Needs only pandas, a core dependency.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import math
import shutil
import tempfile
import zipfile
from functools import reduce
from pathlib import Path
from typing import IO, Any, Callable, Optional

import numpy as np
import pandas as pd

import pydemi
from pydemi.io.base import VolumetricData

PARTITIONS = ("nearest", "power", "becke")
DOMAINS = ("bonding", "structural", "heterogeneity")

_COMPRESSED: dict[bytes, Callable[..., Any]] = {     # opener(file object) -> file object
    b"\x1f\x8b": gzip.open,
    b"BZh": bz2.open,
    b"\xfd7zXZ\x00": lzma.open,
}


def _zip_member(zf: zipfile.ZipFile) -> zipfile.ZipInfo:
    files = [m for m in zf.infolist() if not m.is_dir()]
    named = [m for m in files if Path(m.filename).name.upper().startswith(("CHGCAR", "CHG"))]
    pick = named if named else files
    if len(pick) != 1:
        raise ValueError("the zip archive must contain exactly one CHGCAR")
    return pick[0]


def _decompressed(src: IO[bytes], dst: IO[bytes]) -> None:
    """Copy ``src`` to ``dst``, undoing gzip / bzip2 / xz / zip compression."""
    head = src.read(6)
    src.seek(0)
    if head.startswith(b"PK"):
        with zipfile.ZipFile(src) as zf, zf.open(_zip_member(zf)) as inner:
            shutil.copyfileobj(inner, dst)
        return
    for magic, opener in _COMPRESSED.items():
        if head.startswith(magic):
            with opener(src) as inner:
                shutil.copyfileobj(inner, dst)
            return
    shutil.copyfileobj(src, dst)


def read_upload(src: IO[bytes] | bytes) -> VolumetricData:
    """``VolumetricData`` from a CHGCAR file object or its bytes (optionally compressed)."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    with tempfile.TemporaryDirectory(prefix="pydemi-") as tmp:
        path = Path(tmp) / "CHGCAR"
        with open(path, "wb") as dst:
            _decompressed(src, dst)
        try:
            return pydemi.read_chgcar(path)
        except Exception as exc:  # parser errors are ValueError/IndexError; report them uniformly
            raise ValueError(f"not a readable VASP CHGCAR ({exc})") from exc


def _formula(counts: dict[str, int]) -> tuple[str, str]:
    def fmt(c: dict[str, int]) -> str:
        return "".join(f"{e}{n if n > 1 else ''}" for e, n in c.items())
    g = reduce(math.gcd, counts.values())
    return fmt(counts), fmt({e: n // g for e, n in counts.items()})


def summary(vd: VolumetricData) -> dict[str, Any]:
    """Structure and grid facts shown next to the descriptors."""
    m = vd.lattice.matrix
    a, b, c = (float(np.linalg.norm(v)) for v in m)

    def angle(u: Any, v: Any) -> float:
        cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
        return math.degrees(math.acos(max(-1.0, min(1.0, cos))))

    formula, reduced = _formula(vd.structure.site_counts())
    return {
        "formula": formula,
        "reduced_formula": reduced,
        "n_atoms": vd.structure.n_atoms,
        "a": a, "b": b, "c": c,
        "alpha": angle(m[1], m[2]), "beta": angle(m[0], m[2]), "gamma": angle(m[0], m[1]),
        "volume": float(vd.structure.volume),
        "grid": "x".join(str(n) for n in vd.shape),
        "electrons": float(vd.rho.data.sum() * vd.rho.dV),
    }


def compute(vd: VolumetricData, partition: str = "nearest") -> tuple[dict[str, float], dict[str, Any]]:
    """All descriptors and the run metadata."""
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition {partition!r}")
    feats, meta = pydemi.featurize(vd, partition=partition, return_metadata=True)
    return feats, meta


def results_table(feats: dict[str, float], meta: Optional[dict[str, Any]] = None) -> pd.DataFrame:
    """One row per descriptor: domain, value, units, formula, sentinel flag."""
    cat = pydemi.catalogue().set_index("name")
    flagged = set()
    if meta and meta.get("sentinels"):
        flagged = {item.split(":")[0] for item in str(meta["sentinels"]).split(";") if item}
    rows = [{"descriptor": n, "domain": cat.at[n, "domain"], "value": v,
             "units": cat.at[n, "units"], "formula": cat.at[n, "formula"],
             "fallback": n in flagged} for n, v in feats.items()]
    table: pd.DataFrame = pd.DataFrame(rows)
    return table


def chgcar_bytes(vd: VolumetricData, comment: str = "written by pydemi") -> bytes:
    """``vd`` as a VASP CHGCAR (total density only), for download."""
    with tempfile.TemporaryDirectory(prefix="pydemi-") as tmp:
        path = Path(tmp) / "CHGCAR"
        pydemi.write_volumetric(path, vd.structure, [vd.rho.data], comment=comment)
        return path.read_bytes()


def wide_row(label: str, feats: dict[str, float]) -> pd.DataFrame:
    """The descriptors as one CSV row, ready to stack into a dataset."""
    row: pd.DataFrame = pd.DataFrame([{"structure": label, **feats}])
    return row
