"""The ChargE3Net example of the documentation: GaAs predicted from its structure alone.

    pydemi predict GaAs.vasp --out-dir predicted --featurize --device cpu
    python docs/tools/make_ml_example.py DFT_CHGCAR predicted/GaAs.CHGCAR SECONDS

Adds "ml" to docs/_generated/examples.json: the density error (NMAPE), every descriptor from
both densities as the Compare view of the interface tabulates them, and the property
estimates from each.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pydemi
from pydemi import props
from pydemi.gui import compare

EX = Path(__file__).resolve().parents[1] / "_generated" / "examples.json"


def main(dft_path: str, ml_path: str, seconds: float) -> None:
    dft, ml = pydemi.read_chgcar(dft_path), pydemi.read_chgcar(ml_path)
    fd, fm = pydemi.featurize(dft), pydemi.featurize(ml)
    table = compare.descriptors(fd, fm)
    rows = [{"name": r.name, "dft": r.DFT, "ml": r.ML,
             "diff": None if not math.isfinite(r._6) else r._6, "note": r.note,
             "transfer": r.transfer} for r in table.itertuples()]
    ex = json.loads(EX.read_text())
    ex["ml"] = {"structure": "GaAs_216", "grid": list(dft.shape), "seconds": seconds,
                "nmape": compare.density_error(dft, ml),
                "electrons_dft": float(dft.rho.data.mean() * dft.structure.volume),
                "electrons_ml": float(ml.rho.data.mean() * ml.structure.volume),
                "descriptors": rows, "summary": compare.descriptor_summary(table),
                "props_dft": props.predict(fd), "props_ml": props.predict(fm)}
    EX.write_text(json.dumps(ex, indent=1, default=float))
    print(json.dumps({k: v for k, v in ex["ml"].items() if k not in ("descriptors", "props_dft", "props_ml")},
                     indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]))
