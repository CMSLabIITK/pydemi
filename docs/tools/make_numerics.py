"""Numbers of the user-guide pages on numerical settings and validation, from real CHGCARs.

    python docs/tools/make_numerics.py /path/to/6000_data_aug13

Writes docs/_generated/numerics.json: for GaAs (cubic) and SbO2 (monoclinic) the change of
every descriptor with the grid (Fourier-resampled to 90 % and 80 % of the points per axis),
with finite differences instead of FFT, with the diagonal Laplacian, with float32 and with
the Becke partition; and the invariance checks on GaAs (2x1x1 supercell, a translation by
whole voxels, a rotation).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import pydemi
from pydemi.validate import convergence, invariance

OUT = Path(__file__).resolve().parents[1] / "_generated" / "numerics.json"


def rel(a: dict, b: dict) -> dict:
    return {k: float(abs(b[k] - a[k]) / max(abs(a[k]), 1e-12)) if abs(a[k]) > 1e-9 or abs(b[k]) > 1e-9
            else 0.0 for k in a}


def main(root: Path) -> None:
    res = {}
    for name in ("GaAs_216", "SbO2_15"):
        vd = pydemi.read_chgcar(root / name / "CHGCAR")
        base = pydemi.featurize(vd)
        r = {"base": base}
        g = convergence.grid_convergence(pydemi.read_chgcar(root / name / "CHGCAR"))
        r["grid"] = json.loads(g.to_json(orient="records"))
        for key, kw in [("fd4", dict(derivative_backend="fd", fd_order=4)),
                        ("diagonal", dict(laplacian_method="diagonal")),
                        ("float32", dict(float32=True)),
                        ("becke", dict(partition="becke"))]:
            r[key] = rel(base, pydemi.featurize(pydemi.read_chgcar(root / name / "CHGCAR"), **kw))
        res[name] = r
        print(name, "done")
    vd = pydemi.read_chgcar(root / "GaAs_216" / "CHGCAR")
    base = res["GaAs_216"]["base"]
    th = np.deg2rad(30.0)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    inv = {}
    for key, v in [("supercell_2x1x1", invariance.supercell(vd, (2, 1, 1))),
                   ("translate_5_7_11", invariance.translate(vd, (5, 7, 11))),
                   ("rotate_30deg_z", invariance.rotate(vd, R))]:
        f = pydemi.featurize(v)
        d = rel(base, f)
        inv[key] = {"max_rel": max(d.values()), "worst": max(d, key=d.get),
                    "differ_1e-6": invariance.compare(base, f)}
    res["invariance"] = inv
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
