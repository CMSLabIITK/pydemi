"""Where the percolation bottlenecks and the density minimum are, and cell-choice tests.

For N random structures (seed 2026):
- the bottleneck voxel of each axis: rho_perc_alpha is the value of one voxel (the one
  whose removal disconnects the spanning cluster), so it is located exactly. Recorded: its
  distance r1 to the nearest nucleus and r1 / R_PAW, the two nearest nuclei and the
  position along their segment (|d1 - d2| / (d1 + d2): 0 at the midpoint), and the
  distance between those two nuclei relative to the shortest interatomic distance;
- the voxel of rho_min: r / R_PAW of its nearest atom, and whether rho_min < 0;
- a 2 x 1 x 1 supercell (density tiled): the three levels must be unchanged;
- a change of cell a1' = a1 + a2 (same lattice, same grid, same volume; only when
  N1 = N2): the level along a1' is a different wrapping class, so it may differ.

Usage:  python bottlenecks.py [N=200]   -> ../data/bottlenecks.csv, cell_tests.csv
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "paper" / "analysis"))
from common import load, sample_ids  # noqa: E402

from pydemi.operators.topology import percolation_levels  # noqa: E402
from pydemi.descriptors.registry import augmentation_radii, geometry, make_options  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data"


def two_nearest(s, frac):
    """Distances to the two nearest nuclei (with images) of a fractional point, and d_min of the pair."""
    lat = s.lattice.matrix
    sh = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    pts = (s.frac_coords[None, :, :] + sh[:, None, :]).reshape(-1, 3)
    owner = np.tile(np.arange(len(s.frac_coords)), len(sh))
    d = np.linalg.norm((pts - frac) @ lat, axis=1)
    o = np.argsort(d)[:2]
    pair = np.linalg.norm((pts[o[0]] - pts[o[1]]) @ lat)
    return d[o[0]], d[o[1]], pair, owner[o[0]], owner[o[1]]


def shortest_distance(s):
    lat = s.lattice.matrix
    sh = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    best = np.inf
    for a in range(len(s.frac_coords)):
        d = np.linalg.norm(((s.frac_coords[None, :, :] + sh[:, None, :]) - s.frac_coords[a]).reshape(-1, 3) @ lat,
                           axis=1)
        best = min(best, d[d > 1e-6].min())
    return best


def one(mid):
    try:
        vd = load(mid)
        v = vd.with_options(make_options())
        rho = vd.rho.data
        s = vd.structure
        geo = geometry(v)
        R = augmentation_radii(v)[0]
        levels = percolation_levels(rho)
        dmin = shortest_distance(s)
        row = {"id": mid, "n_atoms": s.n_atoms, "d_min": dmin}
        n = np.array(rho.shape)
        for ax, lab in enumerate("abc"):
            idx = np.argwhere(rho == levels[ax])
            k = tuple(idx[0])
            frac = np.array(k) / n
            d1, d2, pair, i1, i2 = two_nearest(s, frac)
            row.update({f"perc_{lab}": float(levels[ax]), f"n_voxels_{lab}": int(len(idx)),
                        f"r_{lab}": float(geo.distance[k]), f"r_over_Rpaw_{lab}": float(geo.distance[k] / R[geo.atom_index[k]]),
                        f"midpointness_{lab}": float(abs(d1 - d2) / (d1 + d2)),
                        f"pair_over_dmin_{lab}": float(pair / dmin),
                        f"pair_elements_{lab}": "-".join(sorted([s.species[i1], s.species[i2]]))})
        kmin = np.unravel_index(np.argmin(rho), rho.shape)
        row.update({"rho_min": float(rho[kmin]), "rmin_over_Rpaw": float(geo.distance[kmin] / R[geo.atom_index[kmin]]),
                    "rmin_A": float(geo.distance[kmin])})
        return row
    except Exception as exc:                                    # noqa: BLE001
        return {"id": mid, "error": repr(exc)}


def cells(mid):
    try:
        vd = load(mid)
        rho = vd.rho.data
        row = {"id": mid, "shape": "x".join(map(str, rho.shape))}
        base = percolation_levels(rho)
        sup = percolation_levels(np.tile(rho, (2, 1, 1)))
        row.update({f"base_{l}": float(x) for l, x in zip("abc", base)})
        row.update({f"super_{l}": float(x) for l, x in zip("abc", sup)})
        n1, n2, _ = rho.shape
        if n1 == n2:
            i, j = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
            # new cell a1' = a1 + a2, a2' = a2: new fractional (f1', f2') = old (f1', f1' + f2')
            rho2 = rho[i, (i + j) % n2, :]
            sheared = percolation_levels(rho2)
            row.update({f"sheared_{l}": float(x) for l, x in zip("abc", sheared)})
        return row
    except Exception as exc:                                    # noqa: BLE001
        return {"id": mid, "error": repr(exc)}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    ids = sample_ids(n, seed=2026)
    with ProcessPoolExecutor(12) as ex:
        b = pd.DataFrame(list(ex.map(one, ids, chunksize=1)))
        c = pd.DataFrame(list(ex.map(cells, ids[:60], chunksize=1)))
    b.to_csv(OUT / "bottlenecks.csv", index=False)
    c.to_csv(OUT / "cell_tests.csv", index=False)
    ok = b[b["error"].isna()] if "error" in b else b
    print(len(ok), "structures")
    cols = [c_ for c_ in ok.columns if any(c_.startswith(p) for p in ("r_", "midpointness", "pair_over", "n_voxels"))]
    print(ok[cols + ["rmin_over_Rpaw"]].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.round(3).to_string())
    print("rho_min < 0:", (ok["rho_min"] < 0).mean(), " rho_min inside PAW:", (ok["rmin_over_Rpaw"] <= 1).mean())
    ok2 = c[c["error"].isna()] if "error" in c else c
    print("supercell max |diff|:", float(np.max(np.abs(ok2[[f"super_{l}" for l in "abc"]].to_numpy()
                                                       - ok2[[f"base_{l}" for l in "abc"]].to_numpy()))))
    sh = ok2.dropna(subset=["sheared_a"])
    rel = (sh["sheared_a"] - sh["base_a"]).abs() / sh["base_a"]
    print("sheared a1+a2:", len(sh), "structures; rel change of the a-level median", rel.median(), "max", rel.max(),
          "; b, c unchanged:", float(np.max(np.abs(sh[["sheared_b", "sheared_c"]].to_numpy()
                                               - sh[["base_b", "base_c"]].to_numpy()))))
