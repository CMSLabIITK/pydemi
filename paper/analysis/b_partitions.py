"""Partition weight convergence (Becke vs Hirshfeld) on a real VASP density,
and dataset-wide sensitivity of shell/site descriptors to the partition."""
import csv
import json
import math

import numpy as np
from scipy.stats import spearmanr

from common import DATASET, OUT, RESULTS
from pydemi import Engine
from pydemi.partition import BeckePartition, HirshfeldPartition, _becke_step

MID = "FeNi3_221"


def becke_weights(d, P, nc, K):
    dd, PP = d[:, :K], P[:, :K]
    R = np.linalg.norm(PP[:, :nc, None] - PP[:, None, :, :], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.nan_to_num((dd[:, :nc, None] - dd[:, None, :]) / R)
    s = _becke_step(mu)
    s[:, np.arange(nc), np.arange(nc)] = 1.0
    cell = s.prod(axis=2)
    return cell / cell.sum(1, keepdims=True)


def main():
    eng = Engine.from_vasp_dir(DATASET / MID)
    x = eng.grid().cart_coords().reshape(-1, 3)[::37]
    out = {"structure": MID, "grid": list(eng.grid().shape), "sample_voxels": len(x)}

    # Becke: products over K competitors (8 candidate cells), reference K = 400
    part = BeckePartition(eng.grid(), eng.structure, k=400, cells=8)
    d, idx = part._query(x)
    P = part._pts[idx]
    ref = becke_weights(d, P, 8, 400)
    out["becke"] = [{"K": K, "radius_A": float(d[:, K - 1].mean()),
                     "max_weight_error": float(np.abs(becke_weights(d, P, 8, K) - ref).max())}
                    for K in (20, 40, 60, 100, 150, 200, 300)]
    out["becke_weight_beyond_8_cells"] = float((1 - becke_weights(d, P, 13, 400)[:, :8].sum(1)).max())

    # Hirshfeld: weights over images within r_cut, reference r_cut = 10 A
    radial = {e: eng.reference.radial(e, "valence", eng.valence()[e]) for e in eng.structure.elements}
    hp = HirshfeldPartition(eng.grid(), eng.structure, radial, r_cut=10.0)
    from scipy.spatial import cKDTree
    tree = cKDTree(hp._pts)
    def weights(rc):
        W = np.zeros((len(x), eng.structure.n_atoms))
        for i, xi in enumerate(x):
            cand = np.array(tree.query_ball_point(xi, rc), int)
            dist = np.linalg.norm(hp._pts[cand] - xi, axis=1)
            dens = np.array([hp.density(hp._elem[hp._owner[c]], np.array([dd]))[0] for c, dd in zip(cand, dist)])
            np.add.at(W[i], hp._owner[cand], dens)
            W[i] /= W[i].sum()
        return W
    Wref = weights(10.0)
    out["hirshfeld"] = [{"r_cut_A": rc, "max_weight_error": float(np.abs(weights(rc) - Wref).max())}
                        for rc in (3.5, 4.5, 5.5, 6.5, 7.5)]

    # dataset-wide partition sensitivity
    rows = list(csv.DictReader(open(RESULTS)))
    def col(c):
        return np.array([float(r[c]) if r[c] not in ("", "nan") else np.nan for r in rows])
    sens = {}
    for X in ("m1", "f_bond", "zeta"):
        base = col(X)
        for scheme in ("power", "hirshfeld"):
            v = col(f"{X}_{scheme}")
            ok = np.isfinite(base) & np.isfinite(v)
            rel = np.abs(v[ok] - base[ok]) / np.abs(base[ok])
            sens[f"{X}_{scheme}"] = {"n": int(ok.sum()), "median_rel_diff": float(np.median(rel)),
                                    "p95_rel_diff": float(np.percentile(rel, 95)),
                                    "spearman_vs_nearest": float(spearmanr(base[ok], v[ok])[0])}
    out["dataset_partition_sensitivity"] = sens
    (OUT / "partitions.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
