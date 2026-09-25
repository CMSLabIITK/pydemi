"""Partition schemes.

1. Becke weights in a periodic solid (FeNi3, 32^3 voxels): maximum weight
   error against k = 400 atom images in each product, for the k used, and the
   Hirshfeld weights' partition of unity.
2. Which descriptors depend on the partition, and by how much: every
   descriptor of every domain under nearest / power / becke / hirshfeld for N
   random structures, relative to the nearest-atom value.

Usage:  python b_partitions.py [N=300]
Output: out/partitions_becke.json, out/partitions.csv, out/partitions_summary.csv
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import pydemi
from common import OUT, WORKERS, load, sample_ids
from pydemi.core.partition import BeckePartition, HirshfeldPartition, hirshfeld_tables

SCHEMES = ("nearest", "power", "becke", "hirshfeld")
DOMAINS = ["bonding", "structural", "magnetic", "heterogeneity"]


def dense(partition, n_vox, n_atoms):
    w = np.zeros((n_vox, n_atoms))
    for c in partition.pairs():
        np.add.at(w, (c.voxel, c.atom), c.weight)
    return w


def becke_convergence():
    vd = load("FeNi3_221")
    shape, s = (32, 32, 32), vd.structure
    n_vox = int(np.prod(shape))
    ref = dense(BeckePartition(shape, s, k=400), n_vox, s.n_atoms)
    out = {"structure": "FeNi3_221", "shape": list(shape), "reference_k": 400, "cells": 8,
           "max_weight_error": {}}
    for k in (8, 20, 60, 100, 200, 300):
        w = dense(BeckePartition(shape, s, k=k), n_vox, s.n_atoms)
        out["max_weight_error"][k] = float(np.abs(w - ref).max())
    small = vd.with_options(vd.options)
    h = dense(HirshfeldPartition(vd.shape, s, hirshfeld_tables(small)), int(np.prod(vd.shape)), s.n_atoms)
    out["hirshfeld_max_unity_error"] = float(np.abs(h.sum(axis=1) - 1).max())
    return out


def one(mid):
    try:
        vd = load(mid)
        row = {"id": mid, "n_atoms": vd.structure.n_atoms}
        for p in SCHEMES:
            f = pydemi.featurize(vd, domains=DOMAINS, partition=p)
            row.update({f"{k}@{p}": v for k, v in f.items()})
        return row
    except Exception as exc:                                    # noqa: BLE001
        return {"id": mid, "error": repr(exc)}


def summary(df):
    names = [c.split("@")[0] for c in df.columns if c.endswith("@nearest")]
    rows = []
    for n in names:
        ref = df[f"{n}@nearest"].astype(float)
        for p in SCHEMES[1:]:
            v = df[f"{n}@{p}"].astype(float)
            ok = np.isfinite(ref) & np.isfinite(v)
            rel = ((v - ref).abs() / np.maximum(np.maximum(v.abs(), ref.abs()), 1e-12))[ok]
            if len(rel) == 0 or rel.max() < 1e-9:
                continue
            rows.append({"descriptor": n, "partition": p, "n": int(ok.sum()), "median": rel.median(),
                         "p95": rel.quantile(0.95), "spearman": ref[ok].rank().corr(v[ok].rank())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    becke = becke_convergence()
    (OUT / "partitions_becke.json").write_text(json.dumps(becke, indent=1))
    print(json.dumps(becke, indent=1))
    with ProcessPoolExecutor(WORKERS) as pool:
        df = pd.DataFrame(list(pool.map(one, sample_ids(n), chunksize=1)))
    df.to_csv(OUT / "partitions.csv", index=False)
    ok = df[df["error"].fillna("") == ""] if "error" in df else df
    print(f"{len(ok)} of {len(df)} structures")
    s = summary(ok)
    s.to_csv(OUT / "partitions_summary.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_rows", 300):
        print(s.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    same = sorted({c.split("@")[0] for c in ok.columns if c.endswith("@nearest")} - set(s["descriptor"]))
    print(f"{len(same)} descriptors independent of the partition")
