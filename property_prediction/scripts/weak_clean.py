"""Dielectric targets without the materials whose PBE gap is below 0.1 eV (metals or near-metals,
for which a DFPT dielectric constant is not meaningful): descriptor LightGBM with the tuned
hyperparameters of study_v1, 3 seeds, scored on all test materials and on those with a gap.

    python scripts/weak_clean.py -> results/weak_targets/clean.json
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402
from train_preview import back, forward  # noqa: E402

d, sets, _ = train.load()
gap = d["band_gap"].to_numpy(float) >= 0.1
X = d[sets["desc"]].to_numpy(float)
res = {}
for t in ("dielectric_electronic", "dielectric_ionic", "dielectric_total"):
    j = json.loads((train.ROOT / f"results/study_v1/jobs/{t}__desc.json").read_text())
    how, floor = j["transform"], j["floor"]
    y = d[t].to_numpy(float)
    ok = ~np.isnan(y)
    z = forward(y, how)
    res[t] = {"n_removed_train_val": int((ok & ~gap & (d.split != "test").to_numpy()).sum()),
              "n_test_without_gap": int((ok & ~gap & (d.split == "test").to_numpy()).sum())}
    for name, keep in (("all", ok), ("gap>=0.1", ok & gap)):
        tr, va = ((d.split == s).to_numpy() & keep for s in ("train", "val"))
        out = {}
        for tname, te in (("test_all", (d.split == "test").to_numpy() & ok),
                          ("test_gap", (d.split == "test").to_numpy() & ok & gap)):
            r, rl, sp = [], [], []
            for s in range(3):
                dtr = lgb.Dataset(X[tr], z[tr], params={"feature_pre_filter": False})
                m = lgb.train(dict(objective="regression", verbose=-1, bagging_freq=1, num_threads=4, seed=s,
                                   feature_pre_filter=False, **j["models"]["lgbm"]["params"]), dtr, 10000,
                              valid_sets=[lgb.Dataset(X[va], z[va], reference=dtr)],
                              callbacks=[lgb.early_stopping(200, verbose=False)])
                pz = m.predict(X[te], num_iteration=m.best_iteration)
                p = np.maximum(back(pz, how), floor)
                r.append(r2_score(y[te], p)); rl.append(r2_score(z[te], pz)); sp.append(spearmanr(y[te], p)[0])
            out[tname] = {"n": int(te.sum()), "r2": float(np.mean(r)), "r2_sd": float(np.std(r)),
                          "r2_log": float(np.mean(rl)), "spearman": float(np.mean(sp))}
        res[t][f"train_{name}"] = out
    print(t, json.dumps(res[t]), flush=True)
(train.ROOT / "results/weak_targets/clean.json").write_text(json.dumps(res, indent=1))
