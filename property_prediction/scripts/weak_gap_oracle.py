"""Upper bound for the dielectric targets: descriptors + the DFT (PBE) band gap as an input.

The gap is not available without a calculation; this only measures how much of the missing
information it carries. LightGBM with the tuned hyperparameters of study_v1 (desc), 3 seeds.

    python scripts/weak_gap_oracle.py -> results/weak_targets/gap_oracle.json
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402
from train_preview import back, forward  # noqa: E402

d, sets, _ = train.load()
res = {}
for t in ("dielectric_electronic", "dielectric_ionic", "dielectric_total", "refractive_index"):
    j = json.loads((train.ROOT / f"results/study_v1/jobs/{t}__desc.json").read_text())
    how, floor = j["transform"], j["floor"]
    y = d[t].to_numpy(float)
    ok = ~np.isnan(y)
    tr, va, te = ((d.split == s).to_numpy() & ok for s in ("train", "val", "test"))
    z = forward(y, how)
    res[t] = {}
    for name, feats in (("desc", sets["desc"]), ("desc+gap", sets["desc"] + ["band_gap"])):
        X = d[feats].to_numpy(float)
        p_all = []
        for s in range(3):
            dtr = lgb.Dataset(X[tr], z[tr], params={"feature_pre_filter": False})
            m = lgb.train(dict(objective="regression", verbose=-1, bagging_freq=1, num_threads=4, seed=s,
                               feature_pre_filter=False, **j["models"]["lgbm"]["params"]), dtr, 10000,
                          valid_sets=[lgb.Dataset(X[va], z[va], reference=dtr)],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            p_all.append(m.predict(X[te], num_iteration=m.best_iteration))
        r = [r2_score(y[te], np.maximum(back(pz, how), floor)) for pz in p_all]
        rl = [r2_score(z[te], pz) for pz in p_all]
        res[t][name] = {"r2": float(np.mean(r)), "r2_sd": float(np.std(r)), "r2_log": float(np.mean(rl)),
                        "spearman": float(np.mean([spearmanr(y[te], pz)[0] for pz in p_all]))}
    print(t, res[t], flush=True)
(train.ROOT / "results/weak_targets/gap_oracle.json").write_text(json.dumps(res, indent=1))
