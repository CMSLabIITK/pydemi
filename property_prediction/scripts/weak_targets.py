"""Why six targets stay below test R2 = 0.70 with the pydemi descriptors, from the study_v1 results.

Targets: Poisson ratio, universal anisotropy, energy above hull, and the electronic, ionic and
total dielectric constants. For each: the data (counts, tails, zeros), the scores of every model
and feature set, a bootstrap interval of the test R2, scores on the modelling scale (log) and
by rank, the error of a constant prediction, and a learning curve (descriptor LightGBM with the
tuned hyperparameters of study_v1, 25-100 % of the training split, three seeds). Plus three
targeted checks: nu from the predicted K and G, a stable/unstable classifier for the hull, and
the Penn-model features for the dielectric constants (results/study_v1_penn).

    OMP_NUM_THREADS=2 python scripts/weak_targets.py
    -> results/weak_targets/{numbers.json, *.pdf}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import skew, spearmanr  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402
from train_preview import back, forward  # noqa: E402

ROOT = train.ROOT
STUDY = ROOT / "results/study_v1"
OUT = ROOT / "results/weak_targets"
WEAK = ["poisson_ratio", "anisotropy_index", "energy_above_hull", "dielectric_electronic",
        "dielectric_ionic", "dielectric_total"]
SYM = {"poisson_ratio": r"$\nu$", "anisotropy_index": "$A^U$", "energy_above_hull": r"$E_\mathrm{hull}$",
       "dielectric_electronic": r"$\varepsilon_\infty$", "dielectric_ionic": r"$\varepsilon_\mathrm{ion}$",
       "dielectric_total": r"$\varepsilon$"}
UNITS = {"energy_above_hull": "eV/atom"}
rng = np.random.default_rng(0)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def jobres(t: str, fs: str, study: Path = STUDY) -> dict:
    return json.loads((study / "jobs" / f"{t}__{fs}.json").read_text())


def boot_r2(y: np.ndarray, p: np.ndarray, n: int = 2000) -> tuple[float, float]:
    idx = rng.integers(0, len(y), (n, len(y)))
    r = [r2_score(y[i], p[i]) for i in idx]
    return float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def learning_curve(t: str, d: pd.DataFrame, feats: list[str], how: str | None, floor) -> list[dict]:
    params = dict(objective="regression", metric="l2", verbose=-1, bagging_freq=1, num_threads=2,
                  feature_pre_filter=False, **jobres(t, "desc")["models"]["lgbm"]["params"])
    y = d[t].to_numpy(float)
    ok = ~np.isnan(y)
    tr, va, te = ((d.split == s).to_numpy() & ok for s in ("train", "val", "test"))
    X, z = d[feats].to_numpy(float), forward(y, how)
    itr = np.flatnonzero(tr)
    rows = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        for s in range(3):
            sub = np.sort(np.random.default_rng(s).permutation(itr)[:int(round(frac * len(itr)))])
            dtr = lgb.Dataset(X[sub], z[sub], params={"feature_pre_filter": False})
            m = lgb.train(dict(params, seed=s), dtr, 10000, valid_sets=[lgb.Dataset(X[va], z[va], reference=dtr)],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            pz = m.predict(X[te], num_iteration=m.best_iteration)
            p = back(pz, how)
            p = np.maximum(p, floor) if floor is not None else p
            rows.append({"frac": frac, "n": len(sub), "seed": s, "r2": r2_score(y[te], p),
                         "r2_model_scale": r2_score(z[te], pz)})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d, sets, _ = train.load()
    num: dict = {}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for ax, t in zip(axes.flat, WEAK):
        jd = jobres(t, "desc")
        how, floor = jd["transform"], jd["floor"]
        y = np.asarray(jd["test_y"])
        p = np.asarray(jd["models"]["lgbm"]["ensemble"]["test_p"])
        all_y = d[t].dropna().to_numpy()
        tr_y = d.loc[d.split == "train", t].dropna().to_numpy()
        e: dict = {"n": {s: int(d.loc[d.split == s, t].notna().sum()) for s in ("train", "val", "test")},
                   "median": float(np.median(all_y)), "q25": float(np.quantile(all_y, 0.25)),
                   "q75": float(np.quantile(all_y, 0.75)), "q99": float(np.quantile(all_y, 0.99)),
                   "max": float(all_y.max()), "std": float(all_y.std()), "skew": float(skew(all_y)),
                   "zero_frac": float(np.mean(all_y == 0)), "transform": how}
        sc = {}
        for fs in ("comp", "desc", "both"):
            j = jobres(t, fs)
            for m in ("ridge", "rf", "lgbm"):
                sc[f"{m}_{fs}"] = float(np.mean([r["r2"] for r in j["models"][m]["seeds"]]))
        e["scores"] = sc
        e["lgbm_desc_sd"] = float(np.std([r["r2"] for r in jd["models"]["lgbm"]["seeds"]]))
        e["ens_r2"] = float(r2_score(y, p))
        e["ens_r2_ci"] = boot_r2(y, p)
        e["spearman"] = float(spearmanr(y, p)[0])
        e["r2_model_scale"] = float(r2_score(forward(y, how), forward(p, how))) if how else e["ens_r2"]
        e["mae"] = float(mean_absolute_error(y, p))
        e["mae_const_median"] = float(mean_absolute_error(y, np.full_like(y, np.median(tr_y))))
        # share of the squared error from the 5 % of test materials with the largest values
        big = y >= np.quantile(y, 0.95)
        e["sse_share_top5pct"] = float(((y - p) ** 2)[big].sum() / ((y - p) ** 2).sum())
        e["r2_without_top5pct"] = float(r2_score(y[~big], p[~big]))
        cache = OUT / f"lc_{t}.json"
        if not cache.exists():
            cache.write_text(json.dumps(learning_curve(t, d, sets["desc"], how, floor)))
        e["learning_curve"] = json.loads(cache.read_text())
        num[t] = e
        print(t, {k: v for k, v in e.items() if k not in ("learning_curve", "scores")}, flush=True)
        # parity
        lo = max(min(y.min(), p.min()), 1e-3) if how else min(y.min(), p.min())
        ax.scatter(y, p, s=5, alpha=0.5, lw=0, color="#1f77b4")
        hi = max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], ls=":", c="0.4", lw=0.8)
        if how == "log1p":
            ax.set_xscale("symlog", linthresh=0.1)
            ax.set_yscale("symlog", linthresh=0.1)
        elif how:
            ax.set_xscale("log")
            ax.set_yscale("log")
        u = f" ({UNITS[t]})" if t in UNITS else ""
        ax.set_xlabel(f"DFT {SYM[t]}{u}")
        ax.set_ylabel(f"predicted {SYM[t]}{u}")
        ax.set_title(f"{SYM[t]}: $R^2$ {e['ens_r2']:.2f}, Spearman {e['spearman']:.2f}, n={len(y)}", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "parity.pdf")
    plt.close(fig)

    # learning curves
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for t in WEAK:
        lc = pd.DataFrame(num[t]["learning_curve"]).groupby("n")
        for k, col in enumerate(("r2", "r2_model_scale")):
            m = lc[col].mean()
            ax[k].errorbar(m.index, m.values, yerr=lc[col].std().values, marker="o", ms=3, lw=0.9, label=SYM[t])
    for a, ttl in zip(ax, ("test $R^2$ (target units)", "test $R^2$ (modelling scale: log for $A^U$, $\\varepsilon$'s)")):
        a.set_xscale("log")
        a.set_xticks([300, 500, 1000, 2000, 5000], ["300", "500", "1000", "2000", "5000"])
        a.minorticks_off()
        a.set_xlabel("training materials")
        a.set_title(ttl, loc="left")
        a.grid(lw=0.3, color="0.85")
    ax[0].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves.pdf")
    plt.close(fig)

    # target distributions
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.8))
    for ax, t in zip(axes.flat, WEAK):
        v = d[t].dropna().to_numpy()
        how = num[t]["transform"]
        ax.hist(forward(v, how) if how else v, bins=60, color="0.55")
        lab = {"log10": f"log$_{{10}}$ {SYM[t]}", "log1p": f"ln(1+{SYM[t]})"}.get(how, SYM[t])
        ax.set_xlabel(lab + (f" ({UNITS[t]})" if t in UNITS and not how else ""))
        ax.set_yscale("log")
        ax.set_title(f"n = {len(v)}, skew {num[t]['skew']:.1f}, zeros {100 * num[t]['zero_frac']:.0f} %", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "distributions.pdf")
    plt.close(fig)

    # Poisson ratio from the predicted K and G (descriptor models, same test materials)
    jk, jg, jn = jobres("bulk_modulus", "desc"), jobres("shear_modulus", "desc"), jobres("poisson_ratio", "desc")
    K = pd.Series(jk["models"]["lgbm"]["ensemble"]["test_p"], jk["test_material_id"])
    G = pd.Series(jg["models"]["lgbm"]["ensemble"]["test_p"], jg["test_material_id"])
    nu = pd.Series(jn["test_y"], jn["test_material_id"])
    Kt, Gt = pd.Series(jk["test_y"], jk["test_material_id"]), pd.Series(jg["test_y"], jg["test_material_id"])
    ids = nu.index
    f = lambda k, g: (3 * k - 2 * g) / (2 * (3 * k + g))  # noqa: E731
    tab = pd.DataFrame({"nu": nu, "true": f(Kt[ids], Gt[ids]), "pred": f(K[ids], G[ids]),
                        "gk_true": Gt[ids] / Kt[ids], "gk_pred": G[ids] / K[ids]})
    good = tab.replace([np.inf, -np.inf], np.nan).dropna()
    num["poisson_checks"] = {
        "n": len(good), "n_dropped": len(tab) - len(good),
        "r2_from_true_KG": float(r2_score(good.nu, good.true)),
        "r2_from_predicted_KG": float(r2_score(good.nu, good.pred)),
        "r2_direct": num["poisson_ratio"]["ens_r2"],
        "r2_G_over_K_pred": float(r2_score(good.gk_true, good.gk_pred)),
        "std_nu": float(nu.std()), "mae_direct": num["poisson_ratio"]["mae"]}
    # energy above the hull: on the hull or not
    y = d["energy_above_hull"].to_numpy(float)
    ok = ~np.isnan(y)
    X = d[sets["desc"]].to_numpy(float)
    tr, va, te = ((d.split == s).to_numpy() & ok for s in ("train", "val", "test"))
    c = (y > 1e-9).astype(float)
    dtr = lgb.Dataset(X[tr], c[tr])
    ps = []
    for s in range(3):
        m = lgb.train({"objective": "binary", "verbose": -1, "learning_rate": 0.03, "num_leaves": 31,
                       "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": s,
                       "num_threads": 4}, dtr, 5000, valid_sets=[lgb.Dataset(X[va], c[va], reference=dtr)],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        ps.append(m.predict(X[te], num_iteration=m.best_iteration))
    num["hull_checks"] = {"on_hull_frac_test": float(np.mean(c[te] == 0)),
                          "on_hull_frac_all": float(np.mean(c[ok] == 0)),
                          "classifier_auc": float(roc_auc_score(c[te], np.mean(ps, axis=0))),
                          "mae_predict_zero": float(np.mean(np.abs(y[te]))),
                          "mae_model": num["energy_above_hull"]["mae"],
                          "median_nonzero": float(np.median(y[ok][y[ok] > 0]))}
    # dielectric: Penn features, and the gap as the missing input
    pen = {}
    for t in ("dielectric_electronic", "dielectric_ionic", "dielectric_total"):
        pen[t] = {fs: float(np.mean([r["r2"] for r in jobres(t, fs, ROOT / "results/study_v1_penn")["models"]["lgbm"]["seeds"]]))
                  for fs in ("descpenn", "penn")}
    eg = d["band_gap"].to_numpy(float)
    epsi = d["dielectric_electronic"].to_numpy(float)
    m_ = ~np.isnan(eg) & ~np.isnan(epsi) & (eg > 0)
    pen["spearman_eps_inf_vs_gap"] = float(spearmanr(epsi[m_], eg[m_])[0])
    ion = d["dielectric_ionic"].dropna()
    pen["ionic_share_of_total_median"] = float((d["dielectric_ionic"] / d["dielectric_total"]).median())
    pen["ionic_gt_50"] = int((ion > 50).sum())
    num["dielectric_checks"] = pen
    (OUT / "numbers.json").write_text(json.dumps(num, indent=1))
    for k in ("poisson_checks", "hull_checks", "dielectric_checks"):
        print(k, num[k])


if __name__ == "__main__":
    main()
