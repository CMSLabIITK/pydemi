# Unused, redundant and unimportant descriptors, and the accuracy without them

Started 2026-10-30. Question: which of the 56 pydemi descriptors are unused, redundant
(strongly correlated with another descriptor) or unimportant, and how accurate are the
17 property models of the paper without them?

`results/` and `report/` are git-ignored (values from the CMS Lab dataset).

```
redundancy/
  report/redundancy_report.pdf   (local) the report: accuracy with all 56 vs without the
                                 redundant descriptors, each redundant descriptor and why
  report/build.py                builds it from results/ (prose in redundancy_report.tex)
  scripts/
    redundancy.py    flags every descriptor (train split only) -> descriptor sets
    train_sets.py    retrains the 17 properties on the reduced sets (paper protocol)
    compare.py       test accuracy vs the 55-descriptor models, with bootstrap intervals
  results/
    redundancy.txt   readable printout: every descriptor, its flags, the 11 groups
    descriptors.csv  per descriptor: domain, NaN/mode fraction, SHAP share, # top-10 lists,
                     max |Spearman| with another descriptor and with any property, flags
    pairs.csv        descriptor pairs with |Spearman rho| >= 0.90
    groups.csv       correlated groups: which descriptor is kept, which are dropped
    sets.json        all55 / no_redundant (37) / important34 (34) / lean (26)
    train/           retrained models: jobs/<target>__<set>.json, models/, train.log
    compare.csv      the accuracy table below, with MAE and bootstrap intervals
```

## Definitions (train split only)

* **unused**: constant (relative std < 1e-9) or one value in >= 95 % of structures.
  Only `lap_concentration` (= 1/2 by construction).
* **redundant**: |Spearman rho| >= 0.90 with another descriptor. Groups by complete linkage
  on 1 - |rho| (every pair in a group >= 0.90); each group keeps the member with the largest
  mean SHAP share over the 17 properties. 18 descriptors dropped, in 11 groups.
* **unimportant**: in none of the 17 top-10 SHAP lists (paper Fig. 5). 21 descriptors.

Sets: `all55` (paper), `no_redundant` = all55 minus redundant (37), `important34` = the 34
that are in at least one top-10 list, `lean` = both filters (26).

## Protocol

As the paper (property_prediction/scripts/train.py): LightGBM, 50 Optuna trials, 5 seeds,
fixed split; only the feature list changes. all55 = study_v1 'desc' (not refitted).
Score = mean over the 5 seeds of test R2 (AUC for the classes). Interval = 95 % paired
bootstrap (2,000 resamples of the test structures) of the difference, on the seed-averaged
predictions.

```bash
cd property_prediction
~/miniconda3/envs/pydemi/bin/python redundancy/scripts/redundancy.py
OMP_NUM_THREADS=1 WORKERS=8 THREADS=2 nice -n 10 ~/miniconda3/envs/pydemi/bin/python redundancy/scripts/train_sets.py  # ~70 min, 16 cores
~/miniconda3/envs/pydemi/bin/python redundancy/scripts/compare.py
```

## Result

Dropping the 18 redundant descriptors, or the 21 unimportant ones, or both (55 -> 26),
changes the test score by less than the noise for every property except the band gap
(and C44 in the 26-descriptor set); most changes are within ±0.01, several slightly
positive. Clear losses (interval excludes 0):

* band gap: -0.020 (37), -0.018 (34), -0.027 (26). It loses many small contributions
  (13–20 % of its SHAP share, no single descriptor above 1.6 %).
* C44: -0.013 with the lean set (26); -0.005 with 34 (interval just touches 0); +0.003 with 37.

The direct-gap AUC drops by 0.01–0.02, but within its wide interval (163 test structures).
Full table: `results/compare.csv` (printout from compare.py).
