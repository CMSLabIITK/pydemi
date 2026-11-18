# Property models without the strongly correlated descriptors

Tables S5 and S6 of the paper: which descriptors correlate strongly with another one, and
how accurate the 17 property models are without them.

```bash
cd property_prediction
python redundancy/scripts/redundancy.py    # groups -> redundancy/results/groups.csv, sets.json
python redundancy/scripts/train_sets.py    # retrain on the 37 remaining descriptors
python redundancy/scripts/compare.py       # change in test score -> redundancy/results/compare.csv
```

- **Groups** (training split only): descriptors with |Spearman ρ| ≥ 0.90 are grouped by
  complete linkage on 1 − |ρ|, so that every pair in a group has |ρ| ≥ 0.90. Each group
  keeps the descriptor with the largest mean SHAP share over the 17 properties; the other
  18 descriptors, in 11 groups, are dropped, which leaves 37.
- **Training**: the protocol of `scripts/train.py` (LightGBM, 50 Optuna trials, five seeds,
  the same split); only the list of descriptors changes. The 55-descriptor models are those
  of `results/study_v1`.
- **Comparison**: the test score is the mean over the five seeds of R² (ROC-AUC for the
  classifications); the 95 % interval of the change is a paired bootstrap (2,000 resamples
  of the test structures) on the seed-averaged predictions.

`results/` is not in git.
