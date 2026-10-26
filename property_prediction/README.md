# Property prediction from pydemi descriptors

Can the 56 charge-density descriptors predict ground-state properties as well as models
that read the whole charge-density grid (the JCIM 2026 study), at a fraction of the cost?
And can we predict more than the five properties of that study?

Dataset: the CMS Lab set of 6,059 VASP calculations (`/data/sai/new_charge/6000_data_aug13`,
read only), with the train / val / test split of the earlier study (4,847 / 606 / 606).

```
scripts/   how every file in data/ was made
data/      inputs and computed tables (not in git: regenerate with scripts/)
results/   models, metrics and figures (not in git)
```

## Steps

1. **Descriptors** (done 2026-10-19): `scripts/run_descriptors.sh` computes the 56 pydemi
   descriptors of every CHGCAR (16 workers, 37 min, 0 errors) into
   `data/descriptors_56.csv`; `data/provenance.txt` records versions and settings.
2. **Targets**: `data/properties.csv` holds the dataset's property table
   (bulk, shear, Young's moduli, Poisson ratio, Debye temperature, and the energy per atom
   relative to free atoms, i.e. the *cohesive* energy, labelled "formation energy" there).
   `scripts/fetch_mp.py` adds Materials Project data for the same `mp-` ids (needs an API key
   in `~/.config/.pmgrc.yaml`; not run yet).
3. **Training tables** (done 2026-10-19): `scripts/build_tables.py` cleans and joins
   everything (rules and counts in `data/build_report.txt`):
   - `data/descriptors.csv`: 6,059 materials x the 56 descriptors (+ ids, split);
     units and formulas in `data/descriptors_columns.csv`.
   - `data/targets.csv`: 6,059 materials x 23 targets (+ ids, split, flags `has_f_element`,
     `born_stable`); units, source, task and counts in `data/targets_columns.csv`.
   - `data/dataset.csv`: the two joined, ready for training.
4. **Composition features** (done 2026-10-20): `scripts/features_composition.py` writes
   `data/composition.csv`, the matminer Magpie element statistics (132) + stoichiometry
   p-norms (6) of each formula, for the composition-only baseline.
5. **Models** (running since 2026-10-20): `scripts/train.py`, one job per target x feature set
   (comp 138 / desc 55 / both 193; `lap_concentration` dropped as constant) with ridge, random
   forest (500 trees) and LightGBM (50 Optuna TPE trials on val, early stopping on val);
   RF and LightGBM refit with 5 seeds, test split scored only at the end. Two extra targets
   `energy_jcim` / `debye_jcim` are the properties.csv columns the JCIM table used (its
   "formation energy" differs from MP's cohesive energy, mean |diff| 0.47 eV/atom).
   Results in `results/study_v1/` (jobs/*.json, models/, train.log);
   `scripts/summarize.py` makes summary.csv, best.csv, jcim.csv, shap.csv.
   `refractive_index` is modelled on log10 (it is sqrt(ε_∞); one validation material, NbAs,
   has n = 216 and drove tuning and selection on the linear scale): first-run jobs kept in
   `results/study_v1/superseded/`.
6. **GUI bundle** (2026-10-20): `scripts/magpie_table.py` writes `data/magpie_table.json`, the
   element table `pydemi.props.composition_features` uses (matches matminer to 3e-16 on all
   6,059 materials); `scripts/build_bundle.py` writes `results/study_v1/properties_v1.json.xz`
   and copies it into the package (`src/pydemi/properties_v1.json.xz`, 35 MB): per target the
   feature set with the lowest validation loss, seed-0 LightGBM, with the per-node training
   statistics dropped from the model text (identical predictions, checked by the script).
   22 targets (energy_above_hull left out: R² < 0.1). pydemi and the GUI use it by default.
8. **Penn-model features for the dielectric targets** (2026-10-23, not adopted):
   `scripts/penn_features.py` writes `data/penn_features.csv` (valence plasma energy from the
   electron count and volume, the v1 band-gap model's prediction, out-of-fold on train, and
   1 + (hw_p/E_g)^2). Retrained with
   `FEATURE_SETS=penn,descpenn ONLY=dielectric_electronic,refractive_index,dielectric_total,dielectric_ionic OUT=results/study_v1_penn python scripts/train.py`
   (results in `results/study_v1_penn/`). Validation loss improved by at most 2 %, and every
   test-R² change lies inside its bootstrap 95 % interval over the 123 test materials, so the
   v1 models are kept.
7. **Preview models** (superseded by 6): `scripts/train_preview.py` (descriptors only, untuned).

Cleaning rules (see build_tables.py):
- K, G, E as published in properties.csv; Poisson ratio >= 0.5 removed.
- MP elastic data kept only where K and G are 0-1000 GPa and equal properties.csv; C11, C12,
  C44 and the anisotropy index only where the stiffness tensor is Born-stable (smallest
  eigenvalue > 1 GPa); A_U is computed from the unrounded tensor.
- is_gap_direct only for materials with a gap; dielectric constants are heavy-tailed (log10).
- cohesive_energy is MP's; it is less reliable for lanthanides and actinides (has_f_element).

Join key: `formula_sp` in properties.csv = the dataset folder name (6,059 one-to-one).

Notes from the data:
- `lap_concentration` is 0.5 for every structure (an identity on a periodic grid): no information.
- `n_NNM` / `Q_NNM` are non-zero in 37 % of structures, likely PAW pseudo-density peaks just
  outside the fixed 0.8 Å cutoff, not true non-nuclear maxima.
- 421 structures have no element with two or more sites: their within-element variances are NaN.
