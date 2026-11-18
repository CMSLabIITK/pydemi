# Property models of the pydemi paper

The scripts that train and evaluate the property models of the paper: 17 properties
predicted from the 55 pydemi descriptors, on the 6,059 VASP charge densities of
Kumar et al., J. Chem. Inf. Model. 66, 7113 (2026), with the training, validation and test
split distributed with that dataset (4,847 / 606 / 606).

```
scripts/      every step below
redundancy/   the models without the strongly correlated descriptors
data/         inputs and tables (not in git; made by the scripts)
results/      models and scores (not in git; made by the scripts)
```

## Inputs

- The `CHGCAR` files of the dataset, one folder per structure.
- `data/properties.csv`: the property table distributed with the dataset (material ids,
  split, the moduli, and the energy and Debye-temperature columns used in Table 4).
- A Materials Project API key for `fetch_mp.py`, read by the MP client from
  `~/.config/.pmgrc.yaml` or the `MP_API_KEY` environment variable.
- For `train_mlheldout.py`: `data/descriptors_chargE3net_test605.csv`, the descriptors of the
  ChargE3Net-predicted densities of the 605 test structures of the density model (in the
  data deposit of the paper).

## Steps

```bash
cd property_prediction
scripts/run_descriptors.sh /path/to/dataset   # 1. descriptors -> data/descriptors_batch.csv
python scripts/fetch_mp.py                    # 2. Materials Project targets -> data/mp_*.csv
python scripts/build_tables.py                # 3. cleaned targets + descriptors -> data/dataset.csv
python scripts/train.py                       # 4. ridge, random forest, LightGBM -> results/study_v1/
python scripts/summarize.py                   # 5. scores and SHAP shares -> results/study_v1/*.csv
python scripts/train_top10.py                 # 6. models on the ten most important descriptors
python redundancy/scripts/redundancy.py       # 7. correlated descriptors (training split)
python redundancy/scripts/train_sets.py       #    retrain without them
python redundancy/scripts/compare.py          #    change in test score, bootstrap intervals
python scripts/train_mlheldout.py             # 8. property models on DFT vs predicted densities
python scripts/build_bundle.py                # 9. the models shipped with pydemi
```

1. **Descriptors.** `run_descriptors.sh` runs `pydemi batch` over every `CHGCAR` with 16
   worker processes; `data/provenance.txt` records the versions.
2. **Targets.** `fetch_mp.py` downloads the Materials Project data (database version
   2026.04.13) for the material ids of `properties.csv`.
3. **Training tables.** `build_tables.py` cleans and joins the targets and descriptors; the
   rules and the counts they remove are written to `data/build_report.txt`:
   - K, G and E are those of `properties.csv`.
   - The Materials Project elastic data are kept only where K and G lie between 0 and
     1000 GPa and equal those of `properties.csv` to 0.001 GPa (6,000 materials); C11, C12
     and C44 only where the stiffness tensor is mechanically stable, its smallest eigenvalue
     above 1 GPa (5,791).
   - The direct-gap label is defined only for materials with a gap (1,836).
4. **Models.** `train.py` trains, for each property, a ridge (logistic) model, a random
   forest (500 trees) and LightGBM (50 Optuna trials on the validation set, early stopping);
   the random forest and LightGBM are refitted with five seeds, and the test set is scored
   only at the end. The refractive index is modelled on a log10 scale. Two extra targets,
   `energy_jcim` and `debye_jcim`, are the columns of `properties.csv` used for Table 4.
5. **Summary.** `summarize.py` writes the test scores of every model, the best model per
   property, the five targets of Table 4 and the SHAP shares of the descriptors.
6. **Ten descriptors.** `train_top10.py` ranks the descriptors of each property by their
   mean |SHAP| in the LightGBM models on the training set and retrains on the ten highest.
7. **Correlated descriptors.** See `redundancy/README.md`.
8. **Predicted densities.** `train_mlheldout.py` makes the 605 test structures of the
   density model the test set, retrains the 17 models without them and applies them to the
   descriptors of the DFT and of the ChargE3Net-predicted densities.
9. **Shipped models.** `build_bundle.py` writes the seed-0 LightGBM model of each property,
   with its test scores, into `src/pydemi/properties_v2.json.xz`, which `pydemi.props` and
   the graphical interface read.
