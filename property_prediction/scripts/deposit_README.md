# Data for "Pydemi: A Python Package for Interpretable Charge-Density Descriptors for Materials Property Prediction"

Shubham K Maurya, MD Faiz Akhtar, Somnath Bhowmick (corresponding author, bsomnath@iitk.ac.in),
Department of Materials Science and Engineering, Indian Institute of Technology Kanpur, India.

These files hold the descriptors, property targets and test-set predictions behind the results
of the paper, and the PAW dataset of every DFT calculation. The descriptors were computed with
pydemi 0.4.0 (https://github.com/CMSLabIITK/pydemi, doi:10.5281/zenodo.24415538) with its
default settings (nearest-atom partition). The structures, charge densities and the split are
those of Kumar et al., J. Chem. Inf. Model. 66, 7113-7123 (2026), doi:10.1021/acs.jcim.6c00235;
the data released with that work are at https://github.com/CMSLabIITK/LatentMatFusion. The
charge densities themselves (about 180 GB) are available from the corresponding author on
reasonable request.

All files are CSV with a header row. Every structure is named by its formula and space-group
number (`structure`, e.g. `ErCdRh2_225`); `material_id` is the Materials Project id and `split`
is `train` (4,847), `val` (606) or `test` (606).

## Descriptors

| File | Rows | Contents |
|---|---|---|
| `descriptors_dft.csv` | 6,059 | `structure`, `material_id`, `formula`, `split` and the 54 descriptors of the VASP density |
| `descriptors_chargE3net_test605.csv` | 605 | the same columns for the density predicted by the fine-tuned ChargE3Net model, for the 605 test structures of the density model (rescaled to the valence-electron count, as in the paper) |
| `descriptors_columns.csv` | 54 | per descriptor: domain, density field used, units, range, whether it is intensive, fallback values and formula |

The full definitions are in the descriptor reference of pydemi (`docs/descriptors.pdf`) and in
Section S1 of the Supporting Information.

## Property targets

`targets.csv` (6,059 rows) holds the 19 targets, with `has_f_element` and `born_stable`
(1 if the stiffness tensor is mechanically stable, smallest eigenvalue above 1 GPa).
`targets_columns.csv` gives the units, source, task and number of values of each. Empty cells
are values that are not available or were removed by the rules in Section S3 of the
Supporting Information.

- `bulk_modulus`, `shear_modulus`, `youngs_modulus`, `energy_jcim` and `debye_jcim` are columns
  of the property table of Kumar et al. The last two are used only for Table 4.
- All other targets are from the Materials Project (database version 2026.04.13), used under
  CC BY 4.0.

## Predictions on the test set

| File | Contents |
|---|---|
| `predictions_lightgbm_test.csv` | LightGBM on the 54 descriptors, 19 targets (the 17 properties and the two Table 4 columns): `structure`, `material_id`, `target`, `true`, `pred_mean` and the five seeds `pred_seed0` to `pred_seed4` |
| `predictions_dft_vs_ml_test605.csv` | LightGBM models of the 17 properties trained without the 605 test structures of the density model, applied to their descriptors from the DFT density (`pred_dft_descriptors`) and from the predicted density (`pred_ml_descriptors`) (Fig. 5) |
| `predictions_matten_test.csv` | MatTen trained by us on the same split, for the five targets of Table 4: `structure`, `target`, `true`, `pred` |

- **Seed columns and `pred_mean`:** the test scores in the paper are the mean of the
  scores of the five seed columns. `pred_mean` is the prediction of the five models averaged
  before the lower limit of the target is applied (and on the log10 scale for the refractive
  index), so it can differ slightly from the mean of the seed columns. The bootstrap intervals
  of the Supporting Information use `pred_mean`.
- **Classification targets** (`is_metal`, `is_gap_direct`): the predictions are the
  probability of class 1.

## PAW datasets

| File | Contents |
|---|---|
| `paw_datasets.csv` | per structure: whether its `OUTCAR` exists (`outcar`), and in the order of the species, the `elements`, the `TITEL` of the PAW dataset (`paw_datasets`) and its valence charge (`zval`) |
| `paw_by_element.csv` | the single dataset and valence charge used for each of the 87 elements |

The fields come from the `OUTCAR` files of the 4,901 calculations that have one. Every element
uses the same dataset in every structure. The 1,158 calculations without an `OUTCAR` are
assumed to use the same datasets.

## Reproducing

The files are written by `property_prediction/scripts/make_deposit.py` in the pydemi
repository. The script checks each table against the results it is copied from: the test
scores of every model, the target values and the Table 4 scores of MatTen.

## License

CC BY 4.0. Please cite the paper, pydemi (doi:10.5281/zenodo.24415538) and Kumar et al. (2026).
