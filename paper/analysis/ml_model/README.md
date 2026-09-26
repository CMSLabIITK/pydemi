# ChargE3Net side of the ML input path

Copies of the two scripts added to the ChargE3Net checkout
(`~/charge3net/charge3net/scripts/`, public code of Koker et al., npj Comput.
Mater. 10, 161 (2024), MIT licence), kept here so that the paper's structure-only
path can be reproduced. They run in the ChargE3Net environment with pydemi on the
path; pydemi itself does not depend on PyTorch.

    scripts/predict_from_cif.sh OUT_DIR structure1.cif structure2.cif ...

1. For each structure, the grid is `pydemi.vasp_grid_shape(cell, encut=500)`, the grid
   VASP would use (it reproduces all 6,059 dataset grids).
2. ChargE3Net's unchanged full-grid test pipeline predicts the density at every grid point.
3. Each prediction is written with `pydemi.write_predicted` as `OUT_DIR/<name>.npz`.

Then `pydemi.read_predicted(path, zval=..., paw_radii=...)` and `pydemi.featurize`.

Validation (`../p_cif_path.py`, inputs in `../out/ml/cif_path/cif/`): the paper's
Section 7 and SI S6.
