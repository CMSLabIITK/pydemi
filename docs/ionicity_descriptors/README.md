# Calibrated ionicity and related descriptors report

`ionicity_descriptors.md` is the report on `grid_ionicity`, `ionicity_residual`,
`V_spread`, `rho_mid_std` and `lap_concentration` (with `lap_concentration_valence`).

`grid_ionicity` and `ionicity_residual` are not part of the current pydemi. They belonged
to the superseded PDF-spec build and were dropped in the prompt.md rebuild.
`scripts/calibrate_ionicity.py` reconstructs them from the legacy definition, outside the
library, on the current descriptor table. The report explains the status and what the
reconstruction shows.

- `figures/`: 8 figures, PNG (200 dpi) and PDF (vector).
- `data/`: every table behind the figures and numbers. It includes a copy of the legacy
  Phillips ionicity table; 34 of its rows are unverified.
- `scripts/`: the scripts that regenerate them. The order and inputs are in section 12 of
  the report.
