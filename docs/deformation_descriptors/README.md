# Deformation-density descriptors report

`deformation_descriptors.md` is the report on `m1_def`, `m2_def`, `sigma_r2_def`,
`f_bond_def`, `f_int_def`, `f_bond_dep` and `def_polarity`, and their `paw`-extension
variants (`*_out`). It covers their definitions, formulas, implementation, physical
meaning, analytic behaviour, the PAW caveat, numerical robustness, ML predictability and a
survey over the 6,059-structure VASP dataset.

- `figures/`: 14 figures, PNG (200 dpi) and PDF (vector).
- `data/`: every table behind the figures and numbers. `analytic_reference/` holds the
  radial files of the Gaussian model atoms.
- `scripts/`: the scripts that regenerate them. The order and inputs are in section 12 of
  the report.
