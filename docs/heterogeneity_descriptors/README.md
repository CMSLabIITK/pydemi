# Site-heterogeneity descriptors report

`heterogeneity_descriptors.md` is the report on `m1_site_std`, `f_bond_site_std`,
`zeta_site_std` and the within/between-element variances (`{m1,f_bond,zeta}_within_element_var`,
`{m1,f_bond,zeta}_between_element_var`). The specification's sentinel table calls the
pair `elemental_bonding_variance_within/between`. The report covers their definitions,
formulas, implementation, physical meaning, analytic behaviour, the symmetry noise floor,
numerical robustness, ML predictability and a survey over the 6,059-structure VASP
dataset.

- `figures/`: 12 figures, PNG (200 dpi) and PDF (vector).
- `data/`: every table behind the figures and numbers.
- `scripts/`: the scripts that regenerate them. The order and inputs are in section 12 of
  the report.
