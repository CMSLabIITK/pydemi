# Changelog

## 0.1.0 (2026-09-27)

First release: the implementation described in the accompanying paper
(`paper/main.tex`).

- **Data model and readers.** One `VolumetricData` object with cached derived
  fields.
  - VASP readers: CHGCAR / CHG (1, 2 or 4 blocks), AECCAR0 + AECCAR2, ELFCAR
    and LOCPOT, with ZVAL and RCORE from the POTCAR or OUTCAR, or from
    per-element tables.
  - Gaussian cube and XSF readers.
- **ML input path.**
  - `vasp_grid_shape` reproduces VASP's CHGCAR grid from the lattice and
    ENCUT alone.
  - `write_predicted` / `read_predicted` exchange densities predicted by a
    model through an `.npz` file, with charge renormalization. The density's
    origin, model and scale factor are recorded in the metadata.
- **Descriptors.** 220 by default: 36 bonding, 21 structural, 8 magnetic,
  23 heterogeneity and 132 compositional (Magpie via matminer).
  - Off-by-default extensions: `paw` (12 descriptors outside the augmentation
    spheres) and `robust` (a bounded ellipticity).
  - Every descriptor is registered with its formula, units, range, sentinel
    cases, references and stability tag.
- **Numerics.**
  - FFT and finite-difference derivatives with the exact metric.
  - A periodic geometry pass with a geometric tie-break.
  - Four partitions: nearest atom, power diagram, Becke and Hirshfeld.
  - Tabulated LDA free atoms (Z = 1–96) for the promolecule.
  - The critical-point census on the Freudenthal triangulation, with an exact
    `euler_consistency`.
- **Interfaces.**
  - Python: `featurize`, `featurize_batch`, `catalogue`, `descriptor_names`,
    `hirshfeld_charges` and `sensitivity_sweep`.
  - Command line: `pydemi featurize`, `batch`, `catalogue` and `sweep`.
- **Validation.** 894 tests, including an invariance harness that checks
  every registered descriptor under supercell expansion, translation and
  rotation. `mypy --strict` passes on the whole package; CI runs the tests
  on Python 3.10, 3.11 and 3.12.
