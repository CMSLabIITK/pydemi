# pydemi — development status

What is built, what is running, and what is left. For how to use the
library see [README.md](README.md).

**Author:** Shubham Maurya, CMS Lab, IIT Kanpur (PI: Prof. Somnath Bhowmick)
**Last updated:** 2026-09-25

---

## At a glance

| Item | State |
|---|---|
| Governing specification | `prompt.md` (professor-verified), since 2026-09-24 |
| Library | **rebuilt to prompt.md**: all 12 milestones of its §14 done, on branch `prompt-spec` |
| Descriptors | 220 by default (36 bonding, 21 structural, 8 magnetic, 23 heterogeneity, 132 compositional) + 12 in the off-by-default `paw` extension |
| Tests | 862 pass; `mypy --strict` clean on io, core, fields, operators |
| Dataset rerun | 6,059 structures, 0 errors, 105 min on 24 workers (`results/prompt_spec/`) |
| Performance | 8.4 s for a 96^3, 16-atom cell on one core; ~10 us per voxel, linear |
| Earlier (PDF-spec) version | tagged `pdf-spec-final`; code in `legacy/`, docs in `legacy/docs/` |
| ML input path | ChargE3Net fine-tuned from the MP checkpoint: running (see below) |
| Paper (`paper/main.tex`) | abstract + Numerical considerations written against the PDF-spec version; needs revising for the new descriptor set |

---

## 1. Decisions (2026-09-24)

1. prompt.md governs: layout, API, registry, invariance tests, sentinels, defaults.
2. Documented corrections, each backed by a test or a dataset figure:
   - `lap_concentration` is identically 1/2 on a periodic grid; kept, plus
     `lap_concentration_valence`.
   - Percolation: the highest spanning level (the lowest is always min rho);
     spanning by winding vectors (the face-contact test makes the level depend
     on the cell origin).
   - `"aeccar0"` reference = AECCAR0 + tabulated free-atom valence (AECCAR0 is
     the frozen core).
   - Hirshfeld charges with Z_i = ZVAL for pseudo-densities.
3. Only the descriptors of prompt.md are kept; the PDF-era extras (Tier 3,
   calibrated ionicity / Cohen modulus, strain response, persistence
   analysis, ...) are dropped.
4. PAW-aware items kept as an off-by-default extension: `rho_min_int`,
   `rho_min_int_ratio`, `n_NNM_paw`, `Q_NNM_paw`, the `*_def_out` variants.

Further implementation decisions where the spec is silent or
self-inconsistent (README §8): saddles from the Freudenthal link (the
26-neighbour shell cannot classify saddles); volume-normalized information
measures (intensivity); a geometric tie-break for equidistant atom images and
a boundary tolerance for shells (needed for exact supercell / translation
invariance with atoms on grid points).

## 2. Build milestones (prompt.md §14)

| # | Milestone | Key checks |
|---|---|---|
| 1 | I/O and data model | volume division, Fortran order, spin block after augmentation lines, non-collinear, cube units vs a known file |
| 2 | Derivatives | FFT exact on band-limited data, FD at nominal order, diagonal Laplacian 12% wrong on triclinic cells, FFT cusp ringing documented |
| 3 | Geometry pass | brute force incl. a cell where a 3x3x3 supercell fails |
| 4 | Tier-1 descriptors | Slater closed forms (m1, m2, sigma_r2, zeta = 0, lnf) |
| 5 | Invariance harness | every descriptor: 2x2x2 supercell, translation, rotation at 1e-6 |
| 6 | Site aggregation | ANOVA identity, heterogeneity meta-operator |
| 7 | Magnetic | sum_i mu_i == M_net, ferro/AFM, non-collinear vector form |
| 8 | Derived fields | ELF_D in [0,1] on real data, uniform-gas limits, Hartree vs erf(r)/r, neutral ESP = 0 |
| 9 | Deformation density | shipped free-atom tables (Z=1-96), real-space promolecule, custom / aeccar0 references, pair charge transfer |
| 10 | Structural | periodic census (1,1,3,3; Euler 0), SC Gaussians percolate at the midpoint density |
| 11 | Partitions | Becke, Hirshfeld, weights sum to 1, charge conservation, invariance per partition |
| 12 | Batch, CLI, docs | featurize_batch with recorded errors, CLI, sweep, convergence, float32, matminer, PAW extension, README |

## 3. Dataset rerun (2026-09-25)

`results/prompt_spec/run_rerun.py`: all domains + `paw`, default options,
CHGCAR only. 6,059 rows, 0 errors; 36.6 CPU-h (median 15 s, max 353 s).
Output `descriptors_6000_data_aug13.csv`; comparison with the earlier run in
`comparison_with_old.csv` (`compare_with_old.py`).

- **ZVAL fix found by the pilot.** 1,158 runs have no OUTCAR; the old
  fallback rule was wrong for 43 of 87 elements (p-block d10 counted, e.g.
  Sb 15 instead of 5; f-block semicore missed), giving reference-charge errors
  up to 80 e. Now: the rule matches 78 of 87 elements, and those runs take
  ZVAL / RCORE from a table of the dataset's 4,901 OUTCARs (one POTCAR per
  element throughout; `dataset_paw_table.json`,
  `tools/paw_table_from_outcars.py`). `zval_source` records the origin.
  |def_charge_mismatch|: median 0.06 e, p99 0.78 e, max 2.9 e (80 atoms).
- **Agreement with the earlier run.** Identical (to 1e-15) where definitions
  are unchanged: m1, m2, sigma_r2, f_core/f_bond/f_int, rho_perc_*,
  rho_min, M_abs, M_net, spin descriptors. With `derivative_backend="fd",
  fd_order=2` the rebuild reproduces the old derivative-based values to
  1e-10 (24-structure check).
- **Expected differences.** Per-volume counts (n_max, n_min, n_saddle*,
  n_NNM), volume-normalized shannon_entropy / disequilibrium, ELF now
  reconstructed for every run (the old ELF values came from the 86 ELFCARs),
  mu_site_std = 0 for non-magnetic runs (was ~1e-5 noise), deformation
  descriptors of the 1,158 no-OUTCAR runs (ZVAL fix; runs with an OUTCAR
  agree, Spearman 0.91-0.9996).
- **Finding: ellipticity is not numerically robust.** ellip_bond_avg / std
  change by 20-70% between FD2, FD4 and FFT on the dataset grids (Spearman
  FD2 vs FFT 0.87 / 0.50), while zeta, fisher_information, charge_FA converge
  (FD4 vs FFT 1e-4 to 6e-3). Needs a decision before these two are used or
  reported.
- **Finding: the critical-point census rarely closes.** euler_consistency = 0
  in 9.1% of structures (|.| <= 4 in 20%): on these pseudo-density grids the
  counts n_max, n_min, n_saddle* are not topologically consistent.

## 4. Running

- **ChargE3Net fine-tuning** (`~/charge3net/charge3net`,
  `scripts/run_finetune.sh`): started 2026-09-24 22:48 from the MP checkpoint,
  50k steps (~16.5 h). Validation NMAPE 1.29% after ~5,200 steps, against
  1.21% best for the from-scratch run after 200k steps.

## 5. Remaining

### 5.1 Library
- [x] Rerun the 6,059-structure dataset with the new pydemi (section 3).
- [ ] Decide on ellip_bond_avg / std (not robust to the derivative scheme)
      and on reporting the census counts given euler_consistency.
- [ ] Merge the `prompt-spec` branch into `main` when approved (not pushed).
- [ ] Optional: a new code guide for the rebuilt package (the old one is in
      `legacy/docs/`).
- [ ] Optional: speed up the second-order Voronoi pair regions (2.3 s of 8.4 s
      on a symmetric 96^3 cell).

### 5.2 ML input path
- [ ] Evaluate the fine-tuned model on the 605 test structures; compare with
      the MP model as-is and the from-scratch model.
- [ ] CIF-only inference (grid from the structure, not an existing CHGCAR).
- [ ] pydemi reader for predicted densities (charge renormalization, source tag).
- [ ] Descriptor-level DFT-vs-ML comparison on the test set.

### 5.3 Paper
- [ ] Revise the numerical-considerations section and abstract for the new
      descriptor set, defaults (FFT) and corrections.
- [ ] Write the remaining sections (framework design, descriptor families,
      validation, application, availability).

### 5.4 Decisions for the user / PI
- [ ] Verify the DOIs in `paper/references.bib`.
- [ ] Confirm the MIT licence (no LICENSE file yet).
- [ ] Journal choice (Computer Physics Communications or Computational
      Materials Science recommended for the current scope).
