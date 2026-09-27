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
| Library | **rebuilt to prompt.md**: all 12 milestones of its §14 done; merged into `main` 2026-09-25; pushed to GitHub (45eba79, bd0438c) |
| Descriptors | 220 by default (36 bonding, 21 structural, 8 magnetic, 23 heterogeneity, 132 compositional) + 12 in the off-by-default `paw` extension |
| Tests | 894 pass on Python 3.10 and 3.11 (zeta_ELF tagged fragile 2026-09-27: 5 fragile descriptors); `mypy --strict` clean on the whole package; CI in `.github/workflows/tests.yml` |
| Dataset rerun | 6,059 structures, 0 errors, 105 min on 24 workers (`results/prompt_spec/`) |
| Performance | 8.4 s for a 96^3, 16-atom cell on one core; ~10 us per voxel, linear |
| Earlier (PDF-spec) version | tagged `pdf-spec-final`; code in `legacy/`, docs in `legacy/docs/` |
| ML input path | ChargE3Net from scratch and fine-tuned from the MP checkpoint, both evaluated on the 605 test structures (2026-09-26): fine-tuned NMAPE 0.72% (scratch 0.96%); 68 of 85 descriptors keep Spearman >= 0.95 (`paper/analysis/o_ml_analysis.py`) |
| Paper (`paper/main.tex`) | all sections written (2026-09-26); open: acknowledgements, data availability, compilation check; SI S1-S6 written |

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
  change by a median 22% / 67% between FFT and FD4 and 21% / 62% on an 80%
  grid (48 structures), while zeta, fisher_information, charge_FA converge
  (FD4 vs FFT 1e-4 to 6e-3).
- **Finding: the critical-point census rarely closes.** euler_consistency = 0
  in 9.1% of structures. It equals the number of extrema the 26- and
  14-neighbour stencils classify differently (identity checked on 200
  structures): a median 5% of all critical points, set by PAW-pseudized
  regions and ripple, not by coarse grids.
- **Decision (2026-09-25, E2 + E3 + C2 of the decision note).**
  `ellip_bond_avg`, `ellip_bond_std`, `n_saddle1`, `n_saddle2` are tagged
  `stability="fragile"` (still in the default output;
  `descriptor_names(include_fragile=False)` gives the model-ready set of 216);
  the off-by-default `robust` extension adds `ellip_bond_bounded_avg` =
  mean(1 - lambda2/lambda1), converged to ~1% (added to the dataset table by
  `add_robust_ellipticity.py`); `euler_consistency` is documented by its
  identity. `Q_NNM` and `rho_perc_*` are the topological descriptors of record.
- **Code-review fixes (same day).** Derivatives of derived fields and site
  statistics were cached without every option they depend on (reusing one
  object with another `laplacian_method` or `fd_order` gave stale values);
  `rho_min_int` used c2 as a length with radius-scaled shells; `featurize`
  hard-coded `fd_order=4`. Fixed, with regression tests that fail on the old
  code (`tests/test_regressions.py`). The dataset rerun is unaffected (one
  `featurize` call per fresh object, absolute shells).

## 4. Running

Nothing. The ChargE3Net fine-tuning (50k steps from the MP checkpoint, best
validation NMAPE 0.83%) and its tests finished 2026-09-26 00:32; the
descriptor-level comparison is in `paper/analysis/out/ml_summary.txt`.

## 5. Remaining

### 5.1 Library
- [x] Rerun the 6,059-structure dataset with the new pydemi (section 3).
- [x] Decide on ellip_bond_avg / std and the census counts (section 3: E2 + E3 + C2).
- [x] Merge the `prompt-spec` branch into `main` (2026-09-25; local, not pushed).
- [x] Push `main` to GitHub (2026-09-25, on request; ask before every push).
- [x] Code guide for the rebuilt package: `docs/CODE_GUIDE.md` (the old one
      is `legacy/docs/CODE_GUIDE_pdf_spec.md`).
- [ ] Optional: speed up the second-order Voronoi pair regions (2.3 s of 8.4 s
      on a symmetric 96^3 cell).

### 5.2 ML input path
- [x] Evaluate the fine-tuned model on the 605 test structures; compare with
      the MP model as-is and the from-scratch model (2026-09-26).
- [x] CIF-only inference (2026-09-26): `pydemi.vasp_grid_shape` (reproduces all 6,059 grids) and
      `~/charge3net/charge3net/scripts/predict_from_cif.sh` (copy in `paper/analysis/ml_model/`);
      8 test structures from CIF agree with the DFT-grid predictions (median descriptor difference 9e-6).
- [x] pydemi reader for predicted densities: `read_predicted` / `write_predicted` (`.npz`), charge
      renormalization, `density_origin` / `density_model` / `charge_scale` metadata; 18 tests.
- [x] Descriptor-level DFT-vs-ML comparison on the test set
      (`paper/analysis/n_ml_descriptors.py`, `o_ml_analysis.py`; main-text
      Section 7, Fig. 8, SI S6; the docs reports' ML sections updated).

### 5.3 Paper
- [x] Revise the numerical-considerations section and abstract for the new
      build (2026-09-25): every number recomputed with the current code by
      `paper/analysis/` (new scripts a-i; the first draft's are in `legacy/`);
      new subsections on the critical-point census (replacing persistence),
      valence counts, stability tags and the robust ellipticity; outline notes
      of the other sections updated. Template untouched.
- [x] Framework design, descriptor families, validation, application,
      ML input path (Section 7, Fig. 8), software availability (2026-09-26).
- [x] Introduction and Conclusions (2026-09-26); consistency pass (specification defined,
      architecture caption, uncited VASP/PBE/ELF/software references added).
- [x] Data availability statement (dataset is the CMS Lab's own; densities on request).
- [ ] Confirm "upon reasonable request" (or a public deposit) and archive a release (DOI).
- [x] SI skeleton `paper/SI.tex` (same preamble as main.tex): S1 catalogue
      (generated by `analysis/k_si_tables.py`), S3 the six descriptor-family
      reports condensed with 18 figures (`paper/figures/si/`), S5 dataset and
      DFT settings (parsed from the OUTCARs by `analysis/j_dft_settings.py`).
      S2 holds the full numerics section moved from the main text; S4
      validation details; S6 ML details and the per-descriptor table. Not compiled (no LaTeX locally).
- [x] Main-text numerics condensed (459 -> ~200 lines) with new figures:
      stability map, PAW pitfalls; architecture and dataset figures added;
      table of documented corrections (`analysis/l_main_figures.py`).
- Target journal: most likely Computational Materials Science (else CPC).

### 5.4 Decisions for the user / PI
- [x] Verify the DOIs in `paper/references.bib` (all 34 entries checked against Crossref, 2026-09-26).
- [x] Acknowledgements (text from the user, 2026-09-26).
- [ ] Compile main.tex and SI.tex (no LaTeX here) and fix what the log reports.
- [ ] `spglib` is listed in `pyproject.toml [full]` but not used by the code: drop it or use it.
- [x] MIT licence: `LICENSE` added (2026-09-25).
- [ ] Journal choice (Computer Physics Communications or Computational
      Materials Science recommended for the current scope).
