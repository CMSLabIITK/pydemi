# pydemi — development status

What has been built so far, what is running now, and what is left.
This file is about the project itself. For how to use the library, see
[README.md](README.md). For a walk-through of the code, see
[docs/CODE_GUIDE.md](docs/CODE_GUIDE.md).

**Author:** Shubham Maurya, CMS Lab, IIT Kanpur (PI: Prof. Somnath Bhowmick)
**Last updated:** 2026-09-24

---

## At a glance

| Item | State |
|---|---|
| Library (entries 1–107 of the Consolidated Descriptor Reference) | **complete**: all 7 build phases done |
| Registry | 257 entries; 222 default scalar quantities in 17 families, plus 28 opt-in Becke variants; 209 from a CHGCAR alone |
| Code | about 5,500 lines in `src/pydemi`, 2,200 lines of tests |
| Tests | 166 pass, 1 skipped (the optional matminer comparison) |
| Documentation | README (full user docs), CODE_GUIDE (code walk-through), numerics.md |
| 6,059-structure dataset | computed with 0 failures; G and A rerun with the latest fixes and merged (224 columns) |
| Paper (`paper/main.tex`) | abstract and "Numerical considerations" written; other sections are outlines |
| ML input path (CIF → CHGCAR) | ChargE3Net trained on the dataset (test NMAPE median 0.79%, max 10.4%); **not yet integrated in pydemi**, no CIF-only inference yet |

---

## 1. Completed work

### 1.1 Build phases (from the descriptor PDF)

The 107 descriptor entries were split into seven phases. Each phase ended with
its tests passing before the next one began.

| Phase | Scope | Result |
|---|---|---|
| 0 Foundation | VASP readers (CHGCAR with all spin blocks and noncollinear data, AECCAR, ELFCAR, LOCPOT); `Grid` with periodic derivatives through the metric tensor (`fd` and `spectral`), Hessian and eigenvalues; nearest-atom geometry pass (KD-tree with a provably complete image search); shells; analytic Slater and Gaussian test densities | foundation for every family |
| 1 Tiers 1–3 (entries 1–39) | density moments, shell fractions, Laplacian, gradient and ζ descriptors; Magpie compositional baseline; cross terms and transforms; both readings for entries the PDF leaves ambiguous | Tier 2 matches matminer to 1e-14 |
| 2 ρ-only fields (F1–F6, I1) | ELF_D reconstruction and Family B; Hartree potential (F2); NCI (F3); whole-grid ellipticity (F4); local energy densities (F5); information measures (F6); charge-anisotropy tensor (I1) | 77 tests |
| 3 Spin, sites, partitions | spin-density Family E; site-resolved Family C with an exact within/between variance decomposition; power-diagram partition; Becke partition | Becke made **opt-in**: in periodic solids its weights converge only algebraically; 96 tests |
| 4 Topology and bonds | percolation thresholds (union-find with winding vectors); piecewise-linear Morse census on the Freudenthal triangulation (Euler sum exactly 0); non-nuclear maxima; density floor; bond-midpoint density (I2) | 111 tests |
| 5 Reference-dependent | own spherical LDA atom solver (PW92 and VWN5, validated against NIST); Hirshfeld partition (the default smooth partition); deformation Family A; Family H; entry 73 | 135 tests |
| 6 Calibration and tooling | Phillips ionicity table (67 rows); sigmoid ionicity fit; Cohen bulk modulus (class λ, constant 1971); ρ-based B₀ proxy; strain response (107); batch runner with resume; `pydemi` CLI; Fourier resampling; grid-convergence report; cube and XSF I/O | 161 tests |

### 1.2 Work after the build

- **Dataset run.** `pydemi compute` on `/data/sai/new_charge/6000_data_aug13`
  processed 6,059 structures with 0 failures. Output:
  `results/descriptors_6000_data_aug13.csv`; the first version is kept as
  `.v1.csv`.
- **Topology fixes (first round).**
  - Per-element non-nuclear cutoff max(c₁, R_PAW), with R_PAW read from the
    POTCAR or OUTCAR RCORE. This removes the spurious maxima on the
    pseudized lobes of Si and similar atoms.
  - 0-dimensional persistence from a maximum spanning forest of the basin
    graph, giving `n_NNM_persistent` and `Q_NNM_persistent`. This removes
    ripple maxima.
  - `rho_min_int` / `rho_min_int_ratio`: the density floor outside the cores,
    because the PAW pseudo-density is negative near many nuclei.
  - Family G was rerun on the whole dataset and merged (`topology_G.csv`).
- **Latest additions (this round).**
  - *Charge filter:* a persistent non-nuclear maximum counts only if its
    merged basin holds ≥ 0.01 e. Relative persistence alone let through up to
    48 tiny peaks holding < 1 e in Yb compounds.
  - *Outside-PAW deformation variant `*_def_out`:* Δρ = CHGCAR − valence
    promolecule, evaluated only outside the PAW augmentation spheres. There
    the CHGCAR is not pseudized, so Family A gives usable numbers without
    AECCARs. Adds eight `*_def_out` quantities plus `def_out_volume_fraction`
    and `def_out_radii_from_paw`.
  - `Engine.augmentation_radii()` is now the single place that decides R_PAW
    (RCORE, or the covalent radius as a flagged fallback).
  - Bug fix: `resample_engine` now carries the PAW radii over.
  - Leftover mentions of the old package removed; the code has no references
    to it.
- **Documentation.**
  - `README.md`: installation, concepts, every descriptor table (generated
    from the registry), PAW notes, calibration, CLI, validation, performance
    and limitations.
  - `docs/CODE_GUIDE.md`: every module and function explained.
  - `docs/numerics.md`: numerical findings on real data.
- **Paper.**
  - `paper/main.tex`: the fixed template; abstract draft; section outline; full
    **Numerical considerations** section with four tables (derivative
    schemes, partition convergence, partition sensitivity, FeNi₃ PAW shells);
    bibliography file.
  - `paper/analysis/`: scripts a–g that regenerate every number in that
    section from the dataset, with fixed random seeds.

### 1.3 Findings behind the design decisions

These came from real VASP data, not synthetic tests, and are used in the paper:

- Stencil choice changes voxel-count Laplacian descriptors by up to about 32%
  for individual structures. Charge-weighted versions (`lnf_rho`) change by at
  most about 10%, so they are recommended.
- 61% of the densities are negative somewhere (PAW pseudo-density). Hence
  `rho_min_int`.
- 87% of the whole-cell deformation density ∫|Δρ| lies inside the PAW
  spheres. Hence whole-cell Family A is AECCAR-only, and `*_def_out` was
  added.
- Becke weights on FeNi₃: error 5.6e-2 with 20 neighbours, 1.0e-3 with 300.
  Hirshfeld: 3.0e-6 at a 6.5 Å cutoff. Hence Hirshfeld is the default.
- Grid convergence: at 80% of the grid points, a median of 20% of the
  reported quantities change by more than 2%. The critical-point counts and
  ellipticity are the most sensitive.

---

## 2. In progress

Nothing is running. The G + A rerun finished (6,059 structures, 0 failures)
and was merged into the main CSV; the previous version is kept as `.v2.csv`,
and all 182 columns outside G and A are bit-for-bit unchanged.

Rerun results: persistent non-nuclear maxima in 90 structures (93 before the
charge filter, which changed only four structures); outside-PAW voxels are 53%
of the cell (median; 21–76%) and `def_polarity_out` is 2.3% (median;
0.8–5.9%). 1,158 structures have no POTCAR/OUTCAR, so their PAW radii fall
back to covalent radii (`def_out_radii_from_paw = 0`).

---

## 3. Remaining work

### 3.1 Right after the rerun (done)

- [x] Merge `rerun_GA.csv` into the main results and verify it.
- [x] Paper, persistence paragraph: give the new count of structures with
      persistent non-nuclear maxima (93 before the charge filter), mention the
      0.01 e requirement, and remove the `% TODO` about the Yb compounds.
- [x] Paper, deformation paragraph: add the outside-PAW variant with dataset
      statistics (distributions of `def_polarity_out` and
      `def_out_volume_fraction`).
- [x] Abstract: change the descriptor count from 212 to 222.
- [x] README / CODE_GUIDE: update any numbers that move after the merge.

### 3.2 Paper sections still to write

| Section | Status | Notes |
|---|---|---|
| Introduction | outline | |
| Framework design | outline | input paths, grid engine, registry, free-atom references; needs an architecture figure |
| Descriptor families | outline | table of the 17 families; include the ellipticity finding (the mean is about 11× the median; mean > 3× median in 98.6% of structures) |
| Numerical considerations | **written** | updated with the rerun results |
| Validation | outline | a table of checks and achieved accuracy (the material is in README §12) |
| Application to the dataset | outline | composition space, throughput, family coverage, distributions; property prediction optional |
| Machine-learning input path | placeholder | waits for the CIF → CHGCAR model |
| Software availability | outline | needs the repository URL |
| Conclusions, acknowledgements | empty | |

### 3.3 Machine-learning input path (the "dual-input" part)

Current model: the public ChargE3Net code (Koker et al., *npj Comput. Mater.*
10, 161 (2024), which must be cited), trained from scratch on this dataset in
`~/charge3net/charge3net`, with a 5,152 / 302 / 605 split and 200k steps.
Validation error was still falling when training stopped. The model predicts
total density only (no spin). Predicted charge is off by 0.1% at the median and
up to 3.4%. A first pydemi check gives DFT and ML descriptors within about 1%
for typical structures and several-fold off for YbPrZn₂.

- [ ] Baselines: the bundled MP model without retraining, and fine-tuning from
      it, besides the from-scratch run; train until converged.
- [ ] CIF-only inference: build the grid from the structure (same spacing as
      the training data) instead of from an existing CHGCAR.

- [ ] Integrate the CIF → CHGCAR model once it is trained. Most likely this is
      a reader or engine constructor that takes a predicted density, plus
      the metadata the model can supply (ZVAL, PAW radii or a fallback).
- [ ] Compare descriptors from DFT and predicted densities on the same
      structures, family by family. This is for the paper's ML section.
- [ ] Check which descriptors a predicted density can support. The PAW-specific
      definitions (R_PAW cutoffs, `*_def_out`) assume a VASP-like
      pseudo-density.

Until this exists, the paper keeps those passages marked `% TODO`. No ML
results are written as if they exist.

### 3.4 Library items that are optional or open

- [ ] Family D calibration: fitting needs CHGCARs of the reference compounds
      (Phillips-table compounds, bulk moduli). The table ships with the
      library; the densities have to be computed.
- [ ] Native readers for Quantum ESPRESSO HDF5 and ABINIT binary densities.
      Only cube and XSF are read now, via `pp.x` / `cut3d`.
- [ ] Possible extensions from README §14: a relativistic or spin-polarized
      atom solver; better treatment of small cells in the site statistics.

### 3.5 Decisions and checks that need you

- [ ] **Verify the 34 "recalled" Phillips ionicity values** in
      `src/pydemi/data/phillips_ionicity.csv` against a primary source. The
      other 33 were checked against secondary tables.
- [ ] **Verify the DOIs** in `paper/references.bib`.
- [ ] **Confirm the license.** `pyproject.toml` says MIT, but there is no
      `LICENSE` file yet.
- [ ] **Choose the journal.** This sets the length; the abstract, at about 290
      words, is long for most journals.
- [ ] **Property prediction:** decide whether elastic-modulus (or other)
      prediction results belong in this paper.
- [ ] **Repository:** decide where the code will be hosted (needed for the
      Software availability section). The project is not under git yet.
- [ ] Acknowledgements text (funding, computing resources).

---

## 4. Where things are

| Path | Contents |
|---|---|
| `src/pydemi/` | the library (engine, grid, geometry, partitions, readers, atom solver, descriptors, calibration, CLI) |
| `tests/` | 166 tests: analytic, identities, numerics, units, reference data, I/O |
| `README.md` | user documentation |
| `docs/CODE_GUIDE.md` | detailed code reference |
| `docs/numerics.md` | numerical findings on real data |
| `paper/main.tex`, `paper/references.bib` | manuscript |
| `paper/analysis/` | scripts that regenerate the paper's numbers (`out/*.json`) |
| `results/descriptors_6000_data_aug13.csv` | main descriptor table (`.v1.csv` = first run, `.v2.csv` = before the G + A rerun) |
| `results/topology_G.csv`, `results/rerun_GA.csv` | family reruns |
| `results/*.log` | run logs |

The dataset at `/data/sai/new_charge/6000_data_aug13` belongs to another user
and is only read, never written to.
