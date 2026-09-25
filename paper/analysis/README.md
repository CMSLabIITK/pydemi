# Analyses behind the paper's numerical-considerations section

Run from this directory with pydemi installed; outputs go to `out/`.
Dataset: `/data/sai/new_charge/6000_data_aug13` (VASP, 6,059 structures),
read as in the dataset rerun (`../../results/prompt_spec/`): CHGCAR, with
ZVAL / RCORE from the run's OUTCAR or, for the 1,158 runs without one, from
the dataset's PAW table. `common.py` fixes the random samples (seeds 2026,
7, 11). Workers: `ANALYSIS_WORKERS` (default 16), one thread each.

| Script | Produces | Used in |
|---|---|---|
| `a_derivatives.py 200` | `derivatives.csv`, `derivatives_summary.csv` | Table `tab:derivatives`; integrated Laplacian; ellipticity |
| `g_isolated_atom.py` | `isolated_atom.json` | FFT round-off in near-empty space |
| `b_partitions.py 300` | `partitions_becke.json`, `partitions.csv`, `partitions_summary.csv` | Tables `tab:partitions`, `tab:partition-sensitivity` |
| `c_paw.py` | `paw.json` | negative densities, CaSi3Pt, Table `tab:feni3`, outside-sphere statistics |
| `d_deformation.py 150` | `deformation.json` | |delta rho| inside the PAW spheres; Hirshfeld charges in binaries |
| `h_topology.py 200` | `topology.json` | critical-point census identity and statistics |
| `e_convergence.py 30` | `convergence.csv` | grid-convergence statistics |
| `i_ellipticity_variants.py` | `ellipticity_variants.csv` | robust ellipticity statistics (decision note) |
| `f_site_symmetry.py` | `site_symmetry.json` | within-element variance noise |
| `j_dft_settings.py` | `dft_settings.csv`, `dft_settings_summary.txt` | DFT settings (SI Table `tab:si-dft`) |
| `k_si_tables.py` | `../si/*_table*.tex` | SI catalogue, DFT-settings and analytic-check tables |
| `l_main_figures.py` | `../figures/fig_{architecture,descriptors,paw,stability,dataset}`, `stability_map.csv` | Figs. 1--4, 6 |
| `m_validation_figures.py` | `../figures/fig_{validation,classes}`, `analytic_checks.csv`, `magpie_overlap.csv` | Figs. 5, 7 |
| `n_ml_descriptors.py SRC OUT [--renorm]` | `ml/desc_*.csv` | descriptors of the DFT and predicted test densities |
| `o_ml_analysis.py` | `ml_scores_*.csv`, `ml_summary.txt`, `../figures/fig_ml`, `../si/ml_table.tex` | Section 7, Fig. 8, SI Section S6 |

`n_ml_descriptors.py` reads the ChargE3Net inputs and full-grid test predictions
(`~/charge3net/charge3net/data`, one `.npy` per structure in e/Å³ on the DFT
grid); `out/ml/` also holds the per-structure NMAPE and timings of the
ChargE3Net tests, copied from its results directory. pydemi does not depend on
the model.

Dataset-wide numbers not listed here come from
`../../results/prompt_spec/descriptors_6000_data_aug13.csv`.

`legacy/` holds the scripts and outputs behind the first draft (earlier,
PDF-specification version of pydemi); they need that version's API
(tag `pdf-spec-final`).
