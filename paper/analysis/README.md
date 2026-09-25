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

Dataset-wide numbers not listed here come from
`../../results/prompt_spec/descriptors_6000_data_aug13.csv`.

`legacy/` holds the scripts and outputs behind the first draft (earlier,
PDF-specification version of pydemi); they need that version's API
(tag `pdf-spec-final`).
