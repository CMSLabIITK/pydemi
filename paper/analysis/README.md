# Analyses behind the paper's numerical-considerations section

Run from this directory with pydemi installed; outputs go to `out/`.
Dataset: `/data/sai/new_charge/6000_data_aug13` (VASP, 6,059 structures);
`common.py` fixes the random samples (seeds 2026, 7, 11).

| Script | Produces | Used in |
|---|---|---|
| `a_laplacian_stencils.py 200` | `laplacian_stencils.json`, `laplacian_stencils_summary.txt` | Table `tab:derivatives`; YNi3 example; integrated Laplacian ≤ 3e-14 |
| `g_spectral_floor.py` | `spectral_floor.json` | spectral round-off example (isolated Gaussian) |
| `b_partitions.py` | `partitions.json` | Tables `tab:partitions`, `tab:partition-sensitivity` |
| `c_paw.py` | `paw.json` | negative densities, CaSi3Pt, Table `tab:feni3` |
| `d_deformation.py 150` | `deformation_chgcar_route.json` | 87% / 45% / 0.6% inside PAW spheres; Hirshfeld charges in binaries |
| `e_convergence.py 30` | `convergence.json` | grid-convergence statistics |
| `f_site_symmetry.py` | `site_symmetry.json` | within-element variance noise |

The persistence and non-nuclear-maxima counts come directly from
`../../results/descriptors_6000_data_aug13.csv`.
