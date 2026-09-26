# pydemi

**Interpretable, named, fixed-length descriptors from DFT charge-density grids.**

pydemi turns a charge-density grid plus its structure into a flat
`dict[str, float]` of named descriptors, at seconds per structure, for
datasets of thousands of structures. It is a featurization layer: there is no
critical-point search anywhere, and every descriptor is computed by direct
grid operations that always terminate.

<!-- counts -->220 descriptors by default (36 bonding, 21 structural, 8 magnetic, 23 heterogeneity, 132 compositional; 4 of them tagged fragile), plus 12 in the off-by-default PAW extension and 1 in the robust extension<!-- /counts -->.
Every one is registered with its formula, units, range, sentinel cases and
references, and the catalogue below is generated from that registry.

**Author:** Shubham Maurya, CMS Lab, IIT Kanpur

A file-by-file, function-by-function walk through the source is in
[docs/CODE_GUIDE.md](docs/CODE_GUIDE.md); development status in
[PROGRESS.md](PROGRESS.md).

---

## Contents

1. [Installation](#1-installation)
2. [Quick start](#2-quick-start)
3. [Design](#3-design)
4. [Input data](#4-input-data)
5. [Options](#5-options)
6. [Descriptor catalogue](#6-descriptor-catalogue)
7. [Sentinels, flags and metadata](#7-sentinels-flags-and-metadata)
8. [Where pydemi departs from the specification](#8-where-pydemi-departs-from-the-specification)
9. [PAW pseudo-densities and the `paw` extension](#9-paw-pseudo-densities-and-the-paw-extension)
10. [Validation](#10-validation)
11. [Performance](#11-performance)
12. [Limitations](#12-limitations)
13. [API](#13-api)
14. [Data sources and citations](#14-data-sources-and-citations)

---

## 1. Installation

```bash
pip install -e .            # numpy, scipy, pandas: the numerical core
pip install -e ".[full]"    # + pymatgen, spglib, matminer (compositional domain)
pip install -e ".[dev]"     # + pytest, mypy
```

Python >= 3.10. The numerical core is pure numpy / scipy; pymatgen is used
only for structure I/O and matminer only for the compositional baseline. No
network calls at run time.

## 2. Quick start

```python
import pydemi

vd = pydemi.read_vasp("CHGCAR", elf="ELFCAR")         # VolumetricData
feats = pydemi.featurize(vd)                          # dict[str, float]

feats, meta = pydemi.featurize(
    vd,
    domains=["bonding", "magnetic"],
    partition="nearest",
    shells=(0.8, 1.5),
    deformation_reference="tabulated",
    elf_source="reconstruct",                         # or "file"
    laplacian_method="metric",
    derivative_backend="fft",
    return_metadata=True,
)

pydemi.catalogue()                                    # DataFrame of descriptor metadata
q = pydemi.hirshfeld_charges(vd)                      # q_i = Z_i - int w_i rho dV

df = pydemi.featurize_batch(paths, n_workers=8, on_error="record", progress=True)
```

```bash
pydemi featurize CHGCAR --out features.json
pydemi batch ./runs --glob "*/CHGCAR" --out features.csv --workers 8 --domains bonding,magnetic
pydemi catalogue --out catalogue.csv
pydemi sweep CHGCAR --param c2 --range 1.0:2.5:0.05 --out sweep.csv
```

## 3. Design

**One geometry pass, many operators, several fields.** For every voxel k the
geometry pass gives the distance r_k to the nearest nucleus (minimum image),
the unit vector u_k from that nucleus to the voxel, and the nucleus index
i_k. It is computed once per structure and cached on the `VolumetricData`,
together with the gradient, the Laplacian and the Hessian eigenvalues. A
small set of operators is written once and applied to any field:

| Operator | Definition | Module |
|---|---|---|
| radial moment | m_n = sum_k w_k r_k^n / sum_k w_k, w = f or \|f\| | `operators/moments.py` |
| shell fraction | f_shell = sum_{k in shell} w_k / sum_k w_k | `operators/fractions.py` |
| gradient anisotropy | zeta = 1 - sum_k \|grad f_k . u_k\| / sum_k \|grad f_k\| | `operators/anisotropy.py` |
| anisotropy tensor | T_ab = sum_k d_a f_k d_b f_k / sum_k \|grad f_k\|^2 | `operators/anisotropy.py` |
| Laplacian statistics | sign fraction, charge-weighted fraction, concentration | `operators/laplacian.py` |
| site aggregation | sums restricted to (or weighted by) atom i | `operators/sitestats.py` |
| percolation, extremum census | see section 6.2 | `operators/topology.py` |

The fields are rho, |m|, the ELF (read or reconstructed), delta rho and the
electrostatic potential (read or FFT-solved).

```
src/pydemi/
  io/          base.py (Lattice, Grid, Structure, VolumetricData), vasp.py, cube.py, xsf.py, predicted.py, registry.py
  core/        grid.py, derivatives.py, geometry.py, partition.py
  fields/      density.py, deformation.py, elf.py, potential.py
  operators/   moments.py, fractions.py, anisotropy.py, laplacian.py, topology.py, sitestats.py
  descriptors/ registry.py, bonding.py, structural.py, magnetic.py, heterogeneity.py, compositional.py
  validate/    analytic.py, invariance.py, convergence.py
  constants.py, data/ (elements.csv, free_atoms.npz, magpie_labels.txt), batch.py, cli.py
tools/         generators for the data files and this README's tables
```

**Derivatives** (`core/derivatives.py`). With B = inv(A)^T (rows b1, b2, b3,
no 2 pi) and fractional coordinates u: grad f = sum_a b_a df/du_a,
lap f = sum_{a,b} G[a,b] d^2f/du_a du_b with G = B B^T (off-diagonal terms
included; `laplacian_method="diagonal"` exists only to quantify that
shortcut), H_cart = B^T H_frac B. Backends: `"fft"` (default; exact for
band-limited data) and `"fd"` (central differences, order 2/4/6/8, default 4).

## 4. Input data

| File | Read as | Notes |
|---|---|---|
| CHGCAR / CHG | rho, magnetization | values are rho * V_cell: divided by V; Fortran order; second block = m (augmentation lines skipped); four blocks = non-collinear (rho, m_x, m_y, m_z), read as a vector |
| AECCAR0 + AECCAR2 | `read_all_electron` | sum = all-electron density, `density_source="all_electron"`; AECCAR0 kept as `core_density` |
| ELFCAR | elf | not volume-scaled; resampled trilinearly onto the density grid (stays in [0, 1]) |
| LOCPOT | potential | eV, not volume-scaled |
| POTCAR / OUTCAR | ZVAL, RCORE | found next to the CHGCAR by `read_vasp` (`potcar="auto"`); without either, per-element tables `read_vasp(zval=, paw_radii=)` / `--paw-table`, then the fallback below |
| `*.cube` | rho | bohr and e/bohr^3 converted to Angstrom and e/Angstrom^3 on read |
| `*.xsf` | rho | Angstrom; periodic duplicate plane dropped |
| `*.npz` (ML prediction) | `read_predicted` | written by `write_predicted`; e/Angstrom^3; rescaled to N = sum ZVAL by default |

`pydemi.read(path)` sniffs the format; `read_vasp(chgcar, elf=, locpot=,
aeccar0=, aeccar2=)` assembles one VASP run. A pymatgen `Structure` is
accepted wherever a structure is.

ZVAL (the electrons a pseudo-density holds per atom) sets the free-atom
reference, the Hirshfeld Z_i and the ionic charges; `zval_source` in the
metadata says where it came from (`potcar`, `outcar`, `table`, `default`).
The fallback `pydemi.data.default_zval` reproduces the standard PBE PAW
datasets (p-block without the filled d10, lanthanides and actinides with
the outer s2 p6), 78 of the 87 elements of the 6,059-structure dataset; it
cannot know the semicore choices (K_pv vs K_sv, Ca_pv, Sr_sv, Y_sv, Zr_sv,
Nb_pv, ...). A run without POTCAR or OUTCAR should get a table from the
other runs of the same POTCAR set; a wrong count shows as a large
`def_charge_mismatch` (integer multiples of the missing electrons).

### Densities predicted by a machine-learning model

A model that predicts the valence density from the crystal structure (for
example ChargE3Net trained on VASP CHGCARs) hands its result to pydemi as a
`.npz` file: lattice, species, fractional coordinates and rho in
e/Angstrom^3, written with `pydemi.write_predicted`. pydemi itself does not
depend on any ML framework.

```python
import pydemi
shape = pydemi.vasp_grid_shape(lattice, encut=500)   # the grid VASP would use (PREC = Accurate)
# ... the model predicts rho at the points of that grid ...
pydemi.write_predicted("x.npz", structure, rho, model="ChargE3Net")
vd = pydemi.read_predicted("x.npz", zval=zval_table, paw_radii=radius_table)
features, meta = pydemi.featurize(vd, return_metadata=True)
```

`vasp_grid_shape` reproduces the CHGCAR grid of every run of the
6,059-structure dataset from the lattice and ENCUT alone, so no DFT
calculation is needed. `read_predicted` treats the prediction as a PAW
pseudo-density (`density_source="pseudo"`) and by default rescales it to
N = sum ZVAL. The metadata record `density_origin="predicted"`, the model and
`charge_scale`. A model of the total density gives no magnetization, so the
magnetic descriptors return their documented non-magnetic sentinels, flagged. Which descriptors
survive the prediction error is measured in `paper/` (Section 7): the
critical-point census and the higher-derivative descriptors do not.

Internal units: Angstrom, electrons / Angstrom^3, eV. ELF_D, g, v, H, the NCI
thresholds and the information measures are evaluated in atomic units, with
explicit conversions (`constants.py`).

## 5. Options

| Option | Default | Values |
|---|---|---|
| `domains` | all five | bonding, structural, magnetic, heterogeneity, compositional |
| `extensions` | none | `"paw"` (section 9), `"robust"` (section 6.6) |
| `partition` | `"nearest"` | nearest, power (covalent radii), becke, hirshfeld |
| `shells` | (0.8, 1.5) | (c1, c2) in Angstrom, or `Shells(s1, s2, scaled=True)` (multiples of covalent radii) |
| `deformation_reference` | `"auto"` | aeccar0, tabulated, custom (`custom_reference=<dir>`); auto = aeccar0 when AECCAR0 was read, else tabulated |
| `elf_source` | `"auto"` | reconstruct, file; auto = file when an ELFCAR was read |
| `potential_source` | `"auto"` | locpot, hartree, esp (Hartree + Gaussian ionic term); auto = locpot when read |
| `laplacian_method` | `"metric"` | metric, diagonal |
| `derivative_backend` | `"fft"` | fft, fd (`fd_order` 2/4/6/8) |
| `float32` | False | cast every grid to float32 first |

The settings used, and the sources actually chosen by the `"auto"` options,
are recorded in the metadata.

## 6. Descriptor catalogue

Generated from the registry (`python tools/generate_docs.py`); the complete
table, with ranges and references, is `docs/catalogue.csv` or
`pydemi.catalogue()`. Every descriptor is intensive (section 10).

### 6.1 Bonding

<!-- catalogue:bonding:start -->
| Name | Units | Definition | Sentinel cases | Stability |
|---|---|---|---|---|
| `zeta` | dimensionless | zeta = 1 - sum_k \|grad rho_k . u_k\| / sum_k \|grad rho_k\| | uniform_density=0.0 | robust |
| `m1` | Angstrom | m1 = sum_k rho_k r_k / sum_k rho_k | zero_density=0.0 | robust |
| `m2` | Angstrom^2 | m2 = sum_k rho_k r_k^2 / sum_k rho_k | zero_density=0.0 | robust |
| `sigma_r2` | Angstrom^2 | sigma_r2 = m2 - m1^2 | zero_density=0.0 | robust |
| `f_core` | dimensionless | f_core = sum_{k: r <= c1} rho_k / sum_k rho_k | zero_density=0.0 | robust |
| `f_bond` | dimensionless | f_bond = sum_{k: c1 < r <= c2} rho_k / sum_k rho_k | zero_density=0.0 | robust |
| `f_int` | dimensionless | f_int = sum_{k: r > c2} rho_k / sum_k rho_k | zero_density=0.0 | robust |
| `lnf` | dimensionless | lnf = (1/N) sum_k 1(lap rho_k < 0) | uniform_density=0.0 | robust |
| `lnf_charge_weighted` | dimensionless | lnf_charge_weighted = sum_{lap rho_k < 0} rho_k / sum_k rho_k | uniform_density=0.0;zero_density=0.0 | robust |
| `lap_concentration` | dimensionless | lap_concentration = sum_{lap rho < 0} \|lap rho_k\| / sum_k \|lap rho_k\| | uniform_density=0.5 | robust |
| `lap_concentration_valence` | dimensionless | lap_concentration_valence = sum_{r > c1, lap rho < 0} \|lap rho_k\| / sum_{r > c1} \|lap rho_k\| | uniform_density=0.5;empty_region=0.5 | robust |
| `ellip_bond_avg` | dimensionless | ellip_bond_avg = mean of lambda1/lambda2 - 1 over {k in bond, lambda2 < 0} | no_bond_voxels=0.0 | **fragile** |
| `ellip_bond_std` | dimensionless | ellip_bond_std = std of lambda1/lambda2 - 1 over {k in bond, lambda2 < 0} | no_bond_voxels=0.0 | **fragile** |
| `f_H_negative` | dimensionless | f_H_negative = (1/N_bond) sum_{k in bond} 1(H_k < 0),  H = g + v | empty_region=0.0 | robust |
| `H_bond_mean` | hartree/bohr^3 | H_bond_mean = <H_k> over the bond shell,  H = g + v = (1/4) lap rho - g (a.u.) | empty_region=0.0 | robust |
| `G_over_rho` | hartree/electron | G_over_rho = <g_k / rho_k> over the bond shell (a.u.) | empty_region=0.0 | robust |
| `f_ELF_localized` | dimensionless | f_ELF_localized = (1/N_bond) sum_{k in bond} 1(ELF_k > 0.5) | empty_region=0.0 | robust |
| `ELF_bond_avg` | dimensionless | ELF_bond_avg = <ELF_k> over the bond shell | empty_region=0.0 | robust |
| `ELF_core_valence_contrast` | dimensionless | ELF_core_valence_contrast = <ELF>_core / <ELF>_bond | empty_region=0.0;zero_denominator=0.0 | robust |
| `zeta_ELF` | dimensionless | zeta_ELF = 1 - sum_k \|grad ELF_k . u_k\| / sum_k \|grad ELF_k\| | uniform_density=0.0 | robust |
| `f_NCI` | dimensionless | f_NCI = (1/N) sum_k 1(s_k < 0.5 and rho_k < 0.05 a.u.),  s = \|grad rho\| / (2 (3 pi^2)^(1/3) rho^(4/3)) | uniform_density=0.0 | robust |
| `NCI_attractive` | dimensionless | NCI_attractive = fraction of NCI voxels with lambda2 < 0 | uniform_density=0.0;no_nci_voxels=0.0 | robust |
| `sign_lambda2_rho_mean` | e/bohr^3 | sign_lambda2_rho_mean = mean of sign(lambda2) rho_k over NCI voxels (rho in a.u.) | uniform_density=0.0;no_nci_voxels=0.0 | robust |
| `V_spread` | eV | V_spread = std over sites i of V(R_i) | - | robust |
| `V_int_min` | eV | V_int_min = min over interstitial voxels of V_k (cell average of V set to 0) | empty_region=0.0 | robust |
| `rho_mid_mean` | e/Angstrom^3 | rho_mid_mean = mean of rho at nearest-neighbour bond midpoints | no_bonds=0.0 | robust |
| `rho_mid_std` | e/Angstrom^3 | rho_mid_std = std of rho at nearest-neighbour bond midpoints | no_bonds=0.0 | robust |
| `m1_def` | Angstrom | m1_def = sum_k \|drho_k\| r_k / sum_k \|drho_k\|,  drho = rho - promolecule | zero_deformation=0.0 | robust |
| `m2_def` | Angstrom^2 | m2_def = sum_k \|drho_k\| r_k^2 / sum_k \|drho_k\| | zero_deformation=0.0 | robust |
| `sigma_r2_def` | Angstrom^2 | sigma_r2_def = m2_def - m1_def^2 | zero_deformation=0.0 | robust |
| `f_bond_def` | dimensionless | f_bond_def = sum_{k in bond, drho > 0} drho_k / sum_{drho > 0} drho_k | no_accumulation=0.0 | robust |
| `f_int_def` | dimensionless | f_int_def = sum_{k in int, drho > 0} drho_k / sum_{drho > 0} drho_k | no_accumulation=0.0 | robust |
| `f_bond_dep` | dimensionless | f_bond_dep = sum_{k in bond, drho < 0} \|drho_k\| / sum_{drho < 0} \|drho_k\| | no_depletion=0.0 | robust |
| `def_polarity` | dimensionless | def_polarity = sum_k \|drho_k\| dV / Q_tot,  Q_tot = sum_k rho_k dV | zero_density=0.0 | robust |
| `bond_charge_transfer_pair_mean` | electrons | bond_charge_transfer_pair_mean = mean over nearest-neighbour pairs of int_{region(i,j)} drho dV | no_bonds=0.0 | robust |
| `bond_charge_transfer_pair_std` | electrons | bond_charge_transfer_pair_std = std over nearest-neighbour pairs of int_{region(i,j)} drho dV | no_bonds=0.0 | robust |
<!-- catalogue:bonding:end -->

### 6.2 Structural

Maxima and minima are found by 26-neighbour comparison; saddles from the
Freudenthal link (section 8). Basins are steepest-ascent over 26 neighbours.
Percolation: {rho > c} is labelled with 6-connectivity and merged across the
periodic faces by a union-find that tracks winding vectors.

<!-- catalogue:structural:start -->
| Name | Units | Definition | Sentinel cases | Stability |
|---|---|---|---|---|
| `rho_perc_a` | e/Angstrom^3 | rho_perc_a = max {c : the super-level set {rho > c} spans a1 under PBC} | - | robust |
| `rho_perc_b` | e/Angstrom^3 | rho_perc_b = max {c : the super-level set {rho > c} spans a2 under PBC} | - | robust |
| `rho_perc_c` | e/Angstrom^3 | rho_perc_c = max {c : the super-level set {rho > c} spans a3 under PBC} | - | robust |
| `perc_anisotropy` | dimensionless | perc_anisotropy = (max_alpha - min_alpha) / mean_alpha of rho_perc | zero_levels=0.0 | robust |
| `n_max` | 1/Angstrom^3 | n_max = (number of local maxima) / V_cell | - | robust |
| `n_min` | 1/Angstrom^3 | n_min = (number of local minima) / V_cell | - | robust |
| `n_saddle1` | 1/Angstrom^3 | n_saddle1 = (number of index-1 saddles) / V_cell | - | **fragile** |
| `n_saddle2` | 1/Angstrom^3 | n_saddle2 = (number of index-2 saddles) / V_cell | - | **fragile** |
| `n_NNM` | 1/Angstrom^3 | n_NNM = (number of local maxima with min_i \|r - R_i\| > r_cut) / V_cell | - | robust |
| `Q_NNM` | dimensionless | Q_NNM = (charge in the steepest-ascent basins of the non-nuclear maxima) / Q_tot | - | robust |
| `rho_min` | e/Angstrom^3 | rho_min = min_k rho_k | - | robust |
| `rho_min_ratio` | dimensionless | rho_min_ratio = rho_min / <rho>_V | zero_density=0.0 | robust |
| `rho_int_mean` | e/Angstrom^3 | rho_int_mean = mean of rho over the interstitial shell (r > c2) | empty_region=0.0 | robust |
| `T_eigenvalues_t1` | dimensionless | T_eigenvalues_t1 = eigenvalue 1 (ascending, t1 <= t2 <= t3) of T_ab = sum_k d_a rho_k d_b rho_k / sum_k \|grad rho_k\|^2 (trace 1) | uniform_density=0.3333333333333333 | robust |
| `T_eigenvalues_t2` | dimensionless | T_eigenvalues_t2 = eigenvalue 2 (ascending, t1 <= t2 <= t3) of T_ab = sum_k d_a rho_k d_b rho_k / sum_k \|grad rho_k\|^2 (trace 1) | uniform_density=0.3333333333333333 | robust |
| `T_eigenvalues_t3` | dimensionless | T_eigenvalues_t3 = eigenvalue 3 (ascending, t1 <= t2 <= t3) of T_ab = sum_k d_a rho_k d_b rho_k / sum_k \|grad rho_k\|^2 (trace 1) | uniform_density=0.3333333333333333 | robust |
| `charge_FA` | dimensionless | charge_FA = sqrt(3/2) \|\|T - (1/3) I\|\|_F / \|\|T\|\|_F | uniform_density=0.0 | robust |
| `shannon_entropy` | dimensionless | shannon_entropy = -int rho~ ln rho~ dV - ln V,  rho~ = rho / N_e | zero_density=0.0 | robust |
| `fisher_information` | 1/bohr^2 | fisher_information = int \|grad rho~\|^2 / rho~ dV  (a.u.; voxels above the density floor) | zero_density=0.0 | robust |
| `disequilibrium` | dimensionless | disequilibrium = V int rho~^2 dV | zero_density=0.0 | robust |
| `LMC_complexity` | dimensionless | LMC_complexity = D e^S  (D, S the unnormalized disequilibrium and entropy; = 1 when uniform) | zero_density=0.0 | robust |
<!-- catalogue:structural:end -->

### 6.3 Magnetic

Every entry is 0.0 for a non-magnetic structure, with `magnetic = False` in
the metadata. Non-collinear runs use vector moments.

<!-- catalogue:magnetic:start -->
| Name | Units | Definition | Sentinel cases | Stability |
|---|---|---|---|---|
| `M_abs_per_atom` | mu_B/atom | M_abs_per_atom = sum_k \|m_k\| dV / n_atoms | non_magnetic=0.0 | robust |
| `M_net_per_atom` | mu_B/atom | M_net_per_atom = \|sum_k m_k dV\| / n_atoms | non_magnetic=0.0 | robust |
| `m1_spin` | Angstrom | m1_spin = sum_k \|m_k\| r_k / sum_k \|m_k\| | non_magnetic=0.0 | robust |
| `sigma_r2_spin` | Angstrom^2 | sigma_r2_spin = sum_k \|m_k\| r_k^2 / sum_k \|m_k\| - m1_spin^2 | non_magnetic=0.0 | robust |
| `f_bond_spin` | dimensionless | f_bond_spin = sum_{k in bond} \|m_k\| / sum_k \|m_k\| | non_magnetic=0.0 | robust |
| `mu_site_std` | mu_B | mu_site_std = std over i of mu_i,  mu_i = sum_k w_i(k) m_k dV | non_magnetic=0.0 | robust |
| `spin_frustration` | dimensionless | spin_frustration = 1 - \|sum_i mu_i\| / sum_i \|mu_i\| | non_magnetic=0.0 | robust |
| `spin_charge_correlation` | dimensionless | spin_charge_correlation = Pearson r(rho_k, \|m_k\|) over voxels | non_magnetic=0.0 | robust |
<!-- catalogue:magnetic:end -->

### 6.4 Heterogeneity

A meta-operator over per-site descriptors X^(i) (m1, f_bond, zeta, mu): std,
range, max, min and the one-way ANOVA decomposition
Var_i(X) = sum_e w_e Var_{i in e}(X) + sum_e w_e (Xbar_e - Xbar)^2, with w_e
the fraction of sites of element e (identity tested). `mu_site_std` is in the
magnetic domain.

<!-- catalogue:heterogeneity:start -->
| Name | Units | Definition | Sentinel cases | Stability |
|---|---|---|---|---|
| `m1_site_std` | Angstrom | m1_site_std = std over sites of X^(i),  X^(i) = sum_k w_i(k) rho_k r_ik / sum_k w_i(k) rho_k | uniform_density=0.0;non_magnetic=0.0 | robust |
| `m1_site_range` | Angstrom | m1_site_range = max - min over sites of X^(i),  X^(i) = sum_k w_i(k) rho_k r_ik / sum_k w_i(k) rho_k | uniform_density=0.0;non_magnetic=0.0 | robust |
| `m1_site_max` | Angstrom | m1_site_max = max over sites of X^(i),  X^(i) = sum_k w_i(k) rho_k r_ik / sum_k w_i(k) rho_k | uniform_density=0.0;non_magnetic=0.0 | robust |
| `m1_site_min` | Angstrom | m1_site_min = min over sites of X^(i),  X^(i) = sum_k w_i(k) rho_k r_ik / sum_k w_i(k) rho_k | uniform_density=0.0;non_magnetic=0.0 | robust |
| `m1_within_element_var` | (Angstrom)^2 | m1_within_element_var = sum_e w_e Var_{i in e} of X^(i),  X^(i) = sum_k w_i(k) rho_k r_ik / sum_k w_i(k) rho_k | uniform_density=0.0;non_magnetic=0.0;one_site_per_element=nan | robust |
| `m1_between_element_var` | (Angstrom)^2 | m1_between_element_var = sum_e w_e (mean_e - mean)^2 of X^(i),  X^(i) = sum_k w_i(k) rho_k r_ik / sum_k w_i(k) rho_k | uniform_density=0.0;non_magnetic=0.0;single_element=0.0 | robust |
| `f_bond_site_std` | dimensionless | f_bond_site_std = std over sites of X^(i),  X^(i) = sum_{c1 < r_ik <= c2} w_i(k) rho_k / sum_k w_i(k) rho_k | uniform_density=0.0;non_magnetic=0.0 | robust |
| `f_bond_site_range` | dimensionless | f_bond_site_range = max - min over sites of X^(i),  X^(i) = sum_{c1 < r_ik <= c2} w_i(k) rho_k / sum_k w_i(k) rho_k | uniform_density=0.0;non_magnetic=0.0 | robust |
| `f_bond_site_max` | dimensionless | f_bond_site_max = max over sites of X^(i),  X^(i) = sum_{c1 < r_ik <= c2} w_i(k) rho_k / sum_k w_i(k) rho_k | uniform_density=0.0;non_magnetic=0.0 | robust |
| `f_bond_site_min` | dimensionless | f_bond_site_min = min over sites of X^(i),  X^(i) = sum_{c1 < r_ik <= c2} w_i(k) rho_k / sum_k w_i(k) rho_k | uniform_density=0.0;non_magnetic=0.0 | robust |
| `f_bond_within_element_var` | dimensionless | f_bond_within_element_var = sum_e w_e Var_{i in e} of X^(i),  X^(i) = sum_{c1 < r_ik <= c2} w_i(k) rho_k / sum_k w_i(k) rho_k | uniform_density=0.0;non_magnetic=0.0;one_site_per_element=nan | robust |
| `f_bond_between_element_var` | dimensionless | f_bond_between_element_var = sum_e w_e (mean_e - mean)^2 of X^(i),  X^(i) = sum_{c1 < r_ik <= c2} w_i(k) rho_k / sum_k w_i(k) rho_k | uniform_density=0.0;non_magnetic=0.0;single_element=0.0 | robust |
| `zeta_site_std` | dimensionless | zeta_site_std = std over sites of X^(i),  X^(i) = 1 - sum_k w_i(k) \|grad rho_k . u_ik\| / sum_k w_i(k) \|grad rho_k\| | uniform_density=0.0;non_magnetic=0.0 | robust |
| `zeta_site_range` | dimensionless | zeta_site_range = max - min over sites of X^(i),  X^(i) = 1 - sum_k w_i(k) \|grad rho_k . u_ik\| / sum_k w_i(k) \|grad rho_k\| | uniform_density=0.0;non_magnetic=0.0 | robust |
| `zeta_site_max` | dimensionless | zeta_site_max = max over sites of X^(i),  X^(i) = 1 - sum_k w_i(k) \|grad rho_k . u_ik\| / sum_k w_i(k) \|grad rho_k\| | uniform_density=0.0;non_magnetic=0.0 | robust |
| `zeta_site_min` | dimensionless | zeta_site_min = min over sites of X^(i),  X^(i) = 1 - sum_k w_i(k) \|grad rho_k . u_ik\| / sum_k w_i(k) \|grad rho_k\| | uniform_density=0.0;non_magnetic=0.0 | robust |
| `zeta_within_element_var` | dimensionless | zeta_within_element_var = sum_e w_e Var_{i in e} of X^(i),  X^(i) = 1 - sum_k w_i(k) \|grad rho_k . u_ik\| / sum_k w_i(k) \|grad rho_k\| | uniform_density=0.0;non_magnetic=0.0;one_site_per_element=nan | robust |
| `zeta_between_element_var` | dimensionless | zeta_between_element_var = sum_e w_e (mean_e - mean)^2 of X^(i),  X^(i) = 1 - sum_k w_i(k) \|grad rho_k . u_ik\| / sum_k w_i(k) \|grad rho_k\| | uniform_density=0.0;non_magnetic=0.0;single_element=0.0 | robust |
| `mu_site_range` | mu_B | mu_site_range = max - min over sites of X^(i),  X^(i) = mu_i = sum_k w_i(k) m_k dV (signed; \|mu_i\| for non-collinear runs) | uniform_density=0.0;non_magnetic=0.0 | robust |
| `mu_site_max` | mu_B | mu_site_max = max over sites of X^(i),  X^(i) = mu_i = sum_k w_i(k) m_k dV (signed; \|mu_i\| for non-collinear runs) | uniform_density=0.0;non_magnetic=0.0 | robust |
| `mu_site_min` | mu_B | mu_site_min = min over sites of X^(i),  X^(i) = mu_i = sum_k w_i(k) m_k dV (signed; \|mu_i\| for non-collinear runs) | uniform_density=0.0;non_magnetic=0.0 | robust |
| `mu_within_element_var` | (mu_B)^2 | mu_within_element_var = sum_e w_e Var_{i in e} of X^(i),  X^(i) = mu_i = sum_k w_i(k) m_k dV (signed; \|mu_i\| for non-collinear runs) | uniform_density=0.0;non_magnetic=0.0;one_site_per_element=nan | robust |
| `mu_between_element_var` | (mu_B)^2 | mu_between_element_var = sum_e w_e (mean_e - mean)^2 of X^(i),  X^(i) = mu_i = sum_k w_i(k) m_k dV (signed; \|mu_i\| for non-collinear runs) | uniform_density=0.0;non_magnetic=0.0;single_element=0.0 | robust |
<!-- catalogue:heterogeneity:end -->

### 6.5 Compositional

The 132 features of `matminer.featurizers.composition.ElementProperty.from_preset("magpie")`,
not reimplemented, named `magpie_<stat>_<property>` (e.g.
`magpie_mean_Electronegativity`) and tagged `domain="compositional"`,
`adopted=True`, so they can be kept out of novelty claims and used as an
explicit baseline. Needs `pydemi[full]`.

### 6.6 Stability tags and the `robust` extension

Four descriptors are computed exactly as specified but are not numerically
converged on typical VASP grids, and are registered as `stability="fragile"`
(the Stability column above). On the 6,059-structure dataset
(`paper/analysis/`):

| Descriptor | FFT vs FD4 | 100% vs 80% grid | Cause |
|---|---|---|---|
| `ellip_bond_avg` | 22% | 21% | lambda1/lambda2 - 1 diverges as lambda2 -> 0 at many bond-shell voxels |
| `ellip_bond_std` | 67% | 62% | same |
| `n_saddle1`, `n_saddle2` | - | 20%, 17% | saddles of near-flat, rippled regions appear and vanish with the grid |

(median relative change; 48 random structures for the derivative schemes and
the ellipticity grid test, 30 for the census grid test.)

They stay in the default `featurize` output, so every table keeps the
specified columns. For models use the robust set:

```python
names = pydemi.descriptor_names(include_fragile=False)   # 216 of the 220 defaults
```

`catalogue()` has a `stability` column. The off-by-default `robust`
extension adds a converged ellipticity, the mean of 1 - lambda2/lambda1 over
the same voxels -- per voxel e / (1 + e), a monotone map of the ellipticity
onto [0, 1) that cannot diverge (median change 0.2% FFT vs FD4, 1.1% with
FD2, 0.2% on an 80% grid):

<!-- catalogue:robust:start -->
| Name | Units | Definition | Sentinel cases | Stability |
|---|---|---|---|---|
| `ellip_bond_bounded_avg` | dimensionless | ellip_bond_bounded_avg = mean of 1 - lambda2/lambda1 over {k in bond, lambda2 < 0} | no_bond_voxels=0.0 | robust |
<!-- catalogue:robust:end -->

`pydemi.featurize(vd, extensions=["robust"])` (or `--extensions robust`).
The topological descriptors of record are `Q_NNM` and the percolation
thresholds `rho_perc_*` (changes below 1% on an 80% grid).

## 7. Sentinels, flags and metadata

A descriptor that is undefined for a degenerate input returns a documented
constant and sets its companion flag `<name>__flag` in the metadata; it never
emits a bare NaN. The one exception allowed by the specification is a
within-element variance when no element has two sites: NaN, flagged, with the
per-element site counts (`site_counts`, `max_sites_per_element`). The adopted
Magpie features are passed through from matminer: an element without Magpie
data gives NaN there, flagged `missing_element_data`.

| Case | Affected | Value |
|---|---|---|
| uniform density | zeta, lnf, lnf_charge_weighted, f_NCI, NCI_*, zeta_ELF, per-site zeta | 0.0 |
| uniform density | lap_concentration(_valence) | 0.5 |
| uniform density | T_eigenvalues_t1..t3 / charge_FA | 1/3 / 0.0 |
| non-magnetic | every magnetic descriptor and every mu statistic | 0.0, `magnetic = False` |
| single element | X_between_element_var | 0.0 |
| no voxels in a region | shell-restricted averages | 0.0 (`empty_region`) |
| rho -> 0 voxels | ELF_D, g, H, s, Fisher information | excluded below RHO_FLOOR_AU = 1e-8 e/bohr^3 |

Metadata of every structure (and every `featurize_batch` row): `n_atoms`,
`volume`, `grid_shape`, `density_source`, `density_origin` (`dft` or
`predicted`), `density_model`, `charge_scale`, `zval_source`, `spin_mode`, `magnetic`, `M_abs`,
`M_net` (extensive, so metadata only), `euler_consistency` (below), `site_counts`,
`partition`, `shells`, `derivative_backend`, `laplacian_method`,
`elf_source`, `potential_source`, `deformation_reference`,
`def_charge_mismatch`, `precision`, the `__flag` columns, `sentinels`,
`wall_time_s`, `pydemi_version`, `error`.

`euler_consistency` = n_max - n_saddle2 + n_saddle1 - n_min (raw counts).
Because a census taken entirely on the 14-neighbour Freudenthal link always
closes, it equals the number of extrema whose status depends on the stencil,
(n_max^26 - n_max^14) - (n_min^26 - n_min^14) (checked on 200 structures).
Zero means both neighbourhoods find the same extrema. On VASP
pseudo-densities it is nonzero for 91% of the 6,059-structure dataset
(median 5% of all critical points), set by PAW-pseudized regions and
low-amplitude ripple rather than by coarse grids (median spacing 0.064 A in
both groups): read the census counts of such structures with care.

## 8. Where pydemi departs from the specification

Each change is documented at the point of use and backed by a test or a
dataset figure.

| Item | Specification | pydemi | Why |
|---|---|---|---|
| `lap_concentration` | sum over lap < 0 of \|lap\| / sum \|lap\| | kept, plus `lap_concentration_valence` (r > c1) | int lap rho dV = 0 on a periodic grid, so the whole-cell value is identically 1/2 (tested for both backends) |
| `rho_perc_*` | lowest spanning level | highest spanning level | the lowest is always min(rho): the whole cell spans |
| percolation test | a cluster touching both faces | a cluster with nonzero winding | a blob across the periodic boundary touches both faces without spanning; the face test makes the level depend on the cell origin (translation test) |
| `"aeccar0"` reference | AECCAR0 as the promolecule | AECCAR0 + tabulated free-atom valence (ZVAL) | AECCAR0 is the frozen core; rho - AECCAR0 would be the whole valence density |
| `hirshfeld_charges` | q_i = Z_i - int w_i rho | Z_i = ZVAL for a pseudo-density, Z for all-electron | the reference must count the electrons the density holds |
| saddles in the census | 26-neighbour | Freudenthal (14-neighbour) link | a saddle cannot be classified on the 26-neighbour shell, which is not a triangulated sphere: every adjacency on it miscounts the 3 + 3 saddles of cos 2 pi x + cos 2 pi y + cos 2 pi z. Extrema stay 26-neighbour, so `euler_consistency` counts the extrema the two neighbourhoods classify differently (section 7) |
| information measures | S = -int rho~ ln rho~, D = int rho~^2 | S - ln V, V D (V in bohr^3) | the unnormalized S shifts by ln 8 and D divides by 8 for a 2x2x2 supercell (spec §10 demands intensivity) |
| extensive quantities | M_abs, M_net, counts, Q_NNM | per atom / per volume / fraction of Q_tot; raw values in metadata | spec §10 |
| equidistant images | - | ordered by the largest fractional displacement; shell and cutoff tests with a 1e-8 A tolerance | atoms on high-symmetry grid points put voxels exactly on Voronoi facets and shell boundaries; without a geometric rule the supercell and translation tests fail |

## 9. PAW pseudo-densities and the `paw` extension

A VASP CHGCAR is the PAW pseudo-density plus compensation charge, not an
all-electron density. On a 6,059-structure VASP dataset: the density is
negative somewhere in 61% of structures, so `rho_min` is usually a PAW
artefact near a nucleus; pseudized atoms can lack a maximum at the nucleus
and show lobes inside the augmentation sphere (CaSi3Pt: 8 lobes 0.81-0.83 A
from Si holding 6.3 e, just beyond a 0.8 A cutoff); and a median 87% of the
whole-cell int |delta rho| lies inside the augmentation spheres, which fill
45% of the volume. The `paw` extension (`extensions=["paw"]`, off by default)
excludes the spheres, with R_PAW = RCORE from the POTCAR / OUTCAR or, when
unknown, the covalent radius (`paw_radii_source` in the metadata):

<!-- catalogue:paw:start -->
| Name | Units | Definition | Sentinel cases | Stability |
|---|---|---|---|---|
| `m1_def_out` | Angstrom | m1_def_out = sum \|drho_k\| r_k / sum \|drho_k\| over voxels outside every PAW augmentation sphere (r > R_PAW of the nearest nucleus) | empty_region=0.0;zero_deformation=0.0 | robust |
| `m2_def_out` | Angstrom^2 | m2_def_out = sum \|drho_k\| r_k^2 / sum \|drho_k\| over voxels outside every PAW augmentation sphere (r > R_PAW of the nearest nucleus) | empty_region=0.0;zero_deformation=0.0 | robust |
| `sigma_r2_def_out` | Angstrom^2 | sigma_r2_def_out = m2_def_out - m1_def_out^2 over voxels outside every PAW augmentation sphere (r > R_PAW of the nearest nucleus) | empty_region=0.0;zero_deformation=0.0 | robust |
| `f_bond_def_out` | dimensionless | f_bond_def_out = sum_{bond, drho > 0} drho / sum_{drho > 0} drho over voxels outside every PAW augmentation sphere (r > R_PAW of the nearest nucleus) | empty_region=0.0;no_accumulation=0.0 | robust |
| `f_int_def_out` | dimensionless | f_int_def_out = sum_{int, drho > 0} drho / sum_{drho > 0} drho over voxels outside every PAW augmentation sphere (r > R_PAW of the nearest nucleus) | empty_region=0.0;no_accumulation=0.0 | robust |
| `f_bond_dep_out` | dimensionless | f_bond_dep_out = sum_{bond, drho < 0} \|drho\| / sum_{drho < 0} \|drho\| over voxels outside every PAW augmentation sphere (r > R_PAW of the nearest nucleus) | empty_region=0.0;no_depletion=0.0 | robust |
| `def_polarity_out` | dimensionless | def_polarity_out = sum \|drho_k\| dV / Q_tot over voxels outside every PAW augmentation sphere (r > R_PAW of the nearest nucleus) | empty_region=0.0;zero_density=0.0 | robust |
| `def_out_volume_fraction` | dimensionless | def_out_volume_fraction = fraction of the cell outside every PAW augmentation sphere | - | robust |

| Name | Units | Definition | Sentinel cases | Stability |
|---|---|---|---|---|
| `rho_min_int` | e/Angstrom^3 | rho_min_int = min of rho over voxels with r > max(c2, R_PAW) of their nearest nucleus | empty_region=0.0 | robust |
| `rho_min_int_ratio` | dimensionless | rho_min_int_ratio = rho_min_int / <rho>_V | empty_region=0.0;zero_density=0.0 | robust |
| `n_NNM_paw` | 1/Angstrom^3 | n_NNM_paw = (number of local maxima with r > max(r_cut, R_PAW,i)) / V_cell | - | robust |
| `Q_NNM_paw` | dimensionless | Q_NNM_paw = (charge in the basins of the maxima counted by n_NNM_paw) / Q_tot | - | robust |
<!-- catalogue:paw:end -->

For all-electron work read AECCAR0 + AECCAR2 (`read_vasp(..., aeccar0=, aeccar2=)`).

## 10. Validation

862 tests (`pytest`), organized by the milestones of the specification:

- **I/O**: volume division, Fortran order, spin block after augmentation
  lines, non-collinear four blocks, VASP 4 headers, ELFCAR / LOCPOT not
  scaled, AECCAR sum, POTCAR ZVAL / RCORE, cube unit conversion against a
  hand-written known file, cube / XSF round trips, format sniffing.
- **Derivatives**: FFT exact (1e-12) on band-limited Gaussians in
  orthorhombic and triclinic cells; finite differences at their nominal order
  (2, 4, 6); the diagonal Laplacian 12% wrong on a triclinic cell; Hessian
  symmetric with trace = Laplacian; chunked eigenvalues exact; FFT ringing on
  a Slater cusp documented by a test.
- **Geometry**: nearest atom and power diagram against brute force, including
  a cell where a fixed 3x3x3 supercell misses the nearest image.
- **Analytic densities** (spec §11): Slater 1s int rho = 1, m1 = 3/(2 zeta),
  m2 = 3/zeta^2, sigma_r2 = 3/(4 zeta^2), zeta = 0, lnf = volume fraction of
  r < 1/zeta; uniform density exercises every sentinel; two-atom
  superposition with a custom reference gives int delta rho = 0 or the
  known N/2; the recommended mesh (0.08 A at 2%) is read off
  `analytic_convergence`.
- **Invariance** (spec §10): every registered descriptor (232) on a
  low-symmetry crystal equals its value on the 2x2x2 supercell, a rigid
  translation and a rigid rotation to 1e-6 relative; also under the power,
  Hirshfeld and Becke partitions.
- **Identities**: ANOVA within + between = total; sum_i mu_i = M_net for
  every tiling partition; partition weights sum to 1; Hirshfeld charges of a
  promolecule equal Z_i - int rho_free_i exactly; lap_concentration = 1/2.
- **Fields**: ELF_D in [0, 1] on real PAW data; uniform-gas limits (ELF_D =
  1/2, H = -g); the Hartree potential of a Gaussian against erf(r)/r with the
  neutralizing background; a neutral electron + ion cell gives V = 0;
  bond-midpoint density against the two-Gaussian closed form.
- **Topology**: the census of a periodic function with known critical points
  (1, 1, 3, 3; Euler 0); a simple cubic lattice of Gaussians percolates at the
  bond-midpoint density; a planted non-nuclear maximum and its basin charge.
- **Tooling**: batch with recorded errors, CLI commands, sweep, grid
  convergence, float32 agreement, matminer equality.

`mypy --strict` passes on `io`, `core`, `fields`, `operators`, `constants`
and `data`.

## 11. Performance

One core (`OMP_NUM_THREADS=1`), a 16-atom Heusler cell (ScAlAu2) on its 96^3
VASP grid: 8.4 s for every default domain (bonding 7.7 s, of which the
promolecule 2.8 s and the pair regions 2.3 s; structural 2.1 s; heterogeneity
1.2 s; compositional 0.3 s). Cost is linear in the voxel count: about 10 us
per voxel from 64^3 to 120^3. Batch parallelism is per structure
(`featurize_batch(n_workers=...)`). float32 halves memory; against float64
the median relative difference is below 1e-5 and the largest is the
ellipticity spread (2%), whose lambda1/lambda2 - 1 diverges as lambda2 -> 0
(section 6.6).

## 12. Limitations

- The FFT backend rings on cusps: for all-electron (AECCAR) densities use
  `derivative_backend="fd"`. PAW pseudo-densities are smooth enough for FFT.
- The promolecule is point-sampled: a nucleus exactly on a grid point
  over-counts its free-atom cusp (+0.18% of the valence charge at 0.06 A,
  +5% at 0.15 A); `def_charge_mismatch` reports int delta rho per structure.
- Becke weights are truncated to the 60 nearest images (converged to ~1e-2 in
  weight); exact ties at the truncation break symmetry at the 1e-5 level.
- The tabulated free atoms are spherical, non-relativistic LDA; ELF_D and
  the energy densities are gradient expansions derived for all-electron
  densities, so on PAW densities they are meaningful from the bond shell out.
- Within-element variances are NaN for structures with one site per element.
- Native Quantum ESPRESSO HDF5 and ABINIT binary densities are not read;
  export cube or XSF (pp.x, cut3d).

## 13. API

| Function | Purpose |
|---|---|
| `read`, `read_vasp`, `read_all_electron` | VolumetricData from files |
| `read_predicted`, `write_predicted`, `vasp_grid_shape` | densities predicted by an ML model; the grid VASP would use |
| `featurize(vd, ...)` | descriptors (and metadata) of one structure |
| `featurize_batch(paths, ...)` | tidy DataFrame of many structures |
| `catalogue()` | descriptor metadata DataFrame |
| `descriptor_names(domains, extensions)` | the fixed column order |
| `hirshfeld_charges(vd)` | q_i = Z_i - int w_i rho dV |
| `sensitivity_sweep(vd, c1_range, c2_range)` | shell descriptors over cutoffs |
| `validate.convergence.grid_convergence`, `analytic_convergence`, `recommended_spacing` | grid adequacy |
| `validate.elf_fidelity(elf_true, elf_reconstructed)` | Pearson r, MAE, RMSE |
| `validate.invariance.supercell`, `translate`, `rotate` | the invariance transformations |

## 14. Data sources and citations

- Free-atom densities: pydemi's spherical LDA solver (Slater exchange + PW92
  correlation; validated against the NIST LDA atomic reference data,
  Kotochigova et al., Phys. Rev. A 55, 191 (1997)), `tools/`.
- Covalent radii: Cordero et al., Dalton Trans. 2832 (2008), via Magpie
  (BSD licence, `src/pydemi/data/LICENSE-matminer`).
- Magpie features: L. Ward et al., npj Comput. Mater. 2, 16028 (2016);
  matminer, Comput. Mater. Sci. 152, 60 (2018).
- ELF from rho: V. G. Tsirelson, A. Stash, Chem. Phys. Lett. 351, 142 (2002).
- Local energy densities: Yu. A. Abramov, Acta Cryst. A53, 264 (1997).
- Becke partition: A. D. Becke, J. Chem. Phys. 88, 2547 (1988).
- Hirshfeld partition: F. L. Hirshfeld, Theor. Chim. Acta 44, 129 (1977).
- Piecewise-linear critical points: T. Banchoff, Amer. Math. Monthly 77, 475 (1970).

Licence: MIT (`LICENSE`; declared in `pyproject.toml`). Author: **Shubham Maurya**,
CMS Lab, IIT Kanpur.
