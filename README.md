# pydemi

**Physically interpretable descriptors from electronic charge densities.**

pydemi turns the charge-density grids written by plane-wave DFT codes (VASP
`CHGCAR`, `AECCAR`, `ELFCAR`, `LOCPOT`; Gaussian cube; XCrySDen XSF) into
tabular descriptors for materials informatics and machine learning. Everything
is built on one grid engine: a single geometry pass (distance, direction and
index of the nearest, or assigned, nucleus for every voxel) and a small set of
cached fields and derivatives — the density ρ, the magnetization |m|, the
electron localization function (ELF, or its reconstruction ELF_D from ρ), the
electrostatic potential and the deformation density. Each field runs through
the same moment, shell-fraction, anisotropy and partition machinery, so an
extra descriptor costs almost nothing once the engine is built.

212 scalar quantities in 17 families by default (plus 28 opt-in Becke
variants), covering entries 1–107 of the *pydemi Consolidated Descriptor
Reference* (`pydemi — Consolidated Descriptor Reference.pdf`). Quantities
that need an optional input (ELFCAR, LOCPOT, AECCAR, calibration files) are
added when it is present, so a CHGCAR-only structure yields 199. Every
quantity has a registry entry recording its reference number, kind, inputs,
formula and caveats.

**Author:** Shubham Maurya, CMS Lab, IIT Kanpur

---

## Contents

1. [Installation](#1-installation)
2. [Quickstart](#2-quickstart)
3. [Concepts](#3-concepts)
4. [Input data](#4-input-data)
5. [Descriptor reference](#5-descriptor-reference)
6. [Free-atom references and PAW](#6-free-atom-references-and-paw)
7. [Calibrated quantities](#7-calibrated-quantities)
8. [Batch processing and the command line](#8-batch-processing-and-the-command-line)
9. [Grid tools: resampling, convergence, strain](#9-grid-tools-resampling-convergence-strain)
10. [Units and numerical conventions](#10-units-and-numerical-conventions)
11. [Where pydemi departs from the specification](#11-where-pydemi-departs-from-the-specification)
12. [Validation](#12-validation)
13. [Performance](#13-performance)
14. [Limitations](#14-limitations)
15. [API overview](#15-api-overview)
16. [Data sources and citations](#16-data-sources-and-citations)
17. [Author and license](#17-author-and-license)

---

## 1. Installation

Requires Python ≥ 3.10.

```bash
git clone <repository> pydemi
cd pydemi
pip install -e ".[dev]"      # dev adds pytest
pytest                       # optional: run the test suite (~1 min)
```

Dependencies: `numpy`, `scipy`, `spglib`, `pymatgen`. No DFT code or
atomic-structure package is needed at run time: the free-atom reference
densities come from pydemi's own atomic solver (§6), and the Magpie element
tables used by Tier 2 are bundled.

The install provides the `pydemi` command (§8).

## 2. Quickstart

### Python

```python
from pydemi import Engine
from pydemi.descriptors import compute_descriptors, describe

# a VASP run directory: CHGCAR plus whichever of AECCAR0/2, ELFCAR, LOCPOT,
# POTCAR / OUTCAR exist
eng = Engine.from_vasp_dir("runs/NaCl_225")

d = compute_descriptors(eng)                     # every default family -> {name: value}
d["zeta"], d["fint_over_lnf"], d["M_net"]

# a subset of families, physical descriptors only
d = compute_descriptors(eng, families=("tier1", "F5"), kinds=("descriptor",))

describe("lnf_rho")          # entry, family, kind, inputs, formula, caveats
```

### Command line

```bash
pydemi compute data_root/ -o descriptors.csv --workers 8   # every run under data_root, resumable
pydemi convergence runs/NaCl_225                           # which descriptors this grid resolves
pydemi resample runs/NaCl_225/CHGCAR CHGCAR_0.10 --spacing 0.10
pydemi list --family G
pydemi describe perc_anisotropy
```

## 3. Concepts

### 3.1 Engine and fields

`Engine` holds one `Structure` (lattice, species, fractional coordinates) and
its named grid fields. Each `Field` computes its derivatives lazily and caches
them: gradient, gradient norm, packed Hessian (6 components), Laplacian and
Hessian eigenvalues λ₁ ≤ λ₂ ≤ λ₃. Fields read from different files may live
on different grids (ELFCAR is usually on VASP's coarse `NGX` grid, CHGCAR on
`NGXF`), so geometry, shell masks and partitions are cached per grid shape.

| Field name | Content | Source |
|---|---|---|
| `rho` | total density (PAW pseudo-valence + compensation), e/Å³ | CHGCAR block 1 |
| `magnetization` | m = ρ↑ − ρ↓ (collinear) | CHGCAR block 2 |
| `magnetization_abs` | \|m\|, or \|m_vec\| for non-collinear runs | CHGCAR blocks 2(–4) |
| `rho_ae` | all-electron density | AECCAR0 + AECCAR2 |
| `elf`, `elf_down` | ELF (spin blocks) | ELFCAR |
| `potential` | electrostatic potential, eV | LOCPOT |
| `elf_d` | ELF reconstructed from ρ (derived, entry 72) | computed |
| `hartree_potential` | electronic Hartree potential, eV (derived) | computed |
| `promolecule_valence`, `promolecule_total` | superposed free atoms (derived) | computed |

Any array can be analysed: `Engine(structure, {"rho": array})`.

### 3.2 Derivatives

Both derivative methods treat the cell as periodic and use the exact metric
tensor, so they are correct for any cell shape (triclinic included):

- `method="fd"` (default): second-order compact central differences.
- `method="spectral"`: exact derivatives of the trigonometric interpolant (FFT).

The choice matters for sign-thresholded descriptors such as `lnf`; see §10 and
`docs/numerics.md`.

### 3.3 The geometry pass

For every voxel, `engine.geometry()` gives the distance r_k to the nearest
nucleus, the unit vector r̂_k from that nucleus to the voxel, and the atom index
i(k). Periodic images are searched with a KD-tree whose image range is widened
until it provably contains every voxel's nearest atom, so the minimum image is
exact even in strongly sheared cells (fractional rounding is not).

### 3.4 Shells

Three radial shells about the nearest nucleus, with c₁ = 0.8 Å and c₂ = 1.5 Å
by default:

    core:          r_k ≤ c₁
    bond:          c₁ < r_k ≤ c₂
    interstitial:  r_k > c₂

`Shells(c1, c2)` changes the absolute cutoffs; `Shells.scaled(s1, s2, radii)`
uses per-element cutoffs s·R_e. Every shell-based descriptor inherits a
cutoff sensitivity; report it.

### 3.5 Partitions

`engine.partition(scheme=...)` assigns voxels to atoms. Every scheme exposes the
same interface — chunks of (voxel, atom, weight, distance, direction) pairs,
with distance and direction measured to *that* atom — so every site-resolved
quantity comes from one code path.

| Scheme | Rule | Notes |
|---|---|---|
| `nearest` | argmin_i \|r − R_i\| | default; reproduces Tier 1 exactly |
| `power` | argmin_i \|r − R_i\|² − R_i² | radius-weighted Voronoi; Magpie covalent radii by default (entry 99) |
| `hirshfeld` | w_i = ρ_i^free / Σ_j ρ_j^free | smooth; free-atom densities (§6); 6.5 Å cutoff (entry 101) |
| `becke` | Becke fuzzy cells | opt-in; slow convergence in periodic solids, see §11 (entry 100) |

### 3.6 The registry, kinds and names

`pydemi.descriptors.REGISTRY` holds a `DescriptorInfo` for every quantity:
`name`, `entry` (reference number), `family`, `kind`, `inputs`, `formula`,
`note`, `opt_in`. Kinds:

| Kind | Meaning |
|---|---|
| `descriptor` | a physical descriptor as specified |
| `variant` | a recommended fix for a flagged ambiguity, reported *alongside* the specified form |
| `cross_term` | combinatorial pairing without a physical derivation (entries 33–37) |
| `preprocessing` | a regression transform, not a descriptor (entries 38–39) |
| `metadata` | flags and bookkeeping (e.g. `is_spin_polarized`, `n_atoms`) |
| `field`, `site`, `dataset` | non-scalar entries (a grid, a per-atom array, a cross-dataset statistic) |

Two naming rules keep result files unambiguous:

1. **A name always keeps the specified formula.** Where the specification
   recommends a fix, the fix gets a new explicit name (`lnf` → `lnf_rho`,
   `moment_ratio` → `moment_ratio_scale_free`), so a column never changes
   meaning between files.
2. **Where two data sources are possible, the name says which**: `ELF_*` (from
   ELFCAR) vs `ELFD_*` (reconstructed), `V_*` (LOCPOT) vs `VH_*` (Hartree from ρ),
   and a `_power` / `_hirshfeld` / `_becke` suffix for partition variants.

```python
from pydemi.descriptors import names, describe
names(family="G")                          # scalar names of one family, in reference order
names(kinds=("descriptor",))               # physical descriptors only
names(kinds=None, opt_in=True)             # everything, including non-scalar and opt-in entries
```

## 4. Input data

### 4.1 VASP

| File | Read as | Notes |
|---|---|---|
| `CHGCAR` / `CHG` | `rho` (+ magnetization) | ρ·V is divided by V; every data block is read: 1 = non-spin, 2 = collinear, 4 = non-collinear; any other count is refused |
| `AECCAR0` + `AECCAR2` | `rho_ae` | all-electron density; required for Family A |
| `ELFCAR` | `elf` (`elf_down`) | usually on the coarse grid |
| `LOCPOT` | `potential` (eV) | enables the `V_*` descriptors; which potential it is depends on LVHAR / LVTOT |
| `POTCAR` or `OUTCAR` | ZVAL and RCORE per species | POTCAR preferred; OUTCAR's `POMASS …; ZVAL …` and `RCORE` lines otherwise |

The POSCAR-style header is parsed in VASP 5 and VASP 4 layouts (pass
`species=` for VASP 4), with negative scale factors, Cartesian coordinates,
Selective dynamics and POTCAR-style labels (`Fe_pv` → Fe). Augmentation
occupancies and per-atom moment lines between blocks are skipped.

```python
from pydemi.io import read_chgcar, read_aeccar, read_elfcar, read_locpot, read_volumetric
cd = read_chgcar("CHGCAR")          # ChargeDensity: .total, .magnetization, .spin_mode, .structure
eng = Engine.from_charge_density(cd)
```

### 4.2 Other codes

`Engine.from_file(path)` reads `.cube` / `.cub` (Gaussian cube) and `.xsf`
(XCrySDen) densities, which covers:

- **Quantum ESPRESSO**: `pp.x` with `plot_num = 0`, `output_format = 6` (cube) or `5` (XSF);
- **ABINIT**: `cut3d`, cube or XSF output.

Density values are taken to be in e/bohr³ (what these tools write) and
converted; pass `density_unit="e/A^3"` if the file already holds e/Å³, or
`density_unit=None` for non-density data. XSF's repeated end points are
dropped. `write_cube` and `write_xsf` export any field, e.g. ELF_D or a
deformation density for VESTA. Native QE `charge-density.hdf5` and ABINIT
`_DEN` files are not read directly.

### 4.3 Valence electrons and PAW radii

The free-atom reference (§6) needs each element's PAW valence count (ZVAL),
and the non-nuclear-maximum test (entry 94) needs its PAW radius (RCORE).
`Engine.from_vasp_dir` reads both from the POTCAR or, failing that, the
OUTCAR. Without either, ZVAL falls back to a documented rule (electrons
outside the noble-gas core, a filled f¹⁴ counted as core) and the PAW radius
to the covalent radius; the flag `paw_radii_known` records which was used.
Explicit values: `Engine(..., zval={"Fe": 14})`, `eng.paw_radii = {...}`.

## 5. Descriptor reference

Families, in the order `compute_descriptors` computes them. The **Definition**
column is the exact formula stored in the registry; r_k, r̂_k are measured to
the nearest nucleus, ⟨·⟩ is a voxel average unless stated, and shells are as
in §3.4.

| Family | Content | Needs |
|---|---|---|
| `tier1` | charge-density moments, shell fractions, anisotropy, Laplacian (1–15) | CHGCAR |
| `tier2` | compositional baseline, Magpie via matminer (16–30) | structure |
| `tier3` | interaction terms and transforms (32–39) | CHGCAR |
| `B` | ELF descriptors (48–52) and ELF_D fidelity (73) | CHGCAR (ELFCAR for `ELF_*`, 73) |
| `F2` | electrostatic potential (74–76) | CHGCAR (LOCPOT for `V_*`) |
| `F3` | non-covalent interactions (77–79) | CHGCAR |
| `F4` | whole-grid ellipticity (80–82) | CHGCAR |
| `F5` | local energy densities (83–85) | CHGCAR |
| `F6` | information-theoretic measures (86–89) | CHGCAR |
| `I1` | charge anisotropy tensor (102–103) | CHGCAR |
| `C` | site-resolved heterogeneity (53–59) | CHGCAR |
| `E` | spin density (63–71) | spin-polarized CHGCAR |
| `H` | partition variants (99–101) | CHGCAR |
| `G` | topology and connectivity (90–98) | CHGCAR |
| `I2` | bond-midpoint density (104–105) | CHGCAR |
| `A` | deformation density (40–47) | AECCAR0 + AECCAR2 |
| `D` | calibrated ionicity and bulk-modulus baselines (60–62, 106) | calibration files (§7) |

Entry 72 (the ELF_D field) is `pydemi.descriptors.elf_d_field`; entry 107
(strain response) is the utility in §9.3.

### 5.1 Tier 1 — charge-density descriptors

The three ambiguities the specification flags are reported both ways (`lnf` /
`lnf_rho`, `moment_ratio` / `moment_ratio_scale_free`, `zeta_over_rvar` /
`zeta_over_sigma_r`). `lap_concentration` as specified is identically ½ for a
periodic density (the cell integral of a Laplacian vanishes), so the
core-excluded `lap_concentration_valence` is added.

| Entry | Name | Kind | Definition | Notes |
|---|---|---|---|---|
| 1 | `zeta` | descriptor | 1 - sum_k \|grad rho_k . r_hat_k\| / sum_k \|grad rho_k\| |  |
| 2 | `m1` | descriptor | sum_k rho_k r_k / sum_k rho_k |  |
| 3 | `m2` | descriptor | sum_k rho_k r_k^2 / sum_k rho_k |  |
| 4 | `sigma_r2` | descriptor | m2 - m1^2 |  |
| 5 | `f_core` | descriptor | sum_{r_k <= c1} rho_k / sum_k rho_k |  |
| 6 | `f_bond` | descriptor | sum_{c1 < r_k <= c2} rho_k / sum_k rho_k |  |
| 7 | `f_int` | descriptor | sum_{r_k > c2} rho_k / sum_k rho_k |  |
| 8 | `lnf` | descriptor | (1/N) sum_k 1(lap rho_k < 0) | voxel-count fraction; sensitive to the Laplacian numerics |
| 8 | `lnf_rho` | variant | sum_{lap rho < 0} rho_k / sum_k rho_k | charge-weighted lnf recommended by the reference |
| 9 | `fint_over_lnf` | descriptor | f_int / lnf |  |
| 9 | `fint_over_lnf_rho` | variant | f_int / lnf_rho |  |
| 10 | `moment_ratio` | descriptor | m2 / m1 | carries units of length; not scale-invariant |
| 10 | `moment_ratio_scale_free` | variant | m2 / m1^2 | the scale-invariant form the reference recommends |
| 11 | `radial_cv` | descriptor | sqrt(m2 - m1^2) / m1 |  |
| 12 | `zeta_over_rvar` | descriptor | zeta / sigma_r2 |  |
| 12 | `zeta_over_sigma_r` | variant | zeta / sigma_r | inverse length; scales consistently across cell sizes |
| 13 | `charge_per_m1` | descriptor | Q_tot / m1 |  |
| 14 | `lap_concentration` | descriptor | sum_{lap<0} \|lap rho_k\| / sum_k \|lap rho_k\| | identically 1/2 for any periodic density: the cell integral of a Laplacian vanishes, so negative and positive parts balance. Carries no information; kept for completeness |
| 14 | `lap_concentration_valence` | variant | same share over r_k > c1 | pydemi proposal, not in the reference: excluding the core breaks the identity (flux crosses the core boundary) |
| 15 | `bond_int_ratio` | descriptor | f_bond / f_int |  |
|  | `Q_tot` | metadata | sum_k rho_k dV | electron count; a sanity check, and the numerator of charge_per_m1 |

### 5.2 Tier 2 — compositional baseline

Magpie statistics as computed by matminer's `ElementProperty` (values agree
with matminer to 1e-14); no novelty is claimed — cite matminer. Missing table
entries are imputed with the all-element mean, as matminer does. The crystal
system comes from spglib (symprec 0.01).

| Entry | Name | Kind | Definition | Notes |
|---|---|---|---|---|
| 16 | `mean_mass` | descriptor | sum_i x_i A_i |  |
| 17 | `max_mass` | descriptor | max_i A_i |  |
| 18 | `mass_range` | descriptor | max_i A_i - min_i A_i |  |
| 19 | `mean_elneg` | descriptor | sum_i x_i chi_i |  |
| 20 | `elneg_diff` | descriptor | max_i chi_i - min_i chi_i |  |
| 21 | `mean_vec` | descriptor | sum_i x_i VEC_i | Magpie NValence |
| 22 | `max_vec` | descriptor | max_i VEC_i |  |
| 23 | `mean_radius` | descriptor | sum_i x_i R_i | Magpie AtomicRadius, Angstrom |
| 24 | `radius_diff` | descriptor | max_i R_i - min_i R_i |  |
| 25 | `n_elements` | descriptor | count |  |
| 26 | `ionicity` | descriptor | 1 - exp(-(elneg_diff)^2 / 4) | Pauling; never looks at the density |
| 27 | `mean_period` | descriptor | sum_i x_i P_i |  |
| 28 | `crystal_system_int` | descriptor | spglib crystal system, 1 = triclinic ... 7 = cubic |  |
|  | `space_group_number` | metadata | spglib |  |
| 29 | `is_f_block` | metadata | every element is f-block | pipeline metadata, not a descriptor |
| 30 | `has_f_block` | metadata | any element is f-block | pipeline metadata, not a descriptor |

### 5.3 Tier 3 — interaction terms and transforms

Entries 33–37 are `cross_term`s and 38–39 `preprocessing`; report them only if
your own analysis ranks them. `laplacian_std_valence` measures bonding
heterogeneity instead of near-nucleus numerical spikes.

| Entry | Name | Kind | Definition | Notes |
|---|---|---|---|---|
| 32 | `laplacian_std` | descriptor | std of lap rho over all k | dominated by near-nucleus spikes rather than bonding |
| 32 | `laplacian_std_valence` | variant | std of lap rho over r_k > c1 | core-excluded form the reference recommends |
| 33 | `vec_x_lnf` | cross_term | mean_vec * lnf |  |
| 34 | `elneg_x_lnf` | cross_term | mean_elneg * lnf |  |
| 35 | `vec_over_rvar` | cross_term | mean_vec / sigma_r2 |  |
| 36 | `bond_over_lnf` | cross_term | f_bond / lnf |  |
| 37 | `lnf_x_m1` | cross_term | lnf * m1 |  |
| 38 | `sqrt_zeta` | preprocessing | sqrt(zeta) |  |
| 39 | `log_lnf` | preprocessing | ln(lnf) |  |

### 5.4 Family B — ELF descriptors and ELF_D (F1)

Computed on ELF_D (entry 72, Tsirelson–Stash reconstruction from ρ, ∇ρ, ∇²ρ in
atomic units) always, with `ELFD` in the names, and on a real ELFCAR when one
is loaded, with the reference names below. Fractions and averages use the
bonding shell, since core shells show ELF ≈ 1 from shell structure; `zeta_ELF`
excludes the core shell. With an ELFCAR, entry 73 compares ELF_D and ELF on
the ELFCAR grid.

| Entry | Name | Kind | Definition | Notes |
|---|---|---|---|---|
| 72 | `elf_d` | field | [1 + (D_P / C_F rho^{5/3})^2]^{-1}, Kirzhnits t_P (Tsirelson-Stash) | registered on the engine as field 'elf_d' |

| Entry | Name | Kind | Definition | Notes |
|---|---|---|---|---|
| 48 | `f_ELF_localized` | descriptor | (1/N_bond) sum_bond 1(ELF > 0.5) | true ELF from ELFCAR; present only when one is loaded |
| 49 | `ELF_bond_avg` | descriptor | mean ELF over the bonding shell | true ELF from ELFCAR; present only when one is loaded |
| 50 | `zeta_ELF` | descriptor | 1 - sum \|grad ELF . r_hat\| / sum \|grad ELF\| over r_k > c1 | true ELF from ELFCAR; present only when one is loaded; core shell excluded |
| 51 | `ELF_threshold_sweep_025` | descriptor | f_ELF(t = 0.25) over the bonding shell | true ELF from ELFCAR; present only when one is loaded |
| 51 | `ELF_threshold_sweep_075` | descriptor | f_ELF(t = 0.75) over the bonding shell | true ELF from ELFCAR; present only when one is loaded |
| 51 | `ELF_threshold_sweep_inflection` | descriptor | t of steepest descent of f_ELF(t) (modal ELF) | true ELF from ELFCAR; present only when one is loaded |
| 52 | `ELF_core_valence_contrast` | descriptor | <ELF>_core / <ELF>_bond | true ELF from ELFCAR; present only when one is loaded |
| 73 | `ELFD_fidelity_r` | descriptor | Pearson r(ELF_D, ELF) over ELFCAR voxels | present only with an ELFCAR |
| 73 | `ELFD_fidelity_mae` | descriptor | mean \|ELF_D - ELF\| |  |
| 73 | `ELFD_fidelity_r_bond` | variant | Pearson r over the bonding shell | where the comparison is meaningful |
| 73 | `ELFD_fidelity_mae_bond` | variant | mean \|ELF_D - ELF\| over the bonding shell |  |

Each `ELF_*` name has an `ELFD_*` twin (`f_ELFD_localized`, `ELFD_bond_avg`,
`zeta_ELFD`, …).

### 5.5 F2 — electrostatic potential

`VH_*` uses the electronic Hartree potential from one FFT Poisson solve
(V_H(G) = 4πρ(G)/|G|², in eV; no ionic term). `V_*` uses a LOCPOT when loaded.
Site values are exact Fourier interpolations at the nuclei
(`pydemi.descriptors.site_potentials`). `*_int_min` is relative to the
cell-average potential.

| Entry | Name | Kind | Definition | Notes |
|---|---|---|---|---|
| 74 | `VH_site` | site | V_H(R_i), Fourier-interpolated | pydemi.descriptors.potential.site_potentials |
| 75 | `VH_spread` | descriptor | std over i of V_H(R_i), eV | electronic Hartree potential only (no ionic term) |
| 76 | `VH_int_min` | descriptor | min over r_k > c2 of V_H, eV | relative to the cell-average potential |
| 74 | `V_site` | site | V(R_i) from LOCPOT |  |
| 75 | `V_spread` | descriptor | std over i of V(R_i), eV | LOCPOT; present only when one is loaded |
| 76 | `V_int_min` | descriptor | min over r_k > c2 of V, eV | LOCPOT; relative to the cell average |

### 5.6 F3 — non-covalent interactions

Reduced density gradient s = |∇ρ| / (2(3π²)^{1/3} ρ^{4/3}) (dimensionless), with
the density threshold in atomic units.

| Entry | Name | Kind | Definition | Notes |
|---|---|---|---|---|
| 77 | `f_NCI` | descriptor | (1/N) sum 1(s < 0.5 and rho < 0.05 a.u.) |  |
| 78 | `NCI_attractive` | descriptor | fraction of NCI voxels with lambda2 < 0 |  |
| 79 | `sign_lambda2_rho_mean` | descriptor | mean of sign(lambda2) rho over NCI voxels, e/bohr^3 |  |

### 5.7 F4 — whole-grid ellipticity

Over bonding-shell voxels where the density is concentrated in two directions
(λ₂ < 0). Near bond paths λ₂ → 0⁻ makes the ratio diverge, so the mean can be
dominated by a few voxels; `ellip_bond_median` is the robust companion. Entry
82 is a dataset-level correlation: `pydemi.descriptors.dataset.zeta_ellip_agreement`.

| Entry | Name | Kind | Definition | Notes |
|---|---|---|---|---|
| 80 | `ellip_bond_avg` | descriptor | <lambda1/lambda2 - 1> over bond voxels with lambda2 < 0 | can be dominated by voxels with lambda2 -> 0^- |
| 80 | `ellip_bond_median` | variant | median of the same set | pydemi addition: robust companion to the mean |
| 81 | `ellip_bond_std` | descriptor | std of the same set |  |
| 82 | `zeta_ellip_agreement` | dataset | correlation of zeta with ellip_bond_avg across the dataset | pydemi.descriptors.dataset.zeta_ellip_agreement |

### 5.8 F5 — local energy densities

Abramov kinetic density g (the same Kirzhnits expression behind ELF_D),
local-virial potential density v = ¼∇²ρ − 2g and total H = g + v, in atomic
units, averaged over the bonding shell.

| Entry | Name | Kind | Definition | Notes |
|---|---|---|---|---|
| 83 | `f_H_negative` | descriptor | (1/N_bond) sum_bond 1(H < 0), H = lap/4 - g (a.u.) |  |
| 84 | `H_bond_mean` | descriptor | <H> over the bonding shell, hartree/bohr^3 |  |
| 85 | `G_over_rho` | descriptor | <g/rho> over the bonding shell, hartree/electron |  |

### 5.9 F6 — information-theoretic measures

On the shape function ρ̃ = ρ/N_e, in bohr units as the specification defines
them; C_LMC is unit-free.

| Entry | Name | Kind | Definition | Notes |
|---|---|---|---|---|
| 86 | `shannon_entropy` | descriptor | -int rho~ ln rho~ dV, rho~ = rho/N_e, bohr units |  |
| 87 | `fisher_information` | descriptor | int \|grad rho~\|^2 / rho~ dV, 1/bohr^2 |  |
| 88 | `disequilibrium` | descriptor | int rho~^2 dV, 1/bohr^3 |  |
| 89 | `LMC_complexity` | descriptor | D e^S (unit-free) |  |

### 5.10 I1 — charge anisotropy tensor

T_ab = Σ ∂_aρ ∂_bρ / Σ|∇ρ|² (trace 1). Isotropic → eigenvalues ⅓; a density
varying along one direction only → `charge_FA` = 1.

| Entry | Name | Kind | Definition | Notes |
|---|---|---|---|---|
| 102 | `T_eig_1` | descriptor | largest eigenvalue of T_ab = sum d_a rho d_b rho / sum \|grad rho\|^2 |  |
| 102 | `T_eig_2` | descriptor | middle eigenvalue of T |  |
| 102 | `T_eig_3` | descriptor | smallest eigenvalue of T |  |
| 103 | `charge_FA` | descriptor | sqrt(3/2) \|\|T - I/3\|\|_F / \|\|T\|\|_F |  |

### 5.11 Family C — site-resolved heterogeneity

For X ∈ {`m1`, `f_bond`, `zeta`}, the site value X^(i) restricts every sum to
atom i's voxels (nearest-atom partition), and seven statistics are reported:

| Entry | Name | Definition |
|---|---|---|
| 53–55 | `X_site_std` | population std over sites of X^(i) |
| 56 | `X_site_range` | max − min over sites |
| 57 | `X_site_max`, `X_site_min` | extreme sites |
| 58 | `X_var_within` | Σ_e w_e Var_{i∈e} X^(i) (w_e = atom fraction): configurational disorder |
| 59 | `X_var_between` | Σ_e w_e (X̄_e − X̄)²: chemical differentiation |
| 58/59 | `X_within_share` | within / (within + between), the single ANOVA ratio |
| — | `n_atoms` | report with every site statistic |

The within and between terms partition the total site variance exactly.
`X_var_within` and `X_within_share` are NaN when no element has two or more
sites (a 0 would falsely read as "no disorder").

### 5.12 Family E — spin density

From the CHGCAR magnetization block, in μ_B and Å. Non-collinear runs are
treated as vectors (|m_vec|, vector sums of the site moments), so
perpendicular moments are not misread as collinear cancellation.
Non-spin-polarized runs return 0 for every entry with `is_spin_polarized = 0`;
spin-polarized runs below 0.01 μ_B per atom keep M_abs, M_net and
`mu_site_std` but return 0 for the ratio-type entries, with `is_magnetic = 0`.
Per-site moments: `pydemi.descriptors.site_moments`.

| Entry | Name | Kind | Definition | Notes |
|---|---|---|---|---|
| 63 | `M_abs` | descriptor | sum \|m\| dV, mu_B |  |
| 64 | `M_net` | descriptor | \|sum m dV\|, mu_B | vector sum for non-collinear runs |
| 65 | `m1_spin` | descriptor | sum \|m\| r / sum \|m\| | 0 when not magnetic |
| 66 | `sigma_r2_spin` | descriptor | sum \|m\| r^2 / sum \|m\| - m1_spin^2 | 0 when not magnetic |
| 67 | `f_bond_spin` | descriptor | sum_bond \|m\| / sum \|m\| | 0 when not magnetic |
| 68 | `mu_site` | site | sum_(k in i) m dV | pydemi.descriptors.spin.site_moments |
| 69 | `mu_site_std` | descriptor | sqrt(mean_i \|mu_i - mean mu\|^2) | signed std when collinear |
| 70 | `spin_frustration` | descriptor | 1 - \|sum mu_i\| / sum \|mu_i\| | vector norms; 0 when not magnetic |
| 71 | `spin_charge_correlation` | descriptor | Pearson r(rho_k, \|m_k\|) | 0 when not magnetic |
|  | `is_spin_polarized` | metadata | 1 if the CHGCAR has magnetization blocks | every Family E value is the sentinel 0 when this is 0 |
|  | `is_magnetic` | metadata | 1 if M_abs > 0.01 mu_B per atom |  |

### 5.13 Family H — partition variants

The Tier 1 radial descriptors (`m1`, `m2`, `sigma_r2`, `f_core`, `f_bond`,
`f_int`, `zeta`) and all Family C statistics, recomputed under the power
diagram and Hirshfeld partitions, with suffixes `_power` and `_hirshfeld`
(e.g. `m1_power`, `f_bond_site_std_hirshfeld`). Becke (`_becke`) is opt-in:
`partition_family(eng, schemes=("power", "hirshfeld", "becke"))`. Comparing a
variant with its nearest-atom value is the partition-sensitivity axis.
Hirshfeld charges q_i = N_i − ∫w_iρ dV: `pydemi.descriptors.hirshfeld_charges`.

### 5.14 Family G — topology and connectivity

All by direct grid operations that always terminate.

- **Percolation (90–91):** the largest level c at which {ρ > c} still contains
  a cluster that wraps around the periodic cell along each lattice direction
  (the density at the bottleneck of the best connecting path), found by exact
  bisection over voxel values; wrapping is detected with a union-find that
  tracks winding vectors.
- **Critical-point census (92–93):** piecewise-linear Morse theory on the
  Freudenthal triangulation (14-neighbour link), ties broken by voxel index.
  The Euler sum is then identically 0 on any grid; `euler_consistency` is a
  self-check, and `pydemi convergence` is the grid-adequacy test.
- **Non-nuclear maxima (94–95):** maxima farther than max(c₁, R_PAW) from their
  nearest nucleus, with steepest-ascent basins for their charge. The per-element
  cutoff matters for PAW data: a pseudized atom can have no maximum at its
  nucleus, only lobes on a shell inside its PAW radius.
- **Ripple:** the raw counts are sensitive to low-amplitude ripple.
  `n_NNM_significant` ignores maxima whose basin holds < 0.01 e, which is not
  enough in nearly-free-electron metals (a flat valence sea splits into many
  basins holding more than 0.01 e each). `n_NNM_persistent` / `Q_NNM_persistent`
  use exact 0-dimensional topological persistence: a maximum is kept when the
  density dips by at least 10% of its height before its region merges with a
  higher maximum's, and the charge of discarded ripple basins passes to the
  maximum that absorbs them. On a 6,059-structure VASP dataset this removes
  every ripple maximum in YMg₃ (124 "significant" → 0 persistent).
- **Interstitial floor (96–98):** PAW pseudo-densities dip below zero inside
  the augmentation spheres (61% of the CHGCARs in the same dataset), so the
  plain `rho_min` / `rho_min_ratio` are usually PAW artefacts.
  `rho_min_int` / `rho_min_int_ratio` take the minimum over voxels farther
  than max(c₂, R_PAW) from their nucleus — the metallicity criterion the
  specification intends.

| Entry | Name | Kind | Definition | Notes |
|---|---|---|---|---|
| 90 | `rho_perc_a` | descriptor | sup{c : {rho > c} has a cluster wrapping along a}, e/A^3 | the reference's min{...} read as the supremum it describes |
| 90 | `rho_perc_b` | descriptor | same along b |  |
| 90 | `rho_perc_c` | descriptor | same along c |  |
| 91 | `perc_anisotropy` | descriptor | (max - min) / mean of rho_perc_a/b/c |  |
| 92 | `n_max` | descriptor | PL maxima (Freudenthal 14-neighbour link) | all census counts are sensitive to ripple in near-flat low-density regions |
| 92 | `n_min` | descriptor | PL minima |  |
| 92 | `n_saddle1` | descriptor | index-1 saddles, sum of (components(lower link) - 1) |  |
| 92 | `n_saddle2` | descriptor | index-2 saddles, sum of (components(upper link) - 1) |  |
| 93 | `euler_consistency` | metadata | n_max - n_saddle2 + n_saddle1 - n_min | identically 0 for a consistent PL census (Banchoff): an implementation self-check, not a grid-adequacy flag |
| 94 | `n_NNM` | descriptor | maxima farther than max(c1, R_PAW) from their nearest nucleus | R_PAW = RCORE from POTCAR/OUTCAR, else covalent radius; counts every ripple maximum, see n_NNM_significant |
| 94 | `n_NNM_significant` | variant | non-nuclear maxima whose basin holds >= 0.01 e | pydemi addition: robust to low-amplitude ripple |
| 94 | `n_NNM_persistent` | variant | non-nuclear maxima with relative persistence (peak - merge) / peak >= 0.1 | pydemi addition: removes ripple, including flat free-electron seas |
| 95 | `Q_NNM` | descriptor | charge in the steepest-ascent basins of the non-nuclear maxima |  |
| 95 | `Q_NNM_persistent` | variant | charge of the persistent non-nuclear maxima, ripple basins merged in | pydemi addition |
|  | `paw_radii_known` | metadata | 1 if the non-nuclear-maximum cutoffs used PAW RCORE values |  |
| 96 | `rho_min` | descriptor | min_k rho_k, e/A^3 | can be negative for PAW pseudo-densities |
| 96 | `rho_min_int` | variant | min rho over r_k > max(c2, R_PAW), e/A^3 | pydemi addition: the interstitial floor, outside every PAW sphere; NaN when that region is empty |
| 97 | `rho_min_ratio` | descriptor | rho_min / <rho>_V | on PAW CHGCARs usually set by negative pseudo-density near a nucleus |
| 97 | `rho_min_int_ratio` | variant | rho_min_int / <rho>_V | pydemi addition: the metallicity criterion the reference intends |
| 98 | `rho_int_mean` | descriptor | <rho_k> over r_k > c2 (volume mean), e/A^3 |  |

### 5.15 I2 — bond-midpoint density

Per-atom first-shell bond census (d ≤ 1.1 d_min for each atom, each pair once,
periodic images included) and exact Fourier interpolation of ρ at the bond
midpoints.

| Entry | Name | Kind | Definition | Notes |
|---|---|---|---|---|
| 104 | `rho_mid_mean` | descriptor | <rho(midpoint)> over first-shell bonds, e/A^3 | first shell: d <= 1.1 d_i per atom |
| 105 | `rho_mid_std` | descriptor | std of rho(midpoint) over the bonds |  |
|  | `n_bonds` | metadata | first-shell bond count |  |
|  | `bond_length_mean` | metadata | <d> over first-shell bonds, Angstrom | the explicit nearest-neighbour convention for the Cohen / rho-based B0 |

### 5.16 Family A — deformation density

Δρ = ρ − Σ_i ρ^free_{e(i)}(|r − R_i|), with the promolecule built in reciprocal
space (exact periodic sum, integral exactly Σ N_i). By default computed from
AECCAR0 + AECCAR2 against all-electron free atoms and NaN without AECCARs;
see §6.2 for why CHGCAR needs a different reference.
`deformation_family(eng, field="rho")` requests the CHGCAR route explicitly.

| Entry | Name | Kind | Definition | Notes |
|---|---|---|---|---|
| 40 | `m1_def` | descriptor | sum \|drho\| r / sum \|drho\| | AECCAR route; NaN without AECCARs |
| 41 | `m2_def` | descriptor | sum \|drho\| r^2 / sum \|drho\| |  |
| 42 | `sigma_r2_def` | descriptor | m2_def - m1_def^2 |  |
| 43 | `f_bond_def` | descriptor | sum_(bond, drho>0) drho / sum_(drho>0) drho |  |
| 44 | `f_int_def` | descriptor | sum_(int, drho>0) drho / sum_(drho>0) drho |  |
| 45 | `f_bond_dep` | descriptor | sum_(bond, drho<0) \|drho\| / sum_(drho<0) \|drho\| |  |
| 46 | `bond_charge_transfer` | descriptor | int_bond drho dV, electrons |  |
| 47 | `def_polarity` | descriptor | int \|drho\| dV / Q_tot |  |
|  | `def_charge_mismatch` | metadata | int drho dV, electrons | should be ~0; large means wrong reference counts |
|  | `def_all_electron` | metadata | 1 if AECCAR0 + AECCAR2 was the field |  |

### 5.17 Family D — calibrated quantities

NaN unless calibration files are passed (§7), except Cohen's formula, which
needs only the composition and the bond length.

| Entry | Name | Kind | Definition | Notes |
|---|---|---|---|---|
| 60 | `grid_ionicity` | descriptor | sigmoid(b . z(features)), fitted to Phillips f_i | NaN without an IonicityCalibration; extrapolation outside tetrahedral compounds |
| 61 | `cohen_B0_predicted` | descriptor | (1971 - 220 lambda) d^-3.5 GPa, lambda = Cohen's class 0/1/2 | a baseline, not a training feature; formula extrapolation when cohen_in_scope = 0 |
|  | `cohen_in_scope` | metadata | 1 for group-IV, III-V and II-VI (1:1) compounds |  |
| 62 | `ionicity_residual` | descriptor | grid_ionicity - ionicity (Pauling) |  |
| 106 | `B0_rho_proxy` | descriptor | a (rho_mid_mean / bond_length_mean^3)^b, fitted to known B0 | NaN without a BulkModulusCalibration |

## 6. Free-atom references and PAW

### 6.1 The atomic solver

Family A and the Hirshfeld partition need spherical free-atom densities.
`pydemi.atoms` solves the spherical, non-spin-polarized Kohn–Sham equations
(LDA: Slater exchange with PW92 correlation, or VWN5) on a logarithmic grid
for every element, with ground-state configurations from pymatgen. Densities
are resolved by orbital, so a valence subset of ZVAL electrons can be taken.
Results are cached in memory and on disk in `$PYDEMI_CACHE_DIR` (default
`~/.cache/pydemi`, ~0.6 MB per element; set it to `""` to disable). All 87
elements of a large intermetallics dataset solve in ~16 s on 16 cores.

The solver is non-relativistic, so valence shapes of 5d/6s/6p elements miss
the relativistic contraction; this is adequate for a promolecule or Hirshfeld
reference.

### 6.2 Which reference for which density

A reference must describe the same electrons, with the same shape, as the
density it is compared with:

- **AECCAR0 + AECCAR2 with all-electron free atoms** is consistent, and is
  what Family A uses.
- **CHGCAR with all-electron free-atom valence** is not: CHGCAR is PAW-pseudized
  inside each atom's augmentation sphere. On fcc FeCoNiCr about 3 e per Fe
  sit 0.8–1.5 Å from the nucleus in CHGCAR that the free atom holds inside
  0.8 Å, so the difference measures the POTCAR, not bonding. The same bias
  enters Hirshfeld *charges* from CHGCAR, which therefore warn (the partition
  itself remains a valid smooth partition).
- **CHGCAR with isolated-atom CHGCARs** run with the same POTCARs is
  consistent — the pseudization cancels:

```python
from pydemi.atoms.reference import IsolatedAtomReference
ref = IsolatedAtomReference({"Fe": "atoms/Fe/CHGCAR", "Ni": "atoms/Ni/CHGCAR"},
                            aeccars={"Fe": ("atoms/Fe/AECCAR0", "atoms/Fe/AECCAR2")})
eng = Engine.from_vasp_dir("runs/FeNi", reference=ref)
```

Hirshfeld weights are always built from the same electrons as the partitioned
field (valence for CHGCAR, total for AECCAR), so a pure free-atom
superposition gets zero charges. `def_charge_mismatch` (∫Δρ dV) flags wrong
electron counts.

## 7. Calibrated quantities

### 7.1 Phillips ionicity table

`pydemi/data/phillips_ionicity.csv` holds 67 Phillips spectroscopic
ionicities f_i (tetrahedral semiconductors, Cu/Ag halides, rock-salt halides
and oxides) with structure type, Cohen class and provenance. Rows are marked
`verified` (matching a secondary table that cites Phillips; the source is
given) or `recalled` (compiled from Phillips 1970 and not yet checked against
a source — verify before publication). GaP carries a note: 0.327 and 0.374
both appear in the literature.

```python
from pydemi.calibration import phillips_table
table = phillips_table()                     # formula -> {f_i, structure, cohen_lambda, status, source}
checked = phillips_table(verified_only=True)
```

### 7.2 grid_ionicity (entry 60)

A sigmoid of standardized grid descriptors fitted to Phillips f_i on reference
compounds you have computed. The default features pair `fint_over_lnf` with the
site-potential spread (use `V_spread` from a LOCPOT when available), so the
calibration does not rest on a single descriptor. Leave-one-out RMSE is
reported with every fit.

```python
from pydemi.calibration import fit_ionicity
rows = {"GaAs": compute_descriptors(Engine.from_vasp_dir("ref/GaAs")),
        "NaCl": compute_descriptors(Engine.from_vasp_dir("ref/NaCl")), ...}
cal = fit_ionicity(rows, features=("fint_over_lnf", "V_spread"))
print(cal.n, cal.rmse, cal.loocv_rmse, cal.n_tetrahedral)
cal.save("ionicity.json")
```

Phillips' scale was derived for tetrahedral semiconductors; outside that class
the prediction is an extrapolation.

### 7.3 Cohen bulk modulus (entry 61)

B₀ = (1971 − 220λ)·d^−3.5 GPa (Cohen, Phys. Rev. B 32, 7988 (1985)), with d the
mean first-shell bond length and λ Cohen's integer class: 0 for group IV, 1
for III–V, 2 for II–VI (1:1 compounds). `cohen_in_scope` is 0 elsewhere, where
λ = 0 is used and the value is a formula extrapolation, not a physics baseline.

### 7.4 ρ-based bulk-modulus proxy (entry 106)

B₀ = a·x^b with x = `rho_mid_mean` / `bond_length_mean`³, fitted in log space
on structures with known bulk moduli:

```python
from pydemi.calibration import fit_bulk_modulus
bulk = fit_bulk_modulus(descriptor_rows, known_B0_GPa)
bulk.save("bulk.json")
```

### 7.5 Using calibrations

```python
from pydemi.calibration import IonicityCalibration, BulkModulusCalibration
cal = {"ionicity": IonicityCalibration.load("ionicity.json"),
       "bulk": BulkModulusCalibration.load("bulk.json")}
d = compute_descriptors(eng, calibrations=cal)
```

or `pydemi compute ... --ionicity-cal ionicity.json --bulk-cal bulk.json`.

## 8. Batch processing and the command line

### 8.1 `pydemi compute`

```bash
pydemi compute INPUT [INPUT ...] -o descriptors.csv [--families F ...] [--workers N]
               [--no-resume] [--ionicity-cal FILE] [--bulk-cal FILE] [-q]
```

Each input is a run directory, a density file (CHGCAR, `.cube`, `.xsf`) or a
root directory, which is searched recursively for run directories containing a
CHGCAR. Rows are appended to the CSV as they finish, so an interrupted run is
resumed by repeating the command (IDs already in the CSV are skipped; use
`--no-resume` to start over). A failure is recorded in the `error` column
instead of stopping the batch; the exit code is 1 if any input failed.

Output columns: `material_id` (directory name or file stem), `path`, `error`,
then every scalar descriptor of the requested families in reference order.
Unavailable quantities (e.g. `ELF_*` without an ELFCAR, Family A without
AECCARs, Family D without calibrations) are written as `nan`.

On shared machines, limit per-process threads and priority:

```bash
nice -n 10 env OMP_NUM_THREADS=1 pydemi compute data/ -o out.csv --workers 24
```

The same from Python:

```python
from pydemi.batch import run_batch, find_runs
run_batch(find_runs("data/"), "out.csv", families=("tier1", "tier2", "E"), workers=8)
```

### 8.2 Other commands

| Command | Purpose |
|---|---|
| `pydemi list [--family F] [--kind K ...] [--all-kinds]` | registered descriptors with entry, family and kind |
| `pydemi describe NAME` | formula, inputs and caveats of one descriptor |
| `pydemi convergence INPUT [--factors 1 0.8 0.6] [--rtol 0.02]` | per-descriptor grid-convergence report (§9.2) |
| `pydemi resample IN OUT --spacing H` | Fourier-resample a density to grid spacing H (Å) |

## 9. Grid tools: resampling, convergence, strain

### 9.1 Resampling

Band-limited Fourier resampling (exact for FFT-grid densities, preserves the
cell integral), so structures with different FFT meshes can be compared at a
common spacing:

```python
from pydemi.resample import resample_engine, shape_for_spacing, fourier_resample
eng_010 = resample_engine(eng, spacing=0.10)     # FFT-friendly shapes, all fields
eng_half = resample_engine(eng, scale=0.5)
```

### 9.2 Grid-convergence report

Recomputes descriptors on Fourier-coarsened copies of the grid and flags those
that change by more than `rtol` between the native grid and the next-coarser
one — a per-structure answer to "does this grid resolve what I measure?".

```python
from pydemi.convergence import convergence_report, format_report
rep = convergence_report(eng, factors=(1.0, 0.8, 0.6), rtol=0.02)
print(format_report(rep))
```

On a 48³ fcc FeCoNiCr grid, 26 of 97 descriptors move by more than 2% at 80%
resolution, including `lnf` (10%) and the critical-point counts; run the
report on representative production grids before relying on Laplacian- or
ELF-based descriptors.

### 9.3 Strain response (entry 107)

From SCF runs at ±ε (a validation experiment on a subset, not a production
feature). The comparison uses the charge per voxel in fractional coordinates,
so a uniform dilation cancels exactly:

```python
from pydemi.strain import strain_response
strain_response("eps-0.01/CHGCAR", "eps+0.01/CHGCAR", eps=0.01)
# {'drho_deps': ..., 'drho_deps_l2': ..., 'charge_drift': ...}
```

## 10. Units and numerical conventions

| Quantity | Unit |
|---|---|
| lengths, distances, moments m1 | Å (m2: Å²) |
| ρ, ρ_mid, percolation levels, `rho_min` | e/Å³ |
| charges, `Q_tot`, `Q_NNM`, `bond_charge_transfer` | electrons |
| magnetization quantities | μ_B |
| potentials (`V_*`, `VH_*`) | eV |
| ELF_D inputs, F5 energy densities, NCI thresholds, F6 | atomic units (hartree, bohr), as specified |
| bulk moduli | GPa |

- Zero denominators give NaN; no epsilon is added.
- Derivatives are periodic; `fd` is the default (§3.2). The spectral
  Laplacian has an absolute round-off floor, so voxel-count `lnf` is
  unreliable with `spectral` in near-empty regions (vacuum, voids); use `fd`
  or `lnf_rho` there.
- A double-width, non-periodic Laplacian stencil (a first-derivative routine
  applied twice) gives `lnf` up to ~25% below the converged value; measured
  comparisons are in `docs/numerics.md`.
- PAW pseudo-densities can be negative near nuclei; `rho_min` may be negative,
  and negative values are clipped to 0 only where a formula requires a
  positive density (ELF_D, F3, F5, F6).

## 11. Where pydemi departs from the specification

Each departure is documented at the point of use and in the registry notes.

| Entry | Specification | pydemi | Reason |
|---|---|---|---|
| 14 `lap_concentration` | Σ_{∇²ρ<0}\|∇²ρ\| / Σ\|∇²ρ\| | kept, plus `lap_concentration_valence` | identically ½ for any periodic density |
| 58 | plain mean over elements | atom-fraction weights | makes within + between add up exactly |
| 61 Cohen | λ = grid_ionicity, constant 1972 | Cohen's class λ = 0/1/2, 1971 | Cohen's λ is an integer class, not a continuous ionicity |
| 90 percolation | min{c : …} | supremum | the literal minimum is always the lowest density |
| 93 Euler check | grid-adequacy flag | implementation self-check | identically 0 for a consistent PL census |
| 94 non-nuclear maxima | fixed cutoff | per-element max(c₁, R_PAW); plus persistence-filtered `n_NNM_persistent` | PAW atoms can have maxima only on a shell inside R_PAW; raw counts pick up ripple |
| 96–97 density floor | min over the cell | kept, plus `rho_min_int` over r > max(c₂, R_PAW) | PAW pseudo-density is negative near many nuclei |
| 40–47 deformation | CHGCAR or AECCAR | AECCAR (NaN otherwise) | CHGCAR minus all-electron free atoms measures pseudization |
| 100 Becke | default smooth partition | opt-in; Hirshfeld is the default | weights converge only algebraically in periodic solids (max error ~2e-2 with 60 neighbours, ~1e-3 with 300); Hirshfeld converges exponentially (~1e-5 at 6.5 Å) |

Robust companions added: `lnf_rho`, `moment_ratio_scale_free`,
`zeta_over_sigma_r`, `laplacian_std_valence`, `lap_concentration_valence`,
`ellip_bond_median`, `n_NNM_significant`, `n_NNM_persistent`, `Q_NNM_persistent`,
`rho_min_int`, `rho_min_int_ratio`.

## 12. Validation

164 tests (`pytest`), organized by what they check against:

- **Closed forms.** Slater 1s and Gaussian superpositions with exact values
  (`pydemi.testing.analytic`): moments, shell fractions, Laplacian sign
  structure, Hessian eigenvalues, lnf, ζ = 0 for spherical atoms, the Hartree
  potential of a Gaussian with neutralizing background (to 0.2%), Shannon /
  Fisher / disequilibrium measures, bond-midpoint density, the Becke
  two-atom weight.
- **Exact identities.** ∫∇²ρ = 0 (lap_concentration = ½), Euler sum = 0 on
  random fields, maxima persistence against an independent voxel-level
  union-find, variance decomposition within + between = total,
  partitions of unity, charge conservation in every partition.
- **Numerics.** Spectral derivatives exact to ~1e-12 on triclinic cells; FD
  errors fall 4× per halving of the grid spacing; periodic boundaries;
  minimum image against brute force in sheared cells; power diagram against
  brute force.
- **Units.** Every atomic-unit quantity against an independent atomic-unit
  evaluation; the uniform electron gas (ELF_D = ½, H = −C_F ρ^{5/3}).
- **Reference data.** The atomic solver against hydrogenic eigenvalues and
  the NIST LDA atomic reference data (He, Ne total energies to ~5e-5 Ha);
  Tier 2 against matminer (1e-14); Cohen's formula against the moduli of Si
  and GaAs.
- **Scale behaviour.** The recommended ratio variants have the scaling the
  specification asks for under a uniform scaling of the system.
- **I/O round trips.** CHGCAR (1, 2 and 4 blocks, augmentation lines, VASP 4
  headers), ELFCAR, AECCAR, cube, XSF, POTCAR / OUTCAR ZVAL and RCORE.

## 13. Performance

Measured on a shared 128-core server:

| Workload | Time |
|---|---|
| 180³ grid, 32 atoms, all default families except H | ~13 s, 2.1 GB peak |
| Hirshfeld family at 180³, 32 atoms | ~34 s |
| 48³ grid, 4 atoms, all default families | ~3 s |
| 6,059-structure VASP dataset (80–160³ grids), 24 workers | ~56 structures/min |

The dominant costs are the Hessian eigendecomposition (F3/F4) and the
Hirshfeld partition; drop families you do not need with `--families`.

## 14. Limitations

- Family A needs AECCAR0 + AECCAR2 (or `IsolatedAtomReference`); without them
  it returns NaN.
- Family D needs calibration files fitted on reference compounds you have
  computed; the literature table is supplied, the densities are not.
- Half of the Phillips table (`recalled` rows) has not been checked against a
  source.
- The atomic solver is non-relativistic and spherical, non-spin-polarized LDA.
- ELF_D, energy densities and NCI are gradient-expansion approximations
  derived for all-electron densities; applied to PAW densities they are
  meaningful in the bonding shell and beyond, not in the core.
- Critical-point counts are sensitive to low-amplitude ripple in near-flat
  regions; prefer the persistence-filtered `n_NNM_persistent` /
  `Q_NNM_persistent`, and check with `pydemi convergence`.
- On PAW CHGCARs the plain density floor (`rho_min`, `rho_min_ratio`) is
  usually set by negative pseudo-density near a nucleus; use `rho_min_int`.
- Small cells: site statistics over a handful of atoms are noisy; report
  `n_atoms`.
- Native Quantum ESPRESSO HDF5 and ABINIT binary density files are not read.

## 15. API overview

| Module | Main contents |
|---|---|
| `pydemi` | `Engine`, `Field`, `Grid`, `Structure`, `Shells`, `nearest_atom`, `assign_atoms`, partitions, readers |
| `pydemi.engine` | `Engine.from_vasp_dir / from_chgcar / from_file / from_charge_density`, `geometry`, `shell_masks`, `partition`, `valence`, `reference` |
| `pydemi.grid` | periodic `gradient`, `hessian`, `laplacian` (`fd` / `spectral`), `hessian_eigenvalues` |
| `pydemi.geometry` | `nearest_atom`, `assign_atoms` (nearest / power diagram) |
| `pydemi.partition` | `HardPartition`, `HirshfeldPartition`, `BeckePartition`, `PairChunk` |
| `pydemi.shells` | `Shells`, `ShellMasks` |
| `pydemi.io` | `read_chgcar`, `read_aeccar`, `read_elfcar`, `read_locpot`, `read_volumetric`, `write_volumetric`; `io.grids`: `read_cube`, `read_xsf`, `write_cube`, `write_xsf`, `read_density` |
| `pydemi.descriptors` | `compute_descriptors`, `describe`, `names`, `REGISTRY`, one function per family (`tier1`, `tier2`, `tier3`, `elf_family`, `potential_family`, `nci_family`, `ellipticity_family`, `energy_family`, `information_family`, `anisotropy_family`, `site_family`, `spin_family`, `partition_family`, `topology_family`, `bond_family`, `deformation_family`), per-site helpers (`site_charges`, `hirshfeld_charges`, `site_moments`, `site_potentials`), fields (`elf_d_field`, `promolecule`) |
| `pydemi.descriptors.dataset` | `zeta_ellip_agreement`, `correlation` |
| `pydemi.atoms` | `solver.solve_atom`; `reference.AtomicLDAReference`, `IsolatedAtomReference`, `default_zval`, `read_potcar_zval`, `read_potcar_rcore` |
| `pydemi.calibration` | `phillips_table`, `fit_ionicity`, `fit_bulk_modulus`, `IonicityCalibration`, `BulkModulusCalibration`, `cohen_bulk_modulus`, `cohen_lambda` |
| `pydemi.resample` | `resample_engine`, `fourier_resample`, `shape_for_spacing` |
| `pydemi.convergence` | `convergence_report`, `format_report` |
| `pydemi.strain` | `strain_response` |
| `pydemi.batch` | `run_batch`, `find_runs`, `compute_one` |
| `pydemi.elements` | Magpie element properties, covalent radii |
| `pydemi.testing` | analytic Slater and Gaussian densities with closed-form statistics |

## 16. Data sources and citations

- **Magpie element data** (Tier 2), bundled from matminer 0.10.1 under its
  BSD licence (`pydemi/data/LICENSE-matminer`): L. Ward, A. Agrawal,
  A. Choudhary, C. Wolverton, *npj Comput. Mater.* **2**, 16028 (2016);
  L. Ward et al., *Comput. Mater. Sci.* **152**, 60 (2018).
- **Phillips ionicity:** J. C. Phillips, *Rev. Mod. Phys.* **42**, 317 (1970);
  *Bonds and Bands in Semiconductors* (1973). Secondary tables used for
  verification: arXiv:1509.01457 (Tables 1–2); *Sci. Adv.* **9**, eadf8706 (2023).
- **Cohen bulk modulus:** M. L. Cohen, *Phys. Rev. B* **32**, 7988 (1985).
- **Approximate ELF:** V. G. Tsirelson, A. Stash, *Chem. Phys. Lett.* **351**, 142 (2002).
- **Local energy densities:** Yu. A. Abramov, *Acta Cryst.* **A53**, 264 (1997).
- **Becke partition:** A. D. Becke, *J. Chem. Phys.* **88**, 2547 (1988).
- **Hirshfeld partition:** F. L. Hirshfeld, *Theor. Chim. Acta* **44**, 129 (1977).
- **LDA correlation:** J. P. Perdew, Y. Wang, *Phys. Rev. B* **45**, 13244 (1992);
  S. H. Vosko, L. Wilk, M. Nusair, *Can. J. Phys.* **58**, 1200 (1980).
- **Atomic reference data (solver validation):** S. Kotochigova, Z. H. Levine,
  E. L. Shirley, M. D. Stiles, C. W. Clark, *Phys. Rev. A* **55**, 191 (1997).
- **Piecewise-linear critical points:** T. Banchoff, *Amer. Math. Monthly* **77**, 475 (1970).

## 17. Author and license

**Shubham Maurya**, CMS Lab, IIT Kanpur.

License: MIT, as declared in `pyproject.toml`. Bundled Magpie data remain
under matminer's BSD licence.
