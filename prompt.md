# pydemi — build specification

Build a Python library called `pydemi` that turns DFT charge-density grids into interpretable,
named, fixed-length feature vectors, at a throughput of seconds per structure across datasets of
thousands of structures.

This document is the contract. Where it is explicit, follow it exactly — several formulas and
file-format details below are easy to get subtly wrong and the errors are silent. Where it leaves a
choice open, it says so.

---

## 0. What this is and is not

**Is:** a featurization layer for the electronic charge density. Input is a grid file plus a
structure; output is a flat `dict[str, float]` of named descriptors.

**Is not:** a topological analysis tool. There is no critical-point search anywhere in this codebase.
Every descriptor is computed by direct grid operations that always terminate. That is the entire
design premise — if a proposed implementation needs a Newton-Raphson search, a convergence loop, or
per-structure human parameter setting, it is the wrong implementation.

**Design principle — one geometry pass, many operators, several fields.** The nearest-atom geometry
(distance to nearest nucleus, unit vector toward it, index of that nucleus) is computed once per
structure and reused by every descriptor. A small set of operators (radial moments, shell fractions,
gradient anisotropy, Laplacian statistics, topological counts, site statistics) is applied across
several scalar fields (ρ, |m|, ELF, Δρ, V). The descriptor count is a consequence of
operators × fields, not a hand-assembled catalogue. Structure the code so this is visible: an
operator should be written once and applied to any field.

---

## 1. Repository layout

```
pydemi/
  __init__.py
  io/
    __init__.py
    base.py              # Grid, VolumetricData dataclasses
    vasp.py              # CHGCAR, ELFCAR, LOCPOT, AECCAR readers
    cube.py              # Gaussian cube format
    xsf.py               # XCrySDen format
    registry.py          # format sniffing and dispatch
  core/
    __init__.py
    grid.py              # grid arithmetic, resampling, integration
    derivatives.py       # gradient, Laplacian, Hessian on periodic non-orthogonal grids
    geometry.py          # nearest-atom pass, shells, PBC handling
    partition.py         # nearest-atom, power diagram, Becke, Hirshfeld weights
  fields/
    __init__.py
    density.py           # rho, magnetization
    deformation.py       # promolecule construction, Delta-rho
    elf.py               # ELF_D reconstruction from rho
    potential.py         # FFT Poisson solve
  operators/
    __init__.py
    moments.py           # radial moments over any field
    fractions.py         # shell fractions over any field
    anisotropy.py        # gradient anisotropy, anisotropy tensor
    laplacian.py         # Laplacian-derived statistics
    topology.py          # percolation, Morse census, extrema
    sitestats.py         # per-site aggregation and variance decomposition
  descriptors/
    __init__.py
    registry.py          # descriptor registration and metadata
    bonding.py
    structural.py
    magnetic.py
    heterogeneity.py
    compositional.py     # thin wrapper over matminer — do not reimplement
  validate/
    __init__.py
    analytic.py          # analytic reference densities for testing
    invariance.py        # translation, rotation, supercell tests
    convergence.py       # grid convergence harness
  cli.py
  batch.py
tests/
docs/
```

---

## 2. Data model

```python
@dataclass(frozen=True)
class Lattice:
    matrix: np.ndarray        # (3,3) float64, ROWS are lattice vectors a1,a2,a3 in Angstrom
    # derived, cached: volume, reciprocal matrix, metric tensors

@dataclass
class Grid:
    data: np.ndarray          # (n1,n2,n3) float64 or float32
    lattice: Lattice
    # data[i,j,k] corresponds to fractional coordinate (i/n1, j/n2, k/n3)
    # ALWAYS periodic. Index arithmetic is modular everywhere.

@dataclass
class Structure:
    lattice: Lattice
    species: list[str]        # element symbols, length n_atoms
    frac_coords: np.ndarray   # (n_atoms, 3)

@dataclass
class VolumetricData:
    structure: Structure
    rho: Grid                       # electrons / Angstrom^3
    magnetization: Grid | None      # electrons / Angstrom^3, signed
    elf: Grid | None                # dimensionless, [0,1]
    potential: Grid | None          # eV
```

Accept a `pymatgen.core.Structure` as an alternative input to `Structure` and convert internally.
Do not depend on pymatgen for anything except structure I/O, composition parsing and spglib symmetry
— the numerical core must be pure numpy/scipy.

---

## 3. File I/O — the details that are easy to get wrong

### CHGCAR
- Header is POSCAR format, then a blank line, then grid dimensions, then the data.
- **Values are ρ × V_cell, not ρ.** Divide every value by the cell volume to get
  electrons/Å³. This is the single most commonly missed detail and it silently breaks every
  absolute-valued descriptor.
- Data is written in **Fortran (column-major) order** — fastest index first. Read with
  `np.fromstring(...).reshape((n1,n2,n3), order='F')`.
- A spin-polarized CHGCAR contains a **second data block** after the first, holding
  m(r) = ρ↑ − ρ↓, same shape, same volume scaling. There may be augmentation-occupancy
  blocks between them — skip lines until the next grid-dimension line matching the first.
- A non-collinear run writes **four** blocks: ρ, then mx, my, mz. Detect this by block count.
  Either handle the vector case properly or raise a clear error. Never silently treat mx as m.

### ELFCAR
- Same header layout as CHGCAR. **Values are not volume-scaled** — do not divide. Range is [0,1].

### LOCPOT
- Same layout. Values in eV, **not** volume-scaled.

### AECCAR0 / AECCAR2
- `AECCAR0` is the frozen core density, `AECCAR2` the self-consistent valence density. Their sum
  approximates the all-electron density. Volume-scaled like CHGCAR.
- Provide `read_all_electron(aeccar0_path, aeccar2_path)` returning their sum. Record on the
  `VolumetricData` which source was used (`"pseudo"` or `"all_electron"`), because it changes the
  interpretation of every core-region descriptor and must appear in the output metadata.

### cube, xsf
- Implement after VASP works. Code-agnostic input is the single largest adoption lever, so these are
  not optional for the 1.0 release. Cube files are in Bohr and atomic units — convert to Å and
  electrons/Å³ on read, and unit-test the conversion against a known file.

---

## 4. Derivatives on a periodic, possibly non-orthogonal grid

This module is where the real numerical work is. Get it right and unit-test it hard.

Let `B` be the matrix whose **rows** are the reciprocal lattice vectors **b₁, b₂, b₃** defined
*without* the 2π factor, i.e. `B = inv(A)ᵀ` where `A` has lattice vectors as rows. Fractional
coordinates are u = (u₁,u₂,u₃).

**Gradient.** In Cartesian components:

```
∇f = Σ_a b_a (∂f/∂u_a)
```

**Laplacian.** Using the contravariant metric tensor g^{ab} = b_a · b_b, i.e. `G = B @ B.T`:

```
∇²f = Σ_{a,b} G[a,b] ∂²f/∂u_a ∂u_b
```

**This must include the off-diagonal terms.** A diagonal-only finite difference is correct only for
orthorhombic cells and is materially wrong (noisier and biased) otherwise. Implement the full form,
and provide the diagonal approximation as `method="diagonal"` **solely** so the paper can quantify
the difference. Default is `method="metric"`.

**Hessian.** The Cartesian Hessian at each voxel:

```
H_cart = Bᵀ @ H_frac @ B      where H_frac[a,b] = ∂²f/∂u_a ∂u_b
```

Return shape `(n1,n2,n3,3,3)`, symmetric. Eigenvalues via `np.linalg.eigvalsh` on the reshaped
`(N,3,3)` array — vectorized, never a Python loop over voxels. Sort ascending: λ₁ ≤ λ₂ ≤ λ₃.

**Two differentiation backends:**
1. `method="fft"` — spectral derivatives via FFT. Exact for band-limited data, no stencil error.
   Default for gradient and Laplacian. Beware ringing for sharp core peaks; document it.
2. `method="fd"` — central finite differences, configurable order (default 4th), with modular
   index wrapping for periodicity.

Provide both and make the choice a parameter, because grid-convergence results will differ between
them and the paper needs to report which was used.

---

## 5. Geometry pass

Computed once per structure, cached on the `VolumetricData`.

For every voxel k:
- `r[k]` — distance to the nearest nucleus **under the minimum-image convention**
- `u[k]` — unit vector from that nucleus to the voxel, Cartesian, shape (N,3)
- `i[k]` — index of that nearest atom

**Implementation.** Replicate atoms into a 3×3×3 supercell (or a smaller shell sufficient for the
cutoff), build a `scipy.spatial.cKDTree`, query each voxel's Cartesian position for the single
nearest image, then map back to the parent atom index by modulo. Verify correctness against brute
force on a small low-symmetry test cell — minimum-image bugs in triclinic cells are common and quiet.

**Shells.** Three radial shells with cutoffs c₁ = 0.8 Å, c₂ = 1.5 Å:
- core: r ≤ c₁
- bond: c₁ < r ≤ c₂
- interstitial: r > c₂

Cutoffs must be parameters, never hardcoded. Also support a radius-scaled mode where c₁, c₂ are
per-element multiples of a tabulated covalent radius. Provide a `sensitivity_sweep(c1_range,
c2_range)` helper that recomputes shell descriptors across a grid of cutoffs and returns a dataframe
— the paper needs these curves and they must be reproducible.

---

## 6. Fields

Each field is a `Grid` with the same shape as ρ. Operators must accept any field.

| Field | Source | Notes |
|---|---|---|
| `rho` | read directly | electrons/Å³ |
| `abs_m` | \|magnetization\| | zero field if non-magnetic |
| `elf` | read, or reconstructed (§6.2) | |
| `delta_rho` | ρ − promolecule (§6.1) | signed |
| `potential` | read, or FFT-solved (§6.3) | |

### 6.1 Deformation density

```
Δρ(r) = ρ_crystal(r) − Σ_i ρ_free[element(i)](r − R_i)
```

The promolecule is a superposition of spherical free-atom densities placed at nuclear positions.
Implement three reference sources, in this order of preference, selected by a parameter:

1. `"aeccar0"` — read AECCAR0 from a matching run; it is already an element-wise superposition.
2. `"tabulated"` — ship radial free-atom densities as a data file, interpolate onto the grid.
   Nearest-neighbour-in-radius plus cubic spline; store as (r, ρ(r)) tables per element.
3. `"custom"` — user supplies a directory of per-element radial densities from their own
   isolated-atom calculations.

Record which was used in output metadata. Δρ is meaningful only relative to its reference and this
must be reproducible.

Build the promolecule by summing over periodic images out to where the free-atom density falls below
a tolerance (default 1e-6 e/Å³), not just the parent cell.

### 6.2 ELF reconstruction (Tsirelson–Stash)

Reconstruct ELF from ρ alone via the Kirzhnits gradient expansion. In **atomic units** —
convert ρ to e/bohr³ first, and convert back after:

```
C_F  = (3/10) (3π²)^(2/3) ≈ 2.871234
t_P  = C_F ρ^(5/3) + (1/72)|∇ρ|²/ρ + (1/6)∇²ρ
D_P  = t_P − |∇ρ|²/(8ρ)
D_h  = C_F ρ^(5/3)
ELF_D = 1 / (1 + (D_P/D_h)²)
```

Guard ρ → 0: clamp the denominator at a floor (default 1e-10) and set ELF_D to 0 where ρ is below a
density threshold. Unit-test that ELF_D ∈ [0,1] everywhere on real data.

Provide `elf_fidelity(elf_true, elf_reconstructed)` returning Pearson r, MAE and RMSE over voxels —
this is a validation metric, not a descriptor, and belongs in `validate/`.

### 6.3 Electrostatic potential

Hartree potential from ρ by one FFT:

```
V_H(G) = 4π ρ(G) / |G|²    for G ≠ 0;    V_H(0) = 0
```

Use `|G| = 2π|B ᵀ n|` for integer triples n. Note this is the electronic Hartree term only; the full
ESP needs the ionic contribution. Expose both, label clearly, and record in metadata which was used.
Prefer a read LOCPOT when available.

---

## 7. Operators

Write each once, applied to any field. Signature shape:

```python
def radial_moment(field: np.ndarray, r: np.ndarray, order: int,
                  weight: Literal["signed","abs"] = "abs") -> float
```

| Operator | Definition |
|---|---|
| Radial moment | `m_n = Σ_k w_k r_k^n / Σ_k w_k`, where `w = field` or `|field|` |
| Shell fraction | `f_shell = Σ_{k∈shell} w_k / Σ_k w_k` |
| Gradient anisotropy | `ζ = 1 − Σ_k |∇f_k · û_k| / Σ_k |∇f_k|` |
| Anisotropy tensor | `T_ab = Σ_k ∂_a f_k ∂_b f_k / Σ_k |∇f_k|²` |
| Laplacian sign fraction | `(1/N) Σ_k 𝟙(∇²f_k < 0)`, and a charge-weighted variant |
| Laplacian concentration | `Σ_{∇²f<0}|∇²f_k| / Σ_k |∇²f_k|` |
| Site aggregation | restrict all sums to `{k : i(k) = i}`, return per-site array |
| Percolation threshold | lowest level whose super-level set spans a direction under PBC |
| Extremum census | local extrema by 26-neighbour comparison with modular wrapping |

---

## 8. Descriptors

Register each descriptor with metadata so the catalogue is introspectable and the SI tables can be
generated from code rather than maintained by hand:

```python
@register(
    name="zeta",
    domain="bonding",            # bonding | structural | magnetic | heterogeneity | compositional
    field="rho",
    requires=["gradient", "geometry"],
    units="dimensionless",
    range=(0.0, 1.0),
    intensive=True,              # MUST be True for anything entering a feature vector
    sentinel_cases={"uniform_density": 0.0},
    references=["..."],
)
def zeta(vd: VolumetricData) -> float: ...
```

`pydemi.catalogue()` returns this metadata as a dataframe.

### 8.1 Bonding domain

| Name | Formula |
|---|---|
| `zeta` | `1 − Σ|∇ρ_k·û_k| / Σ|∇ρ_k|` |
| `m1` | `Σ ρ_k r_k / Σ ρ_k` |
| `m2` | `Σ ρ_k r_k² / Σ ρ_k` |
| `sigma_r2` | `m2 − m1²` |
| `f_core`, `f_bond`, `f_int` | shell fractions of ρ |
| `lnf` | `(1/N) Σ 𝟙(∇²ρ_k < 0)` |
| `lnf_charge_weighted` | `Σ_{∇²ρ<0} ρ_k / Σ ρ_k` — compute both, they are not interchangeable |
| `lap_concentration` | `Σ_{∇²ρ<0}|∇²ρ_k| / Σ|∇²ρ_k|` |
| `ellip_bond_avg` | mean of `λ₁/λ₂ − 1` over `{k ∈ bond, λ₂ < 0}` |
| `ellip_bond_std` | std of the same |
| `f_H_negative` | `(1/N_bond) Σ_{k∈bond} 𝟙(H_k < 0)`, `H = g + v` (see below) |
| `H_bond_mean` | `⟨H_k⟩` over bond shell |
| `G_over_rho` | `⟨g_k/ρ_k⟩` over bond shell |
| `f_ELF_localized` | `(1/N_bond) Σ_{k∈bond} 𝟙(ELF_k > 0.5)` |
| `ELF_bond_avg` | mean ELF over bond shell |
| `ELF_core_valence_contrast` | `⟨ELF⟩_core / ⟨ELF⟩_bond` |
| `zeta_ELF` | anisotropy operator on the ELF field |
| `m1_def`, `m2_def` | radial moments of Δρ, weighted by `|Δρ|` |
| `sigma_r2_def` | `m2_def − m1_def²` |
| `f_bond_def`, `f_int_def` | `Σ_{shell,Δρ>0} Δρ_k / Σ_{Δρ>0} Δρ_k` |
| `f_bond_dep` | `Σ_{shell,Δρ<0}|Δρ_k| / Σ_{Δρ<0}|Δρ_k|` |
| `def_polarity` | `Σ_k |Δρ_k| dV / Q_tot` |
| `bond_charge_transfer_pair` | `∫Δρ dV` over the region between each neighbour pair; return mean and std |
| `rho_mid_mean`, `rho_mid_std` | ρ interpolated at nearest-neighbour bond midpoints |
| `f_NCI` | `(1/N) Σ 𝟙(s_k < 0.5 ∧ ρ_k < 0.05 au)`, `s = |∇ρ|/(2(3π²)^(1/3) ρ^(4/3))` |
| `NCI_attractive` | fraction of NCI voxels with `λ₂ < 0` |
| `sign_lambda2_rho_mean` | mean of `sign(λ₂)·ρ_k` over NCI voxels |
| `V_site` array, `V_spread` | potential at each nucleus; std across sites |
| `V_int_min` | min potential over interstitial voxels |

Local energy densities, atomic units:
```
g = C_F ρ^(5/3) + (1/72)|∇ρ|²/ρ + (1/6)∇²ρ
v = (1/4)∇²ρ − 2g
H = g + v
```
These share their computation with ELF_D — compute `g` once and reuse.

### 8.2 Structural domain

| Name | Formula |
|---|---|
| `rho_perc_a/b/c` | lowest level c whose super-level set `{ρ > c}` spans direction α under PBC |
| `perc_anisotropy` | `(max_α − min_α) / mean_α` |
| `n_max`, `n_min`, `n_saddle1`, `n_saddle2` | 26-neighbour extremum census, **normalized per unit volume** |
| `euler_consistency` | `n_max − n_saddle2 + n_saddle1 − n_min`; must be 0 on a 3-torus |
| `n_NNM` | local maxima with `min_i \|r − R_i\| > r_cut`, **per unit volume** |
| `Q_NNM` | charge in non-nuclear basins, **as a fraction of Q_tot** |
| `rho_min` | `min_k ρ_k` |
| `rho_min_ratio` | `rho_min / ⟨ρ⟩_V` |
| `rho_int_mean` | mean ρ over interstitial |
| `T_eigenvalues` | `t1,t2,t3` of the anisotropy tensor, trace 1 |
| `charge_FA` | `√(3/2)·‖T − (1/3)I‖_F / ‖T‖_F` |
| `shannon_entropy` | `−∫ ρ̃ ln ρ̃ dV`, `ρ̃ = ρ/N_e` |
| `fisher_information` | `∫ \|∇ρ̃\|²/ρ̃ dV` |
| `disequilibrium` | `∫ ρ̃² dV` |
| `LMC_complexity` | `D · e^S` |

**Percolation implementation:** bisect on the threshold c. At each c, label connected components of
the super-level set with `scipy.ndimage.label` using 6-connectivity, then merge labels across
opposite faces with a union-find to enforce PBC. A cluster spans direction α if it touches both the
`u_α = 0` and `u_α = 1` faces *and* those contacts belong to the same component after the periodic
merge. Test on a known case: a simple cubic lattice of Gaussians percolates in all three directions
at a computable level.

### 8.3 Magnetic domain

Every entry returns 0.0 with a `magnetic=False` metadata flag for non-magnetic structures — never
NaN.

| Name | Formula |
|---|---|
| `M_abs` | `Σ \|m_k\| dV` — report also per atom, which is the intensive version |
| `M_net` | `\|Σ m_k dV\|` — per atom likewise |
| `m1_spin` | `Σ \|m_k\| r_k / Σ \|m_k\|` |
| `sigma_r2_spin` | second moment of `\|m\|` minus `m1_spin²` |
| `f_bond_spin` | `Σ_{bond} \|m_k\| / Σ \|m_k\|` |
| `mu_site` array | `μ_i = Σ_{i(k)=i} m_k dV` |
| `mu_site_std` | std over i of μ_i |
| `spin_frustration` | `1 − \|Σ_i μ_i\| / Σ_i \|μ_i\|` |
| `spin_charge_correlation` | Pearson r between ρ_k and `\|m_k\|` |

**Assert in a test** that `Σ_i μ_i == M_net` to within float tolerance. The nearest-atom partition
tiles space, so this must hold exactly; it is the correctness property that distinguishes this from
VASP's RWIGS-sphere moments, which do not tile space and do not sum to the total.

`spin_frustration` is defined for collinear magnetism only. If the reader detected a non-collinear
file, either compute the vector generalization (`1 − ‖Σ_i **μ**_i‖ / Σ_i ‖**μ**_i‖`) or raise.

### 8.4 Heterogeneity domain (meta-operator)

For any base descriptor X computable from per-voxel quantities, generate:

- `X_site_std` — std over sites of X^(i)
- `X_site_range`, `X_site_max`, `X_site_min`
- `X_within_element_var` — mean over elements of `Var_{i∈e}(X^(i))`
- `X_between_element_var` — `Var_e(mean_{i∈e} X^(i))`

Apply to at least `m1`, `f_bond`, `zeta`, `mu_site`.

Implement as a decorator or higher-order function over the base descriptor, not by copy-paste. The
two variance terms are an ANOVA decomposition and must satisfy

```
Var_i(X) = Σ_e w_e Var_{i∈e}(X) + Var_e(X̄_e)
```

with `w_e` the fraction of sites of element e. **Unit-test this identity** — it is a free correctness
check on the whole site-aggregation path.

Attach `n_atoms` and per-element site counts to the output, because these statistics are noisy for
small cells and downstream filtering needs the counts.

### 8.5 Compositional

Thin wrapper over `matminer.featurizers.composition.ElementProperty.from_preset("magpie")`. **Do not
reimplement these.** Tag every one with `domain="compositional"` and `adopted=True` in the registry so
they can be filtered out of novelty claims and used as an explicit baseline.

---

## 9. Partitioning schemes

All descriptors that assign voxels to atoms must accept a `partition=` parameter:

| Scheme | Rule |
|---|---|
| `"nearest"` (default) | `i(k) = argmin_i \|r_k − R_i\|` |
| `"power"` | `i(k) = argmin_i (\|r_k − R_i\|² − R_i_rad²)`, radius-weighted Voronoi |
| `"becke"` | Becke fuzzy-cell weights `w_i(r)`, pure geometry |
| `"hirshfeld"` | `w_i(r) = ρ_free_i(r−R_i) / Σ_j ρ_free_j(r−R_j)`, reuses the promolecule |

For the smooth schemes, replace the indicator `𝟙(i(k)=i)` with `w_i(r_k)` in every site-restricted
sum. Write the site aggregation so this substitution is a single code path, not four.

Also expose `hirshfeld_charges(vd) -> np.ndarray` returning `q_i = Z_i − ∫ w_i ρ dV`, for external
comparison against published Bader charges.

---

## 10. Correctness requirements

These are non-negotiable and each needs a test.

**Intensivity.** Every descriptor entering a feature vector must be unchanged when a cell is
replaced by its 2×2×2 supercell of the same material. Extensive quantities (`Q_tot`, `M_abs`, raw
extremum counts, `Q_NNM`) must be normalized by volume, atom count, or a total before being
registered as descriptors. **Write a parametrized test that runs every registered descriptor on a
primitive cell and its 2×2×2 supercell and asserts agreement to 1e-6 relative.** This test will
initially fail for several descriptors; that is the point.

**Translation invariance.** Rigidly translating the structure and grid together changes nothing.

**Rotation invariance.** For scalar descriptors, rotating the cell changes nothing. `T_eigenvalues`
is invariant as a sorted triple; the tensor itself is not, and that is correct.

**Sentinel behaviour.** Define and document the return value for every degenerate case:

| Case | Affected | Required behaviour |
|---|---|---|
| Uniform density | `zeta` (0/0), `f_NCI`, anisotropy tensor | Documented constant, not NaN. `zeta` → 0.0 with a flag. |
| Non-magnetic | all magnetic | 0.0 plus `magnetic=False` |
| Single element | `elemental_bonding_variance_between` | 0.0 plus flag |
| One atom per element | within-element variance | NaN is acceptable here *if* accompanied by the site count; document it |
| `lnf → 0` | any ratio dividing by it | return inf-guarded sentinel and flag; never emit a silent large value |
| ρ → 0 voxels | `ELF_D`, `g`, `s`, information measures | clamp at a floor, document the floor |

**Grid adequacy.** Compute `euler_consistency` for every structure and emit it as a quality flag. A
nonzero value means the grid does not resolve the topology.

---

## 11. Analytic validation

`validate/analytic.py` must build test densities with closed-form descriptor values, on a grid, and
assert agreement.

**Single Slater 1s**, `ρ(r) = (ζ³/π) e^{−2ζr}`, normalized to 1 electron, in a large box:

| Quantity | Exact value |
|---|---|
| `∫ρ dV` | 1 |
| `m1` | `3/(2ζ)` |
| `m2` | `3/ζ²` |
| `sigma_r2` | `3/(4ζ²)` |
| `zeta` | 0 exactly — the gradient is purely radial |
| `∇²ρ` | `ρ(4ζ² − 4ζ/r)`, so negative for `r < 1/ζ` |
| `lnf` | the volume fraction of the box with `r < 1/ζ` |

**Superposition of two Slater 1s functions** at separation d gives a nonzero `zeta` and a
deformation density with an analytically known integral — use for the Δρ path.

**Uniform density** — exercises every sentinel.

These tests are the backbone of the paper's verification section. Report convergence of each
quantity against grid spacing so the recommended minimum mesh comes out of the test suite rather
than a guess.

---

## 12. API

```python
import pydemi

vd = pydemi.read_vasp("CHGCAR", elf="ELFCAR")           # VolumetricData
feats = pydemi.featurize(vd)                             # dict[str, float]

feats = pydemi.featurize(
    vd,
    domains=["bonding", "magnetic"],
    partition="nearest",
    shells=(0.8, 1.5),
    deformation_reference="tabulated",
    elf_source="reconstruct",                            # or "file"
    laplacian_method="metric",
    derivative_backend="fft",
)

pydemi.catalogue()                                       # DataFrame of descriptor metadata
```

Batch:

```python
df = pydemi.featurize_batch(
    paths,
    n_workers=8,
    on_error="record",          # record | raise | skip
    progress=True,
)
```

Return a tidy DataFrame: one row per structure, descriptor columns, plus metadata columns
(`n_atoms`, `volume`, `grid_shape`, `density_source`, `magnetic`, `euler_consistency`,
`wall_time_s`, `pydemi_version`, `error`). Errors go in the frame, never crash a 6,000-structure run.

CLI:

```
pydemi featurize CHGCAR --out features.json
pydemi batch ./runs --glob "*/CHGCAR" --out features.csv --workers 8 --domains bonding,magnetic
pydemi catalogue --out catalogue.csv
pydemi sweep CHGCAR --param c2 --range 1.0:2.5:0.05 --out sweep.csv
```

---

## 13. Performance

Target: **under 10 s per structure** for a 100³ grid on one core, and linear scaling in voxel count.

- Compute the geometry pass, gradient, Laplacian and Hessian eigenvalues **once** and cache on the
  `VolumetricData`. Never recompute per descriptor.
- All operators vectorized. No Python loops over voxels anywhere. Loops over *atoms* or *sites* are
  fine.
- Offer `float32` mode for memory-constrained runs; default `float64`. Verify descriptor agreement
  between the two in a test and report the discrepancy.
- Batch parallelism is per structure via `concurrent.futures.ProcessPoolExecutor`, not per
  descriptor.
- The Hessian is the memory peak: `(N,3,3)` float64 for a 200³ grid is about 576 MB. Compute
  eigenvalues in chunks and free the Hessian immediately.

---

## 14. Build order

Do not build all of this at once. Each milestone should be working and tested before the next.

1. **I/O and data model** — CHGCAR read including the volume scaling, Fortran ordering and the spin
   second block. Round-trip test.
2. **Derivatives** — gradient, metric-tensor Laplacian, Hessian, both backends. Validate against
   analytic Slater densities in orthorhombic *and* triclinic cells.
3. **Geometry pass** — nearest-atom with minimum image, verified against brute force.
4. **Tier-1 descriptors** — ζ, moments, shell fractions, lnf, lap_concentration. Analytic tests.
5. **Invariance test harness** — run it and fix what fails before adding anything further.
6. **Site aggregation** — heterogeneity meta-operator, ANOVA identity test.
7. **Magnetic domain** — including the `Σμ_i == M_net` assertion.
8. **Derived fields** — ELF_D, potential, NCI, energy densities.
9. **Deformation density** — promolecule construction and the A-family descriptors.
10. **Structural domain** — percolation, extremum census, information measures.
11. **Alternative partitions** — power, Becke, Hirshfeld.
12. **Batch, CLI, extra formats, docs.**

---

## 15. Constraints

- Python ≥ 3.10. numpy, scipy required. pymatgen, spglib, matminer optional extras behind
  `pydemi[full]`.
- Type hints throughout; `mypy --strict` on the core modules.
- Docstrings carry the formula in the same notation as this spec, and the analytic limiting values
  where known — the SI descriptor catalogue will be generated from them.
- No hardcoded cutoffs, thresholds or element data outside a single `constants.py` / data directory.
- No network calls at runtime.
- No plotting module. Return dataframes and let users plot. Do not reimplement what matminer,
  pymatgen or pandas already do.
- Deterministic: same input, same output, no RNG anywhere.
- Every descriptor that could be NaN must document when, and set a companion flag rather than
  emitting a bare NaN.

---

## 16. Things that will go wrong

Flagged because they are silent failures, not crashes:

- Forgetting the CHGCAR volume division. Every absolute descriptor is then wrong by a constant
  factor that varies between structures.
- Reading CHGCAR in C order instead of Fortran order. Produces a transposed but plausible-looking
  density.
- Minimum-image bugs in triclinic cells. Nearest-atom distances come out slightly too large near
  cell corners.
- Diagonal-only Laplacian on non-orthogonal cells. Noisier and biased, but never raises.
- Treating a non-collinear CHGCAR's `mx` block as the collinear `m`.
- Extensive descriptors entering the feature vector. Doubles under supercell expansion; invalidates
  everything downstream.
- ELF_D computed in Å-based units instead of atomic units. The `ρ^(5/3)` term then has the wrong
  scale and ELF_D leaves [0,1].
