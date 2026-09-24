# pydemi — code guide

A file-by-file, function-by-function walk through the pydemi source, written
as a personal reference. It explains what every piece of code does, the
algorithm behind it, why it was written that way (including the measurements
that forced a design decision), how the pieces connect, and which tests pin
the behaviour down.

The user-facing documentation is `README.md`; this guide is about the code.

Author: Shubham Maurya, CMS Lab, IIT Kanpur.

---

## Contents

- Part I — Orientation
  - 1. Repository layout
  - 2. Architecture and data flow
  - 3. Conventions used everywhere
- Part II — The engine layer
  - 4. `constants.py`
  - 5. `structure.py`
  - 6. `grid.py`
  - 7. `field.py`
  - 8. `geometry.py`
  - 9. `shells.py`
  - 10. `partition.py`
  - 11. `engine.py`
  - 12. `elements.py`
- Part III — Input / output
  - 13. `io/vasp.py`
  - 14. `io/grids.py`
- Part IV — Free-atom references
  - 15. `atoms/solver.py`
  - 16. `atoms/reference.py`
- Part V — Descriptors
  - 17. `descriptors/__init__.py` (the dispatcher)
  - 18. `descriptors/registry.py`
  - 19. `descriptors/primitives.py`
  - 20. `descriptors/tier1.py`
  - 21. `descriptors/composition.py`
  - 22. `descriptors/tier3.py`
  - 23. `descriptors/kinetic.py` (F1 ELF_D, F5)
  - 24. `descriptors/elf.py` (Family B, entry 73)
  - 25. `descriptors/potential.py` (F2)
  - 26. `descriptors/hessian.py` (F3, F4)
  - 27. `descriptors/information.py` (F6)
  - 28. `descriptors/anisotropy.py` (I1)
  - 29. `descriptors/sites.py` (C, H)
  - 30. `descriptors/spin.py` (E)
  - 31. `descriptors/topology.py` (G)
  - 32. `descriptors/bonds.py` (I2)
  - 33. `descriptors/deformation.py` (A)
  - 34. `descriptors/dataset.py`
- Part VI — Calibration and tools
  - 35. `calibration/__init__.py` (D, 106)
  - 36. `resample.py`
  - 37. `strain.py` (107)
  - 38. `convergence.py`
  - 39. `batch.py`
  - 40. `cli.py`
  - 41. `testing/analytic.py`
- Part VII — Data, tests, docs
  - 42. Data files
  - 43. The test suite, file by file
  - 44. Packaging (`pyproject.toml`)
- Part VIII — Cross-cutting topics
  - 45. Caching: what is computed once, and where it lives
  - 46. Units: where every conversion happens
  - 47. NaN, zero and sentinel policy
  - 48. Recipes: adding a descriptor, a family, a field, a reader
  - 49. Design decisions and the measurements behind them
  - 50. Known rough edges

---

# Part I — Orientation

## 1. Repository layout

```
pydemi/
├── pyproject.toml                  packaging, dependencies, the `pydemi` console script
├── README.md                       user documentation
├── pydemi — Consolidated Descriptor Reference.pdf   the specification (entries 1–107)
├── docs/
│   ├── numerics.md                 measured effect of stencils / boundaries / minimum image
│   └── CODE_GUIDE.md               this file
├── results/                        output of dataset runs (not part of the package)
├── src/pydemi/
│   ├── __init__.py                 public top-level API
│   ├── constants.py                unit conversions, C_F, shell cutoffs
│   ├── structure.py                Structure (lattice, species, fractional coords)
│   ├── grid.py                     Grid: periodic derivatives (fd / spectral)
│   ├── field.py                    Field: a grid array with cached derivatives
│   ├── geometry.py                 nearest-atom / power-diagram assignment (KD-tree)
│   ├── shells.py                   core / bond / interstitial masks
│   ├── partition.py                Hard, Becke and Hirshfeld partitions
│   ├── engine.py                   Engine: structure + fields + all caches; constructors
│   ├── elements.py                 Magpie element-property lookup
│   ├── resample.py                 Fourier resampling
│   ├── strain.py                   strain response (entry 107)
│   ├── convergence.py              per-descriptor grid-convergence report
│   ├── batch.py                    parallel, resumable CSV runs
│   ├── cli.py                      `pydemi` command line
│   ├── io/
│   │   ├── vasp.py                 CHGCAR / AECCAR / ELFCAR / LOCPOT reader + writer
│   │   └── grids.py                cube and XSF readers + writers
│   ├── atoms/
│   │   ├── solver.py               spherical LDA atom (free-atom densities)
│   │   └── reference.py            reference-density providers, ZVAL / RCORE parsing
│   ├── calibration/__init__.py     Phillips table, ionicity and B0 fits, Cohen
│   ├── descriptors/                one module per family + registry + dispatcher
│   ├── testing/analytic.py         analytic Slater / Gaussian densities for tests
│   └── data/                       magpie_elements.csv, phillips_ionicity.csv, LICENSE-matminer
└── tests/                          164 tests (pytest)
```

Size: about 5,400 lines of library code and 2,100 lines of tests.

## 2. Architecture and data flow

pydemi is organized around one idea from the specification: **one grid engine
applied to several physical fields**. Every descriptor is a reduction over
voxels of some field (ρ, |m|, ELF, V, Δρ), weighted and masked by a small set
of geometric quantities that are computed once.

The flow for one structure:

```
files on disk
   │   io/vasp.py, io/grids.py            parse header + every data block, divide by V
   ▼
ChargeDensity / VolumetricData             numpy arrays + Structure
   │   Engine.from_vasp_dir / from_file
   ▼
Engine ─────────────────────────────────────────────────────────────────┐
 ├─ fields: {"rho": Field, "magnetization": Field, "elf": Field, ...}   │
 │     Field: lazily cached gradient, Hessian, Laplacian, eigenvalues   │
 ├─ grid(shape)          one Grid per grid shape                        │
 ├─ geometry(shape)      NearestAtom: r_k, r̂_k, i(k)   (KD-tree, once)  │
 ├─ shell_masks(shape)   core / bond / interstitial booleans            │
 ├─ partition(shape, scheme)   nearest / power / becke / hirshfeld      │
 ├─ reference            free-atom densities (atoms/)                   │
 └─ cache                shared derived arrays, e.g. kinetic terms      │
   │                                                                    │
   ▼  descriptors/__init__.compute_descriptors(engine, families, kinds) │
family functions (tier1, tier2, ..., topology_family, ...) ─────────────┘
   │   each returns {name: float}
   ▼
filter by registry kind and family, order by registry  →  {name: value}
   │   batch.run_batch / cli.py
   ▼
one CSV row per structure
```

Key properties of this design:

- **Everything expensive is computed once** and cached on the `Engine` or the
  `Field` (§45). Asking for another descriptor from an already-computed field
  costs a reduction, not a recomputation.
- **Grids can differ between fields.** ELFCAR is usually on a coarser grid
  than CHGCAR; every cache is keyed by grid shape, so each field sees geometry
  on its own grid.
- **Families are independent functions** of `(engine, field, shells)`
  returning plain dicts. The dispatcher only orders and filters.
- **The registry is the single source of truth** for names, entry numbers,
  kinds and formulas. Output order, filtering, CSV columns, `pydemi list` and
  the README tables all come from it.

## 3. Conventions used everywhere

- **Lattice rows are lattice vectors.** Cartesian position `x = u @ lattice`
  for a fractional row vector `u`. The inverse map is `u = x @ inv(lattice)`.
- **Grid indexing.** Voxel `(i, j, k)` sits at fractional
  `(i/n1, j/n2, k/n3)`; arrays are `(n1, n2, n3)` in C order in memory.
  VASP files store x-fastest (Fortran order), so readers reshape with
  `order="F"`.
- **Units.** Å and e/Å³ internally; atomic units only inside formulas the
  specification defines in atomic units (§46).
- **Directions.** `direction` / `r̂` is the unit vector *from the nucleus to
  the voxel*; it is 0 on a voxel that coincides with a nucleus.
- **Return types.** Family functions return `dict[str, float]` (ints for
  counts and flags). Undefined ratios are `nan`, never regularized with an
  epsilon (§47).
- **Naming.** A registry name always keeps the specification's formula; fixes
  get new names; data sources are encoded in names (`ELF_*` / `ELFD_*`,
  `V_*` / `VH_*`, `_power` / `_hirshfeld` / `_becke`).

---

# Part II — The engine layer

## 4. `constants.py`

Pure constants, no functions.

| Name | Value | Used for |
|---|---|---|
| `BOHR_ANGSTROM` | 0.529177210903 (CODATA 2018) | every Å ↔ bohr conversion |
| `ANGSTROM_BOHR` | 1 / a₀ | volumes in bohr³ (F6) |
| `DENSITY_TO_AU` | a₀³ | e/Å³ → e/bohr³ |
| `DENSITY_FROM_AU` | 1 / a₀³ | e/bohr³ → e/Å³ (atomic solver, cube files) |
| `GRADIENT_TO_AU` | a₀⁴ | ∇ρ from e/Å⁴ to e/bohr⁴ |
| `LAPLACIAN_TO_AU` | a₀⁵ | ∇²ρ from e/Å⁵ to e/bohr⁵ |
| `HARTREE_EV` | 27.211386245988 | |
| `COULOMB_EV_ANGSTROM` | E_h · a₀ = 14.3996 eV·Å | e²/4πε₀ in the Hartree potential (F2) |
| `C_F` | (3/10)(3π²)^{2/3} ≈ 2.871 | Thomas–Fermi constant (F1, F5) |
| `SHELL_C1`, `SHELL_C2` | 0.8, 1.5 Å | default shell cutoffs |

Why the gradient and Laplacian factors are powers 4 and 5: ρ carries L⁻³, each
derivative another L⁻¹, and converting a quantity with units L⁻ⁿ from Å to
bohr multiplies by a₀ⁿ.

## 5. `structure.py`

### `class Structure` (frozen dataclass, `eq=False`)

Fields: `lattice` (3×3, Å), `species` (tuple of symbols, one per atom),
`frac_coords` (n×3).

- `__post_init__` normalizes inputs (arrays of float, tuple of str), checks
  that there is one label per atom and that the lattice is not singular. It
  uses `object.__setattr__` because the dataclass is frozen.
- Cached properties (`functools.cached_property` works on a frozen dataclass
  because it writes to `__dict__` directly): `volume`, `cart_coords`,
  `inv_lattice`, `reciprocal_lattice` (rows b_j with a_i·b_j = 2πδ_ij),
  `elements` (distinct symbols in order of first appearance, via
  `dict.fromkeys`), `element_index` (per-atom index into `elements`).
- `from_pymatgen` / `to_pymatgen` convert both ways; pymatgen is imported
  lazily inside `to_pymatgen`.
- `eq=False` makes instances hash by identity, so they can be used as keys
  and never trigger elementwise array comparison.

`elements` order matters: the POTCAR / OUTCAR ZVAL list is matched to it
positionally in `Engine.from_vasp_dir` (§11), which is correct when each
element appears in exactly one species group (the normal VASP layout).

## 6. `grid.py`

### Module docstring

States the two derivative methods, the chain rule for non-orthogonal cells,
and why `fd` is the default (sign-thresholded descriptors, the spectral
round-off floor in empty regions).

### `HESSIAN_PAIRS`

`((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))` — the packed storage order of the
symmetric 3×3 Hessian. Packing saves a third of the memory at large grids.

### `class Grid`

Constructed from `lattice` and `shape`. Geometry properties:

- `inv_lattice`, `volume`, `n_voxels`, `dV = V / N`.
- `spacing`: voxel edge length along each lattice vector, |a_i| / n_i.
- `metric`: the contravariant metric G = A⁻ᵀA⁻¹, where A is the lattice
  matrix. Used by the FD Laplacian.
- `frac_coords()` / `cart_coords()`: full (n1, n2, n3, 3) coordinate arrays.
  Deliberately **not** cached (they are large and used once per geometry
  pass).
- `integrate(f)` = Σ f · dV.

#### The chain rule (how any cell shape is handled)

With x = uA and voxel index i_m = n_m u_m:

    ∂/∂u_m = n_m ∂/∂i_m
    ∇ₓf    = A⁻¹ (∂f/∂u)                        (gradient)
    Hₓ     = A⁻¹ H_u A⁻ᵀ,  H_u[m,q] = n_m n_q ∂²f/∂i_m∂i_q
    ∇²f    = tr Hₓ = Σ_mq G[m,q] H_u[m,q],  G = A⁻ᵀA⁻¹

This includes the cross-axis terms that a "diagonal" Laplacian drops, and is
exact for triclinic cells.

#### Spectral helpers

- `_frequencies()` returns integer frequencies per axis as broadcastable
  arrays on the `rfftn` half grid (full `fftfreq` on axes 0–1, `rfftfreq` on
  axis 2), and a second copy `m_odd` with the Nyquist frequency of every
  even-length axis zeroed. Reason: for an even grid the Nyquist mode's
  derivative is ambiguous (its sine component is not representable), so it is
  removed from **odd-order** terms; keeping it would make the result complex
  or asymmetric.
- `_B` = 2π A⁻¹, so the Cartesian wavevector is G_a = Σ_j B[a,j] m_j.
- `_second_order_multiplier(W)` computes Σ_jq W[j,q] m_j m_q, using the full
  frequencies for j = q and the Nyquist-free ones for j ≠ q (a mixed term is
  odd in each frequency). Both the spectral Hessian and the spectral Laplacian
  go through this one function, which is what makes `laplacian == trace(hessian)`
  hold exactly in spectral mode (this was a bug fixed during Phase 0: the
  Laplacian originally used |G|² with Nyquist kept in the cross terms).
- `g_squared()` = multiplier with W = BᵀB, i.e. |G|². Also used by the
  Hartree solve (F2) and the promolecule (A).
- `frequencies()` full-grid integer frequencies (public helper).

#### Derivatives

- `_check(f, method)` validates the method name and the shape.
- `gradient(f, method)`:
  - `spectral`: `rfftn`, multiply by i·G_a (Nyquist-free), `irfftn` with
    `axes=(0,1,2)` (numpy 2.x requires explicit axes when `s` is given).
  - `fd`: periodic central differences `n_m·(f[i+1] − f[i−1])/2` via
    `np.roll`, then the A⁻¹ contraction.
- `_hessian_frac(f)`: index-space second differences, compact stencils:
  diagonal `n_m²(f[i+1] − 2f + f[i−1])`, mixed
  `n_m n_q (f[++] − f[+−] − f[−+] + f[−−])/4`. Returns a dict keyed by
  `(m, q)` with m ≤ q.
- `hessian(f, method)`: packed (…, 6). Spectral: multiplier
  W = ½(B_a⊗B_b + B_b⊗B_a) per pair. FD: A⁻¹ H_u A⁻ᵀ, skipping zero
  coefficients (orthogonal cells).
- `laplacian(f, method)`: spectral `−|G|² F`; FD `Σ G[m,q] H_u[m,q]` (off
  diagonal counted twice).

Why compact stencils and not `np.gradient` twice: applying a first-derivative
routine twice produces a stencil of double width (the Laplacian of a smoothed
density) and, in `np.gradient`, one-sided differences on the boundary planes.
Over 200 VASP CHGCARs this changes `lnf` by 2% for the median structure and up to 21% (`docs/numerics.md`).

### `unpack_hessian(packed)` and `hessian_eigenvalues(packed, chunk)`

`unpack_hessian` expands (…, 6) to (…, 3, 3). `hessian_eigenvalues` runs
`np.linalg.eigvalsh` in chunks of 2²⁰ voxels so the full (N, 3, 3) array never
exists at once; eigenvalues come out ascending (λ₁ ≤ λ₂ ≤ λ₃).

Tests: `tests/test_grid.py` (spectral exact to 1e-9 on a triclinic Gaussian,
FD error ratio 3–5 per halving, Laplacian = trace for both methods, boundary
plane no worse than interior, eigenvalues vs dense `eigvalsh`, input checks).

## 7. `field.py`

### `class Field`

A named grid array (`values`, `grid`, `name`, `method`). All derivatives are
`functools.cached_property`:

- `gradient` (…, 3), `gradient_norm`, `hessian` (packed), `laplacian`,
  `hessian_eigenvalues`.
- `laplacian` checks `"hessian" in self.__dict__` — if the Hessian was
  already computed it returns its trace instead of computing the Laplacian
  again (saves one pass; they agree to round-off).
- `hessian_full()` unpacks on demand (not cached, large).
- `integral()` = Σ values · dV.
- `clear_cache()` pops the cached keys from `__dict__`.

The method (`fd` / `spectral`) is fixed per field, inherited from the engine.

## 8. `geometry.py`

The geometry pass every descriptor shares.

### `class NearestAtom` (frozen dataclass)

`distance` (n1,n2,n3), `direction` (n1,n2,n3,3), `atom_index` (n1,n2,n3).
Despite the name it also holds a power-diagram assignment: the arrays refer to
the **assigned** atom, which for the nearest scheme is the nearest one.

### `_image_points(structure, reps)`

Builds all periodic images of all atoms for integer shifts in
[−r, r] per axis (after wrapping fractional coordinates into [0, 1)). Returns
Cartesian points and `owner` (atom index of each image point).

### `_reps_for(distance, structure)`

How many cells of images are needed so that every image within `distance` of
any point in the cell is included. A displacement of length d changes
fractional coordinate j by at most d·|column j of A⁻¹|; atoms and voxels are
both in [0, 1), so `ceil(d·h_j) + 1` shifts per axis suffice.

### `assign_atoms(grid, structure, power_radii=None, chunk, workers)`

1. **Lifting for the power diagram.** With radii r_i, weights w_i = r_i² and
   C = max w, every image gets a fourth coordinate √(C − w_i); a query point
   gets 0. Then the squared 4-D distance is |x − R_i|² − r_i² + C, so the
   ordinary nearest neighbour in 4-D is the power-diagram owner. With no
   radii the fourth coordinate is 0 for everyone, i.e. plain nearest atom.
2. **Provable image range.** Start with one shell of images (reps = 1), build
   a `cKDTree`, query every voxel (in chunks of 2²⁰, `workers=-1`). The largest
   lifted distance found bounds every voxel's true nearest (a better image
   would have a smaller lifted distance, and lifted ≥ Euclidean). If
   `_reps_for(max distance)` asks for more shells than were loaded, widen and
   repeat. The loop always ends, usually after one pass.
3. Recompute Euclidean distance and direction to the chosen image; direction
   is 0 where the distance is below 1e-12.

Why not "round the fractional difference": that is only the minimum image for
near-orthogonal cells. `tests/test_geometry.py::test_fractional_rounding_is_not_enough`
builds a sheared cell where rounding overestimates distances by > 0.1 Å.

### `nearest_atom(grid, structure, **kwargs)`

`assign_atoms` with no radii.

Tests: brute force over 9³ image shells in three cells including a strongly
sheared one; power diagram vs brute force; direction reconstructs the voxel
position; zero direction on a nucleus.

## 9. `shells.py`

### `class ShellMasks`

Three boolean arrays: `core`, `bond`, `interstitial`.

### `class Shells` (frozen dataclass)

`c1`, `c2`, optional `radii`. `__post_init__` enforces 0 ≤ c1 < c2.

- `Shells.scaled(s1, s2, radii)`: radius-scaled mode — the same dataclass with
  `radii` set, where c1/c2 are *multipliers* of each element's radius.
- `atom_cutoffs(structure)`: per-atom (c1_i, c2_i); raises `KeyError` for
  elements without a radius.
- `masks(geometry, structure)`: `r ≤ c1`, `c1 < r ≤ c2`, `r > c2`, with the
  cutoffs of the voxel's assigned atom. Every voxel is in exactly one shell.

The engine caches masks per (shape, c1, c2, radii) — `Shells` is frozen and
hashable except for the `radii` dict, so the cache key uses a sorted tuple of
the radii items.

## 10. `partition.py`

### The interface

`PairChunk(voxel, atom, weight, distance, direction)`: memory-bounded chunks
of voxel-atom pairs. `distance` and `direction` are measured from **that**
atom's image (the one the scheme used) to the voxel. Hard schemes emit one
pair per voxel with weight 1; soft schemes emit several with weights summing
to 1.

`class Partition` (base): `pairs()` abstract; `site_sum(f)` = Σ_k w_i(k) f_k
per atom (via `np.bincount` per chunk); `site_count()` = effective voxel
count per atom.

Every site-resolved quantity in pydemi (Family C, H, spin moments, site
charges) is a weighted sum over pairs, so one code path serves every scheme.

### `class HardPartition`

Wraps a `NearestAtom` assignment (`scheme` = "nearest" or "power").
`pairs()` yields contiguous voxel ranges of 2²⁰. `site_sum` / `site_count`
are overridden with a single `bincount` (faster than iterating pairs).

### `_becke_step(mu)`

Becke's cell function s(μ) = ½(1 − f(f(f(μ)))), f(p) = 1.5p − 0.5p³, written
as `mu * (1.5 - 0.5 * mu * mu)`. The comment explains why: `mu ** 3` goes
through numpy's generic power and was 20× slower (profiling showed 20 of 25 s
in this function).

### `class BeckePartition` (opt-in)

Parameters: `k` (competitors per product, default 60), `cells` (candidate
cells receiving weight, default 8), optional `radii` (heteronuclear size
adjustment), `min_weight` (1e-10).

- `_build()` / `_query(x)`: KD-tree of images, queried for the `k` nearest;
  the same "widen until provably complete" loop as `assign_atoms`, using the
  k-th distance as the bound.
- `_adjust(mu, owners, nc)`: Becke's size adjustment
  ν = μ + a(1 − μ²), χ = R_i/R_j, u = (χ − 1)/(χ + 1), a = u/(u² − 1),
  clipped to |a| ≤ ½. Off when `radii` is None (plain Becke is pure
  geometry).
- `pairs()`: per chunk of 2¹³ voxels, compute μ_ij = (d_i − d_j)/R_ij for the
  `nc` candidate cells against all `k` competitors (array (M, nc, K)), apply
  s, set the diagonal to 1, take products over competitors, normalize over
  candidates, emit weights above `min_weight`.

Why candidates and competitors are separated: if a kept cell's equidistant
competitors are cut off by the K-nearest truncation, that cell's product
misses factors of ≈ ½ and its weight is badly overestimated. Measured: the
weight is always carried by the 8 nearest images (weight beyond them
< 1e-9), but the products need many more competitors. Even so, Becke
converges only algebraically in a periodic solid (max weight error ~2e-2 at
60 competitors, ~1e-3 at 300), which is why it is opt-in (user decision).

### `class HirshfeldPartition` (default smooth scheme)

w_i(r) = ρ_i^free(r) / Σ_j ρ_j^free(r) over every image within `r_cut`
(default 6.5 Å).

- **Density tables.** In `__init__`, each element's radial density is
  log-log interpolated once onto a uniform distance table of 2¹⁵ points from 0
  to r_cut, plus a guard slot of 0 for d ≥ r_cut. `density(e, d)` is then a
  linear lookup (`t = d/step`, integer part and fraction). This replaced
  `np.interp` with log/exp per call, which was 31% of the run time.
- **Blocks.** The grid is processed in 8³-voxel blocks. For each block, the
  candidate images within `r_cut + block_radius` of the block centre come from
  one KD-tree ball query; then all voxel-image distances in the block are
  computed densely (M × C). This replaced per-voxel k-nearest queries
  (about 4 min at 180³) with ~34 s.
- **Empty voxels.** A voxel with no image within r_cut (possible in vacuum)
  is given entirely to its nearest image.
- Image range: built once with `_reps_for(r_cut + block_radius)`.

Why Hirshfeld and not Becke as the default: free-atom densities decay
exponentially, so the weight error falls exponentially with the cutoff
(3.7e-2 at 3.5 Å, 6.8e-5 at 5.5 Å, 3.0e-6 at 6.5 Å on FeNi₃).

Tests: `tests/test_sites_spin.py` (partition of unity, Becke two-atom closed
form, size adjustment direction, opt-in behaviour), `tests/test_reference.py`
(Hirshfeld neutral for the promolecule, detects charge transfer, r_cut
convergence, weights follow the field's electrons).

## 11. `engine.py`

### Field-name constants

`RHO = "rho"`, `MAG = "magnetization"`, `MAG_ABS = "magnetization_abs"`,
`RHO_AE = "rho_ae"`, `ELF = "elf"`, `ELF_DOWN = "elf_down"`,
`POT = "potential"`. Always use these rather than string literals.

### `class Engine`

Constructor arguments: `structure`, optional `fields` dict, `method`
(`"fd"`), `shells`, `spin_mode`, `magnetization_vector` (non-collinear
(3, …) array), `zval` (element → ZVAL), `reference` (free-atom provider).

Instance state:

| Attribute | Content |
|---|---|
| `structure`, `method`, `shells`, `spin_mode` | as given |
| `zval` | element → ZVAL or None (then the default rule applies) |
| `paw_radii` | element → PAW RCORE in Å, or None; set by `from_vasp_dir` |
| `magnetization_vector` | (3, n1, n2, n3) for non-collinear runs |
| `_fields` | name → Field |
| `_grids`, `_geometry`, `_shell_masks`, `_partitions` | caches keyed by shape (+ options) |
| `cache` | shared derived arrays keyed by tuples `(kind, source_field, …)` |
| `_reference` | the reference provider, lazily defaulted |

Methods:

- `add_field(name, values)`: wraps the array in a `Field` on the grid of its
  shape. **Invalidation:** it drops every `cache` entry whose key tuple
  contains `name` (so replacing `rho` drops `("kinetic", "rho")`, but adding
  `elf_d` does not drop the kinetic terms it was computed from — that was a
  real bug found in Phase 2 and is tested).
- `__getitem__`, `__contains__`, `field_names`.
- `_default_shape(shape)`: the explicit shape, else `rho`'s, else the first
  field's.
- `grid(shape)`, `geometry(shape)`, `shell_masks(shape, shells)`: cached
  per shape (and per shells for masks).
- `atom_radii(radii)`: per-atom radii from an element map, default Magpie
  covalent radii (used for the power diagram).
- `augmentation_radii(fallback)`: per-atom PAW augmentation radius (RCORE
  from `paw_radii`) or, when unknown, the covalent radius (or `fallback` for
  elements without one), plus a from-PAW-data flag. Used by the
  non-nuclear-maximum cutoff (§31) and the outside-PAW deformation variant
  (§33).
- `partition(shape, scheme, radii, k, cells, r_cut, part)`: builds and caches
  one partition per distinct configuration. Cache keys:
  - nearest: `(shape, "nearest")`
  - power: `(shape, "power", tuple(radii))`
  - becke: `(shape, "becke", radii or None, k, cells)`
  - hirshfeld: `(shape, "hirshfeld", id(reference), r_cut, part)` — `part`
    ("total" / "valence") selects which electrons the free-atom weights
    describe (§33).

  The new partition object is built in a local called `built`, so the `part`
  string argument is never shadowed.
- `reference` (property): the provider; defaults to the process-wide
  `AtomicLDAReference` from `atoms.reference.default_reference()`.
- `valence()`: element → ZVAL, from `zval` or `default_zval`.

Constructors:

- `from_charge_density(cd, **kwargs)`: adds `rho` (or `rho_ae` if
  `cd.all_electron`), and for collinear runs `magnetization` and
  `magnetization_abs = |m|`; for non-collinear runs stores the vector and adds
  `magnetization_abs = |m_vec|`.
- `from_chgcar(path, species, read_spin, **kwargs)`.
- `from_file(path, density_unit, **kwargs)`: `.cube` / `.cub` / `.xsf` go
  through `io.grids.read_density`, anything else through `from_chgcar`.
- `from_vasp_dir(directory, species, **kwargs)`:
  1. reads `CHGCAR`;
  2. picks `POTCAR`, else `OUTCAR`, and parses ZVAL (unless `zval=` was
     passed) and RCORE; each is used only if it has one entry per element
     (otherwise a warning, and ZVAL falls back to the rule);
  3. adds `AECCAR0 + AECCAR2`, `ELFCAR` (and `elf_down`), `LOCPOT` when
     present, each checked to have the same lattice as the CHGCAR.

## 12. `elements.py`

Lookup into `data/magpie_elements.csv` (Magpie tables from matminer 0.10.1).

- `_tables()` (cached): parses the CSV once into symbol → row index and
  property → numpy column. Comment lines start with `#`.
- `element_property(symbol, prop, impute_nan=True)`: value, or the mean over
  all elements when the table has no entry (matminer's `impute_nan=True`).
  `KeyError` for symbols not in the table (e.g. placeholders "X0").
- `atomic_number(symbol)`: row index + 1.
- `covalent_radii(elements)`: Magpie `CovalentRadius` (pm) / 100 → Å.

Why vendored: installing matminer would have downgraded pandas in the shared
environment; the six properties Tier 2 uses are copied verbatim with the BSD
licence, and values match matminer to 1e-14 (tested when matminer is
importable).

---

# Part III — Input / output

## 13. `io/vasp.py`

### Data classes

- `VolumetricData(structure, blocks, kind, source)`: every data block of one
  file, each an (n1, n2, n3) array already scaled.
- `ChargeDensity(structure, total, magnetization, spin_mode, all_electron,
  source, extra)`: the interpreted CHGCAR. `magnetization` is None, an
  (n1, n2, n3) array (collinear) or a (3, n1, n2, n3) array (non-collinear).
  `spin_mode` is one of `SPIN_NONE`, `SPIN_COLLINEAR`, `SPIN_NONCOLLINEAR`.

### `_parse_header(lines, species)`

Parses the POSCAR part of any VASP volumetric file:

1. **Scale line.** One number = global scale; negative = target volume
   (scale = (|s| / det A)^{1/3}); three numbers = per-axis scale.
2. **Lattice** (lines 2–4) times the scale.
3. **Species line or not.** If line 5 is all integers it is VASP 4 (no
   species line): labels come from `species=` or become placeholders `X0, X1,
   …` with a warning. POTCAR-style labels (`Fe_pv`, `O/abc`) are cleaned to
   the element symbol.
4. **Counts**, optional **Selective dynamics** line, the **Direct / Cartesian**
   line (Cartesian coordinates are multiplied by the scale — VASP convention;
   an early test had this wrong, not the code).
5. Positions (first three tokens of each line; extra tokens such as `T T T`
   or labels are ignored).

Returns the `Structure` and the index of the next line.

### `_find_dims(lines, start, dims=None)`

Finds the next grid-dimension line: exactly three integer tokens, and, when
`dims` is given, equal to it. This is what skips everything VASP writes
between blocks: `augmentation occupancies i n` lines (four tokens), their
float lines, and the per-atom magnetic-moment line of spin-polarized CHGCARs
(floats, not integers).

### `_read_block(lines, start, n_values)`

Fast path: from the number of values on the first data line, compute how many
lines the block needs, join them, and parse with `np.fromstring(sep=" ")` (C
parser). Fallback for irregular line lengths: accumulate line by line.
Raises if the block is truncated.

### `read_volumetric(path, scaled_by_volume, species, kind, max_blocks)`

Reads the header, then loops: find dims → read block → reshape with
`order="F"` (VASP is x-fastest) → divide by the cell volume if the file is
stored as ρ·V (CHGCAR, AECCAR) → look for the next block with the same
dims. `max_blocks=1` skips spin data.

### File-specific readers

- `_charge_from_blocks(vol)`: 1 block → non-spin; 2 → collinear (block 2 is
  m = ρ↑ − ρ↓); 4 → non-collinear (blocks 2–4 are m_x, m_y, m_z); any other
  count raises "Refusing to guess" rather than silently misreading.
- `read_chgcar(path, species, read_spin=True)`.
- `read_aeccar(aeccar0, aeccar2)`: sums core + valence, checks shapes and
  lattices, keeps the core density in `extra["core"]`.
- `read_elfcar(path)`: unscaled; one or two (spin) blocks; usually on the
  coarse grid.
- `read_locpot(path)`: unscaled (eV).

### `write_volumetric(path, structure, blocks, scaled_by_volume, per_line, comment)`

Minimal VASP 5 writer (no augmentation section) used by tests and
`pydemi resample`. Atoms are written grouped by element (VASP requires
grouping), so a structure with interleaved species is reordered on write.

Tests: `tests/test_io.py` — round trip, Fortran order check (first line of
values must be `0 1 2 10 11` for f = i + 10j + 100k), collinear with inserted
augmentation and magmom lines, non-collinear, 3-block refusal, volume
scaling, VASP 4 + negative scale + Selective dynamics + Cartesian, POTCAR
labels, ELFCAR with 10 values per line, AECCAR sum, `from_vasp_dir`.

## 14. `io/grids.py`

Cube and XSF, which cover Quantum ESPRESSO (`pp.x`) and ABINIT (`cut3d`).

- `_UNITS`: `"e/bohr^3"` → multiply by 1/a₀³; `"e/A^3"` → 1; `None` → 1
  (non-density data). `_scale(unit)` validates.
- `read_cube(path, density_unit)`: line 3 = number of atoms (negative for
  orbital cubes, which add one line of orbital indices) and origin; lines 4–6
  = count and voxel vector per axis (positive count → bohr, negative → Å);
  atom lines `Z charge x y z`; data in C order (z fastest). The lattice is
  count × voxel vector. Fractional coordinates are taken relative to the
  origin and wrapped.
- `write_cube(...)`: standard 6 significant digits (`%13.5E`, the format's
  convention — kept for VESTA / VMD compatibility, which is why the round-trip
  test tolerance is 1e-5).
- `read_xsf(path, density_unit)`: `PRIMVEC` (lattice), `PRIMCOORD` (atoms,
  Z or symbol), first `BEGIN_DATAGRID_3D` block: counts, origin, three
  spanning vectors, data in Fortran order until `END_DATAGRID_3D`. XSF
  "general grids" repeat the first plane at the end of every axis, so the
  last index of each axis is dropped.
- `write_xsf(...)`: pads with `mode="wrap"` to add the periodic duplicate.
- `read_density(path, density_unit)`: dispatch by extension to a
  `ChargeDensity`.

---

# Part IV — Free-atom references

## 15. `atoms/solver.py`

A spherical, non-spin-polarized, non-relativistic Kohn–Sham LDA atom, written
because no atomic-DFT package was installed and installing one would have
changed the shared environment. It produces orbital-resolved radial densities
for the promolecule (Family A) and Hirshfeld weights.

### The radial equation on a log grid

r = eˣ, u(r) = rP(r) = e^{x/2}φ(x). The radial equation
−u''/2 + [V + l(l+1)/2r²]u = εu becomes

    −φ''/2 + [(l + ½)²/2 + r²V] φ = ε r² φ

— a generalized symmetric eigenproblem with a tridiagonal left side (second
differences in x) and a diagonal metric r². Scaling rows and columns by 1/r
gives an ordinary symmetric tridiagonal matrix (`diag`, `off` in `_solve_l`).
With ψ = rφ, Σ ψ² h = 1 and the orbital density is ψ²/(4πr³).

### `RadialGrid` (frozen dataclass)

x from ln(1e-10) to ln(80) bohr, 8000 points. The inner boundary is far
inside because s orbitals behave as φ ∝ r^{1/2} toward the nucleus: a
Dirichlet boundary at 1e-7 bohr cost 0.4 mHa on Ne; at 1e-10 the error is
negligible up to Z ≈ 100.

### `AtomResult`

Z, r (bohr), `orbitals` = list of (n, l, occupation, eigenvalue, density(r)),
total density, total energy, iterations, converged flag, `extra`
(electron count, residual). `orbital_density(selection)` sums chosen (n, l).

### Exchange–correlation

- `_lda_xc(n, correlation)`: Slater exchange ε_x = −¾(3n/π)^{1/3},
  v_x = −(3n/π)^{1/3}; correlation PW92 (default) or VWN5.
- PW92: ε_c(rs) = −2A(1 + α₁rs) ln[1 + 1/(2A(β₁rs^{½} + β₂rs + β₃rs^{3/2} + β₄rs²))],
  v_c = ε_c − (rs/3) dε_c/drs, with the analytic derivative.
- `_vwn_c(rs)`: VWN5 paramagnetic, analytic derivative in x = √rs,
  v_c = ε_c − (x/6) dε_c/dx. VWN exists to validate against NIST, whose LDA
  tables use it (PW92 vs NIST differs by ~0.4 mHa on He, which is how the
  difference was diagnosed).

### Hartree and initial guess

- `_cumtrapz(y, x)`: cumulative trapezoid.
- `_hartree(n, r, x)`: V_H(r) = 4π[(1/r)∫₀^r n r'² dr' + ∫_r^∞ n r' dr'] with
  dr = r dx (so the integrands are n r³ and n r² in x).
- `_thomas_fermi_guess(Z, r)`: screened nuclear potential −Zφ_TF(r/b)/r from a
  rational fit to the Thomas–Fermi function, b = 0.8853 Z^{−1/3}, floored at
  −1/r.

### `_solve_l(V, l, grid, n_states, Z)`

Builds the scaled tridiagonal matrix and calls
`scipy.linalg.eigh_tridiagonal(select="i", lapack_driver="stebz",
tol=1e-13·Z²)`. **The explicit tolerance is essential**: the matrix is
extremely graded (diagonal ~1e25 at 1e-10 bohr against eigenvalues of order
1), and LAPACK's default bisection tolerance ε‖C‖ would be larger than the
eigenvalues themselves (the first version returned +218 Ha for the He 1s
level). Sturm-count bisection is accurate on graded tridiagonals when given
an absolute tolerance. The MRRR driver (`stemr`) was tried and fails for Z = 1.

### `solve_atom(Z, configuration, grid, mixing, tol, max_iter, correlation)`

Configuration: list of (n, l, occupation), l as int or s/p/d/f; fractional
occupations are allowed (spherical average of open shells). For each l the
number of states needed is max(n) − l.

SCF loop: solve every l-channel in the current potential, build ρ from the
occupied orbitals (orbital k of channel l is n = l + 1 + k), compute
V_out = −Z/r + V_H + v_xc, stop when the density-weighted potential residual
√(Σ (V_out − V_in)² ρ r³ h / N) < tol, else linear mixing (0.4). Converges in
~40 iterations for every element tested (He … Pu).

Total energy: Σ f ε − ½∫ρV_H + ∫ρ(ε_xc − v_xc), with ∫ f dV = Σ f 4πr³ h.

Validation (`tests/test_reference.py`): hydrogen-like eigenvalues to 2e-5
relative; He and Ne against NIST LDA (VWN) — total energies to 2e-5 and 1e-4
Ha, 1s and 2p eigenvalues to 5e-5 Ha.

## 16. `atoms/reference.py`

### Constants and helpers

- `NOBLE`: noble-gas Z → symbol.
- `CACHE_VERSION = 1`: part of the disk-cache file name, bump it when the
  solver changes.
- `_configuration(element)`: Z and the ground-state configuration from
  `pymatgen.core.Element.full_electronic_structure`.
- `default_zval(element)`: electrons outside the preceding noble-gas core,
  with a filled f¹⁴ counted as core. Matches the standard VASP POTCARs for
  most transition metals (Fe 8, Co 9, Ni 10, Cr 6, Cu 11, Hf 4, Ta 5, W 6,
  Au 11); differs for `_pv` / `_sv` / `_d` variants and lanthanides — hence
  ZVAL from POTCAR / OUTCAR takes precedence.
- `read_potcar_zval(path)`: ZVAL per dataset from the `POMASS = …; ZVAL = …`
  header line (present in both POTCAR and OUTCAR). The OUTCAR also has a later
  summary line `ZVAL = 11.00 11.00 4.00`; matching only the header line avoids
  counting it. Falls back to any `ZVAL =` if no header line exists.
- `read_potcar_rcore(path)`: `RCORE = …` per dataset (bohr), the outermost PAW
  cutoff radius.

### `class AtomicLDAReference` (`pseudized = False`)

- Constructor: correlation, grid, `cache_dir` (default `$PYDEMI_CACHE_DIR` or
  `~/.cache/pydemi`; empty string disables the disk cache).
- `_cache_path(element)`: file name encodes element, correlation, grid and
  cache version, so a changed grid never reads a stale file.
- `atom(element)`: memory memo → disk cache → solve. Disk writes are
  **write-then-rename** (`tmp` file with the PID, then `os.replace`) so
  parallel batch workers never read a half-written file. Warns if the SCF did
  not converge.
- `radial(element, part, zval)`: (r in Å, density in e/Å³). `part="total"`
  uses all occupations; `part="valence"` fills ZVAL electrons from the
  highest-energy orbital down (warns if ZVAL splits a shell, raises if ZVAL
  exceeds Z).
- `electrons(element, part, zval)`.

### `class IsolatedAtomReference` (`pseudized = True`)

Reference route 3: element → isolated-atom CHGCAR (one atom in a box, same
POTCAR as production), optional element → (AECCAR0, AECCAR2). `_profile`
bins the density radially about the atom (400 bins up to half the shortest
box height, via `nearest_atom` + `np.bincount`). Because it carries the same
PAW pseudization as the crystal CHGCAR, it is the consistent reference for
the CHGCAR route of Family A and for Hirshfeld charges from CHGCAR.

The `pseudized` class attribute is how `deformation_family` and
`hirshfeld_charges` decide whether to warn.

### `default_reference()` and `resolve_zval(elements, zval)`

`default_reference` is `lru_cache(maxsize=1)` — one shared LDA reference per
process. `resolve_zval` merges explicit values with the default rule.

---

# Part V — Descriptors

## 17. `descriptors/__init__.py` — the dispatcher

- `FAMILIES`: the 17 family keys in computation order:
  `tier1 tier2 tier3 B F2 F3 F4 F5 F6 I1 C E H G I2 A D`.
- `_D_INPUTS = ("tier1", "tier2", "F2", "I2")`: what Family D needs.
- `_family_b(engine, field, shells)`: ELF descriptors on ELF_D always
  (prefix `ELFD`), and on the real ELFCAR plus the fidelity check when `elf`
  is loaded.
- `_RUNNERS`: family key → `callable(engine, field, shells)`. Lambdas adapt
  families that take fewer arguments. Family A is called with `field=None`
  (AECCAR if present, else NaN).

### `compute_descriptors(engine, families, kinds, field, shells, calibrations)`

1. Remember the requested families; if D is requested, add its inputs to the
   families actually computed (their values are needed but not returned).
2. Validate family names.
3. Tier 1 is computed if tier1 or tier3 is requested; Tier 2 if tier2 or
   tier3; Tier 3 last among the tiers (it reads Tier 1 and 2 values from the
   same dict).
4. Every other family through `_RUNNERS`, in `FAMILIES` order.
5. Family D via `calibration.calibration_family` with the accumulated values.
6. Output: registry names of the requested `kinds` (opt-in included, so an
   explicitly requested Becke run is kept), only those present in the values
   and belonging to a **requested** family, in registry order.

Missing optional data means a name is simply absent (e.g. no `ELF_*` without
ELFCAR); the batch writer then fills the CSV cell with nothing, read back as
NaN.

## 18. `descriptors/registry.py`

### Kinds

`DESCRIPTOR`, `VARIANT`, `CROSS_TERM`, `PREPROCESSING`, `METADATA` (scalar,
`SCALAR_KINDS`), plus `FIELD`, `SITE`, `DATASET` (non-scalar, documentation
only).

### `DescriptorInfo` (frozen dataclass)

`name, entry, family, kind, inputs, formula, note, opt_in`.

### Entry generators

- `_elf_entries(prefix, inputs, note)`: the seven Family B names for
  `prefix` = `ELF` or `ELFD` (`f_{p}_localized`, `{p}_bond_avg`, `zeta_{p}`,
  `{p}_threshold_sweep_025/_075/_inflection`, `{p}_core_valence_contrast`).
- `_SITE_ENTRY = {"m1": 53, "f_bond": 54, "zeta": 55}` and `_site_entries(suffix,
  family, note)`: the seven Family C statistics per site quantity.
- `_CELL_FORMULA` and `_partition_entries(scheme, entry, opt_in)`: the Tier 1
  radial descriptors plus all Family C statistics with a `_power` /
  `_hirshfeld` / `_becke` suffix (Family H); Becke entries carry
  `opt_in=True`.

### `_ENTRIES`, `REGISTRY`, `describe`, `names`

`_ENTRIES` is the ordered list, and output order follows it: Tier 1, Tier
2, Tier 3, Family B (ELF names, then ELFD names), F1–F6, I1, C, E, H (power,
Becke, Hirshfeld), A, D, the entry-73 fidelity names (family B), G, I2 —
the order in which the families were added. `REGISTRY` maps name → info. `names(family, kinds=SCALAR_KINDS, opt_in=False)` filters in order;
`kinds=None` returns every kind.

The registry currently holds 257 entries; 222 are scalar and on by default,
and 28 more (the Becke variants) are scalar but opt-in.

## 19. `descriptors/primitives.py`

Field-agnostic reductions reused across families:

- `safe_div(a, b)`: NaN when b is 0 or either value is not finite.
- `radial_moments(weights, r)`: (m1, m2, m2 − m1²) with weights w.
- `shell_fractions(weights, masks)`: shares of total weight in core / bond /
  interstitial.
- `gradient_anisotropy(gradient, direction, mask)`: ζ = 1 − Σ|g·r̂| / Σ|g|,
  optionally over a mask (used for ζ on ρ, and on ELF with the core excluded).
- `negative_fraction(values, weights)`: voxel fraction (or weight share) with
  values < 0 — `lnf` and `lnf_rho`.
- `negative_magnitude_share(values)`: Σ_{v<0}|v| / Σ|v| — `lap_concentration`.

## 20. `descriptors/tier1.py`

`tier1(engine, field="rho", shells)` computes entries 1–15 plus variants from
one field: Q_tot, (m1, m2, σ²) from `radial_moments(ρ, r)`, shell fractions,
ζ, lnf (voxel count) and lnf_rho (charge weighted) from the Laplacian's sign,
then the derived ratios with `safe_div`. Variants:

- `moment_ratio` = m2/m1 (as specified, a length) and
  `moment_ratio_scale_free` = m2/m1² (invariant under uniform scaling).
- `zeta_over_rvar` = ζ/σ² and `zeta_over_sigma_r` = ζ/σ (an inverse length).
- `lap_concentration` is identically ½ for any periodic field (∫∇²ρ = 0 over
  the cell), so `lap_concentration_valence` restricts it to the non-core
  voxels, where flux through the core boundary breaks the balance.

`field="rho_ae"` runs the same machinery on the all-electron density.

## 21. `descriptors/composition.py`

- `composition_features(amounts, impute_nan)`: entries 16–27 and 29–30 from
  element → amount (normalized internally): means, maxima and ranges of
  Magpie AtomicWeight, Electronegativity, NValence, AtomicRadius, Row;
  `n_elements`; Pauling-type `ionicity` = 1 − exp(−Δχ²/4); `is_f_block`
  (all elements f-block) and `has_f_block` (any).
- `crystal_system(structure, symprec=0.01)`: spglib space group → crystal
  system 1–7 (triclinic … cubic). spglib ≥ 2.5 returns a dataclass and warns
  on every call about legacy error handling; the call is wrapped in
  `warnings.catch_warnings` to silence it locally, and both the dataclass and
  the dict API are handled.
- `tier2(structure)`: the two together.

## 22. `descriptors/tier3.py`

`tier3(engine, t1, t2, field, shells)`: `laplacian_std` over all voxels,
`laplacian_std_valence` over non-core voxels (the core-excluded form the
specification recommends), the cross terms 33–37 from Tier 1 and 2 values,
and the transforms `sqrt_zeta`, `log_lnf` = ln(lnf).

## 23. `descriptors/kinetic.py` — F1 (ELF_D) and F5

Everything here is in atomic units.

- `RHO_FLOOR_AU = 1e-8`: below this density a voxel is "invalid" (the
  gradient expansion is meaningless; |∇ρ|²/ρ would blow up).
- `KineticTerms` (dataclass): ρ (clipped at 0), |∇ρ|², ∇²ρ, `valid`,
  t_F = C_Fρ^{5/3}, `weizsacker8` = |∇ρ|²/8ρ, and g = t_P (NaN where invalid).
- `kinetic_terms(engine, field)`: converts ρ, |∇ρ|², ∇²ρ with the constants of
  §46 and computes g = C_Fρ^{5/3} + |∇ρ|²/(72ρ) + ∇²ρ/6. **Cached** in
  `engine.cache[("kinetic", field)]`, so ELF_D, F5 and F3 share it.
- `elf_d_values(kt)`: χ = (g − |∇ρ|²/8ρ)/(C_Fρ^{5/3}), ELF_D = 1/(1 + χ²),
  0 where invalid.
- `elf_d_field(engine, field)`: registers the result as the field `elf_d`
  (or `elf_d_<field>`), so Family B, the fidelity check and anything else can
  treat it like a real ELFCAR.
- `energy_densities(engine, field)`: g, v = ∇²ρ/4 − 2g, H = g + v.
- `energy_family(engine, field, shells)`: over valid bonding-shell voxels,
  `f_H_negative`, `H_bond_mean` (hartree/bohr³), `G_over_rho` (hartree/e).

Tests: independent atomic-unit evaluation on a Gaussian specified in bohr; the
uniform electron gas (ELF_D = ½, H = −C_Fρ^{5/3}); below-floor voxels
excluded; the cache survives `elf_d` registration but not replacing `rho`.

## 24. `descriptors/elf.py` — Family B and entry 73

- `threshold_sweep(elf_bond, thresholds)`: f(t) = fraction of bonding-shell
  voxels with ELF > t, via `searchsorted` on the sorted values.
- `sweep_inflection(elf_bond, bins=50)`: the threshold of steepest descent of
  f(t), which is the mode of the ELF distribution (−df/dt is the histogram).
- `elf_family(engine, field, prefix, shells)`: entries 48–52 on the field's
  own grid (geometry and masks for that shape): `f_{p}_localized` (ELF > 0.5
  in the bond shell), `{p}_bond_avg`, `zeta_{p}` (anisotropy of ∇ELF with the
  core shell excluded — ELF gradients near cores are sharp and noisy), sweep
  points 0.25 / 0.75 and inflection, core/valence contrast.
- `resample(values, shape)`: exact stride subsampling when the target divides
  the source (ELFCAR's NGX is typically NGXF/2), else periodic trilinear
  (`ndimage.map_coordinates(order=1, mode="grid-wrap")`).
- `elf_fidelity(engine, elf_field, elfd_field, shells)`: resamples ELF_D onto
  the ELFCAR grid and reports Pearson r and MAE over all voxels and over the
  bonding shell.

## 25. `descriptors/potential.py` — F2

- `hartree_potential(engine, field)`: V_H(G) = 4π·14.40·ρ(G)/|G|² (eV), G = 0
  term set to 0 (cell-average potential zero); registered as the field
  `hartree_potential`.
- `fourier_interpolate(values, frac_points, batch=256)`: exact
  trigonometric interpolation at arbitrary points. Separable evaluation: the
  first contraction over the last axis is one matrix product for a whole
  batch of points ((n1·n2, n3) @ (n3, M)), the other two are einsums. Exact at
  grid points; exact everywhere for band-limited fields (tested).
- `site_potentials(engine, source, field)`: potential at each nucleus from the
  Hartree solve or the LOCPOT (entry 74).
- `_stats(...)` and `potential_family(engine, field, shells)`: `VH_spread`,
  `VH_int_min` always; `V_spread`, `V_int_min` when a LOCPOT is loaded.
  `*_int_min` is relative to the cell average (the G = 0 convention).

Tests include the analytic potential of a Gaussian with a neutralizing
background, V(0) − V(r) = kN[2√(α/π) − erf(√α r)/r] − (2π/3)k(N/V)r², to
0.2% — which pins the sign, 4π and the eV·Å constant.

## 26. `descriptors/hessian.py` — F3 and F4

- `reduced_gradient(engine, field)`: s = |∇ρ|/(2(3π²)^{1/3}ρ^{4/3}), ∞ where
  invalid (from the cached kinetic terms).
- `nci_family(engine, field, s_max=0.5, rho_max_au=0.05)`: NCI voxels are
  s < 0.5 and ρ < 0.05 a.u.; `f_NCI`, the attractive share (λ₂ < 0) and the
  mean of sign(λ₂)ρ.
- `ellipticity_family(engine, field, shells)`: over bonding-shell voxels
  with λ₂ < 0, ε = λ₁/λ₂ − 1; mean, std and **median** (the median was added
  because a few voxels with λ₂ → 0⁻ dominate the mean: over the 6,057
  structures of the dataset the mean is typically 11× the median, and it
  exceeds three times the median in 98.6% of them).

## 27. `descriptors/information.py` — F6

On the shape function ρ̃ = ρ/N in bohr units: Shannon entropy
−∫ρ̃ ln ρ̃, Fisher information ∫|∇ρ̃|²/ρ̃ (voxels above the density floor),
disequilibrium ∫ρ̃², LMC complexity D·e^S (unit-free). Negative ρ is clipped
to 0 (0·ln 0 = 0). Tested against the Gaussian closed forms
S = 1.5(1 + ln(π/α)), I_F = 6α, D = (α/2π)^{3/2}.

## 28. `descriptors/anisotropy.py` — I1

`anisotropy_tensor(engine, field)`: T = gᵀg / trace, from the flattened
gradient (NaN matrix if the gradient is zero everywhere).
`anisotropy_family`: eigenvalues descending and fractional anisotropy
√(3/2)‖T − I/3‖_F / ‖T‖_F. Isotropic → ⅓, ⅓, ⅓ and FA = 0; a density varying
along one reciprocal direction → FA = 1 (tested in a triclinic cell).

## 29. `descriptors/sites.py` — Families C and H

### `site_sums(engine, field, scheme, shells, **partition_kwargs)`

One pass over the partition's pairs accumulating, per atom, weighted sums of
ρ, ρr, ρr², ρ in core / bond / interstitial (using the cutoffs of *that*
atom and the distance *to that atom*), |∇ρ·r̂| and |∇ρ|. Returns the sums plus
dV. `_hirshfeld_part(scheme, field, kwargs)` injects `part="valence"` or
`"total"` for Hirshfeld so the weights describe the same electrons as the
field.

### `site_values(sums)` and `whole_cell_values(sums)`

Per-site X^(i) (charge, m1, m2, σ², shell fractions, ζ) and the whole-cell
versions obtained by summing over atoms first. With the nearest partition,
`whole_cell_values` reproduces Tier 1 to 1e-12 (tested).

### `variance_decomposition(x, groups)`

Population statistics over finite values:
within = Σ_e w_e Var_e(x), between = Σ_e w_e (x̄_e − x̄)², w_e = atom
fraction — these add up exactly to Var(x) (tested). The specification writes
an unweighted mean over elements for the within term; the weights are what
make the decomposition exact. `within` and `within_share` are **NaN when no
element has two or more sites** (a 0 would read as "no disorder"); the share
is also NaN when the total variance is at round-off level.

### `site_statistics(values, species, quantities, suffix)`

For each X ∈ {m1, f_bond, ζ}: std, range, max, min, within, between, share,
ignoring NaN sites (e.g. an empty power cell).

### Family functions

- `site_family(engine, field, shells)`: Family C with the nearest partition,
  plus `n_atoms`.
- `partition_family(engine, field, shells, schemes=("power", "hirshfeld"),
  **partition_kwargs)`: Family H — whole-cell radial descriptors and Family C
  statistics per scheme with a `_<scheme>` suffix. Becke is included only if
  requested; `k`, `cells`, `radii` go to Becke and `r_cut`, `part` to
  Hirshfeld.
- `site_charges(engine, scheme, field)`: electrons per atom.
- `hirshfeld_charges(engine, field)`: q_i = N_i − ∫w_iρ with N_i = Z (AECCAR)
  or ZVAL (CHGCAR). Warns on CHGCAR with an all-electron reference: the
  pseudized valence pushed into the bonding region is credited to atoms with
  diffuse free-atom valence (on 31 VASP binaries the sign of the charge
  transfer followed electronegativity in only 58% of cases).

## 30. `descriptors/spin.py` — Family E

- `MAGNETIC_TOL = 0.01` μ_B per atom.
- `_magnetization(engine)`: (n_comp, …) — one component collinear, three
  non-collinear.
- `site_moments(engine, scheme)`: μ_i = Σ_{k∈i} m dV per component, (n,) or
  (n, 3); zeros for non-spin runs.
- `spin_family(engine, field, shells)`:
  - non-spin: every value 0 with `is_spin_polarized = 0` (the specification
    asks for a sentinel, not NaN);
  - otherwise M_abs = Σ|m|dV, M_net = |Σ m dV| (vector norm),
    `mu_site_std` = √(mean |μ_i − μ̄|²) (the signed std when collinear);
  - if M_abs < 0.01 μ_B per atom: `is_magnetic = 0` and the ratio-type values
    (m1_spin, σ²_spin, f_bond_spin, frustration, spin-charge correlation) are
    0 — otherwise they would be ratios of numerical noise;
  - else radial moments of |m|, the bond-shell share of |m|,
    `spin_frustration` = 1 − |Σμ_i| / Σ|μ_i| with vector norms (perpendicular
    moments give 1 − 1/√2, not a collinear misreading), Pearson r(ρ, |m|).

## 31. `descriptors/topology.py` — Family G

The largest descriptor module; five algorithms.

### Constants

- `NNM_MIN_CHARGE = 0.01` e and `NNM_MIN_PERSISTENCE = 0.1`.
- `_POSITIVE`, `OFFSETS`: the Freudenthal triangulation's 7 edge vectors and
  their negatives (14 neighbours). `_FULL` = 2¹⁴ − 1.

### Critical-point census (entries 92–93)

- `_link_components()` (cached): for every one of the 2¹⁴ subsets of the
  14-vertex link, the number of connected components (link vertices u, v are
  adjacent when v − u is itself an edge vector). A 16,384-entry lookup table
  built once (0.14 s).
- `_shift(a, off)`: the field at voxel k + off (periodic `np.roll`).
- `lower_link_mask(values)`: per voxel, a uint16 with bit k set when neighbour
  k is lower. "Lower" is lexicographic in (value, flat index) — a symbolic
  perturbation that makes every comparison strict, so ties cannot break the
  Euler identity.
- `morse_census(values)`: min = empty lower link, max = empty upper link,
  1-saddle multiplicity = components(lower) − 1, 2-saddle multiplicity =
  components(upper) − 1. Returns the counts, `euler_consistency` =
  n_max − n_s2 + n_s1 − n_min, the maxima map and the mask.

With this consistent piecewise-linear census the Euler sum equals χ(T³) = 0
**exactly for any sampled field** (Banchoff's theorem; tested on random
fields with ties). The specification hoped a nonzero value would flag a
coarse grid; it cannot, so the entry is registered as a self-check and the
convergence report (§38) does the grid-adequacy job.

### Basins

`ascent_basins(values, lattice, lower_mask)`: each voxel points to the upper
neighbour of steepest slope (Δρ/|d| with Cartesian step lengths, so skewed
cells are handled); pointer jumping (`ptr = ptr[ptr]`) until stable gives each
voxel's maximum. Every basin ends at a census maximum (tested).

### Persistence

`maxima_persistence(values, basin)` returns, for every maximum, the level at
which its superlevel-set component merges into one with a higher peak, and
the peak it merges into.

1. Relabel basins 0 … P−1.
2. For each of the 7 positive edge directions, collect edges whose end voxels
   lie in different basins, weighted by min(end values); keep the maximum
   weight per basin pair (lexsort + first-occurrence).
3. Kruskal in decreasing weight order with union-find; when two components
   join, the one with the lower peak (ties by index) dies at that weight
   (elder rule).

Why this is exact: every voxel of a basin reaches its maximum by an ascending
path, so within {ρ ≥ c} a basin's voxels are connected to its peak; two
components are joined at level c exactly when some boundary edge has both ends
≥ c. So the maximum spanning forest of the basin graph is the merge tree.
Tested against an independent voxel-level union-find on random grids — equal
to round-off for every maximum.

### Percolation (90–91)

- `_wraps(mask)`: face-connected labelling with `ndimage.label`, then a
  union-find over labels that also tracks each label's integer lattice offset
  relative to its root. For each axis, labels touching across the periodic
  face are joined with offset e_axis. If two labels are already in the same
  set, the winding w = oa + e − ob is nonzero exactly when the cluster meets
  its own periodic image; the axes with w ≠ 0 are spanned. Diagonal windings
  (1, 1, 0) span both a and b; a finite rod crossing the boundary does not
  span (tested).
- `percolation_levels(values)`: bisection over the sorted unique values,
  jointly for the three axes (each labelling updates every axis whose interval
  contains the midpoint). Returns the supremum of spanning levels (the value
  at which the network breaks) — the specification writes min{…}, which is
  degenerate.

### `nuclear_radii(engine, shells)`

Per-atom cutoff for "non-nuclear": max(c₁, R_PAW), with R_PAW from
`engine.augmentation_radii(fallback=c₁)` — RCORE when known, else the Magpie
covalent radius (or c₁ for placeholder elements). Returns the radii and whether PAW data were used.
Motivation: in CaSi₃Pt the Si atoms have **no maximum at the nucleus** — the
pseudo-density even goes negative there — only lobes at 0.81–0.83 Å, which a
0.8 Å cutoff counted as 8 non-nuclear maxima holding 6.3 e.

### `topology_family(engine, field, shells, r_cut, min_basin_charge, min_persistence)`

Runs percolation, the census, basins and persistence, then:

- `n_NNM` / `Q_NNM`: maxima beyond their atom's cutoff and the charge of their
  basins.
- `n_NNM_significant`: those whose basin holds ≥ 0.01 e.
- `n_NNM_persistent` / `Q_NNM_persistent`: maxima kept by relative
  persistence (peak − merge)/peak ≥ 0.1; each discarded maximum's basin
  charge follows its absorber chain to a kept peak (the global maximum is
  always kept). A kept non-nuclear maximum is counted only if its merged
  basin holds ≥ 0.01 e (`min_basin_charge`): relative persistence alone
  admitted up to 48 tiny peaks holding < 1 e in Yb compounds. Maxima that
  remain after both filters can be real density features, e.g. the persistent
  maxima 1.455 Å from Zn in YbPrZn₂, outside the 1.22 Å Zn PAW sphere. Added after the dataset run found Mg₃(TiAl₉)₂ with 576
  "significant" ripple maxima in a flat free-electron sea; YMg₃ goes from 124
  significant to 0 persistent.
- `rho_min`, `rho_min_ratio` (as specified) and `rho_min_int`,
  `rho_min_int_ratio`: the minimum over voxels farther than max(c₂, R_PAW)
  from their nucleus — the interstitial floor, outside every PAW sphere.
  Added because 61% of the dataset's CHGCARs have negative pseudo-density near
  a nucleus, which made the plain `rho_min` a PAW artefact.
- `rho_int_mean`, `paw_radii_known`.

## 32. `descriptors/bonds.py` — I2

- `BondCensus(i, j, shift, length)`; `midpoints_frac(structure)` =
  ½(u_i + u_j + shift).
- `bond_census(structure, tol=0.1)`: per-atom first shell, d ≤ (1 + tol)·d_i
  with d_i the atom's nearest-neighbour distance, including periodic images
  of the atom itself in small cells. Images are generated with widening reps
  until the shell radius is covered. Each bond is stored once under a
  canonical key ((i, j, t) vs (j, i, −t)); shifts computed for wrapped
  coordinates are converted back so midpoints use the stored (possibly
  unwrapped) coordinates (tested with unwrapped input).
- `bond_family(engine, field, tol)`: Fourier-interpolated ρ at every midpoint
  → `rho_mid_mean`, `rho_mid_std`, plus `n_bonds` and `bond_length_mean`
  (the d of Cohen's formula and the 106 proxy). fcc gives 24 bonds at a/√2.

## 33. `descriptors/deformation.py` — Family A

- `form_factor(r, n, G)`: f(G) = 4π∫n r² sin(Gr)/(Gr) dr on the log grid
  (dr = r·d(ln r), `np.sinc` handles G = 0).
- `reference_part(field)`: "total" for `rho_ae`, "valence" otherwise.
- `promolecule(engine, field, n_table)`: in reciprocal space — each element's
  form factor tabulated on 4000 |G| points and interpolated to every grid G,
  times its structure factor Σ exp(−iG·R_i) (separable per-axis phases on the
  rfft half grid), summed and inverse-FFT'd. This is the exact periodic sum,
  band-limited like VASP's own grid densities, and integrates exactly to
  Σ N_i. Registered as `promolecule_total` / `promolecule_valence`.
- `deformation_density(engine, field)`: (ρ − ρ_pro, field used).
- `deformation_family(engine, field, shells)`: entries 40–47 plus
  `def_charge_mismatch` (∫Δρ, should be 0) and `def_all_electron`.
  - `field=None`: AECCAR if loaded, else **NaN everywhere** (the default used
    by `compute_descriptors`).
  - `field="rho"` with a non-pseudized reference **warns**: inside the PAW
    spheres CHGCAR is pseudized, and on 121 VASP CHGCARs 87% of ∫|Δρ| lies
    inside them (45% of the volume), although the charge inside is conserved
    to 0.6% — so |Δρ|-weighted whole-cell descriptors measure the POTCAR
    more than the bonding. (An earlier, much larger estimate came from an
    unrepresentative test file that was not raw VASP output.)
  - Warns when |∫Δρ| > 0.1 e (wrong ZVAL / reference counts).
- `deformation_outside_paw(engine, shells)`: the `*_def_out` variants — the
  same eight quantities from the CHGCAR and the valence free-atom
  reference, with Δρ zeroed inside every PAW sphere (r ≤ R_PAW of the
  voxel's nearest atom, from `engine.augmentation_radii`). Outside the
  spheres the CHGCAR is not pseudized, so this route is consistent without
  AECCARs and never warns. Adds `def_out_volume_fraction` and
  `def_out_radii_from_paw`. The shell fractions refer to the parts of the
  shells outside the spheres (an atom with R_PAW > c₂ contributes no bond
  shell). On real data: FeNi₃ `def_polarity_out` 0.25%, CaSi₃Pt 3.5% with
  0.57 e accumulated in the bonding shell — covalent bonding shows up, as it
  should. The Family A runner in the dispatcher returns both the whole-cell
  entries and these variants.

## 34. `descriptors/dataset.py`

Entry 82 is a dataset-level quantity: `correlation(x, y)` (Pearson,
Spearman, n over finite pairs) and `zeta_ellip_agreement(rows)` (ζ against the
mean and the median ellipticity across structures).

---

# Part VI — Calibration and tools

## 35. `calibration/__init__.py` — Family D and entry 106

### Literature

- `phillips_table(verified_only=False)`: parses `data/phillips_ionicity.csv`
  into formula → {f_i, structure, cohen_lambda, status, source}.
  `verified_only` keeps the 33 rows checked against a secondary source.
- `reduced_formula(species)`: pymatgen reduced formula (GaAs, NaCl, SiC), the
  key for matching computed structures to the table.

### Cohen (61)

- `cohen_lambda(species)`: Cohen's integer class from composition — 0 for
  elemental group-14 or 1:1 group-14 compounds, 1 for 1:1 III–V (groups
  13/15), 2 for 1:1 II–VI (groups 2 or 12 with 16), None otherwise.
- `cohen_bulk_modulus(d, lam)` = (1971 − 220λ)·d^−3.5 GPa. The published
  constant is 1971 and λ is the class, not a continuous ionicity (the
  specification wrote 1972 and λ = grid_ionicity).

### Ionicity (60)

- `IonicityCalibration` (dataclass): features, standardization mean/scale,
  coefficients [b₀, b₁, …], n, rmse, loocv_rmse, n_tetrahedral, compounds.
  `predict(descriptors)` = sigmoid(b₀ + z·b), NaN if any feature is missing;
  `save` / `load` as JSON.
- `_fit_sigmoid(Z, y)`: nonlinear least squares (`scipy.optimize.least_squares`)
  of sigmoid(b₀ + Zb) against y — the sigmoid keeps predictions on Phillips'
  [0, 1] scale.
- `fit_ionicity(descriptor_rows, features, targets, verified_only)`:
  descriptor_rows maps formula → descriptor dict; targets default to the
  table. Skips formulas without a target or with non-finite features;
  standardizes; fits; computes in-sample RMSE and leave-one-out RMSE (refits
  n times); counts tetrahedral compounds (scope of Phillips' scale). Needs at
  least len(features) + 2 compounds. Default features:
  `("fint_over_lnf", "VH_spread")` — two independent ingredients, as the
  specification recommends against calibrating on the SHAP feature alone.

### Bulk-modulus proxy (106)

- `bulk_proxy_x(descriptors)` = rho_mid_mean / bond_length_mean³.
- `BulkModulusCalibration(a, b, n, rmse_log)`: predict a·x^b; save / load.
- `fit_bulk_modulus(rows, B0)`: `np.polyfit` of ln B0 on ln x over finite,
  positive pairs (≥ 3).

### `calibration_family(structure, descriptors, ionicity, bulk)`

Called by the dispatcher after all other families: `grid_ionicity`,
`ionicity_residual` (minus the Pauling `ionicity`), `cohen_B0_predicted`
(λ = 0 outside Cohen's scope, flagged by `cohen_in_scope`), `B0_rho_proxy`.

## 36. `resample.py`

- `fft_friendly(n)`: smallest n' ≥ n with prime factors in {2, 3, 5, 7}.
- `shape_for_spacing(lattice, spacing, friendly)`: grid shape with at most
  `spacing` Å along each lattice vector.
- `fourier_resample(values, shape)`: `scipy.signal.resample` axis by axis —
  zero-padding or truncating the spectrum, exact for band-limited data,
  preserves the cell average (hence the integral).
- `resample_engine(engine, spacing | shape | scale)`: a new Engine with every
  primary field resampled; fields on other grids keep their ratio to the main
  grid; derived fields (`elf_d*`, `hartree_potential*`, `promolecule*`) are
  dropped and recomputed on demand; the non-collinear magnetization vector is
  resampled too. Structure, method, shells, spin mode, ZVAL and reference are
  carried over.

## 37. `strain.py` — entry 107

`strain_response(minus, plus, eps, unstrained, field)`: engines or CHGCAR
paths at ±ε. Checks that the runs have the same atoms (order and fractional
positions within 0.05). Compares charge per voxel n_k = ρ_k V/N in fractional
coordinates (resampling to the `plus` grid if shapes differ), so a uniform
dilation cancels exactly. Returns `drho_deps` = Σ|n⁺ − n⁻| / (2εQ) (fraction
of the charge that moves per unit strain), the relative L2 form, and the net
charge drift.

## 38. `convergence.py`

- `DEFAULT_FAMILIES`: the cheap families (no H, A, D).
- `convergence_report(engine, factors=(1, 0.8, 0.6), families, rtol=0.02)`:
  computes descriptors on the native grid and on Fourier-coarsened copies
  (`resample_engine(scale=f)`), then for each descriptor the relative change
  between native and the next-coarser grid, sorted descending, with a
  `converged` flag (≤ rtol). NaN on both grids counts as converged; a value
  that becomes NaN counts as infinitely changed.
- `format_report(report, limit)`: the table printed by `pydemi convergence`.

On 30 random VASP CHGCARs a median 20% of the reported quantities changed by
more than 2% at 80% resolution — the report is the evidence for or against a
grid.

## 39. `batch.py`

- `find_runs(root)`: every directory under root (inclusive) containing a
  CHGCAR, sorted.
- `material_id(path)`: directory name or file stem.
- `_load(path)`: `from_vasp_dir` for directories, `from_file` for files.
- `_calibrations(ionicity_cal, bulk_cal)`: loads JSON calibration files.
- `compute_one(path, families, ionicity_cal, bulk_cal)`: one row
  (`material_id`, `path`, `error`, descriptors). Any exception is caught and
  recorded in `error` (plus a short traceback) so the batch never stops.
  Runs inside worker processes, so it only takes picklable arguments.
- `columns(families)`: the fixed CSV header — id, path, error, then every
  default scalar registry name of the families, in registry order. Absent
  values become empty cells.
- `done_ids(out_csv)`: material IDs already written (for resume).
- `run_batch(inputs, out_csv, families, workers, resume, ionicity_cal,
  bulk_cal, progress)`: a single root directory without its own CHGCAR is
  expanded with `find_runs`; already-done IDs are skipped; rows are written as
  they finish and the file is flushed after each (so an interruption loses at
  most the rows in flight); `workers > 1` uses `ProcessPoolExecutor` with
  `as_completed`. Returns counts of done / skipped / failed.

## 40. `cli.py`

`main(argv)` builds an `argparse` CLI with subcommands:

| Subcommand | Function | Notes |
|---|---|---|
| `compute` | `_cmd_compute` | `run_batch`; exit code 1 if any input failed |
| `list` | `_cmd_list` | registry rows (entry, family, kind, name); `--all-kinds` includes field/site/dataset |
| `describe` | `_cmd_describe` | every `DescriptorInfo` field |
| `convergence` | `_cmd_convergence` | `convergence_report` + `format_report` |
| `resample` | `_cmd_resample` | `resample_engine(spacing)` then `write_volumetric` of `rho` |

`_load` mirrors the batch loader. The console script `pydemi` points to
`pydemi.cli:main` (pyproject).

## 41. `testing/analytic.py`

Analytic densities for the test suite and for anyone validating a change.

- `_RadialSuperposition(structure, params, electrons, tol)`: a spherical
  profile per atom, summed over atoms and periodic images. `params` and
  `electrons` can be scalars, per-atom arrays or element → value dicts.
  - `_images(center, cutoff)`: lattice shifts whose image of the centre can
    reach the cell (image-to-cell-centre distance ≤ cutoff + cell circumradius)
    — pruning this made the Slater tests 5× faster.
  - `evaluate(points, derivatives)`: ρ, ∇ρ, packed Hessian from the profile f
    and its radial derivatives: ∇ρ = f'r̂, H = f''r̂r̂ᵀ + (f'/r)(I − r̂r̂ᵀ).
  - `on_grid(grid, derivatives)`.
- `SlaterSuperposition`: f = ζ³/π e^{−2ζr} (hydrogenic 1s, with a cusp —
  tests moments and fractions).
- `GaussianSuperposition`: f = (α/π)^{3/2} e^{−αr²} (smooth — tests
  derivatives).
- Closed forms per unit charge: `slater_moment(n, ζ)` = (n+2)!/(2(2ζ)ⁿ),
  `slater_fraction_within(R, ζ)` = 1 − e^{−x}(1 + x + x²/2), x = 2ζR,
  `gaussian_moment(n, α)` = Γ((n+3)/2)/(Γ(3/2) α^{n/2}),
  `gaussian_fraction_within(R, α)` = erf(√αR) − 2√(α/π) R e^{−αR²}.

---

# Part VII — Data, tests, docs

## 42. Data files (`src/pydemi/data/`)

- `magpie_elements.csv`: Z, symbol and seven Magpie properties (AtomicWeight,
  Electronegativity, NValence, AtomicRadius, CovalentRadius, Row, IsFBlock)
  for Z = 1–112, `nan` where the source is missing. Generated from the
  matminer 0.10.1 wheel's `magpie_elementdata/*.table` files (all rows kept
  so the imputation mean matches matminer's exactly).
- `LICENSE-matminer`: matminer's BSD licence (required for redistribution).
- `phillips_ionicity.csv`: 67 compounds; columns formula, f_i, structure,
  cohen_lambda, status (`verified` / `recalled`), source. Comment lines at
  the top explain the status values.

All three ship in the wheel (checked with `pip wheel`).

## 43. The test suite, file by file

Run with `pytest` (166 tests, about a minute). `tests/conftest.py` sets
`PYDEMI_CACHE_DIR` to a temporary directory **before** pydemi is imported, so
tests never write to `~/.cache`, and provides three structures: a one-atom
cube, a two-atom triclinic cell (`TRICLINIC`) and a strongly sheared
three-atom cell.

| File | What it pins down |
|---|---|
| `test_io.py` | VASP reader/writer: round trip, Fortran order, collinear + augmentation + magmom lines, non-collinear, 3-block refusal, volume scaling, VASP 4 / negative scale / Selective dynamics / Cartesian, POTCAR labels, ELFCAR 10-per-line and two blocks, AECCAR sum, `from_vasp_dir` with every file type |
| `test_grid.py` | spectral exactness on a triclinic cell, FD second-order convergence, Laplacian = Hessian trace (both methods), periodic boundaries, eigenvalues, Field caching, input validation, integration |
| `test_geometry.py` | nearest atom vs brute force (3 cells), why fractional rounding fails, direction consistency, zero direction on a nucleus, partition site sums, shells cover every voxel, scaled shells, per-shape caching |
| `test_analytic.py` | closed forms vs quadrature, Slater moments and shell fractions converging with the grid, Gaussian moments, Laplacian sign structure (fd and spectral), Hessian eigenvalues |
| `test_descriptors.py` | Tier 1 vs Slater closed forms, ζ = 0 / > 0, lnf closed form (and the spectral round-off-floor effect), lap_concentration ≡ ½, scaling of the ratio variants, NaN on zero denominators, Tier 2 values / f-block flags / imputation / matminer agreement (skipped without matminer) / crystal systems, registry completeness and filters, the physical-descriptor count, `rho_ae` field |
| `test_rho_fields.py` | kinetic terms in atomic units, ELF_D and H vs independent formulas, uniform gas, low-density exclusion, cache invalidation, Family B on constant ELF, sweep and inflection, ELF vs ELFD naming, Hartree Poisson and analytic Gaussian, Fourier interpolation, site potentials and LOCPOT names, NCI, ellipticity ordering, information closed forms, LMC invariance, anisotropy isotropic / layered / degenerate, all families + dataset helpers |
| `test_sites_spin.py` | nearest partition reproduces Tier 1, power = nearest for equal radii, power vs brute force, power favours larger atoms, Becke partition of unity and two-atom closed form and size adjustment, Becke opt-in, variance decomposition exact and NaN for singletons, site values vs Slater, chemical differentiation, NaN sites ignored, ferromagnet, antiferromagnet, non-collinear vector sums, sentinels, every default scalar computed |
| `test_topology.py` | link-component table, Euler identity on random fields (with ties), census of a generic cosine field (1, 1, 3, 3), basins, non-nuclear maximum found, ripple vs robust counts, PAW shell lobes not counted, persistence vs voxel union-find, persistence removes ripple, interstitial floor, winding detection, layered percolation, bottleneck level, metal vs ionic percolation, bond census (fcc, simple cubic, unwrapped coordinates), midpoint density closed form, G/I2 registration |
| `test_reference.py` | hydrogenic eigenvalues, NIST He/Ne, default ZVAL, reference electron counts and disk cache, POTCAR / OUTCAR ZVAL parsing, form factor of a Gaussian, promolecule vs analytic, Δρ = 0 for the promolecule, Δρ sees bond charge, CHGCAR route opt-in and warning, mismatch warning, isolated-atom profile, Hirshfeld partition of unity / neutrality / charge transfer / r_cut convergence / valence weights / default in H, resampling for fidelity, fidelity perfect and degraded, `from_vasp_dir` ZVAL sources |
| `test_tooling.py` | Phillips table, Cohen λ and formula, ionicity fit (LOOCV, save/load, noise), minimum compounds, B0 power law, Family D in the dispatcher, strain dilation and redistribution, FFT-friendly shapes, Fourier resampling exactness, `resample_engine`, convergence report, cube / XSF round trips and units, batch run / errors / resume / parallel, CLI subcommands |

## 44. Packaging (`pyproject.toml`)

- Build backend: hatchling; wheel packages `src/pydemi` (data files are
  included automatically).
- Dependencies: numpy, scipy, spglib, pymatgen (pymatgen is required: free-atom
  configurations, Cohen's class, element symbols in cube/XSF).
- Extra `dev`: pytest.
- Console script `pydemi = "pydemi.cli:main"`.
- Author: Shubham Maurya, CMS Lab, IIT Kanpur. License field: MIT (no LICENSE
  file yet).

---

# Part VIII — Cross-cutting topics

## 45. Caching: what is computed once, and where it lives

| What | Where | Key | Invalidated by |
|---|---|---|---|
| gradient, Hessian, Laplacian, eigenvalues | `Field.__dict__` (cached_property) | per field | `Field.clear_cache()` |
| Grid | `Engine._grids` | shape | never (immutable) |
| NearestAtom geometry | `Engine._geometry` | shape | never |
| Shell masks | `Engine._shell_masks` | shape, c1, c2, radii | never |
| Partitions | `Engine._partitions` | shape, scheme, options | never |
| Kinetic terms (F1, F3, F5) | `Engine.cache` | `("kinetic", field)` | `add_field(field)` |
| ELF_D, Hartree potential, promolecule | registered as fields | field name | re-adding |
| Link-component table | `lru_cache` on `_link_components` | — | process |
| Magpie tables | `lru_cache` on `elements._tables` | — | process |
| Free-atom solutions | `AtomicLDAReference._memo` + disk `$PYDEMI_CACHE_DIR/atoms/*.npz` | element, correlation, grid, version | delete files / bump `CACHE_VERSION` |
| Default reference | `lru_cache` on `default_reference` | — | process |

Memory at 180³ (5.8M voxels): each full-grid float array is 47 MB; the
Hessian is 6 of those, the geometry direction 3, the gradient 3. Peak for the
full default set was ~2 GB per process.

## 46. Units: where every conversion happens

| Quantity | Conversion | Location |
|---|---|---|
| CHGCAR / AECCAR ρ·V → ρ | ÷ V | `read_volumetric` |
| cube / XSF e/bohr³ → e/Å³ | × 1/a₀³ | `io/grids._scale` |
| ρ, ∇ρ, ∇²ρ → atomic units | × a₀³, a₀⁴, a₀⁵ | `kinetic_terms` |
| volumes for F6 | × (1/a₀)³ | `information_family` |
| free-atom densities e/bohr³ → e/Å³, r bohr → Å | × 1/a₀³, × a₀ | `AtomicLDAReference.radial` |
| Hartree potential | 4π · 14.40 eV·Å · ρ(G)/G² | `hartree_potential` |
| RCORE bohr → Å | × a₀ | `Engine.from_vasp_dir` |
| covalent radii pm → Å | ÷ 100 | `elements.covalent_radii` |

Every atomic-unit quantity is tested against an independent atomic-unit
evaluation (a Gaussian specified in bohr and built in Å), so a wrong power of
a₀ fails a test.

## 47. NaN, zero and sentinel policy

- **Undefined ratio → NaN** (`safe_div`); no epsilons.
- **Input not available → name absent** from `compute_descriptors` output
  (ELF_*, V_*), empty CSV cell in batch output.
- **Family A without AECCAR → NaN** for the whole-cell entries (deliberate:
  inside the PAW spheres the CHGCAR route is dominated by pseudization); the
  `*_def_out` variants are computed from the CHGCAR.
- **Family D without calibration → NaN** (Cohen is always computed).
- **Spin on a non-spin run → 0** with `is_spin_polarized = 0`; ratio entries
  of a non-magnetic spin run → 0 with `is_magnetic = 0` (the specification
  asks for a sentinel, not NaN).
- **Within-element variance with no repeated element → NaN.**
- **Empty regions** (no interstitial voxels, no NCI voxels, no bond-shell
  voxels) → NaN for averages over them; counts and fractions → 0.

## 48. Recipes

### Adding a descriptor to an existing family

1. Compute it in the family function and add it to the returned dict.
2. Add a `DescriptorInfo` in `registry.py` at the right position (output order
   follows the registry): entry number, family, kind, inputs, formula, note.
3. Add a test with a closed form or an exact identity.
4. Regenerate the README table for the family (the README tables are the
   registry rendered as markdown).

### Adding a new family

1. New module in `descriptors/` with `my_family(engine, field, shells) ->
   dict`.
2. Import it in `descriptors/__init__.py`, add the key to `FAMILIES` and a
   runner to `_RUNNERS`.
3. Registry entries with the new family key.
4. Tests; if it needs inputs from other families, compute them internally or
   follow the `_D_INPUTS` pattern.

### Adding a new field (e.g. a new grid quantity)

Compute the array and `engine.add_field(name, values)`; every existing
machinery (moments, shells, partitions, topology) then works on it by passing
`field=name`. If it is derived from another field, cache under a tuple key
containing the source field's name so `add_field` invalidates it.

### Adding a reader

Return a `ChargeDensity` (or `VolumetricData`) with densities in e/Å³ and a
`Structure`; hook it into `Engine.from_file` by extension.

### Changing the free-atom solver

Bump `CACHE_VERSION` in `atoms/reference.py` so cached `.npz` files are not
reused, and rerun `tests/test_reference.py` (NIST checks).

## 49. Design decisions and the measurements behind them

| Decision | Evidence |
|---|---|
| Periodic compact FD Laplacian as default | over 200 VASP CHGCARs, np.gradient-twice lnf differs from spectral by 2% (median), up to 21%; compact FD by 0.2% (median) |
| Keep specified names, add fixes under new names | a column must never change meaning between result files |
| KD-tree minimum image | fractional rounding overestimates by > 0.1 Å in sheared cells |
| Charge-weighted `lnf_rho` alongside `lnf` | spectral voxel-count lnf is noise in near-empty voxels (0.07–0.15 vs 0.008 exact) |
| `lap_concentration_valence` | specified entry 14 is identically ½ |
| Median ellipticity | mean ≈ 11× median for the typical structure of the dataset |
| Becke opt-in, Hirshfeld default | Becke 2e-2 error at 60 competitors; Hirshfeld 9e-6 at 6.5 Å |
| Hirshfeld blocks + lookup table | 4 min → 34 s at 180³ |
| Own LDA atom solver | no atomic-DFT package installed; matches NIST to 1e-4 Ha |
| Family A whole-cell entries AECCAR-only | CHGCAR vs free-atom valence: 87% of ∫\|Δρ\| inside the PAW spheres (121 structures) |
| `*_def_out` variants from CHGCAR | outside the spheres the two agree (FeNi₃ within ~3% beyond 0.8 Å) |
| Charge requirement on persistent maxima | Yb compounds: up to 48 persistent maxima holding < 1 e |
| Hirshfeld weights from the field's own electrons | total-density weights gave nonzero charges for a pure promolecule |
| Consistent PL census (Freudenthal) | Euler identity then exact; 26-neighbour census is not a triangulation link |
| Percolation supremum | literal min{c : …} is always the lowest density |
| PAW-aware non-nuclear cutoff | CaSi₃Pt: 8 spurious lobes at 0.81–0.83 Å holding 6.3 e |
| Persistence filter | Mg₃(TiAl₉)₂: 576 "significant" ripple maxima; YMg₃ 124 → 0 |
| Interstitial floor `rho_min_int` | 61% of 6,059 CHGCARs have negative pseudo-density near a nucleus |
| Vendored Magpie tables | installing matminer would downgrade pandas in the shared env |
| ZVAL / RCORE from OUTCAR | the dataset has no POTCARs; default rule gives Ac = 3 vs POTCAR 11 |
| Atomic write of cache files | parallel workers solving the same element |

## 50. Known rough edges

- A structure whose element appears in two separate species groups gets its
  POTCAR/OUTCAR ZVAL list ignored (list length ≠ number of distinct
  elements) and falls back to the default rule, with a warning.
- The KD-tree queries use `workers=-1`, so each batch worker may use many
  threads; set `OMP_NUM_THREADS=1` and a moderate `--workers` on shared
  machines.
- `pymatgen` emits a warning for noble gases without a Pauling
  electronegativity (He, Ne, Ar) when compositions are parsed; harmless.
- Persistence in `topology_family` resolves absorber chains with a Python
  loop over maxima (fine for thousands of maxima, quadratic in the worst
  case).
- Native Quantum ESPRESSO HDF5 and ABINIT `_DEN` files are not read (no
  sample files to validate a reader against).
