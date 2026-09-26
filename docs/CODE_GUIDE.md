# pydemi — code guide

A file-by-file, function-by-function walk through the pydemi source, written
as a personal reference. It explains what every piece of code does, the
algorithm and formulas behind it, why it was written that way (including the
measurements and test failures that forced a decision), how the pieces
connect, and which tests pin the behaviour down.

This guide describes the current package in `src/pydemi/`, rebuilt to the
specification in `prompt.md`. It replaces `legacy/docs/CODE_GUIDE_pdf_spec.md`,
which describes the earlier (PDF-specification) version kept under `legacy/`.
The user-facing documentation is `README.md`, and the development status is
in `PROGRESS.md`. This guide is about the code.

Author: Shubham Maurya, CMS Lab, IIT Kanpur.

---

## Contents

- Part I — Orientation
  - 1. Repository layout
  - 2. Architecture and data flow
  - 3. Conventions used everywhere
- Part II — Data model and I/O
  - 4. `io/base.py`
  - 5. `io/vasp.py`
  - 6. `io/cube.py`
  - 7. `io/xsf.py`
  - 8. `io/registry.py` and `io/__init__.py`
- Part III — Core numerics
  - 9. `constants.py`
  - 10. `data/__init__.py` and the data files
  - 11. `core/grid.py`
  - 12. `core/derivatives.py`
  - 13. `core/geometry.py`
  - 14. `core/partition.py`
- Part IV — Fields
  - 15. `fields/density.py`
  - 16. `fields/elf.py`
  - 17. `fields/potential.py`
  - 18. `fields/deformation.py`
- Part V — Operators
  - 19. `operators/moments.py`
  - 20. `operators/fractions.py`
  - 21. `operators/anisotropy.py`
  - 22. `operators/laplacian.py`
  - 23. `operators/topology.py`
  - 24. `operators/sitestats.py`
- Part VI — Descriptors
  - 25. `descriptors/registry.py`
  - 26. `descriptors/__init__.py` (`featurize`, metadata, `catalogue`)
  - 27. `descriptors/bonding.py`
  - 28. `descriptors/structural.py`
  - 29. `descriptors/magnetic.py`
  - 30. `descriptors/heterogeneity.py`
  - 31. `descriptors/compositional.py`
  - 32. `pydemi/__init__.py` (the public API)
- Part VII — Validation and tooling
  - 33. `validate/__init__.py` (`elf_fidelity`)
  - 34. `validate/analytic.py`
  - 35. `validate/invariance.py`
  - 36. `validate/convergence.py`
  - 37. `batch.py`
  - 38. `cli.py`
  - 39. `tools/`
  - 40. Packaging (`pyproject.toml`)
- Part VIII — Tests
  - 41. The test suite, file by file
- Part IX — Cross-cutting topics
  - 42. Caching: what is computed once, and under which key
  - 43. Units: where every conversion happens
  - 44. Sentinel, flag and NaN policy
  - 45. Where the code departs from the specification
  - 46. Known rough edges
- 47. How to extend

---

# Part I — Orientation

## 1. Repository layout

```
pydemi/
├── pyproject.toml                  packaging (hatchling), dependencies, `pydemi` console script,
│                                   pytest and mypy settings
├── README.md                       user documentation; descriptor tables generated from the registry
├── PROGRESS.md                     development status, dataset rerun, open decisions
├── prompt.md                       the governing build specification
├── pydemi — Consolidated Descriptor Reference.pdf   the earlier specification (superseded)
├── docs/
│   ├── catalogue.csv               the descriptor registry as a table (tools/generate_docs.py)
│   └── CODE_GUIDE.md               this file
├── src/pydemi/
│   ├── __init__.py                 public API
│   ├── constants.py                every unit factor, cutoff and threshold
│   ├── batch.py                    featurize_batch: one process per structure, errors recorded
│   ├── cli.py                      the `pydemi` command line
│   ├── io/                         base.py (data model), vasp.py, cube.py, xsf.py, predicted.py, registry.py
│   ├── core/                       grid.py, derivatives.py, geometry.py, partition.py
│   ├── fields/                     density.py, elf.py, potential.py, deformation.py
│   ├── operators/                  moments.py, fractions.py, anisotropy.py, laplacian.py,
│   │                               topology.py, sitestats.py
│   ├── descriptors/                registry.py, __init__.py (featurize), bonding.py, structural.py,
│   │                               magnetic.py, heterogeneity.py, compositional.py
│   ├── validate/                   __init__.py (elf_fidelity), analytic.py, invariance.py,
│   │                               convergence.py
│   └── data/                       __init__.py, elements.csv, free_atoms.npz, magpie_labels.txt,
│                                   LICENSE-matminer
├── tests/                          13 files, 873 tests after parametrization (pytest)
├── tools/                          atomic_solver.py, generate_element_data.py,
│                                   generate_free_atom_tables.py, generate_docs.py,
│                                   paw_table_from_outcars.py, magpie_elements_source.csv
├── results/                        dataset runs (not part of the package); prompt_spec/ holds the
│                                   6,059-structure rerun and dataset_paw_table.json
├── paper/                          main.tex, references.bib, analysis/
└── legacy/                         the earlier version: pydemi_pdf_spec/, tests_pdf_spec/, docs/
```

Size: about 6,000 lines of library code (docstrings included), 1,600 lines
of tests and 460 lines of tools. The layout of `src/pydemi/` is the one
prescribed in `prompt.md` §1, module for module.

## 2. Architecture and data flow

The specification's design premise (§0) is **one geometry pass, many
operators, several fields**. The nearest-atom geometry (distance to the
nearest nucleus, unit vector from it, its index) is computed once per
structure; a small set of operators (radial moments, shell fractions,
gradient anisotropy, Laplacian statistics, topological counts, site
statistics) is written once and applied to several scalar fields (rho, |m|,
the ELF, delta rho, the potential). The second premise is that there is no
critical-point search anywhere: every descriptor is a direct grid operation
that always terminates.

The flow for one structure:

```
files on disk
   │   io/vasp.py, io/cube.py, io/xsf.py      (io/registry.read sniffs the format)
   │   CHGCAR / AECCAR divided by V; Fortran order; bohr -> Angstrom; e/bohr^3 -> e/Angstrom^3
   ▼
VolumetricData  (io/base.py)
   structure, rho, magnetization | magnetization_vector, elf, potential, core_density,
   density_source, zval, paw_radii, sources, options, cache
   │
   │   featurize(vd, domains=..., partition=..., ...)       descriptors/__init__.py
   │     opts = make_options(...)            -> FeatureOptions (frozen, validated)
   │     v    = vd.with_options(opts)        same grids, the SAME cache dict
   ▼
for spec in selected(opts):                  registry order, filtered by domain and extension
    result = spec.func(v)                    every descriptor is f(vd) -> float | Sentinel
      │
      ├── geometry(v)             core/geometry.geometry_of      cache[("geometry",)]
      ├── masks(v)                core/geometry.shells_of        cache[("shells", key)]
      ├── field_derivatives(v, n) fields/density.derivatives     cache[("derivatives", n, ...)]
      ├── kinetic, elf_d, potential, promolecule, census, ...    cache[(...)]
      ├── partition_of(v, scheme) core/partition                 cache[("partition", scheme)]
      └── operators/*             pure functions: arrays in, float out
    Sentinel(value, case) -> the documented value, plus the flag  <name>__flag = 1
   │
   ▼
features: dict[str, float]      (+ metadata: settings, METADATA_HOOKS output, __flag columns)
   │   batch.featurize_batch: one process per structure (ProcessPoolExecutor)
   ▼
pandas DataFrame: path | descriptors in registry order | metadata columns
```

Key properties of the design:

- **Descriptors are independent functions of one `VolumetricData`.** Each
  registered function computes one number. They share work only through
  `vd.cache`: the first descriptor that needs the gradient of rho computes
  it, every later one reuses it. There is no family function returning a
  dict and no dispatcher logic beyond filtering and ordering.
- **Cache keys carry the options they depend on.** `featurize` attaches the
  options by creating a view (`with_options`) that shares the grids and the
  cache dict with the caller's object. A second `featurize` call on the same
  object with other options reuses what does not depend on them (the
  geometry pass, the percolation levels, ...) and recomputes what does
  (section 42 lists every key).
- **Operators are field-agnostic.** `radial_moment`, `shell_fraction`,
  `gradient_anisotropy`, `concentration`, `site_sums` take arrays; the
  descriptor layer chooses the field.
- **The registry is the single source of truth.** Names, output order, units,
  ranges, sentinel cases, stability tags and formulas live in one
  `DescriptorSpec` per descriptor. `featurize`, `descriptor_names`,
  `catalogue()`, the README tables, `docs/catalogue.csv`, the batch CSV
  columns and the parametrization of the invariance test all come from it.
- **Metadata hooks always run.** Functions registered with
  `@metadata_hook` add metadata whatever domains were requested, which is
  how `euler_consistency`, `magnetic` and `zval_source` appear in every row.
- **Everything terminates.** Pointer jumping for basins, bisection with a
  step cap for percolation, adaptive image ranges that stop once a proven
  bound is met.

## 3. Conventions used everywhere

- **Lattice rows are lattice vectors.** `Lattice.matrix` = A holds a1, a2,
  a3 as rows, in Angstrom. A Cartesian position is `x = u @ A` for a
  fractional row vector `u`; the inverse is `u = x @ inv(A)`.
- **Reciprocal vectors without 2 pi.** `B = inv(A)^T` (`Lattice.reciprocal`)
  has rows b1, b2, b3 with a_i . b_j = delta_ij. The contravariant metric is
  `G = B B^T` (`Lattice.metric`, 1/Angstrom^2); the covariant one is
  `A A^T` (`Lattice.covariant_metric`). The distance between opposite faces
  a is `1/|b_a|` (`Lattice.heights`). Every derivative formula (section 12)
  and every image-range bound (section 13) is written with these.
- **Grid indexing.** `Grid.data[i, j, k]` is the value at fractional
  coordinate (i/n1, j/n2, k/n3). Grids are always periodic; every index
  operation wraps (`np.roll`, modular stencils, FFTs). Arrays are C-ordered
  in memory. VASP files store the first index fastest (Fortran order), so
  the VASP reader reshapes with `order="F"`; cube files store the last index
  fastest (C order); XSF stores Fortran order with a duplicated periodic
  plane.
- **Units.** Angstrom, electrons / Angstrom^3 and eV internally. Quantities
  the specification defines in atomic units (ELF_D, g, v, H, the reduced
  gradient s, the NCI density threshold, the information measures) are
  converted at the point of use with the factors in `constants.py`
  (section 43). An integral is `sum(f) * dV` with `dV = V_cell / N`.
- **Directions.** `u_k` is the unit vector from the nucleus (the image that
  was used) to voxel k, in Cartesian components. It is 0 on a voxel that
  coincides with a nucleus (`r <= 1e-12`).
- **Distances within `GEOMETRY_EPS` (1e-8 Angstrom) are equal.** Shell
  boundaries, the non-nuclear cutoff and the PAW radii are all tested as
  `r <= c + GEOMETRY_EPS`, and equidistant atom images are ordered by a
  geometric rule (section 13). Both exist to make supercell and translation
  invariance exact for atoms on grid points.
- **Return values.** A descriptor function returns a `float` or a
  `Sentinel(value, case)`. `featurize` reports the value and sets
  `<name>__flag` to 1 in the metadata. Operators return NaN for 0/0; the
  descriptor layer turns every such NaN into a documented sentinel. The
  only NaNs that reach the output are the within-element variances of a
  structure with one site per element and a Magpie feature matminer cannot
  compute; both are flagged (section 44).
- **Intensivity.** Every registered descriptor must be unchanged under a
  2x2x2 supercell (spec §10). `register` refuses `intensive=False`. Counts
  are per volume, totals per atom or as a fraction of Q_tot, the
  information measures are volume-normalized; the extensive raw values
  (`M_abs`, `M_net`) appear only in the metadata, and so does the raw-count
  quality flag `euler_consistency`.
- **Determinism.** No random numbers anywhere. Ties are broken by fixed
  rules: equidistant atom images by their fractional displacement, equal
  grid values by flat voxel index.

---

# Part II — Data model and I/O

## 4. `io/base.py`

The data model of spec §2. Three type aliases are defined at the top:
`FloatArray` (any floating numpy array), `DensitySource =
Literal["pseudo", "all_electron"]` and `SpinMode = Literal["none",
"collinear", "noncollinear"]`.

### `class Lattice` (frozen dataclass, `eq=False`)

One field, `matrix`. `__post_init__` copies it to a 3x3 float64 array,
raises `ValueError("lattice matrix is singular")` when |det A| < 1e-12, marks
it read-only and stores it with `object.__setattr__` (the dataclass is
frozen). Frozen because every `Grid` of a structure shares the same lattice;
`eq=False` because array equality is not a boolean.

Derived quantities are `functools.cached_property`, computed on first use:

| Property | Value |
|---|---|
| `volume` | abs(det A), Angstrom^3 |
| `inverse` | A^{-1}; fractional coordinates are `x @ inverse` |
| `reciprocal` | B = inv(A)^T, rows b1, b2, b3 without 2 pi |
| `metric` | G = B B^T (1/Angstrom^2), the contravariant metric of the Laplacian |
| `covariant_metric` | A A^T (Angstrom^2) |
| `lengths` | abs(a1), abs(a2), abs(a3) |
| `heights` | 1/abs(b_a): distance between opposite faces |

`allclose(other, atol=1e-6)` compares matrices; it is how every reader
checks that companion files (ELFCAR, LOCPOT, AECCAR) describe the same cell.

Tests: `test_io.py::test_lattice_conventions` (A B^T = I to 1e-14, metric =
B B^T, singular matrix rejected).

### `class Grid` (dataclass, `eq=False`)

`data` and `lattice`. `__post_init__` requires a 3-D array with at least one
point per axis and casts non-floating data to float64; float32 data stay
float32 (the float32 mode of spec §13). Properties `shape`, `n_voxels` and
`dV` (voxel volume V/N). `astype(dtype)` returns a new `Grid` without
copying when the dtype already matches.

### `class Structure` (dataclass, `eq=False`)

`lattice` (a `Lattice`, or anything convertible), `species` (one element
symbol per atom) and `frac_coords` (n_atoms x 3). `__post_init__` coerces
the three and checks that the counts agree.

- `n_atoms`, `volume`, `cart_coords` (`frac_coords @ A`).
- `elements`: the distinct elements **in order of first appearance**. This
  order matters: `read_vasp` pairs the ZVAL / RCORE values of a POTCAR,
  which come one per dataset in file order, with `elements`.
- `element_index`: index of each atom's element in `elements`.
- `site_counts()`: `{element: number of sites}`; it goes into the metadata
  and into the Magpie composition.
- `from_pymatgen(structure)` / `to_pymatgen()`: structure I/O is the only
  role pymatgen has (spec §2); `to_pymatgen` imports it lazily.

### `as_structure(obj)`

Returns a pydemi `Structure` unchanged; converts anything with `lattice`,
`frac_coords` and `sites` attributes (a pymatgen `Structure`) by duck typing,
so pymatgen never has to be imported to recognize one. Anything else raises
`TypeError`. Test: `test_io.py::test_pymatgen_structure_accepted`.

### `class VolumetricData` (dataclass, `eq=False`)

One structure with its grid fields.

| Field | Meaning |
|---|---|
| `structure` | `Structure` (a pymatgen one is converted) |
| `rho` | `Grid`, electrons / Angstrom^3; always present |
| `magnetization` | collinear m = rho_up - rho_down, or None |
| `elf` | ELF from an ELFCAR, in [0, 1], or None |
| `potential` | LOCPOT in eV, or None |
| `density_source` | `"pseudo"` (CHGCAR) or `"all_electron"` (AECCAR0 + AECCAR2) |
| `magnetization_vector` | (m_x, m_y, m_z) for a non-collinear run, else None |
| `core_density` | AECCAR0 for an all-electron density, else None |
| `zval` | element -> valence electrons of the PAW dataset, or None |
| `paw_radii` | element -> PAW augmentation radius RCORE in Angstrom, or None |
| `sources` | provenance: file per field, and `"table"` for tabulated zval / paw_radii |
| `options` | the `FeatureOptions` attached by `featurize` |
| `cache` | every derived array (section 42); `repr=False` |

`__post_init__` enforces the invariants the rest of the code relies on:
`density_source` is one of the two values; every grid's lattice matches the
structure's (`allclose`); **every grid has the shape of rho** (the error
message points to `pydemi.core.grid.resample`); and `magnetization` and
`magnetization_vector` are not both set. The shape rule is what lets every
operator index every field with the same geometry arrays; resampling
happens once, at read time (`read_vasp`), never inside a descriptor.

- `_grids()`: (name, grid) pairs of every field present, for validation.
- `spin_mode`: `"noncollinear"` when `magnetization_vector` is set,
  `"collinear"` when `magnetization` is set, else `"none"`.
- `lattice`, `shape`: shortcuts to the structure's lattice and rho's shape.
- `with_options(options)`: `dataclasses.replace(self, options=options)`.
  `replace` passes every field, including the existing `cache` dict, to the
  constructor, so the new object shares the grids **and the cache** with the
  original. This is how `featurize` attaches its options without copying
  data, and why work computed in one `featurize` call is reused by the next.
  `replace` reruns `__post_init__`, which is cheap.
- `astype(dtype)`: a copy with every grid cast and a **fresh** cache, so a
  float32 run never reuses float64 derivatives.

Tests: `test_io.py::test_volumetric_data_validates_fields`,
`test_geometry.py::test_geometry_is_computed_once_per_structure` (the view
from `with_options` returns the identical geometry object).

### Helpers

- `species_labels(tokens, counts)`: expands the per-group labels of a POSCAR
  species line to one symbol per atom, stripping POTCAR suffixes (`Fe_pv` ->
  `Fe`, `O/abc` -> `O`). Test: `test_io.py::test_potcar_style_labels`.
- `expand_species(labels, counts)`: accepts either one label per group or
  one per atom (a user-supplied `species=` list may be either).
- `check_mapping(m, elements, what)`: raises `KeyError` naming the elements
  a mapping lacks. It is defined but not called anywhere in the package.

## 5. `io/vasp.py`

Readers for CHGCAR / CHG, AECCAR0 / AECCAR2, ELFCAR and LOCPOT, ZVAL and
RCORE from a POTCAR or OUTCAR, and a writer. These are the details spec §3
and §16 single out as silent failures: the volume division, the Fortran
order, the spin block after the augmentation lines, and never taking the
m_x block of a non-collinear file as a collinear m.

### Parsing

- `_is_int(token)`: whether a token parses as an integer.
- `_parse_header(lines, species)`: POSCAR header -> (`Structure`, index of
  the first line after the coordinates).
  - Line 2 is the scale. Three numbers scale the lattice rows one by one
    (Cartesian coordinates are then not rescaled). One number scales
    everything; a negative one is a target volume, turned into the factor
    `(-s / abs(det A))^(1/3)`.
  - Line 6 is the species line (VASP 5) when it is not all integers; then
    come the counts.
  - Species come from the `species=` argument if given (through
    `expand_species`), else from the species line (through
    `species_labels`); a VASP 4 file without either raises a `ValueError`
    that says to pass `species=` or keep the POTCAR next to the file.
  - A line starting with `s` / `S` (selective dynamics) is skipped. A
    coordinate-mode line starting with `c` / `k` means Cartesian; those
    coordinates are scaled and converted to fractional with `inv(A)`.
- `_find_dims(lines, start, dims=None)`: the index of the next line holding
  exactly three integers (-1 if none). With `dims` given, only a line equal
  to those dimensions counts. That is how the reader skips the PAW
  augmentation-occupancy lines and magnetic-moment lines VASP writes between
  and after the blocks: it looks for the next line equal to the first
  block's dimensions.
- `_read_block(lines, start, n_values)`: reads `n_values` floats. Fast path:
  the number of values per line is taken from the first line, the needed
  lines are joined and parsed with one `np.fromstring`; if the count comes
  out wrong (irregular line lengths), it falls back to reading line by line
  and raises `ValueError("data block truncated ...")` when the file ends
  early. Returns the values and the index of the next line.
- `read_blocks(path, scaled_by_volume, species=None, max_blocks=None)`:
  every data block of a file. Each block is reshaped with
  `reshape(dims, order="F")` and divided by the cell volume when
  `scaled_by_volume` is True (CHGCAR and AECCAR store rho * V_cell).
  Returns `(structure, [block, ...])`. `max_blocks` stops early (the
  non-spin read of a CHGCAR, AECCAR, LOCPOT).

Tests: `test_io.py::test_volume_division` (raw block = 0.25 V, read block =
0.25), `test_fortran_order` (the first line of a written file is
f(0,0,0), f(1,0,0), f(2,0,0), f(0,1,0), f(1,1,0) for f = i + 10 j + 100 k),
`test_vasp4_negative_scale_cartesian_selective` (scale -27 -> 3 Angstrom
cube, Cartesian coordinates, selective dynamics, species required).

### File readers

- `read_chgcar(path, species=None, read_spin=True)`: one block ->
  non-spin; two blocks -> collinear, the second is `magnetization`; four
  blocks -> non-collinear, blocks 2-4 go to `magnetization_vector` and
  `magnetization` stays None; any other count raises `ValueError(...
  "Refusing to guess.")`. `read_spin=False` reads the first block only.
  Tests: `test_collinear_spin_block_after_augmentation` (VASP-style
  augmentation and magnetic-moment lines inserted between and after the
  blocks), `test_noncollinear_four_blocks_as_vector`,
  `test_three_blocks_refused`, `test_roundtrip_single_block`.
- `read_aeccar(path, species=None)`: one AECCAR (volume-scaled) ->
  `(Structure, Grid)`.
- `read_all_electron(aeccar0_path, aeccar2_path, species=None)`: rho =
  AECCAR0 (frozen core) + AECCAR2 (valence), `density_source =
  "all_electron"`, `core_density` = AECCAR0. The two grids and lattices
  must agree. Test: `test_all_electron_sum_and_source`.
- `read_elfcar(path, species=None)`: not volume-scaled; a spin-polarized
  ELFCAR holds ELF_up and ELF_down and the first (spin-up) block is
  returned. Test: `test_elfcar_not_volume_scaled`.
- `read_locpot(path, species=None)`: eV, not volume-scaled, first block.
  Whether it holds the total local potential or the electrostatic one
  depends on LVTOT / LVHAR (the docstring says so; pydemi cannot tell).

### POTCAR / OUTCAR

- `read_potcar_zval(path)`: ZVAL of every PAW dataset in file order. It
  matches the `POMASS = ...; ZVAL = ...` header line each dataset has, so
  the OUTCAR's later summary line `ZVAL = ...` is not counted twice; only
  when no header line exists does it fall back to every `ZVAL =`.
- `read_potcar_rcore(path)`: RCORE (the outermost PAW cutoff radius, bohr)
  of every dataset.
- `read_potcar_elements(path)`: the element of every dataset from the
  `TITEL = PAW_PBE Fe_pv ...` lines, suffix stripped; used to name the atoms
  of a VASP 4 CHGCAR.
- `_find_paw_source(chgcar)`: POTCAR, else OUTCAR, in the CHGCAR's
  directory, else None.

### `read_vasp(chgcar, elf=None, locpot=None, aeccar0=None, aeccar2=None, potcar="auto", species=None, zval=None, paw_radii=None)`

Assembles one VASP run into a `VolumetricData`, in this order:

1. **PAW source.** `potcar="auto"` looks for a POTCAR, then an OUTCAR,
   next to the CHGCAR; a path uses that file; None skips it.
2. **VASP 4 files.** Without `species=` and with a PAW source, if line 6 of
   the CHGCAR is all integers (no species line) the species come from the
   POTCAR / OUTCAR titles.
3. **Density.** `read_chgcar`. With both `aeccar0` and `aeccar2`, rho is
   replaced by the all-electron sum (lattice and grid must match the
   CHGCAR's; the CHGCAR still supplies the magnetization). Exactly one of
   the two raises `ValueError`.
4. **ELFCAR.** The cell must match (`_same_cell`). VASP writes the ELFCAR
   on the coarse NGX grid and the CHGCAR on the fine NGXF grid, so a
   different shape is resampled with `linear_resample` and clipped to
   [0, 1]. Trilinear, not Fourier, because interpolation between
   neighbouring values stays inside the data range, while a band-limited
   resampling overshoots outside [0, 1] near sharp ELF features.
5. **LOCPOT.** A different shape is resampled with `fourier_resample`
   (the potential is smooth and band-limited).
6. **ZVAL and RCORE from the run's own file.** The number of datasets must
   equal the number of distinct elements; otherwise a warning is issued and
   ZVAL is ignored. RCORE is converted from bohr to Angstrom.
7. **Tables.** `zval=` / `paw_radii=` are per-element tables for runs
   without a POTCAR or OUTCAR. They are used only when the run's own file
   gave nothing, only the structure's elements are kept, and
   `sources[key] = "table"` records the origin.
8. `vd.__post_init__()` revalidates the assembled object (the fields were
   set after construction).

Why the tables exist: the 6,059-structure rerun (PROGRESS.md §3) found 1,158
runs without an OUTCAR. The earlier fallback rule for ZVAL was wrong for 43
of 87 elements (for example Sb 15 instead of 5), giving reference-charge
errors up to 80 e. The fix has two parts: the corrected rule
`pydemi.data.default_zval` (section 10), and a per-element table read from
the dataset's 4,901 OUTCARs (`tools/paw_table_from_outcars.py`, section 39)
passed as `zval=` / `paw_radii=` (or `--paw-table` on the command line).

Tests: `test_io.py::test_read_vasp_assembles_a_run` (all-electron sum,
collinear magnetization, a 3x3x3 ELFCAR resampled to 6x6x6 and kept in
[0, 1], POTCAR ZVAL {Fe: 14, O: 6}, RCORE 2.2 bohr in Angstrom),
`test_paw_tables_for_runs_without_potcar` (tables fill in, extra elements
dropped, partial tables give `table+default`, the POTCAR wins over a
table).

### `_same_cell(s, lattice, path)`

Raises `ValueError` when a companion file's lattice differs from the
CHGCAR's.

### `write_volumetric(path, structure, blocks, scaled_by_volume=True, per_line=5, comment="written by pydemi")`

The inverse of `read_blocks`: a VASP 5 header (scale 1.0, species and
counts, Direct coordinates) and every block preceded by a blank line and its
dimensions, multiplied by V when `scaled_by_volume`, written in Fortran
order with `.11E`. Atoms are written grouped by element in first-appearance
order, as the format requires, so a structure with interleaved species comes
back reordered. No augmentation section is written. The tests use it to
build every synthetic VASP file.

## 6. `io/cube.py`

Gaussian cube files, e.g. from Quantum ESPRESSO `pp.x` (`output_format =
6`) or ABINIT `cut3d`.

### `read_cube(path, quantity="density")`

- Line 3: number of atoms and the origin. Lines 4-6: the voxel count and
  voxel vector of each axis; a positive count means bohr, a negative one
  Angstrom. The lattice is voxel vector x count, converted to Angstrom.
  The bohr / Angstrom flag used for the origin and the atom positions is the
  one of the last axis line (the three are assumed to agree).
- Atom lines: atomic number (-> symbol with `pydemi.data.symbol_of`),
  charge, position. A negative atom count marks an orbital cube with one
  extra line of orbital indices, which is skipped.
- Data: the last index fastest, i.e. C order, so a plain `reshape(counts)`.
  `quantity="density"` converts e/bohr^3 to e/Angstrom^3
  (`DENSITY_FROM_AU`); `"raw"` leaves the values alone (potentials, ELF).
- Fractional coordinates are measured from the grid origin and wrapped to
  [0, 1), so voxel (0, 0, 0) sits at fractional 0 as pydemi's convention
  requires.

### `write_cube(path, structure, data, quantity="density", comment="written by pydemi")`

Origin 0, bohr, e/bohr^3 for densities, six values per line. Inverse of
`read_cube`.

Tests: `test_io.py::test_cube_units_against_known_file` (a hand-written
10 bohr cube of 0.1 e/bohr^3: lattice 10 a0, density 0.1 / a0^3, integral
100 e, a Li atom at fractional (0.25, 0.5, 0.75)),
`test_cube_and_xsf_roundtrip`.

## 7. `io/xsf.py`

XCrySDen XSF periodic data grids (`pp.x` `output_format = 5`, `cut3d`).

### `read_xsf(path, quantity="density")`

Blank and `#` lines are dropped. The lattice comes from `PRIMVEC`, the atoms
from `PRIMCOORD` (atomic number or symbol, Cartesian Angstrom). The grid
block `BEGIN_DATAGRID_3D...END_DATAGRID_3D` gives the counts, the origin and
three spanning vectors (skipped; the lattice is PRIMVEC), then the values in
Fortran order. XSF general grids repeat the first point at the end of every
axis, so the last plane of each axis is dropped (`[:-1, :-1, :-1]`); this is
done unconditionally. Density values are taken to be e/bohr^3 (what pp.x
and cut3d write) and converted; `quantity="raw"` skips the conversion.

### `write_xsf(path, structure, data, quantity="density", name="pydemi")`

Pads the grid with its own first planes (`np.pad(..., mode="wrap")`) to add
the periodic duplicate, and writes Fortran order.

## 8. `io/registry.py` and `io/__init__.py`

### `sniff(path)`

Returns one of `"chgcar"`, `"aeccar"`, `"elfcar"`, `"locpot"`, `"cube"`,
`"xsf"`, `"predicted"`. In order: the suffix (`.cube` / `.cub`, `.xsf`, `.npz`); the file name
stem before the first `.` and `_`, upper-cased, looked up in `_NAMES`
(CHGCAR, CHG, AECCAR0/1/2, ELFCAR, LOCPOT); a name prefix; then the content
of the first 8 lines (`PRIMVEC`, or `CRYSTAL` on the first line -> XSF; an
integer on line 3 and a 4-token numeric line 4 -> cube); otherwise
`"chgcar"`.

### `read(path, **kwargs)`

VASP densities go through `read_vasp(path, **kwargs)` (so `elf=`,
`locpot=`, `aeccar0=`, `aeccar2=`, `zval=`, ... pass through). Cube and XSF
files hold one density and become a `VolumetricData` with `sources={"rho":
path}`. A `.npz` goes through `io/predicted.read_predicted` (`zval=`,
`paw_radii=`, `renormalize=`). An AECCAR, ELFCAR or LOCPOT raises `ValueError(... "is not a charge
density" ...)` with a pointer to the `read_vasp` keywords: an ELFCAR read as
a density would give plausible-looking nonsense.

Test: `test_io.py::test_sniff_and_read`.

`io/__init__.py` re-exports the data model, `read`, `sniff`, every VASP
reader and writer except `read_potcar_elements`, and the cube / XSF readers
and writers, and the ML-prediction functions of `io/predicted.py`.

### `io/predicted.py`

The ML input path. `vasp_grid_shape(lattice, encut, prec)` is VASP 5's grid
rule: along each lattice vector, x = |a_i| sqrt(ENCUT/Ry)/(2 pi) in atomic
units (VASP's own RYTOEV and AUTOA, so boundary cases round as in VASP); the
wavefunction grid is the smallest even size with prime factors 2, 3, 5, 7
that is at least nint(2 WFACT x), WFACT = 2 for Accurate, 1.5 for Normal;
the density grid is twice that. For Accurate it reproduces all 6,059 grids
of the paper's dataset. `predicted_grid` adds the Cartesian positions of the
grid points. `write_predicted` stores lattice, species, fractional
coordinates, rho and JSON metadata (model, checkpoint, grid rule) in an
`.npz`; `read_predicted` loads it (or takes arrays), rejects NaN, keeps
`density_source="pseudo"`, sets `sources["origin"]="predicted"`, and with
`renormalize=True` rescales rho to N = sum ZVAL (default_zval with a warning
for elements the table does not cover), recording `charge_scale` and
`n_electrons_raw`. The metadata hook `_origin_metadata` in
`descriptors/registry.py` turns these into the `density_origin`,
`density_model` and `charge_scale` columns (`dft`, "", 1.0 for every other
reader).

Test: `test_predicted.py` (grid rule against ten dataset runs chosen at
rounding boundaries; round trip; renormalization; errors; metadata; a
prediction equal to a DFT density gives identical descriptors).

---

# Part III — Core numerics

## 9. `constants.py`

Pure constants, no functions. Spec §15 asks that no cutoff, threshold or
element datum be hard-coded outside one constants module and one data
directory; this is the constants module.

| Name | Value | Used for |
|---|---|---|
| `BOHR_ANGSTROM` | 0.529177210903 (CODATA 2018) | every Angstrom <-> bohr conversion |
| `ANGSTROM_BOHR` | 1 / a0 | volumes in bohr^3 (information measures) |
| `DENSITY_TO_AU` | a0^3 | e/Angstrom^3 -> e/bohr^3 |
| `DENSITY_FROM_AU` | 1 / a0^3 | e/bohr^3 -> e/Angstrom^3 (cube, XSF) |
| `GRADIENT_TO_AU` | a0^4 | grad rho, e/Angstrom^4 -> e/bohr^4 |
| `LAPLACIAN_TO_AU` | a0^5 | lap rho and the Hessian, e/Angstrom^5 -> e/bohr^5 |
| `HARTREE_EV` | 27.211386245988 | |
| `COULOMB_EV_ANGSTROM` | E_h a0 = 14.40 eV Angstrom | e^2 / (4 pi eps0) in the Poisson solve |
| `C_F` | (3/10)(3 pi^2)^(2/3) = 2.871234 | Thomas-Fermi constant (ELF_D, g) |
| `S_PREFACTOR` | 2 (3 pi^2)^(1/3) | reduced density gradient s |
| `SHELL_C1`, `SHELL_C2` | 0.8, 1.5 Angstrom | default shell cutoffs |
| `DERIVATIVE_BACKEND` | `"fft"` | default derivative backend |
| `LAPLACIAN_METHOD` | `"metric"` | default Laplacian |
| `FD_ORDER` | 4 | default finite-difference order |
| `HESSIAN_CHUNK` | 2^18 voxels | chunk of the eigenvalue decomposition |
| `RHO_FLOOR_AU` | 1e-8 e/bohr^3 | below it a voxel is empty for ELF_D, g, H, s, Fisher |
| `ELF_DENOMINATOR_FLOOR` | 1e-10 | clamp of rho and C_F rho^(5/3) in denominators |
| `ELF_LOCALIZED` | 0.5 | threshold of `f_ELF_localized` |
| `NCI_S_MAX`, `NCI_RHO_MAX_AU` | 0.5, 0.05 e/bohr^3 | NCI region |
| `NNM_R_CUT` | 0.8 Angstrom | non-nuclear maximum cutoff |
| `BOND_TOL` | 0.1 | first shell: abs(R_ij) <= (1 + BOND_TOL) d_i |
| `MAGNETIC_TOL` | 0.01 mu_B per atom | magnetic when sum abs(m) dV exceeds it per atom |
| `PROMOLECULE_TOL` | 1e-6 e/Angstrom^3 | free-atom table cutoff (image sums) |
| `IMAGE_SUM_BLOCK` | 8 | voxels per edge of the image-sum blocks |
| `BECKE_K`, `BECKE_CELLS` | 60, 8 | Becke truncation (section 14) |
| `MIN_PAIR_WEIGHT` | 1e-10 | smooth-partition pairs below this are dropped |
| `ION_GAUSSIAN_WIDTH` | 0.5 Angstrom | width of the ionic Gaussians of the ESP |
| `GEOMETRY_EPS` | 1e-8 Angstrom | distance tolerance, tie rule |
| `UNIFORM_TOL` | 1e-12 | a field with range below this fraction of max abs(f) is uniform |

Why the gradient and Laplacian factors are a0^4 and a0^5: rho carries
L^-3, each derivative another L^-1, and a quantity with units L^-n in
Angstrom becomes a quantity in bohr after multiplication by a0^n.

A few numerical guards stay local to their module because they are not
physical parameters: the 1e-12 Angstrom below which a direction is set to 0,
the 1e-9 slack of the tie rule's displacement comparison, the uniform
radial-table step `TABLE_STEP = 2e-4` Angstrom in `fields/deformation.py`.

## 10. `data/__init__.py` and the data files

### Data files

| File | Content | Generated by |
|---|---|---|
| `elements.csv` | Z = 1..103: symbol, covalent radius in Angstrom (Magpie `CovalentRadius`, Cordero et al. 2008; empty where the source has none), ground-state configuration such as `1s2 2s2 2p6 3s2 3p6 4s2 3d6` | `tools/generate_element_data.py` |
| `free_atoms.npz` | Z = 1..96 spherical LDA free atoms: `X_r` (700 radii, log grid 1e-5 .. 30 Angstrom), `X_orbitals` (K x 4: n, l, occupation, eigenvalue in hartree, sorted by eigenvalue), `X_density` (K x 700, float32: the density of ONE electron in each orbital, e/Angstrom^3), `_meta` | `tools/generate_free_atom_tables.py` |
| `magpie_labels.txt` | the 132 feature labels of `ElementProperty.from_preset("magpie")` (matminer 0.10.1) | once, from matminer |
| `LICENSE-matminer` | BSD licence of the Magpie data | copied |

The free-atom file stores orbital densities per electron, not total
densities, so one table serves both the all-electron free atom (weights =
occupations) and any valence count (weights filled from the highest orbital
down, section 18).

### Functions

- `DATA_DIR`: the directory of this file.
- `_table()` (`lru_cache`): parses `elements.csv` once into
  `{symbol: (Z, radius or None, configuration)}`, skipping comments and the
  header. No pymatgen at run time.
- `_row(symbol)`: one row; `KeyError("unknown element ...")`.
- `atomic_number(symbol)`, `symbol_of(z)` (linear scan),
  `covalent_radius(symbol)` (`KeyError` when the source has no value),
  `configuration(symbol)` -> list of (n, l, occupation) triples.

### `default_zval(symbol)`

The valence electron count of the standard (unsuffixed) VASP PBE PAW dataset,
from the configuration alone. It is the last resort when neither a POTCAR,
an OUTCAR nor a table gives ZVAL, and it sets the valence free-atom
reference, the Hirshfeld Z_i and the ionic charges of the ESP.

The rule:

1. Remove the preceding noble-gas core (the configuration of the largest
   noble gas with Z below the element's).
2. Lanthanides (Z 57-71) and actinides (Z 89-103): everything outside that
   core plus 8, i.e. the core's outer s2 p6 is kept together with the f
   shell (La 11, Er 22, Lu 25, U 14).
3. Everything else: the electrons outside the core, minus a filled f14
   shell, and, for a p-block element (a p shell of the highest s/p principal
   quantum number is occupied), minus the filled (n-1)d10 shell (Ga 3,
   Sn 4, Sb 5, Bi 5, Tl 3). Zn, Cd and Hg keep their d10 (12), since they
   have no p electrons.

Why: README §4 and PROGRESS.md §3. This rule matches 78 of the 87 elements
of the 6,059-structure dataset; the earlier rule (counting the d10 of the
p-block and missing the f-block semicore) was wrong for 43 of them. It
cannot know per-run semicore choices (K_pv vs K_sv, Ca_pv, Sr_sv, Y_sv,
Zr_sv, Nb_pv, ...), hence the tables of `read_vasp`. A wrong count shows as
a large `def_charge_mismatch` in whole multiples of the missing electrons.

Tests: `test_io.py::test_default_zval_matches_standard_paw_datasets`
(parametrized over 30 elements whose ZVAL was read from the dataset's
OUTCARs, from O and Fe to Er, Lu, Th, U and Pu),
`test_deformation.py::test_tables_hold_the_right_electron_counts`.

## 11. `core/grid.py`

Grid arithmetic on periodic grids.

- `frac_coords(shape)`: (n1, n2, n3, 3) fractional voxel coordinates.
- `cart_coords(shape, lattice)`: the same in Cartesian Angstrom.
- `integrate(grid)`: sum f_k dV, accumulated in float64 (also for float32
  grids).
- `mean(grid)`: the cell average.
- `fourier_resample(data, shape)`: band-limited periodic resampling with
  `scipy.signal.resample` along each axis that changes (zero-padding or
  truncating the spectrum). Exact for band-limited data, which plane-wave
  densities are, and it preserves the cell average and so the integral.
  Used for a LOCPOT on another grid and by `grid_convergence`.
- `linear_resample(data, shape)`: periodic trilinear resampling that stays
  within the data range (the ELF). When every target count divides the
  source count it is exact stride subsampling; otherwise
  `ndimage.map_coordinates(order=1, mode="grid-wrap")`.
- `resample(grid, shape, method="fourier")`: a `Grid` on another number of
  voxels, either method; returns the input unchanged when the shape
  already matches.
- `fourier_interpolate(data, frac_points, batch=256)`: trigonometric
  interpolation at arbitrary fractional points. The FFT coefficients
  F = fftn(f) / N are evaluated as the series sum_m F_m exp(2 pi i m . u)
  without forming an N x M matrix: for a batch of points, the phase
  factors of each axis are outer products, the (n1 n2, n3) coefficient
  matrix is multiplied by the axis-3 phases, then contracted with the
  axis-2 and axis-1 phases by `einsum`. Exact at grid points and
  band-limited in between. It gives rho at bond midpoints and the potential
  at the nuclei, where a linear interpolation would be biased by the
  curvature.
- `roll(data, offset)`: data at voxel k + offset; not used inside the
  package (the modules call `np.roll` directly).

Tests: `test_io.py::test_cube_units_against_known_file` (`integrate`),
`test_fields.py::test_bond_midpoint_density` (`fourier_interpolate` to 1e-6
against a closed form).

## 12. `core/derivatives.py`

Gradient, Laplacian and Hessian on a periodic, possibly non-orthogonal grid
(spec §4, "where the real numerical work is").

### The chain rule

With fractional coordinates u, x = u A, so d/du_a = a_a . grad and
grad = sum_a b_a d/du_a (a_a . b_b = delta_ab). Hence:

```
gradient    grad f = sum_a b_a (df/du_a)
Laplacian   lap f  = sum_{a,b} G[a, b] d^2 f/(du_a du_b),   G = B B^T
Hessian     H_cart = B^T H_frac B,   H_frac[a, b] = d^2 f/(du_a du_b)
```

Every cell shape is handled by differentiating along the grid axes and
applying B. The Laplacian **includes the off-diagonal metric terms**; the
spec demands it, because a diagonal-only Laplacian is exact only for
orthorhombic cells. `method="diagonal"` keeps only `G[a, a] d^2f/du_a^2` and
exists only so the error of that shortcut can be quantified.

### Coefficient tables and constants

`_FIRST` and `_SECOND` hold central-difference coefficients for orders 2,
4, 6 and 8 (first derivative: c_k for k = 1..p/2; second derivative:
c_0..c_{p/2}). `PAIRS = ((0,0), (1,1), (2,2), (0,1), (0,2), (1,2))` is the
packed storage order of the symmetric 3x3 Hessian. `_check(backend, order)`
rejects an unknown backend or an order not in the table.

### Spectral helpers

- `_frequencies(shape)`: the integer frequencies m_a of every axis on the
  `rfftn` half-grid (`rfftfreq` on the last axis, `fftfreq` on the others),
  broadcastable, as two lists: `m_full`, and `m_odd` with the Nyquist
  frequency of every even axis set to 0. The Nyquist mode of a real signal
  has no well-defined odd derivative (its derivative would be imaginary and
  is discarded differently by different code paths). Zeroing it in every
  odd-order factor keeps first derivatives real and makes the mixed second
  derivatives use the same convention in both factors, so the Hessian is
  exactly symmetric. Diagonal second derivatives keep the Nyquist term
  (m^2 is real).
- `_rfft(f)`, `_irfft(F, shape, dtype)`: `scipy.fft` wrappers that restore
  the input dtype (float32 stays float32).
- `g_squared(lattice, shape)`: |G|^2 = (2 pi)^2 sum G[a,b] m_a m_b on the
  half-grid, with the same Nyquist convention as the spectral Laplacian.
  Used only by the tests (the Poisson solver has its own
  `g_squared_full`, section 17).

### Fractional derivatives

- `_d1_fd(f, axis, n, order)`: sum_k c_k (f[i+k] - f[i-k]) * n, with
  `np.roll` wrapping (du = 1/n).
- `_d2_fd(f, axis, n, order)`: (c_0 f + sum_k c_k (f[i+k] + f[i-k])) * n^2.
- `frac_gradient(f, backend="fft", order=FD_ORDER)`: [df/du_1, df/du_2,
  df/du_3]. FFT: `irfft(2 pi i m_odd F)`.
- `frac_hessian(f, backend="fft", order=FD_ORDER, pairs=PAIRS)`: the
  requested (a, b) second derivatives. FD: diagonal pairs with the
  second-derivative stencil; mixed pairs apply the first-derivative stencil
  along a (computed once and reused) and then along b. FFT:
  `irfft(-(2 pi)^2 m_a m_b F)` with `m_full^2` on the diagonal and
  `m_odd m_odd` off it.

### Cartesian derivatives

- `gradient(f, lattice, backend="fft", order=FD_ORDER)`: shape
  (n1, n2, n3, 3); component c is sum_a B[a, c] df/du_a.
- `laplacian(f, lattice, method="metric", backend="fft", order=FD_ORDER)`:
  only the pairs actually needed are computed: the three diagonals, plus
  each off-diagonal pair whose metric element is nonzero (method
  `"metric"`). An orthorhombic cell therefore computes three second
  derivatives in either method, which is why the two methods agree to the
  last bit there. Sum `G[a,a] h_aa + 2 G[a,b] h_ab`.
- `cartesian_hessian_packed(H_frac, lattice)`: the six components of
  B^T H_frac B, skipping zero coefficients.
- `hessian(f, lattice, backend, order)`: the full (n1, n2, n3, 3, 3)
  symmetric array (tests only; the descriptors never hold it).
- `eigenvalues_packed(packed, chunk=HESSIAN_CHUNK)`: ascending eigenvalues
  lambda1 <= lambda2 <= lambda3, by `np.linalg.eigvalsh` on chunks of 2^18
  voxels assembled into (M, 3, 3). Spec §13 names the Hessian as the memory
  peak (576 MB of float64 for a full (N, 3, 3) array on a 200^3 grid); the
  chunks bound that.
- `hessian_eigenvalues(f, lattice, backend, order)`: the composition.

### Choosing a backend

`"fft"` is the default (spec §4): no stencil error on band-limited data,
and PAW pseudo-densities are smooth enough. Its weakness is a cusp: an
all-electron density (AECCAR0 + AECCAR2) has a nuclear cusp that no finite
Fourier series represents, and the spectral Laplacian rings across the whole
cell without converging under refinement. In near-empty regions the sign of
the spectral Laplacian is decided by round-off (about 1e-16 of the peak), so
sign-counting descriptors (`lnf`) are noisy there. For AECCAR input use
`derivative_backend="fd"` (README §12). `"fd"` with `fd_order=2`
reproduces the derivative-based values of the earlier pydemi version to
1e-10 (PROGRESS.md §3, 24-structure check).

The backend matters for the Hessian descriptors: on the dataset grids
`ellip_bond_avg` / `ellip_bond_std` change by 20-70% between FD2, FD4 and
FFT (Spearman FD2 vs FFT 0.87 / 0.50), while `zeta`, `fisher_information`
and `charge_FA` converge (FD4 vs FFT 1e-4 to 6e-3) (PROGRESS.md §3). The two
ellipticity descriptors are therefore tagged `stability="fragile"`, and the
`robust` extension adds a bounded ellipticity that converges (sections 25
and 27; README §6.6).

Tests (`test_derivatives.py`):

- `test_fft_exact_for_band_limited` (orthorhombic and triclinic): gradient,
  Laplacian and Hessian of two Gaussians to 1e-10 relative.
- `test_fd_convergence_order` (orders 2, 4, 6, both cells): the error ratio
  between 32^3 and 64^3 gives a rate above order - 0.6.
- `test_metric_laplacian_needs_off_diagonal_terms`: diagonal = metric on
  the orthorhombic cell; on the triclinic one the metric form is exact to
  1e-10 and the diagonal form is off by more than 5% (README quotes 12%).
- `test_hessian_symmetric_trace_and_eigenvalues` (both backends): exact
  symmetry, trace = Laplacian, ascending eigenvalues, and chunked
  eigenvalues (chunk 1000) identical to unchunked.
- `test_slater_fd_converges_away_from_the_cusp`: FD errors beyond 0.5
  Angstrom from the nuclei fall by more than 8x from 48^3 to 96^3.
- `test_fft_rings_on_a_cusp`: the documented limitation; the FFT Laplacian
  error there stays above 50% at both resolutions.
- `test_g_squared_matches_spectral_laplacian`, `test_invalid_arguments`.

## 13. `core/geometry.py`

The geometry pass every descriptor shares (spec §5), the shells, the
periodic-image machinery behind the promolecule and the partitions, the
nearest-neighbour bond census and the pair regions.

### `class NearestAtom` (frozen dataclass)

`distance` (n1, n2, n3) in Angstrom, `direction` (n1, n2, n3, 3) unit vector
nucleus -> voxel (0 at a nucleus), `atom_index` (n1, n2, n3); property
`shape`.

### `image_points(structure, reps)`

Cartesian positions of every atom image with integer lattice shifts in
[-reps_a, reps_a], after wrapping the atoms into [0, 1). Returns
(points (M, 3), owner atom (M,), shift (M, 3)).

### `reps_for(distance, lattice)`

The image range that contains every image within `distance` of any point of
the cell. A displacement of length d changes fractional coordinate a by at
most d times the norm of column a of A^{-1}, so `ceil(d * h_a) + 1` shifts
are enough (the +1 covers the voxel's own position in [0, 1)).

### Tie-breaking: `_rank` and `_nearest_images`

Atoms on high-symmetry grid points put voxels exactly on Voronoi facets:
two or more images are equidistant. A KD-tree then returns whichever it
happens to enumerate first, which depends on the image list, so the
assignment of those voxels differed between a cell, its 2x2x2 supercell and
a translated copy, and the invariance tests failed (README §8, last row).
The fix is a geometric rule: among the images within `GEOMETRY_EPS` of the
nearest, the one with the **lexicographically largest fractional
displacement** (voxel - image) wins. Fractional displacements are unchanged
by a rigid rotation or translation and are scaled by a positive factor per
axis in a supercell, so the rule makes the same physical choice in all four
cells, and it does not depend on atom numbering.

- `_rank(dk, disp, n_pick, eps)`: for each voxel, picks `n_pick` columns in
  (distance, -displacement) order: at each pick, the candidates within eps
  of the smallest remaining distance are filtered component by component to
  the largest displacement (with a 1e-9 slack), the first survivor is
  taken and removed.
- `_nearest_images(tree, x, pts4, lattice, n_pick, chunk=1 << 20)`:
  queries `n_pick + 1` neighbours per voxel in chunks. A voxel is "tied"
  when any consecutive gap among them is within `GEOMETRY_EPS` (including
  the gap just past the last pick, which could swap an image in or out).
  Only for tied voxels, more neighbours are queried (doubling k until the
  last one is clearly farther than the n_pick-th, or all points are in) and
  `_rank` orders them. Untied voxels keep the plain KD-tree answer, so the
  rule costs little.

### `assign_atoms(shape, structure, power_radii=None)`

Assigns every voxel to an atom: nearest (`power_radii=None`) or power
diagram, argmin_i abs(x - R_i)^2 - R_i^2.

- **Power diagram by lifting.** With w_i = R_i^2 and C = max w, every image
  gets a fourth coordinate sqrt(C - w_i) and every voxel 0. The squared
  4-D distance is abs(x - R_i)^2 + C - R_i^2, so the power-diagram cell is
  an ordinary nearest-neighbour query in 4-D. The lifted distance is at
  least the Euclidean one, so the image-range bound still holds.
- **Adaptive image range.** Start with one shift in each direction, query
  every voxel, compute `reps_for(max nearest distance)`, and widen until the
  range provably covers every voxel. Spec §5 suggests a fixed 3x3x3
  supercell; `test_extreme_shear_where_a_3x3x3_supercell_fails` builds a
  cell where that misses the nearest image by more than 0.01 Angstrom.
- Distance and direction are measured (Euclidean) to the chosen image; the
  direction is 0 where r <= 1e-12.

### `nearest_atom(shape, structure)`

The geometry pass itself: `assign_atoms(shape, structure, None)`.

Tests (`test_geometry.py`): `test_nearest_atom_matches_brute_force_in_a_sheared_cell`
(distances to 1e-12, indices, unit directions with u r = x - R),
`test_extreme_shear_where_a_3x3x3_supercell_fails`,
`test_power_diagram_matches_brute_force`,
`test_atom_on_a_grid_point_has_zero_direction`.

### `class Shells` (frozen dataclass) and `class ShellMasks`

`Shells(c1=SHELL_C1, c2=SHELL_C2, scaled=False)` requires 0 <= c1 < c2.
Absolute mode: c1, c2 in Angstrom. Radius-scaled mode (`scaled=True`):
multiples of the covalent radius of the nearest atom's element.
`atom_cutoffs(structure)` returns per-atom (c1_i, c2_i) in Angstrom;
`key()` returns `(c1, c2, scaled)` for cache keys. `ShellMasks` holds three
boolean arrays `core`, `bond`, `interstitial`.

### `shell_masks(geometry, structure, shells)`

core r <= c1, bond c1 < r <= c2, interstitial r > c2, with the cutoffs of
each voxel's nearest atom **plus `GEOMETRY_EPS`**. A voxel whose distance
equals a cutoff mathematically (an atom on a grid point, a cutoff that is a
multiple of the spacing) can come out 1e-15 on either side in different
cells; the tolerance puts it consistently inside. The three masks partition
the cell. Test: `test_shells_partition_the_cell_and_scale_with_radii`.

### Image sums: `class ImageBlock` and `image_blocks(shape, structure, cutoff, block=8)`

The promolecule and the Hirshfeld weights need, at every voxel, every atom
image within the free-atom cutoff. `image_blocks` walks the grid in blocks of
block^3 voxels. For each block it finds the candidate images once with a
KD-tree ball query around the block centre, of radius cutoff + the block's
own radius, and then computes all voxel-image differences and distances
densely, yielding an `ImageBlock` (`voxel` flat indices, `points`, `diff`
(M, C, 3), `distance` (M, C), `owner` (C,)). Memory is bounded by one block,
and the per-voxel KD-tree work is replaced by one query per block. Test:
`test_image_blocks_cover_every_image_within_the_cutoff` (against a
brute-force sum over 13^3 shifts, 1e-12).

### Cached accessors

- `geometry_of(vd)`: `nearest_atom(vd.shape, vd.structure)` cached under
  `("geometry",)`.
- `shells_of(vd, shells)`: masks cached under `("shells", shells.key())`.

Test: `test_geometry_is_computed_once_per_structure`.

### Nearest-neighbour bond census: `class BondCensus`, `bond_census(structure, tol)`, `bond_census_of(vd, tol)`

A `BondCensus` holds first-shell pairs (i, j, shift t): arrays `i`, `j`,
`shift` (lattice translation of j's image for the stored coordinates),
`length`, `shift_wrapped` (the same translation for positions wrapped into
[0, 1)). `midpoints_frac(structure)` is 0.5 (f_i + f_j + t).

`bond_census` uses a **per-atom first shell**: j (any image) is a neighbour
of i when abs(R_j + t - R_i) <= (1 + tol) d_i, where d_i is the distance
from i to its closest other image. That is unambiguous in multi-element
cells, where a single global cutoff would not be. Steps: wrap the atoms, grow
the image range until it covers (1 + tol) max d_i, take d_i as the
second-nearest image distance (the nearest is the atom itself), collect the
ball of each atom, skip the atom itself, and store each pair under a
canonical key ((i, j, t) or (j, i, -t), whichever is smaller) so a pair seen
from both ends is counted once. Finally the wrapped shifts are converted to
the stored coordinates, `t = t_w - floor(f_j) + floor(f_i)`.
`bond_census_of` caches under `("bond_census", tol)`.

### `pair_regions(shape, structure)` and `_lex_less(u, v)`

The second-order Voronoi assignment behind
`bond_charge_transfer_pair_*`: for every voxel, its two nearest atom images
(`_nearest_images` with `n_pick=2`, the same tie rule, the same adaptive
image range grown on the second distance). The pair is written canonically
as in the census with wrapped positions: t = shift of the second - shift of
the first, swapped when b < a or (a == b and -t < t lexicographically,
`_lex_less`). The region of pair (i, j, t) is then every voxel whose two
nearest images are i and j + t. These regions tile the cell, need no bond
path, and are invariant for the same reason as the geometry pass. Returns
(i, j, t). Cost: 2.3 s of the 8.4 s of a 96^3, 16-atom cell (README §11;
a speed-up is listed as optional in PROGRESS.md §5).

Test: `test_deformation.py::test_pair_regions_tile_the_cell_and_match_the_census`
(one pair per voxel, canonical order, every census bond has a region, finite
charge transfers).

## 14. `core/partition.py`

Assignment of voxels to atoms behind one interface (spec §9).

### The interface

`PairChunk` (frozen dataclass): `voxel`, `atom`, `weight`, `distance`,
`direction`, where distance and direction are measured from **that atom's
image** (the one the scheme used) to the voxel. Every scheme yields
memory-bounded chunks of such pairs: hard schemes one pair per voxel with
weight 1, smooth schemes several pairs per voxel with weights summing to 1.
A site-restricted sum is then one code path for every scheme,

```
X^(i) = sum over pairs with atom i of  w * g(field_k, r_ik, u_ik)
```

i.e. the indicator 1(i(k) = i) replaced by w_i(r_k), which is what spec §9
asks for ("a single code path, not four").

`class Partition`: `scheme`, `n_atoms`, `pairs()` (abstract generator) and
`site_sum(f)` = sum_k w_i(k) f_k for every atom, by `np.bincount` over the
chunks.

### `class HardPartition(Partition)`

`HardPartition(assignment, n_atoms, scheme, chunk=1 << 20)` wraps a
`NearestAtom` (nearest atom or power diagram). `pairs()` yields chunks of
2^20 voxels with weight 1 and the assignment's distance and direction.
`site_sum` overrides the generic one with a single `bincount` over the
labels.

### `power_radii(vd)` and `partition_of(vd, scheme)`

`power_radii` is the covalent radius of every atom. `partition_of` builds and
caches the partition under `("partition", scheme)`: `"nearest"` wraps the
cached geometry pass itself (no second assignment), `"power"` calls
`assign_atoms` with the covalent radii, `"becke"` and `"hirshfeld"` build
the smooth schemes; anything else raises `ValueError`.

### Becke: `_renormalized`, `_becke_step`, `class BeckePartition`

```
w_i(r) = P_i(r) / sum_j P_j(r),   P_i = prod_{j != i} s(mu_ij),
mu_ij = (|r - R_i| - |r - R_j|) / |R_i - R_j|,
s(mu) = (1 - f(f(f(mu)))) / 2,   f(p) = 3p/2 - p^3/2
```

(Becke, J. Chem. Phys. 88, 2547 (1988); pure geometry, no atomic size
adjustment.) `_becke_step(mu)` is s(mu).

Becke's product was designed for molecules. In a solid the number of
competing images at distance R grows as R^2 while each factor approaches 1
only as (d/R)^8, so the weights converge slowly with the number of images in
the product. The module docstring gives the measurement: FeNi3, maximum
weight error 1.6e-2 with 60 competitors and 1.0e-3 with 300. The
implementation therefore fixes a truncation and makes it part of the
definition, recorded in the metadata as `partition = "becke(k=60,cells=8)"`:

- `BeckePartition(shape, structure, k=BECKE_K, cells=BECKE_CELLS,
  chunk=1 << 13)`, with 1 <= cells <= k.
- For each chunk of 8,192 voxels, the k = 60 nearest images are queried;
  the image range is widened until it contains every image within the 60th
  distance of every voxel.
- The cell functions P_i are computed for the `cells` = 8 nearest images
  only, each product running over all 60; mu_ii = 0/0 is replaced by 0 and
  s_ii by 1.
- w = P / sum P over those 8 cells; pairs below `MIN_PAIR_WEIGHT` are
  dropped and `_renormalized` rescales the rest so each voxel's weights sum
  to exactly 1.

The truncation breaks exact ties between equidistant images at the 60th
place, so two symmetric atoms get equal charges only to about 1e-5 (README
§12; `test_symmetric_atoms_get_equal_charges` uses rel 1e-5 for Becke and
1e-10 for Hirshfeld).

### Hirshfeld: `class HirshfeldPartition`, `hirshfeld_tables(vd)`

```
w_i(r) = rho_free_i(r - R_i) / sum_j rho_free_j(r - R_j)      over all periodic images
```

`HirshfeldPartition(shape, structure, tables)` reuses the promolecule's
machinery (spec §9): `fields.deformation.atom_density_blocks` gives, block by
block, the free-atom density of every contributing image at every voxel, the
weights are those densities over their sum, small pairs are dropped and the
rest renormalized. A voxel beyond every free-atom cutoff (total 0) goes
wholly to its nearest image. Free-atom densities decay exponentially, so
Hirshfeld has none of Becke's convergence problem.

`hirshfeld_tables(vd)` picks the tables that match the density: tabulated
all-electron atoms for an all-electron density, tabulated valence atoms
(ZVAL electrons, `vd.zval` or `default_zval`) for a pseudo-density.

### `reference_electrons(vd)` and `hirshfeld_charges(vd)`

`hirshfeld_charges(vd)` returns q_i = Z_i - int w_i rho dV for comparison
with Bader charges, from the cached Hirshfeld partition. `reference_electrons`
gives Z_i: the atomic number for an all-electron density, **ZVAL for a
pseudo-density**. This is one of the four documented corrections: the spec
writes Z_i, but a CHGCAR holds only ZVAL electrons per atom, and q_i = Z_i -
N_i needs the electrons the density actually holds; with the atomic number
every charge would be off by the core count. Inside the augmentation spheres
the free-atom weights keep their all-electron shape while the CHGCAR is
pseudized, so the partition there is approximate (docstring).

Tests (`test_partitions.py`):

- `test_weights_sum_to_one_and_charge_is_conserved` (all four schemes,
  1e-12).
- `test_symmetric_atoms_get_equal_charges` (Becke, Hirshfeld).
- `test_becke_two_atom_weight_is_becke_s` (s(mu) + s(-mu) = 1, s(0) = 1/2).
- `test_hirshfeld_charges_of_a_promolecule`: the density is the free-atom
  superposition itself, so w_i rho is atom i's own free density and q_i =
  ZVAL_i - int rho_free_i dV exactly (1e-10), which is zero up to point
  sampling (5e-3).
- `test_site_descriptors_follow_the_partition`,
  `test_partitioned_descriptors_are_invariant` (power, Hirshfeld and Becke
  under supercell, translation and rotation at 1e-6 relative).
- `test_sites.py::test_power_partition_is_a_different_tiling`,
  `test_magnetic.py::test_site_moments_sum_to_the_net_moment`.

---

# Part IV — Fields

Each field is an array with the shape of rho (spec §6). The descriptor layer
names them in a small field registry (`descriptors/registry._FIELDS`):
`rho`, `abs_m`, `elf_file`, `elf_d`, `potential`, `delta_rho`. The numerical
work lives in the four modules below; where a field is resolved from the
options (which ELF, which potential, which reference) is in
`descriptors/bonding.py` (section 27).

## 15. `fields/density.py`

The density and magnetization fields, and the derivative cache every field
shares.

- `rho(vd)`: `vd.rho.data`.
- `abs_m(vd)`: abs(m) for a collinear run, the vector norm
  sqrt(m_x^2 + m_y^2 + m_z^2) for a non-collinear run, and a zero field for a
  run without spin; cached under `("field", "abs_m")`. The zero field keeps
  every operator applicable; the magnetic descriptors test `is_magnetic`
  before using it (section 29).

### `class Derivatives`

Lazily computed, cached derivatives of one field:
`Derivatives(values, lattice, backend, order)`.

- `gradient`: (n1, n2, n3, 3), computed on first access.
- `gradient_norm`: abs(grad f), from the cached gradient.
- `laplacian(method="metric")`: one cached array per method.
- `hessian_eigenvalues`: forms the fractional Hessian once; if the metric
  Laplacian has not been computed yet it is assembled from the same six
  fractional second derivatives (no second set of FFTs); the Cartesian
  packed Hessian is formed, the fractional one is deleted, and the
  eigenvalues are computed in chunks. Only the (N, 3) eigenvalue array is
  kept, never the Hessian (spec §13).

### `derivatives(vd, name, values, backend="fft", order=FD_ORDER)`

Returns the cached `Derivatives` of field `name`, creating it with
`values(vd)` on first use, under `("derivatives", name, backend, order)`
(the order is stored as 0 for the FFT backend, which has none). Descriptors
call it through `registry.field_derivatives`, which supplies the backend and
order from the options. So the gradient of rho is computed once per backend
and shared by `zeta`, the kinetic terms, the anisotropy tensor, the Fisher
information and the per-site zeta.

## 16. `fields/elf.py`

ELF reconstruction from rho alone (Tsirelson and Stash, Chem. Phys. Lett.
351, 142 (2002)) and the local energy densities that share its kinetic term
(Abramov, Acta Cryst. A53, 264 (1997)); spec §6.2 and §8.1. Everything here
is in **atomic units**:

```
C_F   = (3/10) (3 pi^2)^(2/3) = 2.871234
t_P   = C_F rho^(5/3) + (1/72) |grad rho|^2 / rho + (1/6) lap rho     (= g)
D_P   = t_P - |grad rho|^2 / (8 rho)
D_h   = C_F rho^(5/3)
ELF_D = 1 / (1 + (D_P / D_h)^2)

g = t_P,   v = (1/4) lap rho - 2 g,   H = g + v = (1/4) lap rho - g
```

The expansion's coefficients belong to atomic units; spec §16 lists ELF_D in
Angstrom-based units among the silent failures.

### `class KineticTerms` (frozen dataclass)

`rho` (e/bohr^3, clipped at 0), `grad2` (abs(grad rho)^2), `lap`, `valid`
(rho > `RHO_FLOOR_AU`), `tf` (D_h), `w8` (abs(grad rho)^2 / (8 rho)), `g`.

### `kinetic_terms(rho_A, gradient_norm_A, laplacian_A)`

Takes rho, abs(grad rho) and lap rho in Angstrom units and converts with
`DENSITY_TO_AU`, `GRADIENT_TO_AU`, `LAPLACIAN_TO_AU`. Guards: negative
values of a PAW pseudo-density are clipped to 0; the rho in the denominator
of the Weizsacker term is clamped at `ELF_DENOMINATOR_FLOOR` (1e-10);
voxels below `RHO_FLOOR_AU` (1e-8 e/bohr^3) are marked invalid.

The kinetic terms are computed once per (backend, order, Laplacian method)
by `descriptors.bonding.kinetic` and shared by ELF_D, `f_H_negative`,
`H_bond_mean`, `G_over_rho` and the NCI descriptors, as the spec asks
("compute g once and reuse").

### `elf_d(kt)`

chi = (g - w8) / max(D_h, floor), ELF_D = 1 / (1 + chi^2), set to 0 where
invalid. ELF_D is in [0, 1] everywhere by construction. In the uniform
electron gas (no gradient, no Laplacian) g = D_h, chi = 1 and ELF_D = 1/2.

### `energy_densities(kt)`

(g, v, H) in hartree / bohr^3.

Tests (`test_fields.py`): `test_elf_d_is_in_unit_interval_on_analytic_data`,
`test_elf_d_is_in_unit_interval_on_real_data` (spec §6.2: a real, partly
negative PAW CHGCAR, FeNi3_221 from the dataset; skipped when the file is not
available), `test_uniform_electron_gas_limits` (ELF_D = 1/2, g = C_F
rho^(5/3), `f_H_negative` = 1, `H_bond_mean` = -g, `G_over_rho` = g / rho,
`ELF_bond_avg` = 1/2).

## 17. `fields/potential.py`

Electrostatic potentials on the density grid, in eV (spec §6.3).

```
V_H(G)   = 4 pi rho(G) / |G|^2              G != 0,   V_H(0) = 0,   |G| = 2 pi |B^T n|
V_esp(G) = 4 pi [rho(G) - rho_ion(G)] / |G|^2
rho_ion(G) = (1/V) sum_i Z_i exp(-|G|^2 sigma^2 / 2) exp(-i G . R_i)
```

times e^2 / (4 pi eps0) = `COULOMB_EV_ANGSTROM`. `V_H` is the potential
energy of an electron in the field of the electron density (positive where
the density is); the full electrostatic potential (`"esp"`) adds
Gaussian-smeared ionic charges Z_i of width `ION_GAUSSIAN_WIDTH` = 0.5
Angstrom. Point charges cannot be represented on the grid; Gaussians can,
and a few widths away from its centre a Gaussian charge acts as a point
charge. Z_i is ZVAL for a
pseudo-density and the atomic number for an all-electron density.

The G = 0 term is arbitrary. Every potential is referenced to its cell
average (V(G=0) = 0 here, and a LOCPOT read from file is shifted the same
way by `centred`), so the potential descriptors do not depend on that
convention.

- `g_squared_full(lattice, shape)`: abs(G)^2 on the `rfftn` half-grid with
  every frequency, Nyquist included (unlike the derivative convention,
  which drops Nyquist from odd factors).
- `_poisson(F, G2, shape)`: 4 pi k F / G^2, zero at G = 0, inverse FFT.
- `hartree_potential(rho, lattice)`: one FFT.
- `ionic_form(structure, charges, shape, sigma=ION_GAUSSIAN_WIDTH)`: the
  FFT coefficients of the Gaussian ionic density in the same convention as
  `rfftn(rho)` (structure factor x Gaussian form factor x N/V).
- `electrostatic_potential(rho, structure, charges, sigma=ION_GAUSSIAN_WIDTH)`:
  Poisson solve of rho - rho_ion.
- `centred(values)`: values minus their cell average.
- `ion_charges(species, all_electron, zval=None)`: Z_i (atomic numbers, or
  ZVAL with the `default_zval` fallback).

Tests (`test_fields.py`):

- `test_hartree_potential_of_a_gaussian`: V(r) - V(0) =
  k N [erf(sqrt(a) r)/r - 2 sqrt(a/pi)] + (2 pi k N / 3V) r^2 in a 10 Angstrom box, to 2e-3; the
  r^2 term is the neutralizing background implied by V(G=0) = 0.
- `test_electrostatic_potential_of_a_neutral_cell_vanishes`: electron
  Gaussians with the ion width and charge cancel the ionic term; the result
  is below 1e-6 of the Hartree potential.
- `test_potential_sources`: Hartree by default, ESP differs, a LOCPOT is
  used when present and centred, `potential_source="locpot"` without one
  raises.

## 18. `fields/deformation.py`

The promolecule and the deformation density (spec §6.1):

```
delta_rho(r) = rho_crystal(r) - sum_i rho_free[element(i)](r - R_i)
```

The promolecule is a superposition of spherical free-atom densities placed
at the nuclei and summed over every periodic image out to where each
free-atom density falls below `PROMOLECULE_TOL` (1e-6 e/Angstrom^3), not
only the parent cell.

### Reference sources

The option `deformation_reference` (resolved in
`descriptors.bonding.deformation_reference`, recorded in the metadata):

- `"aeccar0"`, for an all-electron density (AECCAR0 + AECCAR2). **Documented
  correction.** The spec describes AECCAR0 as "already an element-wise
  superposition" to be used as the promolecule. It is the frozen-core
  density: rho - AECCAR0 is the whole valence density (AECCAR2), not a
  deformation density. pydemi uses AECCAR0 for the core part and adds the
  tabulated free-atom **valence** densities (ZVAL electrons per atom):
  promolecule = AECCAR0 + sum_i rho_free_val,i, so delta_rho = AECCAR2 -
  sum_i rho_free_val,i.
- `"tabulated"`: the shipped spherical LDA atoms (`data/free_atoms.npz`),
  all electrons for an all-electron density, the ZVAL highest-energy
  electrons for a PAW pseudo-density. For a CHGCAR the free-atom valence
  keeps its all-electron shape inside the augmentation spheres, where the
  CHGCAR is pseudized, so delta_rho there also reflects the pseudization
  (hence the `paw` extension).
- `"custom"`: a directory of `<Element>.dat` (or `.txt`, `.csv`) files with
  two columns, r in Angstrom and rho in e/Angstrom^3, from the user's own
  isolated-atom calculations.
- `"auto"`: `"aeccar0"` when AECCAR0 was read, else `"tabulated"`.

### Radial references

- `TABLE_STEP = 2e-4`: step (Angstrom) of the uniform lookup tables.
- `_tabulated()` (`lru_cache`): the npz file loaded once into a dict.
- `tabulated_radial(element, part, zval=None)`: (r, rho) of the shipped
  atom. `part="total"` weights every orbital density by its occupation.
  `part="valence"` fills `zval` electrons (default `default_zval`) from the
  highest-energy orbital down (a stable sort on the eigenvalue), taking a
  fraction of an orbital when ZVAL ends inside it; a ZVAL above the
  element's electron count raises `ValueError`. Highest energy first
  because the valence electrons of a PAW dataset are the outermost ones:
  Fe with ZVAL 8 gets 4s2 3d6, Fe_pv with ZVAL 14 also the 3p6.
- `class CustomReference(directory)`: `radial(element)` loads the first of
  `<El>.dat`, `.txt`, `.csv` (comma-separated for `.csv`, `#` comments).
- `class RadialTable` (frozen): `values` on a uniform grid, `step`,
  `r_max`; calling it with distances does a nearest-neighbour lookup
  `values[rint(d / step)]`, clipped to the last entry, which is a 0 guard.
- `radial_table(r, rho, tol=PROMOLECULE_TOL, step=TABLE_STEP)`: sorts the
  data, sets r_max at the first radius after the last point where rho >=
  tol (so the density is below tol "for good"), fits one cubic spline
  (`CubicSpline`, no extrapolation) and samples it on 0 .. r_max; negative
  or non-finite values become 0, values beyond r_max 0. The spec asks for
  "nearest-neighbour-in-radius plus cubic spline": the spline is evaluated
  once per element, and every voxel-image distance afterwards costs one
  array index.

### Image sums

- `class AtomDensityBlock` (frozen): `voxel`, `density` (M, C) free-atom
  density of image c at voxel m, `owner`, `distance`, `diff`.
- `atom_density_blocks(shape, structure, tables, block=IMAGE_SUM_BLOCK)`:
  runs `core.geometry.image_blocks` with the largest r_max of the elements
  as the cutoff and fills the density columns element by element. Shared
  by the promolecule and the Hirshfeld partition.
- `promolecule(shape, structure, tables)`: the per-voxel sum of the blocks.
- `reference_tables(structure, kind, part, zval=None, custom=None)`: one
  `RadialTable` per element for `"tabulated"` (with part and ZVAL) or
  `"custom"` (the file as given; part and ZVAL do not apply).

### Point sampling and `def_charge_mismatch`

The promolecule is sampled at the voxel positions, not averaged over voxels.
A nucleus exactly on a grid point over-counts the cusp of its free-atom
density; the docstring measures one O atom (valence): +5.2% at 0.15 Angstrom
spacing, +0.18% at 0.06 Angstrom; off the grid points the integral is exact
to 0.1%. The resulting int delta_rho dV is reported per structure as
`def_charge_mismatch` in the metadata rather than hidden by a renormalization.
On the 6,059-structure rerun its absolute value has median 0.06 e, 99th
percentile 0.78 e and maximum 2.9 e (80 atoms) (PROGRESS.md §3).

### Helpers not used inside the package

- `split_shell_warning(element, zval)`: a message when ZVAL splits a
  partly-counted shell of the tabulated atom, else None.
- `radial_profile(values, structure, bins=400)`: spherical average about the
  single atom of a one-atom cell out to half the smallest face distance;
  builds a `custom` reference table from an isolated-atom calculation.

Tests (`test_deformation.py`):

- `test_tables_hold_the_right_electron_counts` (H, O, Si, Fe, Ni, Zn, Yb,
  U): total = Z, valence = `default_zval`, `zval=1` gives 1 electron, all to
  2e-4.
- `test_radial_table_cutoff_and_lookup`: r_max of exp(-2r) at tol 1e-6
  within 2% of -ln(1e-6)/2; lookups to 1e-3; 0 beyond r_max.
- `test_promolecule_of_one_atom_integrates_to_its_electrons`: an O atom off
  the grid points integrates to 6 within 2e-3; on a grid point the error
  falls from 60^3 to 90^3 to 150^3 and is below 0.012 e at 150^3.
- `test_custom_reference_equal_to_the_crystal_atoms_gives_zero`,
  `test_custom_reference_with_half_the_charge` (int delta_rho = N/2 = 1),
  `test_aeccar0_reference_is_core_plus_tabulated_valence` (delta_rho = 0 to
  1e-12 when rho is exactly core + tabulated valence, while `"tabulated"`
  gives a nonzero polarity), `test_aeccar0_needs_the_core_density`.

---

# Part V — Operators

Each operator is a pure function of arrays, written once and applied to any
field (spec §7). They return NaN for 0/0; the descriptor layer converts that
into a flagged sentinel.

## 19. `operators/moments.py`

```
m_n = sum_k w_k r_k^n / sum_k w_k,      w = field ("signed") or |field| ("abs")
```

- `_weights(field, weight)`: the field or its absolute value; any other
  `weight` raises.
- `radial_moment(field, r, order, weight="abs", mask=None)`: m_n over all
  voxels or the masked ones; NaN when the weight sum is 0 or not finite.
- `radial_variance(field, r, weight="abs", mask=None)`: m2 - m1^2.

The bonding moments of rho use `"signed"` (rho itself, as the spec writes
sum rho_k r_k; a slightly negative pseudo-density simply contributes with its
sign); the deformation and spin moments use `"abs"`, since delta rho and m
change sign.

## 20. `operators/fractions.py`

```
f_shell = sum_{k in shell} w_k / sum_k w_k
```

`shell_fraction(weights, shell, within=None)`: `within` restricts both sums
(e.g. to delta_rho > 0). NaN when the total is 0.

## 21. `operators/anisotropy.py`

```
zeta = 1 - sum_k |grad f_k . u_k| / sum_k |grad f_k|
T_ab = sum_k d_a f_k d_b f_k / sum_k |grad f_k|^2        (tr T = 1)
FA   = sqrt(3/2) ||T - I/3||_F / ||T||_F
```

- `gradient_anisotropy(gradient, direction, mask=None)`: zeta = 0 when every
  gradient is radial about the nearest nucleus (well-separated spherical
  atoms), growing as density concentrates off the radial directions. NaN
  when the gradient vanishes everywhere.
- `anisotropy_tensor(gradient)`: T = g^T g / trace; NaN-filled when the
  trace is 0.
- `fractional_anisotropy(T)`: 0 for T = I/3, 1 when every gradient is
  parallel (T = e e^T).

## 22. `operators/laplacian.py`

```
sign fraction            (1/N) sum_k 1(lap f_k < 0)
charge-weighted variant  sum_{lap f < 0} w_k / sum_k w_k
concentration            sum_{lap f < 0} |lap f_k| / sum_k |lap f_k|
```

- `negative_fraction(lap, mask=None)`: NaN on an empty mask.
- `weighted_negative_fraction(lap, weights)`.
- `concentration(lap, mask=None)`.

**Why `lap_concentration` is 1/2** (the first documented correction). On a
periodic grid sum_k lap f_k = 0 exactly: the G = 0 Fourier coefficient of a
spectral Laplacian is 0, and the coefficients of every periodic difference
stencil sum to 0. So the negative and positive parts of lap f have equal
magnitude and the whole-cell concentration is 1/2 for any field, up to
round-off. It becomes informative only restricted to a region (`mask`),
e.g. the valence region r > c1, where flux through the core boundary breaks
the balance. Test: `test_tier1.py::test_lap_concentration_is_identically_one_half`
(both backends, to 1e-10; the valence variant differs from 1/2 by more than
1e-3).

## 23. `operators/topology.py`

Grid topology by direct operations that always terminate (spec §7, §8.2).

### Constants and `_shift`

`NEIGHBOURS_26` (the 26 offsets), `_POS` (seven positive edge vectors) and
`FREUDENTHAL_14` = +-`_POS`: the edges of the Freudenthal triangulation of
the grid, which splits each cube into six tetrahedra along the (1, 1, 1)
diagonal. `_shift(a, off)` is `a` at voxel k + off, periodic.

### `lower_mask(values, offsets)`

An int64 bitmask per voxel: bit k is set when neighbour `offsets[k]` is
lower. "Lower" is the lexicographic order of (value, flat index): equal
values are ordered by index (symbolic perturbation), so every comparison is
strict and plateaus cannot produce ambiguous extrema.

### `_link_components()` (`lru_cache`)

A table of 2^14 = 16,384 entries: for every subset of the 14 Freudenthal
neighbours (a bitmask), the number of connected components of that subset
in the link, where two link vertices are adjacent when their difference is
itself a Freudenthal edge. Computed once per process in plain Python.

### `extremum_census(values)`

- **Maxima and minima: 26-neighbour comparison** with modular wrapping, as
  the spec asks: a voxel whose 26 neighbours are all lower (all bits set) is
  a maximum, one with none lower a minimum.
- **Saddles: the Freudenthal link.** The spec's 26-neighbour census cannot
  classify saddles: the 26-neighbour shell is not a triangulated sphere, and
  every choice of adjacency on it miscounts the 3 + 3 saddles of
  cos 2 pi x + cos 2 pi y + cos 2 pi z (README §8). The link of a vertex in the
  Freudenthal triangulation is a triangulated 2-sphere, so piecewise-linear
  Morse theory (Banchoff, Amer. Math. Monthly 77, 475 (1970)) applies: a
  non-extremal voxel whose lower link has c components contributes c - 1
  index-1 saddles, and one whose upper link has c components contributes
  c - 1 index-2 saddles (multiplicities count monkey saddles correctly):

  ```
  n_saddle1 += components(lower link) - 1
  n_saddle2 += components(upper link) - 1
  ```

- **`euler_consistency = n_max - n_saddle2 + n_saddle1 - n_min`.** With
  Freudenthal extrema this sum is identically 0 (the Euler characteristic
  of the 3-torus), so n_saddle1 - n_saddle2 = n_min^14 - n_max^14. With
  26-neighbour extrema it therefore equals

  ```
  euler_consistency = (n_max^26 - n_max^14) - (n_min^26 - n_min^14)
  ```

  the number of extrema whose status depends on the stencil. It is 0
  exactly when both neighbourhoods find the same extrema; a nonzero value is
  the quality flag of spec §10.

Returns the four counts, `euler_consistency`, the boolean map `maxima` and
the 26-neighbour mask `lower26` (reused by the basins).

On real data the flag is informative: on the 6,059-structure dataset
`euler_consistency` is nonzero for 91% of the structures (median 5% of all
critical points), and the identity above was checked on 200 of them (README
§7). The cause is stencil-scale structure, PAW-pseudized regions and
low-amplitude ripple, not coarse grids: the median grid spacing is 0.064
Angstrom both where it is zero and where it is not. For the same reason the
saddle counts are tagged fragile (section 28).

Test: `test_structural.py::test_census_of_a_periodic_function_with_known_critical_points`
(cos 2 pi x + cos 2 pi y + cos 2 pi z plus a small periodic tilt: exactly
(1, 1, 3, 3) and Euler 0 at 12^3, 24^3 and 48^3).

### `ascent_basins(values, lattice, lower26)`

Steepest-ascent basins over the 26 neighbours. Each voxel points to its
upper neighbour (a clear bit in `lower26`) of largest slope, value difference
over the Cartesian length of the offset; a maximum points to itself. Pointer
jumping (`flat = flat[flat]` until it stops changing) resolves every chain
in O(log N) sweeps. Every chain strictly increases in (value, index) order,
so it terminates at a 26-neighbour maximum. Returns the flat index of that
maximum for every voxel.

### `spans(mask)`

(3,) booleans: does a face-connected cluster of `mask` wrap along a1, a2,
a3?

1. `scipy.ndimage.label(mask)` with the default 6-connectivity labels the
   clusters inside the unwrapped cell.
2. A union-find over the labels stores, for every node, its lattice
   offset relative to its parent (`position(node) = position(parent) +
   offset`); `find` compresses paths and accumulates offsets.
3. For each axis, every pair of labels facing each other across the
   periodic boundary (last plane, first plane) is merged with offset e_axis.
   If the two already share a root, the loop closes: a nonzero
   `offset[a] + e - offset[b]` is a winding vector, and every axis where it
   is nonzero spans.

**Why winding instead of face contact** (part of the second documented
correction). The spec's test is "touches both faces and those contacts belong
to the same component after the merge". A blob sitting across the periodic
boundary touches both faces without spanning anything, and whether a finite
blob crosses a face depends on where the cell origin lies, so the face test
makes the percolation level change under a translation. A cluster spans
exactly when it meets its own periodic image, i.e. has a nonzero winding
vector. Tests: `test_spans_needs_a_winding_cluster` (a blob across the
a-face does not span; a rod along a1 spans a1 only),
`test_percolation_does_not_depend_on_the_cell_origin` (an atom at the cell
corner vs the translated cell).

### `percolation_levels(values, max_steps=256)`

(3,) spanning levels by bisection over the sorted distinct voxel values u.
Per axis a bracket [lo, hi] of indices into u: lo = -1 (the level below the
minimum, where the whole cell is included and spans), hi = the last index
({f > max} is empty). Each step bisects the axis with the widest bracket;
one `spans` call at the midpoint updates every axis whose bracket contains
it. It stops when every bracket has width 1 (or after `max_steps`) and
returns u[hi]: the lowest voxel value whose strict super-level set no longer
spans. That is the supremum of the spanning levels, the density at the
bottleneck of the best connecting path.

**Why the highest spanning level** (the second documented correction). The
spec writes "the lowest level whose super-level set spans". The lowest such
level is always the minimum density, where the whole cell spans, so it
carries no information about connectivity; the bottleneck density is the
highest level that still spans. Tests:
`test_simple_cubic_gaussians_percolate_at_the_midpoint_density` (the spec's
test case: a simple-cubic lattice of Gaussians percolates at rho(a/2, 0, 0)
in all three directions, to 1e-12, with `perc_anisotropy` 0),
`test_percolation_level_is_the_highest_spanning_level`.

## 24. `operators/sitestats.py`

Site aggregation and the variance decomposition (spec §7, §8.4).

### `class SiteSums` and `site_sums(partition, field, gradient, c1, c2, dV)`

One pass over a partition's pairs accumulates, per atom i, every
site-restricted sum of one field f:

| Attribute | Sum |
|---|---|
| `total` | sum w f |
| `r1`, `r2` | sum w f r, sum w f r^2 |
| `core` | sum over r <= c1_i of w f |
| `bond` | sum over c1_i < r <= c2_i of w f |
| `radial` | sum w abs(grad f . u) |
| `gnorm` | sum w abs(grad f) |

plus `dV`. Here w is the pair weight, r and u are measured from atom i's
image (for a smooth partition a voxel contributes to several atoms, each
with its own distance and direction), and the shell tests use atom i's
cutoffs with `GEOMETRY_EPS`. `gradient=None` skips the two gradient sums.

### Per-site values

`_ratio(num, den)` is num/den with NaN where den = 0.

```
site_m1     m1^(i)     = sum w f r / sum w f
site_f_bond f_bond^(i) = sum_{c1 < r <= c2} w f / sum w f
site_zeta   zeta^(i)   = 1 - sum w |grad f . u| / sum w |grad f|
site_charge Q^(i)      = sum w f dV
```

### `variance_decomposition(x, groups)` and `site_statistics(x, groups)`

The one-way ANOVA over elements, with population variances and the site
fractions w_e = n_e / n as weights:

```
Var_i(X) = sum_e w_e Var_{i in e}(X)  +  sum_e w_e (Xbar_e - Xbar)^2
           ---------- within ---------    ----------- between ---------
```

This is the law of total variance, exact for any grouping. The spec's text
defines the within term as "the mean over elements" and the between term as
"Var_e of the element means"; with unequal site counts those unweighted
forms do not add up to Var_i(X). Its own identity carries the weights w_e,
and the code uses them in both terms, which is what makes the identity hold.

`site_statistics` returns a `SiteStatistics` (frozen): `std`, `range`,
`max`, `min`, `within` (NaN when no element has two or more sites),
`between`, `n_sites`, `n_elements`, `max_sites_per_element`.

Tests (`test_sites.py`): `test_anova_identity_holds_exactly` (random values,
five uneven groups, 1e-12), `test_anova_identity_on_computed_site_values`,
`test_nearest_partition_tiles_space` (site charges sum to the cell charge,
and sum r1 / sum total reproduces the whole-cell `m1` to 1e-12).

---

# Part VI — Descriptors

## 25. `descriptors/registry.py`

Registration and metadata (spec §8), the options `featurize` takes, and the
shared accessors every descriptor uses.

### Allowed values

`DOMAINS = ("bonding", "structural", "magnetic", "heterogeneity",
"compositional")`, `EXTENSIONS = ("paw", "robust")`, `STABILITY = ("robust",
"fragile")`, `PARTITIONS = ("nearest",
"power", "becke", "hirshfeld")`, `DEFORMATION_REFERENCES = ("auto",
"aeccar0", "tabulated", "custom")`, `ELF_SOURCES = ("auto", "reconstruct",
"file")`, `POTENTIAL_SOURCES = ("auto", "locpot", "hartree", "esp")`.

### `class FeatureOptions` (frozen dataclass)

The settings of one `featurize` call, attached to the `VolumetricData` as
`vd.options` and recorded in the metadata.

| Field | Default |
|---|---|
| `domains` | all five `DOMAINS` |
| `partition` | `"nearest"` |
| `shells` | `Shells(SHELL_C1, SHELL_C2)` |
| `deformation_reference` | `"auto"` |
| `custom_reference` | None |
| `elf_source` | `"auto"` |
| `potential_source` | `"auto"` |
| `laplacian_method` | `LAPLACIAN_METHOD` (`"metric"`) |
| `derivative_backend` | `DERIVATIVE_BACKEND` (`"fft"`) |
| `fd_order` | `FD_ORDER` (4) |
| `nnm_r_cut` | `NNM_R_CUT` (0.8 Angstrom) |
| `extensions` | () |

`__post_init__` checks every value against its allowed list and requires a
directory for `deformation_reference="custom"`. Frozen, so an options object
can be shared between views without being mutated. `nnm_r_cut` has no
`featurize` keyword; it is set only by building a `FeatureOptions` directly.

### `make_options(shells=None, **kwargs)`

Normalizes the user-facing forms: `shells` may be None (defaults), a
`Shells`, or a (c1, c2) tuple in Angstrom; `domains` and `extensions` may be
comma-separated strings or sequences; `domains=None` means all five.

### `options(vd)`

`vd.options` when it is a `FeatureOptions`, else a default one. So every
descriptor function can also be called on a bare `VolumetricData`, which
the tests do (`vd.with_options(FeatureOptions(...))` or no options at all).

### `class Sentinel`, `Result`, `DescriptorFn`

`Sentinel(value, case)` (frozen): the documented constant `value` for the
documented degenerate `case`. A descriptor returns `Result = float |
Sentinel`; `DescriptorFn` is `Callable[[VolumetricData], Result]`.

### `class DescriptorSpec` (frozen dataclass) and `REGISTRY`

`name`, `domain`, `field`, `requires` (a tuple of tags such as `"gradient"`,
`"geometry"`, `"shells"`, `"hessian"`, `"partition"`, `"paw"`, used for
documentation and by `sensitivity_sweep`), `units`, `range`, `intensive`,
`sentinel_cases` (case -> value), `references`, `adopted`, `extension`,
`func`, `doc`, `stability` (`"robust"` by default). The property `formula` is
the first paragraph of the docstring, joined onto one line; that is the
"Definition" column of the README tables. `REGISTRY` is a `dict[str,
DescriptorSpec]`; its insertion order is the output order.

### `register(name, domain, field, requires, units, range=(-inf, inf), intensive=True, sentinel_cases=None, references=(), adopted=False, extension=None, doc=None, stability="robust")`

The decorator of spec §8. It refuses an unknown domain or extension, a
stability other than `"robust"` / `"fragile"`, and **refuses
`intensive=False`** with a message telling to normalize the quantity first;
a name registered twice is an error. The docstring (or the
`doc=` argument, used by the factories that register families of names in a
loop) supplies the formula. Tests:
`test_tier1.py::test_register_refuses_extensive_quantities`,
`test_stability.py::test_register_validates_stability`.

### Stability tags

`stability="fragile"` marks a descriptor that is computed exactly as
specified but is not numerically converged on typical VASP grids: its value
changes by more than about 10% between derivative schemes or when the grid
is coarsened to 80% (measured on the 6,059-structure dataset with the
scripts in `paper/analysis/`). Four are tagged: `ellip_bond_avg`,
`ellip_bond_std` (section 27) and `n_saddle1`, `n_saddle2` (section 28).
Fragile descriptors stay in the default `featurize` output, so every table
keeps the specified columns; `descriptor_names(..., include_fragile=False)`
gives the model-ready set without them (216 of the 220 defaults), and
`catalogue()` has a `stability` column. Tests:
`test_stability.py::test_fragile_set_is_the_measured_one` (the tagged set is
exactly those four), `test_fragile_descriptors_stay_in_the_default_output`
(and the order is kept when they are removed).

### `selected(opts)`, `finite_or(value, sentinel, case)`

`selected` lists the registered specs whose domain is requested and whose
extension (if any) is enabled, in registration order. `finite_or` returns the
value when finite, else `Sentinel(sentinel, case)`; it is the usual way an
operator's NaN becomes a flagged sentinel.

### Shared accessors

- `geometry(vd)` = `geometry_of(vd)`; `masks(vd)` = `shells_of(vd,
  options(vd).shells)`.
- `_FIELDS`: the field registry, `{"rho": rho, "abs_m": abs_m}` here;
  `bonding.py` adds `elf_file`, `elf_d`, `potential` and `delta_rho` with
  `register_field(name, fn)`. `field_values(vd, name)` evaluates one
  (unknown names raise with the list of known ones).
- `field_derivatives(vd, name)`: the cached `Derivatives` of a registered
  field with the backend and order of the options. For a derived field
  (anything but `rho`, `abs_m`, `elf_file`) the cache name also carries
  every option its values depend on (backend, order, Laplacian method,
  deformation reference, custom reference, ELF and potential sources), so
  featurizing one object twice with different options never reuses stale
  derivatives (`tests/test_regressions.py`; before this, `zeta_ELF` under
  `laplacian_method="diagonal"` could reuse the metric-Laplacian ELF).
- `laplacian(vd, name="rho")`: the Laplacian with the options' method.
- `is_uniform(values)`: True when max - min <= `UNIFORM_TOL` max abs(f), or
  the field is zero; the "uniform density" sentinel case.

### Metadata hooks

`METADATA_HOOKS` is a list of functions `vd -> dict`; the decorator
`metadata_hook` appends one. `featurize` evaluates every hook for every
structure, whatever domains were requested (each hook may check the options
itself). Hooks exist in `registry.py` (`_zval_metadata`), `bonding.py`,
`structural.py`, `magnetic.py` and `heterogeneity.py`.

### `augmentation_radii(vd)`

(Per-atom PAW augmentation radius R_PAW in Angstrom, source label) for the
`paw` extension. R_PAW is RCORE from the POTCAR / OUTCAR or a table; an
element without one falls back to its covalent radius. The label is
`"covalent"` when nothing is known, `"potcar"` or `"table"` when every
element is covered, with `"+covalent"` appended when only some are (a
`paw_radii` dict set directly on the object also reports `"potcar"`).
Inside R_PAW a CHGCAR is pseudized; outside it is the all-electron valence
density.

### `zval_source(vd)` and `_zval_metadata`

Where the valence counts of a pseudo-density came from: `"potcar"`,
`"outcar"` (from the file name recorded in `sources["zval"]`), `"table"`,
`"given"` (set on the object), `"default"` (`default_zval`), with
`"+default"` when only some elements are covered; `"not_used"` for an
all-electron density. The hook writes it as `zval_source` for every
structure. Tests: `test_io.py::test_paw_tables_for_runs_without_potcar`,
`test_tooling.py::test_batch_passes_read_options`.

## 26. `descriptors/__init__.py` — `featurize`, metadata, `catalogue`

Importing the package imports the five domain modules, which registers every
descriptor as a side effect, in the order bonding, structural, magnetic,
heterogeneity, compositional (and within a module, in definition order).

### `featurize(vd, domains=None, partition="nearest", shells=None, deformation_reference="auto", elf_source="auto", laplacian_method="metric", derivative_backend="fft", fd_order=FD_ORDER, potential_source="auto", custom_reference=None, extensions=(), float32=False, return_metadata=False)`

1. `make_options(...)` builds and validates the `FeatureOptions` (an
   unknown partition or domain raises here;
   `test_tier1.py::test_options_are_validated`).
2. `float32=True` works on `vd.astype(np.float32)` (fresh cache); otherwise
   on `vd` itself.
3. `v = base.with_options(opts)`: a view sharing grids and cache.
4. For every spec in `selected(opts)`: call it; a `Sentinel` contributes its
   value and records its case in `flags`; anything else is cast to float.
5. Without `return_metadata`, return the dict.
6. Metadata, in this order:
   - fixed keys: `n_atoms`, `volume`, `grid_shape` (`"48x48x48"`),
     `density_source`, `spin_mode`, `site_counts` (`"Fe:2,O:1"`),
     `partition` (with `(k=60,cells=8)` for Becke), `shells`
     (`"c1,c2"` plus `",scaled"`), `derivative_backend` (`"fft"` or e.g.
     `"fd4"`), `laplacian_method`, `precision`;
   - the hooks: `zval_source`; `elf_source`, `potential_source`,
     `deformation_reference`, `def_charge_mismatch` (bonding requested);
     `euler_consistency` (always); `paw_radii_source` (`paw` enabled);
     `magnetic`, `M_abs`, `M_net` (always); `n_elements`,
     `max_sites_per_element` (always);
   - `<name>__flag` (0 or 1) for every selected descriptor that has
     sentinel cases, so the flag columns have a fixed schema;
   - `sentinels` (`"name:case;..."`), `wall_time_s`, `pydemi_version`,
     `error` (empty; `featurize_batch` fills it on failure).

Because the census hook runs unconditionally, the extremum census is
computed for every structure even when the structural domain is not
requested; that is the price of emitting `euler_consistency` everywhere, as
spec §10 asks.

Tests: `test_tier1.py::test_featurize_names_and_metadata`,
`test_uniform_density_sentinels`.

### `descriptor_names(domains=None, extensions=(), include_fragile=True)`

The fixed column order: the names `featurize` returns for those domains and
extensions, without computing anything. `include_fragile=False` drops the
descriptors tagged `stability="fragile"`, keeping the order of the rest.

### `catalogue()`

A pandas DataFrame, one row per registered descriptor (all domains and both
extensions): `name`, `domain`, `field`, `requires`, `units`, `range_min`,
`range_max`, `intensive`, `sentinel_cases` (`"case=value;..."`), `adopted`,
`extension`, `stability`, `formula`, `references`. `tools/generate_docs.py`
renders it into the README tables and `docs/catalogue.csv`. Test:
`test_tier1.py::test_catalogue_is_generated_from_the_registry`.

## 27. `descriptors/bonding.py`

The bonding domain (spec §8.1): 36 descriptors by default, plus 8 in the `paw`
extension and 1 in the `robust` extension. Distances and directions are those
of the geometry pass; shells are core r <= c1, bond c1 < r <= c2, interstitial
r > c2. The module also resolves the derived fields (kinetic terms, ELF,
potential, promolecule) from the options, so the other domains import them
from here.

### Tier 1: `zeta`, `m1`, `m2`, `sigma_r2`, `f_core`, `f_bond`, `f_int`

```
zeta     = 1 - sum_k |grad rho_k . u_k| / sum_k |grad rho_k|
m1       = sum_k rho_k r_k / sum_k rho_k
m2       = sum_k rho_k r_k^2 / sum_k rho_k
sigma_r2 = m2 - m1^2
f_shell  = sum_{k in shell} rho_k / sum_k rho_k,   shell = core, bond, int
```

- `zeta` is 0 exactly for a single spherical atom and grows as density
  concentrates off the radial directions, e.g. in bonds. A uniform density
  (0/0) returns `Sentinel(0.0, "uniform_density")`, checked with
  `is_uniform` before the gradient is used, and again through `finite_or`.
- The moments use signed weights (`radial_moment(..., "signed")`); a zero
  total gives `zero_density` = 0.0. Single Slater 1s atom
  (rho ~ exp(-2 zeta r)): m1 = 3/(2 zeta), m2 = 3/zeta^2, sigma_r2 =
  3/(4 zeta^2).
- The three shell fractions are registered in a loop by `_shell(which)`,
  which generates the docstring (`doc=`) with the shell's inequality.

Tests (`test_tier1.py`): `test_slater_closed_forms` (both backends, 96^3:
integral 1 to 1e-4, m1 to 5e-4, m2 to 1.5e-3, sigma_r2 to 1e-2, zeta below
5e-5; the small m2 bias comes from nearest-image distances in the 4 Angstrom
box), `test_slater_moments_converge_with_the_grid`,
`test_two_atoms_give_nonzero_anisotropy`.

### Tier 1: `lnf`, `lnf_charge_weighted`, `lap_concentration`, `lap_concentration_valence`

```
lnf                       = (1/N) sum_k 1(lap rho_k < 0)
lnf_charge_weighted       = sum_{lap rho_k < 0} rho_k / sum_k rho_k
lap_concentration         = sum_{lap rho < 0} |lap rho_k| / sum_k |lap rho_k|
lap_concentration_valence = the same over r > c1 only
```

- `lnf` is the volume fraction of charge concentration. For a Slater 1s
  atom in a box it is the volume fraction with r < 1/zeta (lap rho = rho
  (4 zeta^2 - 4 zeta / r)). It counts voxels, so voxels at near-zero density
  weigh as much as any other; with the FFT backend the sign of lap rho there
  is round-off, which is why `lnf_charge_weighted` exists (the spec: "compute
  both, they are not interchangeable"). Uniform density: 0.0, flagged.
- `lap_concentration` is identically 1/2 on a periodic grid (section 22,
  the first documented correction). It is kept because the spec lists it;
  `lap_concentration_valence` restricts it to r > c1, where it is not fixed.
  Sentinels 0.5 (`uniform_density`, and `empty_region` when no voxel lies
  beyond c1).

Tests: `test_slater_lnf_is_the_sphere_volume_fraction` (FD within 5% of the
exact fraction; FFT biased above 1.1 times it by the cusp ringing),
`test_lap_concentration_is_identically_one_half`,
`test_uniform_density_sentinels`.

### Derived-field plumbing

- `kinetic(vd)`: the `KineticTerms` of rho from the cached gradient norm and
  Laplacian, cached under `("kinetic", backend, fd_order,
  laplacian_method)`.
- `elf_source(vd)`: resolves `"auto"` to `"file"` when an ELFCAR was read,
  else `"reconstruct"`; `"file"` without an ELFCAR raises `ValueError`
  (message names the ELFCAR).
- `_elf_file(vd)`, `_elf_reconstructed(vd)`: the two ELF fields, registered
  as `elf_file` and `elf_d`; the reconstruction is cached under `("field",
  "elf_d", backend, fd_order, laplacian_method)`. `elf_field_name(vd)` picks
  the registered name for the resolved source. Registering the ELF as a field
  gives it cached derivatives like any other field (for `zeta_ELF`).
- `potential_source(vd)`: `"auto"` -> `"locpot"` when a LOCPOT was read,
  else `"hartree"`; `"locpot"` without one raises.
- `potential(vd)`: the potential in eV with zero cell average, from the
  resolved source (`centred` LOCPOT, `hartree_potential`, or
  `electrostatic_potential` with `ion_charges`), cached under `("field",
  "potential", source)` and registered as the field `potential`.
- `site_potentials(vd)`: V(R_i) at each nucleus by `fourier_interpolate`.
- `_field_metadata` (hook): `elf_source` and `potential_source` when the
  bonding domain is requested.

Test: `test_fields.py::test_elf_source_selection` (auto picks the
reconstruction, then the file once one is attached, `elf_source=
"reconstruct"` overrides it, `"file"` without an ELFCAR raises).

### Ellipticity: `ellip_bond_avg`, `ellip_bond_std` (fragile) and `ellip_bond_bounded_avg` (`robust` extension)

```
ellip_bond_avg / _std   = mean / std of  lambda1/lambda2 - 1   over {k in bond, lambda2 < 0}
ellip_bond_bounded_avg  = mean of        1 - lambda2/lambda1   over the same voxels
```

with lambda1 <= lambda2 <= lambda3 the Hessian eigenvalues of rho
(`_eigenvalues` = the cached `hessian_eigenvalues`; `_ellipticity` selects the
voxels). Mean and population std; no such voxel -> `no_bond_voxels` = 0.0.

Wherever lambda2 -> 0- the ratio diverges and a few voxels dominate the mean.
The two specified descriptors are kept exactly as written but tagged
`stability="fragile"`: on the 6,059-structure dataset they change by a
median 22% (avg) and 67% (std) between FFT and FD4 derivatives and by 21%
and 62% on an 80% grid (48 structures; docstrings, README §6.6); they are
also the largest float32 discrepancy (2%, README §11).

`ellip_bond_bounded_avg` (range [0, 1), `extension="robust"`, off by
default) is the converged alternative. Per voxel 1 - lambda2/lambda1 =
e / (1 + e) with e = lambda1/lambda2 - 1: a monotone map of the ellipticity
onto [0, 1), 0 for cylindrical symmetry and tending to 1 as lambda2 -> 0-,
so it cannot diverge. It changes by a median 0.2% between FFT and FD4, 1.1%
with FD2 and 0.2% on an 80% grid (48 structures).

Tests (`test_stability.py`): `test_robust_extension_is_off_by_default`,
`test_bounded_ellipticity_matches_the_exact_hessian` (an anisotropic
Gaussian, 1e-3 against the exact Hessian),
`test_bounded_ellipticity_is_zero_for_a_spherical_atom` (both forms 0),
`test_bounded_ellipticity_is_a_monotone_map_of_the_ellipticity`.

### Local energy densities: `f_H_negative`, `H_bond_mean`, `G_over_rho`

```
g = C_F rho^(5/3) + (1/72)|grad rho|^2/rho + (1/6) lap rho,   v = (1/4) lap rho - 2 g,   H = g + v
f_H_negative = (1/N_bond) sum_{k in bond} 1(H_k < 0)
H_bond_mean  = <H_k> over the bond shell           (hartree/bohr^3)
G_over_rho   = <g_k / rho_k> over the bond shell   (hartree/electron)
```

All in atomic units, over bond-shell voxels above the density floor
(`_bond_valid`); none -> `empty_region` = 0.0. H < 0 marks shared-shell
(covalent) interaction in Cremer-Kraka terms. The gradient expansion was
derived for all-electron densities, so on PAW densities these are meaningful
from the bond shell out (README §12), which is why the averages are over the
bond shell. Test: `test_fields.py::test_uniform_electron_gas_limits`.

### ELF: `f_ELF_localized`, `ELF_bond_avg`, `ELF_core_valence_contrast`, `zeta_ELF`

```
f_ELF_localized           = (1/N_bond) sum_{k in bond} 1(ELF_k > 0.5)
ELF_bond_avg              = <ELF_k> over the bond shell
ELF_core_valence_contrast = <ELF>_core / <ELF>_bond
zeta_ELF                  = 1 - sum_k |grad ELF_k . u_k| / sum_k |grad ELF_k|
```

The ELF is the ELFCAR (resampled at read time) or ELF_D, per `elf_source`.
Sentinels: `empty_region` = 0.0 when the bond (or core) shell is empty,
`zero_denominator` = 0.0 when the bond average is 0, `uniform_density` =
0.0 for a uniform ELF. `zeta_ELF` differentiates the ELF field itself
through `field_derivatives(vd, elf_field_name(vd))`.

### Non-covalent interactions: `f_NCI`, `NCI_attractive`, `sign_lambda2_rho_mean`

```
s = |grad rho| / (2 (3 pi^2)^(1/3) rho^(4/3))         (a.u.)
NCI voxels: s < 0.5 and rho < 0.05 e/bohr^3
f_NCI                 = (1/N) sum_k 1(k is an NCI voxel)
NCI_attractive        = fraction of NCI voxels with lambda2 < 0
sign_lambda2_rho_mean = mean of sign(lambda2) rho_k over NCI voxels (rho in a.u.)
```

- `reduced_gradient(kt)`: s from the kinetic terms, +inf below the density
  floor so empty voxels never qualify.
- `_nci(vd)`: the NCI mask, or `Sentinel(0.0, "uniform_density")` (a
  uniform density has s = 0 everywhere but no non-covalent region).
- No NCI voxel: `no_nci_voxels` = 0.0 for the two conditional averages.

Tests: `test_nci_region_between_two_distant_atoms`,
`test_uniform_nci_sentinel`.

### Electrostatic potential: `V_spread`, `V_int_min`

```
V_spread  = std over sites i of V(R_i)
V_int_min = min over interstitial voxels of V_k       (cell average of V set to 0)
```

`V_spread` interpolates the potential at the nuclei exactly (the spec's
`V_site` array is an intermediate, not a descriptor). The source is recorded
in `potential_source`. `V_int_min` has `empty_region` = 0.0. Both are
independent of the potential's G = 0 convention (std; referenced to the cell
average).

### Bond midpoints: `rho_mid_mean`, `rho_mid_std`

```
rho_mid = rho at the midpoints of nearest-neighbour bonds (first shell, (1 + BOND_TOL) d_i)
```

`_rho_mid(vd)` takes the census (`bond_census_of(vd, BOND_TOL)`), the
midpoints in stored fractional coordinates, and `fourier_interpolate`,
cached under `("rho_mid", BOND_TOL)`. No bond -> `no_bonds` = 0.0. Test:
`test_fields.py::test_bond_midpoint_density` (two Gaussians 2 Angstrom
apart: rho(mid) = 2 N (a/pi)^(3/2) exp(-a d^2/4) to 1e-6, std 0).

### Deformation density: resolution and metadata

- `deformation_reference(vd)`: resolves `"auto"` to `"aeccar0"` when
  `core_density` is present, else `"tabulated"`; `"aeccar0"` without
  AECCAR0 raises.
- `promolecule_density(vd)`: the promolecule on the density grid, cached
  under `("promolecule", reference, custom_reference)`. For `"aeccar0"`:
  `core_density` + the tabulated **valence** promolecule with ZVAL
  electrons per atom (section 18). Otherwise the `"tabulated"` or
  `"custom"` tables, all electrons for an all-electron density and ZVAL
  electrons for a pseudo-density.
- `delta_rho(vd)` = rho - promolecule, registered as the field `delta_rho`.
- `_deformation_metadata` (hook, bonding requested):
  `deformation_reference` and `def_charge_mismatch` = int delta_rho dV.

### Deformation density: `m1_def`, `m2_def`, `sigma_r2_def`, `f_bond_def`, `f_int_def`, `f_bond_dep`, `def_polarity`

```
m1_def       = sum_k |drho_k| r_k / sum_k |drho_k|
m2_def       = sum_k |drho_k| r_k^2 / sum_k |drho_k|
sigma_r2_def = m2_def - m1_def^2
f_bond_def   = sum_{k in bond, drho > 0} drho_k / sum_{drho > 0} drho_k      (f_int_def: interstitial)
f_bond_dep   = sum_{k in bond, drho < 0} |drho_k| / sum_{drho < 0} |drho_k|
def_polarity = sum_k |drho_k| dV / Q_tot,   Q_tot = sum_k rho_k dV
```

`_drho_parts` returns (delta_rho, abs, distance); `_accumulated_share`
computes the accumulation shares. The moments are weighted by abs(delta_rho)
because delta_rho changes sign; the shell shares separate accumulation from
depletion because their sum over the cell is `def_charge_mismatch`, not 0.
Sentinels: `zero_deformation`, `no_accumulation`, `no_depletion`,
`zero_density`, each 0.0.

### Pair charge transfer: `bond_charge_transfer_pair_mean`, `bond_charge_transfer_pair_std`

```
q_ij = int_{region(i,j)} drho dV,   region(i, j) = voxels whose two nearest nuclei are i and j
```

`pair_charge_transfer(vd)` computes the second-order Voronoi regions
(`pair_regions`, cached under `("pair_regions",)`), encodes every voxel's
(i, j, t) and every census pair's (i, j, t_wrapped) as one integer (the
shift offset by 50 in base 101), sums delta_rho dV per region code with
`np.unique` and `bincount`, and looks up each census pair (a pair without a
region gets 0). Mean and population std over the census pairs; no bond ->
`no_bonds` = 0.0. The spec asks for "the region between each neighbour
pair"; the second-order Voronoi cell defines one that tiles the cell and
needs no bond-path search.

Tests (`test_deformation.py`): `test_pair_regions_tile_the_cell_and_match_the_census`,
`test_deformation_names_and_no_nan`, and the reference tests of section 18.

### The `paw` extension (bonding part): the `*_def_out` family and `def_out_volume_fraction`

Beyond R_PAW of its nucleus a CHGCAR voxel is not pseudized, so there the
CHGCAR and the free-atom valence reference agree in kind. On the
6,059-structure dataset a median 87% of the whole-cell int abs(delta_rho) lies
inside the augmentation spheres, which fill 45% of the volume (README §9), so
the whole-cell family measures mostly pseudization. The extension repeats it
on the outside voxels:

- `_drho_out(vd)`: mask r > R_PAW(nearest atom) + `GEOMETRY_EPS`
  (`augmentation_radii`), and delta_rho zeroed inside.
- `_out(name, fn_doc, units, rng, compute, cases)`: registers one extension
  descriptor that returns `empty_region` = 0.0 when no voxel is outside and
  otherwise applies `compute` to the masked delta_rho.
- The computations: `_moment(n)` (m1_def_out, m2_def_out), `_sigma`
  (sigma_r2_def_out), `_share(shell)` (f_bond_def_out, f_int_def_out; the
  bond and interstitial masks are the ordinary shells), `_dep`
  (f_bond_dep_out), `_polarity` (def_polarity_out, still divided by the
  whole-cell Q_tot).
- `def_out_volume_fraction`: the fraction of the cell outside every sphere.

## 28. `descriptors/structural.py`

The structural domain (spec §8.2): 21 descriptors by default plus 4 in the
`paw` extension. Intensivity is built in: every count is per unit volume, the
non-nuclear charge is a fraction of Q_tot, and the information measures are
normalized by the cell volume.

### Percolation: `rho_perc_a`, `rho_perc_b`, `rho_perc_c`, `perc_anisotropy`

```
rho_perc_alpha  = max {c : {rho > c} spans a_alpha under PBC}
perc_anisotropy = (max_alpha - min_alpha) / mean_alpha of rho_perc
```

`_percolation(vd)` caches `percolation_levels(rho)` under
`("percolation",)`; `_perc(axis, label)` registers the three levels. The
level is the density at the bottleneck of the best connecting path: high for
metals, near zero for ionic and molecular solids. The two corrections behind
it (highest level; winding test) are in section 23. `perc_anisotropy` has
`zero_levels` = 0.0.

### Extremum census: `n_max`, `n_min`, `n_saddle1`, `n_saddle2`, and `euler_consistency`

```
n_X = (number of X) / V_cell
```

`census(vd)` caches `extremum_census(rho)` under `("census",)`; `_count(which,
text, stability, note)` registers the four per-volume counts. Maxima and
minima are 26-neighbour, saddles from the Freudenthal link (section 23).
`n_saddle1` and `n_saddle2` are tagged `stability="fragile"` (`_SADDLE_NOTE`):
they change by a median 17-20% when the grid is coarsened to 80% (30
structures), because saddles of near-flat, rippled regions appear and vanish
with the grid. `_census_metadata` (hook, always) writes the raw
`euler_consistency`, the number of extrema the two neighbourhoods classify
differently (section 23). Raw counts are extensive (they double in a 2x2x2
supercell), hence the division by V (spec §10). Test:
`test_census_descriptors_are_per_volume` (a 4 Angstrom cell: n_max = 1/64,
n_saddle1 = 3/64, Euler 0).

### Non-nuclear maxima: `n_NNM`, `Q_NNM`

```
n_NNM = (number of local maxima with min_i |r - R_i| > r_cut) / V_cell
Q_NNM = (charge in the steepest-ascent basins of those maxima) / Q_tot
```

`_nnm(vd, r_cut)` takes a per-atom cutoff array (so the PAW variant can
reuse it): the 26-neighbour maxima farther than the cutoff of their nearest
atom (plus `GEOMETRY_EPS`), and the rho summed over the voxels whose basin
(`ascent_basins`, cached under `("basins",)`) ends at one of them, divided by
the total. The default cutoff is `options(vd).nnm_r_cut` (0.8 Angstrom).
Test: `test_a_non_nuclear_maximum_is_counted_with_its_basin_charge` (a
planted "ghost" Gaussian of 0.5 e between two 1 e atoms: n_NNM = 1/216,
Q_NNM between 0.15 and 0.25).

### Density floor: `rho_min`, `rho_min_ratio`, `rho_int_mean`

```
rho_min       = min_k rho_k
rho_min_ratio = rho_min / <rho>_V
rho_int_mean  = <rho> over the interstitial shell (r > c2)
```

For a PAW pseudo-density `rho_min` is usually negative and near a nucleus:
on the 6,059-structure dataset the density is negative somewhere in 61% of
the structures (README §9), hence `rho_min_int` in the extension.
Sentinels: `zero_density` for the ratio, `empty_region` for the interstitial
mean.

### Charge-anisotropy tensor: `T_eigenvalues_t1`, `T_eigenvalues_t2`, `T_eigenvalues_t3`, `charge_FA`

```
T_ab = sum_k d_a rho_k d_b rho_k / sum_k |grad rho_k|^2      (trace 1)
t1 <= t2 <= t3 its eigenvalues,   charge_FA = sqrt(3/2) ||T - I/3||_F / ||T||_F
```

`_tensor(vd)` returns (sorted eigenvalues, FA) or the uniform sentinel;
`_t(k)` registers the three eigenvalues with sentinel 1/3 (an isotropic
tensor), `charge_FA` has 0.0. The tensor itself rotates with the cell; its
sorted eigenvalues are invariant, which is why only they are registered
(spec §10). Tests: `test_anisotropy_tensor_of_a_layered_density` (rho varying
along a3 only: t3 = 1, t1 = 0, FA = 1), `test_uniform_limits_and_sentinels`.

### Information measures: `shannon_entropy`, `fisher_information`, `disequilibrium`, `LMC_complexity`

With the shape function rho~ = rho / N_e, in atomic units (rho clipped at 0,
volumes in bohr^3):

```
shannon_entropy    = -int rho~ ln rho~ dV - ln V
fisher_information = int |grad rho~|^2 / rho~ dV            (voxels above the density floor; 1/bohr^2)
disequilibrium     = V int rho~^2 dV
LMC_complexity     = D e^S                                  (D, S the unnormalized ones)
```

`_shape_function(vd)` computes all four once, cached under `("information",
backend, fd_order)` (the Fisher term needs the gradient).

**Why volume-normalized** (an extra decision; README §8). The spec defines
S = -int rho~ ln rho~ and D = int rho~^2. For a 2x2x2 supercell rho~ is
spread over 8 times the volume, so S shifts by ln 8 and D divides by 8: they
are not intensive, which spec §10 forbids. S - ln V is minus the
Kullback-Leibler divergence of rho~ from the uniform distribution: intensive,
independent of units, 0 for a uniform density and negative otherwise (range
(-inf, 0]). V D is 1 for a uniform density and at least 1 otherwise. The
Fisher information is already intensive. LMC = D e^S is unchanged by the
normalization (V D e^{S - ln V} = D e^S) and is 1 for a uniform density.
Test: `test_uniform_limits_and_sentinels` (S = 0, D = 1, LMC = 1 on a
uniform density, no NaN).

### The `paw` extension (structural part): `rho_min_int`, `rho_min_int_ratio`, `n_NNM_paw`, `Q_NNM_paw`

Pseudized atoms can have no maximum at the nucleus, only lobes on a shell
inside their augmentation sphere (CaSi3Pt: 8 lobes 0.81-0.83 Angstrom from
Si holding 6.3 e, just beyond a 0.8 Angstrom cutoff; README §9), and the
negative pseudo-density sits inside the spheres too. These variants exclude
the spheres:

```
rho_min_int       = min of rho over voxels with r > max(c2, R_PAW) of their nearest nucleus
rho_min_int_ratio = rho_min_int / <rho>_V
n_NNM_paw         = (number of local maxima with r > max(r_cut, R_PAW,i)) / V_cell
Q_NNM_paw         = (charge in the basins of those maxima) / Q_tot
```

`_outside_spheres(vd, inner)` builds the mask from per-atom arrays: `inner`
is `shells.atom_cutoffs(structure)[1]`, so with radius-scaled shells c2 is
converted to a length per atom before max(c2_i, R_PAW,i) is taken
(`tests/test_regressions.py::test_rho_min_int_with_radius_scaled_shells`); `_paw_cut(vd)` the per-atom
cutoff max(r_cut, R_PAW,i) for `_nnm`. `_paw_metadata` (hook, extension
enabled) records `paw_radii_source`.

Tests (`test_tooling.py`): `test_paw_extension_is_off_by_default_and_uses_the_paw_radii`,
`test_rho_min_int_ignores_negative_pseudo_density_inside_the_spheres`
(rho_min < 0 < rho_min_int), `test_n_nnm_paw_ignores_lobes_inside_the_augmentation_sphere`
(six lobes at 0.9 Angstrom: n_NNM > 0, n_NNM_paw = 0 with R_PAW = 1.1).

## 29. `descriptors/magnetic.py`

The magnetic domain (spec §8.3), from m = rho_up - rho_down (e/Angstrom^3;
integrated values are in mu_B). 8 descriptors.

### Helpers

- `magnetization_components(vd)`: [m] (collinear), [m_x, m_y, m_z]
  (non-collinear) or [] (no spin).
- `total_moments(vd)`: (M_abs, M_net) = (sum abs(m) dV, abs(sum m dV)); the
  vector norm of the component sums for a non-collinear run. Extensive, so
  metadata only.
- `is_magnetic(vd)`: M_abs > `MAGNETIC_TOL` x n_atoms (0.01 mu_B per atom).
  Below that the ratios would be ratios of numerical noise; the threshold is
  per atom so that it is itself intensive.
- `site_moments(vd)`: mu_i = sum_k w_i(k) m_k dV under the requested
  partition, cached under `("site_moments", partition)`: shape (n_atoms,)
  collinear, (n_atoms, 3) non-collinear, zeros without spin.
- `site_moment_magnitudes(vd)`: signed mu_i (collinear) or abs(mu_i)
  (non-collinear), for the heterogeneity statistics.
- `_magnetic_metadata` (hook, always): `magnetic`, `M_abs`, `M_net`.

### Registration pattern: `_magnetic` and `_reg`

Every entry must return 0.0 for a non-magnetic structure, never NaN.
`_magnetic(fn)` wraps a function so it returns `Sentinel(0.0,
"non_magnetic")` unless `is_magnetic`; `_reg(name, units, requires, rng)`
registers the wrapped function with the original docstring and
`sentinel_cases = {"non_magnetic": 0.0}`, and returns the unwrapped one. So
the module-level names (`magnetic.m1_spin`, ...) are the raw functions; the
registry holds the guarded ones.

### The descriptors

```
M_abs_per_atom          = sum_k |m_k| dV / n_atoms
M_net_per_atom          = |sum_k m_k dV| / n_atoms
m1_spin                 = sum_k |m_k| r_k / sum_k |m_k|
sigma_r2_spin           = sum_k |m_k| r_k^2 / sum_k |m_k| - m1_spin^2
f_bond_spin             = sum_{k in bond} |m_k| / sum_k |m_k|
mu_site_std             = std over i of mu_i          (non-collinear: sqrt(mean_i |mu_i - mean mu|^2))
spin_frustration        = 1 - |sum_i mu_i| / sum_i |mu_i|
spin_charge_correlation = Pearson r(rho_k, |m_k|) over voxels
```

- `M_abs` and `M_net` themselves are extensive; the spec says to report them
  "also per atom, which is the intensive version", so only the per-atom
  forms are registered.
- `spin_frustration` is 0 for a ferromagnet and 1 for a perfectly compensated
  antiferromagnet; for a non-collinear run it uses vector norms (the spec's
  vector generalization). `spin_charge_correlation` returns the
  `non_magnetic` sentinel when either field has zero variance.
- The site moments tile space: sum_i mu_i = M_net exactly for every hard or
  smooth partition, unlike VASP's RWIGS-sphere moments (spec §8.3 asks for
  this assertion).

Tests (`test_magnetic.py`): `test_site_moments_sum_to_the_net_moment`
(nearest and power, 1e-12), `test_ferro_and_antiferro` (frustration 0 and 1,
M_net 0 for AFM, equal M_abs, the name order),
`test_noncollinear_vector_generalization` (two equal moments at 90 degrees:
frustration 1 - sqrt(2)/2), `test_non_magnetic_is_zero_never_nan`
(unpolarized, and moments below `MAGNETIC_TOL`).

## 30. `descriptors/heterogeneity.py`

The heterogeneity domain (spec §8.4): a meta-operator over per-site
descriptors, 23 names.

### Per-site values

- `density_site_sums(vd)`: `site_sums` of rho (with its gradient and the
  per-atom shell cutoffs) under the requested partition, cached under
  `("site_sums", "rho", partition, shells key, backend, fd_order)`.
- `_per_site_m1`, `_per_site_f_bond`, `_per_site_zeta` (uniform density ->
  `Sentinel(0.0, "uniform_density")`), `_per_site_mu` (not magnetic ->
  `Sentinel(0.0, "non_magnetic")`; otherwise `site_moment_magnitudes`).
- `_stats(vd, per_site, base)`: `site_statistics` over the species, or the
  per-site sentinel passed through, cached under `("site_stats", base,
  partition, shells key, backend, fd_order, laplacian_method)`.

### `heterogeneity(base, per_site, units, formula, requires, skip=())`

The higher-order function the spec asks for ("a decorator or higher-order
function, not copy-paste"). For a base X it registers

```
X_site_std              std over sites of X^(i)
X_site_range            max_i X^(i) - min_i X^(i)
X_site_max, X_site_min
X_within_element_var    sum_e w_e Var_{i in e}(X^(i))
X_between_element_var   sum_e w_e (Xbar_e - Xbar)^2
```

with the squared units for the variances, `requires + ["partition"]`, and a
generated docstring holding the per-site formula. The closure `fn` returns,
in order: the per-site sentinel if any; `Sentinel(nan,
"one_site_per_element")` for the within variance when no element has two
sites (the one NaN the spec allows, "if accompanied by the site count");
`Sentinel(0.0, "single_element")` for the between variance of a
one-element structure; otherwise the statistic. Every registered name lists
the cases `uniform_density` and `non_magnetic` (0.0) plus its own, so the
catalogue shows the same cases for all four bases, although `uniform_density`
can only occur for zeta and `non_magnetic` only for mu.

Applied to (with the per-site formula in the registry):

| Base | X^(i) | Names |
|---|---|---|
| `m1` | sum_k w_i(k) rho_k r_ik / sum_k w_i(k) rho_k | 6 |
| `f_bond` | sum_{c1 < r_ik <= c2} w_i(k) rho_k / sum_k w_i(k) rho_k | 6 |
| `zeta` | 1 - sum_k w_i(k) abs(grad rho_k . u_ik) / sum_k w_i(k) abs(grad rho_k) | 6 |
| `mu` | mu_i = sum_k w_i(k) m_k dV (signed; abs(mu_i) non-collinear) | 5: `mu_site_std` is skipped because it belongs to the magnetic domain (spec §8.3) |

Every site-restricted sum honours the `partition` option through the pair
interface (section 14).

### `_site_metadata` (hook, always)

`n_elements` and `max_sites_per_element`; with `site_counts` from
`featurize` this is the per-element count the spec wants attached, since the
statistics are noisy for small cells.

Tests (`test_sites.py`): `test_heterogeneity_names_and_values` (names,
max >= min, range = max - min, std^2 = within + between),
`test_heterogeneity_sentinels` (one site per element -> NaN and flag with
`site_counts = "Fe:1,Ni:1,O:1,S:1"`; one element -> between 0 and flag;
non-magnetic -> every mu statistic 0 and flagged),
`test_power_partition_is_a_different_tiling`;
`test_partitions.py::test_site_descriptors_follow_the_partition`.

## 31. `descriptors/compositional.py`

The compositional baseline (spec §8.5): a thin wrapper over
`matminer.featurizers.composition.ElementProperty.from_preset("magpie")`,
132 features. **Nothing is reimplemented** (the spec's instruction).

- `_labels()`: the labels from `data/magpie_labels.txt`, so the names are
  known (and `catalogue()`, `descriptor_names()` work) without matminer
  installed. `LABELS` is that list.
- `_name(label)`: `"MagpieData mean Electronegativity"` ->
  `magpie_mean_Electronegativity`.
- `magpie_features(vd)`: imports matminer and pymatgen lazily (an
  `ImportError` says to install `pydemi[full]`), checks that the installed
  matminer's labels equal the shipped list (`RuntimeError` otherwise: the
  column meaning must not drift), featurizes `Composition(site_counts)` and
  caches the dict under `("magpie",)`.
- `_register(label)`: one descriptor per label, `domain="compositional"`,
  `adopted=True` (so they can be excluded from novelty claims and used as
  an explicit baseline), units `"(Magpie)"`. A value matminer returns as NaN
  (element data missing from the Magpie tables) is reported as NaN with the
  flag `missing_element_data`.
- `magpie_names()`: the 132 names.

Test: `test_tooling.py::test_compositional_is_matminer_magpie` (all 132
values equal matminer's for the same composition; `adopted` exactly for the
compositional rows).

## 32. `pydemi/__init__.py` — the public API

`__version__ = "0.1.0.dev0"`, `__author__`, and the exports: `Grid`,
`Lattice`, `Structure`, `VolumetricData`, `read`, `read_vasp`,
`read_all_electron`, `featurize`, `featurize_batch`, `catalogue`,
`descriptor_names` (with `include_fragile=`), `hirshfeld_charges`,
`sensitivity_sweep`. The rest is
reached by module path (`pydemi.validate.convergence.grid_convergence`,
`pydemi.validate.invariance.supercell`, `pydemi.validate.elf_fidelity`,
`pydemi.core.derivatives.laplacian`, ...).

---

# Part VII — Validation and tooling

## 33. `validate/__init__.py` — `elf_fidelity`

`elf_fidelity(elf_true, elf_reconstructed)` returns `{"pearson_r", "mae",
"rmse"}` over voxels: how well ELF_D reproduces a real ELFCAR (spec §6.2).
It is a validation metric, not a descriptor. The two arrays must have the
same shape (resample first with `core.grid.linear_resample`); Pearson r is
NaN when either has zero variance. Test: `test_fields.py::test_elf_fidelity`.

## 34. `validate/analytic.py`

Analytic reference densities with closed-form descriptor values (spec §11),
the backbone of the verification tests.

### `class _RadialSuperposition`

A sum over atoms and all periodic images of a spherical profile f(r).
`_RadialSuperposition(structure, params, electrons=1.0, tol=1e-14)`:
`params` (the exponent) and `electrons` may be a scalar, one value per atom,
or an element -> value mapping (`_per_atom`).

- `profile(r, p)`: f, f', f'' per unit charge (subclasses).
- `cutoff(p)`: the radius where the profile falls below `tol`.
- `_images(center, cutoff)`: the lattice shifts of the images that can reach
  the cell (within cutoff + the cell's circumradius of its centre).
- `evaluate(points)`: rho (N,), gradient (N, 3) and Hessian (N, 3, 3) at
  Cartesian points, **exactly**: grad = f' u, H = f'' u u^T + (f'/r)(I - u
  u^T). The exact derivatives are what the derivative tests compare against.
- `on_grid(shape)`: the three on a grid.
- `volumetric(shape, magnetization=None)`: a `VolumetricData` with this
  density and optionally a second superposition as a collinear
  magnetization.

### `SlaterSuperposition` and `GaussianSuperposition`

- Slater 1s, `params` = zeta (1/Angstrom): rho_i = N_i (zeta^3/pi)
  exp(-2 zeta r). It has a nuclear cusp like a real density, so it tests the
  cusp behaviour of the backends and gives the closed forms of spec §11:
  int rho = 1, m1 = 3/(2 zeta), m2 = 3/zeta^2, sigma_r2 = 3/(4 zeta^2),
  zeta = 0 exactly, lap rho = rho (4 zeta^2 - 4 zeta/r), lnf = volume
  fraction with r < 1/zeta.
- Gaussian, `params` = alpha (1/Angstrom^2): rho_i = N_i (alpha/pi)^(3/2)
  exp(-alpha r^2). Smooth everywhere and effectively band-limited, so it
  measures derivative accuracy and convergence order.

### Other builders and closed forms

- `uniform(structure, shape, value=0.1)`: a constant density; it exercises
  every sentinel case (spec §10, §11).
- `cubic_cell(a, species=("H",), frac=((0.5, 0.5, 0.5),))`.
- `slater_moment(n, zeta)` = (n+2)! / (2 (2 zeta)^n);
  `slater_fraction_within(R, zeta)`; `gaussian_moment(n, alpha)`;
  `gaussian_fraction_within(R, alpha)`; `slater_lnf(zeta, volume)` =
  (4/3) pi / zeta^3 / V.

## 35. `validate/invariance.py`

The transformations of spec §10, under which every registered descriptor
must be unchanged.

- `_map_grids(vd, fn, lattice, structure)`: applies `fn` to every grid
  (including the vector magnetization) and returns a new `VolumetricData`
  with an empty cache.
- `supercell(vd, reps=(2, 2, 2))`: the lattice scaled row by row, atoms
  replicated (shift-major order), every grid tiled with `np.tile`. Exact:
  the supercell's grid is the same sampling of the same density.
- `translate(vd, voxels)`: structure and grid translated together by whole
  voxels (fractional shift voxels / shape, grid `np.roll`ed), so the
  sampling stays exact.
- `rotation(axis, angle_deg)`: a Rodrigues rotation matrix.
- `rotate(vd, R)`: lattice rows a_i -> R a_i; fractional coordinates and
  grid values unchanged, the Cartesian frame rotated.
- `compare(a, b, rtol=1e-6, atol=1e-12)`: `{name: (a, b)}` for every entry
  with abs(a - b) > atol + rtol max(abs(a), abs(b)); two NaNs count as equal.

These transformations are what exposed the need for the geometric tie rule
and the `GEOMETRY_EPS` shell tolerance (section 13), the winding test for
percolation (section 23), and the volume normalization of the information
measures (section 28).

## 36. `validate/convergence.py`

- `_resampled(vd, shape)`: every grid Fourier-resampled to `shape` (the ELF
  clipped to [0, 1]), fresh cache.
- `grid_convergence(vd, scales=(0.9, 0.8), **featurize_kwargs)`: every
  descriptor on the full grid and on grids with `scale` times the points per
  axis (at least 4): columns `descriptor`, `full`, `x0.9`,
  `rel_change_x0.9`, ... So grid adequacy can be judged per descriptor.
- `analytic_convergence(spacings=(0.12, 0.09, 0.07, 0.05), zeta=3.0,
  box=4.0, backend="fd")`: the Slater 1s closed forms against grid spacing:
  relative errors of the integral, m1, m2 and lnf, and the absolute zeta.
  FD by default because the FFT Laplacian rings on the cusp.
- `recommended_spacing(table, tol=0.01)`: the coarsest spacing at which
  every relative error is below `tol` (NaN if none). Spec §11 wants the
  recommended mesh to "come out of the test suite rather than a guess";
  the test reads 0.08 Angstrom at 2%.
- `sensitivity_sweep(vd, c1_range=None, c2_range=None, scaled=False,
  **featurize_kwargs)`: every descriptor whose registry entry requires
  `"shells"`, over the grid of (c1, c2) (pairs with c1 >= c2 skipped),
  as a DataFrame (spec §5, for the cutoff sensitivity curves). Each point is
  a full `featurize` call on the same object, so everything that does not
  depend on the shells is reused from the cache.

Tests (`test_tooling.py`): `test_sensitivity_sweep` (skipped pairs, f_core +
f_bond + f_int = 1), `test_grid_convergence_report` (m1 changes by less than
1e-3 at 0.8), `test_recommended_mesh_comes_from_the_analytic_tests`
(integral error decreasing; `recommended_spacing(tol=0.02)` = 0.08; m1 and
m2 reach a ~5e-4 floor set by the nearest-image distance in the 4 Angstrom
box).

## 37. `batch.py`

Featurizing many structures (spec §12, §13): one process per structure via
`ProcessPoolExecutor`, never per descriptor.

- `COMPANIONS = {"elf": "ELFCAR", "locpot": "LOCPOT", "aeccar0": "AECCAR0",
  "aeccar2": "AECCAR2"}`.
- `_read(path, companions, read_options=None)`: a directory means its
  CHGCAR. For VASP files, `read_options` (e.g. `{"zval": ..., "paw_radii":
  ...}`) go to `read_vasp`; with `companions`, the ELFCAR, LOCPOT and AECCAR
  pair found next to the CHGCAR are passed too (a lone AECCAR is dropped).
  Cube and XSF files ignore both.
- `_one(path, companions, options, read_options=None)`: runs in the worker;
  returns `{"path", **features, **metadata}` or, on any exception, a row
  with `error = "Type: message"`, `wall_time_s`, `pydemi_version` and a
  3-frame `traceback`.
- `featurize_batch(paths, n_workers=8, on_error="record", progress=True,
  companions=False, read_options=None, **featurize_kwargs)`: rows are kept
  in input order whatever the completion order. `on_error`: `"record"`
  (the error in the `error` column, descriptors NaN), `"raise"`
  (`RuntimeError` with the traceback at the first failure), `"skip"` (drop
  the row). Progress lines go to stderr. The schema is fixed: `path`, then
  `descriptor_names(domains, extensions)` in registry order (a column
  missing because every row failed is filled with NaN), then every metadata
  column; `error` is always present and empty for good rows, and the
  traceback column is dropped. `n_workers=1` runs in-process (no pool).
  `companions=False` is the default so that every row of a dataset is
  computed from the same kind of input (otherwise the 86 runs with an ELFCAR
  would silently use the file and the rest the reconstruction).
- `find_runs(root, glob="*/CHGCAR")`: sorted matching paths.

The 6,059-structure rerun (PROGRESS.md §3): 0 errors, 105 min on 24
workers, 36.6 CPU-h, median 15 s and maximum 353 s per structure.

Tests (`test_tooling.py`): `test_batch_records_errors_and_keeps_a_fixed_schema`
(1 and 2 workers; a broken file gives an error row with NaN descriptors),
`test_batch_skip_and_raise`, `test_batch_passes_read_options`.

## 38. `cli.py`

The `pydemi` console script (spec §12), with `argparse` subcommands:

```
pydemi featurize CHGCAR --out features.json          (.json, or .csv)
pydemi batch ./runs --glob "*/CHGCAR" --out features.csv --workers 8 --domains bonding,magnetic
pydemi catalogue --out catalogue.csv
pydemi sweep CHGCAR --param c2 --range 1.0:2.5:0.05 --out sweep.csv
```

- `_options(p)`: the shared flags: `--domains`, `--extensions`,
  `--partition`, `--shells c1,c2`, `--deformation-reference`,
  `--custom-reference`, `--elf-source`, `--potential-source`,
  `--laplacian-method`, `--derivative-backend`, `--fd-order`, `--float32`,
  `--paw-table`. `featurize` and `sweep` also take `--elf`, `--locpot`,
  `--aeccar0`, `--aeccar2`; `batch` takes `--glob`, `--workers`,
  `--on-error`, `--companions`, `--quiet`.
- `_kwargs(a)`: the flags as `featurize` keywords (`--shells` parsed to a
  tuple; there is no flag for the radius-scaled mode).
- `_read_options(a)`: `--paw-table FILE.json`, `{"Sb": {"zval": 5,
  "rcore_bohr": 2.3}, ...}` (either key optional), to `read_vasp`'s `zval`
  and `paw_radii` (RCORE converted from bohr); the run's own POTCAR / OUTCAR
  still wins. This is the format written by
  `tools/paw_table_from_outcars.py`.
- `_read_one(a)`: reads the path with the companion files and, for VASP
  files, the PAW table.
- `_json_safe(x)`: NaN and inf -> `null`, numpy scalars -> Python.
- `main(argv=None)`: `featurize` writes JSON `{"path", "features",
  "metadata"}` or a one-row CSV; `batch` writes the DataFrame and prints the
  error count (exit code 1 when nothing matches the glob); `catalogue`
  writes the CSV; `sweep` builds the inclusive range `start:stop:step`
  (rounded to 10 digits) and runs `sensitivity_sweep` on one of c1 / c2 with
  the other at its default.

Test: `test_tooling.py::test_cli_commands` (featurize to JSON, catalogue,
batch with a broken run, `--paw-table` giving `zval_source = "table"` and
`paw_radii_source = "table+covalent"`, sweep over three c2 values).

## 39. `tools/`

Development-time scripts; none is imported by the package.

### `tools/atomic_solver.py`

The spherical, non-spin-polarized, non-relativistic LDA free-atom solver
behind `data/free_atoms.npz` (Hartree atomic units throughout). (Its module
docstring still carries the module path and entry numbers of the earlier
package layout.)

- **Radial equation on a log grid.** With r = e^x and u(r) = r P(r) =
  e^(x/2) phi(x), -u''/2 + [V + l(l+1)/(2r^2)] u = eps u becomes
  -phi''/2 + [(l + 1/2)^2/2 + r^2 V] phi = eps r^2 phi, a generalized
  symmetric eigenproblem with a tridiagonal left side and diagonal metric
  r^2; scaling by r^-1 makes it an ordinary symmetric tridiagonal problem.
- `RadialGrid` (frozen): x from ln(1e-10) to ln(80) bohr, 8,000 points. The
  inner boundary must sit far inside because s orbitals behave as
  phi ~ r^(1/2): r_min = 1e-7 bohr already costs about 0.4 mHa on Ne
  (docstring).
- `AtomResult`: Z, r, orbitals [(n, l, occupation, eigenvalue, density)],
  total density, energy, iterations, converged, extra;
  `orbital_density(selection)`.
- `_vwn_c(rs)` (VWN5 correlation), `_lda_xc(n, correlation="pw92")`
  (Slater exchange plus PW92 by default, or VWN5).
- `_cumtrapz`, `_hartree(n, r, x)` (V_H = 4 pi [(1/r) int_0^r n r'^2 dr' +
  int_r^inf n r' dr']), `_thomas_fermi_guess(Z, r)` (a rational fit to the
  Thomas-Fermi screening function as the starting potential).
- `_solve_l(V, l, grid, n_states, Z=1.0)`: `eigh_tridiagonal` with the
  `stebz` driver and an absolute tolerance 1e-13 max(1, Z^2). The matrix is
  strongly graded (diagonal ~1e25 at r = 1e-10 bohr), so the default relative
  tolerance would swamp the eigenvalues; bisection with Sturm counts and an
  absolute tolerance is accurate on graded tridiagonals.
- `solve_atom(Z, configuration, grid=RadialGrid(), mixing=0.4, tol=1e-9,
  max_iter=400, correlation="pw92")`: linear potential mixing until the
  density-weighted potential residual is below `tol`; orbitals sorted by
  eigenvalue. Validated against the NIST LDA atomic reference data
  (Kotochigova et al., Phys. Rev. A 55, 191 (1997)) with VWN5; discretization
  error of eigenvalues about 1e-5 relative (docstring). Limitations:
  non-relativistic (5d/6s/6p valence shapes miss the relativistic
  contraction), spin-restricted, LDA; adequate for a promolecule reference.

### `tools/generate_free_atom_tables.py`

Solves Z = 1..96 in parallel (`--workers`, default 16) with the
configurations of `elements.csv`, stops on an unconverged SCF, interpolates
each orbital's one-electron density onto `geomspace(1e-5, 30, 700)`
Angstrom (linear in log r, zero beyond the solver grid), stores float32
densities and writes `free_atoms.npz` with `_meta`.

### `tools/generate_element_data.py`

Writes `elements.csv` for Z = 1..103: covalent radii from
`tools/magpie_elements_source.csv` (Magpie tables copied from matminer
0.10.1, pm -> Angstrom, empty where NaN) and configurations from pymatgen
(`Element.full_electronic_structure`). Needs pymatgen; pydemi itself then
reads the CSV without it.

### `tools/generate_docs.py`

Regenerates `docs/catalogue.csv` and the README tables between the markers
`<!-- catalogue:<domain>:start -->` / `:end -->` for bonding, structural,
magnetic, heterogeneity, the two `paw` tables and the `robust` table (every
table has a Stability column, **fragile** in bold), and the count sentence
between `<!-- counts -->` and `<!-- /counts -->` (defaults per domain, how
many of them are fragile, and the sizes of the two extensions). The
compositional features are not tabulated (they are matminer's). Run it after
adding or changing a descriptor.

### `tools/paw_table_from_outcars.py`

`python tools/paw_table_from_outcars.py ROOT OUT.json [--glob "*/OUTCAR"]
[--workers 16]`: reads TITEL, ZVAL (`read_potcar_zval`) and RCORE
(`read_potcar_rcore`) from every OUTCAR under a dataset root and writes
`{element: {"potcar", "zval", "rcore_bohr", "n_runs"}}` with the most common
values. It refuses (exit 1) when an element appears with two ZVALs, i.e.
the runs mix POTCARs. `results/prompt_spec/dataset_paw_table.json` was made
with it from the dataset's 4,901 OUTCARs (one POTCAR per element
throughout).

## 40. Packaging (`pyproject.toml`)

Hatchling build of `src/pydemi`, version 0.1.0.dev0, `license = "MIT"`,
Python >= 3.10. Dependencies numpy >= 1.24, scipy >= 1.10, pandas >= 1.5;
extra `full` = pymatgen, spglib, matminer (the compositional domain and
pymatgen structures); extra `dev` = pytest, mypy, pandas-stubs, scipy-stubs.
Console script `pydemi = "pydemi.cli:main"`. pytest collects `tests/` only
(`legacy`, `paper`, `tools` excluded). `mypy --strict` covers `io`, `core`,
`fields`, `operators`, `constants.py` and `data`, the modules spec §15 calls
the core; the descriptor modules are outside it.

---

# Part VIII — Tests

## 41. The test suite, file by file

873 tests after parametrization (`pytest --collect-only`). Each file follows
a milestone of spec §14; `test_stability.py` covers the stability decision
taken after the milestones.

| File | Tests | Milestone and what it pins down |
|---|---|---|
| `test_io.py` | 48 | 1. Data model (lattice conventions, field validation, pymatgen input); CHGCAR round trip, volume division, Fortran order, spin block after augmentation lines, four blocks as a vector, three blocks refused, VASP 4 header with negative scale / Cartesian / selective dynamics, POTCAR-style labels; ELFCAR not scaled; AECCAR sum; `read_vasp` assembling a run; `default_zval` for 30 elements; ZVAL / RCORE tables; cube units against a hand-written file; cube and XSF round trips; sniffing and `read` |
| `test_derivatives.py` | 16 | 2. FFT exact on band-limited data (both cells); FD at its nominal order (2, 4, 6); the diagonal Laplacian wrong on triclinic cells; Hessian symmetric, trace = Laplacian, chunked eigenvalues exact; FD converging away from a Slater cusp; FFT ringing on the cusp; `g_squared`; argument checks |
| `test_geometry.py` | 7 | 3. Nearest atom and power diagram against brute force in a sheared cell; the cell where a fixed 3x3x3 supercell fails; zero direction on a nucleus; shells (absolute and scaled); the geometry cached once and shared by `with_options` views; image blocks against brute force |
| `test_tier1.py` | 12 | 4. Slater closed forms (both backends), moment convergence, lnf = sphere volume fraction (FD) and its FFT bias, two atoms give zeta > 0, `lap_concentration` = 1/2 (both backends), uniform sentinels and flags, names and metadata, catalogue columns, extensive quantities refused, option validation |
| `test_invariance.py` | 700 | 5. Every registered descriptor including the `paw` and `robust` extensions (233) on a low-symmetry Fe2O cell, equal on its 2x2x2 supercell, a translation by (3, 7, 11) voxels and a rotation by 37 degrees, to 1e-6 relative (699 parametrized tests), plus a check that the transforms are what they claim. Parametrized over the registry, so new descriptors are checked automatically |
| `test_sites.py` | 6 | 6. ANOVA identity (random and computed values), the nearest partition tiling space, heterogeneity names and values, sentinels, the power partition as a different tiling |
| `test_magnetic.py` | 6 | 7. sum_i mu_i = M_net (nearest, power), ferro vs antiferro, the non-collinear vector form, non-magnetic -> 0 and flagged (unpolarized and below `MAGNETIC_TOL`) |
| `test_fields.py` | 11 | 8. ELF_D in [0, 1] (analytic and real data), uniform-gas limits, ELF source selection, `elf_fidelity`, the Hartree potential of a Gaussian, the neutral cell's ESP, potential sources, the NCI region, the NCI uniform sentinel, the bond-midpoint density |
| `test_deformation.py` | 16 | 9. Free-atom tables' electron counts (8 elements), radial-table cutoff and lookup, promolecule point sampling, custom references (zero and half charge), the `aeccar0` correction, AECCAR0 required, pair regions vs the census, no NaN |
| `test_structural.py` | 11 | 10. Census (1, 1, 3, 3; Euler 0) at three resolutions, counts per volume, simple-cubic percolation at the midpoint density, origin independence, the winding test, the highest spanning level, a planted non-nuclear maximum, uniform limits, the layered-density tensor |
| `test_partitions.py` | 18 | 11. Weights sum to 1 and charge is conserved (four schemes), symmetric atoms, Becke's s(mu), Hirshfeld charges of a promolecule, site descriptors follow the partition, invariance of partitioned descriptors (three schemes x three transforms) |
| `test_tooling.py` | 14 | 12. Batch (record, skip, raise, read options), the CLI, sweep, grid convergence, the recommended mesh, float32 vs float64 (median relative difference below 1e-5, at most 5% of descriptors above 1e-3), matminer equality, the PAW extension (off by default, radii source, `rho_min_int`, `n_NNM_paw`), no bare NaN in any domain (both extensions on) on a uniform density |
| `test_stability.py` | 8 | The fragile set is exactly the four measured descriptors (registry and catalogue); fragile names stay in the default output and `include_fragile=False` removes only them; the `robust` extension is off by default; `register` validates `stability`; `ellip_bond_bounded_avg` against the exact Hessian of anisotropic Gaussians, 0 for a spherical atom, and a monotone map of the ellipticity |

Notes:

- The invariance fixture (`test_invariance.py::base`) is a Gaussian
  superposition with a collinear magnetization, a synthetic ELF and a
  synthetic potential attached, so every domain is exercised (the
  compositional one through matminer), and the ELF and potential take the
  "file" / "locpot" paths.
- `test_fields.py::test_elf_d_is_in_unit_interval_on_real_data` reads
  `/data/sai/new_charge/6000_data_aug13/FeNi3_221/CHGCAR` and is skipped
  when it is not available; `test_compositional_is_matminer_magpie` and
  `test_pymatgen_structure_accepted` skip without matminer / pymatgen.
- Features are cached per test module with `functools.lru_cache` where
  several tests share an expensive `featurize` call.

---

# Part IX — Cross-cutting topics

## 42. Caching: what is computed once, and under which key

Everything derived is stored in `vd.cache`, a plain dict shared by every
`with_options` view of one object (section 4). A key names the quantity and
every option its value depends on, so views with different options can share
one cache.

| Key | Value | Set by |
|---|---|---|
| `("geometry",)` | `NearestAtom` of the density grid | `core.geometry.geometry_of` |
| `("shells", (c1, c2, scaled))` | `ShellMasks` | `core.geometry.shells_of` |
| `("bond_census", tol)` | `BondCensus` | `core.geometry.bond_census_of` |
| `("pair_regions",)` | second-order Voronoi (i, j, t) | `bonding.pair_charge_transfer` |
| `("partition", scheme)` | `Partition` | `core.partition.partition_of` |
| `("field", "abs_m")` | abs(m) | `fields.density.abs_m` |
| `("derivatives", name, backend, order or 0)` | `Derivatives`: gradient, norm, Laplacians, eigenvalues | `fields.density.derivatives` |
| `("kinetic", backend, fd_order, laplacian_method)` | `KineticTerms` | `bonding.kinetic` |
| `("field", "elf_d", backend, fd_order, laplacian_method)` | ELF_D | `bonding._elf_reconstructed` |
| `("field", "potential", source)` | potential, cell average 0 | `bonding.potential` |
| `("rho_mid", BOND_TOL)` | rho at the bond midpoints | `bonding._rho_mid` |
| `("promolecule", reference, custom_reference)` | promolecule | `bonding.promolecule_density` |
| `("percolation",)` | three levels | `structural._percolation` |
| `("census",)` | census dict with `maxima`, `lower26` | `structural.census` |
| `("basins",)` | steepest-ascent basins | `structural._nnm` |
| `("information", backend, fd_order)` | S, D, I (or the sentinel) | `structural._shape_function` |
| `("site_moments", partition)` | mu_i | `magnetic.site_moments` |
| `("site_sums", "rho", partition, shells, backend, fd_order)` | `SiteSums` | `heterogeneity.density_site_sums` |
| `("site_stats", base, partition, shells, backend, fd_order, laplacian_method)` | `SiteStatistics` or sentinel | `heterogeneity._stats` |
| `("magpie",)` | 132 Magpie values | `compositional.magpie_features` |

Process-wide caches (`functools.lru_cache`): the element table
(`data._table`), the free-atom npz (`fields.deformation._tabulated`) and the
Freudenthal link table (`operators.topology._link_components`).

Not cached: `delta_rho(vd)` is recomputed (one subtraction) by each
deformation descriptor from the cached promolecule. `astype` (float32) and
every transformation in `validate/` start from an empty cache.

## 43. Units: where every conversion happens

| Where | Conversion |
|---|---|
| `io/vasp.read_blocks` | CHGCAR, AECCAR values divided by V_cell (rho * V -> rho) |
| `io/vasp.read_vasp`, `cli._read_options` | RCORE bohr -> Angstrom |
| `io/cube.read_cube` | lengths bohr -> Angstrom (unless the count is negative), density e/bohr^3 -> e/Angstrom^3 |
| `io/xsf.read_xsf` | density e/bohr^3 -> e/Angstrom^3 (lengths are Angstrom) |
| `tools/generate_free_atom_tables.py` | solver output bohr, e/bohr^3 -> stored Angstrom, e/Angstrom^3 |
| `fields/elf.kinetic_terms` | rho, grad rho, lap rho to a.u.; ELF_D dimensionless; g, v, H in hartree/bohr^3 |
| `descriptors/bonding` NCI | s and the 0.05 e/bohr^3 threshold in a.u.; `sign_lambda2_rho_mean` in e/bohr^3 |
| `descriptors/structural._shape_function` | volumes in bohr^3, rho and grad rho in a.u.; Fisher in 1/bohr^2 |
| `fields/potential` | `COULOMB_EV_ANGSTROM` gives eV from e/Angstrom^3 and Angstrom |
| magnetic domain | m in e/Angstrom^3 times dV in Angstrom^3 = electrons of net spin = mu_B |

## 44. Sentinel, flag and NaN policy

Spec §10 and §15: every degenerate case returns a documented constant with a
companion flag, never a bare NaN.

| Case name | Value | Descriptors |
|---|---|---|
| `uniform_density` | 0.0 | zeta, lnf, lnf_charge_weighted, zeta_ELF, f_NCI, NCI_attractive, sign_lambda2_rho_mean, charge_FA, per-site zeta statistics |
| `uniform_density` | 0.5 | lap_concentration, lap_concentration_valence |
| `uniform_density` | 1/3 | T_eigenvalues_t1..t3 |
| `zero_density` | 0.0 | m1, m2, sigma_r2, f_core, f_bond, f_int, lnf_charge_weighted, def_polarity(_out), rho_min_ratio, rho_min_int_ratio, the four information measures |
| `empty_region` | 0.0 (0.5 for lap_concentration_valence) | shell-restricted averages: f_H_negative, H_bond_mean, G_over_rho, the ELF shell averages, V_int_min, rho_int_mean, rho_min_int(_ratio), every `*_def_out` |
| `no_bond_voxels` | 0.0 | ellip_bond_avg, ellip_bond_std, ellip_bond_bounded_avg |
| `zero_denominator` | 0.0 | ELF_core_valence_contrast |
| `no_nci_voxels` | 0.0 | NCI_attractive, sign_lambda2_rho_mean |
| `no_bonds` | 0.0 | rho_mid_mean, rho_mid_std, bond_charge_transfer_pair_mean/std |
| `zero_deformation`, `no_accumulation`, `no_depletion` | 0.0 | the deformation moments and shares |
| `zero_levels` | 0.0 | perc_anisotropy |
| `non_magnetic` | 0.0 | every magnetic descriptor, every mu statistic |
| `single_element` | 0.0 | X_between_element_var |
| `one_site_per_element` | NaN | X_within_element_var |
| `missing_element_data` | NaN | a Magpie feature matminer returns as NaN |

Mechanics: a descriptor returns `Sentinel(value, case)` (directly, or via
`finite_or` when an operator produced NaN); `featurize` writes the value and
sets `<name>__flag = 1`, lists `name:case` in `sentinels`, and writes
`<name>__flag = 0` for every other selected descriptor that has sentinel
cases. Descriptors without sentinel cases (the percolation levels, the census
counts, n_NNM, Q_NNM, rho_min, V_spread, def_out_volume_fraction, n_NNM_paw,
Q_NNM_paw) are defined for every input. Low-density voxels are not a sentinel
case: ELF_D is 0 there and g, H, s and the Fisher integrand exclude them
(below `RHO_FLOOR_AU`). The spec's "lnf -> 0 in a ratio" case does not arise:
no registered descriptor divides by lnf.

Tests: `test_tier1.py::test_uniform_density_sentinels`,
`test_structural.py::test_uniform_limits_and_sentinels`,
`test_magnetic.py::test_non_magnetic_is_zero_never_nan`,
`test_sites.py::test_heterogeneity_sentinels`,
`test_tooling.py::test_uniform_density_no_bare_nan_in_any_domain` (every
domain plus the extension: the only NaNs are within-element variances and
Magpie features, and they are flagged).

## 45. Where the code departs from the specification

The four documented corrections and the extra decisions, with where they live
and what pins them down (README §8, PROGRESS.md §1).

| Item | Specification | Code | Where | Test |
|---|---|---|---|---|
| `lap_concentration` | ratio over the cell | kept (identically 1/2) plus `lap_concentration_valence` (r > c1) | `operators/laplacian.py`, `bonding.lap_concentration_valence` | `test_lap_concentration_is_identically_one_half` |
| percolation level | lowest spanning level | highest spanning level (the lowest is min rho) | `topology.percolation_levels` | `test_percolation_level_is_the_highest_spanning_level`, `test_simple_cubic_gaussians_percolate_at_the_midpoint_density` |
| spanning test | touches both faces in one component | nonzero winding vector | `topology.spans` | `test_spans_needs_a_winding_cluster`, `test_percolation_does_not_depend_on_the_cell_origin` |
| `"aeccar0"` reference | AECCAR0 is the promolecule | AECCAR0 + tabulated free-atom valence (ZVAL) | `bonding.promolecule_density` | `test_aeccar0_reference_is_core_plus_tabulated_valence` |
| Hirshfeld charges | Z_i | ZVAL for a pseudo-density, Z for all-electron | `partition.reference_electrons` | `test_hirshfeld_charges_of_a_promolecule` |
| saddles | 26-neighbour census | Freudenthal (14-neighbour) link; extrema stay 26-neighbour | `topology.extremum_census` | `test_census_of_a_periodic_function_with_known_critical_points` |
| information measures | S = -int rho~ ln rho~, D = int rho~^2 | S - ln V, V D | `structural._shape_function` | invariance tests, `test_uniform_limits_and_sentinels` |
| extensive quantities | normalize before registering | per atom, per volume, fraction of Q_tot; raw in metadata; `register` refuses `intensive=False` | `magnetic`, `structural`, `registry.register` | `test_register_refuses_extensive_quantities`, invariance tests |
| equidistant images | - | lexicographically largest fractional displacement; `GEOMETRY_EPS` in shell and cutoff tests | `geometry._rank`, `_nearest_images`, `shell_masks` | invariance tests |
| image range | 3x3x3 supercell | adaptive range with a proven bound | `geometry.assign_atoms`, `reps_for` | `test_extreme_shear_where_a_3x3x3_supercell_fails` |
| ANOVA terms | unweighted mean / variance over elements | site-fraction weights in both terms (exact identity) | `sitestats.variance_decomposition` | `test_anova_identity_holds_exactly` |
| Becke | fuzzy cells | 8 cells, products over 60 images, renormalized | `partition.BeckePartition` | `test_weights_sum_to_one_and_charge_is_conserved` |
| PAW-aware items | - | off-by-default `paw` extension (12 descriptors) | `bonding`, `structural` | `test_paw_extension_is_off_by_default_and_uses_the_paw_radii` |
| unconverged descriptors | - | kept as specified, tagged `stability="fragile"`; `ellip_bond_bounded_avg` in the off-by-default `robust` extension | `registry.register`, `bonding`, `structural._count` | `test_stability.py` |

Also decided where the spec is silent: FFT derivatives by default with FD
2/4/6/8 as the option (FFT rings at cusps, so use FD for AECCAR); the ZVAL
fallback `default_zval` plus per-element `zval` / `paw_radii` tables; the
promolecule point-sampled on the grid with the resulting overcount reported
as `def_charge_mismatch`; Hirshfeld reusing the promolecule's image blocks.

## 46. Known rough edges

Behaviours worth knowing when reading results or extending the code.

- **Repeated element groups.** `read_vasp` pairs POTCAR datasets with the
  distinct elements. A CHGCAR listing one element in two separate groups has
  more datasets than distinct elements, so its ZVAL list is ignored with a
  warning (and its RCORE list silently), falling back to a table or
  `default_zval`.
- **Writers regroup atoms.** `write_volumetric` writes atoms grouped by
  element, so a structure with interleaved species reads back reordered.
- **Header corner cases.** A POSCAR with three per-axis scale factors does
  not rescale Cartesian coordinates; `read_cube` takes the bohr / Angstrom
  flag of the origin and atoms from the last axis line; `read_xsf` always
  drops the last plane of each axis (it assumes a general, duplicated grid).
- **The census runs for every structure**, through the metadata hook, even
  when the structural domain is not requested.
- **Becke symmetry.** Truncating the products breaks exact ties: symmetric
  atoms agree to about 1e-5.
- **Fragile descriptors are in the default output.** `ellip_bond_avg`,
  `ellip_bond_std`, `n_saddle1` and `n_saddle2` are returned by `featurize`
  and written by `featurize_batch`; use `descriptor_names(include_fragile=
  False)` to select model inputs. `euler_consistency` is nonzero for most
  VASP pseudo-densities, so read the census counts of such structures with
  care.
- **Options without a front end.** `nnm_r_cut` has no `featurize` keyword,
  and the CLI has no flag for radius-scaled shells.
- **Unused helpers.** `io.base.check_mapping`,
  `fields.deformation.split_shell_warning`, `fields.deformation.radial_profile`
  and `core.grid.roll` are not called inside the package;
  `core.derivatives.g_squared` is used only by a test.

---

# 47. How to extend

### Adding a descriptor

1. Choose the domain module (`descriptors/bonding.py`, ...). Write
   `def my_name(vd: VolumetricData) -> Result`. The first paragraph of the
   docstring is the formula in spec notation, starting `my_name = ...`
   (`catalogue()` takes it as the definition, and
   `test_catalogue_is_generated_from_the_registry` checks that every formula
   contains `=`); the second paragraph gives limiting values and caveats.
2. Decorate it with `@register(name="my_name", domain=..., field=...,
   requires=[...], units=..., range=(lo, hi), sentinel_cases={case: value},
   references=[...])`. Add `"shells"` to `requires` if it depends on the
   cutoffs, so `sensitivity_sweep` includes it. Measure its sensitivity to
   the derivative backend and to an 80% grid (`grid_convergence`); if it
   changes by more than about 10%, pass `stability="fragile"` and extend the
   expected set in `test_stability.py::test_fragile_set_is_the_measured_one`.
3. Build it from the accessors and operators: `geometry(vd)`, `masks(vd)`,
   `field_derivatives(vd, "rho")`, `laplacian(vd)`, `kinetic(vd)`,
   `delta_rho(vd)`, `partition_of(vd, options(vd).partition)`, and
   `radial_moment`, `shell_fraction`, `gradient_anisotropy`, ... Read options
   with `options(vd)`.
4. For every degenerate input return `Sentinel(value, case)` with the case
   listed in `sentinel_cases`; wrap operator results in `finite_or`. Never
   return a bare NaN.
5. Make it intensive (per volume, per atom, or a ratio). The invariance test
   is parametrized over the registry, so the new name is checked under
   supercell, translation and rotation automatically.
6. Cache anything expensive in `vd.cache` under a key that contains every
   option the value depends on (section 42).
7. Add a test with a closed form or an exact identity (`validate/analytic.py`
   builds the densities), then run `python tools/generate_docs.py` to
   regenerate the README tables, the count sentence and `docs/catalogue.csv`.

For a family of related names, follow the factories that register in a loop
with `doc=`: `_shell` (bonding), `_perc`, `_count`, `_t` (structural), `_out`
(the `paw` extension). An extension descriptor passes `extension="paw"` or
`extension="robust"`, or a new extension name added to `EXTENSIONS`. A new
heterogeneity base is one call, `heterogeneity(base, per_site, units, formula,
requires)`, with a per-site function returning an (n_atoms,) array or a
`Sentinel`. New metadata is a `@metadata_hook` function returning a dict;
check `options(vd).domains` if it belongs to one domain, because hooks run for
every structure.

### Adding a field

Write `fn(vd) -> array` on the density grid, cache the array in `vd.cache`
with a key that includes its options, and call `register_field(name, fn)`.
`field_values(vd, name)` and `field_derivatives(vd, name)` then work, and
every operator applies. The `Derivatives` cache key holds only (name,
backend, order): if the field's values depend on another option, register
one name per value of that option (as `elf_file` / `elf_d` do) or accept the
caveat of section 46. A field read from a file needs a new optional `Grid`
attribute on `VolumetricData`, added to `_grids()` (validation), `astype`,
`validate.invariance._map_grids` and `validate.convergence._resampled`, and
resampled to rho's grid at read time.

### Adding a reader

Return a `Structure` and a `Grid` (or a `VolumetricData`) in pydemi's
conventions: Angstrom and e/Angstrom^3, `data[i, j, k]` at (i/n1, j/n2,
k/n3), periodic without duplicated planes, fractional coordinates measured
from the grid origin. Recognize the format in `io/registry.sniff` (suffix or
name), dispatch it in `read`, export it from `io/__init__.py`, and add a round
trip plus a known-units test like `test_cube_units_against_known_file`.

### Adding a partition scheme

Subclass `Partition` and implement `pairs()` yielding `PairChunk`s whose
weights sum to 1 per voxel (use `_renormalized` after dropping small
weights), with distance and direction measured to the image the scheme used.
Add the name to `registry.PARTITIONS`, build it in `partition_of`, add it to
the `--partition` choices in `cli._options`, and record any parameters in the
`partition` metadata string. Parametrize
`test_weights_sum_to_one_and_charge_is_conserved` and
`test_partitioned_descriptors_are_invariant` over it.

### Changing the element or free-atom data

`python tools/generate_element_data.py` (needs pymatgen) rewrites
`elements.csv`; `python tools/generate_free_atom_tables.py` rewrites
`free_atoms.npz` with the solver in `tools/atomic_solver.py`. Then run
`test_deformation.py::test_tables_hold_the_right_electron_counts` and
`test_io.py::test_default_zval_matches_standard_paw_datasets`, and rerun the
invariance and partition tests, since the promolecule and the Hirshfeld
weights depend on the tables.
