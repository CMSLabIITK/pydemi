# 1. One CHGCAR, in depth

This tutorial walks through everything `featurize` can do for one structure. The example is
zinc-blende GaAs (8-atom conventional cell, 84 × 84 × 84 grid).

## Reading the file

```python
import pydemi

vd = pydemi.read_chgcar("GaAs/CHGCAR")
vd.structure.lattice.matrix        # rows a, b, c in Å (here 5.748 Å cubic)
vd.structure.species               # one symbol per atom
vd.structure.frac_coords           # (8, 3) fractional coordinates
vd.rho.data                        # (84, 84, 84) density in e/Å³
```

What the reader does:

- the file may be plain or compressed with gzip, bzip2 or xz (detected from its first bytes);
- the values of a CHGCAR are ρ·V; they are divided by the cell volume;
- only the first block, the **total** density, is read. A spin-polarised run writes the
  magnetisation density after it; it is not used;
- a VASP 4 file has no species line: pass `species=["Ga", "As"]`.

```{note}
The density in a CHGCAR is the **valence pseudo-density** of the PAW method: the deep cores
are absent and it is smooth (and can be slightly negative) inside the augmentation spheres.
See {doc}`../descriptors/theory` for what this means for the descriptors.
```

## All descriptors, or some domains

```python
feats = pydemi.featurize(vd)                                  # all 54
bond = pydemi.featurize(vd, domains="bonding")                # 18
some = pydemi.featurize(vd, domains=["bonding", "structural"])
pydemi.descriptor_names("heterogeneity")                      # the 18 names, in order
```

The order of the dict is the registry order, the same as `descriptor_names()` and the columns
of `catalogue()`, so the vectors of different structures line up.

## Settings

Every option is recorded in the metadata. The defaults are the settings of the paper and of
the shipped property models; change them only with a reason.

| argument | default | alternatives | see |
|---|---|---|---|
| `partition` | `"nearest"` | `"becke"` | {doc}`../guide/partitions` |
| `shells` | `(0.79, 1.14)` | any `(a, b)` with `a < b` | {doc}`../guide/regions` |
| `derivative_backend` | `"fft"` | `"fd"` (with `fd_order` 2, 4, 6, 8) | {doc}`../guide/numerics` |
| `laplacian_method` | `"metric"` | `"diagonal"` (for testing only) | {doc}`../guide/numerics` |
| `float32` | `False` | `True` (half the memory) | {doc}`../guide/performance` |

```python
feats_b = pydemi.featurize(vd, partition="becke")            # Becke cells for m1 and zeta sites
feats_s = pydemi.featurize(vd, shells=(0.75, 1.20))          # other sphere factors
```

## Metadata and flags

```python
feats, meta = pydemi.featurize(vd, return_metadata=True)
meta["shells"], meta["partition"], meta["grid_shape"]   # 'spheres(0.79,1.14)', 'nearest', '84x84x84'
meta["euler_consistency"]                                # -48 for GaAs (see the guide)
meta["sentinels"]                                        # '' : no descriptor fell back
meta["f_NCI__flag"]                                      # 0
```

A descriptor that is undefined for a special input (a uniform density, an empty region, no
bonds) returns a documented constant and sets its `<name>__flag` to 1; the names of all such
descriptors are listed in `meta["sentinels"]`. No descriptor returns NaN.
{doc}`../guide/metadata` lists every key.

## A table row, and the catalogue

```python
import pandas as pd

row = pd.DataFrame([feats], index=["GaAs"])
cat = pydemi.catalogue()          # name, domain, field, requires, units, range, intensive,
                                  # sentinel cases, formula
cat[cat.domain == "structural"][["name", "units", "formula"]]
```

From the command line the same in one go:

```bash
pydemi featurize GaAs/CHGCAR --out gaas.json                  # features + metadata
pydemi featurize GaAs/CHGCAR --out gaas.csv --domains bonding  # one row
pydemi catalogue --out catalogue.csv
```

The JSON holds `path`, `features` and `metadata`.

## Reading the numbers

The values for GaAs (all 54 are in the {doc}`quick start <../getting-started/quickstart>`):

- `f_core` 0.31, `f_bond` 0.45, `f_int` 0.24: much of the valence charge is in the bonds, as
  in a covalent sp³ semiconductor;
- `zeta` 0.125: the gradient is far from radial, so the atoms are strongly deformed by
  directional bonds (Cu: 0.0025);
- `V_spread` 7.07 eV: the Hartree potential differs between the Ga and As nuclei, the
  signature of charge transfer from Ga to As (in elemental Si it is 0);
- `perc_anisotropy` 0 and `T_eigenvalues_t1..t3` all 1/3: cubic symmetry, as Neumann's
  principle requires;
- `rho_mid_std` ≈ 7·10⁻⁹ e/Å³: all Ga–As bonds are equivalent.

Every descriptor's full entry, with an interactive figure for GaAs, is in
{doc}`../descriptors/index`.
