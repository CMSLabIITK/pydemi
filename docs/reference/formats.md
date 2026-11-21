# File formats

## Input

| input | used by | notes |
|---|---|---|
| VASP **CHGCAR** or **CHG** | everything | VASP 5 (species line) or VASP 4 (pass `species=`); plain, `.gz`, `.bz2`, `.xz` (the interface also takes `.zip`). Only the first block, the total density, is read. |
| **CIF**, **POSCAR/CONTCAR** (`.vasp`, `.poscar`), `.xyz`, `.extxyz`, `.json`, `.in` | `pydemi predict`, `pydemi.ml` | any format ASE reads; the first structure of a multi-structure CIF |

What a CHGCAR holds: a POSCAR header (comment, scale, lattice, species, counts, coordinates),
a blank line, the grid dimensions $n_1\,n_2\,n_3$, then $\rho\cdot V$ on the grid in Fortran
order (first index fastest). A spin-polarised run then writes the PAW augmentation
occupancies and the magnetisation density; pydemi stops after the first block.

```{note}
The CHGCAR density is the PAW **valence pseudo-density** of the POTCARs used. Descriptors of
densities from different POTCAR sets (for example, `_pv` and plain variants of an element)
are not directly comparable. See {doc}`../descriptors/theory`.
```

## Output

| command | format | contents |
|---|---|---|
| `featurize --out x.json` | JSON | `path`, `features` (name → value), `metadata` |
| `featurize --out x.csv` | CSV | one row: `path`, descriptors, metadata columns |
| `batch --out x.csv` | CSV | one row per structure, same columns, `error` filled on failure |
| `catalogue --out x.csv` | CSV | name, domain, field, requires, units, range_min, range_max, intensive, sentinel_cases, formula |
| `sweep --out x.csv` | CSV | `a`, `b`, and the fifteen region-dependent descriptors |
| `predict --out-dir d` | CHGCAR | `d/<name>.CHGCAR`, VASP 5 layout, no augmentation section; with `--featurize` also `d/<name>_descriptors.json` |

`pydemi.write_volumetric` writes any density in the same CHGCAR layout, so a predicted or
modified density can be read back by pydemi, VESTA or other tools.
