# 2. A dataset of descriptors

`pydemi batch` (or `pydemi.featurize_batch`) featurizes many CHGCARs in parallel, one process
per structure, and writes one row per structure. A failure on one file is recorded in its row
and never stops the run: pydemi computed the descriptors of all 6,059 structures of the CMS
Lab dataset without one.

## Lay out the runs

Any layout works; you give a root and a glob:

```text
runs/
├── Cu_225/CHGCAR
├── GaAs_216/CHGCAR
├── MgO_225/CHGCAR
├── NaCl_225/CHGCAR
├── SbO2_15/CHGCAR
└── Si_227/CHGCAR
```

## Run it

```bash
pydemi batch runs --glob "*/CHGCAR" --out all.csv --workers 6
```

```text
...
[5/6] runs/GaAs_216/CHGCAR: ok (8.1 s)
[6/6] runs/NaCl_225/CHGCAR: ok (8.1 s)
wrote all.csv: 6 structures, 0 errors
```

With 6 workers the six took 12 s of wall time. The same from Python:

```python
import pydemi
from pydemi.batch import find_runs

paths = find_runs("runs", "*/CHGCAR")
df = pydemi.featurize_batch(paths, n_workers=6, on_error="record")
```

Options: `--domains`, `--partition`, `--shells` and the numerical settings as for
`featurize`; `--on-error record|raise|skip`; `--quiet`.

## What is in the table

117 columns: `path`, the 54 descriptors in registry order, then the metadata (`n_atoms`,
`volume`, `grid_shape`, `site_counts`, the settings, `euler_consistency`, the 54
`<name>__flag` columns, `sentinels`, `wall_time_s`, `pydemi_version`, `error`). A few columns
of the real output:

| structure | f_core | f_bond | f_int | f_NCI | zeta | V_spread | rho_perc_a | n_NNM | m1_between_element_var |
|---|---|---|---|---|---|---|---|---|---|
| Cu (fcc) | 0.8467 | 0.1482 | 0.0050 | 0.5661 | 0.0025 | 0 | 0.257 | 0 | 0 |
| GaAs (zinc blende) | 0.3123 | 0.4510 | 0.2368 | 0.1713 | 0.1255 | 7.069 | 0.2177 | 0 | 0.0026 |
| MgO (rock salt) | 0.4023 | 0.5483 | 0.0494 | 0.1643 | 0.0242 | 60.17 | 0.1133 | 0 | 0.0444 |
| NaCl (rock salt) | 0.4187 | 0.4633 | 0.1180 | 0.0412 | 0.0248 | 35.37 | 0.0295 | 0 | 0.0696 |
| SbO₂ (Sb₂O₄) | 0.4133 | 0.3987 | 0.1880 | 0.1611 | 0.0260 | 54.97 | 0.2897 | 0 | 0.0254 |
| Si (diamond) | 0.2425 | 0.4510 | 0.3065 | 0.1700 | 0.2121 | 0 | 0.3865 | 0.1957 | 0 |

Already a few chemical trends are visible:

- **Cu**, a metal, has the largest `f_NCI` (0.57): most of its cell holds low, slowly varying,
  free-electron-like density. Its d shell keeps 85 % of the valence charge in the core
  spheres, and its atoms are almost spherical (`zeta` 0.0025).
- **NaCl** has the lowest percolation level (0.029 e/Å³): any path through the crystal crosses
  nearly empty space, as expected of an ionic solid. Its `m1_between_element_var` is the
  largest: small Na⁺, large Cl⁻.
- **Si** is the only one with non-nuclear maxima (`n_NNM` > 0): its valence pseudo-density
  peaks on the bonds. It also has the largest `zeta`: the most directional bonding.
- **MgO** and **NaCl** have the largest `V_spread`: strong charge transfer.

## Into pandas

```python
import pandas as pd

df = pd.read_csv("all.csv")
df = df[df.error.isna()]                                   # rows that succeeded
X = df[pydemi.descriptor_names()]                          # the feature matrix
flagged = df.filter(like="__flag").sum()                   # how often each descriptor fell back
```

## Large datasets

- **Workers**: one process per structure; each holds one density and its derived fields. Use
  {doc}`../guide/performance` to size `--workers` by memory, not by cores.
- **Restart**: write one CSV per chunk of runs and concatenate; rows are independent.
- **Consistency**: densities of one dataset should come from one set of POTCARs, because the
  POTCAR sets the number of valence electrons (see {doc}`../descriptors/theory`).
