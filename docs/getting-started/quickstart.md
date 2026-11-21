# Quick start

Five minutes from a CHGCAR to descriptors, a plot and property estimates. The numbers below
are those of zinc-blende GaAs (`mp-2534`, conventional cell, 8 atoms, 84³ grid) from the
CMS Lab VASP dataset.

## 1. Read the CHGCAR

```python
import pydemi

vd = pydemi.read_chgcar("CHGCAR")      # also CHG, and .gz / .bz2 / .xz
vd.structure.species                   # ['Ga', 'Ga', 'Ga', 'Ga', 'As', 'As', 'As', 'As']
vd.shape                               # (84, 84, 84)
```

The structure comes from the header and the density from the first data block, divided by
the cell volume so that it is in electrons / Å³. Nothing else is read: no POTCAR, OUTCAR,
AECCAR, ELFCAR or LOCPOT.

## 2. Compute the descriptors

```python
feats = pydemi.featurize(vd)           # dict: name -> float, all 54
feats["f_core"], feats["f_bond"], feats["f_int"]
# (0.3123, 0.4510, 0.2368): a third of the valence charge in the cores, 45 % in the bonds
feats["V_spread"]                      # 7.07 eV: Ga and As feel very different potentials
feats["perc_anisotropy"]               # 0.0: cubic, the charge connects equally along a, b, c
```

On one machine this took about 7 s. `pydemi.featurize(vd, return_metadata=True)` also returns
the settings and quality flags ({doc}`../guide/metadata`).

````{dropdown} All 54 values for GaAs
```{include} ../_generated/md/gaas_features.md
```
````

## 3. Look up what a descriptor means

```python
cat = pydemi.catalogue()               # pandas DataFrame, one row per descriptor
cat.set_index("name").loc["zeta", ["domain", "units", "formula"]]
```

Every descriptor has a page entry with its definition, formula, physics, limits and an
interactive figure: see {doc}`../descriptors/index`.

## 4. Estimate properties

```python
from pydemi import props               # pip install "pydemi[props]"
p = props.predict(feats)
p["bulk_modulus"], p["band_gap"]       # (67.4 GPa, 0.62 eV)
```

GaAs is in the test set of the models, so this is a fair check: its DFT bulk modulus is
60.7 GPa and its PBE gap 0.18 eV ({doc}`../tutorials/properties`).

## 5. Do the same from the command line or the browser

```bash
pydemi featurize CHGCAR --out features.json
pydemi gui
```

## Next

- {doc}`../tutorials/dataset`: a table of descriptors for thousands of structures.
- {doc}`../tutorials/predicted-densities`: no CHGCAR? Predict it from a CIF.
- {doc}`../guide/regions`: what "core", "bond" and "interstitial" mean here.
