---
html_theme.sidebar_secondary.remove: true
---

# pydemi

```{image} _static/logo.png
:class: hero-logo only-light
:alt: pydemi
```
```{image} _static/logo_dark.png
:class: hero-logo only-dark
:alt: pydemi
```

```{rst-class} hero-tagline
```
**Interpretable, fixed-length descriptors of a crystal's electron density, from a VASP
`CHGCAR` and nothing else.** 54 named numbers of bonding, structure and site heterogeneity,
each with its formula, units, range and physical meaning; property estimates from them; and,
when there is no DFT calculation, a density predicted from the crystal structure alone.

::::{grid} 2 2 4 4
:gutter: 2

:::{grid-item}
<div class="big-number">54</div><div class="big-label">documented descriptors</div>
:::
:::{grid-item}
<div class="big-number">1 file</div><div class="big-label">the CHGCAR: no POTCAR, OUTCAR or AECCAR</div>
:::
:::{grid-item}
<div class="big-number">17</div><div class="big-label">property models (test R² mostly 0.86–0.95)</div>
:::
:::{grid-item}
<div class="big-number">3 ways</div><div class="big-label">Python, command line, graphical interface</div>
:::
::::

```bash
pip install pydemi
```

## What it does

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`beaker;1.2em` Describe the bonding
:link: descriptors/bonding
:link-type: doc

Where the charge sits (core, bond, interstitial), how spherical the atoms are, where it is
concentrated (Laplacian), the shape of the bonds, weak interactions (NCI) and the
electrostatic potential: **18 bonding descriptors**.
:::

:::{grid-item-card} {octicon}`globe;1.2em` Describe the structure
:link: descriptors/structural
:link-type: doc

How well the charge connects through the crystal along each axis, its peaks and voids,
non-nuclear maxima, the density floor, its directionality and its information content:
**18 structural descriptors**.
:::

:::{grid-item-card} {octicon}`people;1.2em` Compare the atoms
:link: descriptors/heterogeneity
:link-type: doc

Per-atom size, bond share and asphericity, summarised over the sites, with the
site variance split exactly into *within-element* (inequivalent sites) and
*between-element* parts: **18 heterogeneity descriptors**.
:::

:::{grid-item-card} {octicon}`graph;1.2em` Estimate properties
:link: tutorials/properties
:link-type: doc

LightGBM models of 17 elastic, thermal, energetic, electronic and optical properties that
read the descriptors alone, each with its test score and typical error.
:::

:::{grid-item-card} {octicon}`cpu;1.2em` Skip the DFT run
:link: tutorials/predicted-densities
:link-type: doc

ChargE3Net, fine-tuned on the CMS Lab VASP dataset, predicts the CHGCAR from a CIF or POSCAR
(0.72 % mean error on 605 held-out structures); every descriptor then follows.
:::

:::{grid-item-card} {octicon}`browser;1.2em` Explore in the browser
:link: tutorials/gui
:link-type: doc

A local graphical interface: structure and density slices, all descriptors with a plot of
each, property estimates, and DFT vs predicted densities side by side.
:::
::::

## Try it

`````{tab-set}
````{tab-item} Python
```python
import pydemi

vd = pydemi.read_chgcar("CHGCAR")       # structure + density, from the one file
feats = pydemi.featurize(vd)            # dict of all 54 descriptors
feats["f_bond"], feats["V_spread"]      # GaAs: (0.451, 7.07 eV)

pydemi.catalogue()                      # formula, units, range of every descriptor
```
````
````{tab-item} Command line
```bash
pydemi featurize CHGCAR --out features.json
pydemi batch ./runs --glob "*/CHGCAR" --out all.csv --workers 8
pydemi predict GaAs.cif --out-dir predicted --featurize
pydemi gui
```
````
````{tab-item} Property estimates
```python
from pydemi import props

props.predict(feats)    # GaAs: bulk modulus 67 GPa (DFT 61), band gap 0.62 eV (PBE 0.18), ...
```
````
`````

## See it

The density of GaAs from its VASP CHGCAR, as the *Structure* view of the graphical interface
shows it. Every descriptor page has a figure like this for the quantity behind the number.

`````{tab-set}
````{tab-item} Density slice
```{raw} html
:file: _generated/figures/_slice.html
```
````
````{tab-item} Structure
```{raw} html
:file: _generated/figures/_structure.html
```
````
````{tab-item} A descriptor: f_bond
```{raw} html
:file: _generated/figures/f_bond.html
```
````
````{tab-item} Site heterogeneity (SbO₂)
```{raw} html
:file: _generated/figures/m1_within_element_var.html
```
````
`````

## Where to go next

| if you want to… | read |
|---|---|
| install pydemi | {doc}`getting-started/installation` |
| compute descriptors in five minutes | {doc}`getting-started/quickstart` |
| learn one task at a time | {doc}`tutorials/index` |
| understand a descriptor | {doc}`descriptors/index` |
| understand regions, partitions and numerical settings | {doc}`guide/index` |
| look up a function or command | {doc}`reference/api`, {doc}`reference/cli` |
| cite pydemi | {doc}`project/citing` |

```{toctree}
:hidden:
:caption: Getting started

getting-started/installation
getting-started/quickstart
tutorials/index
getting-started/faq
```

```{toctree}
:hidden:
:caption: Descriptors

descriptors/index
```

```{toctree}
:hidden:
:caption: User guide

guide/index
```

```{toctree}
:hidden:
:caption: Reference

reference/api
reference/cli
reference/formats
descriptors/references
```

```{toctree}
:hidden:
:caption: Project

project/citing
project/changelog
project/development
project/license
```
