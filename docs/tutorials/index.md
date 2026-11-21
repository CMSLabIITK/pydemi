# Tutorials

Each tutorial is one task, start to finish, with the commands and the real output on VASP
CHGCARs of the CMS Lab dataset (PAW-PBE, ENCUT 500 eV).

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 1. One CHGCAR, in depth
:link: one-chgcar
:link-type: doc
Read a density, choose domains and settings, keep the metadata, export a row.
:::
:::{grid-item-card} 2. A dataset of descriptors
:link: dataset
:link-type: doc
Thousands of CHGCARs in parallel, errors recorded, ready for pandas.
:::
:::{grid-item-card} 3. The graphical interface
:link: gui
:link-type: doc
Every view of `pydemi gui`, what it shows and what it exports.
:::
:::{grid-item-card} 4. Densities without DFT
:link: predicted-densities
:link-type: doc
Predict a CHGCAR from a CIF with ChargE3Net and compare it with DFT.
:::
:::{grid-item-card} 5. Property estimates
:link: properties
:link-type: doc
17 properties from the descriptors, with test scores and the dataset check.
:::
:::{grid-item-card} 6. Your own model
:link: own-model
:link-type: doc
Train a model of your property on pydemi descriptors and read it with SHAP.
:::
:::{grid-item-card} 7. How much do the regions matter?
:link: sensitivity
:link-type: doc
Sweep the sphere factors and see which descriptors move.
:::
::::

```{toctree}
:hidden:

one-chgcar
dataset
gui
predicted-densities
properties
own-model
sensitivity
```
