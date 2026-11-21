# 3. The graphical interface

```bash
pip install "pydemi[gui]"      # add ,ml for the Structure (ML) input and the Compare view
pydemi gui
```

This starts a local server and opens the interface in your browser. It listens on
`localhost`, so only your computer can reach it and your files never leave your machine.
Stop it with Ctrl+C.

| option | effect |
|---|---|
| `--port 8600` | another port (default 8501, or the next free one) |
| `--no-browser` | do not open a browser (e.g. on a remote machine; then forward the port with `ssh -L 8501:localhost:8501 host`) |
| `--host 0.0.0.0` | let other machines on your network reach it |

The screenshots below are of zinc-blende GaAs (its CHGCAR from the CMS Lab dataset).

## The left panel

```{image} ../_static/screenshots/gui_start.png
:alt: The interface before a file is opened
:width: 100%
```

- **Input.** *CHGCAR*: open a CHGCAR or CHG, plain or `.gz`, `.bz2`, `.xz`, `.zip`.
  *Structure (ML)*: open a CIF or POSCAR/CONTCAR and let ChargE3Net predict its density (see
  {doc}`predicted-densities`).
- **Summary.** Formula, atoms, volume, electrons, grid and lattice of the open file.
- **Settings.** The atomic partition of the per-site descriptors of `m1` and `zeta`
  ({doc}`../guide/partitions`).
- **Help.** The descriptor reference as a PDF.

The descriptors are computed once per file and partition; choices are kept when you switch
views.

## Structure

```{image} ../_static/screenshots/gui_structure.png
:alt: Structure view
:width: 100%
```

The cell and its atoms in 3D, and a slice of the density through any lattice plane, with a
log or linear colour scale and the atoms near the plane marked. The same figures, live, are on
the {doc}`front page <../index>`.

## Descriptors

```{image} ../_static/screenshots/gui_descriptors.png
:alt: Descriptors view
:width: 100%
```

All 54 descriptors in a table, with descriptive names, a filter by domain and a search over
names and formulas. Select a row to see its plot, value, units and formula beside the table.
A ticked *fallback* box marks a descriptor that took its documented value for a special case
({doc}`../guide/metadata`).

**Export**:

- *CSV, one row*: one row per structure, the format to build datasets from;
- *CSV, table*: one row per descriptor, with units and formulas;
- *JSON*: the values with the run metadata.

## Plots

```{image} ../_static/screenshots/gui_plots.png
:alt: Plots view
:width: 100%
```

One figure per descriptor, showing the quantity it summarises with its value marked, stepping
through each domain. The plotted data reproduce the descriptor values (this is tested). Every
entry of {doc}`../descriptors/index` embeds the same figure for a real material.

## Properties

```{image} ../_static/screenshots/gui_properties.png
:alt: Properties view
:width: 100%
```

The 17 property estimates, each with the typical error and the test score of its model.
Select a row for the parity plot of the model on the test set and the descriptors it relies
on most (mean |SHAP|). When the open file is a material of the training dataset, its DFT
values are shown alongside, with its split. GaAs is in the **test** set, so the comparison
above is a genuine check. A warning lists the descriptors that lie outside the 1st to 99th
percentile of the training materials, where an estimate is less reliable. See
{doc}`properties`.

## Compare

```{image} ../_static/screenshots/gui_compare.png
:alt: Compare view before the prediction
:width: 100%
```

Needs the `ml` extra. For the open CHGCAR, ChargE3Net predicts the density of the same
structure on the same grid, and the two are set side by side:

- the density error (NMAPE over the grid);
- every descriptor from both, with its relative difference and how well that descriptor
  transfers to predicted densities in general (from the 605 held-out test structures);
- every property estimated from each, marked where the two differ by more than the model's
  typical error.

Export as CSV, and download the predicted CHGCAR.
