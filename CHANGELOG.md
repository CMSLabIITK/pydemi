# Changelog

## 0.2.1 (2026-10-28)

- **Compare view** in the graphical interface: the open CHGCAR against the density
  ChargE3Net predicts for the same structure, on the CHGCAR's own grid. It shows the
  density error, the 56 descriptors from both (relative difference, values that are
  numerical noise in both, and each descriptor's transfer to predicted densities over the
  605 held-out structures of the density model), and the 22 properties predicted from
  each, marked where they differ by more than the model's typical error. CSV export.
- Background predictions accept a grid (`pydemi.gui.predict.Jobs.submit(..., shape=)`).

## 0.2.0 (2026-10-26)

Descriptors of a crystal's electron density from one input, the VASP CHGCAR,
with a graphical interface, densities predicted from the structure, and
property estimates.

- **Descriptors: 56**, all computed by default from the CHGCAR alone:
  19 bonding, 19 structural and 18 site-heterogeneity. Each is registered with
  its formula, units, range and fallback values (`pydemi.catalogue()`).
- **Not included, on purpose:** ELF and local energy-density descriptors (from
  a CHGCAR, a PAW pseudo-density, they would rest on a density-only
  kinetic-energy estimate, not the true quantities), the unbounded ellipticity
  and the saddle-point counts (numerically unstable on VASP grids), and
  anything that needs a file other than the CHGCAR. Partitions: nearest,
  power, Becke.
- **Graphical interface:** `pip install "pydemi[gui]"`, then `pydemi gui`.
  It runs on the user's machine and opens in the browser; nothing is uploaded
  anywhere.
  - Laid out like desktop software: an input panel (open a CHGCAR, plain,
    gz, bz2, xz or zip, or a structure for ChargE3Net; properties; settings)
    and four views: the structure and density slices; the descriptor table
    with search, export and the selected descriptor's plot; a plot for
    each descriptor, built so that the plotted data reproduce its value; and
    material properties estimated from the descriptors.
  - Streamlit and plotly are an optional extra; the library itself still
    needs only numpy, scipy and pandas.
- **Densities predicted from the structure** (optional `ml` extra:
  PyTorch, e3nn, ASE): `pydemi.ml.predict`, `pydemi predict`, and the
  *Crystal structure (ML)* tab of the GUI.
  - The ChargE3Net network (Koker et al. 2024, MIT) is included, adapted to
    one structure at a time on any device; its predictions equal those of the
    original code (tested to 1e-5 of the peak density). It ships with a
    checkpoint fine-tuned by CMS Lab on its VASP dataset (8 MB).
  - The grid is the one VASP would use (`pydemi.ml.vasp_grid_shape`; all
    6,059 dataset grids reproduced), and the density is rescaled to the
    valence electrons of the training POTCARs (`pydemi.ml.ZVAL`, 87 elements).
  - The GUI predicts in the background, with progress, time left and cancel,
    then offers the three tasks on the prediction and a predicted-CHGCAR download.
- **Property estimates** (LightGBM; in the `gui` extra, or `props` for the
  library alone): `pydemi.props` and the *Properties* view of the GUI estimate
  22 properties from the descriptors and the composition with the models that
  ship with pydemi (35 MB, trained on the CMS Lab dataset with Materials
  Project targets), and show each estimate with its test-set error and score,
  a parity plot and the inputs the model relies on most. Models with a test R² below 0.5 are marked ⚠ as rough
  indications; descriptors outside the training range are flagged.
- **Command line:** `pydemi featurize`, `batch`, `catalogue`, `sweep`, `gui` and
  `predict`.
- **Tests:** 342, including an end-to-end test of the GUI (Streamlit's
  AppTest) for both inputs. CI runs on Linux, macOS and Windows, Python
  3.10–3.13, with and without the GUI extra, and the ML extra on all three
  systems; `mypy --strict` covers the library.
