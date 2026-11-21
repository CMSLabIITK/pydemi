# Changelog

## 0.4.0 (unreleased)

- **54 descriptors**: `fisher_information` is removed (breaking). It needed two cut-offs of its
  own, a density floor and a 0.5 Angstrom core exclusion (`FISHER_CORE_RADIUS`, also removed),
  and added little that the Shannon entropy, disequilibrium and LMC complexity do not already
  carry. `featurize`, `descriptor_names()`, the catalogue and CSV outputs have one column
  fewer; the structural domain has 18 descriptors, and the graphical interface has no Fisher
  plot.
- **Power-diagram partition removed** (breaking): the site descriptors use the nearest-atom
  partition (default) or Becke cells; `partition="power"` and the `power_radii` argument of
  the geometry pass are gone. The power diagram reordered the structures by their site
  statistics, which made those descriptors depend on an extra choice of atomic radii; the
  bond-region fraction per atom already uses cells scaled by the covalent radii.
- **Regions by the sphere rule** (breaking): the core, bond and interstitial regions are no
  longer shells at fixed distances (0.8 / 1.5 Angstrom) from the nearest nucleus. Around every
  nucleus i lie a core sphere of radius a R_i and an outer sphere of radius b R_i (R_i the
  covalent radius, a = 0.79, b = 1.14); a voxel is core inside any core sphere, bond inside an
  outer sphere but in no core sphere, and interstitial outside every outer sphere. The factors
  are the 1st and 99th percentiles of d / (R_A + R_B) over the nearest neighbour of every atom
  of the 6,059-structure dataset. This is the only region definition: `Shells(a, b)` takes the
  two factors (the `scaled` mode and Angstrom cutoffs are gone; `SHELL_C1`/`SHELL_C2` are
  replaced by `SPHERE_A`/`SPHERE_B`), the command line takes `--shells a,b`, and
  `pydemi sweep` varies `--param a` or `b`. Affected: f_core, f_bond, f_int,
  lap_concentration_valence, ellip_bond_bounded_avg, V_int_min, rho_int_mean and the f_bond
  site statistics, whose per-site sums give each voxel to the atom with the smallest r / R.
- **Non-nuclear maxima outside the core spheres**: n_NNM and Q_NNM count the local maxima
  outside every core sphere instead of those farther than 0.8 Angstrom (`NNM_R_CUT` and the
  `nnm_r_cut` option are removed). The shell maximum of the pseudo-density around a heavy atom
  no longer counts as non-nuclear (GaBi: Q_NNM 1.0 before, 0 now).
- **Within-element variances are 0, not NaN**, when no element has two or more sites: the
  value of every supercell of the structure, so the three descriptors are now unchanged under
  a supercell for every structure (flagged as before). No descriptor returns NaN.
- `pydemi.props` reads model bundles on the descriptors only: `props.composition_features`
  and the `counts=` argument of `props.predict` are removed (the shipped models never used
  them), and the graphical interface no longer has an inputs column.
- **Lower peak memory**: the sphere-rule cells store no direction field and an int32 atom index,
  and they skip the gradient sums that only the nearest-atom cells need. On the 216³ grid of
  SnBi2Te4 the peak falls from 3.05 to 2.77 GB with the GUI plot data. The results are unchanged.
- The covalent radii are credited to Cordero et al. (2008) directly; the values are unchanged.
- `property_prediction/` keeps only the scripts behind the results of the paper.

## 0.3.0 (2026-11-02)

- **55 descriptors**: `lap_concentration` is removed. On a periodic grid the Laplacian
  integrates to zero over the cell, so this ratio was 1/2 for every density and carried no
  information. Its informative form, `lap_concentration_valence` (the same ratio outside the
  cores), stays. `featurize`, `descriptor_names()`, the catalogue and CSV outputs have one
  column fewer; the bonding domain has 18 descriptors. The shipped property models never used
  it and are unchanged.
- The graphical interface shows descriptive names ("Interstitial charge fraction") in its
  tables, plots and property charts, with the short name (`f_int`) in the descriptor panel
  and in the exports; the command line and the Python API keep the short names.
- `read_chgcar`, `featurize`, `pydemi featurize` and `pydemi batch` read gzip-, bzip2- and
  xz-compressed CHGCAR files, recognised by their first bytes, so the file name does not
  matter; a run directory that holds only `CHGCAR.gz` (or `.bz2`, `.xz`) is found too. Until
  now only the graphical interface accepted compressed files.

## 0.2.2 (2026-11-02)

- **Property models v2** (`properties_v2.json.xz`, replaces v1): the 17 properties of the
  pydemi paper, each predicted from the descriptors alone (tuned LightGBM). v1 chose
  between the composition, the descriptors and both for each property and had 22
  properties; the Poisson ratio, the anisotropy index, the energy above the hull and the
  three dielectric constants are dropped (descriptor-only test R² below 0.70).
  `props.predict` no longer needs `counts=` for the shipped models; bundles that read the
  composition still work.
- pydemi logo (the valence density of Si around a Si–Si bond) in the README and in
  the graphical interface (sidebar and browser tab).
- Descriptor reference (`docs/descriptors.pdf`, also in the GUI): every descriptor with its
  definition, formula, physical meaning, limits and caveats.

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
