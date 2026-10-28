# pydemi

Interpretable, fixed-length descriptors of a crystal's electron density, computed from a
VASP CHGCAR and nothing else. The structure comes from the CHGCAR header and the total
density from its first data block; no POTCAR, OUTCAR, AECCAR, ELFCAR or LOCPOT is needed.
Without a DFT calculation, the density can instead be predicted from the crystal structure
alone by ChargE3Net (optional `ml` extra).

pydemi runs on your own computer (Linux, macOS or Windows; Python 3.10 or newer), as a
Python library, a command-line tool, or a graphical interface in your browser.

## Install

```bash
pip install "pydemi[gui]"     # library, command line and graphical interface (with property estimates)
pip install "pydemi[gui,ml]"  # ... and densities predicted from a structure (PyTorch, e3nn, ASE)
pip install "pydemi[props]"   # library and command line, with property estimates (LightGBM)
pip install pydemi            # library and command line only (numpy, scipy, pandas)
```

The `ml` extra pulls in PyTorch. For a CPU-only machine the CPU build is much smaller:
`pip install torch --index-url https://download.pytorch.org/whl/cpu` first. With an NVIDIA
GPU, install the CUDA build of PyTorch first (see pytorch.org); pydemi then uses the GPU.

On macOS, the property models (LightGBM) need the OpenMP runtime: `brew install libomp`.

A separate environment keeps it tidy, e.g. `python -m venv pydemi-env` (then
`source pydemi-env/bin/activate`, or `pydemi-env\Scripts\activate` on Windows), or
`conda create -n pydemi python=3.12`.

## Graphical interface

```bash
pydemi gui
```

This starts a local server and opens the interface in your browser. It listens on
`localhost`, so only your computer can reach it and your files stay on your machine.
Stop it with Ctrl+C. Options: `--port 8600`, `--no-browser`, and `--host 0.0.0.0` to
let other machines on your network reach it.

The window is laid out like desktop software:

- **Left panel.** *Input*: open a CHGCAR (or CHG; plain or `.gz`, `.bz2`, `.xz`, `.zip`),
  or, in the *Structure (ML)* tab, a CIF or POSCAR/CONTCAR whose density ChargE3Net
  predicts (see below). *Summary*: formula, atoms, volume, electrons, grid and lattice.
  *Settings*: the atomic partition for the per-site descriptors. *Help*: the descriptor
  reference (PDF).
- **Views**, in tabs across the top:
  - *Structure*: the cell and atoms in 3D, and a slice of the density through any lattice
    plane (log or linear colour, atoms near the plane marked).
  - *Descriptors*: all 56 in a table with a domain filter and search; select a row to see
    its plot, value, units and formula alongside. *Export* gives a one-row CSV (for
    building datasets), a long CSV with units and formulas, or JSON with the run metadata.
  - *Plots*: one figure per descriptor, showing the quantity it summarises with its value
    marked, stepping through each domain. The plotted data reproduce the descriptor
    values (this is tested).
  - *Properties*: 22 material properties (elastic, thermal, energetic, electronic and
    dielectric) estimated from the descriptors and the composition, each with its typical
    error and test score; select a row for the parity plot on the test set and the inputs
    the model relies on most. Models with a test R² below 0.5 are marked ⚠ and give rough
    indications only. If the material is in the training dataset, its DFT values are
    shown alongside.
  - *Compare* (needs the `ml` extra): for the open CHGCAR, ChargE3Net predicts the
    density of the same structure on the same grid, and the two are set side by side: the
    density error (NMAPE), every descriptor from both with its relative difference and how
    well it transfers to predicted densities in general, and every property predicted from
    each, marked where the two differ by more than the model's typical error. Export as CSV,
    with the predicted CHGCAR.

Choices are kept when you switch views; the descriptors are computed once per file and
partition.

## Densities predicted from the structure (ChargE3Net)

When there is no CHGCAR, pydemi can predict one with ChargE3Net (Koker et al., npj Comput.
Mater. 10, 161 (2024); MIT licence), an E(3)-equivariant network. pydemi includes the
network and a checkpoint that CMS Lab fine-tuned on its own VASP dataset (PAW-PBE, ENCUT
500 eV, PREC Accurate), so a prediction stands for the CHGCAR of those settings:

- the grid is the one VASP would use for the cell (it reproduces the CHGCAR grids of all
  6,059 training-set runs);
- the density is rescaled to the valence electron count of the training POTCARs
  (`pydemi.ml.ZVAL`); the model knows those 87 elements only;
- the error on 605 held-out structures (NMAPE over the full grid) is 0.72 % on average,
  0.58 % median, and up to 10 % for f-electron compounds.

In the interface, the *Crystal structure (ML)* tab predicts the density in the background
with a progress bar and a cancel button; then all three tasks work on the prediction,
which is marked as ML-predicted, and the predicted CHGCAR can be downloaded.

```bash
pydemi predict Si.cif CONTCAR --out-dir predicted               # writes predicted/<name>.CHGCAR
pydemi predict Si.cif --out-dir predicted --featurize --device cuda
```

```python
from pydemi import ml
vd = ml.predict("Si.cif")        # VolumetricData, like read_chgcar; vd.sources records the model
feats = pydemi.featurize(vd)
```

Prediction cost grows with the number of atom-grid point pairs within 4 Å. On one CPU core
it takes about 0.35 ms per pair: a 40³ grid of a two-atom cell (BeCo, 2 million pairs)
took 12 minutes. All cores, and above all a GPU, make it much faster. Memory is about
160 kB per pair in flight; pydemi works through the grid in chunks sized to stay near
2.6 GB on a CPU, and halves the chunk if memory runs out.

If you use predicted densities, please cite ChargE3Net as well as pydemi.

## Property estimates

pydemi ships LightGBM models of 22 properties, trained on the CMS Lab dataset (6,059
structures; train 4,847, validation 606, test 606) with the composition features of
Magpie (Ward et al., npj Comput. Mater. 2, 16028 (2016)) and targets from the Materials
Project (Jain et al., APL Mater. 1, 011002 (2013); CC BY 4.0). For each property the
inputs (descriptors, composition, or both) were chosen on the validation set. On the test
set, for example, the bulk modulus has R² 0.96, the shear modulus 0.93, the Debye
temperature 0.94 and the band gap 0.93; the anisotropy index and the total and ionic
dielectric constants are weak (R² 0.29–0.47). The estimates are not DFT values.

```python
from pydemi import props
p = props.predict(feats, counts=vd.structure.site_counts())   # property -> value (or probability)
```

The models and the scripts that train them are in `property_prediction/`.
`PYDEMI_PROPERTY_MODEL` points pydemi at another model bundle.

## Python and command line

```python
import pydemi

vd = pydemi.read_chgcar("CHGCAR")
feats = pydemi.featurize(vd)     # dict of all 56 descriptors
pydemi.catalogue()               # their metadata: domain, formula, units, range, fallbacks
```

```bash
pydemi featurize CHGCAR --out features.json            # one structure (.json or .csv)
pydemi batch ./runs --glob "*/CHGCAR" --out all.csv     # many structures, in parallel
pydemi catalogue --out catalogue.csv
pydemi sweep CHGCAR --param c2 --range 1.0:2.5:0.05 --out sweep.csv   # shell-radius sensitivity
```

## Descriptors (56)

| domain | n | descriptors |
|---|---|---|
| bonding | 19 | `zeta`, `m1`, `m2`, `sigma_r2`, `f_core`, `f_bond`, `f_int`, `lnf`, `lnf_charge_weighted`, `lap_concentration`, `lap_concentration_valence`, `ellip_bond_bounded_avg`, `f_NCI`, `NCI_attractive`, `sign_lambda2_rho_mean`, `V_spread`, `V_int_min`, `rho_mid_mean`, `rho_mid_std` |
| structural | 19 | `rho_perc_a`, `rho_perc_b`, `rho_perc_c`, `perc_anisotropy`, `n_max`, `n_min`, `n_NNM`, `Q_NNM`, `rho_min`, `rho_min_ratio`, `rho_int_mean`, `T_eigenvalues_t1`, `T_eigenvalues_t2`, `T_eigenvalues_t3`, `charge_FA`, `shannon_entropy`, `fisher_information`, `disequilibrium`, `LMC_complexity` |
| heterogeneity | 18 | `m1_site_std`, `m1_site_range`, `m1_site_max`, `m1_site_min`, `m1_within_element_var`, `m1_between_element_var`, `f_bond_site_std`, `f_bond_site_range`, `f_bond_site_max`, `f_bond_site_min`, `f_bond_within_element_var`, `f_bond_between_element_var`, `zeta_site_std`, `zeta_site_range`, `zeta_site_max`, `zeta_site_min`, `zeta_within_element_var`, `zeta_between_element_var` |

The formulas and their meaning are in `docs/descriptors.pdf` (also a download button in
the interface). The potential is the Hartree potential of the CHGCAR density. Shells and
the `power` partition use tabulated covalent radii shipped with the package.

Not included, on purpose: the ELF and local energy densities (from a CHGCAR, a PAW
pseudo-density, they would rest on a density-only kinetic-energy estimate, not the true
quantities), the unbounded ellipticity and saddle counts (numerically unstable on VASP
grids), and anything that needs a file other than the CHGCAR.

## Speed and memory

Measured on one CPU core (Linux workstation), nearest partition, descriptors plus the plot data:

| CHGCAR | grid | time | peak memory |
|---|---|---|---|
| FeNi3, 6 MB | 56³ | 0.7 s | 0.25 GB |
| Al2Te5, 31 MB | 64×108×128 | 4.1 s | 0.45 GB |
| SnBi2Te4, 350 MB | 216³ | 54 s | 2.6 GB |

The Becke partition takes about ten times longer on large grids.

## Tests

```bash
pip install -e ".[gui,ml,dev]"
pytest -q
```

The GUI tests (including an end-to-end run of the interface) skip when the `gui` extra
is not installed, and the model tests when the `ml` extra is not. The model tests check
that the predictions equal those of the original ChargE3Net code.

## Citing

See `CITATION.cff`. pydemi 0.2.0: doi:10.5281/zenodo.23664506; all versions:
doi:10.5281/zenodo.22995680.

## License

MIT. Author: Shubham Maurya, CMS Lab, IIT Kanpur. The ChargE3Net network in
`pydemi/ml/model.py` is from https://github.com/AIforGreatGood/charge3net (MIT,
`pydemi/ml/LICENSE-charge3net`).
