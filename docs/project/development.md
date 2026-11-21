# Development

## Set up

```bash
git clone https://github.com/CMSLabIITK/pydemi.git
cd pydemi
pip install -e ".[gui,ml,dev]"
pytest -q
```

The GUI tests (including an end-to-end run of the interface) skip when the `gui` extra is not
installed, and the model tests when the `ml` extra is not.

## Layout

| path | contents |
|---|---|
| `src/pydemi/io` | CHGCAR reader and writer, `Structure`, `Lattice`, `Grid`, `VolumetricData` |
| `src/pydemi/core` | geometry pass and sphere rule, derivatives, partitions |
| `src/pydemi/fields` | density fields and the Hartree potential |
| `src/pydemi/operators` | moments, fractions, Laplacian statistics, anisotropy tensor, topology, site statistics |
| `src/pydemi/descriptors` | the registry and the 54 descriptors (`bonding`, `structural`, `heterogeneity`) |
| `src/pydemi/validate` | invariance, analytic densities, convergence and sensitivity |
| `src/pydemi/ml` | ChargE3Net, the fine-tuned checkpoint, VASP grids |
| `src/pydemi/props.py` | property-model bundles |
| `src/pydemi/gui` | the Streamlit interface and its plots |
| `docs/` | this website and the descriptor reference (`descriptors.tex`) |
| `property_prediction/` | the scripts behind the property models and the paper |

## Adding a descriptor

A descriptor is a function of a `VolumetricData` registered with its metadata. Use the
cached fields (`geometry`, `masks`, `field_derivatives`, …) rather than recomputing them,
return a `Sentinel` for an undefined case, and add:

- its entry in `docs/descriptors.tex` (the website and the PDF follow);
- a figure in `pydemi/gui/plots.py` whose data reproduce the value;
- tests: analytic value if there is one, invariance, and the special cases.

## This website

The site is built with Sphinx and the Furo theme from `docs/`:

```bash
pip install -r docs/requirements.txt -e .
sphinx-build -b html docs docs/_build/html
```

The descriptor pages are generated on every build from `docs/descriptors.tex`
(`docs/tools/descriptors_to_myst.py`), so the PDF and the website say the same thing. The
interactive figures, example values and screenshots come from real CHGCARs and are committed
under `docs/_generated` and `docs/_static/screenshots`. To remake them (needs the dataset):

```bash
python docs/tools/make_figures.py /path/to/dataset
python docs/tools/make_numerics.py /path/to/dataset
pydemi gui --no-browser --port 8611 &
python docs/tools/make_screenshots.py /path/to/CHGCAR http://localhost:8611
```

A push to `main` that touches `docs/` or `src/` rebuilds the site and publishes it to GitHub
Pages (`.github/workflows/docs.yml`).

## Releases

Version in `src/pydemi/__init__.py`, an entry in `CHANGELOG.md`, then a GitHub release: the
`publish` workflow builds and uploads to PyPI, and Zenodo archives the release with a DOI.
