# Installation

pydemi runs on your own computer: Linux, macOS or Windows, Python 3.10 or newer. It is on
PyPI; pick the extras you need.

| command | what you get |
|---|---|
| `pip install pydemi` | library and command line (numpy, scipy, pandas) |
| `pip install "pydemi[props]"` | … plus the property estimates (LightGBM) |
| `pip install "pydemi[gui]"` | … plus the graphical interface (Streamlit, Plotly, LightGBM) |
| `pip install "pydemi[gui,ml]"` | … plus densities predicted from a structure (PyTorch, e3nn, ASE) |
| `pip install "pydemi[full]"` | … plus pymatgen `Structure` in and out |

```{tip}
A separate environment keeps it tidy:

`````{tab-set}
````{tab-item} venv
```bash
python -m venv pydemi-env
source pydemi-env/bin/activate        # Windows: pydemi-env\Scripts\activate
pip install "pydemi[gui]"
```
````
````{tab-item} conda
```bash
conda create -n pydemi python=3.12
conda activate pydemi
pip install "pydemi[gui]"
```
````
`````
```

## The `ml` extra and PyTorch

The `ml` extra pulls in PyTorch, which is large. On a machine without a GPU, install the CPU
build first, it is much smaller:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "pydemi[gui,ml]"
```

With an NVIDIA GPU, install the CUDA build of PyTorch that matches your driver first (see
[pytorch.org](https://pytorch.org)); pydemi then uses the GPU. If PyTorch reports that the
driver is too old, the installed PyTorch was built for a newer CUDA than your driver supports:
install an older CUDA build of PyTorch, or run on the CPU with `--device cpu`.

## macOS

The property models (LightGBM) need the OpenMP runtime:

```bash
brew install libomp
```

Without it, `pydemi.props.load_error()` reports the failed import and the interface hides the
*Properties* view.

## From source

```bash
git clone https://github.com/CMSLabIITK/pydemi.git
cd pydemi
pip install -e ".[gui,ml,dev]"
pytest -q
```

## Check the installation

```bash
python -c "import pydemi; print(pydemi.__version__, len(pydemi.descriptor_names()))"
```

prints `{{ version }} 54`.
