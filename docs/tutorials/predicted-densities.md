# 4. Densities without DFT

When there is no CHGCAR, pydemi can predict one from the crystal structure alone with
**ChargE3Net** (Koker et al., *npj Comput. Mater.* **10**, 161 (2024); MIT licence), an
E(3)-equivariant graph network. pydemi ships the network and a checkpoint that CMS Lab
fine-tuned on its own VASP dataset (PAW-PBE, ENCUT 500 eV, PREC Accurate), so a prediction
stands for the CHGCAR of those settings.

```bash
pip install "pydemi[ml]"     # PyTorch, e3nn, ASE; see the installation page for CPU/GPU builds
```

## What the model knows

- **The grid** is the one VASP would use for the cell at ENCUT 500 eV; it reproduces the
  CHGCAR grids of all 6,059 training runs. `--grid n1,n2,n3` overrides it, and `--encut` sets
  another cut-off.
- **The electrons**: the density is rescaled to the valence electron count of the training
  POTCARs (`pydemi.ml.ZVAL`; `pydemi.ml.n_valence(structure)`). The model knows those 87
  elements only; `pydemi.ml.unsupported(structure)` lists any others.
- **The error** on the 605 held-out structures of the density model (NMAPE over the full
  grid): 0.72 % on average, 0.58 % median, up to 10 % for f-electron compounds.

## Predict

```bash
pydemi predict GaAs.vasp --out-dir predicted --featurize --device cpu
```

```text
wrote predicted/GaAs.CHGCAR (grid 84x84x84, scale 0.99938261, 1348.1 s)
```

`scale` is the factor that brought the raw prediction to the 32 valence electrons of GaAs:
the network alone was 0.06 % off. `--featurize` also writes `predicted/GaAs_descriptors.json`.

```python
import pydemi
from pydemi import ml

vd = ml.predict("GaAs.vasp")               # VolumetricData, like read_chgcar
vd.sources                                 # records the model and its settings
feats = pydemi.featurize(vd)
pydemi.write_volumetric("GaAs_pred.CHGCAR", vd.structure, [vd.rho.data])
```

**Time.** This 8-atom cell on its 84³ grid (6.7 million atom–point pairs within 4 Å) took
22 minutes on the CPUs of a workstation (about 50 cores busy). A GPU is much faster, and
`--grid 40,40,40` gives a quick preview. See {doc}`../guide/performance`.

## How close is it? GaAs, DFT against predicted

GaAs is one of the held-out structures of the property models. Its prediction against the
VASP CHGCAR on the same grid:

- **density error (NMAPE): 0.40 %**, better than the average of the test set;
- **descriptors**: 37 of the 54 within 1 %, 6 differ by more than 5 %; median difference
  0.22 %.

`````{dropdown} All 54 descriptors from both densities
The last column is how well each descriptor transfers to predicted densities in general:
the Spearman correlation between its DFT and predicted values over the 605 test structures
(excellent ≥ 0.99, good ≥ 0.95, fair ≥ 0.8). "≈ 0 in both" marks values that are zero up to
noise, where a relative difference means nothing.

```{include} ../_generated/md/gaas_ml.md
```
`````

The pattern is general. Descriptors built from the density itself (moments, region
fractions, potential, percolation, information measures) transfer almost exactly. Those built
from its second derivatives (the Laplacian, ellipticity, NCI) and from counting grid extrema
(`n_min`, `n_NNM`) are more sensitive to small ripples in the prediction. `rho_min`, which
lies in the PAW-pseudized core, differs by 8 %.

## Properties from a predicted density

```python
from pydemi import props
props.predict(pydemi.featurize(vd))
```

```{include} ../_generated/md/gaas_ml_props.md
```

For GaAs the two sets of estimates agree to about the typical error of the models: the
largest change is the bulk modulus, 10 GPa or about one MAE. Against the DFT values the
picture is mixed. The predicted density brings the bulk modulus, Debye temperature, sound
velocities and refractive index closer, and moves Young's modulus and C11 further away. That
is one material. On the 605 held-out structures the pydemi paper found the two routes nearly
equally accurate, which is the point: **no DFT calculation is needed** to get the
descriptors and the estimates.

## In the interface

The *Structure (ML)* input tab predicts the density of an uploaded CIF or POSCAR in the
background, with a progress bar and a cancel button; every view then works on the
prediction, which is marked as ML-predicted, and the CHGCAR can be downloaded. The *Compare*
view does what this page did for any open CHGCAR ({doc}`gui`).

If you use predicted densities, please cite ChargE3Net as well as pydemi
({doc}`../project/citing`).
