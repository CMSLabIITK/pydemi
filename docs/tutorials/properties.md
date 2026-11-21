# 5. Property estimates

pydemi ships LightGBM models of 17 properties that read the **54 descriptors alone**, with no
composition or structure features. They are the models of the pydemi paper.

```bash
pip install "pydemi[props]"     # macOS: brew install libomp
```

```python
import pydemi
from pydemi import props

feats = pydemi.featurize(pydemi.read_chgcar("GaAs/CHGCAR"))
est = props.predict(feats)            # property -> value, or probability for a classification
props.outside_range(feats)            # descriptors outside the 1st-99th percentile of training
```

## The models

Trained on the 6,059-structure dataset of Kumar et al., *J. Chem. Inf. Model.* **66**, 7113
(2026) (train 4,847, validation 606, test 606), with targets from that dataset and from the
Materials Project (Jain et al., *APL Mater.* **1**, 011002 (2013), CC BY 4.0). Test scores are
the mean over five seeds; the MAE is the typical error shown in the interface.

```{include} ../_generated/md/property_models.md
```

The number of materials differs between properties because not every material has every
value: the elastic tensor needs a mechanically stable calculation, the direct-gap model
covers only materials with a gap, and the refractive index is known for the fewest.

## A worked example: GaAs

GaAs (`mp-2534`) is in the **test** set: the models never saw it during training. Its
estimates against its DFT values:

```{include} ../_generated/md/gaas_properties.md
```

The elastic moduli are within the typical error. The PBE band gap is overestimated by about
twice the typical error, and the direct-gap classifier calls it indirect (probability of
direct 0.37), while PBE gives a direct gap at Γ. The interface also flags that one descriptor
of GaAs, `sign_lambda2_rho_mean`, lies outside the central 98 % of the training materials,
which is a reason for caution.

```{important}
The estimates are not DFT values. The models were trained on densities from the PAW
datasets of the CMS Lab VASP calculations. For densities from other PAW datasets (for
example those of the Materials Project), the descriptors shift and the estimates can be less
accurate.
```

## Inside the bundle

```python
b = props.load()                      # the shipped bundle (cached)
b.name, b.split                       # 'v2 (tuned LightGBM, descriptors only)', 'train 4847 / val 606 / test 606'
t = b.targets["bulk_modulus"]
t.units, t.score, t.error             # 'GPa', ('R²', 0.951), 9.54
t.importance                          # descriptor -> mean |SHAP| on the test set
b.match(feats)                        # the dataset material with these descriptors, if any
```

`PYDEMI_PROPERTY_MODEL=/path/to/bundle` points pydemi at another bundle, for example one you
trained with the scripts in `property_prediction/` of the repository.
