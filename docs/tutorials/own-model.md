# 6. Your own model

The descriptors are an ordinary fixed-length feature vector, so any tabular learner works.
This is the recipe behind the shipped models, scaled down. You need a table of descriptors
({doc}`dataset`) and a table of your target with the same structure names.

```python
import pandas as pd
import pydemi
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

X = pd.read_csv("all.csv")                      # from: pydemi batch runs --out all.csv
X = X[X.error.isna()]
X["structure"] = X.path.str.split("/").str[-2]  # runs/<structure>/CHGCAR
y = pd.read_csv("my_property.csv")              # columns: structure, value

data = X.merge(y, on="structure")
names = pydemi.descriptor_names()
Xtr, Xte, ytr, yte = train_test_split(data[names], data["value"], test_size=0.1, random_state=0)

model = LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=31,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
model.fit(Xtr, ytr)
p = model.predict(Xte)
print(r2_score(yte, p), mean_absolute_error(yte, p))
```

## Which descriptors matter?

```python
import shap

sv = shap.TreeExplainer(model).shap_values(Xte)
imp = pd.Series(abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
imp.head(10)
```

Each descriptor has a physical meaning, so the ranking can be read: for instance a model of
the shipped bulk-modulus model leans most on `rho_perc_c`, `rho_mid_mean`, `rho_int_mean` and
`f_int`: how well the charge connects through the crystal, how much of it sits in the bonds,
and how densely the space between the atoms is filled. Look up any name in
{doc}`../descriptors/index`. In the pydemi paper, ten descriptors per property were almost as
good as all 54.

## Good practice

- **Split by material, not by row**, if your table holds several entries per material.
- **One PAW dataset** for all densities: the POTCAR sets the valence electrons, and the
  descriptors follow it.
- **Keep the defaults** (`shells`, `partition`, derivatives) or record what you changed: the
  metadata columns of the batch table do this for you.
- **Check the flags**: `X.filter(like="__flag").sum()` shows how often a descriptor fell back
  to its special-case value. Such a value is valid, but it means "not applicable", which a
  model should be able to see.
