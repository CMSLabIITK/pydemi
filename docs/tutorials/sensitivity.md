# 7. How much do the regions matter?

Fifteen descriptors depend on the core, bond and interstitial regions, and so on the two sphere
factors $a$ (core) and $b$ (outer) of the {doc}`sphere rule <../guide/regions>`. `pydemi
sweep` recomputes them over a range of one factor and shows how much they move.

```bash
pydemi sweep GaAs/CHGCAR --param b --range 1.04:1.24:0.05 --out sweep.csv
```

```python
import numpy as np
import pydemi

vd = pydemi.read_chgcar("GaAs/CHGCAR")
df = pydemi.sensitivity_sweep(vd, b_range=np.arange(1.04, 1.25, 0.05))
```

The real output for GaAs (some columns):

| a | b | f_core | f_bond | f_int | lap_concentration_valence | ellip_bond_bounded_avg | V_int_min (eV) |
|---|---|---|---|---|---|---|---|
| 0.79 | 1.04 | 0.3123 | 0.3355 | 0.3523 | 0.2464 | 0.5045 | −11.32 |
| 0.79 | 1.09 | 0.3123 | 0.3967 | 0.2910 | 0.2464 | 0.5309 | −11.32 |
| 0.79 | **1.14** | 0.3123 | **0.4510** | **0.2368** | 0.2464 | 0.5542 | −11.32 |
| 0.79 | 1.19 | 0.3123 | 0.4953 | 0.1925 | 0.2464 | 0.5637 | −11.32 |
| 0.79 | 1.24 | 0.3123 | 0.5317 | 0.1560 | 0.2464 | 0.5667 | −11.32 |

What it shows:

The sweep returns all fifteen: `f_core`, `f_bond`, `f_int`, `lap_concentration_valence`,
`ellip_bond_bounded_avg`, `V_int_min`, `n_NNM`, `Q_NNM`, `rho_int_mean` and the six
`f_bond` site statistics.

- `f_core`, `lap_concentration_valence`, `n_NNM` and `Q_NNM` depend only on $a$, so they do
  not move with $b$.
- `f_bond` and `f_int` trade charge as the outer spheres grow: by construction, a larger $b$
  moves interstitial charge into the bond region. Their **ranking across materials** is what
  a model uses, and it is far more stable than the values.
- `V_int_min` is the minimum of the potential in the interstitial, which lies far from the
  atoms, so it does not move over this range.

Why the defaults are 0.79 and 1.14: they are the 1st and 99th percentiles of $d/(R_A+R_B)$
over the nearest neighbour of every atom of the 6,059-structure dataset. The core spheres of
bonded atoms then overlap for fewer than 1 % of the atoms, and the outer spheres cover the
whole bond line for all but 1 %. Sweep `--param a` the same way.
