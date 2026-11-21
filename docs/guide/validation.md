# Validation

pydemi's test suite (346 tests) checks the descriptors against closed forms,
symmetry and invariance. The same tools are in `pydemi.validate` for your own checks.

## Invariance

Every descriptor must be unchanged when the same crystal is described differently.
`pydemi.validate.invariance` builds the transformed copies exactly:

| transformation | function | GaAs: largest relative change over the 54 |
|---|---|---|
| 2 × 1 × 1 supercell (grid tiled, atoms replicated) | `supercell(vd, (2, 1, 1))` | 2.9·10⁻⁹ |
| translation by (5, 7, 11) voxels | `translate(vd, (5, 7, 11))` | 2.9·10⁻⁹ |
| rigid rotation by 30° about z | `rotate(vd, R)` | 4.4·10⁻¹⁵ |

```python
import numpy as np
from pydemi.validate import invariance

f0 = pydemi.featurize(vd)
f1 = pydemi.featurize(invariance.supercell(vd, (2, 1, 1)))
invariance.compare(f0, f1)          # {} : nothing differs beyond rtol 1e-6
```

The largest changes are round-off on `rho_mid_std`, which is ~10⁻⁸ e/Å³ for GaAs because all
its bonds are equivalent. Invariance under supercells is why no extensive quantity can be a
descriptor, and why the within-element variances are 0 (not undefined) when no element has two
sites: that is the value of every supercell.

## Analytic densities

`pydemi.validate.analytic` builds densities with known answers:

- **Slater superposition**, $\rho_i \propto e^{-2\zeta r}$: for one atom in a large box
  $m_1 = 3/(2\zeta)$, $m_2 = 3/\zeta^2$, $\sigma_r^2 = 3/(4\zeta^2)$, `zeta` = 0 exactly (the
  gradient is radial), and the Laplacian is negative for $r < 1/\zeta$;
- **Gaussian superposition**: smooth everywhere, so it measures derivative accuracy and
  convergence order;
- **uniform**: every special case ({doc}`metadata`).

## Symmetry

Tensor descriptors must respect the point group (Neumann's principle): a cubic crystal has
$t_1 = t_2 = t_3 = 1/3$ and `charge_FA` = 0. GaAs gives 0.333333333333 for all three and
`charge_FA` = 3·10⁻¹². Symmetry-equivalent atoms have identical site values, so their site
spreads vanish.

## Plots reproduce values

Each figure of the graphical interface is drawn from reduced data (histograms, per-site
values). A test recomputes every descriptor from its plotted data and checks it against the
value from `featurize`.

## Predicted densities

The `ml` tests check that pydemi's copy of ChargE3Net gives the same predictions as the
original code, independent of how the grid is chunked, and invariant to rotating the crystal,
permuting its atoms and translating it by a lattice vector. The grid pydemi chooses for a cell
reproduces the CHGCAR grids of all 6,059 training runs.
