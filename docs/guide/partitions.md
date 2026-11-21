# Atomic partitions

The 18 heterogeneity descriptors summarise per-atom values: the radius of each atom's charge
cloud $m_1^{(i)}$, the share of its charge in its bond region $f_{\rm bond}^{(i)}$, and its
asphericity $\zeta^{(i)}$. That needs a rule for which voxels belong to which atom.

| `partition` | rule | weights |
|---|---|---|
| `"nearest"` (default) | each voxel to its nearest nucleus: Voronoi cells | 0 or 1 |
| `"becke"` | Becke fuzzy cells, pure geometry (no size adjustment) | smooth, summing to 1 |

The partition applies to the **12 site descriptors of `m1` and `zeta`**. The six `f_bond` site
descriptors always use the cells of the sphere rule (each voxel to the atom with the smallest
$r/R$), because the sphere that puts a voxel in the bond region also names its atom; the bond
regions of the atoms then add up exactly to that of the cell.

## How much it matters

For SbO₂ the two partitions give identical values for the 42 descriptors that do not use a
partition, and different values for all 12 site statistics of `m1` and `zeta` (in GaAs the
within-element variances stay 0 under both, since all Ga and all As sites are equivalent). A few of the
changes (relative to the nearest-atom values):

| descriptor | GaAs | SbO₂ |
|---|---|---|
| `m1_site_std` | 33 % | 12 % |
| `m1_between_element_var` | 78 % | 30 % |
| `zeta_site_std` | 16 % | 27 % |
| `zeta_between_element_var` | 35 % | 77 % |

So the choice is part of the definition: compare values only within one partition. The
property models use `nearest`.

## Becke cells in a solid

Becke's weights were built for molecules: the cell of atom $i$ is a product over all
competitors $j$. In a solid the number of competitors at distance $R$ grows as $R^2$ while each
factor approaches 1 only slowly, so the product must run over many periodic images (FeNi₃: a
largest weight error of 1.6·10⁻² with 60 competitors, 1.0·10⁻³ with 300). pydemi uses a fixed
number of nearest images, recorded in the metadata as part of the definition. On large grids
Becke takes about ten times longer than `nearest`.
