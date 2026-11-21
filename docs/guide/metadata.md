# Metadata and flags

```python
feats, meta = pydemi.featurize(vd, return_metadata=True)
```

The same keys are the metadata columns of `pydemi batch` and the `metadata` object of
`pydemi featurize --out x.json`.

| key | example (GaAs) | meaning |
|---|---|---|
| `n_atoms` | 8 | atoms in the cell |
| `volume` | 189.907 | cell volume (Å³) |
| `grid_shape` | `84x84x84` | the CHGCAR grid |
| `site_counts` | `Ga:4,As:4` | sites per element |
| `n_elements`, `max_sites_per_element` | 2, 4 | for the heterogeneity descriptors |
| `partition` | `nearest` | `becke(k=…,cells=…)` records the image counts of the Becke cells |
| `shells` | `spheres(0.79,1.14)` | sphere factors $a$, $b$ |
| `derivative_backend` | `fft` | or `fd4` etc. |
| `laplacian_method` | `metric` | or `diagonal` |
| `precision` | `float64` | or `float32` |
| `euler_consistency` | −48 | critical-point quality check, below |
| `<name>__flag` | 0 | 1 if that descriptor took its special-case value |
| `sentinels` | (empty) | `name:case;…` for every flagged descriptor |
| `wall_time_s` | 7.2 | time of the call |
| `pydemi_version` | `0.4.0` | |
| `error` | (empty) | the error message of a failed batch row |

## Special cases (sentinels)

Some descriptors are undefined for special inputs: a ratio with nothing in its denominator,
a region with no voxels. Rather than NaN, pydemi returns a documented constant, the value the
descriptor tends to in that limit, and sets the flag. The catalogue lists the cases of every
descriptor (`sentinel_cases`), and each descriptor entry gives them under **Limits**.

A perfectly uniform density in the fcc Cu cell (`pydemi.validate.analytic.uniform`) triggers
20 of them, for example:

| descriptor | value | case |
|---|---|---|
| `zeta` | 0 | `uniform_density`: no gradient, nothing non-radial |
| `lap_concentration_valence` | 1/2 | `uniform_density` |
| `T_eigenvalues_t1`, `_t2`, `_t3` | 1/3 each | `uniform_density`: isotropic |
| `ellip_bond_bounded_avg` | 0 | `no_bond_voxels` |
| `m1_between_element_var` | 0 | `single_element` |

while `shannon_entropy` comes out as 0 and `disequilibrium` as 1 from their formulas, as they
should for a uniform density. On real CHGCARs flags are rare: none of the 54 is set for GaAs
or SbO₂.

## The Euler check

On a periodic cell the critical points of a smooth density obey
$n_{\max} - n_{\rm bond} + n_{\rm ring} - n_{\min} = 0$ (the Euler characteristic of the
3-torus is zero). pydemi counts the extrema with the 26-neighbour stencil and the saddles with
the 14-neighbour (Freudenthal) triangulation. `euler_consistency` is the resulting sum, which
equals the number of extrema whose status depends on the stencil. It is 0 exactly when the
two stencils find the same extrema.

On VASP pseudo-densities it is nonzero for 91 % of the 6,059-structure dataset (median 5 % of
all critical points), set by PAW-pseudized regions and low-amplitude ripple rather than by
coarse grids. It is a quality indicator for `n_max` and `n_min`, not an error. The saddle
counts themselves are not descriptors: they changed by 17 to 20 % on a coarser grid.
