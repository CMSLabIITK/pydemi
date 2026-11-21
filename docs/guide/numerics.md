# Numerical settings

## Derivatives

The gradient $\nabla\rho$, the Laplacian $\nabla^2\rho$ and the Hessian are taken on the
periodic, possibly non-orthogonal grid. With $A$ the lattice vectors as rows and
$B = (A^{-1})^{T}$ the reciprocal vectors (without $2\pi$), in fractional coordinates $u$:

$$
\nabla f = \sum_a \mathbf b_a\,\frac{\partial f}{\partial u_a},\qquad
\nabla^2 f = \sum_{a,b} G_{ab}\,\frac{\partial^2 f}{\partial u_a\,\partial u_b},\quad G = BB^{T},\qquad
H_{\rm cart} = B^{T} H_{\rm frac} B .
$$

| `derivative_backend` | method | notes |
|---|---|---|
| `"fft"` (default) | spectral: $\partial/\partial u_a \to 2\pi i m_a$ | exact for a plane-wave density; the Nyquist term of an even axis is dropped from odd orders so the results stay real and the Hessian symmetric |
| `"fd"` | central finite differences, `fd_order` 2, 4 (default), 6 or 8 | mixed derivatives apply the first-derivative stencil along both axes |

Switching GaAs and SbO₂ to fourth-order finite differences changed no descriptor by more than
1.2 % (`sign_lambda2_rho_mean` of SbO₂); all others by less than 1 %.

## The Laplacian on a skewed cell

`laplacian_method="metric"` (default) keeps the off-diagonal metric terms $G_{ab}$, $a\neq b$.
`"diagonal"` drops them, a common shortcut that is exact only for orthogonal cells. It exists
so that its error can be measured, and the error is large:

| descriptor | GaAs (cubic) | SbO₂ (monoclinic) |
|---|---|---|
| `lnf` | 0 | 7.6 % |
| `lnf_charge_weighted` | 0 | 6.7 % |
| `lap_concentration_valence` | 0 | 416 % |

Keep the default.

## Grid

The descriptors are computed on the grid of the CHGCAR. `pydemi.validate.convergence.
grid_convergence` Fourier-resamples the density to coarser grids and reports the relative
change of every descriptor:

```python
from pydemi.validate.convergence import grid_convergence
table = grid_convergence(vd, scales=(0.9, 0.8))     # points per axis: 90 % and 80 %
table.sort_values("rel_change_x0.8", ascending=False).head(10)
```

For SbO₂ (72 × 80 × 96), 47 of the 54 descriptors changed by less than 1 % on the 80 % grid;
the largest change was 3.7 % (`perc_anisotropy`). The descriptors that move most on coarse
grids are the per-site statistics of small spreads and the values that are zero by symmetry,
whose *relative* change is not meaningful (in GaAs, `perc_anisotropy` is exactly 0 and
`charge_FA` is 3·10⁻¹²). Look at the absolute change for those.

`pydemi.validate.convergence.analytic_convergence` and `recommended_spacing` compare the
radial moments with their closed forms for a Slater density at several grid spacings, and
give the coarsest spacing that meets a tolerance.

## Precision

`float32=True` casts every grid to single precision first and halves the memory of the
derivative fields. For GaAs and SbO₂ it changed the descriptors by a median of
2·10⁻⁹ and 1·10⁻⁸ (relative). Only values that are zero up to round-off moved visibly
(`charge_FA` of cubic GaAs, 3·10⁻¹²). The default is float64.

## Units

Internally lengths are in Å, densities in e/Å³ and potentials in eV. Quantities defined in
atomic units (the NCI thresholds and the information measures) are converted at the point of
use, with CODATA 2018 factors from `pydemi.constants`:
1 e/bohr³ = 6.748 e/Å³.
