# Numerical choices that change descriptor values

Some descriptors depend strongly on how derivatives are discretized. The
defaults below were chosen by measurement; `pydemi convergence` reports, per
structure, which descriptors the grid actually resolves.

## Laplacian (`lnf`, `laplacian_std` and everything built on them)

pydemi uses compact, periodic second-difference stencils with the exact
metric tensor (`method="fd"`, default) or spectral derivatives
(`method="spectral"`). Two common shortcuts give quite different values:
applying a first-derivative routine twice (a stencil of double width, i.e.
the Laplacian of a smoothed density) and one-sided differences on the
boundary planes of a periodic cell. On fcc FeCoNiCr (48^3 grid):

| Laplacian | lnf | laplacian_std |
|---|---|---|
| double-width stencil, one-sided boundaries | 0.159 | 2.41 |
| double-width stencil, periodic | 0.178 | 3.16 |
| pydemi `fd` (default) | 0.206 | 3.63 |
| pydemi `spectral` | 0.211 | 4.13 |

The two converged methods agree to within 0.005; the shortcuts are up to
~25% low, and `fint_over_lnf` inherits the difference. Voxel-count `lnf` also
depends on the method in near-empty regions: the spectral Laplacian's
round-off floor makes its sign noise there, so prefer `fd` (or the
charge-weighted `lnf_rho`) for slabs and porous structures.

## Gradient (`zeta`)

Central differences, periodic. One-sided differences on the boundary planes
change zeta by ~4% on a 48^3 grid, shrinking as the grid gets finer.

## Minimum image

Wrapping fractional differences with `round` is only the minimum image for
near-orthogonal cells; in strongly sheared cells it overestimates distances
by >0.1 A (see `tests/test_geometry.py`). pydemi uses a periodic KD-tree
with a provably sufficient image range.

## Division

Zero denominators give NaN; no epsilon is added.
