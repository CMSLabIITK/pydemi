# Numerical choices that change descriptor values

Some descriptors depend strongly on how derivatives are discretized. The
defaults below were chosen by measurement; `pydemi convergence` reports, per
structure, which descriptors the grid actually resolves.

## Laplacian (`lnf`, `laplacian_std` and everything built on them)

pydemi uses compact, periodic second-difference stencils with the exact
metric tensor (`method="fd"`, default) or spectral derivatives
(`method="spectral"`). Two common shortcuts change the values: applying a
first-derivative routine twice (a stencil of double width, i.e. the
Laplacian of a smoothed density) and one-sided differences on the boundary
planes of a periodic cell. Absolute relative difference to the spectral
result over 200 random VASP CHGCARs (grid spacing 0.060-0.070 A):

| Descriptor | Discretization | Median | 95th pct. | Max. |
|---|---|---|---|---|
| `lnf` (voxel count) | double width, one-sided | 2.0% | 8.8% | 20.8% |
| | double width, periodic | 1.0% | 3.0% | 6.9% |
| | compact, periodic (`fd`) | 0.2% | 1.6% | 31.5% |
| `lnf_rho` (charge weighted) | double width, one-sided | 1.0% | 2.9% | 10.3% |
| | double width, periodic | 0.7% | 2.0% | 3.6% |
| | compact, periodic (`fd`) | 0.1% | 0.6% | 1.9% |

Voxel-count `lnf` is the sensitive one: in flat interstitial regions many
voxels have a Laplacian near zero and their sign flips with small
discretization differences (YNi3: `lnf` 0.111 with `fd`, 0.084 spectral;
`lnf_rho` 0.511 vs 0.502). Prefer `lnf_rho`. In near-empty regions the
spectral Laplacian's absolute round-off floor makes its sign noise (isolated
Gaussian: exact `lnf` 0.0079, `fd` 0.0079, spectral 0.072), so prefer `fd`
or `lnf_rho` for slabs and porous structures. The scripts behind these
numbers are in `paper/analysis/`.

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
