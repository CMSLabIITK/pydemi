# How pydemi works


One call, `pydemi.featurize(vd)`, runs these steps on one CHGCAR. Every intermediate field is
computed once and cached on the `VolumetricData`, so the 54 descriptors share the work.

| step | what is computed | used by |
|---|---|---|
| 1. read | structure from the header; ρ = first data block / V (e/Å³) | everything |
| 2. geometry pass | for every voxel: nearest nucleus (periodic images), distance r, direction û | radial moments, `zeta`, site statistics |
| 3. sphere rule | for every voxel: the atom with the smallest r/R (R covalent radius); core, bond, interstitial masks | region fractions, `n_NNM`, `V_int_min`, `f_bond` sites, … |
| 4. derivatives | ∇ρ, ∇²ρ, Hessian and its eigenvalues λ₁ ≤ λ₂ ≤ λ₃, spectrally (FFT) with the full cell metric | Laplacian, ellipticity, NCI, anisotropy tensor |
| 5. potential | Hartree potential from ρ by one FFT (Poisson) | `V_spread`, `V_int_min` |
| 6. topology | grid maxima and minima (26 neighbours), steepest-ascent basins, percolation by bisection | critical-point census, `Q_NNM`, `rho_perc_*` |
| 7. bonds | first-shell neighbours (within 1.1 × nearest distance), midpoint densities by Fourier interpolation | `rho_mid_*` |
| 8. partition | voxels to atoms (nearest or Becke) for the per-site sums | heterogeneity descriptors |

Then each registered descriptor reads what it needs. The registry knows each descriptor's
domain, field, requirements, units, range, intensivity, special cases and formula: that is
`pydemi.catalogue()`.

## Design rules

- **One file.** Only the CHGCAR. No POTCAR, OUTCAR, AECCAR, ELFCAR or LOCPOT.
- **Intensive and invariant.** Every descriptor is unchanged when the cell is replaced by a
  supercell, translated or rotated ({doc}`validation`).
- **No NaN.** An undefined case returns a documented constant and a flag ({doc}`metadata`).
- **No hidden numbers.** Every cut-off and unit factor is in `pydemi.constants`, and the
  settings of a run are in its metadata.
- **Stable.** Candidate descriptors that changed by more than about 10 % with the grid or the
  derivative method were dropped ({ref}`not-included`).

```{toctree}
:hidden:

regions
partitions
numerics
metadata
validation
performance
```
