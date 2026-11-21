# Core, bond and interstitial regions

Many descriptors ask *where* the charge is: near the nuclei, in the bonds, or in the space
between the atoms. pydemi answers with one geometric rule, the **sphere rule**, scaled by the
size of each atom.

```{figure} ../_static/regions.svg
:width: 300px
:align: center

Core and outer spheres of a large atom $i$ and a small atom $j$.
```

Around every nucleus $i$ lie a core sphere of radius $aR_i$ and an outer sphere of radius
$bR_i$, with $R_i$ the covalent radius of the element (Cordero et al., *Dalton Trans.* 2832
(2008), shipped with the package) and

$$
a = 0.79,\qquad b = 1.14 .
$$

With $s_k = \min_i r_{ik}/R_i$, the distance from voxel $k$ to the nearest nucleus in units of
that atom's radius:

| region | condition | meaning |
|---|---|---|
| core | $s_k \le a$ | inside some core sphere |
| bond | $a < s_k \le b$ | inside an outer sphere but in no core sphere |
| interstitial | $s_k > b$ | outside every outer sphere |

## Why these factors

A bond of length $d$ between atoms $A$ and $B$ has $d \approx R_A + R_B$. Over the nearest
neighbour of every atom of the 6,059-structure dataset, $d/(R_A+R_B)$ has median 0.95; its 1st
and 99th percentiles are 0.79 and 1.14. So:

- the core spheres of bonded atoms overlap for fewer than 1 % of the atoms;
- the outer spheres cover the whole bond line for all but 1 %.

Because the spheres scale with the atoms, the regions mean the same thing for H and for Cs: a
large atom does not by itself raise the interstitial fraction.

## Which descriptors use the regions

`f_core`, `f_bond`, `f_int`, `lap_concentration_valence`, `ellip_bond_bounded_avg`,
`V_int_min`, `n_NNM`, `Q_NNM`, `rho_int_mean` and the six `f_bond` site statistics: fifteen
in all. {doc}`../tutorials/sensitivity` shows how they move with $a$ and $b$.

## Changing them

```python
pydemi.featurize(vd, shells=(0.75, 1.20))
```

```bash
pydemi featurize CHGCAR --out f.json --shells 0.75,1.20
```

The setting is recorded as `shells: spheres(0.75,1.20)` in the metadata. The shipped property
models were trained with the defaults.

```{note}
Earlier versions (before 0.4.0) used shells at fixed distances (0.8 / 1.5 Å) from the nearest
nucleus. Those are gone: the sphere rule is the only region definition.
```
