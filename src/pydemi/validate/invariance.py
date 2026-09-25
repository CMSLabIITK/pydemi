"""
pydemi.validate.invariance
==========================
Transformations under which every descriptor must be unchanged (spec §10):

* :func:`supercell` -- the cell replaced by an (n1, n2, n3) supercell of the
  same material: every grid is tiled exactly and the atoms are replicated.
  Intensive descriptors are unchanged; extensive ones scale, which is why
  none may be registered.
* :func:`translate` -- structure and grid translated together by a whole
  number of voxels (a rigid translation that keeps the sampling exact).
* :func:`rotate` -- the cell rigidly rotated: fractional coordinates and grid
  values unchanged, Cartesian frame rotated. Scalar descriptors are
  invariant; the anisotropy tensor itself is not, its sorted eigenvalues are.

:func:`compare` lists the descriptors that differ between two feature dicts.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ..io.base import FloatArray, Grid, Lattice, Structure, VolumetricData


def _map_grids(vd: VolumetricData, fn: Callable[[FloatArray], FloatArray],
               lattice: Lattice, structure: Structure) -> VolumetricData:
    def g(grid: Optional[Grid]) -> Optional[Grid]:
        return None if grid is None else Grid(fn(grid.data), lattice)
    mv = None
    if vd.magnetization_vector is not None:
        a, b, c = (Grid(fn(x.data), lattice) for x in vd.magnetization_vector)
        mv = (a, b, c)
    rho = g(vd.rho)
    assert rho is not None
    return dataclasses.replace(vd, structure=structure, rho=rho, magnetization=g(vd.magnetization),
                               elf=g(vd.elf), potential=g(vd.potential),
                               core_density=g(vd.core_density), magnetization_vector=mv, cache={})


def supercell(vd: VolumetricData, reps: Sequence[int] = (2, 2, 2)) -> VolumetricData:
    """Exact (n1, n2, n3) supercell: grids tiled, atoms replicated in lattice order."""
    r = np.array([int(x) for x in reps])
    s = vd.structure
    lattice = Lattice(s.lattice.matrix * r[:, None])
    shifts = np.array([(i, j, k) for i in range(r[0]) for j in range(r[1]) for k in range(r[2])],
                      dtype=np.float64)
    frac = ((s.frac_coords[None, :, :] + shifts[:, None, :]) / r).reshape(-1, 3)
    species = [e for _ in range(len(shifts)) for e in s.species]
    structure = Structure(lattice, species, frac)
    return _map_grids(vd, lambda a: np.tile(a, tuple(r)), lattice, structure)


def translate(vd: VolumetricData, voxels: Sequence[int]) -> VolumetricData:
    """Translate structure and grid together by ``voxels`` grid steps along a1, a2, a3."""
    v = np.array([int(x) for x in voxels])
    s = vd.structure
    structure = Structure(s.lattice, list(s.species),
                          (s.frac_coords + v / np.array(vd.shape)) % 1.0)
    return _map_grids(vd, lambda a: np.roll(a, tuple(v), axis=(0, 1, 2)), s.lattice, structure)


def rotation(axis: Sequence[float], angle_deg: float) -> NDArray[np.float64]:
    """Rotation matrix about ``axis`` by ``angle_deg`` (Rodrigues)."""
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    t = math.radians(angle_deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.asarray(np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K, dtype=np.float64)


def rotate(vd: VolumetricData, R: NDArray[np.float64]) -> VolumetricData:
    """Rigidly rotate the cell: every lattice vector a_i -> R a_i."""
    s = vd.structure
    lattice = Lattice(s.lattice.matrix @ np.asarray(R).T)
    structure = Structure(lattice, list(s.species), s.frac_coords.copy())
    return _map_grids(vd, lambda a: a, lattice, structure)


def compare(a: Mapping[str, float], b: Mapping[str, float], rtol: float = 1e-6,
            atol: float = 1e-12) -> dict[str, tuple[float, float]]:
    """{name: (a, b)} for every descriptor where |a - b| > atol + rtol max(|a|, |b|)."""
    out: dict[str, tuple[float, float]] = {}
    for k in a:
        x, y = float(a[k]), float(b[k])
        if math.isnan(x) and math.isnan(y):
            continue
        if not abs(x - y) <= atol + rtol * max(abs(x), abs(y)):
            out[k] = (x, y)
    return out
