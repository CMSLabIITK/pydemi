"""
pydemi.core.partition
=====================
Assignment of voxels to atoms (spec §9), behind one interface.

Every scheme yields memory-bounded chunks of (voxel, atom, weight, distance,
direction) pairs, where distance and direction are measured from THAT atom's
nucleus (the periodic image the scheme used) to the voxel. Hard schemes emit
one pair per voxel with weight 1; smooth schemes emit several with weights
summing to 1. A site-restricted sum is then one code path for every scheme,

    X^(i) = sum over pairs with atom i of  w * g(field_k, r_ik, u_ik),

i.e. the indicator 1(i(k) = i) replaced by w_i(r_k).

Schemes
-------
``nearest``    i(k) = argmin_i |r_k - R_i|                        (default)
``power``      i(k) = argmin_i |r_k - R_i|^2 - R_i^2, R_i the covalent radius
``becke``      Becke fuzzy cells, pure geometry
``hirshfeld``  w_i(r) = rho_free_i(r - R_i) / sum_j rho_free_j(r - R_j)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from ..data import covalent_radius
from ..io.base import VolumetricData
from .geometry import NearestAtom, assign_atoms, geometry_of

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class PairChunk:
    voxel: I64          # (M,) flat voxel index
    atom: I64           # (M,) atom index
    weight: F64         # (M,)
    distance: F64       # (M,) Angstrom, voxel to this atom's image
    direction: F64      # (M, 3) unit vector nucleus -> voxel


class Partition:
    scheme = "abstract"

    def __init__(self, n_atoms: int) -> None:
        self.n_atoms = n_atoms

    def pairs(self) -> Iterator[PairChunk]:
        raise NotImplementedError

    def site_sum(self, f: F64) -> F64:
        """sum_k w_i(k) f_k for every atom i."""
        flat = np.asarray(f, dtype=np.float64).ravel()
        out = np.zeros(self.n_atoms)
        for p in self.pairs():
            out += np.bincount(p.atom, weights=p.weight * flat[p.voxel], minlength=self.n_atoms)
        return out


class HardPartition(Partition):
    """Every voxel belongs wholly to one atom (nearest atom or power diagram)."""

    def __init__(self, assignment: NearestAtom, n_atoms: int, scheme: str, chunk: int = 1 << 20):
        super().__init__(n_atoms)
        self.scheme = scheme
        self.assignment = assignment
        self.labels = assignment.atom_index.ravel()
        self.chunk = chunk

    def pairs(self) -> Iterator[PairChunk]:
        dist = self.assignment.distance.ravel()
        direction = self.assignment.direction.reshape(-1, 3)
        n = dist.size
        for s in range(0, n, self.chunk):
            v = np.arange(s, min(s + self.chunk, n), dtype=np.int64)
            yield PairChunk(v, self.labels[v], np.ones(v.size), dist[v], direction[v])

    def site_sum(self, f: F64) -> F64:
        return np.asarray(np.bincount(self.labels, weights=np.asarray(f, np.float64).ravel(),
                                      minlength=self.n_atoms), dtype=np.float64)


def power_radii(vd: VolumetricData) -> F64:
    """Covalent radius of every atom (the power-diagram weights)."""
    return np.array([covalent_radius(s) for s in vd.structure.species], dtype=np.float64)


def partition_of(vd: VolumetricData, scheme: str) -> Partition:
    """The partition ``scheme`` for ``vd``, cached on ``vd``."""
    key = ("partition", scheme)
    if key not in vd.cache:
        n = vd.structure.n_atoms
        if scheme == "nearest":
            vd.cache[key] = HardPartition(geometry_of(vd), n, "nearest")
        elif scheme == "power":
            vd.cache[key] = HardPartition(assign_atoms(vd.shape, vd.structure, power_radii(vd)),
                                          n, "power")
        else:
            raise ValueError(f"partition {scheme!r} needs its builder (see pydemi.core.partition)")
    out: Partition = vd.cache[key]
    return out
