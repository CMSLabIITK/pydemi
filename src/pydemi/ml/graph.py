"""Periodic neighbour lists for ChargE3Net: atom-atom and atom-probe edges within the cutoff.

NumPy and SciPy only. Atoms are replicated over every image cell that can lie within
the cutoff of the home cell, and a KD-tree of the replicas answers the queries; the
same construction as the KD-tree graph of the original code.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from pydemi.io.base import Structure


@dataclass
class Replicas:
    """The atoms of ``structure`` and their periodic images within ``cutoff`` of the cell."""

    positions: NDArray[np.float64]   # (n_rep, 3) Cartesian
    atom: NDArray[np.int64]          # (n_rep,) index of the home atom
    tree: Any

    @classmethod
    def build(cls, structure: Structure, cutoff: float) -> "Replicas":
        A = structure.lattice.matrix
        home = (structure.frac_coords % 1.0) @ A
        reps = [int(math.ceil(cutoff / h)) for h in structure.lattice.heights]
        offsets = np.array(list(itertools.product(*(range(-n, n + 1) for n in reps))), dtype=float)
        positions = (home[:, None, :] + (offsets @ A)[None, :, :]).reshape(-1, 3)
        atom = np.repeat(np.arange(len(home)), len(offsets))
        return cls(positions, atom, cKDTree(positions))


Edges = Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]


def _edges(rep: Replicas, points: NDArray[np.float64], cutoff: float, skip_self: bool) -> Edges:
    """(src atom, dst point, vector point - atom image) for every pair closer than ``cutoff``."""
    hits = rep.tree.query_ball_point(points, r=cutoff, return_sorted=False)
    counts = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
    image = np.fromiter(itertools.chain.from_iterable(hits), dtype=np.int64, count=int(counts.sum()))
    dst = np.repeat(np.arange(len(points)), counts)
    vec = points[dst] - rep.positions[image]
    if skip_self:  # an atom is not its own neighbour (its periodic images are)
        keep = np.einsum("ij,ij->i", vec, vec) > 1e-16
        image, dst, vec = image[keep], dst[keep], vec[keep]
    return rep.atom[image], dst, vec


def atom_edges(rep: Replicas, structure: Structure, cutoff: float) -> Edges:
    """Edges between atoms: (src, dst, r_dst - r_src)."""
    home = (structure.frac_coords % 1.0) @ structure.lattice.matrix
    return _edges(rep, home, cutoff, skip_self=True)


def probe_edges(rep: Replicas, probes: NDArray[np.float64], cutoff: float) -> Edges:
    """Edges from atoms to probe points: (atom, probe, r_probe - r_atom)."""
    return _edges(rep, probes, cutoff, skip_self=False)
