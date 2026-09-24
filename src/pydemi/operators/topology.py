"""
pydemi.operators.topology
=========================
Grid topology by direct operations that always terminate (spec §7, §8.2):
no critical-point search, no Newton iteration.

Extremum census
---------------
Ties are broken by flat voxel index (symbolic perturbation), so every
comparison is strict.

* Maxima and minima: 26-neighbour comparison with modular wrapping -- a
  voxel higher (lower) than all 26 neighbours.
* Saddles: the specification asks for a 26-neighbour census but a saddle
  cannot be classified consistently on the 26-neighbour shell, which is not
  a triangulated sphere (on cos 2 pi x + cos 2 pi y + cos 2 pi z every
  adjacency on that shell miscounts the six saddles). Saddle multiplicities
  are therefore taken from the lower / upper link of the Freudenthal
  triangulation of the grid (14 neighbours), whose link IS a triangulated
  2-sphere:  n_saddle1 += components(lower link) - 1,
             n_saddle2 += components(upper link) - 1.
* euler_consistency = n_max - n_saddle2 + n_saddle1 - n_min. With the
  Freudenthal extrema this sum is identically 0 (chi(T^3) = 0, Banchoff);
  with the 26-neighbour extrema it is 0 exactly when both neighbourhoods
  find the same extrema, i.e. when the grid resolves every extremum, and
  nonzero otherwise -- the grid-adequacy flag the specification asks for.

Basins
------
Steepest ascent over the 26 neighbours (slope = value difference / distance);
every basin ends at a 26-neighbour maximum. Pointer jumping, so it
terminates in O(log N) sweeps.

Percolation
-----------
For a level c, the super-level set {f > c} is labelled with
``scipy.ndimage.label`` (6-connectivity); labels are merged across opposite
faces with a union-find that tracks each cluster's lattice offset, so a
cluster spans direction alpha when, after the periodic merge, it meets its
own image with a winding vector w_alpha != 0. (A cluster that merely touches
both faces -- a blob across the periodic boundary -- does not span; the
winding test makes the level independent of where the cell origin lies.)
The spanning level is found by bisection over the sorted voxel values:
rho_perc_alpha is the HIGHEST level whose super-level set still spans alpha
(documented correction: the lowest such level is always the density minimum,
at which the whole cell spans).
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]

#: the 26 neighbour offsets
NEIGHBOURS_26 = np.array([o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)],
                         dtype=np.int64)
_POS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                dtype=np.int64)
#: the 14 edge vectors of the Freudenthal triangulation
FREUDENTHAL_14 = np.vstack([_POS, -_POS])


def _shift(a: NDArray[Any], off: NDArray[np.int64]) -> NDArray[Any]:
    """``a`` at voxel k + off (periodic)."""
    return np.roll(a, shift=tuple(-int(o) for o in off), axis=(0, 1, 2))


def lower_mask(values: NDArray[Any], offsets: NDArray[np.int64]) -> I64:
    """Bit k set when neighbour ``offsets[k]`` is lower, (value, flat index) lexicographically."""
    idx = np.arange(values.size, dtype=np.int64).reshape(values.shape)
    mask = np.zeros(values.shape, dtype=np.int64)
    for k, off in enumerate(offsets):
        vn, jn = _shift(values, off), _shift(idx, off)
        mask |= (((vn < values) | ((vn == values) & (jn < idx))).astype(np.int64) << k)
    return mask


@lru_cache(maxsize=1)
def _link_components() -> NDArray[np.int8]:
    """components[mask] of the Freudenthal link vertices selected by ``mask`` (2^14 entries)."""
    edges = {tuple(o) for o in FREUDENTHAL_14.tolist()}
    n = len(FREUDENTHAL_14)
    adj = [[j for j in range(n) if j != i
            and tuple((FREUDENTHAL_14[j] - FREUDENTHAL_14[i]).tolist()) in edges] for i in range(n)]
    out = np.zeros(1 << n, dtype=np.int8)
    for mask in range(1 << n):
        seen, count = 0, 0
        for start in range(n):
            if not (mask >> start) & 1 or (seen >> start) & 1:
                continue
            count += 1
            stack = [start]
            seen |= 1 << start
            while stack:
                v = stack.pop()
                for w in adj[v]:
                    if (mask >> w) & 1 and not (seen >> w) & 1:
                        seen |= 1 << w
                        stack.append(w)
        out[mask] = count
    return out


def extremum_census(values: NDArray[Any]) -> dict[str, Any]:
    """Counts n_max, n_min (26-neighbour), n_saddle1, n_saddle2 (Freudenthal link),
    euler_consistency, plus the boolean map of maxima and the 26-neighbour lower mask."""
    m26 = lower_mask(values, NEIGHBOURS_26)
    full26 = (1 << 26) - 1
    is_max = m26 == full26
    n_max, n_min = int(is_max.sum()), int((m26 == 0).sum())
    m14 = lower_mask(values, FREUDENTHAL_14)
    full14 = (1 << 14) - 1
    mid = (m14 != 0) & (m14 != full14)
    comps = _link_components()
    n_s1 = int(np.sum(comps[m14[mid]].astype(np.int64) - 1))
    n_s2 = int(np.sum(comps[full14 ^ m14[mid]].astype(np.int64) - 1))
    return {"n_max": n_max, "n_min": n_min, "n_saddle1": n_s1, "n_saddle2": n_s2,
            "euler_consistency": n_max - n_s2 + n_s1 - n_min, "maxima": is_max,
            "lower26": m26}


def ascent_basins(values: NDArray[Any], lattice: NDArray[np.float64], lower26: I64) -> I64:
    """Flat index of the 26-neighbour maximum each voxel's steepest-ascent path reaches."""
    shape = values.shape
    step = np.asarray(lattice) / np.array(shape, dtype=np.float64)[:, None]
    lengths = np.linalg.norm(NEIGHBOURS_26 @ step, axis=1)
    idx = np.arange(values.size, dtype=np.int64).reshape(shape)
    best = np.full(shape, -np.inf)
    ptr = idx.copy()
    for k, off in enumerate(NEIGHBOURS_26):
        upper = ((lower26 >> k) & 1) == 0
        slope = (_shift(values, off) - values) / lengths[k]
        better = upper & (slope > best)
        best[better] = slope[better]
        ptr[better] = _shift(idx, off)[better]
    flat: I64 = np.asarray(ptr.ravel(), dtype=np.int64)
    while True:
        nxt = np.asarray(flat[flat], dtype=np.int64)
        if np.array_equal(nxt, flat):
            return flat.reshape(shape)
        flat = nxt


def spans(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """(3,) bools: does a face-connected cluster of ``mask`` wrap along a1, a2, a3?"""
    labels, n = ndimage.label(mask)
    if n == 0:
        return np.zeros(3, dtype=bool)
    parent = np.arange(n + 1)
    offset = np.zeros((n + 1, 3), dtype=np.int64)       # position(node) = position(parent) + offset

    def find(a: int) -> int:
        path = []
        while parent[a] != a:
            path.append(a)
            a = int(parent[a])
        root, acc = a, np.zeros(3, np.int64)
        for node in reversed(path):
            acc = acc + offset[node]
            parent[node], offset[node] = root, acc
        return root

    out = np.zeros(3, dtype=bool)
    for axis in range(3):
        last = np.take(labels, -1, axis=axis)
        first = np.take(labels, 0, axis=axis)
        both = (last > 0) & (first > 0)
        e = np.zeros(3, np.int64)
        e[axis] = 1
        for a, b in set(zip(last[both].tolist(), first[both].tolist())):
            ra, rb = find(a), find(b)
            oa, ob = offset[a], offset[b]
            if ra == rb:
                out |= (oa + e - ob) != 0
            else:
                parent[rb] = ra
                offset[rb] = oa + e - ob
    return out


def percolation_levels(values: NDArray[Any], max_steps: int = 256) -> F64:
    """(3,) highest levels c whose super-level set {f > c} spans a1, a2, a3."""
    u = np.unique(values)
    lo = np.full(3, -1)                      # known to span (-1: every voxel is included)
    hi = np.full(3, len(u) - 1)              # known not to span ({f > max} is empty)
    for _ in range(max_steps):
        width = hi - lo
        if np.all(width <= 1):
            break
        axis = int(np.argmax(width))
        mid = (lo[axis] + hi[axis]) // 2
        s = spans(values > u[mid])
        for a in range(3):
            if lo[a] < mid < hi[a]:
                if s[a]:
                    lo[a] = mid
                else:
                    hi[a] = mid
    return np.asarray(u[hi], dtype=np.float64)
