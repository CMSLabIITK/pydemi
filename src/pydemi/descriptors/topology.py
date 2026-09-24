"""
pydemi.descriptors.topology
---------------------------
Family G -- topology and connectivity by direct grid operations that always
terminate (reference entries 90-98).

G1 percolation (90-91)
    rho_perc_a/b/c: the largest level c at which {rho > c} still contains a
    cluster that wraps around the periodic cell along lattice direction
    a / b / c, i.e. the density at the bottleneck of the best connecting
    path. (The reference writes min{c : ...}; taken literally that is the
    lowest density, since the whole cell always spans. The text -- metals
    percolate at high c, ionic solids only near zero -- describes the
    supremum, which is what is computed.) Spanning is detected exactly: the
    face-connected clusters are labelled in the box, then joined across the
    periodic faces with a union-find that tracks each cluster's lattice
    offset; a cluster spans direction alpha when it meets its own image
    with a winding vector w that has w_alpha != 0. The level is found by
    bisection over the sorted voxel values, so it is exact at voxel
    resolution.

    perc_anisotropy = (max - min) / mean over the three directions.

G2-G3 critical-point census (92-95)
    Piecewise-linear Morse theory on the Freudenthal triangulation of the
    grid (14-neighbour link), ties broken by voxel index. For each voxel
    the lower link L- and upper link L+ are subsets of that 14-vertex
    2-sphere:

        min: L- empty      max: L+ empty
        1-saddle multiplicity = components(L-) - 1
        2-saddle multiplicity = components(L+) - 1

    With this consistent census the Euler relation
    n_max - n_saddle2 + n_saddle1 - n_min = chi(T^3) = 0 holds *exactly*
    for any sampled function (Banchoff), so ``euler_consistency`` is an
    implementation self-check. It cannot flag an under-resolved grid, as
    the reference hopes: coarse grids change the counts, not the identity.
    (A 26-neighbour comparison, as the reference suggests, is not the link
    of any triangulation, and there the relation can fail for reasons
    unrelated to resolution.)

    Basins: steepest ascent over the same 14 neighbours, (rho_n - rho)/|d_n|,
    so every basin ends at one of the census maxima. n_NNM counts maxima
    farther than r_cut from every nucleus; Q_NNM is the charge in their
    basins.

    PAW pseudo-densities need a per-element r_cut: a pseudized atom can have
    no maximum at its nucleus at all, only lobes on a shell inside its PAW
    radius (on CaSi3Pt, 8 lobes 0.81-0.83 A from Si holding 6.3 e, just past
    a 0.8 A cutoff). The cutoff for atom i is therefore max(c1, R_i), with R_i
    the PAW RCORE read from the POTCAR / OUTCAR (``engine.paw_radii``) --
    pseudization only acts inside it -- or, when unknown, the element's
    covalent radius; ``paw_radii_known`` records which.

    Robustness: every critical-point count is sensitive to small ripples in
    near-flat, low-density regions (a 1e-3 noise floor on a synthetic 180^3
    density produced ~3e4 spurious maxima). Q_NNM is naturally robust,
    since noise basins hold negligible charge; ``n_NNM_significant``
    (a pydemi variant) counts only non-nuclear maxima whose basin holds at
    least ``NNM_MIN_CHARGE`` electrons.

    Persistence: every critical-point count is sensitive to ripple, and the
    basin-charge filter above does not remove ripple in nearly-free-electron
    metals, where a flat valence sea splits into many basins that each hold
    more than 0.01 e (Mg3(TiAl9)2: 576 "significant" maxima holding 120 of
    136 e). ``n_NNM_persistent`` / ``Q_NNM_persistent`` use 0-dimensional
    topological persistence instead: a maximum's persistence is peak - s,
    with s the level at which its superlevel-set component merges into one
    with a higher peak (elder rule). It is computed exactly from the ascent
    basins: every voxel of a basin reaches its maximum by an ascending path,
    so two components join at the highest boundary edge between their
    basins (min of the two end values), and a maximum-spanning merge over
    the basin adjacency graph reproduces the superlevel-set filtration.
    A maximum is kept when its relative persistence (peak - s) / peak is at
    least ``NNM_MIN_PERSISTENCE``; the charge of a discarded basin goes to the
    maximum that absorbed it.

G4 interstitial floor (96-98)
    rho_min, rho_min / <rho>_V, and <rho> over the interstitial shell.
    PAW pseudo-densities dip below zero inside the augmentation spheres
    (on a 6,059-structure VASP dataset, 61% of CHGCARs), so the plain
    rho_min is usually a PAW artefact rather than the interstitial floor
    the reference means. ``rho_min_int`` / ``rho_min_int_ratio`` take the
    minimum over voxels farther than max(c2, R_PAW) from their nucleus, the
    region pseudization cannot reach; NaN when that region is empty.
"""

from functools import lru_cache
from typing import Optional

import numpy as np
from scipy import ndimage

from ..engine import RHO, Engine
from ..shells import Shells
from .primitives import safe_div

NNM_MIN_CHARGE = 0.01   # electrons; interstitial electrons in electrides carry ~0.1-1 e
NNM_MIN_PERSISTENCE = 0.1   # (peak - merge level) / peak; ripple in a flat electron sea is ~0.01

# Freudenthal triangulation: edge vectors are the nonzero 0/1 vectors and
# their negatives
_POSITIVE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
                      [0, 1, 1], [1, 1, 1]])
OFFSETS = np.vstack([_POSITIVE, -_POSITIVE])
_FULL = (1 << len(OFFSETS)) - 1


@lru_cache(maxsize=None)
def _link_components() -> np.ndarray:
    """components[mask]: connected components of the link vertices in ``mask``."""
    edges = {tuple(o) for o in OFFSETS}
    n = len(OFFSETS)
    adj = [[j for j in range(n) if j != i and tuple(OFFSETS[j] - OFFSETS[i]) in edges]
           for i in range(n)]
    comps = np.zeros(1 << n, dtype=np.int8)
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
        comps[mask] = count
    return comps


def _shift(a, off):
    """a at voxel k + off (periodic)."""
    return np.roll(a, shift=tuple(-o for o in off), axis=(0, 1, 2))


def lower_link_mask(values: np.ndarray) -> np.ndarray:
    """uint16 per voxel: bit k set when neighbour OFFSETS[k] is lower.

    'Lower' is lexicographic in (value, flat index), a symbolic perturbation
    that makes every comparison strict.
    """
    idx = np.arange(values.size).reshape(values.shape)
    mask = np.zeros(values.shape, dtype=np.uint16)
    for k, off in enumerate(OFFSETS):
        vn, jn = _shift(values, off), _shift(idx, off)
        lower = (vn < values) | ((vn == values) & (jn < idx))
        mask |= lower.astype(np.uint16) << k
    return mask


def morse_census(values: np.ndarray) -> dict:
    """Entries 92-93 plus the boolean map of maxima."""
    mask = lower_link_mask(values)
    comps = _link_components()
    is_min = mask == 0
    is_max = mask == _FULL
    mid = ~(is_min | is_max)
    n_s1 = int(np.sum(comps[mask[mid]].astype(np.int64) - 1))
    n_s2 = int(np.sum(comps[_FULL ^ mask[mid]].astype(np.int64) - 1))
    n_max, n_min = int(is_max.sum()), int(is_min.sum())
    return {"n_max": n_max, "n_min": n_min, "n_saddle1": n_s1, "n_saddle2": n_s2,
            "euler_consistency": n_max - n_s2 + n_s1 - n_min, "maxima": is_max,
            "lower_mask": mask}


def ascent_basins(values: np.ndarray, lattice: np.ndarray,
                  lower_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Flat index of the maximum each voxel's steepest-ascent path reaches."""
    shape = values.shape
    if lower_mask is None:
        lower_mask = lower_link_mask(values)
    step = lattice / np.array(shape)[:, None]         # voxel edge vectors (rows)
    lengths = np.linalg.norm(OFFSETS @ step, axis=1)
    idx = np.arange(values.size).reshape(shape)
    best = np.full(shape, -np.inf)
    ptr = idx.copy()
    for k, off in enumerate(OFFSETS):
        upper = ((lower_mask >> k) & 1) == 0
        slope = (_shift(values, off) - values) / lengths[k]
        better = upper & (slope > best)
        best[better] = slope[better]
        ptr[better] = _shift(idx, off)[better]
    ptr = ptr.ravel()
    while True:                                      # pointer jumping
        nxt = ptr[ptr]
        if np.array_equal(nxt, ptr):
            return ptr.reshape(shape)
        ptr = nxt


def maxima_persistence(values: np.ndarray, basin: np.ndarray):
    """0-dimensional persistence of every maximum from its ascent basins.

    Returns (peaks, merge_level, absorber): flat indices of the maxima, the
    level at which each merges into a component with a higher peak (-inf for
    the global maximum), and the peak it merges into (itself for the global
    maximum). Exact for the superlevel-set filtration on the 14-neighbour
    Freudenthal graph (module docstring).
    """
    v = values.ravel()
    b = basin.ravel()
    peaks = np.unique(b)
    index = np.full(v.size, -1, dtype=np.int64)
    index[peaks] = np.arange(peaks.size)
    lab = index[b].reshape(values.shape)          # basin label 0..P-1 per voxel

    # boundary edges between basins, weighted by min(end values); keep the
    # highest edge per basin pair
    keys, weights = [], []
    for off in _POSITIVE:
        ln = _shift(lab, off)
        diff = ln != lab
        if not diff.any():
            continue
        a, c = lab[diff], ln[diff]
        w = np.minimum(values[diff], _shift(values, off)[diff])
        lo, hi = np.minimum(a, c), np.maximum(a, c)
        keys.append(lo * peaks.size + hi)
        weights.append(w)
    merge = np.full(peaks.size, -np.inf)
    absorber = np.arange(peaks.size)
    if keys:
        keys, weights = np.concatenate(keys), np.concatenate(weights)
        order = np.lexsort((-weights, keys))
        keys, weights = keys[order], weights[order]
        first = np.r_[True, keys[1:] != keys[:-1]]
        keys, weights = keys[first], weights[first]
        edges = np.stack([keys // peaks.size, keys % peaks.size], axis=1)
        parent = np.arange(peaks.size)
        top = np.arange(peaks.size)                # highest peak of each component

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        height = v[peaks]
        for k in np.argsort(-weights, kind="stable"):
            ra, rc = find(edges[k, 0]), find(edges[k, 1])
            if ra == rc:
                continue
            pa, pc = top[ra], top[rc]
            # elder rule: the component with the lower peak dies here
            young, old = (pa, pc) if (height[pa], peaks[pa]) < (height[pc], peaks[pc]) else (pc, pa)
            merge[young] = weights[k]
            absorber[young] = old
            parent[rc] = ra
            top[ra] = old
    return peaks, merge, peaks[absorber]


def _wraps(mask: np.ndarray) -> np.ndarray:
    """(3,) bools: does some face-connected cluster of ``mask`` wrap along each axis?"""
    labels, n = ndimage.label(mask)
    if n == 0:
        return np.zeros(3, bool)
    parent = np.arange(n + 1)
    offset = np.zeros((n + 1, 3), dtype=np.int64)    # position(node) = position(parent) + offset

    def find(a):
        path = []
        while parent[a] != a:
            path.append(a)
            a = parent[a]
        root, acc = a, np.zeros(3, np.int64)
        for node in reversed(path):                  # compress, accumulating offsets
            acc = acc + offset[node]
            parent[node], offset[node] = root, acc
        return root

    spans = np.zeros(3, bool)
    for axis in range(3):
        last = np.take(labels, -1, axis=axis)
        first = np.take(labels, 0, axis=axis)
        both = (last > 0) & (first > 0)
        e = np.zeros(3, np.int64)
        e[axis] = 1
        for a, b in set(zip(last[both].tolist(), first[both].tolist())):
            # the copy of b one cell along +axis touches a
            ra, rb = find(a), find(b)
            oa, ob = offset[a], offset[b]           # relative to the roots (roots: 0)
            if ra == rb:
                w = oa + e - ob
                spans |= w != 0
            else:
                parent[rb] = ra
                offset[rb] = oa + e - ob
    return spans


def percolation_levels(values: np.ndarray, max_steps: int = 200) -> np.ndarray:
    """(3,) rho_perc for lattice directions a, b, c (see module docstring)."""
    u = np.unique(values)
    # spans at level u[k] means {values > u[k]} wraps; true at k = -1 (everything)
    lo = np.full(3, -1)                  # known to span
    hi = np.full(3, len(u) - 1)          # known not to span ({v > max} is empty)
    for _ in range(max_steps):
        width = hi - lo
        if np.all(width <= 1):
            break
        axis = int(np.argmax(width))
        mid = (lo[axis] + hi[axis]) // 2
        spans = _wraps(values > u[mid])
        for a in range(3):
            if lo[a] < mid < hi[a]:
                if spans[a]:
                    lo[a] = mid
                else:
                    hi[a] = mid
    # supremum of the spanning levels: the value at which the network breaks
    return u[hi]


def nuclear_radii(engine: Engine, shells: Optional[Shells] = None):
    """(per-atom r_cut for non-nuclear maxima in Angstrom, from-PAW-data flag)."""
    from ..elements import covalent_radii
    shells = shells if shells is not None else engine.shells
    known = engine.paw_radii is not None
    if known:
        radii = engine.paw_radii
    else:
        radii = {}
        for e in engine.structure.elements:
            try:
                radii[e] = covalent_radii([e])[e]
            except KeyError:                 # placeholder labels (e.g. VASP 4 "X0")
                radii[e] = shells.c1
    R = np.array([radii[s] for s in engine.structure.species])
    return np.maximum(shells.c1, R), known


def topology_family(engine: Engine, field: str = RHO, shells: Optional[Shells] = None,
                    r_cut: Optional[float] = None,
                    min_basin_charge: float = NNM_MIN_CHARGE,
                    min_persistence: float = NNM_MIN_PERSISTENCE) -> dict:
    """Entries 90-98. ``r_cut`` (Angstrom) overrides the per-atom cutoffs
    for non-nuclear maxima (see module docstring)."""
    f = engine[field]
    rho = f.values
    shape = f.grid.shape
    shells = shells if shells is not None else engine.shells
    if r_cut is None:
        atom_cut, paw_known = nuclear_radii(engine, shells)
    else:
        atom_cut, paw_known = np.full(engine.structure.n_atoms, float(r_cut)), engine.paw_radii is not None

    perc = percolation_levels(rho)
    census = morse_census(rho)
    basin = ascent_basins(rho, f.grid.lattice, census["lower_mask"])
    maxima = np.flatnonzero(census["maxima"])
    geo = engine.geometry(shape)
    dist, owner = geo.distance.ravel(), geo.atom_index.ravel()
    nnm = maxima[dist[maxima] > atom_cut[owner[maxima]]]
    basin_charge = np.bincount(basin.ravel(), weights=rho.ravel(), minlength=rho.size) * f.grid.dV
    in_nnm = np.isin(basin.ravel(), nnm)
    inter = engine.shell_masks(shape, shells).interstitial

    # persistence-filtered non-nuclear maxima; ripple basins pass their charge
    # up to the maximum that absorbs them
    peaks, merge, absorber = maxima_persistence(rho, basin)
    height = rho.ravel()[peaks]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(height > 0, (height - merge) / height, 0.0)
    keep = ~np.isfinite(merge) | (rel >= min_persistence)   # the global maximum always stays
    pos = {p: k for k, p in enumerate(peaks)}
    owner_peak = np.arange(peaks.size)
    for k in range(peaks.size):                    # follow absorbers to a kept peak
        j = k
        while not keep[j]:
            j = pos[absorber[j]]
        owner_peak[k] = j
    peak_charge = np.bincount(owner_peak, weights=basin_charge[peaks], minlength=peaks.size)
    kept_nnm = np.flatnonzero(keep & (dist[peaks] > atom_cut[owner[peaks]]))

    far = geo.distance > np.maximum(shells.c2, atom_cut[geo.atom_index])
    rho_min_int = float(rho[far].min()) if far.any() else float("nan")

    return {
        "rho_perc_a": float(perc[0]),
        "rho_perc_b": float(perc[1]),
        "rho_perc_c": float(perc[2]),
        "perc_anisotropy": safe_div(perc.max() - perc.min(), perc.mean()),
        "n_max": census["n_max"],
        "n_min": census["n_min"],
        "n_saddle1": census["n_saddle1"],
        "n_saddle2": census["n_saddle2"],
        "euler_consistency": census["euler_consistency"],
        "n_NNM": int(nnm.size),
        "n_NNM_significant": int(np.sum(basin_charge[nnm] >= min_basin_charge)),
        "n_NNM_persistent": int(kept_nnm.size),
        "Q_NNM_persistent": float(peak_charge[kept_nnm].sum()),
        "Q_NNM": float(rho.ravel()[in_nnm].sum() * f.grid.dV),
        "rho_min": float(rho.min()),
        "rho_min_ratio": safe_div(rho.min(), rho.mean()),
        "rho_min_int": rho_min_int,
        "rho_min_int_ratio": safe_div(rho_min_int, rho.mean()),
        "rho_int_mean": float(rho[inter].mean()) if inter.any() else float("nan"),
        "paw_radii_known": int(paw_known),
    }
