"""One plot per descriptor: the quantity behind it, with its value marked.

``plot_data`` runs right after ``featurize`` on the same ``VolumetricData`` and
reuses every field featurize cached there (geometry, derivatives, census,
basins, potential). It reduces the grids to small arrays (histograms, per-site
values, a capped set of points), so the result is cheap to cache and needs no
UI. ``figure`` turns it into a plotly figure for one descriptor.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Optional

import numpy as np
import plotly.graph_objects as go

import pydemi
from pydemi.constants import (ANGSTROM_BOHR, BOND_TOL, DENSITY_TO_AU, GEOMETRY_EPS, GRADIENT_TO_AU,
                              NCI_RHO_MAX_AU, NCI_S_MAX, RHO_FLOOR_AU)
from pydemi.core.geometry import bond_census_of
from pydemi.descriptors import bonding as B
from pydemi.descriptors import heterogeneity as H
from pydemi.descriptors import structural as S
from pydemi.descriptors.registry import (FeatureOptions, Sentinel, field_derivatives, geometry,
                                         laplacian, masks, options)
from pydemi.io.base import VolumetricData
from pydemi.operators.anisotropy import anisotropy_tensor
from pydemi.operators.sitestats import site_charge

RADIAL_BIN = 0.05     # Angstrom; the default shell radii 0.8 and 1.5 fall on bin edges
UNIT_BINS = 50        # bins on [0, 1] for cosines and ellipticities
VALUE_BINS = 120      # bins for density and potential histograms
MAX_POINTS = 3000     # extrema drawn in the 3D views (a random subset beyond that)
NCI_X, NCI_S = 0.08, 2.0   # NCI map window: |sign(lambda2) rho| <= NCI_X a.u., s <= NCI_S

Data = dict[str, Any]


# ----------------------------------------------------------------------
# data: reduce the grids once per upload
# ----------------------------------------------------------------------

def _hist(x: np.ndarray, bins: int, range: tuple[float, float],
          weights: Optional[np.ndarray] = None) -> Data:
    lo, hi = range
    if not hi > lo:
        lo, hi = lo - 0.5, hi + 0.5
    counts, edges = np.histogram(x, bins=bins, range=(lo, hi), weights=weights)
    return {"edges": edges, "counts": counts.astype(np.float64)}


def _points(mask: np.ndarray, vd: VolumetricData, rng: np.random.Generator) -> Data:
    idx = np.argwhere(mask)
    total = len(idx)
    if total > MAX_POINTS:
        idx = idx[np.sort(rng.choice(total, MAX_POINTS, replace=False))]
    frac = idx / np.array(vd.shape, dtype=np.float64)
    return {"xyz": frac @ vd.lattice.matrix, "rho": vd.rho.data[tuple(idx.T)], "total": total}


def site_labels(species: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for s in species:
        seen[s] = seen.get(s, 0) + 1
        out.append(f"{s}{seen[s]}")
    return out


def _per_site(fn: Callable[[VolumetricData], Any], vd: VolumetricData) -> Optional[np.ndarray]:
    x = fn(vd)
    return None if isinstance(x, Sentinel) else np.asarray(x, dtype=np.float64)


def plot_data(vd: VolumetricData, partition: str = "nearest") -> Data:
    """Everything the descriptor plots need, as small arrays (call after ``featurize``)."""
    v = vd.with_options(FeatureOptions(partition=partition))
    o = options(v)
    st = v.structure
    rho = v.rho.data
    dV = v.rho.dV
    geo = geometry(v)
    r = geo.distance
    sh = masks(v)
    rng = np.random.default_rng(0)
    out: Data = {"species": list(st.species), "labels": site_labels(list(st.species)),
                 "c1": o.shells.c1, "c2": o.shells.c2, "scaled_shells": o.shells.scaled}

    # radial profiles about the nearest nucleus; bin k is (k w, (k+1) w], like the shell masks
    k = np.clip(np.ceil((r - GEOMETRY_EPS) / RADIAL_BIN).astype(np.int64) - 1, 0, None).ravel()
    nb = int(k.max()) + 1

    def rbin(w: np.ndarray) -> np.ndarray:
        return np.bincount(k, weights=np.asarray(w, dtype=np.float64).ravel(), minlength=nb)

    lap = laplacian(v)
    neg = lap < 0
    alap = np.abs(lap)
    out["radial"] = {
        "edges": RADIAL_BIN * np.arange(nb + 1),
        "charge": rbin(rho) * dV,
        "volume": np.bincount(k, minlength=nb) * dV,
        "vol_neg": rbin(neg) * dV,
        "charge_neg": rbin(np.where(neg, rho, 0.0)) * dV,
        "lap_neg": rbin(np.where(neg, alap, 0.0)) * dV,
        "lap_pos": rbin(np.where(neg, 0.0, alap)) * dV,
    }
    del alap
    out["r2"] = _hist(r.ravel() ** 2, 100, (0.0, float(r.max()) ** 2), rho.ravel())

    # gradient direction: |cos| to the radial direction (zeta) and to T's axes (t1..t3)
    der = field_derivatives(v, "rho")
    g = der.gradient.reshape(-1, 3)
    gn = der.gradient_norm.ravel()
    ok = gn > 0
    radial = np.abs(np.einsum("ij,ij->i", g[ok], geo.direction.reshape(-1, 3)[ok]))
    out["zeta_cos"] = _hist(radial / gn[ok], UNIT_BINS, (0.0, 1.0), gn[ok])
    del radial
    T = anisotropy_tensor(g)
    if np.isfinite(T).all():
        evals, evecs = np.linalg.eigh(T)
        g2 = gn[ok] ** 2
        out["tensor"] = {"eigenvalues": evals, "eigenvectors": evecs, "cos2": [
            _hist((g[ok] @ evecs[:, i]) ** 2 / g2, UNIT_BINS, (0.0, 1.0), g2) for i in range(3)]}
    else:
        out["tensor"] = None

    # Hessian: bounded ellipticity in the bond shell, and the NCI map
    ev = der.hessian_eigenvalues
    lam2 = ev[..., 1]
    sel = sh.bond & (lam2 < 0)
    out["ellip"] = _hist(1.0 - lam2[sel] / ev[..., 0][sel], UNIT_BINS, (0.0, 1.0)) if sel.any() else None
    rho_au, s = B._rho_au(v)
    x = np.sign(lam2) * rho_au
    win = np.isfinite(s) & (np.abs(x) <= NCI_X) & (s <= NCI_S)
    counts, xe, se = np.histogram2d(x[win], s[win], bins=(120, 80), range=((-NCI_X, NCI_X), (0, NCI_S)))
    nci = B._nci(v)
    nci_x = None if isinstance(nci, Sentinel) or not nci.any() else x[nci]
    out["nci"] = {"counts": counts, "x_edges": xe, "s_edges": se, "n_voxels": int(rho.size),
                  "n_nci": 0 if nci_x is None else int(nci_x.size),
                  "x": None if nci_x is None else _hist(nci_x, 100, (-NCI_RHO_MAX_AU, NCI_RHO_MAX_AU))}
    del x, win

    # electrostatic potential
    V = B.potential(v)
    lo, hi = float(V.min()), float(V.max())
    out["potential"] = {"sites": B.site_potentials(v),
                        "all": _hist(V.ravel(), VALUE_BINS, (lo, hi)),
                        "int": _hist(V[sh.interstitial], VALUE_BINS, (lo, hi))}

    # bond midpoints
    census = bond_census_of(v, BOND_TOL)
    mid = B._rho_mid(v)
    out["bonds"] = None if isinstance(mid, Sentinel) else {
        "length": census.length, "rho": np.asarray(mid),
        "pair": ["-".join(sorted((st.species[i], st.species[j]))) for i, j in zip(census.i, census.j)]}

    # percolation: the best spanning path must cross every slab, so rho_perc <= every slab max
    out["percolation"] = {"levels": S._percolation(v), "profiles": [
        {"max": rho.max(axis=tuple(a for a in range(3) if a != ax)),
         "mean": rho.mean(axis=tuple(a for a in range(3) if a != ax))} for ax in range(3)]}

    # extrema, non-nuclear maxima and their basins
    c = S.census(v)
    out["cell"] = v.lattice.matrix
    out["atoms"] = (np.mod(st.frac_coords, 1.0)) @ v.lattice.matrix
    out["maxima"] = _points(c["maxima"], v, rng)
    out["minima"] = _points(c["lower26"] == 0, v, rng)
    r_cut = o.nnm_r_cut
    maxima = np.flatnonzero(c["maxima"])
    nnm, _ = S._nnm(v, np.full(st.n_atoms, r_cut))
    basin = v.cache[("basins",)].ravel()
    in_nnm = np.isin(basin, nnm)
    uniq, inv = np.unique(basin[in_nnm], return_inverse=True)
    q = np.bincount(inv, weights=rho.ravel()[in_nnm], minlength=uniq.size) / float(rho.sum())
    out["nnm"] = {"r_cut": r_cut, "max_distance": geo.distance.ravel()[maxima],
                  "q": q, "distance": geo.distance.ravel()[uniq],
                  "atom": [st.species[i] for i in geo.atom_index.ravel()[uniq]]}

    # density values
    lo, hi = float(rho.min()), float(rho.max())
    imin = np.unravel_index(int(np.argmin(rho)), rho.shape)
    out["rho"] = {"all": _hist(rho.ravel(), VALUE_BINS, (lo, hi)),
                  "int": _hist(rho[sh.interstitial], VALUE_BINS, (lo, hi)),
                  "mean": float(rho.mean()),
                  "min_distance": float(r[imin]), "min_atom": st.species[int(geo.atom_index[imin])]}

    # information theory (atomic units, as in pydemi.descriptors.structural)
    dVb = dV * ANGSTROM_BOHR ** 3
    pa = np.maximum(rho, 0.0) * DENSITY_TO_AU
    N = float(pa.sum() * dVb)
    if N > 0:
        p = pa / N
        pos = p > 0
        lnr = np.log(p[pos] * st.volume * ANGSTROM_BOHR ** 3)      # ln(rho / mean rho)
        valid = pa > RHO_FLOOR_AU
        contrib = np.zeros_like(p)
        contrib[valid] = (gn.reshape(p.shape)[valid] * GRADIENT_TO_AU / N) ** 2 / p[valid] * dVb
        out["info"] = {"ln_ratio": _hist(lnr, 100, (float(lnr.min()), float(lnr.max())), p[pos] * dVb),
                       "fisher_radial": rbin(contrib)}
    else:
        out["info"] = None

    # per-site values (heterogeneity)
    out["sites"] = {"m1": _per_site(H._per_site_m1, v), "f_bond": _per_site(H._per_site_f_bond, v),
                    "zeta": _per_site(H._per_site_zeta, v),
                    "charge": site_charge(H.density_site_sums(v))}
    return out


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------

CORE, BOND, INT = "#4C78A8", "#F58518", "#54A24B"
ACCENT, MUTED, NEG, POS = "#E45756", "#BAB0AC", "#4C78A8", "#E45756"
ELEMENT_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2",
                  "#EECA3B", "#9D755D", "#FF9DA6", "#79706E"]


@lru_cache(maxsize=1)
def _units() -> dict[str, str]:
    cat = pydemi.catalogue()
    return dict(zip(cat["name"], cat["units"]))


UNIT_TEXT = {"Angstrom": "Å", "Angstrom^2": "Å²", "(Angstrom)^2": "Å²", "e/Angstrom^3": "e/Å³",
             "1/Angstrom^3": "Å⁻³", "e/bohr^3": "e/bohr³", "1/bohr^2": "bohr⁻²", "eV": "eV",
             "dimensionless": ""}


def _fmt(x: float) -> str:
    return f"{x:.4g}" if np.isfinite(x) else str(x)


def _layout(fig: go.Figure, name: str, value: float, x: str, y: str, **kw: Any) -> go.Figure:
    unit = _units().get(name, "")
    unit = UNIT_TEXT.get(unit, unit)
    fig.update_layout(title=f"{name} = {_fmt(value)}{' ' + unit if unit else ''}",
                      xaxis_title=x, yaxis_title=y, bargap=kw.pop("bargap", 0), height=430, margin=dict(l=60, r=20, t=60, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1, xanchor="right"),
                      **kw)
    return fig


def _empty(name: str, value: float, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                       font=dict(size=15))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, name, value, "", "")


def _centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[1:] + edges[:-1])


def _bar(edges: np.ndarray, y: np.ndarray, name: str, color: str, **kw: Any) -> go.Bar:
    return go.Bar(x=_centers(edges), y=y, width=np.diff(edges), name=name, marker_color=color,
                  marker_line_width=0, **kw)


def _vline(fig: go.Figure, x: float, text: str, color: str = ACCENT, dash: str = "dash",
           position: str = "top right") -> None:
    fig.add_vline(x=x, line_dash=dash, line_color=color, annotation_text=text,
                  annotation_position=position, annotation_font_color=color)


def _hline(fig: go.Figure, y: float, text: str, color: str = ACCENT, dash: str = "dash") -> None:
    fig.add_hline(y=y, line_dash=dash, line_color=color, annotation_text=text,
                  annotation_position="top left", annotation_font_color=color)


def _weighted_fraction(h: Data) -> np.ndarray:
    total = h["counts"].sum()
    return h["counts"] / total if total else h["counts"]


# --- bonding: radial distribution about the nearest nucleus -----------

def _shell_colors(edges: np.ndarray, d: Data, highlight: Optional[str] = None) -> list[str]:
    hi = edges[1:]
    out = []
    for e in hi:
        shell = "core" if e <= d["c1"] + 1e-9 else "bond" if e <= d["c2"] + 1e-9 else "int"
        color = {"core": CORE, "bond": BOND, "int": INT}[shell]
        out.append(color if highlight in (None, shell) else MUTED)
    return out


def _radial_charge(d: Data, name: str, value: float, highlight: Optional[str] = None,
                   shells: bool = True) -> go.Figure:
    rad = d["radial"]
    e = rad["edges"]
    colors = _shell_colors(e, d, highlight) if shells else CORE
    fig = go.Figure(go.Bar(x=_centers(e), y=rad["charge"] / RADIAL_BIN, width=np.diff(e),
                           marker_color=colors, marker_line_width=0, showlegend=False,
                           hovertemplate="r = %{x:.3f} Å<br>dQ/dr = %{y:.4g} e/Å<extra></extra>"))
    if shells and not d["scaled_shells"]:
        for c, t in ((d["c1"], "c1"), (d["c2"], "c2")):
            fig.add_vline(x=c, line_dash="dot", line_color="gray", annotation_text=t,
                          annotation_position="top")
    return _layout(fig, name, value, "distance to the nearest nucleus r (Å)", "dQ/dr (e/Å)")


def fig_m1(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    fig = _radial_charge(d, name, f[name], shells=False)
    _vline(fig, f["m1"], f"m1 = {_fmt(f['m1'])} Å")
    return fig, ("Charge per unit distance from the nearest nucleus. m1 is the charge-weighted "
                 "mean distance: how far the electrons sit from their atom on average.")


def fig_m2(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    h = d["r2"]
    fig = go.Figure(_bar(h["edges"], _weighted_fraction(h), "charge", CORE, showlegend=False))
    _vline(fig, f["m2"], f"m2 = {_fmt(f['m2'])} Å²")
    return (_layout(fig, name, f[name], "squared distance to the nearest nucleus r² (Å²)",
                    "charge fraction"),
            "Charge distribution over r². m2 is its mean, the charge-weighted mean squared "
            "distance; it weights far-out (diffuse) density more than m1 does.")


def fig_sigma_r2(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    fig = _radial_charge(d, name, f[name], shells=False)
    sd = float(np.sqrt(max(f["sigma_r2"], 0.0)))
    fig.add_vrect(x0=f["m1"] - sd, x1=f["m1"] + sd, fillcolor=ACCENT, opacity=0.15, line_width=0,
                  annotation_text=f"m1 ± σ (σ = {_fmt(sd)} Å)", annotation_position="top left")
    _vline(fig, f["m1"], "m1")
    return fig, ("Charge per unit distance from the nearest nucleus; the band is m1 ± σ with "
                 "σ² = sigma_r2 = m2 − m1², the radial spread of the charge.")


def _fig_shell(which: str) -> Callable[[Data, str, dict[str, float]], tuple[go.Figure, str]]:
    text = {"core": "r ≤ c1", "bond": "c1 < r ≤ c2", "int": "r > c2"}[which]

    def fn(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
        fig = _radial_charge(d, name, f[name], highlight=which)
        cap = (f"Charge per unit distance from the nearest nucleus, coloured by shell "
               f"(core blue, bond orange, interstitial green). f_{which} is the coloured "
               f"area ({text}) over the total: {100 * f[name]:.1f}% of the charge.")
        if d["scaled_shells"]:
            cap += " Shells are scaled per element, so their edges are not drawn."
        return fig, cap
    return fn


def _fig_lap(kind: str) -> Callable[[Data, str, dict[str, float]], tuple[go.Figure, str]]:
    def fn(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
        rad = d["radial"]
        e = rad["edges"]
        if kind == "volume":
            neg, pos, y = rad["vol_neg"], rad["volume"] - rad["vol_neg"], "volume per shell (Å³)"
            cap = ("Grid volume at each distance from the nearest nucleus, split by the sign of "
                   "∇²ρ. lnf is the blue (∇²ρ < 0, charge concentration) share of the total volume.")
        elif kind == "charge":
            neg, pos, y = rad["charge_neg"], rad["charge"] - rad["charge_neg"], "charge per shell (e)"
            cap = ("Charge at each distance from the nearest nucleus, split by the sign of ∇²ρ. "
                   "lnf_charge_weighted is the blue (∇²ρ < 0) share of the total charge.")
        else:
            neg, pos, y = rad["lap_neg"], rad["lap_pos"], "∫|∇²ρ| dV per shell (e/Å²)"
            cap = ("|∇²ρ| at each distance from the nearest nucleus, for ∇²ρ < 0 (blue) and "
                   "∇²ρ > 0 (red). ")
            cap += ("On a periodic grid both totals are equal, so lap_concentration is always 1/2."
                    if kind == "lap" else
                    "lap_concentration_valence is the blue share beyond c1 (the grey core "
                    "region is excluded), where the two need not balance.")
        fig = go.Figure([_bar(e, neg, "∇²ρ < 0", NEG), _bar(e, pos, "∇²ρ ≥ 0", POS)])
        fig.update_layout(barmode="stack" if kind in ("volume", "charge") else "overlay")
        if kind not in ("volume", "charge"):
            fig.update_traces(opacity=0.65)
        if kind == "valence" and not d["scaled_shells"]:
            fig.add_vrect(x0=0, x1=d["c1"], fillcolor="gray", opacity=0.2, line_width=0,
                          annotation_text="core: excluded", annotation_position="top left")
        return _layout(fig, name, f[name], "distance to the nearest nucleus r (Å)", y), cap
    return fn


def fig_zeta(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    h = d["zeta_cos"]
    fig = go.Figure(_bar(h["edges"], _weighted_fraction(h), "|∇ρ|-weighted", CORE, showlegend=False))
    _vline(fig, 1.0 - f["zeta"], f"mean = 1 − ζ = {_fmt(1.0 - f['zeta'])}", position="top left")
    return (_layout(fig, name, f[name], "|cos θ| between ∇ρ and the direction to the nearest nucleus",
                    "|∇ρ|-weighted fraction"),
            "How radial the density gradient is. A pile-up at 1 means the gradient points at "
            "the atoms (spherical atoms, ζ → 0); weight at small |cos θ| means density bent "
            "sideways into bonds. ζ = 1 − the |∇ρ|-weighted mean.")


def fig_ellip(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    h = d["ellip"]
    if h is None:
        return _empty(name, f[name], "no bond-shell voxel with λ2 < 0"), "Fallback value."
    fig = go.Figure(_bar(h["edges"], _weighted_fraction(h), "bond-shell voxels", BOND, showlegend=False))
    _vline(fig, f[name], f"mean = {_fmt(f[name])}")
    return (_layout(fig, name, f[name], "1 − λ2/λ1  (0 cylindrical, → 1 flat)", "voxel fraction"),
            "Bounded ellipticity of every bond-shell voxel where the density curves down in two "
            "directions (λ2 < 0). 0: the bond is cylindrical (σ-like); towards 1: flattened "
            "(π-like or planar).")


# --- bonding: NCI ----------------------------------------------------

def fig_f_nci(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    n = d["nci"]
    z = np.log10(n["counts"].T, where=n["counts"].T > 0, out=np.full(n["counts"].T.shape, np.nan))
    fig = go.Figure(go.Heatmap(x=_centers(n["x_edges"]), y=_centers(n["s_edges"]), z=z,
                               colorscale="Viridis", colorbar=dict(title="log₁₀ voxels"),
                               hovertemplate="sign(λ2)ρ = %{x:.3f}<br>s = %{y:.2f}<extra></extra>"))
    fig.add_shape(type="rect", x0=-NCI_RHO_MAX_AU, x1=NCI_RHO_MAX_AU, y0=0, y1=NCI_S_MAX,
                  line=dict(color=ACCENT, width=2))
    fig.add_annotation(x=0, y=NCI_S_MAX, yshift=10, text="NCI region", showarrow=False,
                       font=dict(color=ACCENT))
    return (_layout(fig, name, f[name], "sign(λ2) ρ (e/bohr³)", "reduced gradient s"),
            f"The NCI map: every voxel by its signed density and reduced gradient. f_NCI is the "
            f"share of all voxels inside the box (s < {NCI_S_MAX}, ρ < {NCI_RHO_MAX_AU} a.u.): "
            f"{n['n_nci']:,} of {n['n_voxels']:,}.")


def _nci_hist(d: Data, name: str, f: dict[str, float], split: bool) -> tuple[go.Figure, Optional[str]]:
    h = d["nci"]["x"]
    if h is None:
        return _empty(name, f[name], "no voxel in the NCI region"), "Fallback value."
    e, c = h["edges"], h["counts"]
    mid = _centers(e)
    if split:
        fig = go.Figure([_bar(e, np.where(mid < 0, c, 0), "λ2 < 0 (attractive)", NEG),
                         _bar(e, np.where(mid >= 0, c, 0), "λ2 > 0 (repulsive)", POS)])
        fig.update_layout(barmode="stack")
    else:
        fig = go.Figure(_bar(e, c, "NCI voxels", CORE, showlegend=False))
    return _layout(fig, name, f[name], "sign(λ2) ρ (e/bohr³)", "NCI voxels"), None


def fig_nci_attractive(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    fig, cap = _nci_hist(d, name, f, split=True)
    return fig, cap or ("NCI-region voxels by signed density. NCI_attractive is the blue share "
                        "(λ2 < 0: bonding-like, attractive interactions).")


def fig_sign_lambda2(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    fig, cap = _nci_hist(d, name, f, split=False)
    if cap is None:
        _vline(fig, f[name], f"mean = {_fmt(f[name])}")
    return fig, cap or ("NCI-region voxels by signed density; the line is their mean. Negative: "
                        "attractive interactions dominate; positive: steric repulsion.")


# --- bonding: potential and bond midpoints ----------------------------

def _site_dots(d: Data, y: np.ndarray, highlight: Optional[int] = None) -> go.Figure:
    """One point per site, coloured by element; ``highlight`` enlarges one site."""
    els = list(dict.fromkeys(d["species"]))
    fig = go.Figure()
    for i, el in enumerate(els):
        idx = [k for k, s in enumerate(d["species"]) if s == el]
        big = [highlight == k for k in idx]
        fig.add_scatter(x=[d["labels"][k] for k in idx], y=y[idx], name=el, mode="markers",
                        marker=dict(color=ELEMENT_COLORS[i % len(ELEMENT_COLORS)],
                                    size=[20 if b else 11 for b in big],
                                    line=dict(width=[3 if b else 0 for b in big], color="black")),
                        hovertemplate="%{x}: %{y:.5g}<extra></extra>")
    fig.update_xaxes(categoryorder="array", categoryarray=d["labels"],
                     showticklabels=len(d["labels"]) <= 60)
    return fig


def fig_v_spread(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    V = d["potential"]["sites"]
    fig = _site_dots(d, V)
    m, sd = float(V.mean()), float(V.std())
    fig.add_hrect(y0=m - sd, y1=m + sd, fillcolor=ACCENT, opacity=0.12, line_width=0)
    _hline(fig, m, f"mean ± std (std = {_fmt(sd)} eV)")
    return (_layout(fig, name, f[name], "site", "Hartree potential at the nucleus V(Rᵢ) (eV)"),
            "The electronic Hartree potential at each nucleus. V_spread is its standard deviation "
            "over sites: 0 when every site sees the same potential.")


def fig_v_int_min(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    p = d["potential"]
    fig = go.Figure([_bar(p["all"]["edges"], p["all"]["counts"], "all voxels", MUTED),
                     _bar(p["int"]["edges"], p["int"]["counts"], "interstitial (r > c2)", INT)])
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(type="log")
    _vline(fig, f[name], f"min = {_fmt(f[name])} eV")
    return (_layout(fig, name, f[name], "Hartree potential V (eV, cell mean 0)", "voxels"),
            "Distribution of the Hartree potential over the grid, interstitial voxels in green. "
            "V_int_min is the lowest interstitial value: the deepest well away from the atoms.")


def fig_rho_mid_mean(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    b = d["bonds"]
    if b is None:
        return _empty(name, f[name], "no nearest-neighbour bonds"), "Fallback value."
    fig = go.Figure()
    pairs = np.array(b["pair"])
    for i, p in enumerate(dict.fromkeys(b["pair"])):
        sel = pairs == p
        fig.add_scatter(x=b["length"][sel], y=b["rho"][sel], mode="markers", name=p,
                        marker=dict(size=9, color=ELEMENT_COLORS[i % len(ELEMENT_COLORS)]))
    _hline(fig, f[name], f"mean = {_fmt(f[name])}")
    return (_layout(fig, name, f[name], "bond length (Å)", "ρ at the bond midpoint (e/Å³)"),
            "Every nearest-neighbour bond: its length and the density at its midpoint. "
            "rho_mid_mean is the mean over bonds; high values mean shared (covalent or metallic) "
            "density between neighbours.")


def fig_rho_mid_std(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    b = d["bonds"]
    if b is None:
        return _empty(name, f[name], "no nearest-neighbour bonds"), "Fallback value."
    fig = go.Figure()
    pairs = np.array(b["pair"])
    for i, p in enumerate(dict.fromkeys(b["pair"])):
        fig.add_box(x=b["rho"][pairs == p], name=p, boxpoints="all", jitter=0.4, pointpos=0,
                    marker_color=ELEMENT_COLORS[i % len(ELEMENT_COLORS)])
    m = f["rho_mid_mean"]
    fig.add_vrect(x0=m - f[name], x1=m + f[name], fillcolor=ACCENT, opacity=0.12, line_width=0,
                  annotation_text="mean ± std", annotation_position="top left")
    return (_layout(fig, name, f[name], "ρ at the bond midpoint (e/Å³)", "bond type"),
            "Midpoint densities grouped by bond type. rho_mid_std is their spread over all "
            "bonds: large when the cell mixes strong and weak bonds.")


# --- structural -------------------------------------------------------

def _fig_perc(axis: int) -> Callable[[Data, str, dict[str, float]], tuple[go.Figure, str]]:
    lab = "abc"[axis]

    def fn(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
        prof = d["percolation"]["profiles"][axis]
        x = np.arange(prof["max"].size) / prof["max"].size
        fig = go.Figure([go.Scatter(x=x, y=prof["max"], name="slab maximum", line_color=CORE),
                         go.Scatter(x=x, y=prof["mean"], name="slab mean", line_color=MUTED)])
        _hline(fig, f[name], f"ρ_perc = {_fmt(f[name])}")
        return (_layout(fig, name, f[name], f"fractional coordinate along {lab}", "ρ (e/Å³)"),
                f"Density in each grid plane across {lab}. A path of density above ρ_perc runs "
                f"through the whole crystal along {lab}, so it must cross every plane: ρ_perc "
                f"lies below the lowest slab maximum, which marks the bottleneck.")
    return fn


def fig_perc_anisotropy(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    lv = [f["rho_perc_a"], f["rho_perc_b"], f["rho_perc_c"]]
    fig = go.Figure(go.Bar(x=["a", "b", "c"], y=lv, marker_color=[CORE, BOND, INT],
                           text=[_fmt(v) for v in lv], textposition="outside"))
    _hline(fig, float(np.mean(lv)), "mean")
    return (_layout(fig, name, f[name], "axis", "ρ_perc (e/Å³)", bargap=0.4),
            "Percolation level along each axis. perc_anisotropy = (max − min) / mean: 0 when "
            "conduction-like paths are equally good in every direction, large for layered or "
            "chain structures.")


def cell_traces(d: Data) -> list[go.Scatter3d]:
    """Cell edges and atoms (one trace per element); ``d`` has cell, atoms, species, labels."""
    m = d["cell"]
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float) @ m
    xs, ys, zs = [], [], []
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count("1") == 1:
                for p in (corners[a], corners[b]):
                    xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
                xs.append(None); ys.append(None); zs.append(None)
    traces = [go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="gray", width=2),
                           hoverinfo="skip", showlegend=False)]
    els = list(dict.fromkeys(d["species"]))
    for i, el in enumerate(els):
        idx = [k for k, s in enumerate(d["species"]) if s == el]
        a = d["atoms"][idx]
        traces.append(go.Scatter3d(x=a[:, 0], y=a[:, 1], z=a[:, 2], mode="markers", name=el,
                                   text=[d["labels"][k] for k in idx], hoverinfo="text",
                                   marker=dict(size=9, color=ELEMENT_COLORS[i % len(ELEMENT_COLORS)],
                                               line=dict(width=1, color="black"))))
    return traces


def _fig_extrema(which: str) -> Callable[[Data, str, dict[str, float]], tuple[go.Figure, str]]:
    def fn(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
        p = d[which]
        fig = go.Figure(cell_traces(d))
        fig.add_scatter3d(x=p["xyz"][:, 0], y=p["xyz"][:, 1], z=p["xyz"][:, 2], mode="markers",
                          name="maxima" if which == "maxima" else "minima",
                          marker=dict(size=2.5, color=p["rho"], colorscale="Viridis",
                                      colorbar=dict(title="ρ (e/Å³)", x=1.02)),
                          hovertemplate="ρ = %{marker.color:.4g}<extra></extra>")
        _layout(fig, name, f[name], "", "")
        fig.update_layout(height=560, scene=dict(aspectmode="data", xaxis_title="x (Å)",
                                                 yaxis_title="y (Å)", zaxis_title="z (Å)"))
        shown = "" if p["total"] <= MAX_POINTS else f" ({MAX_POINTS:,} of them drawn)"
        kind = "local maxima" if which == "maxima" else "local minima"
        return fig, (f"All {p['total']:,} density {kind} in the cell{shown}, coloured by ρ, with "
                     f"the atoms. {name} is their count per Å³. Drag to rotate.")
    return fn


def fig_n_nnm(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    n = d["nnm"]
    dist = n["max_distance"]
    top = max(float(dist.max()) if dist.size else 1.0, n["r_cut"]) + RADIAL_BIN
    e = np.arange(0, top + RADIAL_BIN, RADIAL_BIN)
    c, _ = np.histogram(dist, bins=e)
    far = _centers(e) > n["r_cut"]
    fig = go.Figure([_bar(e, np.where(far, 0, c), "at a nucleus", MUTED),
                     _bar(e, np.where(far, c, 0), "non-nuclear", ACCENT)])
    fig.update_layout(barmode="stack")
    _vline(fig, n["r_cut"], f"r_cut = {n['r_cut']} Å", color="gray", dash="dot")
    return (_layout(fig, name, f[name], "distance of the maximum from the nearest nucleus (Å)",
                    "local maxima"),
            f"Every density maximum by its distance from the nearest nucleus. Those beyond "
            f"r_cut are non-nuclear maxima (red; {int((dist > n['r_cut']).sum())} here); n_NNM "
            f"is their count per Å³. In a PAW pseudo-density the peak around a heavy nucleus can "
            f"sit on a shell just outside r_cut instead of at the nucleus, and then counts as "
            f"non-nuclear.")


def fig_q_nnm(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    n = d["nnm"]
    if n["q"].size == 0:
        return _empty(name, f[name], "no non-nuclear maxima: Q_NNM = 0"), (
            "No density maximum lies farther than r_cut from a nucleus.")
    order = np.argsort(n["q"])[::-1][:40]
    lab = [f"#{i + 1} near {n['atom'][k]} ({n['distance'][k]:.2f} Å)" for i, k in enumerate(order)]
    fig = go.Figure(go.Bar(x=lab, y=100 * n["q"][order], marker_color=ACCENT))
    return (_layout(fig, name, f[name], "non-nuclear maximum", "basin charge (% of total)",
                    bargap=0.2),
            "Charge of the steepest-ascent basin of each non-nuclear maximum (largest first). "
            "Q_NNM is their sum over the total charge: electrons held between atoms, as in "
            "electrides or some metals. Check the n_NNM plot: maxima just beyond r_cut are "
            "usually pseudo-density shells around heavy atoms, not interstitial electrons.")


def fig_rho_min(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    h = d["rho"]["all"]
    fig = go.Figure(_bar(h["edges"], h["counts"], "voxels", CORE, showlegend=False))
    fig.update_yaxes(type="log")
    _vline(fig, f[name], f"min = {_fmt(f[name])}", position="top right")
    where = f"{d['rho']['min_distance']:.2f} Å from a {d['rho']['min_atom']} nucleus"
    return (_layout(fig, name, f[name], "ρ (e/Å³)", "voxels"),
            f"Distribution of the density over all voxels. rho_min is the lowest value, found "
            f"{where}; a negative value near a nucleus is the PAW pseudo-density, not a real hole.")


def fig_rho_min_ratio(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    h, m = d["rho"]["all"], d["rho"]["mean"]
    if m == 0:
        return _empty(name, f[name], "zero mean density"), "Fallback value."
    fig = go.Figure(_bar(h["edges"] / m, h["counts"], "voxels", CORE, showlegend=False))
    fig.update_yaxes(type="log")
    _vline(fig, f[name], f"ratio = {_fmt(f[name])}")
    _vline(fig, 1.0, "mean", color="gray", dash="dot", position="top left")
    return (_layout(fig, name, f[name], "ρ / ⟨ρ⟩", "voxels"),
            "The same distribution in units of the mean density. rho_min_ratio = ρ_min / ⟨ρ⟩ "
            "compares the floor with the average, independent of the electron count.")


def fig_rho_int_mean(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    a, i = d["rho"]["all"], d["rho"]["int"]
    fig = go.Figure([_bar(a["edges"], a["counts"], "all voxels", MUTED),
                     _bar(i["edges"], i["counts"], "interstitial (r > c2)", INT)])
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(type="log")
    _vline(fig, f[name], f"interstitial mean = {_fmt(f[name])}")
    return (_layout(fig, name, f[name], "ρ (e/Å³)", "voxels"),
            "Density distribution, interstitial voxels (farther than c2 from every nucleus) in "
            "green. rho_int_mean is their mean: the electron gas between atoms.")


def _fig_t(k: int) -> Callable[[Data, str, dict[str, float]], tuple[go.Figure, str]]:
    def fn(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
        t = d["tensor"]
        if t is None:
            return _empty(name, f[name], "uniform density: T undefined"), "Fallback value."
        h = t["cos2"][k]
        e = t["eigenvectors"][:, k]
        fig = go.Figure(_bar(h["edges"], _weighted_fraction(h), "|∇ρ|²-weighted",
                             [CORE, BOND, INT][k], showlegend=False))
        _vline(fig, f[name], f"mean = t{k + 1} = {_fmt(f[name])}")
        _vline(fig, 1 / 3, "isotropic 1/3", color="gray", dash="dot", position="bottom left")
        return (_layout(fig, name, f[name], f"cos² θ between ∇ρ and axis e{k + 1}",
                        "|∇ρ|²-weighted fraction"),
                f"How much the density gradient lines up with the tensor's axis "
                f"e{k + 1} = ({e[0]:.2f}, {e[1]:.2f}, {e[2]:.2f}). t{k + 1} is the |∇ρ|²-weighted "
                f"mean of cos² θ; the three add to 1, and all equal 1/3 when no direction is "
                f"preferred.")
    return fn


def fig_charge_fa(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    t = [f[f"T_eigenvalues_t{i}"] for i in (1, 2, 3)]
    fig = go.Figure(go.Bar(x=["t1", "t2", "t3"], y=t, marker_color=[CORE, BOND, INT],
                           text=[_fmt(v) for v in t], textposition="outside"))
    _hline(fig, 1 / 3, "isotropic 1/3", color="gray", dash="dot")
    return (_layout(fig, name, f[name], "eigenvalue of T", "value", bargap=0.4),
            "The three eigenvalues of the gradient tensor T. charge_FA measures how far they are "
            "from all being 1/3: 0 for an isotropic gradient distribution, 1 if every gradient "
            "is parallel.")


def _support(h: Data, tol: float = 1e-4) -> tuple[float, float]:
    """Range of the bins holding more than ``tol`` of the largest bin."""
    keep = np.flatnonzero(h["counts"] > tol * h["counts"].max())
    return float(h["edges"][keep[0]]), float(h["edges"][keep[-1] + 1])


def fig_shannon(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    i = d["info"]
    if i is None:
        return _empty(name, f[name], "no positive density"), "Fallback value."
    h = i["ln_ratio"]
    fig = go.Figure(_bar(h["edges"], h["counts"], "charge", CORE, showlegend=False))
    _vline(fig, -f[name], f"mean = −S = {_fmt(-f[name])}")
    _vline(fig, 0.0, "uniform", color="gray", dash="dot", position="top left")
    lo, hi = _support(h)
    fig.update_xaxes(range=[min(lo, -f[name], 0.0) - 0.2, max(hi, 0.0) + 0.2])
    return (_layout(fig, name, f[name], "ln(ρ / ⟨ρ⟩)", "charge fraction"),
            "Where the electrons sit relative to the mean density, by charge. Its mean is −S: "
            "S = 0 for a uniform density, more negative the more the charge is packed into "
            "dense regions.")


def fig_disequilibrium(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    i = d["info"]
    if i is None:
        return _empty(name, f[name], "no positive density"), "Fallback value."
    h = i["ln_ratio"]
    e = np.exp(h["edges"])
    fig = go.Figure(go.Bar(x=np.sqrt(e[1:] * e[:-1]), y=h["counts"], width=np.diff(e),
                           marker_color=BOND, marker_line_width=0, showlegend=False))
    lo, hi = _support(h)
    ln10 = np.log(10)
    fig.update_xaxes(type="log", range=[(min(lo, 0.0) - 0.2) / ln10,
                                        (max(hi, np.log(f[name])) + 0.2) / ln10])
    # on a log axis, annotations take log10(x); add_vline's own labels would land off-plot
    for x, text, color, dash, side in ((f[name], f"mean = D = {_fmt(f[name])}", ACCENT, "dash", "left"),
                                       (1.0, "uniform", "gray", "dot", "right")):
        fig.add_vline(x=x, line_dash=dash, line_color=color)
        fig.add_annotation(x=np.log10(x), y=1, xref="x", yref="paper", text=text, showarrow=False,
                           xanchor=side, yanchor="bottom", font=dict(color=color))
    return (_layout(fig, name, f[name], "ρ / ⟨ρ⟩ (log scale)", "charge fraction"),
            "The density each electron sees, relative to the mean. Its charge-weighted mean is "
            "the disequilibrium D: 1 for a uniform density, larger the more concentrated it is.")


def fig_fisher(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    i = d["info"]
    if i is None:
        return _empty(name, f[name], "no positive density"), "Fallback value."
    e = d["radial"]["edges"]
    fig = go.Figure(_bar(e, i["fisher_radial"], "contribution", INT, showlegend=False))
    return (_layout(fig, name, f[name], "distance to the nearest nucleus r (Å)",
                    "contribution to I (1/bohr²)"),
            "Where the Fisher information ∫|∇ρ̃|²/ρ̃ dV comes from. The bars add up to "
            "fisher_information; it is dominated by the steep density close to the nuclei.")


def fig_lmc(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
    vals = [f["disequilibrium"], float(np.exp(f["shannon_entropy"])), f["LMC_complexity"]]
    fig = go.Figure(go.Bar(x=["D", "exp(S)", "C = D·exp(S)"], y=vals,
                           marker_color=[BOND, CORE, ACCENT],
                           text=[_fmt(v) for v in vals], textposition="outside"))
    _hline(fig, 1.0, "uniform density", color="gray", dash="dot")
    return (_layout(fig, name, f[name], "", "value", bargap=0.4),
            "LMC complexity combines order (disequilibrium D) and disorder (exp of the entropy). "
            "Both are 1 for a uniform density; C stays near 1 for both perfect order and perfect "
            "disorder and grows in between.")


# --- heterogeneity ---------------------------------------------------

BASE_TEXT = {"m1": ("m1 (Å)", "mean distance of each atom's charge from its nucleus"),
             "f_bond": ("f_bond", "share of each atom's charge in its bond shell"),
             "zeta": ("ζ", "gradient anisotropy of each atom's region")}


def _fig_site(base: str, stat: str) -> Callable[[Data, str, dict[str, float]], tuple[go.Figure, str]]:
    ylab, what = BASE_TEXT[base]

    def fn(d: Data, name: str, f: dict[str, float]) -> tuple[go.Figure, str]:
        x = d["sites"][base]
        if x is None:
            return _empty(name, f[name], "per-site values undefined"), "Fallback value."
        species = np.array(d["species"])
        cap = f"Per-site {what}, one point per atom (partition-dependent). "
        if stat in ("site_max", "site_min"):
            k = int(np.argmax(x) if stat == "site_max" else np.argmin(x))
            fig = _site_dots(d, x, highlight=k)
            cap += f"{name} is the {'highest' if stat == 'site_max' else 'lowest'}: {d['labels'][k]}."
        elif stat == "site_range":
            fig = _site_dots(d, x)
            _hline(fig, float(x.max()), "max")
            _hline(fig, float(x.min()), "min", color=NEG)
            cap += f"{name} is the gap between the highest and lowest site."
        elif stat == "site_std":
            fig = _site_dots(d, x)
            m, sd = float(x.mean()), float(x.std())
            fig.add_hrect(y0=m - sd, y1=m + sd, fillcolor=ACCENT, opacity=0.12, line_width=0)
            _hline(fig, m, "mean ± std")
            cap += f"{name} is the standard deviation over sites (the band)."
        elif stat == "within_element_var":
            fig = go.Figure()
            for i, el in enumerate(dict.fromkeys(d["species"])):
                xe = x[species == el]
                fig.add_box(y=xe, name=f"{el} (var {_fmt(float(xe.var()))})", boxpoints="all",
                            jitter=0.4, pointpos=0,
                            marker_color=ELEMENT_COLORS[i % len(ELEMENT_COLORS)])
            cap += (f"Sites grouped by element. {name} is the site-weighted mean of the variance "
                    f"inside each element: how different chemically identical atoms are.")
            if not np.isfinite(f[name]):
                cap += " No element has two sites here, so it is undefined (NaN)."
        else:
            fig = go.Figure()
            m = float(x.mean())
            for i, el in enumerate(dict.fromkeys(d["species"])):
                xe = x[species == el]
                fig.add_bar(x=[el], y=[float(xe.mean()) - m], base=[m], name=el,
                            marker_color=ELEMENT_COLORS[i % len(ELEMENT_COLORS)],
                            text=[f"w = {xe.size / x.size:.2f}"], textposition="outside",
                            hovertemplate=f"{el} mean: %{{y:.5g}}<extra></extra>")
            _hline(fig, m, "mean over all sites")
            cap += (f"Each bar runs from the mean over all sites to the element's mean; w = share "
                    f"of sites. {name} = Σ w (element mean − overall mean)²: how different the "
                    f"elements are from each other.")
            return _layout(fig, name, f[name], "element", f"element mean of {ylab}",
                           bargap=0.4), cap
        return _layout(fig, name, f[name], "site" if stat != "within_element_var" else "element",
                       ylab), cap
    return fn


FIGURES: dict[str, Callable[[Data, str, dict[str, float]], tuple[go.Figure, str]]] = {
    "zeta": fig_zeta, "m1": fig_m1, "m2": fig_m2, "sigma_r2": fig_sigma_r2,
    "f_core": _fig_shell("core"), "f_bond": _fig_shell("bond"), "f_int": _fig_shell("int"),
    "lnf": _fig_lap("volume"), "lnf_charge_weighted": _fig_lap("charge"),
    "lap_concentration": _fig_lap("lap"), "lap_concentration_valence": _fig_lap("valence"),
    "ellip_bond_bounded_avg": fig_ellip, "f_NCI": fig_f_nci, "NCI_attractive": fig_nci_attractive,
    "sign_lambda2_rho_mean": fig_sign_lambda2, "V_spread": fig_v_spread, "V_int_min": fig_v_int_min,
    "rho_mid_mean": fig_rho_mid_mean, "rho_mid_std": fig_rho_mid_std,
    "rho_perc_a": _fig_perc(0), "rho_perc_b": _fig_perc(1), "rho_perc_c": _fig_perc(2),
    "perc_anisotropy": fig_perc_anisotropy, "n_max": _fig_extrema("maxima"),
    "n_min": _fig_extrema("minima"), "n_NNM": fig_n_nnm, "Q_NNM": fig_q_nnm,
    "rho_min": fig_rho_min, "rho_min_ratio": fig_rho_min_ratio, "rho_int_mean": fig_rho_int_mean,
    "T_eigenvalues_t1": _fig_t(0), "T_eigenvalues_t2": _fig_t(1), "T_eigenvalues_t3": _fig_t(2),
    "charge_FA": fig_charge_fa, "shannon_entropy": fig_shannon,
    "fisher_information": fig_fisher, "disequilibrium": fig_disequilibrium,
    "LMC_complexity": fig_lmc,
    **{f"{b}_{s}": _fig_site(b, s) for b in BASE_TEXT
       for s in ("site_std", "site_range", "site_max", "site_min", "within_element_var",
                 "between_element_var")},
}


def figure(name: str, data: Data, feats: dict[str, float]) -> tuple[go.Figure, str]:
    """(plotly figure, one-paragraph caption) for descriptor ``name``."""
    return FIGURES[name](data, name, feats)
