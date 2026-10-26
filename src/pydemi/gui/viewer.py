"""Structure and density viewer: the cell with its atoms, and planar slices of rho.

``preview`` needs only the CHGCAR (no featurize), so it is ready as soon as the
file is read. It keeps at most ``MAX_GRID`` grid points per axis (every n-th
point, so the values shown are real grid values) to stay small enough to cache.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go

from pydemi.gui import plots
from pydemi.io.base import Structure, VolumetricData

MAX_GRID = 96        # grid points kept per axis
NEAR_PLANE = 0.6     # Angstrom; atoms this close to a slice are drawn on it
LOG_FLOOR = 1e-4     # e/Angstrom^3; lower (and negative PAW) densities share the bottom colour
AXES = ("a", "b", "c")

Data = dict[str, Any]


def preview(vd: VolumetricData) -> Data:
    """Cell, atoms and a strided copy of the density (e/Å³) for the viewer."""
    step = tuple(max(1, math.ceil(n / MAX_GRID)) for n in vd.shape)
    species = list(vd.structure.species)
    frac = np.mod(np.asarray(vd.structure.frac_coords, dtype=np.float64), 1.0)
    m = np.array(vd.lattice.matrix, dtype=np.float64)
    return {"cell": m, "species": species, "labels": plots.site_labels(species),
            "frac": frac, "atoms": frac @ m, "shape": tuple(vd.shape), "step": step,
            "grid": np.ascontiguousarray(vd.rho.data[::step[0], ::step[1], ::step[2]],
                                         dtype=np.float32),
            "axes": [np.arange(0, n, s) / n for n, s in zip(vd.shape, step)]}


def structure_only(s: Structure) -> Data:
    """The cell and atoms of a structure without a density (enough for ``structure_figure``)."""
    frac = np.mod(np.asarray(s.frac_coords, dtype=np.float64), 1.0)
    m = np.array(s.lattice.matrix, dtype=np.float64)
    species = list(s.species)
    return {"cell": m, "species": species, "labels": plots.site_labels(species),
            "frac": frac, "atoms": frac @ m}


def structure_figure(p: Data) -> go.Figure:
    """The cell and its atoms in 3D, lattice vectors labelled."""
    fig = go.Figure(plots.cell_traces(p))
    tips = 1.08 * p["cell"]
    fig.add_scatter3d(x=tips[:, 0], y=tips[:, 1], z=tips[:, 2], mode="text", text=list(AXES),
                      textfont=dict(size=16, color="gray"), hoverinfo="skip", showlegend=False)
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(title="element", orientation="h", y=0, x=0, itemsizing="constant"),
                      scene=dict(aspectmode="data", xaxis_title="x (Å)", yaxis_title="y (Å)",
                                 zaxis_title="z (Å)",
                                 # off the cell diagonal, where atoms at 0 and 1/2 would overlap
                                 camera=dict(eye=dict(x=1.7, y=0.75, z=0.85))))
    return fig


def slice_plane(p: Data, axis: int, position: float) -> tuple[np.ndarray, float]:
    """The density on the stored grid plane nearest to fractional ``position`` along ``axis``,
    and that plane's actual fractional position."""
    ax = p["axes"][axis]
    d = np.abs(ax - position % 1.0)
    i = int(np.argmin(np.minimum(d, 1.0 - d)))
    return np.take(p["grid"], i, axis=axis), float(ax[i])


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _images(fu: float, fv: float, edge: float = 0.02) -> list[tuple[float, float]]:
    """An in-plane position and its periodic copies on the far edges of the cell."""
    us = [fu, fu + 1.0] if fu < edge else [fu]
    vs = [fv, fv + 1.0] if fv < edge else [fv]
    return [(a, b) for a in us for b in vs]


def slice_figure(p: Data, axis: int, position: float, log: bool = True) -> tuple[go.Figure, str]:
    """A heatmap of rho on the plane spanned by the two other lattice vectors."""
    u, v = (k for k in range(3) if k != axis)
    plane, at = slice_plane(p, axis, position)          # shape (n_u, n_v)
    plane = np.pad(plane, ((0, 1), (0, 1)), mode="wrap")   # repeat the first row and column at 1
    m = p["cell"]
    lu, lv = (float(np.linalg.norm(m[k])) for k in (u, v))
    x, y = np.append(p["axes"][u], 1.0) * lu, np.append(p["axes"][v], 1.0) * lv
    z = np.log10(np.maximum(plane, LOG_FLOOR)) if log else plane
    fig = go.Figure(go.Heatmap(
        x=x, y=y, z=z.T, customdata=plane.T, zsmooth="best", colorscale="Viridis",
        colorbar=dict(title="log₁₀ ρ" if log else "ρ (e/Å³)"),
        hovertemplate=(f"{AXES[u]} = %{{x:.2f}} Å<br>{AXES[v]} = %{{y:.2f}} Å<br>"
                       "ρ = %{customdata:.4g} e/Å³<extra></extra>")))

    # atoms within NEAR_PLANE of the slice, measured along the plane normal
    normal = np.cross(m[u], m[v])
    spacing = abs(float(np.linalg.det(m))) / float(np.linalg.norm(normal))   # interplanar distance
    df = p["frac"][:, axis] - at
    near = np.abs(df - np.round(df)) * spacing <= NEAR_PLANE
    els = list(dict.fromkeys(p["species"]))
    for i, el in enumerate(els):
        pts = [(fu, fv, p["labels"][k]) for k in np.flatnonzero(near) if p["species"][k] == el
               for fu, fv in _images(p["frac"][k, u], p["frac"][k, v])]
        if pts:
            fu, fv, text = zip(*pts)
            fig.add_scatter(x=np.array(fu) * lu, y=np.array(fv) * lv, mode="markers+text",
                            name=el, text=list(text), textposition="top right", hoverinfo="text",
                            textfont=dict(color="black", size=13),
                            marker=dict(size=12, color=plots.ELEMENT_COLORS[i % len(plots.ELEMENT_COLORS)],
                                        line=dict(width=2, color="white")))
    gamma = math.degrees(math.acos(float(m[u] @ m[v]) / (lu * lv)))
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10),
                      title=f"ρ on the {AXES[u]}–{AXES[v]} plane at {AXES[axis]} = {at:.3f}",
                      legend=dict(orientation="h", y=-0.15),
                      xaxis=dict(title=f"along {AXES[u]} (Å)", range=[-0.03 * lu, 1.03 * lu],
                                 constrain="domain"),
                      yaxis=dict(title=f"along {AXES[v]} (Å)", range=[-0.03 * lv, 1.03 * lv],
                                 scaleanchor="x"))
    stride = p["step"][u] * p["step"][v] > 1
    every = {k: "every grid point" if p["step"][k] == 1 else f"every {_ordinal(p['step'][k])} point"
             for k in (u, v)}
    caption = (f"Density on the {AXES[u]}–{AXES[v]} lattice plane, drawn as a rectangle; the real "
               f"angle between {AXES[u]} and {AXES[v]} is {gamma:.1f}°. Atoms within {NEAR_PLANE} Å "
               f"of the plane are marked."
               + (f" Shown: {every[u]} along {AXES[u]}, {every[v]} along {AXES[v]} "
                  f"(full grid {' × '.join(map(str, p['shape']))})." if stride else "")
               + (f" Log scale; values below {LOG_FLOOR} e/Å³, including the negative PAW "
                  "pseudo-density near nuclei, share the darkest colour." if log else ""))
    return fig, caption
