"""pydemi GUI, laid out like desktop software: an input panel on the left (open a CHGCAR,
or a structure whose density ChargE3Net predicts, its summary and the settings) and
the views on the right (structure, descriptors, plots, properties predicted from the
descriptors).

Started by ``pydemi gui`` (see ``pydemi.gui.launch``), which runs this file with
Streamlit on the user's own machine.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import streamlit as st

import pydemi
from pydemi import ml, props
from pydemi.gui import app_views, core, plots, predict, viewer
from pydemi.gui import properties as P
from pydemi.gui.formulas import formula as typeset
from pydemi.io.base import Structure, VolumetricData

PDF = Path(__file__).with_name("descriptors.pdf")
MAX_JOBS = int(os.environ.get("PYDEMI_MAX_JOBS", "2"))
N_DESCRIPTORS = len(pydemi.descriptor_names())
PARTITION_HELP = {
    "nearest": "Each voxel belongs to its nearest atom. Fast; the default.",
    "power": "Nearest atom after subtracting the covalent radius squared (radical Voronoi).",
    "becke": "Smooth Becke weights between neighbouring atoms. Much slower on large grids.",
}
# seconds per voxel, measured on this machine (FeNi3, Al2Te5, SnBi2Te4; descriptors + plot data)
SEC_PER_VOXEL = 5.3e-6
BECKE_SEC_PER_VOXEL = 5.5e-5
# ChargE3Net fine-tune on its 605 held-out test structures (full grids, NMAPE)
ML_ERROR = "0.72 % mean, 0.58 % median, up to 10 % (f-electron compounds)"
INPUTS = ("CHGCAR", "Structure (ML)")
VIEWS = app_views()
DEVICE_NAMES = {"cuda": "GPU", "cpu": "CPU"}
CHART = {"displaylogo": False}

CSS = """
<style>
[data-testid="stMainBlockContainer"] { padding-top: 4rem; padding-bottom: 1rem;
  padding-left: 2rem; padding-right: 2rem; max-width: none; }
[data-testid="stSidebarHeader"] { height: 2.4rem; margin-bottom: 0; }
[data-testid="stSidebarUserContent"] { padding-top: 0; }
.pd-brand { display: flex; align-items: baseline; gap: .5rem; margin: -.4rem 0 .2rem; }
.pd-brand b { font-size: 1.25rem; letter-spacing: -.01em; }
.pd-brand span, .pd-muted { color: rgba(128, 128, 128, .95); font-size: .78rem; }
.pd-head { font-size: .7rem; font-weight: 600; letter-spacing: .07em; text-transform: uppercase;
  color: rgba(128, 128, 128, .95); margin: .9rem 0 .25rem; }
table.pd-props { width: 100%; border-collapse: collapse; font-size: .8rem; }
table.pd-props td { padding: 3px 2px; border-bottom: 1px solid rgba(128, 128, 128, .18);
  vertical-align: top; }
table.pd-props td:first-child { color: rgba(128, 128, 128, .95); white-space: nowrap; }
table.pd-props td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
.pd-title { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; }
.pd-title b { font-size: 1.15rem; }
.pd-chip { font-size: .72rem; padding: 1px 8px; border-radius: 10px;
  border: 1px solid rgba(128, 128, 128, .35); color: rgba(128, 128, 128, 1); }
.pd-chip.ml { border-color: #8b5cf6; color: #8b5cf6; }
.pd-empty { border: 1px dashed rgba(128, 128, 128, .45); border-radius: 8px; padding: 2.5rem 2rem;
  text-align: center; color: rgba(128, 128, 128, 1); margin-top: .5rem; }
.pd-empty b { display: block; font-size: 1.05rem; margin-bottom: .4rem; }
.pd-empty table { margin: 1rem auto 0; font-size: .8rem; text-align: left; }
.pd-empty td { padding: 2px 10px; }
[data-testid="stCaptionContainer"] .katex { font-size: 1.02em; }
</style>
"""


# ----------------------------------------------------------------------
# cached work, keyed by the input's content hash
# ----------------------------------------------------------------------

@dataclass
class Source:
    """Where the density comes from: an uploaded CHGCAR or a finished prediction."""

    key: str                                  # cache key (content hash, plus device for ML)
    name: str                                 # file name shown in the title bar
    label: str                                # stem for downloaded files
    load: Callable[[], VolumetricData] = field(repr=False)
    sources: Optional[dict] = None            # vd.sources of a prediction


def _upload_source(upload: Any) -> Source:
    def load() -> VolumetricData:
        upload.seek(0)
        return core.read_upload(upload)
    label = Path(upload.name).name
    for ext in (".gz", ".bz2", ".xz", ".zip"):
        label = label.removesuffix(ext)
    return Source(hashlib.sha256(upload.getbuffer()).hexdigest(), upload.name, label, load)


@st.cache_resource
def _jobs() -> predict.Jobs:
    return predict.Jobs()


@st.cache_resource(show_spinner="Checking for a GPU...")
def _devices() -> tuple[str, ...]:
    import torch
    return ("cuda", "cpu") if torch.cuda.is_available() else ("cpu",)


@st.cache_data(max_entries=32, show_spinner=False)
def _structure(digest: str, name: str, _upload: Any) -> Structure:
    return ml.read_structure(bytes(_upload.getbuffer()), name)


@st.cache_resource
def _slots() -> threading.Semaphore:
    """Caps simultaneous reads and featurizations so concurrent users cannot exhaust memory."""
    return threading.Semaphore(MAX_JOBS)


@st.cache_data(max_entries=32, show_spinner=False)
def _preview(key: str, _src: Source) -> tuple[dict, dict]:
    """Read one input: its summary and the viewer data (no descriptors yet)."""
    with _slots():
        vd = _src.load()
        return core.summary(vd), viewer.preview(vd)


@st.cache_data(max_entries=32, show_spinner=False)
def _run(key: str, partition: str, _src: Source) -> tuple[dict, dict, dict]:
    """Featurize one input and reduce it for the plots."""
    with _slots():
        vd = _src.load()
        feats, meta = core.compute(vd, partition)
        return feats, meta, plots.plot_data(vd, partition)


def _finite(x: Any) -> Any:
    return None if isinstance(x, float) and not math.isfinite(x) else x


def _fmt(v: float) -> str:
    return f"{v:.6g}" if math.isfinite(v) else str(v)


def _duration(s: float) -> str:
    if s < 2:
        return "about a second"
    if s < 90:
        return f"about {5 * math.ceil(s / 5):.0f} s" if s > 10 else f"about {math.ceil(s):.0f} s"
    return f"about {math.ceil(s / 60):.0f} min"


def _estimate(n_voxels: int, partition: str) -> float:
    return n_voxels * (SEC_PER_VOXEL + (BECKE_SEC_PER_VOXEL if partition == "becke" else 0.0))


# ----------------------------------------------------------------------
# small pieces
# ----------------------------------------------------------------------

def _kept(name: str, default: Any) -> Any:
    """A widget's last value. Streamlit drops the state of widgets that are not drawn, so
    each widget also saves its value under ``keep_<name>`` and starts from it when redrawn."""
    return st.session_state.setdefault(f"keep_{name}", default)


def _keep(name: str) -> None:
    st.session_state[f"keep_{name}"] = st.session_state[name]


def _head(text: str) -> None:
    st.html(f'<div class="pd-head">{html.escape(text)}</div>')


def _props(rows: list[tuple[str, str]]) -> None:
    """A compact two-column property table."""
    body = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in rows)
    st.html(f'<table class="pd-props">{body}</table>')


def _empty(title: str, text: str, extra: str = "") -> None:
    st.html(f'<div class="pd-empty"><b>{html.escape(title)}</b>{html.escape(text)}{extra}</div>')


def _lattice_rows(m: Any) -> list[tuple[str, str]]:
    import numpy as np
    a, b, c = (float(np.linalg.norm(v)) for v in m)

    def angle(u: Any, v: Any) -> float:
        cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
        return math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    return [("a, b, c (Å)", f"{a:.4f}, {b:.4f}, {c:.4f}"),
            ("α, β, γ (°)", f"{angle(m[1], m[2]):.2f}, {angle(m[0], m[2]):.2f}, {angle(m[0], m[1]):.2f}")]


def _info_rows(info: dict, predicted: Optional[dict]) -> list[tuple[str, str]]:
    shape = info["grid"].replace("x", " × ")
    rows = [("Formula", info["formula"]), ("Atoms", str(info["n_atoms"])),
            ("Volume (Å³)", f"{info['volume']:.3f}"), ("Electrons", f"{info['electrons']:.3f}"),
            ("Grid", f"{shape}"), ("Grid points", f"{info['n_voxels']:,}"),
            ("a, b, c (Å)", f"{info['a']:.4f}, {info['b']:.4f}, {info['c']:.4f}"),
            ("α, β, γ (°)", f"{info['alpha']:.2f}, {info['beta']:.2f}, {info['gamma']:.2f}")]
    if predicted:
        rows += [("Density", "ML-predicted"), ("Charge scale", f"{float(predicted['charge_scale']):.4f}"),
                 ("Raw electrons", f"{float(predicted['n_electrons_raw']):.3f}")]
    else:
        rows += [("Density", "DFT (CHGCAR)")]
    return rows


# ----------------------------------------------------------------------
# input panel: ChargE3Net prediction
# ----------------------------------------------------------------------

@dataclass
class Pending:
    """A structure in the ML tab whose density is not predicted yet."""

    structure: Structure
    name: str
    shape: tuple[int, int, int]


@st.fragment(run_every=1.0)
def _progress(key: Any) -> None:
    job = _jobs().get(key)
    if job is None or job.state != "running":
        st.rerun()  # the whole page: show the result
        return
    left = job.seconds_left()
    frac = job.done / job.total if job.total else 0.0
    where = DEVICE_NAMES.get(job.device or "", job.device)
    if job.done == 0:
        text = f"Loading the model on {where}..."
    else:
        eta = f", {_duration(left)} left" if left is not None else ""
        text = f"Predicting on {where}: {100 * frac:.0f} %{eta}"
    st.progress(frac, text=text)
    st.button("Cancel", icon=":material/close:", width="stretch", on_click=job.cancel,
              key="cancel_prediction")


def _start(key: Any, s: Structure, device: str) -> None:
    try:
        _jobs().submit(key, s, device)
    except RuntimeError as exc:
        st.session_state.predict_error = str(exc)


def _ml_panel() -> tuple[Optional[Source], Optional[Pending]]:
    """The structure tab of the input panel: (finished prediction, or the structure waiting)."""
    gone = ml.missing()
    if gone:
        st.info(f"Needs the ML extra ({', '.join(gone)} missing): `{ml.INSTALL_HINT}`, "
                "then restart.", icon=":material/download:")
        return None, None
    upload = st.file_uploader(
        "Crystal structure", type=None, key="structure_upload", label_visibility="collapsed",
        help="CIF, POSCAR/CONTCAR (.vasp) or another format ASE reads. ChargE3Net predicts "
             "its valence charge density, as a VASP CHGCAR of the training settings.")
    st.caption(f"ChargE3Net · {len(ml.ZVAL)} elements · error {ML_ERROR.split(',')[0]} NMAPE",
               help=f"{ml.CITATION}. Fine-tuned by CMS Lab on its VASP dataset (PAW-PBE, ENCUT "
                    f"500 eV, PREC Accurate). Error on 605 held-out structures: {ML_ERROR}.")
    if upload is None:
        return None, None
    digest = hashlib.sha256(upload.getbuffer()).hexdigest()
    try:
        s = _structure(digest, upload.name, upload)
    except ValueError as exc:
        st.error(f"Could not read {upload.name}: {exc}")
        return None, None
    bad = ml.unsupported(s)
    if bad:
        st.error(f"The model was not trained on {', '.join(bad)}; it cannot predict this "
                 "structure.")
        return None, None
    shape = ml.vasp_grid_shape(s.lattice)
    pending = Pending(s, upload.name, shape)

    devices = _devices()
    device = devices[0]
    if len(devices) > 1:
        device = st.segmented_control("Device", devices, default=_kept("device", devices[0]),
                                      required=True, format_func=DEVICE_NAMES.get, key="device",
                                      on_change=_keep, args=("device",), width="stretch")
    key = f"ml:{digest}:{device}"
    job = _jobs().get(key)
    if job is not None and job.state == "running":
        _progress(key)
        return None, pending
    if job is None or job.state in ("error", "cancelled"):
        if job is not None and job.state == "error":
            st.error(f"Prediction failed: {job.error}")
        if err := st.session_state.pop("predict_error", None):
            st.warning(err)
        st.button("Predict density", type="primary", icon=":material/play_arrow:",
                  width="stretch", on_click=_start, args=(key, s, device), key="start_prediction",
                  help=f"{math.prod(shape):,} grid points, about "
                       f"{math.prod(shape) * ml.pairs_per_point(s) / 1e6:.1f} million atom–point "
                       "pairs. Seconds to minutes on a GPU, longer on a CPU.")
        return None, pending

    vd = job.result
    assert vd is not None
    src = vd.sources
    st.caption(f":material/check_circle: Predicted in {_duration(job.elapsed())} on "
               f"{DEVICE_NAMES.get(job.device or '', job.device)}")
    stem = Path(upload.name).stem if Path(upload.name).suffix else Path(upload.name).name
    st.download_button(
        "Predicted CHGCAR", lambda: core.chgcar_bytes(vd, f"{stem}: predicted by {ml.MODEL_NAME}, "
                                                          "not DFT (pydemi)"),
        file_name=f"{stem}_predicted.CHGCAR", mime="text/plain", width="stretch",
        icon=":material/download:", on_click="ignore", key="download_chgcar",
        help="VASP CHGCAR format (total density only), e.g. for VESTA or the pydemi command line.")
    return Source(key, upload.name, f"{stem}_predicted", lambda: vd, sources=dict(src)), pending


# ----------------------------------------------------------------------
# views
# ----------------------------------------------------------------------

def _compute(src: Source, partition: str, info: dict) -> tuple[dict, dict, dict]:
    key = (src.key, partition)
    if key in st.session_state.done:
        return _run(src.key, partition, src)
    est = _duration(_estimate(info["n_voxels"], partition))
    with st.spinner(f"Computing {N_DESCRIPTORS} descriptors ({est})..."):
        try:
            out = _run(src.key, partition, src)
        except MemoryError:
            st.error("Out of memory for this grid.")
            st.stop()
    st.session_state.done.add(key)
    return out


def _figure(name: str, pdata: dict, feats: dict, key: str) -> str:
    fig, caption = plots.figure(name, pdata, feats)
    st.plotly_chart(fig, width="stretch", key=key, config=CHART)
    return caption


def _detail(name: str, feats: dict, cat: Any, caption: str) -> None:
    """The property panel of one descriptor."""
    units = cat.at[name, "units"]
    _props([("Descriptor", name), ("Domain", cat.at[name, "domain"]),
            ("Value", _fmt(feats[name])), ("Units", plots.UNIT_TEXT.get(units, units) or units)])
    _head("Formula")
    tex, symbols = typeset(name)
    st.latex(tex)
    for line in symbols:
        if line.startswith("$$"):
            st.latex(line.strip("$"))
        else:
            st.caption(line)
    _head("What the plot shows")
    st.caption(caption)


def _view_structure(prev: Optional[dict], s: Optional[Structure] = None) -> None:
    left, right = st.columns([2, 3], gap="medium")
    with left:
        _head("Cell and atoms")
        p = prev if prev is not None else viewer.structure_only(s)  # type: ignore[arg-type]
        st.plotly_chart(viewer.structure_figure(p), width="stretch", key="structure", config=CHART)
        st.caption("Drag to rotate, scroll to zoom, hover an atom for its label.")
    with right:
        _head("Density slice")
        if prev is None:
            _empty("No density yet", "Predict the density in the left panel to slice through it.")
            return
        c1, c2, c3 = st.columns([3, 4, 1.4], vertical_alignment="bottom")
        axis = c1.segmented_control("Normal to", (0, 1, 2), required=True,
                                    default=_kept("slice_axis", 2), on_change=_keep,
                                    args=("slice_axis",),
                                    format_func=lambda k: f"{viewer.AXES[k]} "
                                                          f"({''.join(a for i, a in enumerate(viewer.AXES) if i != k)})",
                                    key="slice_axis")
        pos = c2.slider(f"Position along {viewer.AXES[axis]}", 0.0, 1.0,
                        _kept("slice_pos", 0.0), 0.01, key="slice_pos", on_change=_keep,
                        args=("slice_pos",))
        log = c3.toggle("Log", value=_kept("slice_log", True), key="slice_log",
                        on_change=_keep, args=("slice_log",))
        fig, caption = viewer.slice_figure(prev, axis, pos, log)
        st.plotly_chart(fig, width="stretch", key="slice", config=CHART)
        st.caption(caption)


def _view_descriptors(label: str, info: dict, feats: dict, meta: dict, pdata: dict,
                      density: Optional[dict] = None) -> None:
    table = core.results_table(feats, meta)
    cat = pydemi.catalogue().set_index("name")
    run_meta = {k: v for k, v in meta.items() if not k.endswith("__flag")}
    if density:  # where a predicted density came from
        run_meta.update({f"density_{k}": v for k, v in density.items()})

    # toolbar: filter, search, export
    counts = table["domain"].value_counts()
    t1, t2, t3 = st.columns([5, 3, 1.3], vertical_alignment="bottom")
    domain = t1.segmented_control(
        "Domain", ("all",) + core.DOMAINS, default=_kept("table_domain", "all"), required=True,
        key="table_domain", on_change=_keep, args=("table_domain",), label_visibility="collapsed",
        format_func=lambda d: f"All {len(table)}" if d == "all" else f"{d.capitalize()} {counts[d]}")
    query = t2.text_input("Search", value=_kept("table_query", ""), key="table_query",
                          placeholder="Search name or formula", on_change=_keep,
                          label_visibility="collapsed", args=("table_query",)).strip().lower()
    with t3.popover("Export", icon=":material/download:", width="stretch"):
        st.download_button("CSV, one row", core.wide_row(label, feats).to_csv(index=False),
                           file_name=f"{label}_descriptors.csv", mime="text/csv", width="stretch",
                           help="All descriptors in one row: stack rows from many files into a dataset.")
        st.download_button("CSV, table", table.to_csv(index=False),
                           file_name=f"{label}_descriptors_table.csv", mime="text/csv",
                           width="stretch", help="One row per descriptor, with units and formula.")
        st.download_button("JSON",
                           json.dumps({"structure": label, **info,
                                       "descriptors": {k: _finite(v) for k, v in feats.items()},
                                       "metadata": {k: _finite(v) for k, v in run_meta.items()}},
                                      indent=2),
                           file_name=f"{label}_descriptors.json", mime="application/json",
                           width="stretch", help="Descriptors, structure summary and run metadata.")

    view = table if domain == "all" else table[table["domain"] == domain]
    if query:
        view = view[view["descriptor"].str.lower().str.contains(query, regex=False)
                    | view["formula"].str.lower().str.contains(query, regex=False)]
    view = view.assign(value=view["value"].map(_fmt),
                       units=view["units"].map(lambda u: plots.UNIT_TEXT.get(u, u) or u)).reset_index(drop=True)

    left, right = st.columns([3, 2], gap="medium")
    with left:
        event = st.dataframe(view[["descriptor", "domain", "value", "units", "fallback"]],
                             hide_index=True, width="stretch", height=560, on_select="rerun",
                             selection_mode="single-row", key=f"table_{domain}_{query}",
                             column_config={
                                 "descriptor": st.column_config.TextColumn(width=210, pinned=True),
                                 "domain": st.column_config.TextColumn(width=100),
                                 "fallback": st.column_config.CheckboxColumn(
                                     "fallback", width=70, help="took its documented fallback value")})
        n_fb = int(table["fallback"].sum())
        st.caption(f"{len(view)} of {len(table)} shown"
                   + (f" · :material/warning: {n_fb} took their fallback value "
                      "(see the reference PDF)" if n_fb else ""))
    with right:
        rows = event.selection.rows
        if rows:
            name = view.at[rows[0], "descriptor"]
            caption = _figure(name, pdata, feats, key="table_fig")
            _detail(name, feats, cat, caption)
        else:
            _empty("No descriptor selected", "Select a row to see its plot and formula here.")

    with st.expander("Run details"):
        st.dataframe([{"key": k, "value": str(v)} for k, v in run_meta.items()],
                     hide_index=True, width="stretch")


def _step(names: list[str], by: int) -> None:
    i = names.index(st.session_state.plot_name)
    st.session_state.plot_name = st.session_state.keep_plot_name = names[(i + by) % len(names)]


def _view_plots(feats: dict, pdata: dict) -> None:
    cat = pydemi.catalogue().set_index("name")
    t1, t2, t3, t4 = st.columns([3.4, 0.45, 4, 0.45], vertical_alignment="bottom")
    domain = t1.segmented_control("Domain", core.DOMAINS, default=_kept("plot_domain", "bonding"),
                                  required=True, format_func=str.capitalize, key="plot_domain",
                                  on_change=_keep, args=("plot_domain",),
                                  label_visibility="collapsed")
    names = list(pydemi.descriptor_names([domain]))
    kept = _kept("plot_name", names[0])
    st.session_state.plot_name = kept if kept in names else names[0]
    t2.button("", icon=":material/arrow_back:", width="stretch", on_click=_step, args=(names, -1),
              key="plot_prev", help="Previous descriptor")
    name = t3.selectbox("Descriptor", names, key="plot_name", label_visibility="collapsed",
                        format_func=lambda n: f"{names.index(n) + 1}/{len(names)}  {n} = {_fmt(feats[n])}",
                        on_change=_keep, args=("plot_name",))
    t4.button("", icon=":material/arrow_forward:", width="stretch", on_click=_step,
              args=(names, 1), key="plot_next", help="Next descriptor")
    left, right = st.columns([3, 1.2], gap="medium")
    with left:
        caption = _figure(name, pdata, feats, key="plot_fig")
    with right:
        _detail(name, feats, cat, caption)


# ----------------------------------------------------------------------
# properties predicted from the descriptors
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _bundle(path: str) -> props.Bundle:
    return props.load(path)


def _view_properties(label: str, feats: dict, meta: dict) -> None:
    lacking, path = props.missing(), props.model_path()
    if lacking:
        _empty("Property models need LightGBM", props.load_error() or
               f"Install it with: {props.INSTALL_HINT} (missing: {', '.join(lacking)}).")
        return
    if path is None or not path.exists():
        _empty("No property model installed",
               f"Set {props.ENV} to a model bundle (.json.gz) and restart pydemi gui."
               + (f" Not found: {path}" if path is not None else ""))
        return
    with st.spinner("Loading the property models..."):
        bundle = _bundle(str(path))
    pred = props.predict(feats, bundle, counts=P.counts(meta))
    match = bundle.match(feats)
    table = P.table(pred, bundle, match)

    # toolbar: group filter, export
    counts = table["group"].value_counts()
    t1, t2 = st.columns([6, 1.3], vertical_alignment="bottom")
    group = t1.segmented_control(
        "Group", ("all",) + tuple(bundle.groups), default=_kept("prop_group", "all"), required=True,
        key="prop_group", on_change=_keep, args=("prop_group",), label_visibility="collapsed",
        format_func=lambda g: f"All {len(table)}" if g == "all" else f"{g} {counts[g]}")
    with t2.popover("Export", icon=":material/download:", width="stretch"):
        st.download_button("CSV", P.export(pred, bundle).to_csv(index=False),
                           file_name=f"{label}_properties.csv", mime="text/csv", width="stretch",
                           help="One row per property: value (or probability), units, test scores.")
        st.download_button("JSON", json.dumps({"structure": label, "model": bundle.name,
                                               "model_file": str(bundle.path), "split": bundle.split,
                                               "properties": pred}, indent=2),
                           file_name=f"{label}_properties.json", mime="application/json",
                           width="stretch")

    view = (table if group == "all" else table[table["group"] == group]).reset_index(drop=True)
    left, right = st.columns([5, 3] if match is not None else [5, 4], gap="medium")
    with left:
        shown = view.drop(columns=["name"] + ([] if group == "all" else ["group"]))
        event = st.dataframe(shown, hide_index=True, width="stretch",
                             height=38 + 35 * len(view), on_select="rerun",
                             selection_mode="single-row", key=f"prop_table_{group}",
                             column_config={
                                 "property": st.column_config.TextColumn(width=180, pinned=True),
                                 "group": st.column_config.TextColumn(width=95),
                                 "prediction": st.column_config.TextColumn(width=118),
                                 "DFT": st.column_config.TextColumn(
                                     width=68, help="the value in the dataset (Materials Project / properties.csv)"),
                                 "typical error": st.column_config.TextColumn(
                                     "± error", width=56, help="mean absolute error on the test set"),
                                 "units": st.column_config.TextColumn(width=62),
                                 "test score": st.column_config.TextColumn(
                                     width=80, help="R² (regression) or ROC-AUC (yes/no) on the test set; "
                                     f"⚠ weak model: R² below {P.WEAK_R2}, a rough indication only")})
        far = props.outside_range(feats, bundle)
        if match is not None:
            st.caption(f":material/dataset: {P.match_text(match)}")
        n_weak = sum(P.weak(t) for t in bundle.targets.values())
        st.caption(f"{bundle.name} · {bundle.split} · trained {bundle.created}. Estimates from "
                   f"{P.inputs_text(bundle)}, not DFT values."
                   + (f" ⚠ {n_weak} weak models (test R² below {P.WEAK_R2}) give rough "
                      "indications only." if n_weak else "")
                   + (f" :material/warning: {len(far)} descriptors lie outside the range of the "
                      f"training materials ({', '.join(far[:4])}{', ...' if len(far) > 4 else ''}): "
                      "treat the values with extra care." if far else ""))
    with right:
        rows = event.selection.rows
        if not rows:
            _empty("No property selected", "Select a row to see how well its model does on the "
                                           "test set and which descriptors it uses.")
            return
        name = view.at[rows[0], "name"]
        t, v = bundle.targets[name], pred[name]
        _props(P.model_rows(t, pred, bundle, match))
        caveat = P.note(t, pred)
        if caveat:
            st.caption(f":material/info: {caveat}")
        fig = P.probabilities(t, v) if t.task == "classification" else P.parity(t, v)
        st.plotly_chart(fig, width="stretch", key="prop_fig", config=CHART)
        _head("Most important inputs")
        st.plotly_chart(P.importance(t), width="stretch", key="prop_importance", config=CHART)
        st.caption(t.description[:1].upper() + t.description[1:] + ".")


# ----------------------------------------------------------------------
# page
# ----------------------------------------------------------------------

st.set_page_config(page_title="pydemi", page_icon=":material/blur_on:", layout="wide",
                   initial_sidebar_state="expanded")
st.html(CSS)
st.session_state.setdefault("done", set())

with st.sidebar:
    st.html(f'<div class="pd-brand"><b>pydemi</b><span>{pydemi.__version__}</span></div>')
    _head("Input")
    # tabs, not a toggle: Streamlit draws every tab, so an uploaded file survives switching;
    # with on_change the active tab is known and picks the input
    dft_tab, ml_tab = st.tabs(list(INPUTS), key="input_tab", on_change="rerun")
    with ml_tab:
        ml_source, pending = _ml_panel()
    with dft_tab:
        upload = st.file_uploader(
            "CHGCAR", type=None, key="chgcar_upload", label_visibility="collapsed",
            help="CHGCAR or CHG, plain or compressed (.gz, .bz2, .xz, .zip). Only the first "
                 "data block (the total density) is used; spin blocks are ignored.")
    use_ml = bool(ml_tab.open)
    source = ml_source if use_ml else (_upload_source(upload) if upload is not None else None)
    summary = st.container()
    _head("Settings")
    current = _kept("partition", "nearest")
    partition = st.selectbox("Atomic partition", core.PARTITIONS, key="partition",
                             index=core.PARTITIONS.index(current), on_change=_keep,
                             args=("partition",), format_func=str.capitalize,
                             help="How voxels are assigned to atoms (only the heterogeneity "
                                  "descriptors depend on it). " + " ".join(
                                      f"{p.capitalize()}: {t}" for p, t in PARTITION_HELP.items()))
    _head("Help")
    if PDF.exists():
        st.download_button("Descriptor reference (PDF)", PDF.read_bytes(),
                           file_name="pydemi_descriptors.pdf", mime="application/pdf",
                           width="stretch", icon=":material/menu_book:", key="pdf")
    st.caption(f"pydemi {pydemi.__version__} · Shubham Maurya, CMS Lab, IIT Kanpur. "
               "Predicted densities: ChargE3Net, Koker et al., npj Comput. Mater. 2024.")

info = prev = None
if source is not None:
    try:
        with st.spinner(f"Reading {source.name}..."):
            info, prev = _preview(source.key, source)
    except ValueError as exc:
        st.error(f"Could not read {source.name}: {exc}")
        st.stop()
    except MemoryError:
        st.error("Out of memory reading this file.")
        st.stop()
    info = {**info, "n_voxels": math.prod(int(n) for n in info["grid"].split("x"))}
    with summary:
        _head("Summary")
        _props(_info_rows(info, source.sources))
elif use_ml and pending is not None:
    s = pending.structure
    formula, _ = core._formula(s.site_counts())
    with summary:
        _head("Summary")
        _props([("Formula", formula), ("Atoms", str(s.n_atoms)), ("Volume (Å³)", f"{s.volume:.3f}"),
                ("Electrons", f"{ml.n_valence(s):g}"),
                ("Grid", " × ".join(map(str, pending.shape))),
                ("Grid points", f"{math.prod(pending.shape):,}"),
                *_lattice_rows(s.lattice.matrix), ("Density", "not predicted yet")])

# --- main area -------------------------------------------------------------
if source is None and not (use_ml and pending is not None):
    _empty("No input open",
           "Open a VASP CHGCAR, or a crystal structure for ChargE3Net to predict its density, "
           "in the panel on the left.",
           extra=("<table>"
                  "<tr><td>CHGCAR tab</td><td>CHGCAR / CHG, plain or .gz .bz2 .xz .zip</td></tr>"
                  "<tr><td>Structure (ML) tab</td><td>CIF, POSCAR / CONTCAR (.vasp), other ASE formats</td></tr>"
                  f"<tr><td>Views</td><td>Structure and density slices · {N_DESCRIPTORS} "
                  "descriptors · a plot for each · properties predicted from them</td></tr></table>"))
    st.stop()

name = source.name if source is not None else pending.name  # type: ignore[union-attr]
formula = info["formula"] if info is not None else core._formula(pending.structure.site_counts())[0]  # type: ignore[union-attr]
chip = ('<span class="pd-chip ml">ML-predicted density</span>' if use_ml
        else '<span class="pd-chip">DFT density</span>')
if use_ml and source is None:
    chip = '<span class="pd-chip ml">structure only · density not predicted</span>'
st.html(f'<div class="pd-title"><b>{html.escape(formula)}</b>'
        f'<span class="pd-muted">{html.escape(name)}</span>{chip}</div>')

views = st.tabs(list(VIEWS), key="view", on_change="rerun")
with views[0]:
    if views[0].open:
        _view_structure(prev, None if pending is None else pending.structure)
for tab, kind in ((views[1], "descriptors"), (views[2], "plots"), (views[3], "properties")):
    with tab:
        if not tab.open:
            continue
        if source is None or info is None:
            _empty("No density yet", "Predict the density in the left panel; the descriptors, "
                                     "plots and properties then use the prediction.")
            continue
        try:
            feats, meta, pdata = _compute(source, partition, info)
        except ValueError as exc:
            st.error(f"Could not process {source.name}: {exc}")
            continue
        if kind == "descriptors":
            _view_descriptors(source.label, info, feats, meta, pdata, source.sources)
        elif kind == "plots":
            _view_plots(feats, pdata)
        else:
            _view_properties(source.label, feats, meta)
