"""The GUI end to end, headless (Streamlit's AppTest): upload, task menu, the three views,
and the structure tab that predicts a density with ChargE3Net."""

import time

import numpy as np
import pytest

pytest.importorskip("plotly")
pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

import pydemi  # noqa: E402
from pydemi.gui import APP  # noqa: E402
from pydemi.io.base import Lattice, Structure  # noqa: E402
from pydemi.validate.analytic import GaussianSuperposition  # noqa: E402


@pytest.fixture(scope="module")
def chgcar(tmp_path_factory) -> bytes:
    s = Structure(Lattice(np.array([[4.0, 0.0, 0.0], [0.5, 4.1, 0.0], [0.3, 0.4, 4.3]])),
                  ["Na", "Cl", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.45, 0.5], [0.2, 0.7, 0.35]])
    vd = GaussianSuperposition(s, [1.4, 2.2, 2.2], [1.0, 8.0, 8.0], tol=1e-20).volumetric((20, 20, 22))
    path = tmp_path_factory.mktemp("chg") / "CHGCAR"
    pydemi.write_volumetric(path, vd.structure, [vd.rho.data])
    return path.read_bytes()


def _ok(at: AppTest) -> AppTest:
    assert not at.exception, [e.value for e in at.exception]
    return at


def _html(at: AppTest) -> str:
    """The text of the st.html blocks (title bar, property tables, empty states)."""
    return "\n".join(h.proto.body for h in at.get("html"))


def _show(at: AppTest, view: int) -> AppTest:
    """Run with view tab ``view`` open (AppTest does not keep a tab choice between runs as
    the browser does, so every run sets it)."""
    from pydemi.gui import app_views
    at.session_state["view"] = app_views()[view]
    return _ok(at.run())


@pytest.fixture
def loaded(chgcar) -> AppTest:
    at = _ok(AppTest.from_file(str(APP), default_timeout=120).run())
    at.file_uploader(key="chgcar_upload").set_value(("CHGCAR", chgcar, "application/octet-stream"))
    return _ok(at.run())


def test_empty_state_before_any_input():
    at = _ok(AppTest.from_file(str(APP), default_timeout=60).run())
    assert "No input open" in _html(at)
    assert [t.label for t in at.tabs] == ["CHGCAR", "Structure (ML)"]   # views appear with a file


def test_upload_fills_the_properties_and_opens_the_views(loaded):
    text = _html(loaded)
    assert "<td>Formula</td><td>NaCl2</td>" in text and "<td>Atoms</td><td>3</td>" in text
    assert "DFT density" in text and "No input open" not in text
    from pydemi.gui import app_views
    assert [t.label for t in loaded.tabs][:len(app_views())] == list(app_views())


def test_structure_view_needs_no_descriptors(loaded):
    assert not loaded.dataframe                       # the structure view opens first
    loaded.segmented_control(key="slice_axis").set_value(0)
    loaded.slider(key="slice_pos").set_value(0.5)
    loaded.toggle(key="slice_log").set_value(False)
    _ok(loaded.run())


def test_descriptor_table_search_export_and_partition(loaded):
    at = _show(loaded, 1)
    assert len(at.dataframe[0].value) == len(pydemi.descriptor_names()) == 56
    labels = {d.label for d in at.get("download_button")}
    assert {"CSV, one row", "CSV, table", "JSON", "Descriptor reference (PDF)"} <= labels
    assert "No descriptor selected" in _html(at)
    at.text_input(key="table_query").set_value("NCI")
    at = _show(at, 1)
    assert set(at.dataframe[0].value["descriptor"]) == {"f_NCI", "NCI_attractive",
                                                         "sign_lambda2_rho_mean"}
    at.selectbox(key="partition").set_value("power")
    at = _show(at, 1)
    assert at.session_state["keep_partition"] == "power" and len(at.dataframe[0].value) == 3


def test_plot_browser_steps_through_a_domain(loaded):
    at = _show(loaded, 2)
    at.segmented_control(key="plot_domain").set_value("heterogeneity")
    at = _show(at, 2)
    names = pydemi.descriptor_names(["heterogeneity"])
    assert at.selectbox(key="plot_name").value == names[0]
    at.button(key="plot_next").click()
    at = _show(at, 2)
    assert at.selectbox(key="plot_name").value == names[1]
    assert f"<td>Descriptor</td><td>{names[1]}</td>" in _html(at)


def test_switching_views_keeps_choices(loaded):
    at = _show(loaded, 1)
    at.text_input(key="table_query").set_value("zeta")
    at = _show(_show(_show(at, 1), 0), 1)
    assert at.text_input(key="table_query").value == "zeta"


def test_properties_view_without_a_model(loaded, monkeypatch):
    from pydemi import props
    monkeypatch.delenv(props.ENV, raising=False)
    monkeypatch.setattr(props, "BUNDLED", props.BUNDLED.with_name("absent.json.gz"))
    monkeypatch.setattr(props, "missing", lambda: [])
    assert "No property model installed" in _html(_show(loaded, 3))


def test_properties_view_without_lightgbm(loaded, monkeypatch):
    from pydemi import props
    monkeypatch.setattr(props, "missing", lambda: ["lightgbm"])
    assert 'pip install &quot;pydemi[props]&quot;' in _html(_show(loaded, 3))
    monkeypatch.setattr(props, "load_error", lambda: "LightGBM is installed but cannot load (x). "
                        "On macOS, install OpenMP with: brew install libomp")
    assert "brew install libomp" in _html(_show(loaded, 3))


def test_properties_view_predicts_from_the_descriptors(loaded, property_bundle, monkeypatch):
    from pydemi import props
    monkeypatch.setenv(props.ENV, str(property_bundle[0]))
    at = _show(loaded, 3)
    table = at.dataframe[0].value
    assert list(table["property"]) == ["Bulk modulus", "Dielectric constant, total", "Metallic",
                                     "Band gap (PBE)"]
    assert "DFT" not in table.columns                  # not a dataset material
    assert {d.label for d in at.get("download_button")} >= {"CSV", "JSON"}
    assert "No property selected" in _html(at)
    at.segmented_control(key="prop_group").set_value("Mechanical")
    at.session_state["view"] = at.session_state["view"]
    assert list(_ok(at.run()).dataframe[0].value["property"]) == ["Bulk modulus"]


def test_unreadable_upload_is_a_clear_error():
    at = _ok(AppTest.from_file(str(APP), default_timeout=60).run())
    at.file_uploader(key="chgcar_upload").set_value(("CHGCAR", b"not a chgcar\n", "text/plain"))
    at.run()
    assert at.error and "Could not read" in at.error[0].value


# ----------------------------------------------------------------------
# structure tab (ChargE3Net)
# ----------------------------------------------------------------------

ML_TAB = "Structure (ML)"
CIF = b"""data_BeCo
_cell_length_a 2.5961
_cell_length_b 2.5961
_cell_length_c 2.5961
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Be1 Be 0 0 0
Co1 Co 0.5 0.5 0.5
"""


def _structure_tab(name: str, data: bytes) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.session_state["input_tab"] = ML_TAB
    _ok(at.run())
    at.file_uploader(key="structure_upload").set_value((name, data, "text/plain"))
    return _ok(at.run())


def test_structure_tab_without_the_ml_extra(monkeypatch):
    from pydemi import ml
    monkeypatch.setattr(ml, "missing", lambda: ["torch"])
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["input_tab"] = ML_TAB
    _ok(at.run())
    assert 'pip install "pydemi[ml]"' in at.info[0].value


def test_structure_tab_rejects_unknown_elements():
    pytest.importorskip("ase")
    at = _structure_tab("PoO.cif", CIF.replace(b"Be1 Be", b"Po1 Po"))
    assert at.error and "not trained on Po" in at.error[0].value


def test_structure_tab_predicts_then_offers_the_views(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    pytest.importorskip("ase")
    from pydemi import ml
    monkeypatch.setattr(ml, "vasp_grid_shape", lambda lattice, encut=500.0, prec="accurate": (6, 6, 6))
    at = _structure_tab("BeCo.cif", CIF)
    text = _html(at)
    assert "<td>Electrons</td><td>11</td>" in text and "<td>Grid points</td><td>216</td>" in text
    assert "density not predicted" in text and "No density yet" in text   # 3D cell, no slice yet
    at = _show(at, 1)
    assert "No density yet" in _html(at) and not at.dataframe
    at.button(key="start_prediction").click()
    at = _show(at, 1)
    for _ in range(240):
        if any(d.key == "download_chgcar" for d in at.get("download_button")):
            break
        time.sleep(0.5)
        at = _show(at, 1)
    assert "Predicted CHGCAR" in [d.label for d in at.get("download_button")]
    text = _html(at)
    assert "ML-predicted density" in text and "<td>Charge scale</td>" in text
    assert len(at.dataframe[0].value) == 56          # the descriptors view, now on the prediction
    details = dict(zip(at.dataframe[1].value["key"], at.dataframe[1].value["value"]))
    assert details["density_origin"] == "predicted" and details["density_grid_rule"].startswith("vasp")
