"""Web-app helpers: compressed uploads, summary and result tables (no Streamlit needed)."""

import bz2
import re
import gzip
import io
import lzma
import zipfile

import numpy as np
import pytest

from pydemi.gui import core
import pydemi
from pydemi.io.base import Lattice, Structure
from pydemi.validate.analytic import GaussianSuperposition


@pytest.fixture(scope="module")
def chgcar(tmp_path_factory) -> bytes:
    s = Structure(Lattice(np.diag([4.0, 4.2, 4.4])), ["Na", "Cl"],
                  [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    vd = GaussianSuperposition(s, [1.4, 2.0], [1.0, 8.0], tol=1e-20).volumetric((16, 16, 18))
    path = tmp_path_factory.mktemp("chg") / "CHGCAR"
    pydemi.write_volumetric(path, vd.structure, [vd.rho.data])
    return path.read_bytes()


def _zip(data: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("run/CHGCAR", data)
    return buf.getvalue()


@pytest.mark.parametrize("pack", [lambda b: b, gzip.compress, bz2.compress, lzma.compress, _zip],
                         ids=["plain", "gz", "bz2", "xz", "zip"])
def test_every_compression_reads_the_same_density(chgcar, pack):
    ref = core.read_upload(chgcar)
    vd = core.read_upload(pack(chgcar))
    np.testing.assert_array_equal(vd.rho.data, ref.rho.data)
    assert vd.structure.species == ["Na", "Cl"]


def test_garbage_is_a_clear_error():
    with pytest.raises(ValueError, match="not a readable VASP CHGCAR"):
        core.read_upload(b"hello\nworld\n")


def test_summary_and_tables(chgcar):
    vd = core.read_upload(chgcar)
    info = core.summary(vd)
    assert info["formula"] == "NaCl" and info["n_atoms"] == 2 and info["grid"] == "16x16x18"
    assert (info["a"], info["b"], info["c"]) == pytest.approx((4.0, 4.2, 4.4))
    assert (info["alpha"], info["beta"], info["gamma"]) == pytest.approx((90.0, 90.0, 90.0))
    assert info["electrons"] == pytest.approx(9.0, rel=1e-6)
    feats, meta = core.compute(vd)
    table = core.results_table(feats, meta)
    assert list(table["descriptor"]) == pydemi.descriptor_names()
    assert table["units"].notna().all() and table["formula"].str.len().gt(0).all()
    row = core.wide_row("NaCl", feats)
    assert row.shape == (1, len(feats) + 1)


def test_formula_reduces():
    assert core._formula({"Fe": 2, "O": 4}) == ("Fe2O4", "FeO2")


def test_unknown_partition_is_rejected(chgcar):
    with pytest.raises(ValueError, match="unknown partition"):
        core.compute(core.read_upload(chgcar), "hirshfeld")


def test_every_descriptor_has_a_typeset_formula():
    from pydemi.gui.formulas import FORMULAS
    assert set(FORMULAS) == set(pydemi.descriptor_names())
    for name, (tex, symbols) in FORMULAS.items():
        for text in [tex, *symbols]:
            assert text.count("{") == text.count("}"), name
            assert text.count("\\left") == text.count("\\right"), name
        assert not re.search(r"(?<![\\a-zA-Z])(rho|sum|lambda|grad|lap)(?![a-zA-Z])", tex), name   # LaTeX, not ASCII
        for line in symbols:
            assert line.count("$") % 2 == 0, name
            if line.startswith("$$"):
                assert line.endswith("$$") and line.count("$") == 4, name
