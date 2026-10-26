"""pydemi.ml: grid rule, neighbour lists, and the ChargE3Net predictions (torch tests skip
without the ml extra)."""

import itertools
from pathlib import Path

import numpy as np
import pytest

import pydemi
from pydemi import ml
from pydemi.io.base import Lattice, Structure
from pydemi.ml.graph import Replicas, atom_edges, probe_edges
from pydemi.ml.grid import grid_points, vasp_grid_shape

DATA = Path(__file__).with_name("data")


@pytest.fixture(scope="module")
def beco():
    with np.load(DATA / "BeCo_221_charge3net.npz") as f:
        s = Structure(Lattice(f["lattice"]), [str(x) for x in f["species"]], f["frac_coords"])
        return s, f["rho_ref"], f["dft"]


def triclinic() -> Structure:
    A = np.array([[3.1, 0.0, 0.0], [0.9, 2.8, 0.0], [0.4, -0.7, 3.6]])
    return Structure(Lattice(A), ["Si", "O", "O"], [[0.02, 0.1, 0.97], [0.5, 0.45, 0.3], [0.8, 0.9, 0.6]])


# ----------------------------------------------------------------------
# no torch needed
# ----------------------------------------------------------------------

def test_vasp_grid_shape(beco):
    assert vasp_grid_shape(beco[0].lattice) == (40, 40, 40)          # its CHGCAR grid
    shape = vasp_grid_shape(np.diag([3.0, 5.0, 11.0]))
    assert all(n % 2 == 0 for n in shape) and shape[0] < shape[1] < shape[2]
    for n in shape:
        m = n // 2
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        assert m == 1
    assert vasp_grid_shape(np.eye(3) * 4, encut=250) < vasp_grid_shape(np.eye(3) * 4)
    with pytest.raises(ValueError):
        vasp_grid_shape(np.eye(3), prec="high")


def test_grid_points_order():
    lat = Lattice(np.diag([2.0, 3.0, 4.0]))
    pts = grid_points(lat, (2, 3, 4), 0, 24)
    assert np.allclose(pts[1], [0, 0, 1.0]) and np.allclose(pts[4], [0, 1.0, 0])
    assert np.allclose(pts[12], [1.0, 0, 0]) and np.allclose(grid_points(lat, (2, 3, 4), 5, 7), pts[5:7])


def test_potcar_table():
    assert len(ml.ZVAL) == 87
    assert ml.ZVAL["Be"] == 2 and ml.ZVAL["Co"] == 9 and ml.ZVAL["Ba"] == 10
    assert ml.POTCAR["Ba"] == "PAW_PBE Ba_sv 06Sep2000"
    s = triclinic()
    assert ml.n_valence(s) == 4 + 6 + 6
    assert ml.unsupported(s) == []
    assert ml.unsupported(Structure(s.lattice, ["Si", "Og", "Fr"], s.frac_coords)) == ["Og", "Fr"]


def _brute(s: Structure, points: np.ndarray, cutoff: float, skip_self: bool):
    found = set()
    home = (s.frac_coords % 1.0) @ s.lattice.matrix
    for n in itertools.product(range(-3, 4), repeat=3):
        shift = np.array(n) @ s.lattice.matrix
        for a, r in enumerate(home):
            d = points - (r + shift)
            for p in np.flatnonzero(np.linalg.norm(d, axis=1) < cutoff):
                if skip_self and p == a and n == (0, 0, 0):
                    continue
                found.add((a, int(p), *np.round(d[p], 8)))
    return found


def test_neighbours_match_brute_force():
    s = triclinic()
    rep = Replicas.build(s, 4.0)
    home = (s.frac_coords % 1.0) @ s.lattice.matrix
    src, dst, vec = atom_edges(rep, s, 4.0)
    assert {(a, p, *np.round(v, 8)) for a, p, v in zip(src, dst, vec)} == _brute(s, home, 4.0, True)
    pts = grid_points(s.lattice, (5, 4, 6), 0, 120)
    src, dst, vec = probe_edges(rep, pts, 4.0)
    assert {(a, p, *np.round(v, 8)) for a, p, v in zip(src, dst, vec)} == _brute(s, pts, 4.0, False)


def test_chunk_size_follows_pair_budget():
    s = triclinic()
    small = ml.chunk_size(s, "cpu")
    assert ml.chunk_size(s, "cuda") > small
    assert abs(small * ml.pairs_per_point(s) - ml.PAIRS_PER_CHUNK["cpu"]) < ml.pairs_per_point(s) + 1
    assert ml.chunk_size(s, "cpu", max_pairs=10) == 64


# ----------------------------------------------------------------------
# the network (pydemi[ml])
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield ml.load(device="cpu")
    torch.set_num_threads(n)


def test_checkpoint_loads(model):
    assert model.device == "cpu" and model.cutoff == 4.0
    assert model.sha256 == "ca2a6d5eb9c21d6d3d1e1075ead67be4ed94b7a189f13e85e38fb987a9b68b46"
    assert sum(p.numel() for p in model.net.parameters()) == 1_909_389


def test_matches_original_code(beco, model):
    """Same numbers as the original ChargE3Net code (torch 1.13, e3nn 0.5) gave this model."""
    s, ref, _ = beco
    rho = ml.predict_grid(s, (8, 8, 8), model=model)
    assert np.abs(rho - ref).max() < 1e-5 * np.abs(ref).max()


def test_independent_of_chunking(beco, model):
    s = beco[0]
    a = ml.predict_grid(s, (4, 4, 6), model=model, max_pairs=10)
    b = ml.predict_grid(s, (4, 4, 6), model=model, max_pairs=10_000)
    assert np.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_invariant_to_rotation_permutation_and_lattice_translation(model):
    s = triclinic()
    base = ml.predict_grid(s, (3, 4, 5), model=model)
    q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
    rotated = Structure(Lattice(s.lattice.matrix @ q.T), s.species, s.frac_coords)
    order = [2, 0, 1]
    shuffled = Structure(s.lattice, [s.species[i] for i in order], s.frac_coords[order] + [1, -2, 3])
    for other in (rotated, shuffled):
        assert np.allclose(ml.predict_grid(other, (3, 4, 5), model=model), base, rtol=1e-4, atol=1e-5)


def test_predict_renormalizes_and_records_sources(beco, model):
    s = beco[0]
    seen = []
    vd = ml.predict(s, shape=(8, 8, 8), device="cpu", progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (512, 512)
    assert vd.rho.data.sum() * vd.rho.dV == pytest.approx(ml.n_valence(s), rel=1e-10)
    src = vd.sources
    assert src["origin"] == "predicted" and src["grid_rule"] == "8x8x8"
    raw = ml.predict(s, shape=(8, 8, 8), device="cpu", renormalize=False)
    assert float(src["n_electrons_raw"]) == pytest.approx(raw.rho.data.sum() * raw.rho.dV, rel=1e-6)
    assert float(src["charge_scale"]) * float(src["n_electrons_raw"]) == pytest.approx(11.0, rel=1e-6)


def test_cancel_and_unsupported(beco, model):
    with pytest.raises(ml.Cancelled):
        ml.predict_grid(beco[0], (8, 8, 8), model=model, should_stop=lambda: True)
    s = triclinic()
    with pytest.raises(ValueError, match="not trained on Og"):
        ml.predict_grid(Structure(s.lattice, ["Si", "Og", "O"], s.frac_coords), (2, 2, 2), model=model)


# ----------------------------------------------------------------------
# structure files (ASE)
# ----------------------------------------------------------------------

CIF = b"""data_BeCo
_cell_length_a 2.5961
_cell_length_b 2.5961
_cell_length_c 2.5961
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P m -3 m'
_symmetry_Int_Tables_number 221
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Be1 Be 0 0 0
Co1 Co 0.5 0.5 0.5
"""

POSCAR = b"""BeCo
1.0
2.5961 0 0
0 2.5961 0
0 0 2.5961
Be Co
1 1
Direct
0 0 0
0.5 0.5 0.5
"""


def test_read_structure_cif_and_poscar(tmp_path):
    pytest.importorskip("ase")
    for name, data in (("BeCo.cif", CIF), ("POSCAR", POSCAR), ("BeCo.vasp", POSCAR)):
        s = ml.read_structure(data, name)
        assert s.species == ["Be", "Co"] and s.volume == pytest.approx(2.5961 ** 3)
    (tmp_path / "CONTCAR").write_bytes(POSCAR)
    assert ml.read_structure(tmp_path / "CONTCAR").n_atoms == 2
    with pytest.raises(ValueError, match="not a readable"):
        ml.read_structure(b"hello", "x.cif")


def test_cli_predict(tmp_path, capsys, model):
    pytest.importorskip("ase")
    from pydemi.cli import main
    (tmp_path / "BeCo.cif").write_bytes(CIF)
    assert main(["predict", str(tmp_path / "BeCo.cif"), "--out-dir", str(tmp_path / "out"),
                 "--grid", "4,4,4", "--device", "cpu"]) == 0
    vd = pydemi.read_chgcar(tmp_path / "out" / "BeCo.CHGCAR")
    assert vd.shape == (4, 4, 4) and vd.rho.data.sum() * vd.rho.dV == pytest.approx(11.0, rel=1e-8)
    assert "Koker" in capsys.readouterr().err
