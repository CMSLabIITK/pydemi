"""The ML input path: grid from the structure, prediction files, charge renormalization."""

import numpy as np
import pytest

import pydemi
from pydemi.io import Lattice, Structure, read, sniff
from pydemi.io.predicted import predicted_grid, read_predicted, vasp_grid_shape, write_predicted

# Lattice-vector lengths (Angstrom) and CHGCAR grids of runs of the paper's dataset
# (ENCUT = 500 eV, PREC = Accurate), chosen near rounding boundaries of the rule.
DATASET_GRIDS = [
    ("Sr3(AlGe)2_71", (4.2317174433, 4.8656467857, 10.1522582607), (64, 72, 160)),
    ("YMn12_139", (8.2856980000, 8.2856980000, 4.6878030000), (120, 120, 72)),
    ("NaYbS2_166", (6.9644741442, 6.9644741442, 6.9644741365), (108, 108, 108)),
    ("RbFeSe2_15", (7.0392421967, 7.0392421967, 5.6137200880), (108, 108, 84)),
    ("Li6WN4_137", (6.6435660000, 6.6435660000, 4.8750030000), (96, 96, 72)),
    ("Nb3B4_71", (3.1552460183, 3.3167717684, 7.4376256338), (48, 48, 108)),
    ("Sb2S3_62", (3.8447330000, 11.3671410000, 11.8001180000), (56, 168, 180)),
    ("Li2InRh_216", (4.4284184446, 4.4284178259, 4.4284180000), (64, 64, 64)),
    ("TmCo3B2_191", (4.9396240000, 4.9396443358, 2.9679330000), (72, 72, 48)),
    ("Ti3Rh5_55", (4.1304260000, 5.4262040000, 10.5314990000), (60, 80, 160)),
]


@pytest.mark.parametrize("name,lengths,shape", DATASET_GRIDS, ids=[g[0] for g in DATASET_GRIDS])
def test_vasp_grid_shape_matches_dataset(name, lengths, shape):
    assert vasp_grid_shape(np.diag(lengths), encut=500.0) == shape


def test_vasp_grid_shape_uses_vector_lengths_and_fft_sizes():
    lat = np.array([[4.0, 0.0, 0.0], [2.0, 3.4641016, 0.0], [0.0, 0.0, 6.0]])    # |a| = |b| = 4
    n = vasp_grid_shape(Lattice(lat))
    assert n[0] == n[1]
    for m in n:
        k = m // 2
        assert m % 2 == 0 and k % 2 == 0
        for p in (2, 3, 5, 7):
            while k % p == 0:
                k //= p
        assert k == 1
    assert vasp_grid_shape(lat, encut=800.0)[2] > n[2]
    assert vasp_grid_shape(lat, prec="normal")[2] < n[2]
    with pytest.raises(ValueError):
        vasp_grid_shape(lat, prec="high")


def _structure():
    lat = np.array([[4.0, 0.0, 0.0], [0.3, 4.2, 0.0], [0.1, 0.2, 4.5]])
    return Structure(Lattice(lat), ["Fe", "Fe", "O"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.1]])


def _density(s, shape=(20, 22, 24), n_electrons=22.0):
    u = np.meshgrid(*(np.arange(n) / n for n in shape), indexing="ij")
    r = 1.0 + 0.5 * np.cos(2 * np.pi * u[0]) * np.cos(2 * np.pi * u[1]) + 0.2 * np.sin(2 * np.pi * u[2])
    return r * n_electrons / (r.sum() * s.volume / r.size)


ZVAL = {"Fe": 8.0, "O": 6.0}                         # N = 22


def test_round_trip_and_sources(tmp_path):
    s = _structure()
    rho = 1.01 * _density(s)
    p = write_predicted(tmp_path / "x", s, rho, model="m1", checkpoint="ckpt.pt", grid_rule="vasp:500")
    assert p.suffix == ".npz" and sniff(p) == "predicted"
    vd = read_predicted(p, zval=ZVAL, renormalize=False)
    assert vd.structure.species == s.species
    np.testing.assert_allclose(vd.structure.frac_coords, s.frac_coords)
    np.testing.assert_allclose(vd.structure.lattice.matrix, s.lattice.matrix)
    np.testing.assert_allclose(vd.rho.data, rho)
    assert vd.density_source == "pseudo"
    assert vd.sources["origin"] == "predicted" and vd.sources["rho"] == "predicted:m1"
    assert vd.sources["checkpoint"] == "ckpt.pt" and vd.sources["grid_rule"] == "vasp:500"
    assert vd.sources["charge_scale"] == "1"
    assert float(vd.sources["n_electrons_raw"]) == pytest.approx(22.22, rel=1e-9)


def test_renormalization_to_zval():
    s = _structure()
    vd = read_predicted(structure=s, rho=1.03 * _density(s), zval=ZVAL)
    assert vd.rho.data.sum() * vd.rho.dV == pytest.approx(22.0, rel=1e-12)
    assert float(vd.sources["charge_scale"]) == pytest.approx(1 / 1.03, rel=1e-6)
    assert vd.sources["zval"] == "table" and vd.zval == ZVAL


def test_missing_zval_warns_and_uses_default():
    s = _structure()
    with pytest.warns(UserWarning, match="default_zval"):
        vd = read_predicted(structure=s, rho=_density(s), zval={"Fe": 8.0})
    assert vd.zval == {"Fe": 8.0}


def test_input_errors(tmp_path):
    s = _structure()
    rho = _density(s)
    p = write_predicted(tmp_path / "x.npz", s, rho)
    with pytest.raises(ValueError):
        read_predicted(p, structure=s, rho=rho)
    with pytest.raises(ValueError):
        read_predicted(structure=s)
    with pytest.raises(ValueError):
        read_predicted(structure=s, rho=rho[0])
    bad = rho.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        read_predicted(structure=s, rho=bad, zval=ZVAL)
    with pytest.raises(ValueError):
        write_predicted(tmp_path / "y.npz", s, rho[0])
    np.savez(tmp_path / "other.npz", a=np.zeros(3))
    with pytest.raises(ValueError, match="prediction file"):
        read_predicted(tmp_path / "other.npz")


def test_read_dispatch_and_metadata(tmp_path):
    s = _structure()
    p = write_predicted(tmp_path / "x.npz", s, 0.98 * _density(s), model="charge3net")
    vd = read(p, zval=ZVAL, paw_radii={"Fe": 1.2, "O": 0.8})
    _, meta = pydemi.featurize(vd, domains=["bonding"], return_metadata=True)
    assert meta["density_origin"] == "predicted"
    assert meta["density_model"] == "charge3net"
    assert meta["charge_scale"] == pytest.approx(1 / 0.98, rel=1e-6)
    assert meta["zval_source"] == "table"
    dft = pydemi.VolumetricData(s, pydemi.Grid(_density(s), s.lattice))
    _, meta = pydemi.featurize(dft, domains=["bonding"], return_metadata=True)
    assert meta["density_origin"] == "dft" and meta["density_model"] == "" and meta["charge_scale"] == 1.0


def test_predicted_equals_same_density_given_directly():
    """A prediction with the DFT density's values gives the DFT descriptors exactly."""
    s = _structure()
    rho = _density(s)
    a = pydemi.featurize(read_predicted(structure=s, rho=rho, zval=ZVAL, renormalize=False),
                         domains=["bonding", "structural"])
    b = pydemi.featurize(pydemi.VolumetricData(s, pydemi.Grid(rho, s.lattice), zval=ZVAL),
                         domains=["bonding", "structural"])
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])), k


def test_predicted_grid_positions():
    s = _structure()
    shape, xyz = predicted_grid(s, encut=300.0)
    assert xyz.shape == (*shape, 3)
    np.testing.assert_allclose(xyz[0, 0, 0], 0.0)
    np.testing.assert_allclose(xyz[1, 0, 0], s.lattice.matrix[0] / shape[0])
    np.testing.assert_allclose(xyz[0, 0, 1], s.lattice.matrix[2] / shape[2])
