"""Milestone 1: I/O and the data model (spec §2, §3, §16)."""

import numpy as np
import pytest

from pydemi.io import Lattice, Structure, VolumetricData, read_blocks, read_chgcar, write_volumetric
from pydemi.io.base import Grid

rng = np.random.default_rng(0)


def _structure():
    lat = np.array([[4.0, 0.0, 0.0], [0.3, 4.2, 0.0], [0.1, 0.2, 4.5]])      # triclinic
    return Structure(Lattice(lat), ["Fe", "Fe", "O"],
                     [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.1]])


def _aug_lines(n_atoms, magmoms):
    lines = []
    for a in range(1, n_atoms + 1):
        lines.append(f"augmentation occupancies {a:3d}  7")
        lines.append("  0.1234567E+00 -0.2345678E-01  0.3456789E-02  0.0000000E+00  0.1E+01")
        lines.append("  0.4444444E+00  0.5555555E+00")
    if magmoms:
        lines.append("  " + "  ".join("2.0000" for _ in range(n_atoms)))
    return lines


def _insert_augmentation(path, n_atoms, magmoms=False):
    """VASP-style augmentation (and magnetic-moment) lines between and after the blocks."""
    lines = path.read_text().splitlines()
    first_blank = lines.index("")
    dims = lines[first_blank + 1]
    next_dims = {i for i, l in enumerate(lines) if l == dims and i > first_blank + 1}
    out = []
    for i, line in enumerate(lines):
        if i + 1 in next_dims and line == "":
            out.extend(_aug_lines(n_atoms, magmoms))
        else:
            out.append(line)
    out.extend(_aug_lines(n_atoms, magmoms=False))
    path.write_text("\n".join(out) + "\n")


# ---------------------------------------------------------------- data model

def test_lattice_conventions():
    A = np.array([[4.0, 0.0, 0.0], [0.3, 4.2, 0.0], [0.1, 0.2, 4.5]])
    lat = Lattice(A)
    B = lat.reciprocal
    np.testing.assert_allclose(A @ B.T, np.eye(3), atol=1e-14)          # a_i . b_j = delta_ij
    np.testing.assert_allclose(lat.metric, B @ B.T)
    assert np.isclose(lat.volume, abs(np.linalg.det(A)))
    with pytest.raises(ValueError, match="singular"):
        Lattice(np.zeros((3, 3)))


def test_volumetric_data_validates_lattice():
    s = _structure()
    other = Lattice(np.eye(3) * 3.0)
    with pytest.raises(ValueError, match="lattice"):
        VolumetricData(s, Grid(rng.random((4, 5, 6)), other))
    vd = VolumetricData(s, Grid(rng.random((4, 5, 6)), s.lattice))
    assert vd.shape == (4, 5, 6)


def test_pymatgen_structure_accepted():
    pytest.importorskip("pymatgen")
    s = _structure()
    vd = VolumetricData(s.to_pymatgen(), Grid(rng.random((3, 3, 3)), s.lattice))
    assert vd.structure.species == ["Fe", "Fe", "O"]


# ---------------------------------------------------------------- CHGCAR

def test_roundtrip_single_block(tmp_path):
    s = _structure()
    rho = rng.random((6, 7, 8))
    p = tmp_path / "CHGCAR"
    write_volumetric(p, s, [rho])
    vd = read_chgcar(p)
    assert vd.structure.species == ["Fe", "Fe", "O"]
    np.testing.assert_allclose(vd.lattice.matrix, s.lattice.matrix)
    np.testing.assert_allclose(vd.structure.frac_coords, s.frac_coords, atol=1e-11)
    np.testing.assert_allclose(vd.rho.data, rho, rtol=1e-10)


def test_volume_division(tmp_path):
    """CHGCAR stores rho * V_cell: the reader must divide by V (spec §3, §16)."""
    s = _structure()
    p = tmp_path / "CHGCAR"
    write_volumetric(p, s, [np.full((4, 4, 4), 0.25)])
    _, raw = read_blocks(p, scaled_by_volume=False)
    np.testing.assert_allclose(raw[0], 0.25 * s.volume)
    np.testing.assert_allclose(read_chgcar(p).rho.data, 0.25)


def test_fortran_order(tmp_path):
    """Value at (i, j, k) = i + 10 j + 100 k survives the fastest-index-first layout."""
    s = _structure()
    i, j, k = np.meshgrid(np.arange(3), np.arange(4), np.arange(5), indexing="ij")
    f = (i + 10 * j + 100 * k).astype(float)
    p = tmp_path / "CHGCAR"
    write_volumetric(p, s, [f], scaled_by_volume=False)
    lines = p.read_text().split("\n")
    first = lines[lines.index("") + 2]
    np.testing.assert_allclose([float(x) for x in first.split()], [0, 1, 2, 10, 11])
    np.testing.assert_allclose(read_blocks(p, scaled_by_volume=False)[1][0], f)


def test_spin_polarized_chgcar_gives_total_density(tmp_path):
    """Only the first block (the total density) is read; the magnetization is skipped."""
    s = _structure()
    rho, mag = rng.random((5, 6, 7)), rng.random((5, 6, 7)) - 0.5
    p = tmp_path / "CHGCAR"
    write_volumetric(p, s, [rho, mag])
    _insert_augmentation(p, s.n_atoms, magmoms=True)
    np.testing.assert_allclose(read_chgcar(p).rho.data, rho, rtol=1e-10)
    blocks = read_blocks(p, scaled_by_volume=True)[1]
    assert len(blocks) == 2                               # the reader still finds both blocks
    np.testing.assert_allclose(blocks[1], mag, rtol=1e-10, atol=1e-14)


def test_noncollinear_chgcar_gives_total_density(tmp_path):
    s = _structure()
    blocks = [rng.random((4, 4, 4)) for _ in range(4)]
    p = tmp_path / "CHGCAR"
    write_volumetric(p, s, blocks)
    _insert_augmentation(p, s.n_atoms)
    np.testing.assert_allclose(read_chgcar(p).rho.data, blocks[0], rtol=1e-10)


def test_vasp4_negative_scale_cartesian_selective(tmp_path):
    text = "\n".join(["vasp4 style", "-27.0", "1 0 0", "0 1 0", "0 0 1", "1 1",
                      "Selective dynamics", "Cartesian", "0.0 0.0 0.0 T T T",
                      "0.5 0.5 0.5 F F F", "", "2 2 2", " ".join(["27.0"] * 8)])
    p = tmp_path / "CHGCAR"
    p.write_text(text + "\n")
    with pytest.raises(ValueError, match="species"):
        read_chgcar(p)
    vd = read_chgcar(p, species=["Cs", "Cl"])
    np.testing.assert_allclose(vd.lattice.matrix, np.eye(3) * 3.0)
    np.testing.assert_allclose(vd.structure.frac_coords[1], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(vd.rho.data, 1.0)
    assert vd.structure.species == ["Cs", "Cl"]


def test_potcar_style_labels(tmp_path):
    s = _structure()
    p = tmp_path / "CHGCAR"
    write_volumetric(p, s, [rng.random((3, 3, 3))])
    p.write_text(p.read_text().replace("  Fe O\n", "  Fe_pv O/abc\n"))
    assert read_chgcar(p).structure.species == ["Fe", "Fe", "O"]
