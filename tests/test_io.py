"""Milestone 1: I/O and the data model (spec §2, §3, §16)."""

import numpy as np
import pytest

from pydemi.constants import BOHR_ANGSTROM
from pydemi.core.grid import integrate
from pydemi.io import (Lattice, Structure, VolumetricData, read, read_all_electron,
                       read_blocks, read_chgcar, read_cube, read_elfcar, read_locpot,
                       read_potcar_rcore, read_potcar_zval, read_vasp, read_xsf, sniff,
                       write_cube, write_volumetric, write_xsf)
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


def test_volumetric_data_validates_fields():
    s = _structure()
    rho = Grid(rng.random((4, 5, 6)), s.lattice)
    with pytest.raises(ValueError, match="shape"):
        VolumetricData(s, rho, elf=Grid(rng.random((2, 2, 2)), s.lattice))
    other = Lattice(np.eye(3) * 3.0)
    with pytest.raises(ValueError, match="lattice"):
        VolumetricData(s, rho, potential=Grid(rng.random((4, 5, 6)), other))
    vd = VolumetricData(s, rho)
    assert vd.spin_mode == "none" and vd.density_source == "pseudo"


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
    assert vd.spin_mode == "none" and vd.magnetization is None
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
    p = tmp_path / "LOCPOT"
    write_volumetric(p, s, [f], scaled_by_volume=False)
    lines = p.read_text().split("\n")
    first = lines[lines.index("") + 2]
    np.testing.assert_allclose([float(x) for x in first.split()], [0, 1, 2, 10, 11])
    np.testing.assert_allclose(read_locpot(p)[1].data, f)


def test_collinear_spin_block_after_augmentation(tmp_path):
    s = _structure()
    rho, mag = rng.random((5, 6, 7)), rng.random((5, 6, 7)) - 0.5
    p = tmp_path / "CHGCAR"
    write_volumetric(p, s, [rho, mag])
    _insert_augmentation(p, s.n_atoms, magmoms=True)
    vd = read_chgcar(p)
    assert vd.spin_mode == "collinear"
    np.testing.assert_allclose(vd.rho.data, rho, rtol=1e-10)
    np.testing.assert_allclose(vd.magnetization.data, mag, rtol=1e-10, atol=1e-14)
    assert read_chgcar(p, read_spin=False).spin_mode == "none"


def test_noncollinear_four_blocks_as_vector(tmp_path):
    s = _structure()
    blocks = [rng.random((4, 4, 4)) for _ in range(4)]
    p = tmp_path / "CHGCAR"
    write_volumetric(p, s, blocks)
    _insert_augmentation(p, s.n_atoms)
    vd = read_chgcar(p)
    assert vd.spin_mode == "noncollinear"
    assert vd.magnetization is None                      # m_x is never taken as m
    np.testing.assert_allclose(vd.magnetization_vector[2].data, blocks[3], rtol=1e-10)


def test_three_blocks_refused(tmp_path):
    s = _structure()
    p = tmp_path / "CHGCAR"
    write_volumetric(p, s, [rng.random((3, 3, 3)) for _ in range(3)])
    with pytest.raises(ValueError, match="Refusing to guess"):
        read_chgcar(p)


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


# ---------------------------------------------------------------- other VASP files

def test_elfcar_not_volume_scaled(tmp_path):
    s = _structure()
    up, dn = rng.random((5, 5, 4)), rng.random((5, 5, 4))
    p = tmp_path / "ELFCAR"
    write_volumetric(p, s, [up, dn], scaled_by_volume=False, per_line=10)
    np.testing.assert_allclose(read_elfcar(p)[1].data, up, rtol=1e-10)


def test_all_electron_sum_and_source(tmp_path):
    s = _structure()
    core, val = rng.random((4, 4, 4)), rng.random((4, 4, 4))
    write_volumetric(tmp_path / "AECCAR0", s, [core])
    write_volumetric(tmp_path / "AECCAR2", s, [val])
    vd = read_all_electron(tmp_path / "AECCAR0", tmp_path / "AECCAR2")
    assert vd.density_source == "all_electron"
    np.testing.assert_allclose(vd.rho.data, core + val, rtol=1e-10)
    np.testing.assert_allclose(vd.core_density.data, core, rtol=1e-10)


POTCAR = """  PAW_PBE Fe_pv 02Aug2007
   POMASS =   55.847; ZVAL   =   14.000    mass and valenz
   RCORE  =    2.200    outmost cutoff radius
   TITEL  = PAW_PBE Fe_pv 02Aug2007
 End of Dataset
  PAW_PBE O 08Apr2002
   POMASS =   16.000; ZVAL   =    6.000    mass and valenz
   RCORE  =    1.520    outmost cutoff radius
   TITEL  = PAW_PBE O 08Apr2002
 End of Dataset
"""


def test_read_vasp_assembles_a_run(tmp_path):
    s = _structure()
    rho, mag = rng.random((6, 6, 6)), rng.random((6, 6, 6)) - 0.5
    write_volumetric(tmp_path / "CHGCAR", s, [rho, mag])
    write_volumetric(tmp_path / "AECCAR0", s, [rho])
    write_volumetric(tmp_path / "AECCAR2", s, [2 * rho])
    write_volumetric(tmp_path / "ELFCAR", s, [rng.random((3, 3, 3))], scaled_by_volume=False)
    write_volumetric(tmp_path / "LOCPOT", s, [rng.random((6, 6, 6))], scaled_by_volume=False)
    (tmp_path / "POTCAR").write_text(POTCAR)
    vd = read_vasp(tmp_path / "CHGCAR", elf=tmp_path / "ELFCAR", locpot=tmp_path / "LOCPOT",
                   aeccar0=tmp_path / "AECCAR0", aeccar2=tmp_path / "AECCAR2")
    assert vd.density_source == "all_electron" and vd.spin_mode == "collinear"
    np.testing.assert_allclose(vd.rho.data, 3 * rho, rtol=1e-10)
    assert vd.elf.shape == (6, 6, 6) and 0.0 <= vd.elf.data.min() <= vd.elf.data.max() <= 1.0
    assert vd.zval == {"Fe": 14.0, "O": 6.0}
    assert np.isclose(vd.paw_radii["Fe"], 2.2 * BOHR_ANGSTROM)
    assert read_potcar_zval(tmp_path / "POTCAR") == [14.0, 6.0]
    assert read_potcar_rcore(tmp_path / "POTCAR") == [2.2, 1.52]


# ---------------------------------------------------------------- cube, xsf

KNOWN_CUBE = """known cube: 10 bohr cube, 2x2x2, 0.1 e/bohr^3 everywhere
periodic
    1    0.000000    0.000000    0.000000
    2    5.000000    0.000000    0.000000
    2    0.000000    5.000000    0.000000
    2    0.000000    0.000000    5.000000
    3    3.000000    2.500000    5.000000    7.500000
 0.1 0.1 0.1 0.1 0.1 0.1
 0.1 0.1
"""


def test_cube_units_against_known_file(tmp_path):
    """Bohr -> Angstrom and e/bohr^3 -> e/Angstrom^3 on read (spec §3)."""
    p = tmp_path / "known.cube"
    p.write_text(KNOWN_CUBE)
    s, g = read_cube(p)
    np.testing.assert_allclose(s.lattice.matrix, np.eye(3) * 10.0 * BOHR_ANGSTROM)
    np.testing.assert_allclose(g.data, 0.1 / BOHR_ANGSTROM ** 3)
    assert np.isclose(integrate(g), 100.0)              # 0.1 e/bohr^3 * 1000 bohr^3
    np.testing.assert_allclose(s.frac_coords[0], [0.25, 0.5, 0.75])
    assert s.species == ["Li"]


def test_cube_and_xsf_roundtrip(tmp_path):
    s = _structure()
    rho = rng.random((5, 6, 7))
    write_cube(tmp_path / "a.cube", s, rho)
    s2, g2 = read_cube(tmp_path / "a.cube")
    np.testing.assert_allclose(g2.data, rho, rtol=1e-7)
    np.testing.assert_allclose(s2.lattice.matrix, s.lattice.matrix, atol=1e-6)
    write_xsf(tmp_path / "a.xsf", s, rho)
    s3, g3 = read_xsf(tmp_path / "a.xsf")
    np.testing.assert_allclose(g3.data, rho, rtol=1e-9)
    np.testing.assert_allclose(s3.frac_coords % 1.0, s.frac_coords % 1.0, atol=1e-9)


def test_sniff_and_read(tmp_path):
    s = _structure()
    rho = rng.random((4, 4, 4))
    write_volumetric(tmp_path / "CHGCAR", s, [rho])
    write_cube(tmp_path / "rho.cube", s, rho)
    write_xsf(tmp_path / "rho.xsf", s, rho)
    assert sniff(tmp_path / "CHGCAR") == "chgcar"
    assert sniff(tmp_path / "rho.cube") == "cube"
    assert sniff(tmp_path / "rho.xsf") == "xsf"
    for name in ("CHGCAR", "rho.cube", "rho.xsf"):
        np.testing.assert_allclose(read(tmp_path / name).rho.data, rho, rtol=1e-7)
    write_volumetric(tmp_path / "ELFCAR", s, [rho], scaled_by_volume=False)
    with pytest.raises(ValueError, match="not a charge density"):
        read(tmp_path / "ELFCAR")
