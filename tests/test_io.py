import numpy as np
import pytest

from pydemi.io import (SPIN_COLLINEAR, SPIN_NONCOLLINEAR, SPIN_NONE, read_aeccar,
                       read_chgcar, read_elfcar, read_locpot, read_volumetric,
                       write_volumetric)
from pydemi import Engine, Structure

rng = np.random.default_rng(0)


def _structure():
    lat = np.array([[4.0, 0.0, 0.0], [0.3, 4.2, 0.0], [0.1, 0.2, 4.5]])
    return Structure(lat, ["Fe", "Fe", "O"],
                     [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.1]])


def _insert_augmentation(path, n_atoms, magmoms=False):
    """Replace the blank line between blocks with VASP-style augmentation
    (and optionally magnetic-moment) lines, and append them after the last."""
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


def _aug_lines(n_atoms, magmoms):
    lines = []
    for a in range(1, n_atoms + 1):
        lines.append(f"augmentation occupancies {a:3d}  7")
        lines.append("  0.1234567E+00 -0.2345678E-01  0.3456789E-02  0.0000000E+00  0.1E+01")
        lines.append("  0.4444444E+00  0.5555555E+00")
    if magmoms:
        lines.append("  " + "  ".join("2.0000" for _ in range(n_atoms)))
    return lines


def test_roundtrip_single_block(tmp_path):
    s = _structure()
    rho = rng.random((6, 7, 8))
    p = tmp_path / "CHGCAR"
    write_volumetric(p, s, [rho])
    cd = read_chgcar(p)
    assert cd.spin_mode == SPIN_NONE and cd.magnetization is None
    assert cd.structure.species == ("Fe", "Fe", "O")
    np.testing.assert_allclose(cd.structure.lattice, s.lattice)
    np.testing.assert_allclose(cd.structure.frac_coords, s.frac_coords, atol=1e-9)
    np.testing.assert_allclose(cd.total, rho, rtol=1e-10)


def test_fortran_order(tmp_path):
    # value at (i, j, k) = i + 10 j + 100 k must survive the x-fastest layout
    s = _structure()
    i, j, k = np.meshgrid(np.arange(3), np.arange(4), np.arange(5), indexing="ij")
    f = (i + 10 * j + 100 * k).astype(float)
    p = tmp_path / "LOCPOT"
    write_volumetric(p, s, [f], scaled_by_volume=False)
    first_values = p.read_text().split("\n")[p.read_text().split("\n").index("") + 2]
    np.testing.assert_allclose([float(x) for x in first_values.split()], [0, 1, 2, 10, 11])
    np.testing.assert_allclose(read_locpot(p).blocks[0], f)


def test_collinear_spin_with_augmentation(tmp_path):
    s = _structure()
    rho, mag = rng.random((5, 6, 7)), rng.random((5, 6, 7)) - 0.5
    p = tmp_path / "CHGCAR"
    write_volumetric(p, s, [rho, mag])
    _insert_augmentation(p, s.n_atoms, magmoms=True)
    cd = read_chgcar(p)
    assert cd.spin_mode == SPIN_COLLINEAR
    np.testing.assert_allclose(cd.total, rho, rtol=1e-10)
    np.testing.assert_allclose(cd.magnetization, mag, rtol=1e-10, atol=1e-14)
    assert read_chgcar(p, read_spin=False).spin_mode == SPIN_NONE


def test_noncollinear_four_blocks(tmp_path):
    s = _structure()
    blocks = [rng.random((4, 4, 4)) for _ in range(4)]
    p = tmp_path / "CHGCAR"
    write_volumetric(p, s, blocks)
    _insert_augmentation(p, s.n_atoms)
    cd = read_chgcar(p)
    assert cd.spin_mode == SPIN_NONCOLLINEAR
    assert cd.magnetization.shape == (3, 4, 4, 4)
    np.testing.assert_allclose(cd.magnetization[2], blocks[3], rtol=1e-10)


def test_three_blocks_refused(tmp_path):
    s = _structure()
    p = tmp_path / "CHGCAR"
    write_volumetric(p, s, [rng.random((3, 3, 3)) for _ in range(3)])
    with pytest.raises(ValueError, match="Refusing to guess"):
        read_chgcar(p)


def test_volume_scaling(tmp_path):
    s = _structure()
    rho = np.full((4, 4, 4), 0.25)
    p = tmp_path / "CHGCAR"
    write_volumetric(p, s, [rho])
    raw = read_volumetric(p, scaled_by_volume=False).blocks[0]
    np.testing.assert_allclose(raw, 0.25 * s.volume)
    np.testing.assert_allclose(read_chgcar(p).total, 0.25)


def test_vasp4_negative_scale_cartesian_selective(tmp_path):
    # negative scale = target volume (27 A^3 -> 3 A cube); Cartesian
    # positions are multiplied by the same scale factor
    text = "\n".join([
        "vasp4 style",
        "-27.0",
        "1 0 0", "0 1 0", "0 0 1",
        "1 1",
        "Selective dynamics",
        "Cartesian",
        "0.0 0.0 0.0 T T T",
        "0.5 0.5 0.5 F F F",
        "",
        "2 2 2",
        " ".join(["27.0"] * 8),
    ])
    p = tmp_path / "CHGCAR"
    p.write_text(text + "\n")
    with pytest.warns(UserWarning, match="species"):
        cd = read_chgcar(p)
    np.testing.assert_allclose(cd.structure.lattice, np.eye(3) * 3.0)
    np.testing.assert_allclose(cd.structure.frac_coords[1], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(cd.total, 1.0)
    cd2 = read_chgcar(p, species=["Cs", "Cl"])
    assert cd2.structure.species == ("Cs", "Cl")


def test_potcar_style_labels(tmp_path):
    s = _structure()
    p = tmp_path / "CHGCAR"
    write_volumetric(p, s, [rng.random((3, 3, 3))])
    text = p.read_text().replace("  Fe O\n", "  Fe_pv O/abc\n")
    p.write_text(text)
    assert read_chgcar(p).structure.species == ("Fe", "Fe", "O")


def test_elfcar_ten_per_line_two_blocks(tmp_path):
    s = _structure()
    up, dn = rng.random((5, 5, 4)), rng.random((5, 5, 4))
    p = tmp_path / "ELFCAR"
    write_volumetric(p, s, [up, dn], scaled_by_volume=False, per_line=10)
    elf = read_elfcar(p)
    assert len(elf.blocks) == 2
    np.testing.assert_allclose(elf.blocks[1], dn, rtol=1e-10)


def test_aeccar_sum(tmp_path):
    s = _structure()
    core, val = rng.random((4, 4, 4)), rng.random((4, 4, 4))
    write_volumetric(tmp_path / "AECCAR0", s, [core])
    write_volumetric(tmp_path / "AECCAR2", s, [val])
    ae = read_aeccar(tmp_path / "AECCAR0", tmp_path / "AECCAR2")
    assert ae.all_electron
    np.testing.assert_allclose(ae.total, core + val, rtol=1e-10)


def test_engine_from_vasp_dir(tmp_path):
    s = _structure()
    rho, mag = rng.random((6, 6, 6)), rng.random((6, 6, 6)) - 0.5
    write_volumetric(tmp_path / "CHGCAR", s, [rho, mag])
    write_volumetric(tmp_path / "AECCAR0", s, [rho])
    write_volumetric(tmp_path / "AECCAR2", s, [rho])
    write_volumetric(tmp_path / "ELFCAR", s, [rng.random((3, 3, 3))], scaled_by_volume=False)
    write_volumetric(tmp_path / "LOCPOT", s, [rng.random((6, 6, 6))], scaled_by_volume=False)
    eng = Engine.from_vasp_dir(tmp_path)
    assert set(eng.field_names) == {"rho", "magnetization", "magnetization_abs",
                                    "rho_ae", "elf", "potential"}
    np.testing.assert_allclose(eng["magnetization_abs"].values, np.abs(mag), rtol=1e-10)
    # ELF on its own coarse grid gets its own geometry
    assert eng.geometry((3, 3, 3)).shape == (3, 3, 3)
    assert eng.geometry().shape == (6, 6, 6)
