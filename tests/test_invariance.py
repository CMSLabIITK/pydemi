"""Milestone 5: invariance harness (spec §10).

Every registered descriptor is run on a low-symmetry test crystal and on its
2x2x2 supercell, a rigid translation and a rigid rotation, and must agree to
1e-6 relative. The test is parametrized over the registry, so descriptors
added later are checked automatically.
"""

import functools

import numpy as np
import pytest

import pydemi
from pydemi.io.base import Grid, Lattice, Structure, VolumetricData
from pydemi.validate.analytic import GaussianSuperposition
from pydemi.validate.invariance import compare, rotate, rotation, supercell, translate

LATTICE = Lattice(np.array([[3.9, 0.0, 0.0], [0.8, 4.1, 0.0], [0.6, 0.5, 4.3]]))
FRAC = [[0.05, 0.10, 0.12], [0.52, 0.43, 0.58], [0.27, 0.71, 0.36]]
SHAPE = (20, 21, 22)
EXT = ("paw",)


@functools.lru_cache(maxsize=None)
def base() -> VolumetricData:
    s = Structure(LATTICE, ["Fe", "Fe", "O"], FRAC)
    rho = GaussianSuperposition(s, [1.6, 1.9, 2.4], [8.0, 8.0, 6.0], tol=1e-18)
    mag = GaussianSuperposition(s, [2.2, 2.0, 3.0], [2.0, -1.2, 0.1], tol=1e-18)
    vd = rho.volumetric(SHAPE, magnetization=mag)
    r = vd.rho.data
    vd.elf = Grid(np.clip(0.5 + 0.4 * np.sin(r / r.max() * 6.0), 0.0, 1.0), s.lattice)
    vd.potential = Grid(-3.0 * r / r.max() + 0.1 * np.cos(r), s.lattice)
    return vd


TRANSFORMS = {
    "supercell_2x2x2": lambda vd: supercell(vd, (2, 2, 2)),
    "translation": lambda vd: translate(vd, (3, 7, 11)),
    "rotation": lambda vd: rotate(vd, rotation((1.0, 2.0, 0.5), 37.0)),
}


@functools.lru_cache(maxsize=None)
def features(name: str) -> dict:
    vd = base() if name == "base" else TRANSFORMS[name](base())
    return pydemi.featurize(vd, extensions=EXT)


NAMES = pydemi.descriptor_names(extensions=EXT)


@pytest.mark.parametrize("transform", sorted(TRANSFORMS))
@pytest.mark.parametrize("name", NAMES)
def test_descriptor_is_invariant(name, transform):
    a, b = features("base")[name], features(transform)[name]
    assert not compare({name: a}, {name: b}), f"{name}: {a!r} vs {b!r} under {transform}"


def test_transforms_are_what_they_claim():
    vd = base()
    sc = supercell(vd)
    assert sc.structure.n_atoms == 8 * vd.structure.n_atoms
    assert np.isclose(sc.structure.volume, 8 * vd.structure.volume)
    assert np.isclose(sc.rho.data.sum() * sc.rho.dV, 8 * vd.rho.data.sum() * vd.rho.dV)
    rot = rotate(vd, rotation((1.0, 2.0, 0.5), 37.0))
    assert np.isclose(rot.structure.volume, vd.structure.volume)
    np.testing.assert_allclose(rot.lattice.covariant_metric, vd.lattice.covariant_metric)
