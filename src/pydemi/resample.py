"""
pydemi.resample
---------------
Fourier resampling of periodic grid fields, so descriptors from structures
with different FFT meshes can be compared at a common grid spacing.

Resampling zero-pads or truncates the spectrum along each axis
(``scipy.signal.resample``), which is exact for band-limited data -- VASP
grid densities are -- and preserves the cell average, hence the integral.
Truncation discards the highest frequencies, i.e. it produces what a
coarser FFT grid would have represented.
"""

from typing import Optional

import numpy as np
from scipy.signal import resample as _resample1d

from .engine import Engine

_SMALL_PRIMES = (2, 3, 5, 7)


def fft_friendly(n: int) -> int:
    """Smallest n' >= n whose prime factors are all in {2, 3, 5, 7}."""
    m = max(1, int(n))
    while True:
        k = m
        for p in _SMALL_PRIMES:
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


def shape_for_spacing(lattice, spacing: float, friendly: bool = True) -> tuple:
    """Grid shape giving at most ``spacing`` Angstrom along every lattice vector."""
    lengths = np.linalg.norm(np.asarray(lattice, float), axis=1)
    n = [int(np.ceil(L / spacing)) for L in lengths]
    return tuple(fft_friendly(k) if friendly else k for k in n)


def fourier_resample(values: np.ndarray, shape) -> np.ndarray:
    """Periodic band-limited resampling of a 3-D field to ``shape``."""
    out = np.asarray(values, float)
    for axis, n in enumerate(shape):
        if out.shape[axis] != n:
            out = _resample1d(out, int(n), axis=axis)
    return out


def resample_engine(engine: Engine, spacing: Optional[float] = None,
                    shape: Optional[tuple] = None, scale: Optional[float] = None) -> Engine:
    """A new Engine with every field resampled.

    Give exactly one of ``spacing`` (Angstrom; FFT-friendly shapes),
    ``shape`` (for the fields on the main grid) or ``scale`` (multiplies each
    field's own shape). Fields on other grids (e.g. ELFCAR) keep their
    ratio to the main grid. Derived fields (ELF_D, potentials,
    promolecules) are not copied; they are recomputed on demand.
    """
    if sum(x is not None for x in (spacing, shape, scale)) != 1:
        raise ValueError("give exactly one of spacing, shape or scale")
    main = engine.grid().shape
    if spacing is not None:
        target_main = shape_for_spacing(engine.structure.lattice, spacing)
    elif shape is not None:
        target_main = tuple(int(n) for n in shape)
    else:
        target_main = tuple(max(2, int(round(n * scale))) for n in main)

    derived = ("elf_d", "hartree_potential", "promolecule")
    new = Engine(engine.structure, method=engine.method, shells=engine.shells,
                 spin_mode=engine.spin_mode, zval=engine.zval,
                 reference=engine._reference)
    for name in engine.field_names:
        if name.startswith(derived):
            continue
        f = engine[name]
        s = f.grid.shape
        target = tuple(max(2, int(round(t * n / m))) for t, n, m in zip(target_main, s, main))
        new.add_field(name, fourier_resample(f.values, target))
    if engine.magnetization_vector is not None:
        mv = engine.magnetization_vector
        new.magnetization_vector = np.stack([fourier_resample(c, target_main) for c in mv])
    return new
