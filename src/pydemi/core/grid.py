"""
pydemi.core.grid
================
Grid arithmetic on periodic grids: voxel coordinates, integration,
resampling and exact trigonometric interpolation.

``data[i, j, k]`` sits at fractional coordinate (i/n1, j/n2, k/n3);
everything here is periodic.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import fft as sfft
from scipy import ndimage, signal

from ..io.base import FloatArray, Grid, Lattice


def frac_coords(shape: Sequence[int]) -> NDArray[np.float64]:
    """(n1, n2, n3, 3) fractional voxel coordinates."""
    axes = [np.arange(n, dtype=np.float64) / n for n in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def cart_coords(shape: Sequence[int], lattice: Lattice) -> NDArray[np.float64]:
    """(n1, n2, n3, 3) Cartesian voxel coordinates, Angstrom."""
    return np.asarray(frac_coords(shape) @ lattice.matrix, dtype=np.float64)


def integrate(grid: Grid) -> float:
    """int f dV = sum_k f_k dV."""
    return float(np.sum(grid.data, dtype=np.float64) * grid.dV)


def mean(grid: Grid) -> float:
    """Cell average <f>_V."""
    return float(np.mean(grid.data, dtype=np.float64))


def fourier_resample(data: FloatArray, shape: Sequence[int]) -> FloatArray:
    """Band-limited periodic resampling to ``shape`` (zero-padding or truncating the spectrum).

    Exact for band-limited data -- plane-wave grid densities are -- and it
    preserves the cell average, hence the integral (``scipy.signal.resample``
    along each axis).
    """
    out = np.asarray(data)
    for axis, n_new in enumerate(int(n) for n in shape):
        if out.shape[axis] != n_new:
            out = np.asarray(signal.resample(out, n_new, axis=axis))
    return np.asarray(out, dtype=np.asarray(data).dtype)


def linear_resample(data: FloatArray, shape: Sequence[int]) -> FloatArray:
    """Periodic trilinear resampling; stays within the data range (use for the ELF)."""
    src = np.array(data.shape)
    dst = np.array([int(n) for n in shape])
    if np.all(src % dst == 0):
        step = src // dst
        return np.ascontiguousarray(data[::step[0], ::step[1], ::step[2]])
    idx = np.meshgrid(*[np.arange(n) * s / n for n, s in zip(dst, src)], indexing="ij")
    return np.asarray(ndimage.map_coordinates(data, idx, order=1, mode="grid-wrap"),
                      dtype=data.dtype)


def resample(grid: Grid, shape: Sequence[int],
             method: Literal["fourier", "linear"] = "fourier") -> Grid:
    """``grid`` on a different number of voxels."""
    if tuple(shape) == grid.shape:
        return grid
    fn = fourier_resample if method == "fourier" else linear_resample
    return Grid(fn(grid.data, shape), grid.lattice)


def fourier_interpolate(data: FloatArray, frac_points: NDArray[Any],
                        batch: int = 256) -> NDArray[np.float64]:
    """Trigonometric interpolation of a periodic grid field at fractional points.

    Exact at grid points, band-limited in between.
    """
    F = sfft.fftn(np.asarray(data, dtype=np.float64)) / data.size
    n1, n2, n3 = data.shape
    freqs = [sfft.fftfreq(n, 1.0 / n) for n in data.shape]
    u = np.asarray(frac_points, dtype=np.float64).reshape(-1, 3) % 1.0
    out = np.empty(len(u))
    for s in range(0, len(u), batch):
        ub = u[s:s + batch]
        p1, p2, p3 = (np.exp(2j * np.pi * np.outer(freqs[ax], ub[:, ax])) for ax in range(3))
        A = (F.reshape(-1, n3) @ p3).reshape(n1, n2, -1)
        B = np.einsum("ijm,jm->im", A, p2)
        out[s:s + batch] = np.real(np.einsum("im,im->m", B, p1))
    return out


def roll(data: FloatArray, offset: Sequence[int]) -> FloatArray:
    """``data`` at voxel k + offset (periodic)."""
    return np.roll(data, shift=tuple(-int(o) for o in offset), axis=(0, 1, 2))
