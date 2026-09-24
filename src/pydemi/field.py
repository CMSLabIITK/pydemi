"""
pydemi.field
------------
A scalar field on a periodic grid with lazily cached derivatives.

The descriptor families run one machinery (moments, fractions,
anisotropy, partitions) over several physical fields -- rho, |m|, ELF,
the electrostatic potential. ``Field`` is the common currency: each
derived quantity is computed at most once per field, so the marginal
cost of an extra descriptor stays near zero.
"""

from functools import cached_property

import numpy as np

from .grid import Grid, hessian_eigenvalues, unpack_hessian


class Field:
    def __init__(self, values, grid: Grid, name: str = "field", method: str = "fd"):
        values = np.asarray(values, dtype=float)
        if values.shape != grid.shape:
            raise ValueError(f"{name}: values {values.shape} != grid {grid.shape}")
        self.values = values
        self.grid = grid
        self.name = name
        self.method = method

    def __repr__(self):
        return f"Field({self.name!r}, shape={self.grid.shape}, method={self.method!r})"

    @cached_property
    def gradient(self) -> np.ndarray:
        """(nx, ny, nz, 3) Cartesian gradient."""
        return self.grid.gradient(self.values, self.method)

    @cached_property
    def gradient_norm(self) -> np.ndarray:
        return np.linalg.norm(self.gradient, axis=-1)

    @cached_property
    def hessian(self) -> np.ndarray:
        """(nx, ny, nz, 6) packed symmetric Hessian (see grid.HESSIAN_PAIRS)."""
        return self.grid.hessian(self.values, self.method)

    def hessian_full(self) -> np.ndarray:
        """(nx, ny, nz, 3, 3) Hessian (allocated on each call)."""
        return unpack_hessian(self.hessian)

    @cached_property
    def laplacian(self) -> np.ndarray:
        # reuse the Hessian's trace when it has already been computed
        if "hessian" in self.__dict__:
            return self.hessian[..., 0] + self.hessian[..., 1] + self.hessian[..., 2]
        return self.grid.laplacian(self.values, self.method)

    @cached_property
    def hessian_eigenvalues(self) -> np.ndarray:
        """(nx, ny, nz, 3) ascending lambda1 <= lambda2 <= lambda3."""
        return hessian_eigenvalues(self.hessian)

    def integral(self) -> float:
        return self.grid.integrate(self.values)

    def clear_cache(self) -> None:
        for key in ("gradient", "gradient_norm", "hessian", "laplacian",
                    "hessian_eigenvalues"):
            self.__dict__.pop(key, None)
