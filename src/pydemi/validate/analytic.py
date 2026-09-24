"""
pydemi.validate.analytic
========================
Analytic reference densities with closed-form descriptor values (spec §11).

``SlaterSuperposition``
    rho_i(r) = N_i (zeta_i^3 / pi) exp(-2 zeta_i r)   (hydrogenic 1s, normalized to N_i).
    Has a nuclear cusp like a real density. For one atom in a large box:

        int rho dV = 1,  m1 = 3/(2 zeta),  m2 = 3/zeta^2,  sigma_r2 = 3/(4 zeta^2),
        zeta (anisotropy) = 0 exactly (the gradient is radial),
        lap rho = rho (4 zeta^2 - 4 zeta / r)  -> negative for r < 1/zeta,
        lnf = volume fraction of the box with r < 1/zeta.

``GaussianSuperposition``
    rho_i(r) = N_i (alpha_i / pi)^(3/2) exp(-alpha_i r^2). Smooth everywhere, so
    it measures derivative accuracy and convergence order.

Both are summed over atoms and all periodic images that reach the cell, and
evaluate the exact gradient and Hessian with the density.

``uniform`` builds a constant density: every sentinel case (spec §10).
"""

from __future__ import annotations

import itertools
import math
from typing import Mapping, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

from ..core.derivatives import PAIRS
from ..core.grid import cart_coords
from ..io.base import Grid, Lattice, Structure, VolumetricData

Param = Union[float, Sequence[float], Mapping[str, float]]
F64 = NDArray[np.float64]


class _RadialSuperposition:
    """Sum over atoms and periodic images of a spherical profile f(r)."""

    def __init__(self, structure: Structure, params: Param, electrons: Param = 1.0,
                 tol: float = 1e-14) -> None:
        self.structure = structure
        self.params = self._per_atom(params)
        self.electrons = self._per_atom(electrons)
        self.tol = tol

    def _per_atom(self, value: Param) -> F64:
        n = self.structure.n_atoms
        if isinstance(value, Mapping):
            return np.array([value[s] for s in self.structure.species], dtype=np.float64)
        return np.array(np.broadcast_to(np.asarray(value, dtype=np.float64), (n,)))

    def profile(self, r: F64, p: float) -> tuple[F64, F64, F64]:
        """f(r), f'(r), f''(r) per unit charge."""
        raise NotImplementedError

    def cutoff(self, p: float) -> float:
        raise NotImplementedError

    def _images(self, center: F64, cutoff: float) -> F64:
        lat = self.structure.lattice
        h = np.linalg.norm(lat.inverse, axis=0)
        reps = [int(np.ceil(cutoff * hj)) + 1 for hj in h]
        shifts = np.array(list(itertools.product(*[range(-r, r + 1) for r in reps])),
                          dtype=np.float64) @ lat.matrix
        corners = np.array(list(itertools.product((0, 1), repeat=3)), dtype=np.float64) @ lat.matrix
        mid = corners.mean(axis=0)
        radius = np.linalg.norm(corners - mid, axis=1).max()
        keep = np.linalg.norm(center + shifts - mid, axis=1) <= cutoff + radius
        return np.asarray(shifts[keep])

    def evaluate(self, points: F64) -> tuple[F64, F64, F64]:
        """rho (N,), gradient (N, 3), Hessian (N, 3, 3) at Cartesian points."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        rho = np.zeros(len(pts))
        grad = np.zeros((len(pts), 3))
        hess = np.zeros((len(pts), 3, 3))
        centers = self.structure.cart_coords
        for i in range(self.structure.n_atoms):
            p, N = float(self.params[i]), float(self.electrons[i])
            cut = self.cutoff(p)
            for shift in self._images(centers[i], cut):
                d = pts - (centers[i] + shift)
                r = np.linalg.norm(d, axis=1)
                near = r < cut
                if not near.any():
                    continue
                dn, rn = d[near], r[near]
                f, f1, f2 = self.profile(rn, p)
                rho[near] += N * f
                safe = np.maximum(rn, 1e-12)
                u = dn / safe[:, None]
                grad[near] += N * f1[:, None] * u
                f1r = f1 / safe
                for a, b in PAIRS:
                    uu = u[:, a] * u[:, b]
                    val = N * (f2 * uu + f1r * (float(a == b) - uu))
                    hess[near, a, b] += val
                    if a != b:
                        hess[near, b, a] += val
        return rho, grad, hess

    def on_grid(self, shape: Sequence[int]) -> tuple[F64, F64, F64]:
        """rho (n1,n2,n3), gradient (...,3), Hessian (...,3,3) on a grid of ``shape``."""
        s = tuple(int(n) for n in shape)
        rho, grad, hess = self.evaluate(cart_coords(s, self.structure.lattice).reshape(-1, 3))
        return rho.reshape(s), grad.reshape(s + (3,)), hess.reshape(s + (3, 3))

    def volumetric(self, shape: Sequence[int], magnetization: Optional["_RadialSuperposition"] = None
                   ) -> VolumetricData:
        """VolumetricData with this density (and, optionally, a collinear magnetization)."""
        rho, _, _ = self.on_grid(shape)
        lat = self.structure.lattice
        mag = None
        if magnetization is not None:
            mag = Grid(magnetization.on_grid(shape)[0], lat)
        return VolumetricData(self.structure, Grid(rho, lat), magnetization=mag)


class SlaterSuperposition(_RadialSuperposition):
    """Superposed 1s Slater densities; ``params`` are exponents zeta (1/Angstrom)."""

    def profile(self, r: F64, p: float) -> tuple[F64, F64, F64]:
        f = p ** 3 / np.pi * np.exp(-2.0 * p * r)
        return f, -2.0 * p * f, 4.0 * p ** 2 * f

    def cutoff(self, p: float) -> float:
        return -math.log(self.tol) / (2.0 * p)


class GaussianSuperposition(_RadialSuperposition):
    """Superposed normalized Gaussians; ``params`` are exponents alpha (1/Angstrom^2)."""

    def profile(self, r: F64, p: float) -> tuple[F64, F64, F64]:
        f = (p / np.pi) ** 1.5 * np.exp(-p * r ** 2)
        return f, -2.0 * p * r * f, (4.0 * p ** 2 * r ** 2 - 2.0 * p) * f

    def cutoff(self, p: float) -> float:
        return math.sqrt(-math.log(self.tol) / p)


def uniform(structure: Structure, shape: Sequence[int], value: float = 0.1) -> VolumetricData:
    """A constant density: exercises every sentinel case."""
    s = tuple(int(n) for n in shape)
    return VolumetricData(structure, Grid(np.full(s, float(value)), structure.lattice))


def cubic_cell(a: float, species: Sequence[str] = ("H",),
               frac: Sequence[Sequence[float]] = ((0.5, 0.5, 0.5),)) -> Structure:
    return Structure(Lattice(np.eye(3) * a), list(species), np.asarray(frac, dtype=np.float64))


# ----------------------------------------------------------------------
# isolated-atom closed forms (per unit charge)
# ----------------------------------------------------------------------

def slater_moment(n: int, zeta: float) -> float:
    """<r^n> for rho ~ exp(-2 zeta r): (n+2)! / (2 (2 zeta)^n); m1 = 3/(2 zeta), m2 = 3/zeta^2."""
    return float(math.factorial(n + 2) / (2.0 * (2.0 * zeta) ** n))


def slater_fraction_within(R: float, zeta: float) -> float:
    """Fraction of a 1s Slater density inside radius R."""
    x = 2.0 * zeta * R
    return 1.0 - math.exp(-x) * (1.0 + x + 0.5 * x * x)


def gaussian_moment(n: int, alpha: float) -> float:
    """<r^n> for rho ~ exp(-alpha r^2)."""
    return float(math.gamma((n + 3) / 2.0) / (math.gamma(1.5) * alpha ** (n / 2.0)))


def gaussian_fraction_within(R: float, alpha: float) -> float:
    s = math.sqrt(alpha) * R
    return float(erf(s)) - 2.0 / math.sqrt(math.pi) * s * math.exp(-s * s)


def slater_lnf(zeta: float, volume: float) -> float:
    """Volume fraction of a box of ``volume`` with r < 1/zeta (sphere inside the box)."""
    return 4.0 / 3.0 * math.pi / zeta ** 3 / volume
