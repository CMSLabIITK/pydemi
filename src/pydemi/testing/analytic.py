"""
pydemi.testing.analytic
-----------------------
Analytic test densities with closed-form values, derivatives and radial
statistics. They back the validation suite: any moment, shell fraction,
Laplacian or gradient that pydemi computes on the grid can be checked
against an exact answer.

Two families, both superposed over atoms and periodic images:

``SlaterSuperposition``
    rho_i(r) = N_i zeta_i^3 / pi * exp(-2 zeta_i r)  (hydrogenic 1s, |psi|^2).
    Has a nuclear cusp, like a real density, so it tests the moment and
    fraction machinery; derivatives near r = 0 are not grid-resolvable.

``GaussianSuperposition``
    rho_i(r) = N_i (alpha_i / pi)^{3/2} exp(-alpha_i r^2).
    Smooth everywhere, so it tests derivative accuracy and convergence.

Isolated-atom closed forms (per unit charge) are exposed as functions so
tests can compare grid results for well-separated atoms.
"""

import itertools

import numpy as np
from scipy.special import erf

from ..grid import HESSIAN_PAIRS, Grid
from ..structure import Structure


class _RadialSuperposition:
    """Sum over atoms and images of a spherical profile f(r)."""

    def __init__(self, structure: Structure, params, electrons=1.0, tol=1e-14):
        self.structure = structure
        n = structure.n_atoms
        self.params = self._per_atom(params, n)
        self.electrons = self._per_atom(electrons, n)
        self.tol = tol

    def _per_atom(self, value, n):
        if isinstance(value, dict):
            return np.array([value[s] for s in self.structure.species], dtype=float)
        arr = np.broadcast_to(np.asarray(value, dtype=float), (n,))
        return arr.copy()

    # radial profile and its first two derivatives, per unit charge
    def _profile(self, r, p):
        raise NotImplementedError

    def _cutoff(self, p):
        raise NotImplementedError

    def _images(self, center, cutoff):
        """Lattice shifts whose image of ``center`` can reach the cell."""
        lat = self.structure.lattice
        h = np.linalg.norm(self.structure.inv_lattice, axis=0)
        reps = [int(np.ceil(cutoff * hj)) + 1 for hj in h]
        shifts = np.array(list(itertools.product(*[range(-r, r + 1) for r in reps])),
                          dtype=float) @ lat
        corners = np.array(list(itertools.product((0, 1), repeat=3)), dtype=float) @ lat
        mid = corners.mean(axis=0)
        radius = np.linalg.norm(corners - mid, axis=1).max()
        keep = np.linalg.norm(center + shifts - mid, axis=1) <= cutoff + radius
        return shifts[keep]

    def evaluate(self, points, derivatives=True):
        """rho, grad (N,3), packed Hessian (N,6) at Cartesian ``points`` (N,3)."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        rho = np.zeros(points.shape[0])
        grad = np.zeros((points.shape[0], 3)) if derivatives else None
        hess = np.zeros((points.shape[0], 6)) if derivatives else None
        centers = self.structure.cart_coords
        for i in range(self.structure.n_atoms):
            p, N = self.params[i], self.electrons[i]
            cutoff = self._cutoff(p)
            for shift in self._images(centers[i], cutoff):
                d = points - (centers[i] + shift)
                r = np.linalg.norm(d, axis=1)
                near = r < cutoff
                if not near.any():
                    continue
                dn, rn = d[near], r[near]
                f, f1, f2 = self._profile(rn, p)
                rho[near] += N * f
                if not derivatives:
                    continue
                safe = np.maximum(rn, 1e-12)
                u = dn / safe[:, None]
                grad[near] += N * f1[:, None] * u
                # H = f'' u u^T + (f'/r)(I - u u^T)
                f1r = f1 / safe
                for q, (a, b) in enumerate(HESSIAN_PAIRS):
                    uu = u[:, a] * u[:, b]
                    hess[near, q] += N * (f2 * uu + f1r * ((a == b) - uu))
        return rho, grad, hess

    def on_grid(self, grid: Grid, derivatives=True):
        """Values on ``grid``: rho (nx,ny,nz), grad (...,3), Hessian (...,6)."""
        rho, grad, hess = self.evaluate(grid.cart_coords().reshape(-1, 3), derivatives)
        s = grid.shape
        if not derivatives:
            return rho.reshape(s), None, None
        return rho.reshape(s), grad.reshape(s + (3,)), hess.reshape(s + (6,))


class SlaterSuperposition(_RadialSuperposition):
    """Superposed 1s Slater densities; ``params`` are exponents zeta (1/Angstrom)."""

    def _profile(self, r, zeta):
        f = zeta ** 3 / np.pi * np.exp(-2.0 * zeta * r)
        return f, -2.0 * zeta * f, 4.0 * zeta ** 2 * f

    def _cutoff(self, zeta):
        # exp(-2 zeta r) < tol
        return -np.log(self.tol) / (2.0 * zeta)


class GaussianSuperposition(_RadialSuperposition):
    """Superposed normalized Gaussians; ``params`` are exponents alpha (1/Angstrom^2)."""

    def _profile(self, r, alpha):
        f = (alpha / np.pi) ** 1.5 * np.exp(-alpha * r ** 2)
        return f, -2.0 * alpha * r * f, (4.0 * alpha ** 2 * r ** 2 - 2.0 * alpha) * f

    def _cutoff(self, alpha):
        return np.sqrt(-np.log(self.tol) / alpha)


# ----------------------------------------------------------------------
# isolated-atom closed forms (per unit charge)
# ----------------------------------------------------------------------

def slater_moment(n: int, zeta: float) -> float:
    """<r^n> for rho ~ exp(-2 zeta r): (n+2)! / (2 (2 zeta)^n)."""
    from math import factorial
    return factorial(n + 2) / (2.0 * (2.0 * zeta) ** n)


def slater_fraction_within(R, zeta):
    """Fraction of a 1s Slater density inside radius R."""
    x = 2.0 * zeta * np.asarray(R, dtype=float)
    return 1.0 - np.exp(-x) * (1.0 + x + 0.5 * x ** 2)


def gaussian_moment(n: int, alpha: float) -> float:
    """<r^n> for rho ~ exp(-alpha r^2): Gamma((n+3)/2) / (Gamma(3/2) alpha^(n/2))."""
    from math import gamma
    return gamma((n + 3) / 2.0) / (gamma(1.5) * alpha ** (n / 2.0))


def gaussian_fraction_within(R, alpha):
    """Fraction of a normalized Gaussian density inside radius R."""
    R = np.asarray(R, dtype=float)
    s = np.sqrt(alpha) * R
    return erf(s) - 2.0 / np.sqrt(np.pi) * s * np.exp(-s ** 2)
