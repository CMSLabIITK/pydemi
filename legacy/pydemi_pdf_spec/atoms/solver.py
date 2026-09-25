"""
pydemi.atoms.solver
-------------------
Spherical, non-spin-polarized, non-relativistic LDA free-atom solver
(Hartree atomic units throughout). It supplies the free-atom reference
densities behind the deformation density (Family A) and Hirshfeld
partition (entry 101) -- the "tabulated atomic-DFT radial densities" route
of the reference, computed on demand instead of shipped as a table.

Radial equation on a logarithmic grid
-------------------------------------
With r = e^x and u(r) = r P(r) = e^{x/2} phi(x), the radial Kohn-Sham
equation -u''/2 + [V + l(l+1)/(2 r^2)] u = eps u becomes

    -phi''/2 + [(l + 1/2)^2 / 2 + r^2 V] phi = eps r^2 phi,

a generalized symmetric eigenproblem with a tridiagonal left side (second-
order differences in x) and diagonal metric r^2. Scaling by r^{-1} makes it
an ordinary symmetric tridiagonal problem, solved for the lowest states of
each l with LAPACK's tridiagonal eigensolver. With psi = r phi,
sum_x psi^2 h = 1 and the orbital's density is psi^2 / (4 pi r^3).

That matrix is strongly graded (diagonal ~1e25 at r = 1e-10 bohr against
eigenvalues of order 1), so the default bisection tolerance eps * ||C||
would swamp the eigenvalues; an explicit absolute tolerance scaled by Z^2
restores full accuracy (bisection with Sturm counts is accurate on graded
tridiagonals). The discretization is second order in h: eigenvalues carry
a relative error ~1e-5 on the default grid, far below what a promolecule
reference needs.

Exchange-correlation: Slater exchange and Perdew-Wang 1992 correlation
(unpolarized); VWN5 is available and reproduces the NIST LDA reference
data (Kotochigova et al., Phys. Rev. A 55, 191 (1997)). Occupations are the spherically averaged ground-state
configuration (pymatgen), fractional within open shells.

Limitations: non-relativistic (valence shapes of 5d/6s/6p elements are
off by the missing relativistic contraction), spin-restricted, and LDA.
Adequate for a promolecule reference, which is what it is for.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh_tridiagonal

L_SYMBOL = {"s": 0, "p": 1, "d": 2, "f": 3}


@dataclass(frozen=True)
class RadialGrid:
    # s orbitals behave as phi ~ r^{1/2} toward the nucleus, so the inner
    # Dirichlet boundary must sit far inside: r_min = 1e-7 bohr already
    # costs ~0.4 mHa on Ne; 1e-10 makes the error negligible up to Z ~ 100
    x_min: float = np.log(1e-10)
    x_max: float = np.log(80.0)
    n: int = 8000

    @property
    def x(self):
        return np.linspace(self.x_min, self.x_max, self.n)

    @property
    def h(self):
        return (self.x_max - self.x_min) / (self.n - 1)

    @property
    def r(self):
        return np.exp(self.x)


@dataclass
class AtomResult:
    Z: int
    r: np.ndarray                      # bohr
    orbitals: list                     # [(n, l, occupation, eigenvalue, density(r))]
    density: np.ndarray                # total, electrons / bohr^3
    energy: float                      # hartree
    iterations: int
    converged: bool
    extra: dict = field(default_factory=dict)

    def orbital_density(self, selection) -> np.ndarray:
        """Sum of orbital densities for (n, l) pairs in ``selection``."""
        keep = set(selection)
        return sum((o[4] for o in self.orbitals if (o[0], o[1]) in keep),
                   np.zeros_like(self.r))


# ---------------------------------------------------------------- xc

def _vwn_c(rs):
    """Vosko-Wilk-Nusair (VWN5) paramagnetic correlation: (eps_c, v_c)."""
    A, x0, b, c = 0.0310907, -0.10498, 3.72744, 12.9352
    x = np.sqrt(rs)
    X = x * x + b * x + c
    X0 = x0 * x0 + b * x0 + c
    Q = np.sqrt(4.0 * c - b * b)
    at = np.arctan(Q / (2.0 * x + b))
    ec = A * (np.log(x * x / X) + 2.0 * b / Q * at
              - b * x0 / X0 * (np.log((x - x0) ** 2 / X) + 2.0 * (b + 2.0 * x0) / Q * at))
    den = (2.0 * x + b) ** 2 + Q * Q
    dec = A * (2.0 / x - (2.0 * x + b) / X - 4.0 * b / den
               - b * x0 / X0 * (2.0 / (x - x0) - (2.0 * x + b) / X - 4.0 * (b + 2.0 * x0) / den))
    return ec, ec - x / 6.0 * dec


def _lda_xc(n, correlation="pw92"):
    """(eps_xc, v_xc) per electron for density n (unpolarized)."""
    n = np.maximum(n, 1e-30)
    kx = (3.0 / np.pi) ** (1.0 / 3.0)
    ex = -0.75 * kx * n ** (1.0 / 3.0)
    vx = -kx * n ** (1.0 / 3.0)
    rs = (3.0 / (4.0 * np.pi * n)) ** (1.0 / 3.0)
    if correlation == "vwn":
        ec, vc = _vwn_c(rs)
        return ex + ec, vx + vc
    A, a1, b1, b2, b3, b4 = 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294
    srs = np.sqrt(rs)
    Q = 2.0 * A * (b1 * srs + b2 * rs + b3 * rs * srs + b4 * rs * rs)
    dQ = A * (b1 / srs + 2.0 * b2 + 3.0 * b3 * srs + 4.0 * b4 * rs)
    L = np.log1p(1.0 / Q)
    ec = -2.0 * A * (1.0 + a1 * rs) * L
    dec = -2.0 * A * a1 * L + 2.0 * A * (1.0 + a1 * rs) * dQ / (Q * Q + Q)
    vc = ec - rs / 3.0 * dec
    return ex + ec, vx + vc


# ---------------------------------------------------------------- Hartree

def _cumtrapz(y, x):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def _hartree(n, r, x):
    """V_H(r) = 4 pi [ (1/r) int_0^r n r'^2 dr' + int_r^inf n r' dr' ]."""
    inner = _cumtrapz(n * r ** 3, x)             # dr = r dx
    outer_rev = _cumtrapz((n * r ** 2)[::-1], -x[::-1])
    outer = outer_rev[::-1]
    return 4.0 * np.pi * (inner / r + outer)


def _thomas_fermi_guess(Z, r):
    """Screened nuclear potential from a rational fit to the Thomas-Fermi function."""
    b = 0.8853 * Z ** (-1.0 / 3.0)
    x = r / b
    sx = np.sqrt(x)
    phi = 1.0 / (1.0 + 0.02747 * sx + 1.243 * x - 0.1486 * x * sx
                 + 0.2302 * x * x + 0.007298 * x * x * sx + 0.006944 * x ** 3)
    return -np.maximum(Z * phi, 1.0) / r


# ---------------------------------------------------------------- solver

def _solve_l(V, l, grid, n_states, Z=1.0):
    r, h = grid.r, grid.h
    diag = (1.0 / (h * h) + (l + 0.5) ** 2 / 2.0 + r * r * V) / (r * r)
    off = -0.5 / (h * h) / (r[:-1] * r[1:])
    w, vec = eigh_tridiagonal(diag, off, select="i", select_range=(0, n_states - 1),
                              lapack_driver="stebz", tol=1e-13 * max(1.0, Z * Z))
    psi = vec / np.sqrt(h)                        # sum psi^2 h = 1
    return w, psi


def solve_atom(Z: int, configuration, grid: RadialGrid = RadialGrid(),
               mixing: float = 0.4, tol: float = 1e-9, max_iter: int = 400,
               correlation: str = "pw92") -> AtomResult:
    """Self-consistent LDA atom.

    ``configuration``: iterable of (n, l, occupation) with l an int or s/p/d/f.
    ``correlation``: "pw92" (default) or "vwn" (VWN5, as in the NIST
    atomic reference data, used to validate the solver).
    """
    config = [(int(n), L_SYMBOL.get(l, l) if isinstance(l, str) else int(l), float(f))
              for n, l, f in configuration]
    n_electrons = sum(f for _, _, f in config)
    r, x = grid.r, grid.x
    V_in = _thomas_fermi_guess(Z, r)
    by_l = {}
    for n, l, f in config:
        by_l[l] = max(by_l.get(l, 0), n - l)

    converged = False
    for it in range(1, max_iter + 1):
        orbitals, density = [], np.zeros_like(r)
        eig_sum = 0.0
        for l, count in by_l.items():
            w, psi = _solve_l(V_in, l, grid, count, Z)
            for n, ll, f in config:
                if ll != l:
                    continue
                k = n - l - 1
                rho_k = psi[:, k] ** 2 / (4.0 * np.pi * r ** 3)
                orbitals.append((n, l, f, float(w[k]), rho_k))
                density += f * rho_k
                eig_sum += f * w[k]
        VH = _hartree(density, r, x)
        exc, vxc = _lda_xc(density, correlation)
        V_out = -Z / r + VH + vxc
        # potential residual weighted by the density-bearing region
        err = np.sqrt(np.sum((V_out - V_in) ** 2 * density * r ** 3) * grid.h / n_electrons)
        if err < tol:
            converged = True
            break
        V_in = (1.0 - mixing) * V_in + mixing * V_out

    w3 = 4.0 * np.pi * r ** 3 * grid.h            # int f dV = sum f 4 pi r^3 h
    energy = (eig_sum - 0.5 * np.sum(density * VH * w3)
              + np.sum(density * (exc - vxc) * w3))
    orbitals.sort(key=lambda o: o[3])
    return AtomResult(Z, r, orbitals, density, float(energy), it, converged,
                      extra={"electrons": float(np.sum(density * w3)),
                             "residual": float(err)})
