"""
pydemi.constants
================
Every unit factor, cutoff and threshold pydemi uses. No other module
hard-codes a number with physical meaning; element data live in
``pydemi/data`` (spec §15).

Internal units
--------------
Lengths in Angstrom, densities in electrons / Angstrom^3, potentials in eV.
Quantities the spec defines in atomic units (the NCI thresholds, the
information measures) are converted explicitly at the point of use with
the factors below.
"""

from __future__ import annotations

import math

# ----------------------------------------------------------------------
# units
# ----------------------------------------------------------------------

BOHR_ANGSTROM: float = 0.529177210903           # CODATA 2018
ANGSTROM_BOHR: float = 1.0 / BOHR_ANGSTROM

#: e/Angstrom^3 -> e/bohr^3
DENSITY_TO_AU: float = BOHR_ANGSTROM ** 3
#: e/bohr^3 -> e/Angstrom^3
DENSITY_FROM_AU: float = 1.0 / DENSITY_TO_AU
#: e/Angstrom^4 -> e/bohr^4 (density gradient)
GRADIENT_TO_AU: float = BOHR_ANGSTROM ** 4

HARTREE_EV: float = 27.211386245988
#: e^2 / (4 pi eps0) in eV Angstrom
COULOMB_EV_ANGSTROM: float = HARTREE_EV * BOHR_ANGSTROM

#: prefactor of the reduced density gradient, 2 (3 pi^2)^(1/3)
S_PREFACTOR: float = 2.0 * (3.0 * math.pi ** 2) ** (1.0 / 3.0)

# ----------------------------------------------------------------------
# shells (spec §5)
# ----------------------------------------------------------------------

#: core: r <= c1 (Angstrom)
SHELL_C1: float = 0.8
#: bond: c1 < r <= c2; interstitial: r > c2 (Angstrom)
SHELL_C2: float = 1.5

# ----------------------------------------------------------------------
# derivatives (spec §4)
# ----------------------------------------------------------------------

DERIVATIVE_BACKEND: str = "fft"
LAPLACIAN_METHOD: str = "metric"
#: order of the central finite differences for backend "fd"
FD_ORDER: int = 4
#: voxels per chunk for the Hessian eigendecomposition
HESSIAN_CHUNK: int = 1 << 18

# ----------------------------------------------------------------------
# low-density guards (spec §6.2, §10)
# ----------------------------------------------------------------------

#: reduced gradient s / information measures: voxels with rho below this
#: (e/bohr^3) are treated as empty (s = inf there, so never NCI)
RHO_FLOOR_AU: float = 1e-8

# ----------------------------------------------------------------------
# descriptor thresholds (spec §8)
# ----------------------------------------------------------------------

#: NCI region: s < NCI_S_MAX and rho < NCI_RHO_MAX_AU (e/bohr^3)
NCI_S_MAX: float = 0.5
NCI_RHO_MAX_AU: float = 0.05
#: a local maximum is non-nuclear when farther than this from every nucleus (Angstrom)
NNM_R_CUT: float = 0.8
#: bond census: j is a first-shell neighbour of i when |R_ij| <= (1 + BOND_TOL) d_i
BOND_TOL: float = 0.1

# ----------------------------------------------------------------------
# partitions (spec §9)
# ----------------------------------------------------------------------

#: Becke: weights go to the BECKE_CELLS nearest atom images; each cell
#: product runs over the BECKE_K nearest images (see pydemi.core.partition)
BECKE_K: int = 60
BECKE_CELLS: int = 8
#: pair weights below this are dropped from smooth partitions
MIN_PAIR_WEIGHT: float = 1e-10

# ----------------------------------------------------------------------
# relative tolerances
# ----------------------------------------------------------------------

#: distances within this of each other (Angstrom) are equal: shell and cutoff tests use it,
#: and equidistant atom images are then ordered by a geometric rule (pydemi.core.geometry)
GEOMETRY_EPS: float = 1e-8
#: a field whose range max - min is below this fraction of max |f| is uniform
#: (the uniform-density sentinel case, spec §10)
UNIFORM_TOL: float = 1e-12
