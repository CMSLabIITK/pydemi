"""
pydemi.constants
================
Every unit factor, cutoff and threshold pydemi uses. No other module
hard-codes a number with physical meaning; element data live in
``pydemi/data`` (spec §15).

Internal units
--------------
Lengths in Angstrom, densities in electrons / Angstrom^3, potentials in eV.
Quantities the spec defines in atomic units (ELF_D, the local energy
densities g, v, H, the NCI thresholds, the information measures) are
converted explicitly at the point of use with the factors below.
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
#: e/Angstrom^5 -> e/bohr^5 (Laplacian, Hessian)
LAPLACIAN_TO_AU: float = BOHR_ANGSTROM ** 5

HARTREE_EV: float = 27.211386245988
#: e^2 / (4 pi eps0) in eV Angstrom
COULOMB_EV_ANGSTROM: float = HARTREE_EV * BOHR_ANGSTROM

#: Thomas-Fermi constant C_F = (3/10)(3 pi^2)^(2/3) ~= 2.871234 (a.u.)
C_F: float = 0.3 * (3.0 * math.pi ** 2) ** (2.0 / 3.0)
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

#: ELF_D / g / s / information measures: voxels with rho below this
#: (e/bohr^3) are treated as empty -- ELF_D = 0 there, and they are left
#: out of every average over g, H and s
RHO_FLOOR_AU: float = 1e-8
#: floor on the denominators C_F rho^(5/3) and rho in ELF_D and g (a.u.)
ELF_DENOMINATOR_FLOOR: float = 1e-10

# ----------------------------------------------------------------------
# descriptor thresholds (spec §8)
# ----------------------------------------------------------------------

#: f_ELF_localized counts bond-shell voxels with ELF above this
ELF_LOCALIZED: float = 0.5
#: NCI region: s < NCI_S_MAX and rho < NCI_RHO_MAX_AU (e/bohr^3)
NCI_S_MAX: float = 0.5
NCI_RHO_MAX_AU: float = 0.05
#: a local maximum is non-nuclear when farther than this from every nucleus (Angstrom)
NNM_R_CUT: float = 0.8
#: bond census: j is a first-shell neighbour of i when |R_ij| <= (1 + BOND_TOL) d_i
BOND_TOL: float = 0.1
#: a spin-polarized structure is magnetic when sum |m| dV exceeds this per atom (mu_B)
MAGNETIC_TOL: float = 0.01

# ----------------------------------------------------------------------
# promolecule, deformation density, Hirshfeld (spec §6.1, §9)
# ----------------------------------------------------------------------

#: periodic images contribute out to where the free-atom density drops below this (e/Angstrom^3)
PROMOLECULE_TOL: float = 1e-6
#: voxels per edge of the blocks the image sums are evaluated in
IMAGE_SUM_BLOCK: int = 8

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
# potential (spec §6.3)
# ----------------------------------------------------------------------

#: Gaussian width (Angstrom) of the ionic charges in the ionic potential
ION_GAUSSIAN_WIDTH: float = 0.5

# ----------------------------------------------------------------------
# relative tolerances
# ----------------------------------------------------------------------

#: a field whose range max - min is below this fraction of max |f| is uniform
#: (the uniform-density sentinel case, spec §10)
UNIFORM_TOL: float = 1e-12
