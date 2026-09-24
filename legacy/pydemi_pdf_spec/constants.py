"""
pydemi.constants
----------------
Unit conventions and physical constants.

Internal units
--------------
pydemi stores lengths in Angstrom and densities in electrons / Angstrom^3,
matching VASP output and the absolute shell cutoffs of the descriptor
reference (0.8 / 1.5 Angstrom). Formulas that the reference defines in
atomic units (ELF_D, Abramov energy densities, NCI thresholds) convert
explicitly using the factors below at the point of use.
"""

import numpy as np

BOHR_ANGSTROM = 0.529177210903  # CODATA 2018
ANGSTROM_BOHR = 1.0 / BOHR_ANGSTROM

# e/Angstrom^3 -> e/bohr^3
DENSITY_TO_AU = BOHR_ANGSTROM ** 3
# e/bohr^3 -> e/Angstrom^3
DENSITY_FROM_AU = 1.0 / DENSITY_TO_AU

# e/Angstrom^4 -> e/bohr^4 (density gradient), e/Angstrom^5 -> e/bohr^5 (Laplacian)
GRADIENT_TO_AU = BOHR_ANGSTROM ** 4
LAPLACIAN_TO_AU = BOHR_ANGSTROM ** 5

HARTREE_EV = 27.211386245988
# e^2 / (4 pi eps0) in eV * Angstrom
COULOMB_EV_ANGSTROM = HARTREE_EV * BOHR_ANGSTROM

# Thomas-Fermi kinetic constant, C_F = (3/10)(3 pi^2)^(2/3) ~= 2.871 (a.u.)
C_F = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0)

# Default absolute shell cutoffs (Angstrom): core <= C1 < bond <= C2 < interstitial
SHELL_C1 = 0.8
SHELL_C2 = 1.5
