"""
pydemi -- one grid engine for charge-density descriptors.

Every descriptor is built from one geometry pass (nearest nucleus
distance, direction and index per voxel) and a small set of cached grid
fields and their derivatives, so each extra descriptor costs almost nothing.
"""

from .engine import ELF, MAG, MAG_ABS, POT, RHO, RHO_AE, Engine
from .field import Field
from .geometry import NearestAtom, assign_atoms, nearest_atom
from .partition import (BeckePartition, HardPartition, HirshfeldPartition, PairChunk,
                        Partition)
from .grid import Grid, hessian_eigenvalues, unpack_hessian
from .io import read_aeccar, read_chgcar, read_elfcar, read_locpot, read_volumetric
from .shells import Shells, ShellMasks
from .structure import Structure

__version__ = "0.0.1"
__author__ = "Shubham Maurya, CMS Lab, IIT Kanpur"

__all__ = [
    "Engine", "Field", "Grid", "Structure", "Shells", "ShellMasks",
    "NearestAtom", "nearest_atom", "assign_atoms",
    "Partition", "HardPartition", "BeckePartition", "HirshfeldPartition", "PairChunk",
    "hessian_eigenvalues", "unpack_hessian",
    "read_chgcar", "read_aeccar", "read_elfcar", "read_locpot", "read_volumetric",
    "RHO", "RHO_AE", "MAG", "MAG_ABS", "ELF", "POT",
]
