"""
pydemi
======
Interpretable, named, fixed-length descriptors from DFT charge-density grids.

    import pydemi
    vd = pydemi.read_vasp("CHGCAR", elf="ELFCAR")      # VolumetricData
    feats = pydemi.featurize(vd)                       # dict[str, float]
    pydemi.catalogue()                                 # DataFrame of descriptor metadata

Author: Shubham Maurya, CMS Lab, IIT Kanpur.
"""

__version__ = "0.1.0.dev0"
__author__ = "Shubham Maurya, CMS Lab, IIT Kanpur"

from .batch import featurize_batch  # noqa: E402
from .core.partition import hirshfeld_charges  # noqa: E402
from .validate.convergence import sensitivity_sweep  # noqa: E402
from .descriptors import catalogue, descriptor_names, featurize  # noqa: E402
from .io import (Grid, Lattice, Structure, VolumetricData, read, read_all_electron,  # noqa: E402
                 read_vasp)

__all__ = ["Grid", "Lattice", "Structure", "VolumetricData", "read", "read_vasp",
           "read_all_electron", "featurize", "featurize_batch", "catalogue", "descriptor_names", "hirshfeld_charges",
           "sensitivity_sweep", "__version__"]
