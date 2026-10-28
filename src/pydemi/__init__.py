"""
pydemi
======
Interpretable, named, fixed-length descriptors from a VASP CHGCAR.

    import pydemi
    vd = pydemi.read_chgcar("CHGCAR")      # VolumetricData: structure + density
    feats = pydemi.featurize(vd)           # dict[str, float]
    pydemi.catalogue()                     # DataFrame of descriptor metadata

Author: Shubham Maurya, CMS Lab, IIT Kanpur.
"""

__version__ = "0.2.1"
__author__ = "Shubham Maurya, CMS Lab, IIT Kanpur"

from .batch import featurize_batch  # noqa: E402
from .validate.convergence import sensitivity_sweep  # noqa: E402
from .descriptors import catalogue, descriptor_names, featurize  # noqa: E402
from .io import Grid, Lattice, Structure, VolumetricData, read_chgcar, write_volumetric  # noqa: E402

__all__ = ["Grid", "Lattice", "Structure", "VolumetricData", "read_chgcar", "write_volumetric",
           "featurize", "featurize_batch", "catalogue", "descriptor_names", "sensitivity_sweep",
           "__version__"]
