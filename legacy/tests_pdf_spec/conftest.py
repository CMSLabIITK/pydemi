import os
import tempfile

import numpy as np
import pytest

# keep the free-atom disk cache out of the user's ~/.cache during tests
os.environ.setdefault("PYDEMI_CACHE_DIR", tempfile.mkdtemp(prefix="pydemi-test-cache-"))

from pydemi import Structure

TRICLINIC = np.array([[5.0, 0.0, 0.0],
                      [1.5, 4.6, 0.0],
                      [0.8, 1.1, 5.3]])


@pytest.fixture
def cubic_one_atom():
    return Structure(np.eye(3) * 8.0, ["Na"], [[0.0, 0.0, 0.0]])


@pytest.fixture
def triclinic_two_atoms():
    return Structure(TRICLINIC, ["Si", "O"], [[0.1, 0.2, 0.3], [0.6, 0.55, 0.7]])


@pytest.fixture
def skewed_cell():
    # strongly sheared cell where fractional rounding is not the minimum image
    lat = np.array([[4.0, 0.0, 0.0],
                    [3.7, 1.2, 0.0],
                    [3.5, 0.9, 1.4]])
    return Structure(lat, ["Fe", "Fe", "Ni"],
                     [[0.0, 0.0, 0.0], [0.5, 0.3, 0.2], [0.25, 0.8, 0.6]])
