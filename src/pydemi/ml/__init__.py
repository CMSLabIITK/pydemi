"""
pydemi.ml
=========
Charge densities predicted from the crystal structure alone, with ChargE3Net.

ChargE3Net (Koker et al., npj Comput. Mater. 10, 161 (2024), MIT licence) is an
E(3)-equivariant network that predicts the valence charge density at any point from
the atoms around it. pydemi ships its network (``pydemi.ml.model``) and a checkpoint
fine-tuned on the CMS Lab dataset of VASP CHGCARs (PAW_PBE, ENCUT = 500 eV, PREC =
Accurate), so a prediction is a prediction of the CHGCAR those settings would give:

    vd = pydemi.ml.predict("Si.cif")       # VolumetricData, like read_chgcar
    feats = pydemi.featurize(vd)

The density is predicted on the grid VASP would use (:func:`vasp_grid_shape`), and
rescaled to the valence electron count of the training POTCARs (:data:`ZVAL`); the
scale factor and the model are recorded in ``vd.sources``.

Needs the ``ml`` extra (``pip install "pydemi[ml]"``: PyTorch, e3nn, ASE). The grid
rule and the element table work without it.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from pydemi.io.base import Grid, Lattice, Structure, VolumetricData, as_structure
from pydemi.ml.grid import grid_points, vasp_grid_shape

if sys.platform == "darwin":
    # PyTorch's macOS wheels bring their own OpenMP runtime and LightGBM uses Homebrew's libomp;
    # with the property models and ChargE3Net in one process (the GUI), the second runtime
    # would abort the process unless duplicates are allowed. Respects a value set by the user.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

HERE = Path(__file__).parent
CHECKPOINT = HERE / "charge3net_cmslab_v1.pt"
MODEL_NAME = "ChargE3Net (CMS Lab fine-tune v1)"
REQUIRES = ("torch", "e3nn", "ase")
INSTALL_HINT = 'pip install "pydemi[ml]"'
CITATION = ("T. Koker, K. Quigley, E. Taw, K. Tibbetts and L. Li, Higher-order equivariant "
            "neural networks for charge density prediction in materials, npj Comput. Mater. "
            "10, 161 (2024)")
STRUCTURE_SUFFIXES = (".cif", ".vasp", ".poscar", ".xyz", ".extxyz", ".json", ".in")
# The network's memory grows with the number of atom-probe pairs within the cutoff
# (about 160 kB per pair, measured on CPU); chunks of grid points are sized to this budget.
PAIRS_PER_CHUNK = {"cpu": 16_384, "cuda": 65_536}

PathLike = Union[str, Path]


def _potcars() -> tuple[dict[str, float], dict[str, str]]:
    with open(HERE / "potcars.csv", newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return ({r["element"]: float(r["zval"]) for r in rows},
            {r["element"]: r["potcar"] for r in rows})


ZVAL, POTCAR = _potcars()
"""Valence electrons and PAW dataset of each element in the training data; the model
knows these elements only."""


class Cancelled(Exception):
    """Raised by :func:`predict` when ``should_stop`` returns True."""


def missing() -> list[str]:
    """Packages of the ``ml`` extra that are not installed."""
    return [m for m in REQUIRES if importlib.util.find_spec(m) is None]


def unsupported(structure: Any) -> list[str]:
    """Elements of ``structure`` the model was not trained on."""
    return [e for e in as_structure(structure).elements if e not in ZVAL]


def n_valence(structure: Any) -> float:
    """Valence electrons of the cell with the training POTCARs (sum of ZVAL)."""
    return float(sum(ZVAL[s] for s in as_structure(structure).species))


# ----------------------------------------------------------------------
# structure input
# ----------------------------------------------------------------------

def _ase_format(name: str) -> Optional[str]:
    low = Path(name).name.lower()
    if low.endswith(".cif"):
        return "cif"
    if low.endswith((".vasp", ".poscar")) or low.startswith(("poscar", "contcar")):
        return "vasp"
    return None  # let ASE guess from the suffix


def read_structure(src: Union[PathLike, bytes], name: Optional[str] = None) -> Structure:
    """A periodic structure from a CIF, POSCAR/CONTCAR or any file ASE reads.

    ``src`` is a path, or the file's bytes with its ``name`` (the name picks the format).
    For a CIF with several structures the first is used.
    """
    from ase.io import read

    if isinstance(src, (bytes, bytearray)):
        fname = Path(name or "structure").name
        with tempfile.TemporaryDirectory(prefix="pydemi-") as tmp:
            path = Path(tmp) / fname
            path.write_bytes(bytes(src))
            return read_structure(path)
    path = Path(src)
    try:
        atoms = read(str(path), index=0, format=_ase_format(path.name))
    except Exception as exc:  # ASE raises many types for unreadable input
        raise ValueError(f"not a readable crystal structure ({exc})") from exc
    if not all(atoms.pbc) or abs(atoms.cell.volume) < 1e-6:
        raise ValueError("the structure has no 3-D periodic cell (a molecule or a slab "
                         "without vacuum cell?)")
    return Structure(Lattice(np.array(atoms.cell)), atoms.get_chemical_symbols(),
                     atoms.get_scaled_positions(wrap=True))


# ----------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------

def default_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Model:
    """A loaded checkpoint: the network, where it runs, and what it is."""

    net: Any
    device: str
    cutoff: float
    checkpoint: str
    sha256: str
    info: dict[str, Any]


_LOCK = threading.Lock()


@lru_cache(maxsize=4)
def load(checkpoint: Optional[str] = None, device: Optional[str] = None) -> Model:
    """The network with the weights of ``checkpoint`` (default: the shipped fine-tune) on
    ``device`` ("cpu", "cuda", "cuda:1", ...; default: a GPU if there is one).

    Also takes training checkpoints of the original ChargE3Net code (the ``model`` entry
    is used; the hyperparameters must be those of its ``e3_density`` config).
    """
    import torch

    from pydemi.ml.model import E3DensityModel

    path = Path(checkpoint) if checkpoint else CHECKPOINT
    data = path.read_bytes()
    state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    config = state.get("config") or dict(num_interactions=3, num_neighbors=20, mul=500, lmax=4,
                                         cutoff=4.0, basis="gaussian", num_basis=20)
    net = E3DensityModel(**config)  # type: ignore[no-untyped-call]
    net.load_state_dict(state["model"])
    dev = device or default_device()
    net.to(dev).eval()
    return Model(net, dev, float(config["cutoff"]), str(path),
                 hashlib.sha256(data).hexdigest(), dict(state.get("info") or {}))


def _tensor(x: Any, dev: str, dtype: Any = None) -> Any:
    import torch
    return torch.as_tensor(x, dtype=dtype or torch.float32, device=dev)


def pairs_per_point(structure: Any, cutoff: float = 4.0) -> float:
    """Expected atoms within ``cutoff`` of a grid point: n_atoms / V * (4/3) pi rc^3."""
    s = as_structure(structure)
    return s.n_atoms / s.volume * 4.0 / 3.0 * np.pi * cutoff ** 3


def chunk_size(structure: Any, device: str, max_pairs: Optional[int] = None) -> int:
    """Grid points per chunk so a chunk holds about ``max_pairs`` atom-probe pairs."""
    budget = max_pairs or PAIRS_PER_CHUNK["cuda" if device.startswith("cuda") else "cpu"]
    return int(max(64, budget // max(1.0, pairs_per_point(structure))))


def predict_grid(structure: Any, shape: Optional[Tuple[int, int, int]] = None, *,
                 model: Optional[Model] = None, max_pairs: Optional[int] = None,
                 progress: Optional[Callable[[int, int], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None) -> NDArray[np.float32]:
    """The raw predicted density (electrons / Angstrom^3) on ``shape`` (default: the VASP grid).

    Point [i, j, k] is at fractional (i/n1, j/n2, k/n3). The points are processed in
    chunks of about ``max_pairs`` atom-point pairs (memory: ~160 kB per pair; default
    :data:`PAIRS_PER_CHUNK`). ``progress(done, total)`` is called after every chunk;
    ``should_stop()`` returning True raises :class:`Cancelled`.
    """
    import torch

    from pydemi.ml.graph import Replicas, atom_edges, probe_edges

    s = as_structure(structure)
    bad = unsupported(s)
    if bad:
        raise ValueError(f"the model was not trained on {', '.join(bad)}; it knows "
                         f"{len(ZVAL)} elements (pydemi.ml.ZVAL)")
    m = model or load()
    n1, n2, n3 = (int(n) for n in (shape or vasp_grid_shape(s.lattice)))
    grid = (n1, n2, n3)
    total = n1 * n2 * n3
    rep = Replicas.build(s, m.cutoff)
    from pydemi.data import atomic_number
    numbers = _tensor([atomic_number(e) for e in s.species], m.device, torch.long)
    out = np.empty(total, dtype=np.float32)
    chunk = chunk_size(s, m.device, max_pairs)
    with _LOCK, torch.inference_mode():  # one prediction at a time: the network is shared
        src, dst, vec = atom_edges(rep, s, m.cutoff)
        atoms = m.net.atom_model(numbers, _tensor(src, m.device, torch.long),
                                 _tensor(dst, m.device, torch.long), _tensor(vec, m.device))
        start = 0
        while start < total:
            if should_stop is not None and should_stop():
                raise Cancelled()
            stop = min(start + chunk, total)
            src, dst, vec = probe_edges(rep, grid_points(s.lattice, grid, start, stop), m.cutoff)
            try:
                rho = m.net.probe_model(atoms, stop - start, _tensor(src, m.device, torch.long),
                                        _tensor(dst, m.device, torch.long), _tensor(vec, m.device))
            except (MemoryError, RuntimeError) as exc:  # torch.OutOfMemoryError is a RuntimeError
                if not _out_of_memory(exc) or chunk <= 64:
                    raise
                del src, dst, vec
                chunk //= 2  # retry this chunk at half the size
                if m.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                continue
            out[start:stop] = rho.float().cpu().numpy()
            start = stop
            if progress is not None:
                progress(stop, total)
    return out.reshape(grid)


def _out_of_memory(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, MemoryError) or "out of memory" in text or "can't allocate" in text


def predict(structure: Union[Any, PathLike], *, shape: Optional[Tuple[int, int, int]] = None,
            encut: float = 500.0, checkpoint: Optional[PathLike] = None,
            device: Optional[str] = None, renormalize: bool = True,
            max_pairs: Optional[int] = None,
            progress: Optional[Callable[[int, int], None]] = None,
            should_stop: Optional[Callable[[], bool]] = None) -> VolumetricData:
    """Predict the charge density of ``structure`` (a Structure, or a structure file).

    The grid is ``shape``, or the one VASP would use at ``encut`` eV with PREC = Accurate
    (the training settings are 500 eV). With ``renormalize`` (default) the density is
    scaled to integrate to :func:`n_valence` electrons. ``vd.sources`` records the model,
    checkpoint hash, raw electron count and scale factor; see :func:`predict_grid` for
    ``progress`` and ``should_stop``.
    """
    s = read_structure(structure) if isinstance(structure, (str, Path)) else as_structure(structure)
    rule = ("x".join(map(str, shape)) if shape else
            f"vasp_grid_shape(encut={encut:g}, prec=accurate)")
    dims = (int(shape[0]), int(shape[1]), int(shape[2])) if shape else vasp_grid_shape(s.lattice, encut)
    m = load(str(checkpoint) if checkpoint else None, device)
    t0 = time.perf_counter()
    rho = predict_grid(s, dims, model=m, max_pairs=max_pairs, progress=progress,
                       should_stop=should_stop).astype(np.float64)
    grid = Grid(rho, s.lattice)
    n_raw = float(rho.sum() * grid.dV)
    sources = {"origin": "predicted", "rho": f"predicted: {MODEL_NAME}",
               "model": MODEL_NAME, "checkpoint": Path(m.checkpoint).name,
               "checkpoint_sha256": m.sha256, "device": m.device,
               "grid_rule": rule,
               "n_electrons_raw": f"{n_raw:.6f}",
               "predict_time_s": f"{time.perf_counter() - t0:.1f}"}
    if isinstance(structure, (str, Path)):
        sources["file"] = str(structure)
    if renormalize:
        if n_raw <= 0:
            raise ValueError(f"the predicted density integrates to {n_raw:.4g} electrons")
        scale = n_valence(s) / n_raw
        grid = Grid(rho * scale, s.lattice)
        sources["charge_scale"] = f"{scale:.8f}"
    else:
        sources["charge_scale"] = "1"
    return VolumetricData(s, grid, sources=sources)


__all__: Sequence[str] = ["predict", "predict_grid", "read_structure", "vasp_grid_shape",
                          "load", "missing", "unsupported", "n_valence", "ZVAL", "POTCAR",
                          "Cancelled", "CITATION", "MODEL_NAME"]
