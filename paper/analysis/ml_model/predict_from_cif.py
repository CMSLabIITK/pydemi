"""
predict_from_cif.py

Predict the valence charge density of crystal structures with a ChargE3Net
checkpoint, starting from the structure alone (CIF, POSCAR or any format ASE
reads), and write one pydemi prediction file (.npz) per structure.

No DFT calculation is needed: the grid is the one VASP would use for the
density (pydemi.vasp_grid_shape: PREC = Accurate, ENCUT = 500 eV by default,
the settings of the training data), and the density is predicted at every
grid point with the unchanged test pipeline of this repository
(src/test_from_config.py, full-grid mode).

Steps, all under --work:
  1. inputs/  <name>.npy (placeholder density of the grid shape; only its
     shape is used), <name>_atoms.pkl, filelist.txt, split.json (all test),
     probe_counts.csv -- the layout of data/your_dataset_inputs/;
  2. the test pipeline writes pred/cubes/<name>.npy (electrons / Angstrom^3);
  3. out/<name>.npz via pydemi.write_predicted, with the model, checkpoint
     and grid rule as metadata.
The NMAPE the pipeline logs is meaningless here (the target is the placeholder).

Then, in any environment with pydemi:
    vd = pydemi.read_predicted("out/<name>.npz", zval=..., paw_radii=...)
    features = pydemi.featurize(vd)

Usage (charge3net environment; pydemi on PYTHONPATH, see scripts/predict_from_cif.sh):
    python scripts/predict_from_cif.py STRUCTURE [STRUCTURE ...] --out DIR
        [--checkpoint CKPT] [--encut 500] [--prec accurate] [--work DIR] [--keep-work]
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read as ase_read

import pydemi

REPO = Path(__file__).resolve().parents[1]
FINETUNED = REPO / "results/charge3net/finetune_mp_v1/2026-09-24/22-48-31/checkpoint.pt"
PY = sys.executable


def name_of(path: Path) -> str:
    stem = path.name
    for suffix in (".cif", ".vasp", ".poscar"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def prepare(structures, inputs: Path, encut: float, prec: str) -> dict:
    inputs.mkdir(parents=True, exist_ok=True)
    rows, names, atoms_of = [], [], {}
    for path in structures:
        atoms = ase_read(str(path))
        name = name_of(Path(path))
        if name in atoms_of:
            raise ValueError(f"two structures named {name}")
        shape = pydemi.vasp_grid_shape(np.array(atoms.get_cell()), encut=encut, prec=prec)
        np.save(inputs / f"{name}.npy", np.ones(shape, dtype=np.float32))
        with open(inputs / f"{name}_atoms.pkl", "wb") as f:
            pickle.dump(atoms, f)
        rows.append({"id": name, "Count": int(np.prod(shape)), "shape_x": shape[0],
                     "shape_y": shape[1], "shape_z": shape[2]})
        names.append(name)
        atoms_of[name] = atoms
        print(f"{name}: {len(atoms)} atoms, grid {shape[0]}x{shape[1]}x{shape[2]}")
    (inputs / "filelist.txt").write_text("\n".join(names) + "\n")
    (inputs / "split.json").write_text(json.dumps({"train": [], "validation": [],
                                                   "test": list(range(len(names)))}))
    pd.DataFrame(rows).to_csv(inputs / "probe_counts.csv", index=False)
    return atoms_of


def run_model(inputs: Path, pred: Path, run_dir: Path, checkpoint: Path) -> None:
    cmd = [PY, "src/test_from_config.py", "-cd", "configs/charge3net", "-cn", "train_mp_e3_final.yaml",
           "nnodes=1", "nprocs=1",
           f"data.data_root={inputs / 'filelist.txt'}",
           f"data.split_file={inputs / 'split.json'}",
           f"data.grid_size_file={inputs / 'probe_counts.csv'}",
           f"checkpoint_path={checkpoint}",
           "data.test_probes=null", f"cube_dir={pred}",
           f"hydra.run.dir={run_dir}", "hydra.job.name=predict_from_cif"]
    subprocess.run(cmd, cwd=REPO, check=True)


def convert(atoms_of: dict, pred: Path, out: Path, checkpoint: Path, encut: float, prec: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, atoms in atoms_of.items():
        cube = pred / "cubes" / f"{name}.npy"
        rho = np.load(cube)
        rho = rho[..., 0] if rho.ndim == 4 else rho
        s = pydemi.Structure(pydemi.Lattice(np.array(atoms.get_cell())), atoms.get_chemical_symbols(),
                             atoms.get_scaled_positions(wrap=True))
        p = pydemi.write_predicted(out / f"{name}.npz", s, rho.astype(np.float64),
                                   model="ChargE3Net", checkpoint=str(checkpoint),
                                   grid_rule=f"vasp_grid_shape(encut={encut:g}, prec={prec})")
        print(f"wrote {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structures", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=FINETUNED)
    ap.add_argument("--encut", type=float, default=500.0)
    ap.add_argument("--prec", default="accurate")
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()
    if not a.checkpoint.exists():
        sys.exit(f"checkpoint not found: {a.checkpoint}")
    work = a.work or Path(tempfile.mkdtemp(prefix="predict_from_cif_"))
    work = work.resolve()
    atoms_of = prepare(a.structures, work / "inputs", a.encut, a.prec)
    run_model(work / "inputs", work / "pred", work / "run", a.checkpoint.resolve())
    convert(atoms_of, work / "pred", a.out, a.checkpoint.resolve(), a.encut, a.prec)
    if not a.keep_work and a.work is None:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
