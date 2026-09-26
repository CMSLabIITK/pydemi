#!/usr/bin/env bash
# Predict charge densities from crystal structures (CIF, POSCAR, ...) with the
# fine-tuned ChargE3Net model and write pydemi prediction files (.npz).
#
#   scripts/predict_from_cif.sh OUT_DIR STRUCTURE [STRUCTURE ...] [-- extra options]
#
# Runs in the charge3net environment with pydemi (pure NumPy/SciPy) on the
# path; see scripts/predict_from_cif.py for the options.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="$1"; shift
export PYTHONPATH="/home/shubham/pydemi/src${PYTHONPATH:+:$PYTHONPATH}"
exec /home/shubham/miniconda3/envs/charge3net/bin/python scripts/predict_from_cif.py "$@" --out "$OUT"
