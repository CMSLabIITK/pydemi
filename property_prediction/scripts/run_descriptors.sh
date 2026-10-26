#!/usr/bin/env bash
# Step 1: the 56 pydemi descriptors of every CHGCAR in the CMS Lab dataset (read-only input).
# 16 worker processes, one thread each, lowest priority. Output: ../data/descriptors_56.csv
set -euo pipefail
cd "$(dirname "$0")/../data"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
REPO="$(cd ../.. && pwd)"
export PYTHONPATH="$REPO/src"
PY="${PYTHON:-python}"
{
  echo "started: $(date -Is)"
  echo "host: $(hostname)  workers: 16  partition: nearest (default settings)"
  echo "input: /data/sai/new_charge/6000_data_aug13/*/CHGCAR"
  echo "pydemi: $($PY -c 'import pydemi; print(pydemi.__version__)')  git: $(git -C "$REPO" rev-parse --short HEAD) (working tree has uncommitted changes: $(git -C "$REPO" status --porcelain | wc -l) files)"
  echo "numpy/scipy: $($PY -c 'import numpy, scipy; print(numpy.__version__, scipy.__version__)')"
} > provenance.txt
exec nice -n 19 $PY -m pydemi.cli batch /data/sai/new_charge/6000_data_aug13 --glob "*/CHGCAR" \
     --out descriptors_56.csv --workers 16 --on-error record
