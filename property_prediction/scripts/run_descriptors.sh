#!/usr/bin/env bash
# The 55 pydemi descriptors of every CHGCAR of the dataset (one folder per structure), with
# 16 worker processes of one thread each.
#   scripts/run_descriptors.sh <dataset directory>
# Output: ../data/descriptors_batch.csv and ../data/provenance.txt
set -euo pipefail
SRC="$(cd "${1:?usage: run_descriptors.sh <dataset directory>}" && pwd)"
cd "$(dirname "$0")/../data"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY="${PYTHON:-python}"
{
  echo "started: $(date -Is)"
  echo "workers: 16  partition: nearest (default settings)"
  echo "input: $SRC/*/CHGCAR"
  echo "pydemi: $($PY -c 'import pydemi; print(pydemi.__version__)')"
  echo "numpy/scipy: $($PY -c 'import numpy, scipy; print(numpy.__version__, scipy.__version__)')"
} > provenance.txt
exec nice -n 19 $PY -m pydemi.cli batch "$SRC" --glob "*/CHGCAR" \
     --out descriptors_batch.csv --workers 16 --on-error record
