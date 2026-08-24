#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
export MPLCONFIGDIR="$ROOT/tmp/matplotlib"

cd "$ROOT"
mkdir -p "$MPLCONFIGDIR"
"$PYTHON" scripts/run_matrix.py \
  --config configs/paper_full.json \
  --mode full \
  --gpu "$GPU" \
  --python "$PYTHON" \
  "$@"
