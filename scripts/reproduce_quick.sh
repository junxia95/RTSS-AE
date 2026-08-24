#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
export MPLCONFIGDIR="$ROOT/tmp/matplotlib"

cd "$ROOT"
mkdir -p "$MPLCONFIGDIR"
"$PYTHON" scripts/run_matrix.py \
  --config configs/quick_cifar10.json \
  --mode quick \
  --gpu "$GPU" \
  --python "$PYTHON" \
  "$@"

LATEST="$(find results/quick -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | sed -n '1s/^[^ ]* //p')"
"$PYTHON" scripts/validate_results.py --run-dir "$LATEST" --quick-profile
"$PYTHON" scripts/plot_quick_results.py --run-dir "$LATEST"
