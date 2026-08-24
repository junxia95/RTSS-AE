#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
OUT="$ROOT/tmp/smoketest/latest"
export MPLCONFIGDIR="$ROOT/tmp/matplotlib"
export RTDFL_TOPO_ROOT="$OUT/topo"

mkdir -p "$OUT"
mkdir -p "$MPLCONFIGDIR"
find "$OUT" -mindepth 1 -delete

cd "$ROOT"
"$PYTHON" scripts/check_environment.py | tee "$OUT/environment.txt"
"$PYTHON" -m unittest discover -s tests -v 2>&1 | tee "$OUT/tests.log"
if [[ -d "$ROOT/expected_results/source" ]]; then
  "$PYTHON" scripts/validate_results.py --expected-only \
    | tee "$OUT/expected-results.log"
else
  printf 'expected_results not present; bundled-evidence validation skipped\n' \
    | tee "$OUT/expected-results.log"
fi
printf 'smoke test passed\n' | tee "$OUT/status.txt"
