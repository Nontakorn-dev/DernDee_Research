#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT/experiments/cnn1d"
PYTHON="${PYTHON:-python3}"
echo "=== CNN1D FP32 @ 100 Hz (shared protocol) ==="
exec "$PYTHON" train.py "$@"
