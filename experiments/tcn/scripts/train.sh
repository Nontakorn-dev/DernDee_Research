#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT/experiments/tcn"
PYTHON="${PYTHON:-python3}"
echo "=== Standard TCN FP32 @ 100 Hz (shared protocol) ==="
exec "$PYTHON" train.py "$@"
