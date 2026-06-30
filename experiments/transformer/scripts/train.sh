#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT/experiments/transformer"
PYTHON="${PYTHON:-python3}"
echo "=== Transformer FP32 @ 100 Hz (shared protocol) ==="
exec "$PYTHON" train.py "$@"
