#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT/experiments/tinytcn"
PYTHON="${PYTHON:-python3}"
echo "=== TinyTCN FP32 @ 100 Hz ==="
exec "$PYTHON" train.py "$@"
