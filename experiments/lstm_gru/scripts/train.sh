#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT/experiments/lstm_gru"
PYTHON="${PYTHON:-python3}"
echo "=== LSTM+GRU FP32 @ 100 Hz (shared protocol) ==="
exec "$PYTHON" train.py "$@"
