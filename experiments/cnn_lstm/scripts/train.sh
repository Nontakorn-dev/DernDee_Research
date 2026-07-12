#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT/experiments/cnn_lstm"
PYTHON="${PYTHON:-python3}"
echo "=== CNN-LSTM FP32 @ 100 Hz (shared protocol) ==="
exec "$PYTHON" train.py "$@"
