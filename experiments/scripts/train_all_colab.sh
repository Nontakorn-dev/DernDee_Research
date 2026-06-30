#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-shared/configs/train_fair_comparison.json}"
DATA_ROOT="${DATA_ROOT:-dataset/Xy}"
DEVICE="${DEVICE:-auto}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --lazy)
      EXTRA_ARGS+=(--lazy)
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

MODELS=(tinytcn cnn1d lstm_gru tcn transformer)

echo "=== Train all models (shared fair-comparison protocol) ==="
echo "Config:    $CONFIG"
echo "Data root: $DATA_ROOT"
echo "Device:    $DEVICE"
echo

for model in "${MODELS[@]}"; do
  echo "--- Training ${model} ---"
  "$PYTHON" "experiments/${model}/train.py" \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --device "$DEVICE" \
    "${EXTRA_ARGS[@]}"
  echo
done

echo "All models finished."
