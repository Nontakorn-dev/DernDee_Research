#!/usr/bin/env bash
# K-fold orchestrator for Google Colab (CUDA + lazy loading by default).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-}"
DEVICE="${DEVICE:-cuda}"
LAZY=1
MAX_JOBS=1
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lazy) LAZY=1; shift ;;
    --no-lazy) LAZY=0; shift ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --max-jobs) MAX_JOBS="$2"; shift 2 ;;
    --sync-only|--next|--reset-running)
      EXTRA+=("$1")
      shift
      ;;
    --model|--phase)
      EXTRA+=("$1" "$2")
      shift 2
      ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ -n "$DATA_ROOT" ]]; then
  mkdir -p dataset
  if [[ ! -e dataset/Xy ]]; then
    ln -sf "$DATA_ROOT" dataset/Xy
    echo "[run_kfold_colab] linked dataset/Xy -> $DATA_ROOT"
  fi
fi

CMD=("$PYTHON" experiments/scripts/run_kfold_orchestrator.py --max-jobs "$MAX_JOBS")
if [[ "$LAZY" -eq 1 ]]; then
  CMD+=(--lazy)
fi
if [[ -n "$DATA_ROOT" ]]; then
  CMD+=(--data-root "$DATA_ROOT")
fi
if [[ -n "$DEVICE" ]]; then
  CMD+=(--device "$DEVICE")
fi
CMD+=("${EXTRA[@]}")

echo "[run_kfold_colab] ${CMD[*]}"
exec "${CMD[@]}"
