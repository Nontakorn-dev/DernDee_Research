#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.matplotlib-cache}"

FOLD="${FOLD:-0}"
SEED="${SEED:-42}"
MANIFEST="${MANIFEST:-experiments/compression/manifest_fold${FOLD}_seed${SEED}.json}"
PHASE_MANIFEST="${PHASE_MANIFEST:-experiments/compression/manifest_fold${FOLD}_seed${SEED}_phase.json}"

"$PYTHON" analysis/build_kfold_illustration_manifest.py \
  --fold "$FOLD" --seed "$SEED" --out "$MANIFEST"
"$PYTHON" analysis/build_kfold_illustration_manifest.py \
  --fold "$FOLD" --seed "$SEED" \
  --configs FP32 INT8 Prune50 INT8+Prune50 \
  --out "$PHASE_MANIFEST"
"$PYTHON" analysis/phase_degradation.py --manifest "$PHASE_MANIFEST" --metric phase_accuracy
"$PYTHON" analysis/plot_pareto.py --manifest "$MANIFEST"
"$PYTHON" analysis/plot_phase_degradation.py --manifest "$PHASE_MANIFEST" --metric phase_accuracy
"$PYTHON" analysis/plot_confusion_matrix.py
"$PYTHON" analysis/plot_training.py
