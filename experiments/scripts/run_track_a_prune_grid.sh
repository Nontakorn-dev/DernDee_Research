#!/usr/bin/env bash
# Track A: prune ratio grid 25/50/75 (single split, exploratory Pareto).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

CHECKPOINT="${CHECKPOINT:-experiments/tinytcn/runs/fp32_100hz/best_model.pt}"
OUT_ROOT="${OUT_ROOT:-experiments/compression/runs/track_a_prune_grid}"
PYTHON="${PYTHON:-.venv/bin/python}"
KEEP_RATIOS=(0.25 0.50 0.75)

for ratio in "${KEEP_RATIOS[@]}"; do
  pct=$("$PYTHON" - <<PY
print(int(round(${ratio} * 100)))
PY
)
  config="Prune${pct}"
  int8_config="INT8+Prune${pct}"
  out_dir="${OUT_ROOT}/keep${pct}"
  echo "=== keep_ratio=${ratio} (${config}, ${int8_config}) ==="
  "$PYTHON" experiments/compression/run_eval.py \
    --checkpoint "$CHECKPOINT" \
    --out-dir "$out_dir" \
    --configs "$config" "$int8_config" \
    --keep-ratio "$ratio" \
    --lazy
done

echo "Track A prune grid complete → ${OUT_ROOT}"
