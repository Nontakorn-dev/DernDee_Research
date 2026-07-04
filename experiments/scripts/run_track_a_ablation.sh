#!/usr/bin/env bash
# Track A: prune fine-tune ablation (single split, exploratory appendix).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

CHECKPOINT="${CHECKPOINT:-experiments/tinytcn/runs/fp32_100hz/best_model.pt}"
OUT_ROOT="${OUT_ROOT:-experiments/compression/runs/track_a_ablation}"
PYTHON="${PYTHON:-.venv/bin/python}"
EPOCHS=(5 15 50)
LRS=(1e-4 5e-4)
CONFIGS=(Prune50 "INT8+Prune50")

for epochs in "${EPOCHS[@]}"; do
  for lr in "${LRS[@]}"; do
    for config in "${CONFIGS[@]}"; do
      tag="epochs${epochs}_lr${lr}_${config//+/_}"
      out_dir="${OUT_ROOT}/${tag}"
      echo "=== ${tag} ==="
      "$PYTHON" experiments/compression/run_eval.py \
        --checkpoint "$CHECKPOINT" \
        --out-dir "$out_dir" \
        --configs "$config" \
        --prune-finetune-epochs "$epochs" \
        --finetune-lr "$lr" \
        --lazy
    done
  done
done

echo "Track A ablation complete → ${OUT_ROOT}"
