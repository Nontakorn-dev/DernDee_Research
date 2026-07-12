#!/usr/bin/env bash
# TCN compression ablations: prune grid + fine-tune schedule.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

PYTHON="${PYTHON:-.venv/bin/python}"
DEVICE="${DEVICE:-auto}"
OUT_ROOT="${OUT_ROOT:-experiments/compression/runs/tcn_family_ablation}"
KEEP_RATIOS=(0.25 0.50 0.75)
FINETUNE_EPOCHS=(5 15 50)
MODELS=(tcn)

run_eval() {
  local model="$1"
  local study="$2"
  local out_dir="$3"
  shift 3
  echo "=== ${model} / ${study} → ${out_dir} ==="
  "$PYTHON" experiments/compression/run_eval.py \
    --model "$model" \
    --out-dir "$out_dir" \
    --device "$DEVICE" \
    --lazy \
    "$@"
}

for model in "${MODELS[@]}"; do
  checkpoint="experiments/${model}/runs/fp32_100hz/best_model.pt"
  grid_dir="${OUT_ROOT}/${model}/pareto_grid/main"
  grid_configs=(FP32 INT8 INT4)

  for ratio in "${KEEP_RATIOS[@]}"; do
    pct=$(python3 - <<PY
print(int(round(${ratio} * 100)))
PY
)
    grid_configs+=("Prune${pct}" "INT8+Prune${pct}" "INT4+Prune${pct}")
  done

  run_eval "$model" pareto_grid "$grid_dir" \
    --checkpoint "$checkpoint" \
    --configs "${grid_configs[@]}"

  for epochs in "${FINETUNE_EPOCHS[@]}"; do
    ft_dir="${OUT_ROOT}/${model}/finetune/epochs${epochs}"
    run_eval "$model" "finetune/${epochs}ep" "$ft_dir" \
      --checkpoint "$checkpoint" \
      --configs Prune50 "INT8+Prune50" \
      --keep-ratio 0.5 \
      --prune-finetune-epochs "$epochs"
  done
done

echo "Aggregating ablation summaries..."
"$PYTHON" analysis/aggregate_ablation.py --root "$OUT_ROOT"
"$PYTHON" analysis/plot_ablation_pareto.py --combined
"$PYTHON" analysis/export_ablation_tex.py

echo "TCN-family ablation complete → ${OUT_ROOT}"
