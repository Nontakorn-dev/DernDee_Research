#!/usr/bin/env bash
# K-fold compression for Standard TCN (no FP32 baseline retrain).
# Uses existing fold×seed checkpoints; evaluates the expanded compression grid.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
DEVICE="${DEVICE:-auto}"
MAX_JOBS="${MAX_JOBS:-9999}"
MODELS=(tcn)
SKIP_CLEAN=0
SKIP_AGGREGATE=0
RESET_RUNNING=0
EXTRA=()

usage() {
  cat <<'EOF'
Usage: bash experiments/scripts/run_kfold_compression.sh [options]

Compression-only k-fold eval (FP32 baselines unchanged):
  TCN × 15 checkpoints × 8 configs = 120 jobs.

Primary Pareto configs: FP32, INT8, Prune25, Prune50, Prune75, INT8+Prune50
Fine-tune ablation: Prune50 (5 ep default), Prune50_ft15, Prune50_ft50

Options:
  --model NAME       Run one model only: tcn (default).
  --skip-clean       Keep existing compression run outputs.
  --skip-aggregate   Skip analysis/aggregate_runs.py at the end.
  --max-jobs N       Stop after N orchestrator jobs.
  --device DEV       cuda | cpu | mps | auto (default: auto)
  --lazy             Lazy dataset loading (recommended on Mac).
  --reset-running    Reset stuck "running" manifest entries before starting.
  -h, --help         Show this help.

Examples:
  # Reset + run full grid:
  .venv/bin/python -c "
from experiments.kfold_manifest import reset_compression_jobs, sync_manifest
reset_compression_jobs()
sync_manifest(reset_running=True)
print('Reset: 120 compression jobs pending')
"

  mkdir -p logs && caffeinate -dimsu bash experiments/scripts/run_kfold_compression.sh --lazy --reset-running \
    > logs/kfold_compression.log 2>&1

  bash experiments/scripts/run_kfold_compression.sh --model tcn --lazy --reset-running
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODELS=("$2")
      shift 2
      ;;
    --skip-clean) SKIP_CLEAN=1; shift ;;
    --skip-aggregate) SKIP_AGGREGATE=1; shift ;;
    --max-jobs) MAX_JOBS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --lazy) EXTRA+=(--lazy); shift ;;
    --reset-running) RESET_RUNNING=1; EXTRA+=(--reset-running); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$SKIP_CLEAN" -eq 0 ]]; then
  echo "=== Cleaning legacy / flat compression runs ==="
  rm -rf \
    experiments/compression/runs/FP32 \
    experiments/compression/runs/INT8 \
    experiments/compression/runs/Prune25 \
    experiments/compression/runs/Prune50 \
    experiments/compression/runs/Prune75 \
    experiments/compression/runs/INT8+Prune50 \
    experiments/compression/runs/tcn \
    experiments/compression/runs/tcn_family_ablation
  rm -f logs/tcn_family_ablation.log
fi

MODELS_CSV=$(IFS=,; echo "${MODELS[*]}")
"$PYTHON" - <<PY
from experiments.kfold_manifest import (
    COMPRESSION_CONFIGS,
    DEFAULT_FOLDS,
    DEFAULT_SEEDS,
    baseline_done,
    baseline_run_dir,
    reset_compression_jobs,
    sync_manifest,
)

models = tuple("${MODELS_CSV}".split(","))
reset_compression_jobs(models=models)
sync_manifest(reset_running=bool(${RESET_RUNNING}))

for model in models:
    missing = []
    for fold in DEFAULT_FOLDS:
        for seed in DEFAULT_SEEDS:
            run_dir = baseline_run_dir(model, fold, seed)
            if not baseline_done(run_dir):
                missing.append(run_dir.name)
    ready = 15 - len(missing)
    print(f"{model}: checkpoints ready {ready}/15")
    if missing:
        print(f"  missing best_model.pt ({len(missing)}):")
        for name in missing[:5]:
            print(f"    - {name}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")
    jobs = 15 * len(COMPRESSION_CONFIGS)
    print(f"  compression jobs queued: {jobs}")
PY

ORCH=(
  "$PYTHON" experiments/scripts/run_kfold_orchestrator.py
  --max-jobs "$MAX_JOBS"
  --device "$DEVICE"
  --phase compression
  "${EXTRA[@]}"
)

for model in "${MODELS[@]}"; do
  echo "=== Compression: ${model} (up to 120 jobs) ==="
  "${ORCH[@]}" --model "$model"
done

if [[ "$SKIP_AGGREGATE" -eq 0 ]]; then
  echo "=== Aggregating k-fold compression (mean ± std, N=15) ==="
  "$PYTHON" analysis/aggregate_runs.py
  "$PYTHON" analysis/export_kfold_compression_tex.py
fi

echo "Done. Outputs: experiments/compression/runs/tcn/fold{0..4}_seed{42,123,456}/{config}/"
