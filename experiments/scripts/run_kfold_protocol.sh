#!/usr/bin/env bash
# Full Track B k-fold protocol (baseline then compression via manifest).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

echo "Syncing manifest..."
.venv/bin/python experiments/scripts/run_kfold_orchestrator.py --sync-only

echo "Starting next pending job (repeat until 135 jobs done)..."
exec caffeinate -dimsu .venv/bin/python experiments/scripts/run_kfold_orchestrator.py \
  --next --max-jobs "${MAX_JOBS:-1}" "$@"
