#!/usr/bin/env bash
# Overnight k-fold runner with manifest skip/resume.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

LAZY=0
MAX_JOBS=1
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lazy) LAZY=1; shift ;;
    --max-jobs) MAX_JOBS="$2"; shift 2 ;;
    --sync-only) EXTRA+=(--sync-only); shift ;;
    --next) EXTRA+=(--next); shift ;;
    --reset-running) EXTRA+=(--reset-running); shift ;;
    --model) EXTRA+=(--model "$2"); shift 2 ;;
    --phase) EXTRA+=(--phase "$2"); shift 2 ;;
    --all-folds) shift ;;
    --all-seeds) shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

CMD=(.venv/bin/python experiments/scripts/run_kfold_orchestrator.py --max-jobs "$MAX_JOBS")
if [[ "$LAZY" -eq 1 ]]; then
  CMD+=(--lazy)
fi
CMD+=("${EXTRA[@]}")

echo "[run_overnight] ${CMD[*]}"
exec "${CMD[@]}"
