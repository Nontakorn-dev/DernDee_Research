#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.matplotlib-cache}"
"$PYTHON" experiments/compression/run_eval.py "$@"
"$PYTHON" analysis/collect_pareto.py
"$PYTHON" analysis/phase_degradation.py --metric phase_accuracy
"$PYTHON" analysis/plot_pareto.py
"$PYTHON" analysis/plot_phase_degradation.py --metric phase_accuracy
"$PYTHON" analysis/plot_confusion_matrix.py
