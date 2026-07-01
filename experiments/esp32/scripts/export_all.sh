#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXPORT_PY="${ROOT}/.venv-export/bin/python"

if [[ ! -x "${EXPORT_PY}" ]]; then
  bash "${ROOT}/experiments/esp32/scripts/setup_export_env.sh"
fi

CONFIGS="${CONFIGS:-FP32 INT8 Prune50 INT8+Prune50}"
ACTIVE_CONFIG="${ACTIVE_CONFIG:-INT8}"
TENSOR_ARENA_KB="${TENSOR_ARENA_KB:-120}"

echo "==> Exporting TFLite models"
"${EXPORT_PY}" "${ROOT}/experiments/esp32/scripts/export_tflite.py" --configs ${CONFIGS}

echo "==> Generating Arduino headers"
"${EXPORT_PY}" "${ROOT}/experiments/esp32/scripts/export_headers.py" \
  --configs ${CONFIGS} \
  --active-config "${ACTIVE_CONFIG}" \
  --tensor-arena-kb "${TENSOR_ARENA_KB}"

echo "==> Host validation"
"${EXPORT_PY}" "${ROOT}/experiments/esp32/scripts/validate_tflite.py" --configs ${CONFIGS}

echo "Done. Flash sketch from experiments/esp32/arduino/tinytcn_benchmark/"
