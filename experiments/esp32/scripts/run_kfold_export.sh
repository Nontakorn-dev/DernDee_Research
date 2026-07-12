#!/usr/bin/env bash
# Export all Table-VI TCN configs from k-fold compression checkpoints, then prepare
# Arduino headers for flashing one ACTIVE_CONFIG at a time.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

EXPORT_PY="${ROOT}/.venv-export/bin/python"
MODEL="${MODEL:-tcn}"
FOLD="${FOLD:-0}"
SEED="${SEED:-42}"
CONFIGS="${CONFIGS:-FP32 INT8 Prune50 INT8+Prune50}"
ACTIVE_CONFIG="${ACTIVE_CONFIG:-INT8}"
TENSOR_ARENA_KB="${TENSOR_ARENA_KB:-120}"

COMMON_ARGS=(
  --model "$MODEL"
  --fold "$FOLD"
  --seed "$SEED"
)

if [[ ! -x "${EXPORT_PY}" ]]; then
  bash "${ROOT}/experiments/esp32/scripts/setup_export_env.sh"
fi

echo "==> K-fold source: ${MODEL} fold${FOLD}_seed${SEED}"
echo "==> Exporting TFLite: ${CONFIGS}"
"${EXPORT_PY}" "${ROOT}/experiments/esp32/scripts/export_tflite.py" \
  --configs ${CONFIGS} \
  "${COMMON_ARGS[@]}"

echo "==> Generating Arduino headers (active=${ACTIVE_CONFIG}, arena=${TENSOR_ARENA_KB} KB)"
"${EXPORT_PY}" "${ROOT}/experiments/esp32/scripts/export_headers.py" \
  --configs ${CONFIGS} \
  --active-config "${ACTIVE_CONFIG}" \
  --tensor-arena-kb "${TENSOR_ARENA_KB}" \
  "${COMMON_ARGS[@]}"

echo "==> Host validation"
"${EXPORT_PY}" "${ROOT}/experiments/esp32/scripts/validate_tflite.py" \
  --configs ${CONFIGS} \
  "${COMMON_ARGS[@]}"

cat <<EOF

Done.
  TFLite:  experiments/esp32/exports/{FP32,INT8,Prune50,INT8+Prune50}/model.tflite
  Sketch:  experiments/esp32/arduino/tcn_benchmark/tcn_benchmark.ino
  Active:  ${ACTIVE_CONFIG} (arena ${TENSOR_ARENA_KB} KB)

Flash loop (repeat for each config):
  ACTIVE_CONFIG=INT8 TENSOR_ARENA_KB=120 bash experiments/esp32/scripts/run_kfold_export.sh
  # Arduino IDE → Upload → Serial Monitor 115200 → copy # summary block

Suggested order:
  1) INT8
  2) INT8+Prune50
  3) Prune50   (try TENSOR_ARENA_KB=120, bump to 160 if AllocateTensors fails)
  4) FP32      (often needs TENSOR_ARENA_KB=160 or 200)
EOF
