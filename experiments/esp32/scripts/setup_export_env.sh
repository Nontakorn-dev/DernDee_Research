#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXPORT_VENV="${ROOT}/.venv-export"

if [[ ! -x "${EXPORT_VENV}/bin/python" ]]; then
  echo "Creating export venv at ${EXPORT_VENV} (Python 3.12)..."
  if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv first: https://docs.astral.sh/uv/" >&2
    exit 1
  fi
  uv python install 3.12
  uv venv "${EXPORT_VENV}" --python 3.12
fi

echo "==> Installing export dependencies"
uv pip install --python "${EXPORT_VENV}/bin/python" -r "${ROOT}/experiments/esp32/requirements-export.txt"

echo "Export env ready: ${EXPORT_VENV}/bin/python"
