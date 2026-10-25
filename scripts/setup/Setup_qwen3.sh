#!/bin/bash
# Download Qwen3.5-9B for local vLLM serving.
# Usage: bash scripts/setup/Setup_qwen3.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/Qwen3.5-9B}"

if [ -n "${PYTHON_BIN:-}" ]; then
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "Configured PYTHON_BIN was not found on PATH: $PYTHON_BIN"
        exit 1
    fi
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "Neither python3 nor python was found on PATH."
    exit 1
fi

mkdir -p "$PROJECT_DIR/models"
"$PYTHON_BIN" -m pip install "huggingface_hub>=0.23.0"

"$PYTHON_BIN" - "$MODEL_ID" "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1],
    local_dir=sys.argv[2],
)
print(f"Downloaded {sys.argv[1]} to {sys.argv[2]}")
PY
