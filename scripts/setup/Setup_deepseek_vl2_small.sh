#!/bin/bash
# Install DeepSeek-VL2 code dependencies and download DeepSeek-VL2-Small.
# Run this on the target machine before the benchmark.
# Usage: bash scripts/setup/Setup_deepseek_vl2_small.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DEEPSEEK_REPO_DIR="${DEEPSEEK_REPO_DIR:-$PROJECT_DIR/third_party/DeepSeek-VL2}"
MODEL_ID="${MODEL_ID:-deepseek-ai/deepseek-vl2-small}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/deepseek-vl2-small}"

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

mkdir -p "$PROJECT_DIR/third_party" "$PROJECT_DIR/models"

if [ ! -d "$DEEPSEEK_REPO_DIR/.git" ]; then
    git clone https://github.com/deepseek-ai/DeepSeek-VL2.git "$DEEPSEEK_REPO_DIR"
else
    git -C "$DEEPSEEK_REPO_DIR" pull --ff-only
fi

"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install -e "$DEEPSEEK_REPO_DIR"
"$PYTHON_BIN" -m pip install "numpy<2"
"$PYTHON_BIN" -m pip install "huggingface_hub[cli]>=0.23.0"

"$PYTHON_BIN" - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="$MODEL_ID",
    local_dir="$MODEL_DIR",
    local_dir_use_symlinks=False,
    resume_download=True,
)
print("Downloaded $MODEL_ID to $MODEL_DIR")
PY

echo ""
echo "DeepSeek-VL2-Small setup is complete."
echo "Model directory: $MODEL_DIR"
