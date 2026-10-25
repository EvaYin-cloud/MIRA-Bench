#!/bin/bash
# Install MiniCPM-V 2.6 dependencies and download the model.
# Usage: bash scripts/setup/Setup_minicpm_v_2_6.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODEL_ID="${MODEL_ID:-openbmb/MiniCPM-V-2_6}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/MiniCPM-V-2_6}"

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

"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install \
    "Pillow>=10.1.0" \
    "torch>=2.1.2" \
    "torchvision>=0.16.2" \
    "transformers>=4.40.0" \
    "sentencepiece>=0.1.99" \
    "decord" \
    "accelerate" \
    "huggingface_hub[cli]>=0.23.0"

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
echo "MiniCPM-V 2.6 setup is complete."
echo "Model directory: $MODEL_DIR"
