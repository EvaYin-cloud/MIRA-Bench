#!/bin/bash
# Serve the Hugging Face format LLaVA-OneVision-7B checkpoint with vLLM.
# Usage: bash scripts/serve/Serve_llava_ov_7b_hf_vllm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/llava-onevision-qwen2-7b-ov-hf}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llava-hf/llava-onevision-qwen2-7b-ov-hf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8004}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DTYPE="${DTYPE:-bfloat16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Missing model directory: $MODEL_PATH"
    printf 'Run: hf download %q --local-dir %q\n' \
        'llava-hf/llava-onevision-qwen2-7b-ov-hf' "$MODEL_PATH"
    exit 1
fi

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

exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --limit-mm-per-prompt '{"image":3,"video":0}' \
    --trust-remote-code
