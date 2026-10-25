#!/bin/bash
# Serve local InternVL3-8B through the vLLM OpenAI-compatible API.
# Usage: bash scripts/serve/Serve_internvl3_vllm.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/InternVL3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-OpenGVLab/InternVL3-8B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
DTYPE="${DTYPE:-bfloat16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
CHAT_TEMPLATE_CONTENT_FORMAT="${CHAT_TEMPLATE_CONTENT_FORMAT:-string}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Missing model directory: $MODEL_PATH"
    exit 1
fi

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
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
    --chat-template-content-format "$CHAT_TEMPLATE_CONTENT_FORMAT" \
    --trust-remote-code
