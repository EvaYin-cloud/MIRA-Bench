#!/bin/bash
# Uses the native DeepSeek-VL2 model and chat template in vLLM 0.19.0.

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/deepseek-vl2-tiny}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-ai/deepseek-vl2-tiny}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8005}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-bfloat16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "Missing model config: $MODEL_PATH/config.json" >&2
    printf 'Run: hf download deepseek-ai/deepseek-vl2-tiny --local-dir "%s"\n' "$MODEL_PATH" >&2
    exit 1
fi

if [ -n "${PYTHON_BIN:-}" ]; then
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "Configured PYTHON_BIN was not found on PATH: $PYTHON_BIN" >&2
        exit 1
    fi
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "Neither python3 nor python was found on PATH." >&2
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
    --hf-overrides '{"architectures":["DeepseekVLV2ForCausalLM"]}' \
    --chat-template-content-format string \
    --generation-config vllm
