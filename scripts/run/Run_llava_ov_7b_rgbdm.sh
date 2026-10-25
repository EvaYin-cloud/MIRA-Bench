#!/bin/bash
# Run the LLaVA-OneVision-7B RGBDM benchmark.
# Usage: bash scripts/run/Run_llava_ov_7b_rgbdm.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

if ! cd "$PROJECT_DIR"; then
    echo "Failed to enter project directory: $PROJECT_DIR"
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

EVAL_ARGS=(--model "llava-ov-7b" --mode rgbdm)
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
LOG_FILE="$LOG_DIR/llava-ov-7b_rgbdm.log"
mkdir -p "$LOG_DIR"

"$PYTHON_BIN" -m src.main "${EVAL_ARGS[@]}" --print-config >/dev/null || exit $?

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

echo "Starting LLaVA-OneVision-7B RGBDM benchmark..."
nohup "$PYTHON_BIN" -m src.main "${EVAL_ARGS[@]}" > "$LOG_FILE" 2>&1 &
PID=$!

echo "PID: $PID"
echo "Log: $LOG_FILE"
echo "Follow log: tail -f $LOG_FILE"
echo "Stop run: kill $PID"

wait "$PID"
STATUS=$?

if [ "$STATUS" -eq 0 ]; then
    echo "=== LLaVA-OneVision-7B RGBDM run completed successfully ==="
else
    echo "=== LLaVA-OneVision-7B RGBDM run failed; check the log above ==="
fi
exit "$STATUS"
