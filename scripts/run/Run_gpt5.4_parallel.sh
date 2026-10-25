#!/bin/bash
# Run GPT-5.4 input conditions in parallel.
# Usage: bash scripts/run/Run_gpt5.4_parallel.sh

set -u

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Please set OPENAI_API_KEY first:"
    echo "  export OPENAI_API_KEY='<your-api-key>'"
    exit 1
fi

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

LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
mkdir -p "$LOG_DIR"

for mode in rgb rgbd rgbm rgbdm; do
    "$PYTHON_BIN" -m src.main --model gpt-5.4 --mode "$mode" --print-config >/dev/null || exit $?
done

start_run() {
    local label="$1"
    local mode="$2"
    local log_file="$3"

    echo "Starting $label..."
    nohup "$PYTHON_BIN" -m src.main --model gpt-5.4 --mode "$mode" \
        > "$log_file" 2>&1 &
}

start_run "RGB only" "rgb" "$LOG_DIR/gpt5.4_rgb.log"
PID_RGB=$!

start_run "RGBD" "rgbd" "$LOG_DIR/gpt5.4_rgbd.log"
PID_RGBD=$!

start_run "RGBM" "rgbm" "$LOG_DIR/gpt5.4_rgbm.log"
PID_RGBM=$!

start_run "RGBDM" "rgbdm" "$LOG_DIR/gpt5.4_rgbdm.log"
PID_RGBDM=$!

echo ""
echo "Four experiments have started:"
echo "  RGB:   PID=$PID_RGB    log: $LOG_DIR/gpt5.4_rgb.log"
echo "  RGBD:  PID=$PID_RGBD   log: $LOG_DIR/gpt5.4_rgbd.log"
echo "  RGBM:  PID=$PID_RGBM   log: $LOG_DIR/gpt5.4_rgbm.log"
echo "  RGBDM: PID=$PID_RGBDM  log: $LOG_DIR/gpt5.4_rgbdm.log"
echo ""
echo "Follow one log: tail -f $LOG_DIR/gpt5.4_rgb.log"
echo "Follow all logs: tail -f $LOG_DIR/gpt5.4_*.log"
echo "Stop all runs: kill $PID_RGB $PID_RGBD $PID_RGBM $PID_RGBDM"

STATUS=0
for pid in "$PID_RGB" "$PID_RGBD" "$PID_RGBM" "$PID_RGBDM"; do
    if ! wait "$pid"; then
        STATUS=1
    fi
done

echo ""
if [ "$STATUS" -eq 0 ]; then
    echo "=== All runs completed successfully ==="
else
    echo "=== One or more runs failed; check the logs above ==="
fi
exit "$STATUS"
