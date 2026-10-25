#!/bin/bash
# Run the InternVL3-8B RGBDM input condition.
# Usage: bash scripts/run/Run_internvl3_parallel.sh

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

LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
mkdir -p "$LOG_DIR"

"$PYTHON_BIN" -m src.main --model internvl3 --mode rgbdm --print-config >/dev/null || exit $?

start_run() {
    local label="$1"
    local mode="$2"
    local log_file="$3"

    echo "Starting $label..."
    nohup "$PYTHON_BIN" -m src.main --model internvl3 --mode "$mode" \
        > "$log_file" 2>&1 &
}

start_run "InternVL3 RGBDM" "rgbdm" "$LOG_DIR/internvl3_rgbdm.log"
PID_RGBDM=$!

echo ""
echo "InternVL3 RGBDM experiment has started:"
echo "  RGBDM: PID=$PID_RGBDM  log: $LOG_DIR/internvl3_rgbdm.log"
echo ""
echo "Follow log: tail -f $LOG_DIR/internvl3_rgbdm.log"
echo "Stop run: kill $PID_RGBDM"

STATUS=0
if ! wait "$PID_RGBDM"; then
    STATUS=1
fi

echo ""
if [ "$STATUS" -eq 0 ]; then
    echo "=== InternVL3 RGBDM run completed successfully ==="
else
    echo "=== InternVL3 RGBDM run failed; check the log above ==="
fi
exit "$STATUS"
