#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
PYTHON_BIN="$(bash "$SCRIPT_DIR/resolve_dev_python.sh")"

cd -- "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m control_station.quest_calibration.launch "$@"
