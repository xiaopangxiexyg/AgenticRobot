#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="$REPO_ROOT/model/perception/graspgenx"
ASSET_ROOT="${DEXTELEOP_ASSET_ROOT:-$REPO_ROOT/assets}"
HOST="127.0.0.1"
PORT="5556"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_graspgenx_server.sh [OPTIONS]

Start the isolated GraspGenX inference service for AgenticRobot point-grasp.
The service is restricted to a loopback endpoint and never connects to a robot.

Options:
  --asset-root PATH  Shared AgenticRobot asset root (default: assets/ or
                     DEXTELEOP_ASSET_ROOT).
  --host HOST        Loopback bind host: 127.0.0.1 or localhost
                     (default: 127.0.0.1).
  --port PORT        TCP port in [1, 65535] (default: 5556).
  -h, --help         Show this help.
EOF
}

while (($#)); do
  case "$1" in
    --asset-root)
      ASSET_ROOT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$HOST" == "localhost" ]]; then
  HOST="127.0.0.1"
elif [[ "$HOST" != "127.0.0.1" ]]; then
  echo "GraspGenX server host must be loopback." >&2
  exit 2
fi
if [[ ! "$PORT" =~ ^[0-9]+$ ]]; then
  echo "GraspGenX server port must be an integer in [1, 65535]." >&2
  exit 2
fi
PORT=$((10#$PORT))
if ((PORT < 1 || PORT > 65535)); then
  echo "GraspGenX server port must be an integer in [1, 65535]." >&2
  exit 2
fi

BUNDLE_ROOT="$ASSET_ROOT/perception/graspgenx-point-grasp"
CHECKPOINT_ROOT="$BUNDLE_ROOT/checkpoints"
GRIPPER_CFG_ROOT="$BUNDLE_ROOT/gripper_descriptions"
VENV_PYTHON="$BUNDLE_ROOT/runtime/.venv/bin/python"

bash "$REPO_ROOT/scripts/install_graspgenx_env.sh" \
  --asset-root "$ASSET_ROOT" \
  --verify-only

export GRASPGENX_CHECKPOINT_DIR="$CHECKPOINT_ROOT"
export GRASPGENX_GRIPPER_CFG_DIR="$GRIPPER_CFG_ROOT"
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$VENV_PYTHON" "$SOURCE_ROOT/client-server/graspgenx_server.py" \
  --config "$CHECKPOINT_ROOT/release" \
  --assets_dir "$GRIPPER_CFG_ROOT" \
  --host "$HOST" \
  --port "$PORT"
