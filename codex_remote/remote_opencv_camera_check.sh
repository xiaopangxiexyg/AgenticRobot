#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

PROJECT=""
ENV_NAME=""
PYTHON_BIN="python"
DEVICE="/dev/video0"
FRAMES=5

usage() {
  cat <<'EOF'
Usage:
  remote_opencv_camera_check.sh -H dex-remote --device /dev/video0 --env dexteleop310

Options:
      --device DEV       OpenCV camera path or index. Default: /dev/video0.
      --frames N         Number of frames to read. Default: 5.
      --project DIR      Remote project directory to cd into before checking.
      --env NAME_OR_DIR  Conda/micromamba env name, or venv directory.
      --python PYTHON    Python executable after env activation. Default: python.

Safety: L1, remote read-only camera open/read test.
EOF
  common_ssh_usage
}

while [[ $# -gt 0 ]]; do
  if parse_common_ssh_arg "$@"; then
    shift "$COMMON_ARGC"
    continue
  fi

  case "$1" in
    --device)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      DEVICE="$2"
      shift 2
      ;;
    --frames)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      FRAMES="$2"
      shift 2
      ;;
    --project)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      PROJECT="$2"
      shift 2
      ;;
    --env)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      ENV_NAME="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ "$FRAMES" =~ ^[0-9]+$ ]] || die "frames must be a number: $FRAMES"
assert_python_bin_name "$PYTHON_BIN"

finalize_common_ssh
print_wrapper_header "L1" "$PROJECT" "$ENV_NAME"

remote_bash "$PROJECT" "$ENV_NAME" "$PYTHON_BIN" "$DEVICE" "$FRAMES" <<'REMOTE'
set -Eeuo pipefail

project=$1
env_name=$2
python_bin=$3
device=$4
frames=$5

resolve_remote_path() {
  local value="$1"
  case "$value" in
    "") return 1 ;;
    "~") printf '%s\n' "$HOME" ;;
    "~/"*) printf '%s/%s\n' "$HOME" "${value#"~/"}" ;;
    /*) printf '%s\n' "$value" ;;
    *) printf '%s/%s\n' "$HOME" "$value" ;;
  esac
}

activate_env() {
  local env_name="$1"
  local conda_sh
  [[ -n "$env_name" ]] || return 0
  if [[ -d "$env_name" && -f "$env_name/bin/activate" ]]; then
    source "$env_name/bin/activate"
    return 0
  fi
  for conda_sh in "$HOME/anaconda3/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/mambaforge/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$conda_sh" ]]; then
      source "$conda_sh"
      conda activate "$env_name"
      return 0
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$env_name"
    return 0
  fi
  if command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook -s bash)"
    micromamba activate "$env_name"
    return 0
  fi
  printf 'Cannot activate env: %s\n' "$env_name" >&2
  return 3
}

if [[ -n "$project" ]]; then
  project="$(resolve_remote_path "$project")"
  [[ -d "$project" ]] || { printf 'Project directory does not exist: %s\n' "$project" >&2; exit 3; }
  cd "$project"
fi

activate_env "$env_name"

"$python_bin" - "$device" "$frames" <<'PY'
import sys
import time

device = sys.argv[1]
frames = int(sys.argv[2])

try:
    import cv2
except Exception as exc:
    print(f"[opencv] import failed: {exc}", file=sys.stderr)
    sys.exit(3)

source = int(device) if device.isdigit() else device
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"[opencv] failed to open: {device}", file=sys.stderr)
    sys.exit(4)

ok_count = 0
shape = None
for _ in range(frames):
    ok, frame = cap.read()
    if ok and frame is not None:
        ok_count += 1
        shape = frame.shape
    time.sleep(0.02)
cap.release()

print(f"[opencv] device={device}")
print(f"[opencv] requested_frames={frames}")
print(f"[opencv] ok_frames={ok_count}")
print(f"[opencv] last_shape={shape}")
sys.exit(0 if ok_count > 0 else 4)
PY
REMOTE
