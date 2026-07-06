#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

DEVICE="/dev/video0"
LOCAL_OUT="./debug/camera"
REMOTE_DIR="/tmp/codex_camera_snapshot"
WIDTH="1280"
HEIGHT="720"
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  remote_camera_snapshot.sh -H dex-remote --device /dev/video0 --local-out ./debug/camera

Options:
      --device DEV       Remote camera device. Default: /dev/video0.
      --local-out DIR    Local output directory. Default: ./debug/camera.
      --remote-dir DIR   Remote temporary output directory. Default: /tmp/codex_camera_snapshot.
      --width N          Capture width for ffmpeg/fswebcam/OpenCV fallback. Default: 1280.
      --height N         Capture height. Default: 720.
      --dry-run          Print actions without capturing or pulling.

Safety: L2, remote non-motion write and local pull.
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
    --local-out|--local)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      LOCAL_OUT="$2"
      shift 2
      ;;
    --remote-dir)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      REMOTE_DIR="$2"
      shift 2
      ;;
    --width)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      WIDTH="$2"
      shift 2
      ;;
    --height)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      HEIGHT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
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

[[ "$WIDTH" =~ ^[0-9]+$ ]] || die "width must be a number: $WIDTH"
[[ "$HEIGHT" =~ ^[0-9]+$ ]] || die "height must be a number: $HEIGHT"

finalize_common_ssh
require_cmd rsync

LOCAL_OUT="$(expand_local_path "$LOCAL_OUT")"
mkdir -p "$LOCAL_OUT"
LOCAL_OUT="$(cd "$LOCAL_OUT" && pwd -P)"

print_wrapper_header "L2"
printf '[camera] device=%s size=%sx%s\n' "$DEVICE" "$WIDTH" "$HEIGHT"
printf '[camera] remote_dir=%s local_out=%s dry_run=%s\n' "$REMOTE_DIR" "$LOCAL_OUT" "$DRY_RUN"

set +e
capture_output="$(
  remote_bash "$DEVICE" "$REMOTE_DIR" "$WIDTH" "$HEIGHT" "$DRY_RUN" <<'REMOTE'
set -Eeuo pipefail

device=$1
remote_dir=$2
width=$3
height=$4
dry_run=$5

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

remote_dir="$(resolve_remote_path "$remote_dir")"
out="$remote_dir/snapshot_$(date +%Y%m%d_%H%M%S).jpg"

printf '[run] capture device=%s out=%s\n' "$device" "$out"
if [[ "$dry_run" == "true" ]]; then
  printf '[result] dry-run only\n'
  exit 0
fi

[[ -e "$device" ]] || { printf 'Camera device does not exist: %s\n' "$device" >&2; exit 4; }
mkdir -p "$remote_dir"

if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -y -hide_banner -loglevel error -f v4l2 -video_size "${width}x${height}" -i "$device" -frames:v 1 "$out"
elif command -v fswebcam >/dev/null 2>&1; then
  fswebcam -q -d "$device" -r "${width}x${height}" --no-banner "$out"
elif command -v python3 >/dev/null 2>&1; then
  python3 - "$device" "$out" "$width" "$height" <<'PY'
import sys

device, out, width, height = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
try:
    import cv2
except Exception as exc:
    print(f"OpenCV fallback unavailable: {exc}", file=sys.stderr)
    sys.exit(3)

cap = cv2.VideoCapture(device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
ok, frame = cap.read()
cap.release()
if not ok or frame is None:
    print(f"Failed to read a frame from {device}", file=sys.stderr)
    sys.exit(4)
if not cv2.imwrite(out, frame):
    print(f"Failed to write {out}", file=sys.stderr)
    sys.exit(6)
PY
else
  printf 'Need ffmpeg, fswebcam, or python3+opencv on remote to capture a frame.\n' >&2
  exit 3
fi

[[ -s "$out" ]] || { printf 'Snapshot file is empty: %s\n' "$out" >&2; exit 6; }
printf 'SNAPSHOT\t%s\n' "$out"
REMOTE
)"
capture_status=$?
set -e

printf '%s\n' "$capture_output"

if [[ "$DRY_RUN" == "true" ]]; then
  exit "$capture_status"
fi
if [[ "$capture_status" -ne 0 ]]; then
  exit "$capture_status"
fi

snapshot_path="$(awk -F '\t' '$1 == "SNAPSHOT" {print $2}' <<< "$capture_output" | tail -n 1)"
[[ -n "$snapshot_path" ]] || die "remote capture succeeded but did not report a snapshot path"

rsh="$(build_rsync_rsh)"
args=(rsync -az -s --no-owner --no-group -e "$rsh" "$(remote_target):$snapshot_path" "$(ensure_trailing_slash "$LOCAL_OUT")")
printf '[run]'
printf ' %q' "${args[@]}"
printf '\n'
"${args[@]}"
printf '[result] local_out=%s/%s\n' "$LOCAL_OUT" "$(basename "$snapshot_path")"
