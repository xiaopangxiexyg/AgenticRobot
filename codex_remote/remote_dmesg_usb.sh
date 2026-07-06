#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

LINES=240

usage() {
  cat <<'EOF'
Usage:
  remote_dmesg_usb.sh -H dex-remote
  remote_dmesg_usb.sh -H dex-remote --lines 500

Options:
      --lines N   Number of recent dmesg lines to inspect. Default: 240.

Safety: L1, remote read-only.
EOF
  common_ssh_usage
}

while [[ $# -gt 0 ]]; do
  if parse_common_ssh_arg "$@"; then
    shift "$COMMON_ARGC"
    continue
  fi

  case "$1" in
    --lines|-n)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      LINES="$2"
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

[[ "$LINES" =~ ^[0-9]+$ ]] || die "lines must be a number: $LINES"

finalize_common_ssh
print_wrapper_header "L1"

remote_bash "$LINES" <<'REMOTE'
set -eu
lines=$1
if ! command -v dmesg >/dev/null 2>&1; then
  printf 'dmesg command not found on remote.\n' >&2
  exit 4
fi

if ! output="$(dmesg -T 2>&1 | tail -n "$lines")"; then
  printf '%s\n' "$output" >&2
  exit 4
fi

printf '%s\n' "$output" | grep -Ei 'usb|uvc|video|camera|ttyUSB|ttyACM|hid|input|xhc|xhci|serial|realsense' || true
REMOTE
