#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

usage() {
  cat <<'EOF'
Usage:
  remote_dev_nodes.sh -H dex-remote

Lists common camera, serial, input, HID, USB bus, and GPU/render nodes.

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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

finalize_common_ssh
print_wrapper_header "L1"

remote_bash <<'REMOTE'
set -eu

print_glob() {
  local label="$1"
  shift
  printf '\n[%s]\n' "$label"
  local found=false
  local node
  for node in "$@"; do
    if [[ -e "$node" ]]; then
      ls -ld "$node"
      found=true
    fi
  done
  if [[ "$found" == "false" ]]; then
    printf '(none)\n'
  fi
}

print_glob "camera" /dev/video*
print_glob "serial" /dev/ttyUSB* /dev/ttyACM*
print_glob "input" /dev/input/event* /dev/input/js*
print_glob "hidraw" /dev/hidraw*
print_glob "usb-bus" /dev/bus/usb/*/*
print_glob "gpu-render" /dev/nvidia* /dev/dri/renderD*
REMOTE
