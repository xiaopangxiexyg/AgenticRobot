#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

DETAILS=false

usage() {
  cat <<'EOF'
Usage:
  remote_camera_list.sh -H dex-remote
  remote_camera_list.sh -H dex-remote --details

Options:
      --details   Include v4l2-ctl --all output when v4l2-ctl is available.

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
    --details)
      DETAILS=true
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

finalize_common_ssh
print_wrapper_header "L1"

remote_bash "$DETAILS" <<'REMOTE'
set -eu
details=$1

if command -v v4l2-ctl >/dev/null 2>&1; then
  printf '[v4l2 devices]\n'
  v4l2-ctl --list-devices || true
  if [[ "$details" == "true" ]]; then
    for dev in /dev/video*; do
      [[ -e "$dev" ]] || continue
      printf '\n[%s]\n' "$dev"
      v4l2-ctl -d "$dev" --all || true
      printf '\n[formats %s]\n' "$dev"
      v4l2-ctl -d "$dev" --list-formats-ext || true
    done
  fi
else
  printf '[v4l2] v4l2-ctl not found\n'
fi

printf '\n[/dev/video*]\n'
found=false
for dev in /dev/video*; do
  if [[ -e "$dev" ]]; then
    ls -l "$dev"
    found=true
  fi
done
if [[ "$found" == "false" ]]; then
  printf '(none)\n'
fi
REMOTE
