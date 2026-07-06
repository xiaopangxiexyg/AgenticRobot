#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

TREE=false

usage() {
  cat <<'EOF'
Usage:
  remote_lsusb.sh -H dex-remote
  remote_lsusb.sh -H dex-remote --tree

Options:
      --tree   Run lsusb -t instead of plain lsusb.

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
    --tree)
      TREE=true
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

remote_bash "$TREE" <<'REMOTE'
set -eu
tree=$1
if ! command -v lsusb >/dev/null 2>&1; then
  printf 'lsusb is not installed on the remote machine.\n' >&2
  exit 4
fi
if [[ "$tree" == "true" ]]; then
  lsusb -t
else
  lsusb
fi
REMOTE
