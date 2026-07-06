#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

usage() {
  cat <<'EOF'
Usage:
  remote_whoami.sh -H dex-remote
  remote_whoami.sh -H zc@192.168.1.50

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
printf '[remote] user=%s\n' "$(id -un)"
printf '[remote] id=%s\n' "$(id)"
printf '[remote] hostname=%s\n' "$(hostname)"
printf '[remote] pwd=%s\n' "$PWD"
printf '[remote] home=%s\n' "$HOME"
printf '[remote] date=%s\n' "$(date -Is 2>/dev/null || date)"
printf '[remote] uname=%s\n' "$(uname -a)"
REMOTE
