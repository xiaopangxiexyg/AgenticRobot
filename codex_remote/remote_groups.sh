#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

usage() {
  cat <<'EOF'
Usage:
  remote_groups.sh -H dex-remote

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
printf '[user] %s\n' "$(id -un)"
printf '[id] %s\n' "$(id)"
printf '[groups] %s\n' "$(groups)"

printf '\n[common hardware groups]\n'
for group_name in video dialout plugdev input render docker; do
  if getent group "$group_name" >/dev/null 2>&1; then
    getent group "$group_name"
  else
    printf '%s: (missing)\n' "$group_name"
  fi
done
REMOTE
