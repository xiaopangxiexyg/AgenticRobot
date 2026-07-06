#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

PATTERN=""
ALL_USERS=false

usage() {
  cat <<'EOF'
Usage:
  remote_ps.sh -H dex-remote
  remote_ps.sh -H dex-remote --grep python
  remote_ps.sh -H dex-remote --all --grep tmux

Options:
      --grep TEXT  Case-insensitive filter on the ps output.
      --all        Show all users. Default shows current remote user's processes.

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
    --grep)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      PATTERN="$2"
      shift 2
      ;;
    --all)
      ALL_USERS=true
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

remote_bash "$PATTERN" "$ALL_USERS" <<'REMOTE'
set -eu
pattern=$1
all_users=$2

if [[ "$all_users" == "true" ]]; then
  output="$(ps -eo pid,ppid,stat,etime,user,comm,args --sort=user,pid)"
else
  output="$(ps -u "$(id -un)" -o pid,ppid,stat,etime,user,comm,args --sort=pid)"
fi

if [[ -n "$pattern" ]]; then
  printf '%s\n' "$output" | awk -v pattern="$pattern" '
    BEGIN { needle = tolower(pattern) }
    NR == 1 || index(tolower($0), needle) > 0
  '
else
  printf '%s\n' "$output"
fi
REMOTE
