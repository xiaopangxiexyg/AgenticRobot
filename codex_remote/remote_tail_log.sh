#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

LOG_PATH=""
LINES=120
FOLLOW=false

usage() {
  cat <<'EOF'
Usage:
  remote_tail_log.sh -H dex-remote --path '~/DexTeleop/logs/latest.log'
  remote_tail_log.sh -H dex-remote --path /tmp/run.log --lines 200 --follow

Options:
      --path PATH   Remote log file path.
      --lines N     Number of lines. Default: 120.
      --follow      Follow the log.

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
    --path|-f)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      LOG_PATH="$2"
      shift 2
      ;;
    --lines|-n)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      LINES="$2"
      shift 2
      ;;
    --follow|-F)
      FOLLOW=true
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

[[ -n "$LOG_PATH" ]] || die "missing --path"
[[ "$LINES" =~ ^[0-9]+$ ]] || die "lines must be a number: $LINES"

finalize_common_ssh
print_wrapper_header "L1"
printf '[log] path=%s lines=%s follow=%s\n' "$LOG_PATH" "$LINES" "$FOLLOW"

remote_bash "$LOG_PATH" "$LINES" "$FOLLOW" <<'REMOTE'
set -Eeuo pipefail
path=$1
lines=$2
follow=$3

case "$path" in
  "~") path=$HOME ;;
  "~/"*) path=$HOME/${path#~/} ;;
  /*) ;;
  *) path=$HOME/$path ;;
esac

[[ -f "$path" ]] || { printf 'Log file does not exist: %s\n' "$path" >&2; exit 4; }

if [[ "$follow" == "true" ]]; then
  exec tail -n "$lines" -F "$path"
fi

tail -n "$lines" "$path"
REMOTE
