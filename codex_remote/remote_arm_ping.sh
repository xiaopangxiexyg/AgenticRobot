#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

ARM_IP=""
COUNT=3
TIMEOUT=1

usage() {
  cat <<'EOF'
Usage:
  remote_arm_ping.sh -H dex-remote --arm-ip 192.168.1.18

Options:
      --arm-ip HOST  Robot/controller IP or hostname.
      --count N      Ping count. Default: 3.
      --timeout SEC  Per-ping timeout. Default: 1.

Safety: L1, remote read-only network check.
EOF
  common_ssh_usage
}

while [[ $# -gt 0 ]]; do
  if parse_common_ssh_arg "$@"; then
    shift "$COMMON_ARGC"
    continue
  fi

  case "$1" in
    --arm-ip|--ip)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      ARM_IP="$2"
      shift 2
      ;;
    --count)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      COUNT="$2"
      shift 2
      ;;
    --timeout)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      TIMEOUT="$2"
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

[[ -n "$ARM_IP" ]] || die "missing --arm-ip"
[[ "$COUNT" =~ ^[0-9]+$ ]] || die "count must be a number: $COUNT"
[[ "$TIMEOUT" =~ ^[0-9]+$ ]] || die "timeout must be a number: $TIMEOUT"

finalize_common_ssh
print_wrapper_header "L1"
printf '[arm] ip=%s\n' "$ARM_IP"

remote_bash "$ARM_IP" "$COUNT" "$TIMEOUT" <<'REMOTE'
set -eu
arm_ip=$1
count=$2
timeout_s=$3
command -v ping >/dev/null 2>&1 || { printf 'ping is not installed on remote.\n' >&2; exit 3; }
ping -c "$count" -W "$timeout_s" "$arm_ip"
REMOTE
