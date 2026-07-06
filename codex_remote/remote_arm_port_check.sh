#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

ARM_IP=""
ARM_PORT=""
TIMEOUT=2

usage() {
  cat <<'EOF'
Usage:
  remote_arm_port_check.sh -H dex-remote --arm-ip 192.168.1.18 --port 8080

Options:
      --arm-ip HOST  Robot/controller IP or hostname.
      --port PORT    TCP port to probe.
      --timeout SEC  Connect timeout. Default: 2.

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
    --port|--arm-port)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      ARM_PORT="$2"
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
[[ -n "$ARM_PORT" ]] || die "missing --port"
[[ "$ARM_PORT" =~ ^[0-9]+$ ]] || die "port must be a number: $ARM_PORT"
[[ "$TIMEOUT" =~ ^[0-9]+$ ]] || die "timeout must be a number: $TIMEOUT"

finalize_common_ssh
print_wrapper_header "L1"
printf '[arm] ip=%s port=%s\n' "$ARM_IP" "$ARM_PORT"

remote_bash "$ARM_IP" "$ARM_PORT" "$TIMEOUT" <<'REMOTE'
set -eu
arm_ip=$1
arm_port=$2
timeout_s=$3

if command -v nc >/dev/null 2>&1; then
  nc -zvw "$timeout_s" "$arm_ip" "$arm_port"
  exit $?
fi

if command -v python3 >/dev/null 2>&1; then
  python3 - "$arm_ip" "$arm_port" "$timeout_s" <<'PY'
import socket
import sys

host, port, timeout_s = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(timeout_s)
try:
    sock.connect((host, port))
except OSError as exc:
    print(f"[port] closed or unreachable: {host}:{port} ({exc})", file=sys.stderr)
    sys.exit(4)
finally:
    sock.close()
print(f"[port] reachable: {host}:{port}")
PY
  exit $?
fi

printf 'Need nc or python3 on remote for TCP port checks.\n' >&2
exit 3
REMOTE
