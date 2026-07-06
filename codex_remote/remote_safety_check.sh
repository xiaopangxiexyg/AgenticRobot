#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=remote_common.sh
source "$SCRIPT_DIR/remote_common.sh"

PROJECT=""
ARM_IP=""
ARM_PORTS=()
REQUIRED_DEVICES=()
NO_TOKEN=false
TOKEN_PATH="/tmp/codex_remote_safety_ok"

usage() {
  cat <<'EOF'
Usage:
  remote_safety_check.sh -H dex-remote --arm-ip 192.168.1.18 --arm-port 8080
  remote_safety_check.sh -H dex-remote --project '~/DexTeleop' --require-device /dev/video0

Options:
      --project DIR        Remote project directory that should exist.
      --arm-ip HOST        Robot/controller IP or hostname to ping.
      --arm-port PORT      TCP port to check. Can be repeated.
      --require-device DEV Device node that must exist. Can be repeated.
      --token-path PATH    Remote safety-ok token path. Default: /tmp/codex_remote_safety_ok.
      --no-token           Do not write a remote safety-ok token.

Safety: L1/L2. Read-only checks, plus optional token file write.
EOF
  common_ssh_usage
}

while [[ $# -gt 0 ]]; do
  if parse_common_ssh_arg "$@"; then
    shift "$COMMON_ARGC"
    continue
  fi

  case "$1" in
    --project)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      PROJECT="$2"
      shift 2
      ;;
    --arm-ip|--ip)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      ARM_IP="$2"
      shift 2
      ;;
    --arm-port|--port)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      ARM_PORTS+=("$2")
      shift 2
      ;;
    --require-device)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      REQUIRED_DEVICES+=("$2")
      shift 2
      ;;
    --token-path)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      TOKEN_PATH="$2"
      shift 2
      ;;
    --no-token)
      NO_TOKEN=true
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

for port in "${ARM_PORTS[@]}"; do
  [[ "$port" =~ ^[0-9]+$ ]] || die "arm port must be a number: $port"
done
if [[ "${#ARM_PORTS[@]}" -gt 0 && -z "$ARM_IP" ]]; then
  die "--arm-port requires --arm-ip"
fi

finalize_common_ssh
print_wrapper_header "L1/L2" "$PROJECT"

remote_bash "$PROJECT" "$ARM_IP" "$NO_TOKEN" "$TOKEN_PATH" "${#ARM_PORTS[@]}" "${ARM_PORTS[@]}" "${#REQUIRED_DEVICES[@]}" "${REQUIRED_DEVICES[@]}" <<'REMOTE'
set -Eeuo pipefail

project=$1
arm_ip=$2
no_token=$3
token_path=$4
port_count=$5
shift 5

arm_ports=()
for ((i = 0; i < port_count; i++)); do
  arm_ports+=("$1")
  shift
done

device_count=$1
shift
required_devices=()
for ((i = 0; i < device_count; i++)); do
  required_devices+=("$1")
  shift
done

resolve_remote_path() {
  local value="$1"
  case "$value" in
    "") return 1 ;;
    "~") printf '%s\n' "$HOME" ;;
    "~/"*) printf '%s/%s\n' "$HOME" "${value#"~/"}" ;;
    /*) printf '%s\n' "$value" ;;
    *) printf '%s/%s\n' "$HOME" "$value" ;;
  esac
}

failures=0
guard_checks=0

check_ok() {
  printf '[ok] %s\n' "$*"
}

check_fail() {
  printf '[fail] %s\n' "$*" >&2
  failures=$((failures + 1))
}

printf '[remote] hostname=%s\n' "$(hostname)"
printf '[remote] user=%s\n' "$(id -un)"
printf '[remote] date=%s\n' "$(date -Is 2>/dev/null || date)"
printf '[remote] groups=%s\n' "$(groups)"

if [[ -n "$project" ]]; then
  resolved_project="$(resolve_remote_path "$project")"
  if [[ -d "$resolved_project" ]]; then
    check_ok "project exists: $resolved_project"
  else
    check_fail "project missing: $resolved_project"
  fi
fi

for dev in "${required_devices[@]}"; do
  guard_checks=$((guard_checks + 1))
  if [[ -e "$dev" ]]; then
    ls -ld "$dev" | sed 's/^/[device] /'
    check_ok "device exists: $dev"
  else
    check_fail "device missing: $dev"
  fi
done

if [[ -n "$arm_ip" ]]; then
  guard_checks=$((guard_checks + 1))
  if command -v ping >/dev/null 2>&1 && ping -c 1 -W 1 "$arm_ip" >/dev/null 2>&1; then
    check_ok "arm ping reachable: $arm_ip"
  else
    check_fail "arm ping failed: $arm_ip"
  fi

  for port in "${arm_ports[@]}"; do
    guard_checks=$((guard_checks + 1))
    if command -v nc >/dev/null 2>&1; then
      if nc -zw 2 "$arm_ip" "$port" >/dev/null 2>&1; then
        check_ok "arm port reachable: $arm_ip:$port"
      else
        check_fail "arm port unreachable: $arm_ip:$port"
      fi
    elif command -v python3 >/dev/null 2>&1; then
      if python3 - "$arm_ip" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(2)
try:
    sock.connect((host, port))
finally:
    sock.close()
PY
      then
        check_ok "arm port reachable: $arm_ip:$port"
      else
        check_fail "arm port unreachable: $arm_ip:$port"
      fi
    else
      check_fail "cannot check TCP port without nc or python3: $arm_ip:$port"
    fi
  done
fi

if [[ "$failures" -ne 0 ]]; then
  printf '[result] safety check failed: %s failure(s)\n' "$failures" >&2
  exit 5
fi

if [[ "$no_token" != "true" ]]; then
  token_path="$(resolve_remote_path "$token_path")"
  if [[ "$guard_checks" -eq 0 ]]; then
    printf 'Refusing to write safety token because no hardware/control guard checks were requested.\n' >&2
    printf 'Use --no-token for informational checks, or add --arm-ip/--require-device.\n' >&2
    exit 5
  fi
  case "$token_path" in
    /tmp/codex_*|"$HOME"/.cache/codex_remote_safety/*) ;;
    *) printf 'Refusing safety token path outside /tmp/codex_* or ~/.cache/codex_remote_safety/: %s\n' "$token_path" >&2; exit 5 ;;
  esac
  mkdir -p "$(dirname "$token_path")"
  {
    printf 'created_at=%s\n' "$(date +%s)"
    printf 'host=%s\n' "$(hostname)"
    printf 'user=%s\n' "$(id -un)"
    printf 'guard_checks=%s\n' "$guard_checks"
    printf 'arm_ip=%s\n' "$arm_ip"
    printf 'project=%s\n' "${resolved_project:-}"
  } > "$token_path"
  chmod 600 "$token_path"
  printf '[result] safety_token=%s\n' "$token_path"
fi

printf '[result] safety check passed\n'
REMOTE
