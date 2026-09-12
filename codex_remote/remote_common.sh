#!/usr/bin/env bash
# Shared helpers for the Connection/codex remote wrappers.
# Source this file from wrapper scripts; do not execute it directly.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'remote_common.sh is a helper library. Source it from a wrapper script.\n' >&2
  exit 1
fi

# The execution target must come from this invocation. Ambient target values can
# silently redirect an otherwise identical command to another machine.
REMOTE_HOST=""
REMOTE_USER=""
REMOTE_PORT=""
KEY_FILE="${KEY_FILE:-$HOME/.ssh/id_ed25519}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-5}"
COMMON_ARGC=0
SSH_ARGS=()

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

info() {
  printf '[INFO] %s\n' "$*"
}

warn() {
  printf '[WARN] %s\n' "$*" >&2
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

expand_local_path() {
  case "$1" in
    "~") printf '%s\n' "$HOME" ;;
    "~/"*) printf '%s/%s\n' "$HOME" "${1#"~/"}" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

shell_single_quote() {
  printf "'"
  printf '%s' "$1" | sed "s/'/'\\\\''/g"
  printf "'"
}

parse_host_input() {
  local input="$1"

  if [[ "$input" == *@* && -z "$REMOTE_USER" ]]; then
    REMOTE_USER="${input%@*}"
    input="${input#*@}"
  fi

  if [[ "$input" != *:*:* && "$input" == *:* ]]; then
    REMOTE_PORT="${input##*:}"
    input="${input%:*}"
  fi

  REMOTE_HOST="$input"
}

remote_target() {
  if [[ -n "$REMOTE_USER" ]]; then
    printf '%s@%s\n' "$REMOTE_USER" "$REMOTE_HOST"
  else
    printf '%s\n' "$REMOTE_HOST"
  fi
}

parse_common_ssh_arg() {
  COMMON_ARGC=0
  case "${1:-}" in
    -H|--host)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      parse_host_input "$2"
      COMMON_ARGC=2
      ;;
    -u|--user)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      REMOTE_USER="$2"
      COMMON_ARGC=2
      ;;
    -p|-P|--port)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      REMOTE_PORT="$2"
      COMMON_ARGC=2
      ;;
    -k|--key)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      KEY_FILE="$(expand_local_path "$2")"
      COMMON_ARGC=2
      ;;
    --connect-timeout)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      CONNECT_TIMEOUT="$2"
      COMMON_ARGC=2
      ;;
    *)
      return 1
      ;;
  esac
  return 0
}

common_ssh_usage() {
  cat <<'EOF'
Common SSH options:
  -H, --host HOST           SSH host or alias. Also accepts user@host or host:port.
  -u, --user USER           SSH username. Optional when ~/.ssh/config handles it.
  -p, -P, --port PORT       SSH port. Optional when ~/.ssh/config handles it.
  -k, --key FILE            SSH private key. Default: ~/.ssh/id_ed25519 when it exists.
      --connect-timeout SEC SSH connect timeout. Default: 5.
EOF
}

finalize_common_ssh() {
  require_cmd ssh
  require_cmd sed

  [[ -n "$REMOTE_HOST" ]] || die "missing -H/--host"
  if [[ -n "$REMOTE_PORT" && ! "$REMOTE_PORT" =~ ^[0-9]+$ ]]; then
    die "SSH port must be a number: $REMOTE_PORT"
  fi
  if [[ ! "$CONNECT_TIMEOUT" =~ ^[0-9]+$ ]]; then
    die "connect timeout must be a number: $CONNECT_TIMEOUT"
  fi

  KEY_FILE="$(expand_local_path "$KEY_FILE")"
}

build_ssh_args() {
  SSH_ARGS=(
    ssh
    -o BatchMode=yes
    -o NumberOfPasswordPrompts=0
    -o ConnectTimeout="$CONNECT_TIMEOUT"
    -o StrictHostKeyChecking=yes
    -o ServerAliveInterval=10
    -o ServerAliveCountMax=3
  )

  if [[ -n "$REMOTE_PORT" ]]; then
    SSH_ARGS+=(-p "$REMOTE_PORT")
  fi
  if [[ -n "$KEY_FILE" && -f "$KEY_FILE" ]]; then
    SSH_ARGS+=(-i "$KEY_FILE")
  fi

  SSH_ARGS+=("$(remote_target)")
}

remote_exec() {
  build_ssh_args
  "${SSH_ARGS[@]}" "$@"
}

remote_bash() {
  local command="bash -s"
  if [[ $# -gt 0 ]]; then
    command+=" --"
    while [[ $# -gt 0 ]]; do
      command+=" $(shell_single_quote "$1")"
      shift
    done
  fi

  build_ssh_args
  "${SSH_ARGS[@]}" "$command"
}

ensure_trailing_slash() {
  if [[ "$1" == "/" ]]; then
    printf '/\n'
  else
    printf '%s/\n' "${1%/}"
  fi
}

build_rsync_rsh() {
  local rsh="ssh"

  rsh+=" -o BatchMode=yes"
  rsh+=" -o NumberOfPasswordPrompts=0"
  rsh+=" -o ConnectTimeout=$(shell_single_quote "$CONNECT_TIMEOUT")"
  rsh+=" -o StrictHostKeyChecking=yes"
  rsh+=" -o ServerAliveInterval=10"
  rsh+=" -o ServerAliveCountMax=3"

  if [[ -n "$REMOTE_PORT" ]]; then
    rsh+=" -p $(shell_single_quote "$REMOTE_PORT")"
  fi
  if [[ -n "$KEY_FILE" && -f "$KEY_FILE" ]]; then
    rsh+=" -i $(shell_single_quote "$KEY_FILE")"
  fi

  printf '%s\n' "$rsh"
}

print_wrapper_header() {
  local safety="$1"
  local project="${2:-}"
  local env_name="${3:-}"

  printf '[local] wrapper=%s safety=%s\n' "$(basename "$0")" "$safety"
  printf '[target] host=%s' "$(remote_target)"
  if [[ -n "$REMOTE_PORT" ]]; then
    printf ' port=%s' "$REMOTE_PORT"
  fi
  if [[ -n "$project" ]]; then
    printf ' project=%s' "$project"
  fi
  if [[ -n "$env_name" ]]; then
    printf ' env=%s' "$env_name"
  fi
  printf '\n'
}

confirm_or_die() {
  local prompt="$1"
  local answer

  if [[ ! -t 0 ]]; then
    die "confirmation required but stdin is not interactive"
  fi

  while true; do
    IFS= read -r -p "$prompt" answer
    case "$answer" in
      [Yy]|[Yy][Ee][Ss]) return 0 ;;
      ""|[Nn]|[Nn][Oo]) die "aborted by user" ;;
      *) printf 'Please answer y or n.\n' >&2 ;;
    esac
  done
}

control_risk_detected() {
  local arg lower

  for arg in "$@"; do
    lower="${arg,,}"
    case "$lower" in
      *confirm-motion*|*confirm_motion*|*confirm-reset*|*confirm_reset*|\
      *no-dry-run*|*no_dry_run*|*disable-dry-run*|*disable_dry_run*|\
      *robot_motion*|*real_robot*|*realrobot*|*rm65*|*gripper*|\
      *arm_home*|*arm-home*|*arm_enable*|*arm-enable*|*arm_disable*|*arm-disable*|\
      *estop*|*e-stop*|*emergency-stop*|*zero_velocity*|*zero-velocity*|\
      *trajectory*|*traj*|*replay*|*servo*|*impedance*|*force_control*|*force-control*|\
      *joint_control*|*joint-control*|*velocity_control*|*velocity-control*|\
      */arm/*|*/arm_*|*/robot/*|*/robot_*|*motion*|*movej*|*movel*)
        return 0
        ;;
    esac
  done

  return 1
}

print_control_risk_matches() {
  local arg lower

  for arg in "$@"; do
    lower="${arg,,}"
    case "$lower" in
      *confirm-motion*|*confirm_motion*|*confirm-reset*|*confirm_reset*|\
      *no-dry-run*|*no_dry_run*|*disable-dry-run*|*disable_dry_run*|\
      *robot_motion*|*real_robot*|*realrobot*|*rm65*|*gripper*|\
      *arm_home*|*arm-home*|*arm_enable*|*arm-enable*|*arm_disable*|*arm-disable*|\
      *estop*|*e-stop*|*emergency-stop*|*zero_velocity*|*zero-velocity*|\
      *trajectory*|*traj*|*replay*|*servo*|*impedance*|*force_control*|*force-control*|\
      *joint_control*|*joint-control*|*velocity_control*|*velocity-control*|\
      */arm/*|*/arm_*|*/robot/*|*/robot_*|*motion*|*movej*|*movel*)
        printf '  %s\n' "$arg" >&2
        ;;
    esac
  done
}

dry_run_marker_present() {
  local arg lower

  for arg in "$@"; do
    lower="${arg,,}"
    case "$lower" in
      --dry-run|--dry_run|--no-motion|--no_motion|--simulate|--simulation|--check-only|--check_only|dry-run|dry_run)
        return 0
        ;;
    esac
  done

  return 1
}

forbidden_dry_run_override_present() {
  local arg lower

  for arg in "$@"; do
    lower="${arg,,}"
    case "$lower" in
      *confirm-motion*|*confirm_motion*|*confirm-reset*|*confirm_reset*|\
      *no-dry-run*|*no_dry_run*|*disable-dry-run*|*disable_dry_run*|\
      --live*|--real*|--execute*|--force*|--home*|--enable*|--move*|--motion*)
        return 0
        ;;
    esac
  done

  return 1
}

assert_dry_run_control_args() {
  if forbidden_dry_run_override_present "$@"; then
    printf 'Refusing dry-control execution because an argument can override dry-run safety:\n' >&2
    print_control_risk_matches "$@"
    exit 5
  fi

  if ! dry_run_marker_present "$@"; then
    die "refusing dry-control execution without an explicit dry-run/no-motion/simulate marker"
  fi
}

is_allowed_dry_flag() {
  case "${1,,}" in
    --dry-run|--dry_run|--no-motion|--no_motion|--simulate|--simulation|--check-only|--check_only)
      return 0
      ;;
  esac

  return 1
}

assert_safe_dry_script_name() {
  local path="$1"
  local base lower

  base="$(basename "$path")"
  lower="${base,,}"
  case "$lower" in
    *dry*|*check*|*validate*|*simulate*|*simulation*|*noop*)
      return 0
      ;;
  esac

  die "dry-run arm wrapper only accepts scripts whose basename clearly indicates dry/check/validate/simulate/noop: $base"
}

assert_python_bin_name() {
  local python_bin="$1"
  local base lower

  base="$(basename "$python_bin")"
  lower="${base,,}"
  case "$lower" in
    python|python[0-9]*|pypy|pypy[0-9]*)
      return 0
      ;;
  esac

  die "--python must point to a Python interpreter, not: $python_bin"
}

assert_no_control_risk() {
  control_risk_detected "$@" || return 0

  printf 'Refusing a control-risk command through a generic L3 wrapper:\n' >&2
  print_control_risk_matches "$@"
  printf 'Use a purpose-built wrapper that constrains the exact action. Preflight evidence does not grant control permission.\n' >&2
  exit 5
}
