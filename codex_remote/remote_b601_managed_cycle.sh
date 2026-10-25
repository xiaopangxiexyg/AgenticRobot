#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/remote_common.sh"

PROJECT="" PYTHON_BIN="python" OPERATION=preflight PREVIEW=false
ARGS=()
usage() {
  cat <<'EOF'
Usage:
  remote_b601_managed_cycle.sh -H HOST --project /repo --python /env/bin/python \
    --operation preflight|start --expected-revision FULL_SHA --config /config.yaml \
    --rebot-source-root /vendor/rebot --motorbridge-root /vendor/bridge --output-root /runs \
    [--recipe near-zero|four-centres] [--field-report /closed-source.json]
  remote_b601_managed_cycle.sh -H HOST --project /repo --python /env/bin/python \
    --operation status|stop --run-directory /runs/managed-ID

Purpose-built B601 single-cycle management. Start requires explicit
hardware authorization. Fixed existing owner/profile/limits; no arbitrary
command, profile, torque override, reset, retry or kill. Stop requests protective
hold only; it cannot request full disable. Status and preflight do not send motor
commands. --dry-run previews without SSH. No authority token is created.
EOF
  common_ssh_usage
}
while [[ $# -gt 0 ]]; do
  if parse_common_ssh_arg "$@"; then shift "$COMMON_ARGC"; continue; fi
  case "$1" in
    --project|--python|--operation)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      case "$1" in
        --project) PROJECT="$2";; --python) PYTHON_BIN="$2";; --operation) OPERATION="$2";;
      esac
      shift 2;;
    --expected-revision|--config|--rebot-source-root|--motorbridge-root|--output-root|--field-report|--run-directory)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      ARGS+=("$1" "$2"); shift 2;;
    --recipe)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      case "$2" in near-zero|four-centres) ;; *) die 'invalid --recipe';; esac
      ARGS+=("$1" "$2"); shift 2;;
    --dry-run) PREVIEW=true; shift;;
    -h|--help) usage; exit 0;;
    *) die "unknown argument: $1";;
  esac
done
[[ -n "$REMOTE_HOST" ]] || die 'explicit -H/--host is required'
[[ "$PROJECT" == /* && "$PROJECT" != / ]] || die '--project must be an absolute checkout path'
case "$OPERATION" in preflight|start|status|stop) ;; *) die 'invalid --operation';; esac
assert_python_bin_name "$PYTHON_BIN"
finalize_common_ssh
print_wrapper_header 'purpose-built B601 single-cycle; preflight/status read-only, start motion, stop protective hold' "$PROJECT"
command="cd -- $(shell_single_quote "$PROJECT") && exec $(shell_single_quote "$PYTHON_BIN") -u -m control_station.b601_torque_console.managed_run"
command+=" --operation $(shell_single_quote "$OPERATION")"
for value in "${ARGS[@]}"; do command+=" $(shell_single_quote "$value")"; done
printf '[managed-cycle] %s\n' "$command"
if [[ "$PREVIEW" == true ]]; then exit 0; fi
remote_exec "$command"
