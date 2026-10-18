#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/remote_common.sh"
repo="" python="" arms="" left_ip="" from_ip="" to_ip="" output="" preview=false apply=false
while [[ $# -gt 0 ]]; do
  if parse_common_ssh_arg "$@"; then shift "$COMMON_ARGC"; continue; fi
  case "$1" in
    --repo|--python|--arm-ip-assignment|--left-controller-ip|--from-ip|--to-ip|--output)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      case "$1" in
        --repo) repo="$2";; --python) python="$2";; --arm-ip-assignment) arms="$2";;
        --left-controller-ip) left_ip="$2";;
        --from-ip) from_ip="$2";; --to-ip) to_ip="$2";; --output) output="$2";;
      esac
      shift 2;;
    --dry-run) preview=true; shift;;
    --apply) apply=true; shift;;
    -h|--help)
      printf '%s\n' 'Preflight or explicitly apply a left-only UDP receiver-IP correction.' \
        'Required: -H HOST --repo PATH --python PATH --arm-ip-assignment PATH' \
        '          --left-controller-ip ADDRESS --from-ip ADDRESS --to-ip ADDRESS --output NEW_DIRECTORY' \
        'Default: preflight only, no SDK. --apply requires explicit per-occasion authorization.' \
        'Only IP changes; ten-second reception; verified rollback on failure; no movement.' \
        '--dry-run performs no SSH. Fixed 60s timeout with detached rollback guardian.'
      exit 0;;
    *) die "unknown argument: $1";;
  esac
done
for value in "$repo" "$python" "$arms" "$output"; do
  [[ "$value" == /* && "$value" != / ]] || die 'all paths must be explicit absolute paths'
done
[[ -n "$left_ip" && -n "$from_ip" && -n "$to_ip" ]] || die 'explicit controller, old and new receiver IPs are required'
finalize_common_ssh
print_wrapper_header 'L4 scoped left telemetry configuration; no movement/HOLD/STOP'
command="/usr/bin/timeout --signal=TERM --kill-after=5s 60s $(shell_single_quote "$python") -u -B -"
command+=" --repo $(shell_single_quote "$repo") --arm-ip-assignment $(shell_single_quote "$arms")"
command+=" --left-controller-ip $(shell_single_quote "$left_ip")"
command+=" --from-ip $(shell_single_quote "$from_ip") --to-ip $(shell_single_quote "$to_ip") --output $(shell_single_quote "$output")"
if [[ "$apply" == true ]]; then command+=" --apply"; fi
printf '[left-udp-destination] %s\n' "$command"
if [[ "$preview" == true ]]; then exit 0; fi
remote_exec "$command" < "$SCRIPT_DIR/../scripts/probes/set_rm65_left_udp_destination.py"
