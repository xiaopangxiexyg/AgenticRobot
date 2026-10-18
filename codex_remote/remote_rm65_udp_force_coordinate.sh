#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/remote_common.sh"
repo="" python="" arms="" left_ip="" right_ip="" output="" preview=false apply=false
while [[ $# -gt 0 ]]; do
  if parse_common_ssh_arg "$@"; then shift "$COMMON_ARGC"; continue; fi
  case "$1" in
    --repo|--python|--arm-ip-assignment|--left-controller-ip|--right-controller-ip|--output)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      case "$1" in
        --repo) repo="$2";; --python) python="$2";; --arm-ip-assignment) arms="$2";;
        --left-controller-ip) left_ip="$2";; --right-controller-ip) right_ip="$2";; --output) output="$2";;
      esac
      shift 2;;
    --dry-run) preview=true; shift;;
    --apply) apply=true; shift;;
    -h|--help)
      printf '%s\n' 'Set only both RM65 UDP force_coordinate fields to tool=2.' \
        'Required: -H HOST --repo PATH --python PATH --arm-ip-assignment PATH' \
        '          --left-controller-ip ADDRESS --right-controller-ip ADDRESS --output NEW_DIRECTORY' \
        'Default: preflight only, no SDK. --apply requires explicit per-occasion authorization.' \
        'Preserves rate/port/IP/custom fields. Ten-second passive verification, rollback on failure.' \
        'No movement/HOLD/STOP/cameras/grippers. --dry-run performs no SSH. Fixed 60s timeout.'
      exit 0;;
    *) die "unknown argument: $1";;
  esac
done
for value in "$repo" "$python" "$arms" "$output"; do
  [[ "$value" == /* && "$value" != / ]] || die 'all paths must be explicit absolute paths'
done
[[ -n "$left_ip" && -n "$right_ip" ]] || die 'explicit left/right controller addresses are required'
finalize_common_ssh
print_wrapper_header 'L4 scoped UDP force coordinates; no movement/HOLD/STOP'
command="/usr/bin/timeout --signal=TERM --kill-after=5s 60s $(shell_single_quote "$python") -u -B -"
command+=" --repo $(shell_single_quote "$repo") --arm-ip-assignment $(shell_single_quote "$arms")"
command+=" --left-controller-ip $(shell_single_quote "$left_ip") --right-controller-ip $(shell_single_quote "$right_ip")"
command+=" --output $(shell_single_quote "$output")"
if [[ "$apply" == true ]]; then command+=" --apply"; fi
printf '[udp-force-coordinate] %s\n' "$command"
if [[ "$preview" == true ]]; then exit 0; fi
remote_exec "$command" < "$SCRIPT_DIR/../scripts/probes/set_rm65_udp_force_coordinate.py"
