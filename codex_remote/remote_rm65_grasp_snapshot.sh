#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/remote_common.sh"
repo="" python="" cameras="" arms="" output="" preview=false
while [[ $# -gt 0 ]]; do
  if parse_common_ssh_arg "$@"; then shift "$COMMON_ARGC"; continue; fi
  case "$1" in
    --repo|--python|--camera-config|--arm-ip-assignment|--output)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      case "$1" in
        --repo) repo="$2";; --python) python="$2";; --camera-config) cameras="$2";;
        --arm-ip-assignment) arms="$2";; --output) output="$2";;
      esac
      shift 2;;
    --dry-run) preview=true; shift;;
    -h|--help)
      printf '%s\n' 'Read-only RM65 qpos + RealSense RGB-D snapshot. No motion/HOLD/STOP/reset.' \
        'Explicit device-read authorization required for each target.' \
        'Required: -H HOST --repo PATH --python PATH --camera-config PATH --arm-ip-assignment PATH --output NEW_DIR' \
        'Optional: --dry-run (no connection). 60s userspace timeout; kernel D-state cannot be forcibly interrupted.'
      exit 0;;
    *) die "unknown argument: $1";;
  esac
done
for value in "$repo" "$python" "$cameras" "$arms" "$output"; do
  [[ "$value" == /* && "$value" != / ]] || die 'all paths must be explicit absolute paths';
done
finalize_common_ssh
print_wrapper_header 'L2 read-only devices; new evidence directory only'
command="/usr/bin/timeout --signal=TERM --kill-after=3s 60s $(shell_single_quote "$python") -u -"
command+=" --repo $(shell_single_quote "$repo") --camera-config $(shell_single_quote "$cameras")"
command+=" --arm-ip-assignment $(shell_single_quote "$arms") --output $(shell_single_quote "$output")"
printf '[readonly-snapshot] %s\n' "$command"
if [[ "$preview" == true ]]; then exit 0; fi
remote_exec "$command" < "$SCRIPT_DIR/../scripts/probes/capture_rm65_grasp_snapshot.py"
