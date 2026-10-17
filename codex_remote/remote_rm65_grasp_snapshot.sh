#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/remote_common.sh"
repo="" python="" cameras="" arms="" output="" preview=false
check_only=false
camera_check=false
camera_check_mode=streams
context_args=()
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
    --check-only) check_only=true; shift;;
    --camera-startup-check)
      camera_check=true; shift
      if [[ "${1:-}" == streams || "${1:-}" == sensors ]]; then camera_check_mode="$1"; shift; fi;;
    --camera-registry|--camera-mount-urdf|--left-tool-profile|--right-tool-profile)
      [[ $# -ge 2 && "$2" == /* && "$2" != / ]] || die "$1 requires an absolute file path"
      context_args+=("$1" "$2"); shift 2;;
    -h|--help)
      printf '%s\n' 'Read-only RM65 qpos + RealSense RGB-D snapshot. No motion/HOLD/STOP/reset.' \
        'Explicit device-read authorization required for each target.' \
        'Required: -H HOST --repo PATH --python PATH --camera-config PATH --arm-ip-assignment PATH --output NEW_DIR' \
        'Optional: --dry-run (no SSH), --check-only (paths/imports only; no devices),' \
        '          --camera-startup-check [streams|sensors] (no arm connection).' \
        'Context: --camera-registry PATH --camera-mount-urdf PATH --left-tool-profile PATH --right-tool-profile PATH' \
        '60s userspace timeout; kernel D-state cannot be forcibly interrupted.'
      exit 0;;
    *) die "unknown argument: $1";;
  esac
done
[[ "$camera_check" != true || "$check_only" != true ]] || die 'camera startup check and check-only are mutually exclusive'
for value in "$repo" "$python" "$cameras" "$arms" "$output"; do
  [[ "$value" == /* && "$value" != / ]] || die 'all paths must be explicit absolute paths';
done
finalize_common_ssh
print_wrapper_header 'L2 read-only devices; new evidence directory only'
command="/usr/bin/timeout --signal=TERM --kill-after=3s 60s $(shell_single_quote "$python") -u -"
command+=" --repo $(shell_single_quote "$repo") --camera-config $(shell_single_quote "$cameras")"
command+=" --arm-ip-assignment $(shell_single_quote "$arms") --output $(shell_single_quote "$output")"
if [[ "$check_only" == true ]]; then
  command+=" --check-only"
  if (( ${#context_args[@]} )); then command+=" --interactive"; fi
fi
if [[ "$camera_check" == true ]]; then command+=" --camera-startup-check $(shell_single_quote "$camera_check_mode")"; fi
for argument in "${context_args[@]}"; do command+=" $(shell_single_quote "$argument")"; done
printf '[readonly-snapshot] %s\n' "$command"
if [[ "$preview" == true ]]; then exit 0; fi
remote_exec "$command" < "$SCRIPT_DIR/../scripts/probes/capture_rm65_grasp_snapshot.py"
