#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/remote_common.sh"
serial="" output="" preview=false
while [[ $# -gt 0 ]]; do
  if parse_common_ssh_arg "$@"; then shift "$COMMON_ARGC"; continue; fi
  case "$1" in
    --usb-serial|--output)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      if [[ "$1" == --usb-serial ]]; then serial="$2"; else output="$2"; fi
      shift 2;;
    --dry-run) preview=true; shift;;
    -h|--help)
      printf '%s\n' 'One D405 raw V4L2 depth/color diagnostic; explicit camera-start authorization required.' \
        'Required: -H HOST --usb-serial USB_SERIAL --output NEW_ABSOLUTE_DIR' \
        'Optional: --dry-run (no SSH).' \
        'Fixed 640x480@15 Z16/YUYV, 30 buffers each, 12s capture limit, no SDK/robot/reset/retry.' \
        'USB serial is not the SDK serial. Resolve the physical identity before invoking.'
      exit 0;;
    *) die "unknown argument: $1";;
  esac
done
[[ "$serial" =~ ^[0-9]{12}$ ]] || die 'a 12-digit USB serial is required'
[[ "$output" == /* && "$output" != / ]] || die 'output must be an explicit absolute new directory'
finalize_common_ssh
print_wrapper_header 'L2 camera-only single-start raw diagnostic; new evidence only'
command="/usr/bin/timeout --signal=TERM --kill-after=5s 30s /usr/bin/python3 -u -"
command+=" --usb-serial $(shell_single_quote "$serial") --output $(shell_single_quote "$output")"
printf '[raw-v4l2-pair] %s\n' "$command"
if [[ "$preview" == true ]]; then exit 0; fi
remote_exec "$command" < "$SCRIPT_DIR/../scripts/probes/capture_d405_v4l2_pair.py"
