#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/remote_common.sh"
port="" serial="" bind=false preview=false
while [[ $# -gt 0 ]]; do
  if parse_common_ssh_arg "$@"; then shift "$COMMON_ARGC"; continue; fi
  case "$1" in
    --usb-port|--usb-serial)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      if [[ "$1" == --usb-port ]]; then port="$2"; else serial="$2"; fi
      shift 2;;
    --bind) bind=true; shift;;
    --dry-run) preview=true; shift;;
    -h|--help)
      printf '%s\n' 'Inspect one D405 USB identity, or restore only its unbound uvcvideo control interface.' \
        'Required: -H HOST --usb-port BUS-PORT[.PORT...] --usb-serial USB_SERIAL' \
        'Optional: --bind (explicit device-binding authorization required; sudo -n only), --dry-run.' \
        'No unbind, USB reset, module reload, SDK import, stream startup or robot access. Fixed 15s limit.'
      exit 0;;
    *) die "unknown argument: $1";;
  esac
done
[[ "$port" =~ ^[1-9][0-9]*-[1-9][0-9]*(\.[1-9][0-9]*)*$ ]] || die 'invalid USB port'
[[ "$serial" =~ ^[0-9]{12}$ ]] || die 'a 12-digit USB serial is required'
finalize_common_ssh
print_wrapper_header 'L2 inspection; explicit --bind is scoped device-state restoration'
command="/usr/bin/timeout --signal=TERM --kill-after=3s 15s"
if [[ "$bind" == true ]]; then command+=" /usr/bin/sudo -n"; fi
command+=" /usr/bin/python3 -I -u - --usb-port $(shell_single_quote "$port") --usb-serial $(shell_single_quote "$serial")"
if [[ "$bind" == true ]]; then command+=" --bind"; fi
printf '[camera-driver-bind] %s\n' "$command"
if [[ "$preview" == true ]]; then exit 0; fi
remote_exec "$command" < "$SCRIPT_DIR/../scripts/probes/restore_d405_driver_binding.py"
