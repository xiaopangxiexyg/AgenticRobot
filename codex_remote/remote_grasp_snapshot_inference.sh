#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/remote_common.sh"
python="" args=()
while [[ $# -gt 0 ]]; do
  if parse_common_ssh_arg "$@"; then shift "$COMMON_ARGC"; continue; fi
  case "$1" in
    --python) [[ $# -ge 2 ]] || die 'missing Python'; python="$2"; shift 2;;
    --) shift; args=("$@"); break;;
    *) die 'Usage: -H HOST --python /absolute/python -- fixed offline probe arguments';;
  esac
done
[[ "$python" == /* ]] || die 'Python must be absolute'
finalize_common_ssh
print_wrapper_header 'L2 offline frozen-array inference; no device connections'
command="/usr/bin/timeout --signal=TERM --kill-after=5s 120s $(shell_single_quote "$python") -u -"
for value in "${args[@]}"; do command+=" $(shell_single_quote "$value")"; done
remote_exec "$command" < "$SCRIPT_DIR/../scripts/probes/infer_frozen_grasp.py"
