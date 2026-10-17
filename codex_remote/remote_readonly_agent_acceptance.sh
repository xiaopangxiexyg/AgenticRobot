#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/remote_common.sh"
repo="" python="" rpent_repo="" rpent_python="" cameras="" registry="" bundle="" output=""
hf_home=""
acceptance_scope=perception
arms="" left_tool="" right_tool="" envelope=""
camera_scope=all
preview=false check=false sdk_debug=false
while [[ $# -gt 0 ]]; do
  if parse_common_ssh_arg "$@"; then shift "$COMMON_ARGC"; continue; fi
  case "$1" in
    --repo|--python|--rpent-repo|--rpent-python|--camera-config|--camera-registry|--localization-bundle|--output)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      case "$1" in
        --repo) repo="$2";; --python) python="$2";; --rpent-repo) rpent_repo="$2";; --rpent-python) rpent_python="$2";;
        --camera-config) cameras="$2";; --camera-registry) registry="$2";; --localization-bundle) bundle="$2";; --output) output="$2";;
      esac
      shift 2;;
    --dry-run) preview=true; shift;;
    --camera-sdk-debug-log) sdk_debug=true; shift;;
    --camera-scope)
      [[ $# -ge 2 && ( "$2" == all || "$2" == main ) ]] || die '--camera-scope must be all or main'
      camera_scope="$2"; shift 2;;
    --acceptance-scope)
      [[ $# -ge 2 && ( "$2" == perception || "$2" == tcp ) ]] || die '--acceptance-scope must be perception or tcp'
      acceptance_scope="$2"; shift 2;;
    --arm-ip-assignment|--left-tool-profile|--right-tool-profile|--envelope-profile)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      case "$1" in
        --arm-ip-assignment) arms="$2";; --left-tool-profile) left_tool="$2";;
        --right-tool-profile) right_tool="$2";; --envelope-profile) envelope="$2";;
      esac
      shift 2;;
    --hf-home)
      [[ $# -ge 2 && "$2" == /* && "$2" != / ]] || die '--hf-home requires an absolute cache root'
      hf_home="$2"; shift 2;;
    --check-only) check=true; shift;;
    -h|--help)
      printf '%s\n' 'Fixed read-only Gateway + existing RPent API CLI acceptance. No motion/gripper/HOLD/STOP/reset.' \
        'Requires current camera-read authorization and explicit absolute paths:' \
        '  -H HOST --repo PATH --python PATH --rpent-repo PATH --rpent-python PATH' \
        '  --camera-config PATH --camera-registry PATH --localization-bundle PATH --output NEW_DIR' \
        '  --acceptance-scope tcp selects dual-arm SDK/FK state only, without cameras; instead provide:' \
        '  --arm-ip-assignment PATH --left-tool-profile PATH --right-tool-profile PATH --envelope-profile PATH' \
        'API credential on stdin only. Optional: --dry-run (no SSH), --check-only (no devices/model request).' \
        '  --hf-home PATH binds an existing offline model cache; never downloads.' \
        '  --camera-scope all|main selects the authorized camera inventory (default: all).' \
        '  --camera-sdk-debug-log retains native SDK log and Gateway startup context in the new output directory.' \
        'Fixed 300s userspace limit, TERM then 90s cleanup allowance. No automatic retry.'
      exit 0;;
    *) die "unknown argument: $1";;
  esac
done
path_names=(repo python rpent_repo rpent_python output)
if [[ "$acceptance_scope" == tcp ]]; then
  [[ -z "$cameras" && -z "$registry" && -z "$bundle" && "$sdk_debug" == false && "$camera_scope" == all ]] || \
    die 'tcp scope cannot start or configure cameras'
  path_names+=(arms left_tool right_tool envelope)
else
  [[ -z "$arms" && -z "$left_tool" && -z "$right_tool" && -z "$envelope" ]] || die 'arm paths require tcp scope'
  path_names+=(cameras registry bundle)
fi
for name in "${path_names[@]}"; do
  value="${!name}"
  [[ "$value" == /* && "$value" != / ]] || die 'all paths must be explicit absolute paths'
done
finalize_common_ssh
print_wrapper_header "L2 read-only $acceptance_scope Agent acceptance; explicit current device authorization required"
command="/usr/bin/timeout --signal=TERM --kill-after=90s 300s $(shell_single_quote "$python") -u"
command+=" $(shell_single_quote "$repo/scripts/probes/run_readonly_agent_acceptance.py")"
for pair in "${path_names[@]}"; do
  case "$pair" in
    rpent_repo) flag=rpent-repo;; rpent_python) flag=rpent-python;; cameras) flag=camera-config;;
    registry) flag=camera-registry;; bundle) flag=localization-bundle;; *) flag="$pair";;
  esac
  case "$pair" in
    arms) flag=arm-ip-assignment;; left_tool) flag=left-tool-profile;;
    right_tool) flag=right-tool-profile;; envelope) flag=envelope-profile;;
  esac
  command+=" --$flag $(shell_single_quote "${!pair}")"
done
if [[ "$check" == true ]]; then command+=" --check-only"; fi
if [[ "$sdk_debug" == true ]]; then command+=" --camera-sdk-debug-log"; fi
if [[ -n "$hf_home" ]]; then command+=" --hf-home $(shell_single_quote "$hf_home")"; fi
command+=" --camera-scope $(shell_single_quote "$camera_scope")"
command+=" --acceptance-scope $(shell_single_quote "$acceptance_scope")"
printf '[readonly-agent] %s\n' "$command"
if [[ "$preview" == true ]]; then exit 0; fi
remote_exec "$command"
