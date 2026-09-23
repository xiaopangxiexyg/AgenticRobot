#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/probes/run_formal_no_capture_teleop.sh \
  --formal-root /absolute/path --expected-head COMMIT \
  --python /absolute/path/to/python --log-dir /absolute/path [--dry-run]

Run the current formal pure-position teleop implementation without collection,
camera, tactile, Q6 or gripper devices. Only an on-site operator may omit
--dry-run. This wrapper never connects OpenPI actions to robot control.
EOF
}

die() { printf '[teleop preflight] %s\n' "$*" >&2; exit 2; }

formal_root=
expected_head=
python_bin=
log_dir=
dry_run=false
while (($#)); do
  case "$1" in
    --formal-root|--expected-head|--python|--log-dir)
      (($# >= 2)) || die "$1 requires a value"
      case "$1" in
        --formal-root) formal_root=$2 ;;
        --expected-head) expected_head=$2 ;;
        --python) python_bin=$2 ;;
        --log-dir) log_dir=$2 ;;
      esac
      shift 2
      ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$formal_root" == /* && -d "$formal_root" ]] || die 'formal root must be an existing absolute directory'
[[ "$log_dir" == /* ]] || die 'log directory must be absolute'
[[ "$python_bin" == /* && -x "$python_bin" ]] || die 'python must be an executable absolute path'
[[ "$expected_head" =~ ^[0-9a-f]{40}$ ]] || die 'expected HEAD must be a full commit SHA'
[[ -f "$formal_root/docs/real/teleop_args.sh" ]] || die 'formal teleop arguments are missing'
[[ -f "$formal_root/teleop_runtime/debug/run_vive_rm65_real_teleop.py" ]] || die 'formal teleop entrypoint is missing'
actual_head=$(git -C "$formal_root" rev-parse HEAD) || die 'cannot resolve formal HEAD'
[[ "$actual_head" == "$expected_head" ]] || die "formal HEAD differs: expected=$expected_head actual=$actual_head"
git -C "$formal_root" diff --quiet -- \
  docs/real/teleop_args.sh teleop_runtime/real_teleop \
  teleop_runtime/session.py teleop_runtime/position_force_guard.py \
  teleop_runtime/backends/arms/rm65.py robot_motion \
  hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json || \
  die 'formal teleop control sources have uncommitted changes'
git -C "$formal_root" diff --cached --quiet -- \
  docs/real/teleop_args.sh teleop_runtime/real_teleop \
  teleop_runtime/session.py teleop_runtime/position_force_guard.py \
  teleop_runtime/backends/arms/rm65.py robot_motion \
  hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json || \
  die 'formal teleop control sources have staged changes'

cd -- "$formal_root"
# shellcheck source=/dev/null
source docs/real/teleop_args.sh
tool_hashes=$(sha256sum \
  hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json) || \
  die 'cannot hash active tool profiles'
for required in \
  --no-position-execution-qvel-feedback \
  --minimum-control-hz \
  --control-rate-warmup-sec \
  --freeze-on-low-control-rate \
  --arm-canfd-low-rate-fault-ratio \
  --arm-canfd-low-rate-fault-window; do
  found=false
  for arg in "${POSITION_FORMAL_V2_BALANCED_60HZ_JERK_V2_ARGS[@]}"; do
    if [[ "$arg" == "$required" ]]; then found=true; break; fi
  done
  [[ "$found" == true ]] || die "formal profile is missing $required"
done

command=(
  taskset -c 0-7 "$python_bin" -m teleop_runtime.debug.run_vive_rm65_real_teleop
  "${NO_GRIPPER_COMMON_ARGS[@]}"
  "${POSITION_FORMAL_V2_BALANCED_60HZ_JERK_V2_ARGS[@]}"
  "${POSITION_COLLECTION_PRODUCTION_ARGS[@]}"
  --no-arm-ip-save-latest
)
printf '[teleop preflight] formal_head=%s root=%s\n' "$actual_head" "$formal_root"
printf '[teleop preflight] tool_profile_sha256:\n%s\n' "$tool_hashes"
printf '[teleop preflight] command:'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "$dry_run" == true ]]; then
  printf '[teleop preflight] dry-run only; no devices opened and no log created\n'
  exit 0
fi

command -v taskset >/dev/null || die 'taskset is unavailable'
mkdir -p -- "$log_dir"
log_file=$(mktemp "$log_dir/pure_teleop_$(date -u +%Y%m%dT%H%M%SZ).XXXXXX.log")
printf '[teleop preflight] terminal_log=%s\n' "$log_file"
printf '[teleop preflight] verify the physical tool matches both DMTAC_M profiles before Home\n'
{
  printf '[teleop preflight] formal_head=%s root=%s\n' "$actual_head" "$formal_root"
  printf '[teleop preflight] tool_profile_sha256:\n%s\n' "$tool_hashes"
  printf '[teleop preflight] command:'
  printf ' %q' "${command[@]}"
  printf '\n'
} >> "$log_file"
set +e
PYTHONPATH="$formal_root" PYTHONUNBUFFERED=1 "${command[@]}" 2>&1 | tee -a "$log_file"
result=("${PIPESTATUS[@]}")
set -e
printf '[teleop exit] runtime=%s logger=%s log=%s\n' "${result[0]}" "${result[1]}" "$log_file" | tee -a "$log_file"
((result[0] == 0 && result[1] == 0))
