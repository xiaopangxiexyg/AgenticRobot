#!/usr/bin/env bash
set -euo pipefail

# Run the complete non-hardware suite in deterministic, target-isolated shards.
# Modules remain the default target; explicitly registered oversized modules
# are split only across statically discovered top-level test functions.
# Mutable pytest/runtime state stays shard-private.  The content-addressed
# MuJoCo model cache is shared only within this exact Run: its implementation
# owns process locking, atomic publication, and hash validation, so identical
# models are compiled once without creating cross-Run state.
# DEXTELEOP_FULL_ACCEPTANCE_RUN_ROOT may name a new absolute artifact directory.
# DEXTELEOP_FULL_ACCEPTANCE_JUNIT may name the final aggregate JUnit report.
# DEXTELEOP_FULL_ACCEPTANCE_MAX_JOBS limits live shards, not planned targets.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
umask 022
export MUJOCO_GL="${MUJOCO_GL:-egl}"

die() {
  echo "[dexteleop-full-acceptance] ERROR: $*" >&2
  exit 2
}

for argument in "$@"; do
  case "$argument" in
    --basetemp|--basetemp=*|-m|--junitxml|--junitxml=*|-o|--override-ini|--override-ini=*)
      die "$argument is owned by this wrapper; pass only shared pytest options such as --durations"
      ;;
    --|.|./|tests/*|./tests/*|*/tests/*|*::* )
      die "explicit test targets are not allowed in the complete acceptance wrapper: $argument"
      ;;
  esac
done

RUN_ROOT=""
JUNIT_DESTINATION=""
ASSET_LINK="$REPO_ROOT/assets"
ASSET_ROOT=""
ASSET_LINK_CREATED=0
SHARD_COUNT=""
MAX_JOBS=""
SHORT_TMP_ROOT=""
SHARD_PIDS=()
SHARD_STATUSES=()
SHARD_LOGS=()
SHARD_XMLS=()

prepare_run_root() {
  local requested_root="${DEXTELEOP_FULL_ACCEPTANCE_RUN_ROOT:-}"
  local parent
  if [[ -z "$requested_root" ]]; then
    RUN_ROOT="$(mktemp -d "/tmp/dt-full-acceptance-${UID:-user}-XXXXXX")"
  else
    case "$requested_root" in
      /*) ;;
      *) die "DEXTELEOP_FULL_ACCEPTANCE_RUN_ROOT must be absolute: $requested_root" ;;
    esac
    if [[ -e "$requested_root" || -L "$requested_root" ]]; then
      die "DEXTELEOP_FULL_ACCEPTANCE_RUN_ROOT must not already exist: $requested_root"
    fi
    parent="$(dirname -- "$requested_root")"
    [[ -d "$parent" ]] || die "run-root parent is not a directory: $parent"
    mkdir -- "$requested_root" || die "cannot create run root: $requested_root"
    RUN_ROOT="$requested_root"
  fi

  JUNIT_DESTINATION="${DEXTELEOP_FULL_ACCEPTANCE_JUNIT:-$RUN_ROOT/full-acceptance.xml}"
  case "$JUNIT_DESTINATION" in
    /*) ;;
    *) die "DEXTELEOP_FULL_ACCEPTANCE_JUNIT must be absolute: $JUNIT_DESTINATION" ;;
  esac
  [[ ! -e "$JUNIT_DESTINATION" && ! -L "$JUNIT_DESTINATION" ]] || die \
    "aggregate JUnit destination already exists: $JUNIT_DESTINATION"
  [[ -d "$(dirname -- "$JUNIT_DESTINATION")" ]] || die \
    "aggregate JUnit parent is not a directory: $(dirname -- "$JUNIT_DESTINATION")"
}

prepare_asset_compat_link() {
  local requested_root="${DEXTELEOP_ASSET_ROOT:-}"
  local existing_root
  if [[ -z "$requested_root" ]]; then
    [[ -e "$ASSET_LINK" ]] || die \
      "assets path is missing; set DEXTELEOP_ASSET_ROOT to an absolute shared asset root"
    return
  fi
  case "$requested_root" in
    /*) ;;
    *) die "DEXTELEOP_ASSET_ROOT must be an absolute directory: $requested_root" ;;
  esac
  [[ -d "$requested_root" ]] || die "DEXTELEOP_ASSET_ROOT is not a directory: $requested_root"
  ASSET_ROOT="$(cd -- "$requested_root" && pwd -P)"
  if [[ -e "$ASSET_LINK" || -L "$ASSET_LINK" ]]; then
    existing_root="$(readlink -f -- "$ASSET_LINK")" || die \
      "cannot resolve existing assets path: $ASSET_LINK"
    [[ "$existing_root" == "$ASSET_ROOT" ]] || die \
      "existing assets path does not match DEXTELEOP_ASSET_ROOT: $existing_root != $ASSET_ROOT"
    return
  fi
  ln -s -- "$ASSET_ROOT" "$ASSET_LINK"
  ASSET_LINK_CREATED=1
}

configure_shards() {
  SHARD_COUNT="${DEXTELEOP_FULL_ACCEPTANCE_SHARDS:-4}"
  case "$SHARD_COUNT" in
    1|2|3|4|5|6) ;;
    *) die "DEXTELEOP_FULL_ACCEPTANCE_SHARDS must be an integer from 1 to 6: $SHARD_COUNT" ;;
  esac
  MAX_JOBS="${DEXTELEOP_FULL_ACCEPTANCE_MAX_JOBS:-$SHARD_COUNT}"
  case "$MAX_JOBS" in
    1|2|3|4|5|6) ;;
    *) die "DEXTELEOP_FULL_ACCEPTANCE_MAX_JOBS must be an integer from 1 to SHARD_COUNT ($SHARD_COUNT): $MAX_JOBS" ;;
  esac
  ((MAX_JOBS <= SHARD_COUNT)) || die \
    "DEXTELEOP_FULL_ACCEPTANCE_MAX_JOBS must not exceed SHARD_COUNT ($SHARD_COUNT): $MAX_JOBS"
}

verify_local_socket_capabilities() {
  python3 "$REPO_ROOT/scripts/pytest_sharding.py" check-sockets || die \
    "complete acceptance requires working loopback TCP and Unix sockets"
}

verify_required_model_input() {
  # Several complete-suite B601 cases load this checkout-relative model.
  # An external source override used by other cases cannot replace this input.
  # Only check availability here; model/physics validation belongs to the tests.
  local model_path="$REPO_ROOT/hardware/reBotArm_control_py/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf"
  [[ -f "$model_path" && -r "$model_path" ]] || die \
    "required B601 URDF is missing or unreadable: $model_path; initialize hardware/reBotArm_control_py at the recorded gitlink before complete acceptance (no shards started)"
}

prepare_shards() {
  local index
  # multiprocessing's forkserver uses TMPDIR, not pytest's short basetemp.
  SHORT_TMP_ROOT="$(mktemp -d "/tmp/dt-full-tmp-${UID:-user}-XXXXXX")"
  mkdir -- "$RUN_ROOT/mujoco-cache"
  python3 "$REPO_ROOT/scripts/pytest_sharding.py" plan \
    --repository "$REPO_ROOT" \
    --tests-root "$REPO_ROOT/tests" \
    --output-dir "$RUN_ROOT" \
    --shards "$SHARD_COUNT" \
    --profile full || die "could not build a complete shard plan"
  for ((index = 0; index < SHARD_COUNT; index++)); do
    SHARD_LOGS[index]="$RUN_ROOT/shard-${index}.log"
    SHARD_XMLS[index]="$RUN_ROOT/shard-${index}.xml"
  done
}

stop_children() {
  local pid
  for pid in "${SHARD_PIDS[@]}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
}

cleanup() {
  local status=$?
  local current_root
  local retain_tmp=0
  trap - EXIT INT TERM
  if ((${#SHARD_PIDS[@]} > 0)); then
    retain_tmp=1
    stop_children
  fi
  if [[ -n "$SHORT_TMP_ROOT" ]]; then
    if ((retain_tmp || status == 130 || status == 143)); then
      echo "[dexteleop-full-acceptance] interrupted temporary state retained: $SHORT_TMP_ROOT"
    else
      case "$SHORT_TMP_ROOT" in
        /tmp/dt-full-tmp-*) rm -rf -- "$SHORT_TMP_ROOT" || status=1 ;;
        *) echo "[dexteleop-full-acceptance] ERROR: unexpected temporary root: $SHORT_TMP_ROOT" >&2; status=1 ;;
      esac
    fi
  fi
  if ((ASSET_LINK_CREATED)); then
    if [[ -L "$ASSET_LINK" ]]; then
      current_root="$(readlink -f -- "$ASSET_LINK")" || current_root=""
      if [[ "$current_root" == "$ASSET_ROOT" ]]; then
        unlink -- "$ASSET_LINK" || status=1
      else
        echo "[dexteleop-full-acceptance] ERROR: refusing to remove changed assets link: $ASSET_LINK" >&2
        status=1
      fi
    else
      echo "[dexteleop-full-acceptance] ERROR: temporary assets link was replaced: $ASSET_LINK" >&2
      status=1
    fi
  fi
  [[ -z "$RUN_ROOT" ]] || echo "[dexteleop-full-acceptance] artifacts: $RUN_ROOT"
  exit "$status"
}

on_signal() {
  stop_children
  local pid
  for pid in "${SHARD_PIDS[@]}"; do
    [[ -n "$pid" ]] || continue
    wait "$pid" 2>/dev/null || true
  done
  SHARD_PIDS=()
  exit 130
}

run_shard() {
  local index="$1"
  shift
  local name="shard-${index}"
  local shard_tmp="$SHORT_TMP_ROOT/$name"
  local -a shard_paths=()
  mkdir -- "$shard_tmp" "$RUN_ROOT/${name}-xdg-cache" "$RUN_ROOT/${name}-matplotlib-cache" \
    "$RUN_ROOT/${name}-numba-cache"
  mapfile -t shard_paths <"$RUN_ROOT/${name}-paths.txt"
  ((${#shard_paths[@]} > 0)) || die "empty shard path list: $name"
  PYTHONNOUSERSITE=1 \
  PYTHONPYCACHEPREFIX="$RUN_ROOT/${name}-python-bytecode" \
  SIM_MUJOCO_CACHE_DIR="$RUN_ROOT/mujoco-cache" \
  XDG_CACHE_HOME="$RUN_ROOT/${name}-xdg-cache" \
  MPLCONFIGDIR="$RUN_ROOT/${name}-matplotlib-cache" \
  NUMBA_CACHE_DIR="$RUN_ROOT/${name}-numba-cache" \
  TMPDIR="$shard_tmp" \
  "$REPO_ROOT/scripts/run_pytest.sh" \
    -m 'not hardware' \
    -q \
    -o "cache_dir=$RUN_ROOT/${name}-pytest-cache" \
    --junitxml="${SHARD_XMLS[index]}" \
    "${shard_paths[@]}" \
    "$@" \
    >"${SHARD_LOGS[index]}" 2>&1
}

wait_shard() {
  local index="$1"
  local shard_status=0
  if wait "${SHARD_PIDS[index]}"; then
    shard_status=0
  else
    shard_status=$?
  fi
  SHARD_STATUSES[index]=$shard_status
  # Never signal a completed child's PID after it could have been reused.
  SHARD_PIDS[index]=""
  ((shard_status == 0)) || OVERALL_STATUS=1
  return 0
}

trap cleanup EXIT
configure_shards
verify_required_model_input
verify_local_socket_capabilities
prepare_run_root
prepare_asset_compat_link
# Wrapper configuration is private to this invocation.  Do not leak it into
# pytest shards, whose tests may launch a nested acceptance wrapper.
unset DEXTELEOP_FULL_ACCEPTANCE_RUN_ROOT
unset DEXTELEOP_FULL_ACCEPTANCE_JUNIT
unset DEXTELEOP_FULL_ACCEPTANCE_SHARDS
unset DEXTELEOP_FULL_ACCEPTANCE_MAX_JOBS
prepare_shards
trap on_signal INT TERM

OVERALL_STATUS=0
for ((index = 0; index < SHARD_COUNT; index++)); do
  if ((index >= MAX_JOBS)); then
    wait_shard "$((index - MAX_JOBS))"
  fi
  run_shard "$index" "$@" &
  SHARD_PIDS[index]=$!
done

for ((index = SHARD_COUNT - MAX_JOBS; index < SHARD_COUNT; index++)); do
  wait_shard "$index"
done
SHARD_PIDS=()

for ((index = 0; index < SHARD_COUNT; index++)); do
  echo "===== full acceptance shard $index (status=${SHARD_STATUSES[index]}) ====="
  if [[ -f "${SHARD_LOGS[index]}" ]]; then
    cat "${SHARD_LOGS[index]}"
  else
    echo "missing shard log: ${SHARD_LOGS[index]}" >&2
    OVERALL_STATUS=1
  fi
done

if ! python3 "$REPO_ROOT/scripts/pytest_sharding.py" merge-junit \
  --output "$JUNIT_DESTINATION" "${SHARD_XMLS[@]}"; then
  OVERALL_STATUS=1
fi
echo "[dexteleop-full-acceptance] aggregate JUnit: $JUNIT_DESTINATION"

if ((OVERALL_STATUS != 0)); then
  exit 1
fi
