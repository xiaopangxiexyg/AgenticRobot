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

CHANGE_BASE=""
ACCEPTANCE_PROFILE="full"
CORE_HEAD=""
PYTEST_ARGUMENTS=()
while (($#)); do
  case "$1" in
    --core) ACCEPTANCE_PROFILE="core" ;;
    --changed-since|--changed-since=*)
      [[ -z "$CHANGE_BASE" ]] || die "--changed-since may be supplied only once"
      if [[ "$1" == --changed-since ]]; then
        (($# >= 2)) || die "--changed-since requires an ancestor commit"
        shift
        CHANGE_BASE="$1"
      else
        CHANGE_BASE="${1#--changed-since=}"
      fi
      [[ -n "$CHANGE_BASE" ]] || die "--changed-since requires an ancestor commit"
      ;;
    *) PYTEST_ARGUMENTS+=("$1") ;;
  esac
  shift
done
set -- "${PYTEST_ARGUMENTS[@]}"
if [[ "$ACCEPTANCE_PROFILE" == core ]]; then
  [[ -z "$CHANGE_BASE" ]] || die "core and change-aware full acceptance are separate scopes"
  [[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all --ignore-submodules=none)" ]] || die "core acceptance requires a clean committed source"
  CORE_HEAD="$(git -C "$REPO_ROOT" rev-parse HEAD)"
fi

[[ -z "${PYTEST_ADDOPTS:-}" ]] || die "acceptance does not allow ambient PYTEST_ADDOPTS"
[[ -z "${PYTEST_PLUGINS:-}" ]] || die "acceptance does not allow ambient PYTEST_PLUGINS"
[[ -z "${PYTHONOPTIMIZE:-}" || "${PYTHONOPTIMIZE}" == 0 ]] || die "acceptance does not allow PYTHONOPTIMIZE"
option_value=""
for argument in "$@"; do
  if [[ -n "$option_value" ]]; then
    [[ "$argument" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid $option_value value: $argument"
    [[ "$option_value" != --durations || "$argument" =~ ^[0-9]+$ ]] || die "invalid --durations value"
    option_value=""
    continue
  fi
  case "$argument" in
    --durations|--durations-min) option_value="$argument" ;;
    --durations=*) [[ "${argument#*=}" =~ ^[0-9]+$ ]] || die "invalid --durations value" ;;
    --durations-min=*) [[ "${argument#*=}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid --durations-min value" ;;
    -q|-v|-vv) ;;
    *) die "acceptance permits only verbosity and duration reporting options: $argument" ;;
  esac
done
[[ -z "$option_value" ]] || die "missing $option_value value"

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
CLEANUP_COMPLETED=0

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

  local report_name="full-acceptance.xml"
  [[ "$ACCEPTANCE_PROFILE" != core ]] || report_name="merge-core-acceptance.xml"
  [[ -z "$CHANGE_BASE" ]] || report_name="changed-acceptance.xml"
  JUNIT_DESTINATION="${DEXTELEOP_FULL_ACCEPTANCE_JUNIT:-$RUN_ROOT/$report_name}"
  if [[ -n "$CHANGE_BASE" && "${JUNIT_DESTINATION##*/}" == full-acceptance.xml ]]; then
    die "change-aware evidence must not be named full-acceptance.xml"
  fi
  if [[ "$ACCEPTANCE_PROFILE" == core && "${JUNIT_DESTINATION##*/}" == full-acceptance.xml ]]; then
    die "core evidence must not be named full-acceptance.xml"
  fi
  case "$JUNIT_DESTINATION" in
    /*) ;;
    *) die "DEXTELEOP_FULL_ACCEPTANCE_JUNIT must be absolute: $JUNIT_DESTINATION" ;;
  esac
  [[ "$(realpath -m -- "$JUNIT_DESTINATION")" != "$(realpath -m -- "$RUN_ROOT/diagnostic-acceptance.xml")" ]] || die \
    "aggregate JUnit destination must differ from the retained diagnostic report"
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
  local default_shards=4
  [[ "$ACCEPTANCE_PROFILE" != core ]] || default_shards=4
  SHARD_COUNT="${DEXTELEOP_FULL_ACCEPTANCE_SHARDS:-$default_shards}"
  case "$SHARD_COUNT" in
    1|2|3|4|5|6|7|8|9|10|11|12) ;;
    *) die "DEXTELEOP_FULL_ACCEPTANCE_SHARDS must be an integer from 1 to 12: $SHARD_COUNT" ;;
  esac
  # More fresh-process batches must not implicitly increase live concurrency.
  if ((SHARD_COUNT > 6)) && [[ -z "${DEXTELEOP_FULL_ACCEPTANCE_MAX_JOBS:-}" ]]; then
    die "DEXTELEOP_FULL_ACCEPTANCE_MAX_JOBS must be explicit when SHARD_COUNT exceeds 6"
  fi
  local default_jobs="$SHARD_COUNT"
  [[ "$ACCEPTANCE_PROFILE" != core ]] || default_jobs=2
  MAX_JOBS="${DEXTELEOP_FULL_ACCEPTANCE_MAX_JOBS:-$default_jobs}"
  case "$MAX_JOBS" in
    1|2|3|4|5|6) ;;
    *) die "DEXTELEOP_FULL_ACCEPTANCE_MAX_JOBS must be an integer from 1 to 6: $MAX_JOBS" ;;
  esac
  if [[ "$ACCEPTANCE_PROFILE" == core ]]; then
    # Native owners occupy even batches, and batch N starts only after batch
    # N-MAX_JOBS finishes, so four live processes hold at most two native
    # batches. Above the two-job default those batches overlap; select it only
    # under the larger memory guard documented in tests/README.md.
    ((MAX_JOBS <= 4)) || die "core acceptance uses at most four live pytest processes"
  fi
  ((MAX_JOBS <= SHARD_COUNT)) || die \
    "DEXTELEOP_FULL_ACCEPTANCE_MAX_JOBS must not exceed SHARD_COUNT ($SHARD_COUNT): $MAX_JOBS"
}

verify_local_socket_capabilities() {
  python3 "$REPO_ROOT/scripts/pytest_sharding.py" check-sockets || die \
    "complete acceptance requires working loopback TCP and Unix sockets"
}

verify_browser_contract_runtime() {
  # Absence remains optional, but an installed incompatible Node fails the suite.
  command -v node >/dev/null 2>&1 || return 0
  node -e 'const assert = require("node:assert/strict"); const value = {ready: true}; assert.equal(value?.ready ?? false, true);' \
    || die "Node.js on PATH cannot run the annotation browser contracts; select a compatible runtime before complete acceptance (no shards started)"
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
  if [[ -n "$CHANGE_BASE" ]]; then
    python3 "$REPO_ROOT/scripts/pytest_impact.py" plan \
      --repository "$REPO_ROOT" --base "$CHANGE_BASE" \
      --output-dir "$RUN_ROOT" --shards "$SHARD_COUNT" || die "could not establish change-aware acceptance"
    python3 "$REPO_ROOT/scripts/pytest_impact.py" verify \
      --repository "$REPO_ROOT" --selection "$RUN_ROOT/selection.json" --shards "$SHARD_COUNT" || die "change-aware source or plan changed"
    echo "[dexteleop-full-acceptance] change-aware run; see selection.json for omissions or full fallback"
  else
    python3 "$REPO_ROOT/scripts/pytest_sharding.py" plan \
      --repository "$REPO_ROOT" \
      --tests-root "$REPO_ROOT/tests" \
      --output-dir "$RUN_ROOT" \
      --shards "$SHARD_COUNT" \
      --profile "$ACCEPTANCE_PROFILE" || die "could not build an acceptance shard plan"
  fi
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

cleanup_resources() {
  local original_status="$1"
  local status=0
  local current_root
  local retain_tmp=0
  ((CLEANUP_COMPLETED == 0)) || return 0
  CLEANUP_COMPLETED=1
  if ((${#SHARD_PIDS[@]} > 0)); then
    retain_tmp=1
    stop_children
  fi
  if [[ -n "$SHORT_TMP_ROOT" ]]; then
    if ((retain_tmp || original_status == 130 || original_status == 143)); then
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
  return "$status"
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  cleanup_resources "$status" || status=1
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
  local -a core_options=()
  [[ "$ACCEPTANCE_PROFILE" != core ]] || core_options=(-p scripts.pytest_core)
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
    -p scripts.pytest_native_memory \
    -m 'not hardware' \
    -q \
    -o "cache_dir=$RUN_ROOT/${name}-pytest-cache" \
    --junitxml="${SHARD_XMLS[index]}" \
    "${shard_paths[@]}" \
    "${core_options[@]}" \
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
verify_browser_contract_runtime
prepare_run_root
python3 "$REPO_ROOT/scripts/pytest_sharding.py" check-junit-publication \
  --directory "$(dirname -- "$JUNIT_DESTINATION")" || die \
  "atomic no-replace JUnit publication is unavailable; no shards started"
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
WRAPPER_ERRORS=()
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

if [[ -n "$CHANGE_BASE" ]]; then
  python3 "$REPO_ROOT/scripts/pytest_impact.py" verify \
    --repository "$REPO_ROOT" --selection "$RUN_ROOT/selection.json" --shards "$SHARD_COUNT" || die "source or plan changed during change-aware acceptance; no aggregate evidence published"
fi

for ((index = 0; index < SHARD_COUNT; index++)); do
  echo "===== full acceptance shard $index (status=${SHARD_STATUSES[index]}) ====="
  if [[ -f "${SHARD_LOGS[index]}" ]]; then
    cat "${SHARD_LOGS[index]}"
  else
    echo "missing shard log: ${SHARD_LOGS[index]}" >&2
    OVERALL_STATUS=1
    WRAPPER_ERRORS+=("shard $index log is missing")
  fi
done

if ! cleanup_resources 0; then
  OVERALL_STATUS=1
  WRAPPER_ERRORS+=("temporary resource cleanup failed")
fi
DIAGNOSTIC_JUNIT="$RUN_ROOT/diagnostic-acceptance.xml"
MERGE_ARGUMENTS=()
if [[ "$ACCEPTANCE_PROFILE" == core ]]; then
  [[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$CORE_HEAD" && -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all --ignore-submodules=none)" ]] || die "core acceptance source changed during execution"
  MERGE_ARGUMENTS+=(--name merge-core-acceptance --head "$CORE_HEAD")
fi
if ! python3 "$REPO_ROOT/scripts/pytest_sharding.py" merge-junit \
  --output "$DIAGNOSTIC_JUNIT" "${MERGE_ARGUMENTS[@]}" "${SHARD_XMLS[@]}"; then
  die "cannot merge shard JUnit; individual shard evidence is retained"
fi

if [[ -n "$CHANGE_BASE" ]]; then
  python3 "$REPO_ROOT/scripts/pytest_impact.py" verify \
    --repository "$REPO_ROOT" --selection "$RUN_ROOT/selection.json" --shards "$SHARD_COUNT" \
    --junit "$DIAGNOSTIC_JUNIT" || die "change-aware evidence is invalid; diagnostic JUnit retained"
fi

PUBLISH_ARGUMENTS=()
for ((index = 0; index < SHARD_COUNT; index++)); do
  PUBLISH_ARGUMENTS+=(--shard-status "${SHARD_STATUSES[index]}")
done
for error in "${WRAPPER_ERRORS[@]}"; do
  PUBLISH_ARGUMENTS+=(--wrapper-error "$error")
done
if ! python3 "$REPO_ROOT/scripts/pytest_sharding.py" publish-junit \
  --input "$DIAGNOSTIC_JUNIT" --output "$JUNIT_DESTINATION" \
  "${PUBLISH_ARGUMENTS[@]}" "${SHARD_XMLS[@]}"; then
  OVERALL_STATUS=1
fi
echo "[dexteleop-full-acceptance] diagnostic JUnit: $DIAGNOSTIC_JUNIT"
[[ ! -f "$JUNIT_DESTINATION" ]] || echo "[dexteleop-full-acceptance] aggregate JUnit: $JUNIT_DESTINATION"

if ((OVERALL_STATUS != 0)); then
  exit 1
fi
