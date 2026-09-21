#!/bin/bash
set -euo pipefail

# Conda activation on collection hosts can leave a PATH containing only the
# environment's bin directory.  Preserve that interpreter precedence while
# restoring the standard system command locations required by the launcher.
# The script must not depend on the caller having started bash or zsh as a
# login shell.
SYSTEM_COMMAND_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PATH="${PATH:+$PATH:}$SYSTEM_COMMAND_PATH"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_rule_based_writing_collection.sh \
    --source-pool-manifest /absolute/source-pool.json [options]

Required:
  --source-pool-manifest P    Cross-dataset source candidate pool; or
  --source-dataset PATH       legacy single-source mode, together with
  --source-episode N          source episode index.
  --detail-annotation-dir P   Exact immutable annotation-set directory; required
                              only in rule-based-writing legacy mode.
  --tactile-bundle-root PATH  Existing external tactile output root; required
                              only when effective tactile capture is enabled.

Options:
  --mode NAME                 rule-based-writing (default) or full-replay.
  --dataset-root PATH         Formal output dataset root.
  --trajectory ID             Writing mode only; one-stroke catalog ID
                              (default: digit_1_v1).
  --trajectory-catalog PATH   Writing mode only; absolute JSON catalog appended
                              to the built-in trajectories.
  --size-m M                  Writing mode only; glyph height in metres
                              (default: 0.08).
  --direction NAME            Writing mode only; up, right, down, or left
                              (default: up).
  --speed-m-s MPS             Writing mode only; override source average speed.
  --initial-alignment-speed-scale S
                              Initial dual-arm alignment speed multiplier in (0, 1].
  --collect-tactile            Request tactile capture for a new/blank dataset.
  --no-collect-tactile         Request no tactile capture for a new/blank dataset.
                              A populated target always follows its existing
                              dataset modality contract. Without a tactile root,
                              the launcher defaults to this mode.
  --source-pool-range A:B     Half-open candidate range (default: whole pool).
  --selection-seed N          Deterministic source selection seed (default: 0).
  --pre-pen-down-replay-candidate-id ID
                              Writing mode only; optional complete source Episode
                              replayed without capture before pen-down.
  --pre-pen-down-replay-normal-episodes N
                              Successful normal Episodes before one configured
                              pre-pen-down replay (default: 0, replay every Episode).
  --task-name NAME            Output collection task name.
  --tactile-min-free-gb N     Required tactile free space (default: 10).
  --camera-frame-max-age-sec S
                              Maximum admitted RGB frame age (default: 0.25).
  --with-left-gripper         Also connect/capture the saved left DMTAC stack.
  --left-gripper-port PATH    Override left DMTAC serial port (optional).
  --right-gripper-port PATH   Override right DMTAC serial port (optional).
  --dry-run                   Print the command; do not start any device.
  -h, --help                  Show this help.

This launcher uses the accepted pure-position profile. Active admittance is
intentionally excluded by the real rule-based deployment preflight.
EOF
}

die() {
  printf '[RuleBased] %s\n' "$*" >&2
  exit 1
}

is_positive_decimal() {
  local value="$1"
  [[ "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || return 1
  [[ ! "$value" =~ ^0*([.]0*)?$ ]]
}

is_unit_interval_decimal() {
  local value="$1"
  is_positive_decimal "$value" || return 1
  [[ "$value" =~ ^0*1([.]0*)?$ || "$value" =~ ^0*[.][0-9]*[1-9][0-9]*$ ]]
}

print_command() {
  printf '[dry-run]'
  printf ' %q' "$@"
  printf '\n'
}

SCRIPT_SOURCE="${BASH_SOURCE[0]}"
SCRIPT_PARENT="${SCRIPT_SOURCE%/*}"
if [[ "$SCRIPT_PARENT" == "$SCRIPT_SOURCE" ]]; then
  SCRIPT_PARENT="."
fi
SCRIPT_DIR="$(cd -- "$SCRIPT_PARENT" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
TCP_POSE_ENVELOPE_PROFILE="teleop_runtime/configs/real_teleop/rule_based_writing_bimanual_workspace_v1.json"
SOURCE_DATASET=""
SOURCE_EPISODE=""
DETAIL_ANNOTATION_DIR=""
SOURCE_POOL_MANIFEST=""
SOURCE_POOL_RANGE=""
SELECTION_SEED="0"
PRE_PEN_DOWN_REPLAY_CANDIDATE_ID=""
PRE_PEN_DOWN_REPLAY_NORMAL_EPISODES="0"
COLLECTION_MODE="rule-based-writing"
TACTILE_ROOT="${TACTILE_BUNDLE_ROOT:-}"
OUTPUT_DATASET_ROOT="${DATASET_BASE:-}"
TRAJECTORY="digit_1_v1"
TRAJECTORY_CATALOG=""
SIZE_M="0.08"
DIRECTION="up"
SPEED_M_S=""
INITIAL_ALIGNMENT_SPEED_SCALE="1.0"
LEFT_GRIPPER_PORT=""
RIGHT_GRIPPER_PORT=""
TASK_NAME="rule_based_writing"
TACTILE_MIN_FREE_GB="${TACTILE_BUNDLE_MIN_FREE_GB:-10}"
TACTILE_MIN_FREE_EXPLICIT=0
COLLECT_TACTILE=""
CAMERA_FRAME_MAX_AGE_SEC="0.25"
WITH_LEFT_GRIPPER=0
DRY_RUN=0

# Keep this shell layer limited to site binding and profile assembly.  The
# option-to-variable table avoids duplicating the same arity check and
# assignment branch for every value option while preserving the CLI contract.
declare -A VALUE_OPTION_TARGET=(
  [--mode]=COLLECTION_MODE
  [--source-dataset]=SOURCE_DATASET
  [--source-episode]=SOURCE_EPISODE
  [--detail-annotation-dir]=DETAIL_ANNOTATION_DIR
  [--source-pool-manifest]=SOURCE_POOL_MANIFEST
  [--source-pool-range]=SOURCE_POOL_RANGE
  [--selection-seed]=SELECTION_SEED
  [--pre-pen-down-replay-candidate-id]=PRE_PEN_DOWN_REPLAY_CANDIDATE_ID
  [--pre-pen-down-replay-normal-episodes]=PRE_PEN_DOWN_REPLAY_NORMAL_EPISODES
  [--tactile-bundle-root]=TACTILE_ROOT
  [--dataset-root]=OUTPUT_DATASET_ROOT
  [--trajectory]=TRAJECTORY
  [--trajectory-catalog]=TRAJECTORY_CATALOG
  [--size-m]=SIZE_M
  [--direction]=DIRECTION
  [--speed-m-s]=SPEED_M_S
  [--initial-alignment-speed-scale]=INITIAL_ALIGNMENT_SPEED_SCALE
  [--left-gripper-port]=LEFT_GRIPPER_PORT
  [--right-gripper-port]=RIGHT_GRIPPER_PORT
  [--task-name]=TASK_NAME
  [--tactile-min-free-gb]=TACTILE_MIN_FREE_GB
  [--camera-frame-max-age-sec]=CAMERA_FRAME_MAX_AGE_SEC
)
declare -A VALUE_OPTION_HINT=(
  [--mode]=NAME
  [--source-dataset]=PATH
  [--source-episode]=N
  [--detail-annotation-dir]=PATH
  [--source-pool-manifest]=PATH
  [--source-pool-range]=START:END
  [--selection-seed]=N
  [--pre-pen-down-replay-candidate-id]=ID
  [--pre-pen-down-replay-normal-episodes]=N
  [--tactile-bundle-root]=PATH
  [--dataset-root]=PATH
  [--trajectory]=ID
  [--trajectory-catalog]=PATH
  [--size-m]=M
  [--direction]=NAME
  [--speed-m-s]=MPS
  [--initial-alignment-speed-scale]=S
  [--left-gripper-port]=PATH
  [--right-gripper-port]=PATH
  [--task-name]=NAME
  [--tactile-min-free-gb]=N
  [--camera-frame-max-age-sec]=S
)

while [[ $# -gt 0 ]]; do
  option="$1"
  if [[ -n "${VALUE_OPTION_TARGET[$option]+present}" ]]; then
    [[ $# -ge 2 ]] || die "$option requires ${VALUE_OPTION_HINT[$option]}."
    if [[ "$option" == "--tactile-min-free-gb" ]]; then
      TACTILE_MIN_FREE_EXPLICIT=1
    fi
    printf -v "${VALUE_OPTION_TARGET[$option]}" '%s' "$2"
    shift 2
    continue
  fi
  case "$1" in
    --collect-tactile)
      [[ -z "$COLLECT_TACTILE" ]] || die \
        "--collect-tactile and --no-collect-tactile are mutually exclusive."
      COLLECT_TACTILE=1
      shift
      ;;
    --no-collect-tactile)
      [[ -z "$COLLECT_TACTILE" ]] || die \
        "--collect-tactile and --no-collect-tactile are mutually exclusive."
      COLLECT_TACTILE=0
      shift
      ;;
    --with-left-gripper)
      WITH_LEFT_GRIPPER=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

case "$COLLECTION_MODE" in
  rule-based-writing)
    RULE_COLLECTION_MODE="rule_based_writing"
    ;;
  full-replay)
    RULE_COLLECTION_MODE="full_trajectory_replay"
    ;;
  *) die "mode must be rule-based-writing or full-replay." ;;
esac
if [[ "$COLLECTION_MODE" == "full-replay" && -n "$SPEED_M_S" ]]; then
  die "--speed-m-s is not valid in full-replay mode; source Episode timestamps are preserved."
fi
if [[ "$COLLECTION_MODE" == "full-replay" && -n "$TRAJECTORY_CATALOG" ]]; then
  die "--trajectory-catalog is not valid in full-replay mode."
fi
if [[ "$COLLECTION_MODE" == "full-replay" && -n "$PRE_PEN_DOWN_REPLAY_CANDIDATE_ID" ]]; then
  die "--pre-pen-down-replay-candidate-id is not valid in full-replay mode."
fi
if [[ "$COLLECTION_MODE" == "full-replay" && "$PRE_PEN_DOWN_REPLAY_NORMAL_EPISODES" != "0" ]]; then
  die "--pre-pen-down-replay-normal-episodes is not valid in full-replay mode."
fi

if [[ -n "$TACTILE_ROOT" ]]; then
  [[ "$TACTILE_ROOT" == /* ]] || die "tactile root must be absolute."
  [[ -d "$TACTILE_ROOT" ]] || die "tactile root does not exist: $TACTILE_ROOT"
  [[ ! -L "$TACTILE_ROOT" ]] || die "tactile root must not be a symlink."
fi
if [[ "$COLLECT_TACTILE" == "1" && -z "$TACTILE_ROOT" ]]; then
  die "--collect-tactile requires --tactile-bundle-root."
fi
if [[ -z "$TACTILE_ROOT" && "$TACTILE_MIN_FREE_EXPLICIT" -eq 1 && "$TACTILE_MIN_FREE_GB" != "0" ]]; then
  if [[ "$COLLECT_TACTILE" == "0" ]]; then
    printf '[RuleBased] --tactile-min-free-gb is ignored because --no-collect-tactile is active.\n' >&2
    TACTILE_MIN_FREE_GB="0"
  else
    die "--tactile-min-free-gb requires --tactile-bundle-root."
  fi
fi
if [[ -z "$TACTILE_ROOT" && "$TACTILE_MIN_FREE_EXPLICIT" -eq 0 ]]; then
  TACTILE_MIN_FREE_GB="0"
fi
[[ "$SELECTION_SEED" =~ ^-?[0-9]+$ ]] || die "selection seed must be an integer."
[[ "$PRE_PEN_DOWN_REPLAY_NORMAL_EPISODES" =~ ^[0-9]+$ ]] || \
  die "pre-pen-down replay normal Episodes must be a non-negative integer."
if ! is_unit_interval_decimal "$INITIAL_ALIGNMENT_SPEED_SCALE"; then
  die "initial alignment speed scale must be in (0, 1]."
fi
if ! is_positive_decimal "$CAMERA_FRAME_MAX_AGE_SEC"; then
  die "camera frame max age must be greater than zero."
fi
if [[ -n "$LEFT_GRIPPER_PORT" && "$WITH_LEFT_GRIPPER" -ne 1 ]]; then
  die "--left-gripper-port requires --with-left-gripper."
fi
if [[ -n "$SOURCE_POOL_MANIFEST" ]]; then
  [[ -z "$SOURCE_DATASET" && -z "$SOURCE_EPISODE" && -z "$DETAIL_ANNOTATION_DIR" ]] || \
    die "source pool mode cannot be mixed with legacy single-source options."
  [[ "$SOURCE_POOL_MANIFEST" == /* ]] || die "source pool manifest must be absolute."
  [[ -f "$SOURCE_POOL_MANIFEST" ]] || die \
    "source pool manifest does not exist: $SOURCE_POOL_MANIFEST"
else
  [[ -z "$SOURCE_POOL_RANGE" ]] || die \
    "--source-pool-range requires --source-pool-manifest."
  [[ -z "$PRE_PEN_DOWN_REPLAY_CANDIDATE_ID" ]] || die \
    "--pre-pen-down-replay-candidate-id requires --source-pool-manifest."
  [[ -n "$SOURCE_DATASET" ]] || die \
    "provide --source-pool-manifest or --source-dataset."
  [[ -n "$SOURCE_EPISODE" ]] || die "--source-episode is required in legacy mode."
  [[ "$SOURCE_DATASET" == /* ]] || die "source dataset must be absolute."
  [[ -d "$SOURCE_DATASET" ]] || die "source dataset does not exist: $SOURCE_DATASET"
  [[ "$SOURCE_EPISODE" =~ ^[0-9]+$ ]] || \
    die "source episode must be a non-negative integer."
  if [[ "$COLLECTION_MODE" == "rule-based-writing" ]]; then
    [[ -n "$DETAIL_ANNOTATION_DIR" ]] || die \
      "--detail-annotation-dir is required in rule-based-writing legacy mode."
  fi
  if [[ -n "$DETAIL_ANNOTATION_DIR" ]]; then
    [[ "$DETAIL_ANNOTATION_DIR" == /* ]] || die "detail annotation dir must be absolute."
    [[ -d "$DETAIL_ANNOTATION_DIR" ]] || die \
      "detail annotation dir does not exist: $DETAIL_ANNOTATION_DIR"
  fi
fi
case "$DIRECTION" in
  up|right|down|left) ;;
  *) die "direction must be up, right, down, or left." ;;
esac
if [[ -n "$TRAJECTORY_CATALOG" ]]; then
  [[ "$TRAJECTORY_CATALOG" == /* ]] || die "trajectory catalog must be absolute."
  [[ -f "$TRAJECTORY_CATALOG" ]] || die \
    "trajectory catalog does not exist: $TRAJECTORY_CATALOG"
  [[ ! -L "$TRAJECTORY_CATALOG" ]] || die "trajectory catalog must not be a symlink."
fi

if [[ -n "$OUTPUT_DATASET_ROOT" ]]; then
  [[ "$OUTPUT_DATASET_ROOT" == /* ]] || die "dataset root must be absolute."
  export DATASET_BASE="$OUTPUT_DATASET_ROOT"
fi
export TACTILE_BUNDLE_ROOT="$TACTILE_ROOT"
export TACTILE_BUNDLE_MIN_FREE_GB="$TACTILE_MIN_FREE_GB"

cd -- "$PROJECT_ROOT"
[[ -f "$TCP_POSE_ENVELOPE_PROFILE" ]] || die \
  "TCP pose envelope profile does not exist: $TCP_POSE_ENVELOPE_PROFILE"
# shellcheck source=../docs/real/teleop_args.sh
source docs/real/teleop_args.sh

# Rule-based writing owns the arm command override and intentionally has no
# human gripper input.  Keep the shared formal profile unchanged while removing
# its opt-in Q6 gesture from this specialized launcher.
RULE_COMMON_REAL_TELEOP_ARGS=()
for arg in "${COMMON_REAL_TELEOP_ARGS[@]}"; do
  if [[ "$arg" != "--q6-hold-recalibration-gesture" ]]; then
    RULE_COMMON_REAL_TELEOP_ARGS+=("$arg")
  fi
done

RULE_ARGS=(
  --rule-collection-mode "$RULE_COLLECTION_MODE"
  --rule-initial-alignment-speed-scale "$INITIAL_ALIGNMENT_SPEED_SCALE"
)
if [[ "$COLLECT_TACTILE" == "1" ]]; then
  RULE_ARGS+=(--rule-collect-tactile --dmtac-with-tactile)
elif [[ "$COLLECT_TACTILE" == "0" || -z "$TACTILE_ROOT" ]]; then
  RULE_ARGS+=(--no-rule-collect-tactile --dmtac-no-tactile)
  RULE_ARGS+=(--ignore-tactile left --ignore-tactile right)
fi
if [[ "$COLLECTION_MODE" == "rule-based-writing" ]]; then
  RULE_ARGS+=(
    --rule-writing-trajectory "$TRAJECTORY"
    --rule-writing-size-m "$SIZE_M"
    --rule-writing-direction "$DIRECTION"
    --rule-pre-pen-down-replay-normal-episodes
    "$PRE_PEN_DOWN_REPLAY_NORMAL_EPISODES"
  )
  if [[ -n "$TRAJECTORY_CATALOG" ]]; then
    RULE_ARGS+=(--rule-writing-trajectory-catalog "$TRAJECTORY_CATALOG")
  fi
fi
if [[ -n "$SOURCE_POOL_MANIFEST" ]]; then
  RULE_ARGS+=(
    --rule-source-pool-manifest "$SOURCE_POOL_MANIFEST"
    --rule-source-selection-seed "$SELECTION_SEED"
  )
  if [[ -n "$SOURCE_POOL_RANGE" ]]; then
    RULE_ARGS+=(--rule-source-pool-range "$SOURCE_POOL_RANGE")
  fi
  if [[ -n "$PRE_PEN_DOWN_REPLAY_CANDIDATE_ID" ]]; then
    RULE_ARGS+=(
      --rule-pre-pen-down-replay-candidate-id
      "$PRE_PEN_DOWN_REPLAY_CANDIDATE_ID"
    )
  fi
else
  RULE_ARGS+=(
    --rule-source-dataset "$SOURCE_DATASET"
    --rule-source-episode "$SOURCE_EPISODE"
  )
  if [[ -n "$DETAIL_ANNOTATION_DIR" ]]; then
    RULE_ARGS+=(--rule-detail-annotation-dir "$DETAIL_ANNOTATION_DIR")
  fi
fi
if [[ -n "$SPEED_M_S" && "$COLLECTION_MODE" == "rule-based-writing" ]]; then
  RULE_ARGS+=(--rule-writing-speed-m-s "$SPEED_M_S")
fi

TACTILE_RUNTIME_ARGS=()
if [[ -n "$TACTILE_ROOT" && "$COLLECT_TACTILE" != "0" ]]; then
  TACTILE_RUNTIME_ARGS=(
    "${DMTAC_CAPTURE_COMMON_ARGS[@]}"
    "${DMTAC_A_ARGS[@]}"
    "${DMTAC_TACTILE_WORKER_2CPU_ARGS[@]}"
  )
fi

LEFT_HAND_BACKEND=none
if [[ "$WITH_LEFT_GRIPPER" -eq 1 ]]; then
  LEFT_HAND_BACKEND=dmtac-gripper
fi
HAND_TOPOLOGY_ARGS=(
  --left-hand-backend "$LEFT_HAND_BACKEND"
  --right-hand-backend dmtac-gripper
  --gripper-input-source none
  --no-q6-default-config
  --special-arm-control-mode manual
)

if [[ -n "${LEFT_GRIPPER_PORT:-}" ]]; then
  HAND_TOPOLOGY_ARGS+=(--left-gripper-port "$LEFT_GRIPPER_PORT")
fi
if [[ -n "${RIGHT_GRIPPER_PORT:-}" ]]; then
  HAND_TOPOLOGY_ARGS+=(--right-gripper-port "$RIGHT_GRIPPER_PORT")
fi

PROGRESS_PROFILE_ID="dmtac_rm65_right_gripper_rule_writing_progress"
STAGE_PROFILE_ID="dmtac_rm65_right_gripper_rule_writing_progress_detailed"
if [[ "$WITH_LEFT_GRIPPER" -eq 1 ]]; then
  PROGRESS_PROFILE_ID="dmtac_rm65_bimanual_position_writing_four_stage_progress"
  STAGE_PROFILE_ID="dmtac_rm65_bimanual_position_writing_four_stage_progress_detailed"
fi
COLLECTION_TASK_ID="dmtac_rm65_rule_based_writing_v1"
COLLECTION_DESCRIPTION="rule-based pen-down replay, fixed canonical-desk writing, and relative pen-up replay"
PROGRESS_ENTRY_STEP="right_write"
PROGRESS_ENTRY_STAGE="right_write.pen_down"
if [[ "$COLLECTION_MODE" == "full-replay" ]]; then
  PROGRESS_PROFILE_ID="dmtac_rm65_full_trajectory_replay_v1_progress"
  STAGE_PROFILE_ID="dmtac_rm65_full_trajectory_replay_v1_progress"
  COLLECTION_TASK_ID="dmtac_rm65_full_trajectory_replay_v1"
  COLLECTION_DESCRIPTION="complete source Episode replay recorded as one independent Episode"
  PROGRESS_ENTRY_STEP="execute_task"
  PROGRESS_ENTRY_STAGE="execute_task.full_replay"
  if [[ "$TASK_NAME" == "rule_based_writing" ]]; then
    TASK_NAME="full_trajectory_replay"
  fi
fi

TELEOP_ARGS=(
  "${RULE_COMMON_REAL_TELEOP_ARGS[@]}"
  "${TACTILE_RUNTIME_ARGS[@]}"
  "${DATASET_COMMON_ARGS[@]}"
  # Rule replay already publishes one safe 60 Hz reference stream and the
  # established acceleration planner updates it without blocking the 120 Hz
  # CAN-FD worker.  The sample-aligned QP profile solves synchronously for every
  # strictly newer reference and therefore remains an explicit opt-in, not the
  # rule-collection default.
  "${POSITION_FORMAL_V2_BALANCED_ARGS[@]}"
  --tcp-pose-envelope-profile "$TCP_POSE_ENVELOPE_PROFILE"
  --tcp-pose-envelope-active
  "${POSITION_COLLECTION_PROCESS_ARGS[@]}"
  "${POSITION_COLLECTION_PRODUCTION_ARGS[@]}"
  --dataset-root "$DATASET_BASE"
  --collection-task-id "$COLLECTION_TASK_ID"
  --collection-task-name "$TASK_NAME"
  --collection-task-description
  "$COLLECTION_DESCRIPTION"
  --collection-progress-profile
  "catalog:$PROGRESS_PROFILE_ID"
  --collection-stage-profile
  "catalog:$STAGE_PROFILE_ID"
  --collection-progress-entry-step "$PROGRESS_ENTRY_STEP"
  --collection-progress-entry-stage "$PROGRESS_ENTRY_STAGE"
  "${HAND_TOPOLOGY_ARGS[@]}"
  --dmtac-config latest
  --require-dmtac-config
  --camera-config teleop_runtime/configs/devices/realsense.json
  --camera-color-format bgr8
  --camera-frame-max-age-sec "$CAMERA_FRAME_MAX_AGE_SEC"
  --write-parquet
  --write-video
  --gui-resizable-sidebars
  --tactile-storage external
  --tactile-compression none
  --tactile-bundle-root "$TACTILE_BUNDLE_ROOT"
  --tactile-bundle-min-free-gb "$TACTILE_BUNDLE_MIN_FREE_GB"
  --tactile-artifact-integrity
  --idle-tactile-compression
  --no-gripper-warmup-probe
  "${RULE_ARGS[@]}"
)

RUNTIME_PYTHON="${PYTHON_BIN:-python}"
RUN_COMMAND=(
  taskset -c 0-7
  "$RUNTIME_PYTHON" -m control_station.real_rule_based_writing_entrypoint
  "${TELEOP_ARGS[@]}"
)

for required_command in taskset "$RUNTIME_PYTHON"; do
  command -v "$required_command" >/dev/null 2>&1 || \
    die "required command is unavailable after PATH normalization: $required_command"
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  print_command "${RUN_COMMAND[@]}"
  exit 0
fi

# Import the complete entrypoint with the exact interpreter that will own the
# real process. This exercises transitive runtime dependencies and local module
# resolution without constructing a camera, arm, or gripper component.
runtime_dependency_preflight() {
  local import_output runtime_python_path
  if import_output="$("$RUNTIME_PYTHON" -c \
      'import scipy; from scipy.spatial.transform import Rotation; import control_station.real_rule_based_writing_entrypoint; print(scipy.__version__)' \
      2>&1)"; then
    printf '[RuleBased] runtime dependency preflight: scipy=%s python=%s\n' \
      "$import_output" "$RUNTIME_PYTHON"
    return 0
  fi

  runtime_python_path="$("$RUNTIME_PYTHON" -c 'import sys; print(sys.executable)' 2>/dev/null || printf '%s' "$RUNTIME_PYTHON")"
  printf '%s\n' \
    '[RuleBased] runtime dependency preflight failed; no device startup was attempted.' \
    "[RuleBased] python=$runtime_python_path" \
    '[RuleBased] required import: control_station.real_rule_based_writing_entrypoint (including scipy.spatial.transform.Rotation)' \
    '[RuleBased] expected environment declaration: scipy==1.13.1 in environment.yml/pyproject.toml' \
    '[RuleBased] repair example: conda install -p /home/user/miniconda3/envs/dexteleop310 -c conda-forge scipy=1.13.1' \
    '[RuleBased] import diagnostic:' \
    "$import_output" >&2
  return 1
}

if ! runtime_dependency_preflight; then
  # This is a fail-closed startup refusal, intentionally distinct from the
  # launcher's CLI validation path; do not turn an environment fault into a
  # Python traceback or the legacy exit-2 validation path.
  exit 1
fi

printf '%s\n' \
  "[RuleBased] starting collection mode=$COLLECTION_MODE on the established real-teleop UI." \
  '[RuleBased] clear the workspace; start one formal Episode only after the existing GUI reports ready.' \
  "[RuleBased] source_pool=${SOURCE_POOL_MANIFEST:-legacy_single_source} range=${SOURCE_POOL_RANGE:-all} seed=$SELECTION_SEED" \
  "[RuleBased] pre_pen_down_replay=${PRE_PEN_DOWN_REPLAY_CANDIDATE_ID:-disabled} normal_episodes_between_replays=$PRE_PEN_DOWN_REPLAY_NORMAL_EPISODES" \
  "[RuleBased] source_dataset=${SOURCE_DATASET:-pool_selected} episode=${SOURCE_EPISODE:-pool_selected}" \
  "[RuleBased] detail_annotation_dir=${DETAIL_ANNOTATION_DIR:-pool_selected}" \
  "[RuleBased] gripper_port_override left=${LEFT_GRIPPER_PORT:-assignment} right=${RIGHT_GRIPPER_PORT:-assignment}" \
  "[RuleBased] trajectory=$TRAJECTORY catalog=${TRAJECTORY_CATALOG:-builtin} size_m=$SIZE_M direction=$DIRECTION speed_m_s=${SPEED_M_S:-source_average}" \
  "[RuleBased] output_dataset_root=$DATASET_BASE tactile_request=${COLLECT_TACTILE:-dataset_contract_or_default} tactile_root=$TACTILE_BUNDLE_ROOT min_free_gb=$TACTILE_MIN_FREE_GB"
exec "${RUN_COMMAND[@]}"
