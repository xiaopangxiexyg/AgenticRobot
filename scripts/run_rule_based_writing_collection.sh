#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_rule_based_writing_collection.sh \
    --source-pool-manifest /absolute/source-pool.json \
    --tactile-bundle-root /absolute/tactile/root [options]

Required:
  --source-pool-manifest P    Cross-dataset source candidate pool; or
  --source-dataset PATH       legacy single-source mode, together with
  --source-episode N          source episode index.
  --detail-annotation-dir P   Exact immutable annotation-set directory; required
                              only in rule-based-writing legacy mode.
  --tactile-bundle-root PATH  Existing external tactile output root.

Options:
  --mode NAME                 rule-based-writing (default) or full-replay.
  --dataset-root PATH         Formal output dataset root.
  --trajectory ID             Writing mode only; one-stroke catalog ID
                              (default: digit_1_v1).
  --size-m M                  Writing mode only; glyph height in metres
                              (default: 0.08).
  --direction NAME            Writing mode only; up, right, down, or left
                              (default: up).
  --speed-m-s MPS             Writing mode only; override source average speed.
  --initial-alignment-speed-scale S
                              Initial dual-arm alignment speed multiplier in (0, 1].
  --source-pool-range A:B     Half-open candidate range (default: whole pool).
  --selection-seed N          Deterministic source selection seed (default: 0).
  --task-name NAME            Output collection task name.
  --tactile-min-free-gb N     Required tactile free space (default: 200).
  --camera-frame-max-age-sec S
                              Maximum admitted RGB frame age (default: 0.25).
  --with-left-gripper         Also connect/capture the saved left DMTAC stack.
  --left-gripper-port PATH    Explicit left DMTAC serial port (optional).
  --right-gripper-port PATH   Explicit right DMTAC serial port (optional).
  --dry-run                   Print the command; do not start any device.
  -h, --help                  Show this help.

This launcher uses the accepted pure-position profile. Active admittance is
intentionally excluded by the real rule-based deployment preflight.
EOF
}

die() {
  printf '[RuleBased] %s\n' "$*" >&2
  exit 2
}

print_command() {
  printf '[dry-run]'
  printf ' %q' "$@"
  printf '\n'
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
TCP_POSE_ENVELOPE_PROFILE="teleop_runtime/configs/real_teleop/rule_based_writing_bimanual_workspace_v1.json"
SOURCE_DATASET=""
SOURCE_EPISODE=""
DETAIL_ANNOTATION_DIR=""
SOURCE_POOL_MANIFEST=""
SOURCE_POOL_RANGE=""
SELECTION_SEED="0"
COLLECTION_MODE="rule-based-writing"
TACTILE_ROOT="${TACTILE_BUNDLE_ROOT:-}"
OUTPUT_DATASET_ROOT="${DATASET_BASE:-}"
TRAJECTORY="digit_1_v1"
SIZE_M="0.08"
DIRECTION="up"
SPEED_M_S=""
INITIAL_ALIGNMENT_SPEED_SCALE="1.0"
LEFT_GRIPPER_PORT=""
RIGHT_GRIPPER_PORT=""
TASK_NAME="rule_based_writing"
TACTILE_MIN_FREE_GB="${TACTILE_BUNDLE_MIN_FREE_GB:-200}"
CAMERA_FRAME_MAX_AGE_SEC="0.25"
WITH_LEFT_GRIPPER=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || die "--mode requires NAME."
      COLLECTION_MODE="$2"
      shift 2
      ;;
    --source-dataset)
      [[ $# -ge 2 ]] || die "--source-dataset requires PATH."
      SOURCE_DATASET="$2"
      shift 2
      ;;
    --source-episode)
      [[ $# -ge 2 ]] || die "--source-episode requires N."
      SOURCE_EPISODE="$2"
      shift 2
      ;;
    --detail-annotation-dir)
      [[ $# -ge 2 ]] || die "--detail-annotation-dir requires PATH."
      DETAIL_ANNOTATION_DIR="$2"
      shift 2
      ;;
    --source-pool-manifest)
      [[ $# -ge 2 ]] || die "--source-pool-manifest requires PATH."
      SOURCE_POOL_MANIFEST="$2"
      shift 2
      ;;
    --source-pool-range)
      [[ $# -ge 2 ]] || die "--source-pool-range requires START:END."
      SOURCE_POOL_RANGE="$2"
      shift 2
      ;;
    --selection-seed)
      [[ $# -ge 2 ]] || die "--selection-seed requires N."
      SELECTION_SEED="$2"
      shift 2
      ;;
    --tactile-bundle-root)
      [[ $# -ge 2 ]] || die "--tactile-bundle-root requires PATH."
      TACTILE_ROOT="$2"
      shift 2
      ;;
    --dataset-root)
      [[ $# -ge 2 ]] || die "--dataset-root requires PATH."
      OUTPUT_DATASET_ROOT="$2"
      shift 2
      ;;
    --trajectory)
      [[ $# -ge 2 ]] || die "--trajectory requires ID."
      TRAJECTORY="$2"
      shift 2
      ;;
    --size-m)
      [[ $# -ge 2 ]] || die "--size-m requires M."
      SIZE_M="$2"
      shift 2
      ;;
    --direction)
      [[ $# -ge 2 ]] || die "--direction requires NAME."
      DIRECTION="$2"
      shift 2
      ;;
    --speed-m-s)
      [[ $# -ge 2 ]] || die "--speed-m-s requires MPS."
      SPEED_M_S="$2"
      shift 2
      ;;
    --initial-alignment-speed-scale)
      [[ $# -ge 2 ]] || die "--initial-alignment-speed-scale requires S."
      INITIAL_ALIGNMENT_SPEED_SCALE="$2"
      shift 2
      ;;
    --left-gripper-port)
      [[ $# -ge 2 ]] || die "--left-gripper-port requires PATH."
      LEFT_GRIPPER_PORT="$2"
      shift 2
      ;;
    --right-gripper-port)
      [[ $# -ge 2 ]] || die "--right-gripper-port requires PATH."
      RIGHT_GRIPPER_PORT="$2"
      shift 2
      ;;
    --task-name)
      [[ $# -ge 2 ]] || die "--task-name requires NAME."
      TASK_NAME="$2"
      shift 2
      ;;
    --tactile-min-free-gb)
      [[ $# -ge 2 ]] || die "--tactile-min-free-gb requires N."
      TACTILE_MIN_FREE_GB="$2"
      shift 2
      ;;
    --camera-frame-max-age-sec)
      [[ $# -ge 2 ]] || die "--camera-frame-max-age-sec requires S."
      CAMERA_FRAME_MAX_AGE_SEC="$2"
      shift 2
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

[[ -n "$TACTILE_ROOT" ]] || die "--tactile-bundle-root is required."
[[ "$TACTILE_ROOT" == /* ]] || die "tactile root must be absolute."
[[ -d "$TACTILE_ROOT" ]] || die "tactile root does not exist: $TACTILE_ROOT"
[[ ! -L "$TACTILE_ROOT" ]] || die "tactile root must not be a symlink."
[[ "$SELECTION_SEED" =~ ^-?[0-9]+$ ]] || die "selection seed must be an integer."
if ! [[ "$INITIAL_ALIGNMENT_SPEED_SCALE" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  die "initial alignment speed scale must be a positive decimal in (0, 1]."
fi
if ! awk "BEGIN { exit !($INITIAL_ALIGNMENT_SPEED_SCALE > 0 && $INITIAL_ALIGNMENT_SPEED_SCALE <= 1) }"; then
  die "initial alignment speed scale must be in (0, 1]."
fi
if ! [[ "$CAMERA_FRAME_MAX_AGE_SEC" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  die "camera frame max age must be a positive decimal."
fi
if ! awk "BEGIN { exit !($CAMERA_FRAME_MAX_AGE_SEC > 0) }"; then
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
if [[ "$COLLECTION_MODE" == "rule-based-writing" ]]; then
  RULE_ARGS+=(
    --rule-writing-trajectory "$TRAJECTORY"
    --rule-writing-size-m "$SIZE_M"
    --rule-writing-direction "$DIRECTION"
  )
fi
if [[ -n "$SOURCE_POOL_MANIFEST" ]]; then
  RULE_ARGS+=(
    --rule-source-pool-manifest "$SOURCE_POOL_MANIFEST"
    --rule-source-selection-seed "$SELECTION_SEED"
  )
  if [[ -n "$SOURCE_POOL_RANGE" ]]; then
    RULE_ARGS+=(--rule-source-pool-range "$SOURCE_POOL_RANGE")
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

HAND_TOPOLOGY_ARGS=(
  --left-hand-backend none
  --right-hand-backend dmtac-gripper
  --gripper-input-source none
  --no-q6-default-config
  --special-arm-control-mode manual
)
if [[ "$WITH_LEFT_GRIPPER" -eq 1 ]]; then
  HAND_TOPOLOGY_ARGS=(
    --left-hand-backend dmtac-gripper
    --right-hand-backend dmtac-gripper
    --gripper-input-source none
    --no-q6-default-config
    --special-arm-control-mode manual
  )
fi

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
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}"
  "${DMTAC_A_ARGS[@]}"
  "${DMTAC_TACTILE_WORKER_2CPU_ARGS[@]}"
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

RUN_COMMAND=(
  taskset -c 0-7
  python -m control_station.real_rule_based_writing_entrypoint
  "${TELEOP_ARGS[@]}"
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  print_command "${RUN_COMMAND[@]}"
  exit 0
fi

printf '%s\n' \
  "[RuleBased] starting collection mode=$COLLECTION_MODE on the established real-teleop UI." \
  '[RuleBased] clear the workspace; start one formal Episode only after the existing GUI reports ready.' \
  "[RuleBased] source_pool=${SOURCE_POOL_MANIFEST:-legacy_single_source} range=${SOURCE_POOL_RANGE:-all} seed=$SELECTION_SEED" \
  "[RuleBased] source_dataset=${SOURCE_DATASET:-pool_selected} episode=${SOURCE_EPISODE:-pool_selected}" \
  "[RuleBased] detail_annotation_dir=${DETAIL_ANNOTATION_DIR:-pool_selected}" \
  "[RuleBased] trajectory=$TRAJECTORY size_m=$SIZE_M direction=$DIRECTION speed_m_s=${SPEED_M_S:-source_average}" \
  "[RuleBased] output_dataset_root=$DATASET_BASE tactile_root=$TACTILE_BUNDLE_ROOT"
exec "${RUN_COMMAND[@]}"
