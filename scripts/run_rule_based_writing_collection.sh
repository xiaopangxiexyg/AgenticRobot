#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_rule_based_writing_collection.sh \
    --source-dataset /absolute/source/dataset \
    --source-episode N \
    --detail-annotation-dir /absolute/detail/set \
    --tactile-bundle-root /absolute/tactile/root [options]

Required:
  --source-dataset PATH       Immutable annotated source dataset.
  --source-episode N          Source episode containing pen_down/write/pen_up.
  --detail-annotation-dir P   Exact immutable detail annotation-set directory.
  --tactile-bundle-root PATH  Existing external tactile output root.

Options:
  --dataset-root PATH         Formal output dataset root.
  --trajectory ID             One-stroke catalog ID (default: digit_1_v1).
  --size-m M                  Glyph height in metres (default: 0.08).
  --direction NAME            up, right, down, or left (default: up).
  --speed-m-s MPS             Override source-derived average TCP speed.
  --task-name NAME            Output collection task name.
  --tactile-min-free-gb N     Required tactile free space (default: 200).
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
TACTILE_ROOT="${TACTILE_BUNDLE_ROOT:-}"
OUTPUT_DATASET_ROOT="${DATASET_BASE:-}"
TRAJECTORY="digit_1_v1"
SIZE_M="0.08"
DIRECTION="up"
SPEED_M_S=""
TASK_NAME="rule_based_writing"
TACTILE_MIN_FREE_GB="${TACTILE_BUNDLE_MIN_FREE_GB:-200}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
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

[[ -n "$SOURCE_DATASET" ]] || die "--source-dataset is required."
[[ -n "$SOURCE_EPISODE" ]] || die "--source-episode is required."
[[ -n "$DETAIL_ANNOTATION_DIR" ]] || die "--detail-annotation-dir is required."
[[ -n "$TACTILE_ROOT" ]] || die "--tactile-bundle-root is required."
[[ "$SOURCE_DATASET" == /* ]] || die "source dataset must be absolute."
[[ "$DETAIL_ANNOTATION_DIR" == /* ]] || die "detail annotation dir must be absolute."
[[ "$TACTILE_ROOT" == /* ]] || die "tactile root must be absolute."
[[ -d "$SOURCE_DATASET" ]] || die "source dataset does not exist: $SOURCE_DATASET"
[[ -d "$DETAIL_ANNOTATION_DIR" ]] || die "detail annotation dir does not exist: $DETAIL_ANNOTATION_DIR"
[[ -d "$TACTILE_ROOT" ]] || die "tactile root does not exist: $TACTILE_ROOT"
[[ ! -L "$TACTILE_ROOT" ]] || die "tactile root must not be a symlink."
[[ "$SOURCE_EPISODE" =~ ^[0-9]+$ ]] || die "source episode must be a non-negative integer."
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

RULE_ARGS=(
  --rule-source-dataset "$SOURCE_DATASET"
  --rule-source-episode "$SOURCE_EPISODE"
  --rule-detail-annotation-dir "$DETAIL_ANNOTATION_DIR"
  --rule-writing-trajectory "$TRAJECTORY"
  --rule-writing-size-m "$SIZE_M"
  --rule-writing-direction "$DIRECTION"
)
if [[ -n "$SPEED_M_S" ]]; then
  RULE_ARGS+=(--rule-writing-speed-m-s "$SPEED_M_S")
fi

TELEOP_ARGS=(
  "${COMMON_REAL_TELEOP_ARGS[@]}"
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}"
  "${DMTAC_A_ARGS[@]}"
  "${DMTAC_TACTILE_WORKER_2CPU_ARGS[@]}"
  "${DATASET_COMMON_ARGS[@]}"
  "${POSITION_FORMAL_V2_BALANCED_60HZ_JERK_V2_ARGS[@]}"
  --tcp-pose-envelope-profile "$TCP_POSE_ENVELOPE_PROFILE"
  --tcp-pose-envelope-active
  "${POSITION_COLLECTION_PROCESS_ARGS[@]}"
  "${POSITION_COLLECTION_PRODUCTION_ARGS[@]}"
  --dataset-root "$DATASET_BASE"
  --collection-task-id dmtac_rm65_rule_based_writing_v1
  --collection-task-name "$TASK_NAME"
  --collection-task-description
  "rule-based pen-down replay, fixed canonical-desk writing, and relative pen-up replay"
  --collection-progress-profile
  catalog:dmtac_rm65_bimanual_position_writing_four_stage_progress
  --collection-progress-gui
  --q6-config latest
  --q6-config-required
  --dmtac-config latest
  --require-dmtac-config
  --camera-config teleop_runtime/configs/devices/realsense.json
  --camera-color-format bgr8
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
  '[RuleBased] starting formal pure-position rule-based writing collection.' \
  '[RuleBased] verify the five-pedal service, clear workspace, and use the GUI to start one formal episode.' \
  "[RuleBased] source_dataset=$SOURCE_DATASET episode=$SOURCE_EPISODE" \
  "[RuleBased] detail_annotation_dir=$DETAIL_ANNOTATION_DIR" \
  "[RuleBased] trajectory=$TRAJECTORY size_m=$SIZE_M direction=$DIRECTION speed_m_s=${SPEED_M_S:-source_average}" \
  "[RuleBased] output_dataset_root=$DATASET_BASE tactile_root=$TACTILE_BUNDLE_ROOT"
exec "${RUN_COMMAND[@]}"
