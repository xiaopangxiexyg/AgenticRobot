#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_formal_60hz_writing_collection.sh foot-pad [--dry-run]
  bash scripts/run_formal_60hz_writing_collection.sh collect \
    --tactile-bundle-root /absolute/path [options]

Commands:
  foot-pad  Start the five-pedal forwarding service.
  collect   Start formal dual-arm 60 Hz force-admittance collection with the
            managed four-stage writing progress profile and detail-overlay
            compatible capture metadata.

Collect options:
  --tactile-bundle-root PATH  Required unless TACTILE_BUNDLE_ROOT is set.
  --dataset-root PATH         Defaults to DATASET_BASE or the site default in
                              docs/real/teleop_args.sh.
  --task-name NAME            Defaults to 4_stage_writing_detail.
  --tactile-min-free-gb N     Defaults to TACTILE_BUNDLE_MIN_FREE_GB or 200.
  --dry-run                   Print the resolved command without starting devices.
  -h, --help                  Show this help.

Run both commands from separate terminals in an activated dexteleop310
environment. The tactile root must already exist, must not be a symlink, and
must be on a different filesystem from the formal dataset root.
EOF
}

die() {
  printf '[DexTeleop] %s\n' "$*" >&2
  exit 2
}

print_command() {
  printf '[dry-run]'
  printf ' %q' "$@"
  printf '\n'
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEXTELEOP_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
COMMAND="${1:-}"
if [[ -z "$COMMAND" ]]; then
  usage >&2
  exit 2
fi
shift

case "$COMMAND" in
  -h|--help|help)
    usage
    exit 0
    ;;
  foot-pad)
    if [[ "${1:-}" == "--dry-run" ]]; then
      shift
      [[ $# -eq 0 ]] || die "foot-pad accepts only --dry-run."
      print_command python -m hardware.foot_pad --count 5
      exit 0
    fi
    [[ $# -eq 0 ]] || die "foot-pad accepts no additional arguments."
    cd -- "$DEXTELEOP_ROOT"
    exec python -m hardware.foot_pad --count 5
    ;;
  collect)
    ;;
  *)
    usage >&2
    die "unknown command: $COMMAND"
    ;;
esac

TACTILE_ROOT="${TACTILE_BUNDLE_ROOT:-}"
DATASET_ROOT="${DATASET_BASE:-}"
TASK_NAME="${COLLECTION_TASK_NAME:-4_stage_writing_detail}"
TACTILE_MIN_FREE_GB="${TACTILE_BUNDLE_MIN_FREE_GB:-200}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tactile-bundle-root)
      [[ $# -ge 2 ]] || die "--tactile-bundle-root requires PATH."
      TACTILE_ROOT="$2"
      shift 2
      ;;
    --dataset-root)
      [[ $# -ge 2 ]] || die "--dataset-root requires PATH."
      DATASET_ROOT="$2"
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
      die "unknown collect option: $1"
      ;;
  esac
done

[[ -n "$TACTILE_ROOT" ]] || die \
  "set TACTILE_BUNDLE_ROOT or pass --tactile-bundle-root PATH."
[[ "$TACTILE_ROOT" == /* ]] || die "tactile bundle root must be absolute: $TACTILE_ROOT"
[[ -d "$TACTILE_ROOT" ]] || die "tactile bundle root does not exist: $TACTILE_ROOT"
[[ ! -L "$TACTILE_ROOT" ]] || die "tactile bundle root must not be a symlink: $TACTILE_ROOT"
[[ -n "${TASK_NAME//[[:space:]]/}" ]] || die "task name must not be blank."

if [[ -n "$DATASET_ROOT" ]]; then
  [[ "$DATASET_ROOT" == /* ]] || die "dataset root must be absolute: $DATASET_ROOT"
  if [[ -e "$DATASET_ROOT" ]]; then
    dataset_device="$(stat -c '%d' -- "$DATASET_ROOT")"
    tactile_device="$(stat -c '%d' -- "$TACTILE_ROOT")"
    [[ "$dataset_device" != "$tactile_device" ]] || die \
      "tactile bundle root must be on a different filesystem from dataset root."
  fi
  export DATASET_BASE="$DATASET_ROOT"
fi
export TACTILE_BUNDLE_ROOT="$TACTILE_ROOT"
export TACTILE_BUNDLE_MIN_FREE_GB="$TACTILE_MIN_FREE_GB"

cd -- "$DEXTELEOP_ROOT"
# shellcheck source=../docs/real/teleop_args.sh
source docs/real/teleop_args.sh

TELEOP_ARGS=(
  "${COMMON_REAL_TELEOP_ARGS[@]}"
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}"
  "${DMTAC_A_ARGS[@]}"
  "${DMTAC_TACTILE_WORKER_2CPU_ARGS[@]}"
  "${DATASET_COMMON_ARGS[@]}"
  "${POSITION_FORMAL_V2_BALANCED_60HZ_JERK_V2_DAMPED_ADMITTANCE_ARGS[@]}"
  "${POSITION_COLLECTION_PROCESS_ARGS[@]}"
  "${POSITION_COLLECTION_PRODUCTION_ARGS[@]}"
  --dataset-root "$DATASET_BASE"
  --collection-task-id dmtac_rm65_bimanual_position_v1
  --collection-task-name "$TASK_NAME"
  --collection-task-description
  "formal dual-arm 60 Hz force-admittance writing collection with four-stage progress and post-capture detail overlay"
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
)

RUN_COMMAND=(
  taskset -c 0-7
  python -m teleop_runtime.debug.run_vive_rm65_real_teleop
  "${TELEOP_ARGS[@]}"
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  print_command "${RUN_COMMAND[@]}"
  exit 0
fi

printf '%s\n' \
  '[DexTeleop] starting formal 60 Hz force-admittance collection.' \
  '[DexTeleop] five-pedal forwarding must be running in a separate terminal.' \
  "[DexTeleop] task=$TASK_NAME" \
  "[DexTeleop] dataset_root=$DATASET_BASE" \
  "[DexTeleop] tactile_bundle_root=$TACTILE_BUNDLE_ROOT"
exec "${RUN_COMMAND[@]}"
