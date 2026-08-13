(
set -euo pipefail

: "${DEXTELEOP_ROOT:=/home/user/zc/DexTeleop}"
: "${DEXTELEOP_ENV_NAME:=dexteleop310}"
: "${DATASET_BASE:=/home/user/zc/datasets/RealmanDMTAC/FormalCollection}"

if [[ ! -d "$DEXTELEOP_ROOT" ]]; then
  printf '[DexTeleop] project root not found: %s\n' "$DEXTELEOP_ROOT" >&2
  exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
  printf '%s\n' '[DexTeleop] conda is not available in this terminal.' >&2
  exit 1
fi

cd -- "$DEXTELEOP_ROOT"
export PYTHONPATH="$DEXTELEOP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
conda activate "$DEXTELEOP_ENV_NAME"

export LEFT_ARM_IP=192.168.101.21
export RIGHT_ARM_IP=192.168.1.20
export DATASET_BASE
export POSITION_ADMITTANCE_TRANSLATION_RESPONSE_WEIGHT=0.5
export POSITION_ADMITTANCE_ROTATION_RESPONSE_WEIGHT=0.5
export POSITION_ADMITTANCE_TRANSLATION_FILTER_TAU_SEC=0.010857362
export POSITION_ADMITTANCE_ROTATION_FILTER_TAU_SEC=0.010857362
export POSITION_ADMITTANCE_TRANSLATION_OUTPUT_FOLLOW_TAU_SEC=0.025
export POSITION_ADMITTANCE_ROTATION_OUTPUT_FOLLOW_TAU_SEC=0.025
export POSITION_ADMITTANCE_TRANSLATION_MAX_OUTPUT_ACCELERATION_M_S2=1.2
export POSITION_ADMITTANCE_ROTATION_MAX_OUTPUT_ANGULAR_ACCELERATION_RAD_S2=3.0

source docs/real/teleop_args.sh

TELEOP_ARGS=(
  "${COMMON_REAL_TELEOP_ARGS[@]}"
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}"
  "${DMTAC_A_ARGS[@]}"
  "${DMTAC_TACTILE_WORKER_2CPU_ARGS[@]}"
  "${DATASET_COMMON_ARGS[@]}"
  "${POSITION_FORMAL_V2_BALANCED_60HZ_JERK_V2_DAMPED_ADMITTANCE_ARGS[@]}"
  "${POSITION_COLLECTION_PROCESS_ARGS[@]}"
  --left-hand-backend none
  --dmtac-allow-missing-devices
  --policy-io-layout auto
  --control-debug off
  --no-debug-action
  --no-actuator-trace
  --no-debug-timing-auto-log
  --no-debug-disk-write
  --dataset-root "$DATASET_BASE"
  --collection-task-name dmtac_rm65_bimanual_60hz_admittance_left_gripper_offline_test
  --collection-task-description "formal bimanual 60hz admittance test with left gripper intentionally offline and right gripper measured"
  --q6-config latest
  --q6-config-required
  --dmtac-config latest
  --require-dmtac-config
  --camera-config teleop_runtime/configs/devices/realsense.json
  --camera-color-format bgr8
  --write-parquet
  --write-video
  --collection-progress-gui
  --gui-resizable-sidebars
  --tactile-artifact-integrity
  --idle-tactile-compression
)

taskset -c 0-7 \
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${TELEOP_ARGS[@]}"
)
