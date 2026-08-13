(
set -euo pipefail

DEXTELEOP_ROOT=/home/user/zc/DexTeleop
cd -- "$DEXTELEOP_ROOT"
export PYTHONPATH="$DEXTELEOP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate /home/user/miniconda3/envs/dexteleop310

export LEFT_ARM_IP=192.168.101.21
export RIGHT_ARM_IP=192.168.1.20
export DATASET_BASE=/home/user/zc/datasets/RealmanDMTAC/FormalCollection
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
  --control-debug off
  --no-debug-action
  --no-actuator-trace
  --no-debug-timing-auto-log
  --no-debug-disk-write
  --dataset-root "$DATASET_BASE"
  --collection-task-name dmtac_rm65_bimanual_60hz_admittance_damped_weight_0p5
  --collection-task-description "formal rm65 bimanual 60hz translation and rotation admittance, jerk v2 damped, response weight 0.5"
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
