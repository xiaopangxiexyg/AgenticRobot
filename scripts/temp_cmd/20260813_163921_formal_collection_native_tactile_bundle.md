(
  set -euo pipefail

  DEXTELEOP_ROOT="${DEXTELEOP_ROOT:-/home/user/zc/DexTeleop}"
  DEXTELEOP_ENV_NAME="${DEXTELEOP_ENV_NAME:-dexteleop310}"
  DATASET_BASE="${DATASET_BASE:-/home/user/zc/datasets/RealmanDMTAC/FormalCollection}"
  TACTILE_BUNDLE_ROOT="${TACTILE_BUNDLE_ROOT:-/media/user/T7_LINUX/DexTeleopTactile}"
  TACTILE_BUNDLE_MIN_FREE_GB="${TACTILE_BUNDLE_MIN_FREE_GB:-200}"
  LEFT_ARM_IP="${LEFT_ARM_IP:-192.168.101.21}"
  RIGHT_ARM_IP="${RIGHT_ARM_IP:-192.168.1.20}"

  if [[ ! -d "$DEXTELEOP_ROOT" ]]; then
    printf '[DexTeleop] project root not found: %s\n' "$DEXTELEOP_ROOT" >&2
    exit 1
  fi
  if [[ ! -d "$TACTILE_BUNDLE_ROOT" || -L "$TACTILE_BUNDLE_ROOT" ]]; then
    printf '[DexTeleop] tactile storage root must be a pre-existing non-symlink directory: %s\n' "$TACTILE_BUNDLE_ROOT" >&2
    exit 1
  fi

  cd -- "$DEXTELEOP_ROOT"
  export PYTHONPATH="$DEXTELEOP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$DEXTELEOP_ENV_NAME"
  export LEFT_ARM_IP RIGHT_ARM_IP DATASET_BASE TACTILE_BUNDLE_ROOT
  source docs/real/teleop_args.sh

  taskset -c 0-7 \
  python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
    "${COMMON_REAL_TELEOP_ARGS[@]}" \
    "${DMTAC_CAPTURE_COMMON_ARGS[@]}" \
    "${DMTAC_A_ARGS[@]}" \
    "${DMTAC_TACTILE_WORKER_2CPU_ARGS[@]}" \
    "${DATASET_COMMON_ARGS[@]}" \
    "${POSITION_FORMAL_V1_ARGS[@]}" \
    "${POSITION_COLLECTION_PROCESS_ARGS[@]}" \
    "${POSITION_COLLECTION_PRODUCTION_ARGS[@]}" \
    --dataset-root "$DATASET_BASE" \
    --collection-task-name dmtac_rm65_bimanual_position_v1_native_tactile \
    --collection-task-description "formal rm65 bimanual position v1 with native separated tactile bundle" \
    --q6-config latest \
    --q6-config-required \
    --dmtac-config latest \
    --require-dmtac-config \
    --camera-config teleop_runtime/configs/devices/realsense.json \
    --camera-color-format bgr8 \
    --write-parquet \
    --write-video \
    --tactile-storage external \
    --tactile-compression zstd \
    --tactile-artifact-integrity \
    --no-idle-tactile-compression \
    --tactile-bundle-root "$TACTILE_BUNDLE_ROOT" \
    --tactile-bundle-min-free-gb "$TACTILE_BUNDLE_MIN_FREE_GB"
)
rc=$?
printf '[DexTeleop] command exited with status=%s\n' "$rc"
