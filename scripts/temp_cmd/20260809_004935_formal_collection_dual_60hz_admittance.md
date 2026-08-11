(
set -euo pipefail

DEXTELEOP_ROOT=/home/user/zc/DexTeleop
if [[ ! -d "$DEXTELEOP_ROOT" ]]; then
  printf '[DexTeleop] project root not found: %s\n' "$DEXTELEOP_ROOT" >&2
  exit 1
fi

cd -- "$DEXTELEOP_ROOT"
export PYTHONPATH="$DEXTELEOP_ROOT${PYTHONPATH:+:$PYTHONPATH}"
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate /home/user/miniconda3/envs/dexteleop310

export LEFT_ARM_IP="${LEFT_ARM_IP:-192.168.101.21}"
export RIGHT_ARM_IP="${RIGHT_ARM_IP:-192.168.1.20}"
export DATASET_BASE="${DATASET_BASE:-/home/user/zc/datasets/RealmanDMTAC/FormalCollection}"

# Relative to the accepted real-robot response. 1.0/1.0 preserves acceptance.
# Larger weight means more compliance before the unchanged hard modifier bounds.
export POSITION_ADMITTANCE_TRANSLATION_RESPONSE_WEIGHT="${POSITION_ADMITTANCE_TRANSLATION_RESPONSE_WEIGHT:-1.0}"
export POSITION_ADMITTANCE_ROTATION_RESPONSE_WEIGHT="${POSITION_ADMITTANCE_ROTATION_RESPONSE_WEIGHT:-1.0}"
# Effective force/torque input damping. Larger tau is smoother and slower.
export POSITION_ADMITTANCE_TRANSLATION_FILTER_TAU_SEC="${POSITION_ADMITTANCE_TRANSLATION_FILTER_TAU_SEC:-0.010857362}"
export POSITION_ADMITTANCE_ROTATION_FILTER_TAU_SEC="${POSITION_ADMITTANCE_ROTATION_FILTER_TAU_SEC:-0.010857362}"

export COLLECTION_TASK_NAME="${COLLECTION_TASK_NAME:-dmtac_rm65_bimanual_position_v2_writing_4_stage}"
export COLLECTION_TASK_DESCRIPTION="${COLLECTION_TASK_DESCRIPTION:-formal rm65 bimanual position v2 balanced with accepted dual-arm 60hz translation and rotation admittance}"

printf '%s\n' '[stage] loading formal collection and accepted 60 Hz admittance arguments'
source docs/real/teleop_args.sh

TELEOP_ARGS=(
  "${COMMON_REAL_TELEOP_ARGS[@]}"
  "${DMTAC_CAPTURE_COMMON_ARGS[@]}"
  "${DMTAC_A_ARGS[@]}"
  "${DMTAC_TACTILE_WORKER_2CPU_ARGS[@]}"
  "${DATASET_COMMON_ARGS[@]}"
  "${POSITION_FORMAL_V2_BALANCED_60HZ_ADMITTANCE_ARGS[@]}"
  "${POSITION_COLLECTION_PROCESS_ARGS[@]}"
  "${POSITION_COLLECTION_PRODUCTION_ARGS[@]}"
  --dataset-root "$DATASET_BASE"
  --collection-task-name "$COLLECTION_TASK_NAME"
  --collection-task-description "$COLLECTION_TASK_DESCRIPTION"
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

for index in "${!TELEOP_ARGS[@]}"; do
  arg=${TELEOP_ARGS[$index]}
  if [[ -z "${arg//[[:space:]]/}" ]]; then
    printf '[DexTeleop] refusing whitespace-only argument at index=%s value=%q\n' \
      "$index" "$arg" >&2
    exit 2
  fi
done

printf '[config] translation_weight=%s rotation_weight=%s translation_filter_tau_sec=%s rotation_filter_tau_sec=%s\n' \
  "$POSITION_ADMITTANCE_TRANSLATION_RESPONSE_WEIGHT" \
  "$POSITION_ADMITTANCE_ROTATION_RESPONSE_WEIGHT" \
  "$POSITION_ADMITTANCE_TRANSLATION_FILTER_TAU_SEC" \
  "$POSITION_ADMITTANCE_ROTATION_FILTER_TAU_SEC"
printf '%s\n' '[stage] launching formal collection; use the GUI Hold/Home flow and Ctrl-C once to stop'

taskset -c 0-7 \
  python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${TELEOP_ARGS[@]}"
)
