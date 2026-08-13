(
set -euo pipefail

: "${DEXTELEOP_ROOT:=/home/user/zc/DexTeleop}"
: "${DEXTELEOP_ENV_NAME:=dexteleop310}"

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

source docs/real/teleop_args.sh

TELEOP_ARGS=(
  "${NO_GRIPPER_COMMON_ARGS[@]}"
  "${POSITION_FORMAL_V2_BALANCED_ARGS[@]}"
  --fix-arm
  --special-arm-control-mode foot-pad
  --no-init-arm-available
  --no-position-admittance-shadow
  --no-position-admittance-rotation-shadow
  --no-position-admittance-active
  --no-position-admittance-rotation-active
  --no-position-admittance-combined-active
  --control-debug off
  --actuator-trace
  --no-actuator-singularity-trace
  --actuator-trace-root teleop_records/actuator_identification/runtime_home_vive_switch_test_20260813_132118
  --actuator-trace-max-duration-sec 180
  --no-debug-action
  --no-debug-timing-auto-log
)

PYTHONUNBUFFERED=1 taskset -c 0-7 \
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${TELEOP_ARGS[@]}"
)
