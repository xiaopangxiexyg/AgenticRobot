(
set -euo pipefail

printf '%s\n' '[stage] activating dexteleop310'
cd /home/user/zc/DexTeleop
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate /home/user/miniconda3/envs/dexteleop310
export POSITION_ADMITTANCE_TRANSLATION_RESPONSE_WEIGHT=0.5
export POSITION_ADMITTANCE_ROTATION_RESPONSE_WEIGHT=0.5
export POSITION_ADMITTANCE_TRANSLATION_FILTER_TAU_SEC=0.010857362
export POSITION_ADMITTANCE_ROTATION_FILTER_TAU_SEC=0.010857362
export POSITION_ADMITTANCE_TRANSLATION_OUTPUT_FOLLOW_TAU_SEC=0.025
export POSITION_ADMITTANCE_ROTATION_OUTPUT_FOLLOW_TAU_SEC=0.025
export POSITION_ADMITTANCE_TRANSLATION_MAX_OUTPUT_ACCELERATION_M_S2=0
export POSITION_ADMITTANCE_ROTATION_MAX_OUTPUT_ANGULAR_ACCELERATION_RAD_S2=0
printf '%s\n' '[stage] loading formal JERK V2 admittance rollback: 25 ms follower, output acceleration limiter disabled'
source docs/real/teleop_args.sh
printf '%s\n' '[stage] launching dual-arm 25 ms output-follow candidate; Ctrl-C once to stop'
PYTHONUNBUFFERED=1 taskset -c 0-7 \
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${NO_GRIPPER_COMMON_ARGS[@]}" \
  "${POSITION_FORMAL_V2_BALANCED_60HZ_JERK_V2_DAMPED_ADMITTANCE_ARGS[@]}" \
  --active-sides both \
  --no-init-arm-available \
  --control-debug off \
  --actuator-trace \
  --no-actuator-singularity-trace \
  --actuator-trace-root teleop_records/actuator_identification/dual_arm_admittance_output_follow_25ms_weight_0p5_20260811_202254 \
  --actuator-trace-max-duration-sec 120 \
  --debug-action \
  --debug-action-log-root teleop_records/action_debug/dual_arm_admittance_output_follow_25ms_weight_0p5_20260811_202254 \
  --debug-action-flush-every 1 \
  --debug-action-queue-size 8192 \
  --debug-action-sample-hz 60 \
  "${CONSOLE_TIMING_ARGS[@]}"
)
