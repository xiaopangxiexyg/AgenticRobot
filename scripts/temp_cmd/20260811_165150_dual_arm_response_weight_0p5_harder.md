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
printf '%s\n' '[stage] loading formal teleop arguments: translation/rotation weight=0.5/0.5, tau=0.010857362/0.010857362 s'
source docs/real/teleop_args.sh
printf '%s\n' '[stage] launching harder response-weight candidate; Ctrl-C once to stop'
PYTHONUNBUFFERED=1 taskset -c 0-7 \
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${NO_GRIPPER_COMMON_ARGS[@]}" \
  "${POSITION_FORMAL_V2_BALANCED_60HZ_JERK_V2_ADMITTANCE_ARGS[@]}" \
  --active-sides both \
  --no-init-arm-available \
  --control-debug off \
  --actuator-trace \
  --no-actuator-singularity-trace \
  --actuator-trace-root teleop_records/actuator_identification/dual_arm_response_weight_0p5_harder_20260811_165150 \
  --actuator-trace-max-duration-sec 120 \
  --debug-action \
  --debug-action-log-root teleop_records/action_debug/dual_arm_response_weight_0p5_harder_20260811_165150 \
  --debug-action-flush-every 1 \
  --debug-action-queue-size 8192 \
  --debug-action-sample-hz 60 \
  "${CONSOLE_TIMING_ARGS[@]}"
)
