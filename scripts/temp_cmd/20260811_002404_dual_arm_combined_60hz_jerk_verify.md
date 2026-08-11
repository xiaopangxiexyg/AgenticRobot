(
set -euo pipefail

printf '%s\n' '[stage] activating dexteleop310'
cd /home/user/zc/DexTeleop
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate /home/user/miniconda3/envs/dexteleop310
printf '%s\n' '[stage] loading formal teleop arguments'
source docs/real/teleop_args.sh
printf '%s\n' '[stage] launching bounded dual-arm combined Active with the default-off 60 Hz jerk-limited qcmd profile'
PYTHONUNBUFFERED=1 taskset -c 0-7 \
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${NO_GRIPPER_COMMON_ARGS[@]}" \
  "${POSITION_FORMAL_V2_BALANCED_60HZ_JERK_ADMITTANCE_ARGS[@]}" \
  --active-sides both \
  --no-init-arm-available \
  --control-debug off \
  --actuator-trace \
  --no-actuator-singularity-trace \
  --actuator-trace-root teleop_records/actuator_identification/dual_arm_combined_active_60hz_jerk_verify_20260811_002404 \
  --actuator-trace-max-duration-sec 120 \
  --debug-action \
  --debug-action-log-root teleop_records/action_debug/dual_arm_combined_active_60hz_jerk_verify_20260811_002404 \
  --debug-action-flush-every 1 \
  --debug-action-queue-size 8192 \
  --debug-action-sample-hz 60 \
  --max-steps 3600 \
  "${CONSOLE_TIMING_ARGS[@]}"
)
