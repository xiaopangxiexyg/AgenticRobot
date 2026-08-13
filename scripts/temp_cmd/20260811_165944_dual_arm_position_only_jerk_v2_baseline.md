(
set -euo pipefail

printf '%s\n' '[stage] activating dexteleop310'
cd /home/user/zc/DexTeleop
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate /home/user/miniconda3/envs/dexteleop310
printf '%s\n' '[stage] loading formal position-control arguments'
source docs/real/teleop_args.sh
printf '%s\n' '[stage] launching dual-arm position-only V2/JERK V2 baseline; Ctrl-C once to stop'
PYTHONUNBUFFERED=1 taskset -c 0-7 \
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${NO_GRIPPER_COMMON_ARGS[@]}" \
  "${POSITION_FORMAL_V2_BALANCED_ARGS[@]}" \
  --position-control-profile rm65_bimanual_position_v2_balanced_60hz_jerk_v2 \
  --active-sides both \
  --no-init-arm-available \
  --no-fix-arm \
  --no-position-admittance-shadow \
  --no-position-admittance-rotation-shadow \
  --no-position-admittance-active \
  --no-position-admittance-rotation-active \
  --no-position-admittance-combined-active \
  --control-debug off \
  --actuator-trace \
  --no-actuator-singularity-trace \
  --actuator-trace-root teleop_records/actuator_identification/dual_arm_position_only_jerk_v2_baseline_20260811_165944 \
  --actuator-trace-max-duration-sec 120 \
  --debug-action \
  --debug-action-log-root teleop_records/action_debug/dual_arm_position_only_jerk_v2_baseline_20260811_165944 \
  --debug-action-flush-every 1 \
  --debug-action-queue-size 8192 \
  --debug-action-sample-hz 60 \
  "${CONSOLE_TIMING_ARGS[@]}"
)
