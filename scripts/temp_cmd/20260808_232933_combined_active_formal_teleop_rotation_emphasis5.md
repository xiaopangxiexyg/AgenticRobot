(
set -euo pipefail

printf '%s\n' '[stage] activating dexteleop310'
cd /home/user/zc/DexTeleop
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate /home/user/miniconda3/envs/dexteleop310
printf '%s\n' '[stage] loading formal teleop arguments'
source docs/real/teleop_args.sh
printf '%s\n' '[stage] launching formal teleop plus bounded translation/rotation admittance; Ctrl-C once to stop'
PYTHONUNBUFFERED=1 taskset -c 0-7 \
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${NO_GRIPPER_COMMON_ARGS[@]}" \
  "${POSITION_FORMAL_V2_BALANCED_ARGS[@]}" \
  --active-sides both \
  --no-init-arm-available \
  "${FIX_ARM_ARGS[@]}" \
  --fix-arm-side right \
  --no-position-admittance-shadow \
  --position-admittance-active \
  --position-admittance-active-side left \
  --position-admittance-active-envelope normal_parity \
  --position-admittance-active-endpoint-policy formal_teleop \
  --position-admittance-rotation-active \
  --position-admittance-combined-active \
  --position-admittance-rotation-active-profile rotation-emphasis-5deg \
  --position-admittance-response-model profile_compliance \
  --position-admittance-shadow-axes xyz \
  --position-admittance-shadow-stiffness-n-m 250 \
  --position-admittance-shadow-deadband-n 0.45 \
  --position-admittance-shadow-filter-tau-sec 0.010857362 \
  --position-admittance-shadow-max-offset-m 0.08 \
  --position-admittance-shadow-max-velocity-m-s 0.06 \
  --position-admittance-shadow-max-return-velocity-m-s 0.03 \
  --position-admittance-shadow-max-slew-m-s 0.06 \
  --position-admittance-shadow-return-deadband-n 0.45 \
  --position-admittance-shadow-return-rate-s 1 \
  --position-admittance-response-reference-hz 40 \
  --position-admittance-shadow-stale-sec 0.15 \
  --position-admittance-shadow-max-force-dt-sec 0.15 \
  --control-debug off \
  --debug-action \
  --debug-action-log-root teleop_records/action_debug/left_combined_active_formal_teleop_rotation_emphasis5_20260808_232933 \
  --debug-action-flush-every 1 \
  --debug-action-queue-size 8192 \
  --debug-action-sample-hz 60 \
  "${CONSOLE_TIMING_ARGS[@]}"
)
