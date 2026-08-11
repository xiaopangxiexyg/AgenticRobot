cd /home/user/zc/DexTeleop && \
source /home/user/miniconda3/etc/profile.d/conda.sh && \
conda activate /home/user/miniconda3/envs/dexteleop310 && \
source docs/real/teleop_args.sh && \
taskset -c 0-7 \
python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  "${NO_GRIPPER_COMMON_ARGS[@]}" \
  "${POSITION_FORMAL_V2_BALANCED_ARGS[@]}" \
  --active-sides both \
  --no-init-arm-available \
  "${FIX_ARM_ARGS[@]}" \
  --fix-arm-side right \
  --position-admittance-shadow \
  --position-admittance-rotation-shadow \
  --position-admittance-rotation-shadow-profile balanced-5deg \
  --no-position-admittance-active \
  --control-debug off \
  --debug-action \
  --debug-action-log-root teleop_records/action_debug/left_rotation_balanced5_shadow_audit_20260807_232116 \
  --debug-action-flush-every 1 \
  --debug-action-queue-size 8192 \
  --debug-action-sample-hz 60 \
  "${CONSOLE_TIMING_ARGS[@]}"
