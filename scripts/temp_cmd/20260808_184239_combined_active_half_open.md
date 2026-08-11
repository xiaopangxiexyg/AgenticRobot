set -euo pipefail
safety_now_sec=$(date +%s)
for safety_spec in \
  /tmp/codex_combined_active_right_safety_ok:192.168.1.20 \
  /tmp/codex_combined_active_left_safety_ok:192.168.101.21
do
  safety_token=${safety_spec%%:*}
  safety_arm_ip=${safety_spec#*:}
  [[ -f "$safety_token" ]]
  grep -qx "arm_ip=$safety_arm_ip" "$safety_token"
  grep -qx 'project=/home/user/zc/DexTeleop' "$safety_token"
  safety_created_sec=$(sed -n 's/^created_at=//p' "$safety_token")
  [[ "$safety_created_sec" =~ ^[0-9]+$ ]]
  safety_age_sec=$((safety_now_sec - safety_created_sec))
  (( safety_age_sec >= 0 && safety_age_sec <= 60 ))
done
cd /home/user/zc/DexTeleop
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate /home/user/miniconda3/envs/dexteleop310
source docs/real/teleop_args.sh
taskset -c 0-7 \
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
  --position-admittance-active-envelope cautious \
  --position-admittance-rotation-active \
  --position-admittance-combined-active \
  --position-admittance-rotation-active-profile cautious-3deg \
  --position-admittance-active-max-tool-excursion-m 0.08 \
  --position-admittance-active-max-tool-rotation-excursion-deg 3.5 \
  --position-admittance-response-model profile_compliance \
  --position-admittance-shadow-axes xyz \
  --position-admittance-shadow-stiffness-n-m 250 \
  --position-admittance-shadow-deadband-n 0.45 \
  --position-admittance-shadow-filter-tau-sec 0.010857362 \
  --position-admittance-shadow-max-offset-m 0.002 \
  --position-admittance-shadow-max-velocity-m-s 0.005 \
  --position-admittance-shadow-max-return-velocity-m-s 0.005 \
  --position-admittance-shadow-max-slew-m-s 0.005 \
  --position-admittance-shadow-return-deadband-n 0.45 \
  --position-admittance-shadow-return-rate-s 1 \
  --position-admittance-response-reference-hz 40 \
  --position-admittance-shadow-stale-sec 0.15 \
  --position-admittance-shadow-max-force-dt-sec 0.15 \
  --control-debug off \
  --debug-action \
  --debug-action-log-root teleop_records/action_debug/left_combined_active_cautious_half_open_20260808_184239 \
  --debug-action-flush-every 1 \
  --debug-action-queue-size 8192 \
  --debug-action-sample-hz 60 \
  --max-steps 3600 \
  "${CONSOLE_TIMING_ARGS[@]}"
