(
set -euo pipefail

fail_preflight() {
  printf 'ERROR: %s\n' "$1" >&2
  printf '%s\n' \
    'Run the paired local preflight first, then launch this command within 60 seconds.' >&2
  exit 5
}

safety_now_sec=$(date +%s)
for safety_spec in \
  /tmp/codex_normal_parity_balanced5_right_safety_ok:192.168.1.20 \
  /tmp/codex_normal_parity_balanced5_left_safety_ok:192.168.101.21
do
  safety_token=${safety_spec%%:*}
  safety_arm_ip=${safety_spec#*:}
  [[ -f "$safety_token" ]] || fail_preflight "missing safety token: $safety_token"
  grep -qx "arm_ip=$safety_arm_ip" "$safety_token" || \
    fail_preflight "safety token arm mismatch: $safety_token"
  grep -qx 'project=/home/user/zc/DexTeleop' "$safety_token" || \
    fail_preflight "safety token project mismatch: $safety_token"
  grep -Eq '^guard_checks=[1-9][0-9]*$' "$safety_token" || \
    fail_preflight "safety token has no guard checks: $safety_token"
  safety_created_sec=$(sed -n 's/^created_at=//p' "$safety_token" | head -n 1)
  [[ "$safety_created_sec" =~ ^[0-9]+$ ]] || \
    fail_preflight "safety token has invalid created_at: $safety_token"
  safety_age_sec=$((safety_now_sec - safety_created_sec))
  (( safety_age_sec >= 0 && safety_age_sec <= 60 )) || \
    fail_preflight "safety token is stale (${safety_age_sec}s): $safety_token"
  printf '[preflight] valid token=%s age=%ss arm=%s\n' \
    "$safety_token" "$safety_age_sec" "$safety_arm_ip"
done

printf '%s\n' '[stage] activating dexteleop310'
cd /home/user/zc/DexTeleop
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate /home/user/miniconda3/envs/dexteleop310
printf '%s\n' '[stage] loading formal teleop arguments'
source docs/real/teleop_args.sh
printf '%s\n' '[stage] launching real teleop; Ctrl-C once to stop'
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
  --position-admittance-rotation-active \
  --position-admittance-combined-active \
  --position-admittance-rotation-active-profile balanced-5deg \
  --position-admittance-active-max-tool-excursion-m 0.08 \
  --position-admittance-active-max-tool-rotation-excursion-deg 5.5 \
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
  --debug-action-log-root teleop_records/action_debug/left_combined_active_normal_parity_balanced5_20260808_211639 \
  --debug-action-flush-every 1 \
  --debug-action-queue-size 8192 \
  --debug-action-sample-hz 60 \
  --max-steps 3600 \
  "${CONSOLE_TIMING_ARGS[@]}"
)
