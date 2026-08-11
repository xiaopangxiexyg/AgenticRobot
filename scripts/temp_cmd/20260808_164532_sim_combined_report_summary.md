for report in \
  /tmp/dexteleop_sim_combined_rotation_active_20260808.json \
  /tmp/dexteleop_sim_combined_rotation_shadow_20260808.json \
  /tmp/dexteleop_sim_combined_normal_shadow_20260808.json \
  /tmp/dexteleop_sim_combined_cautious_shadow_20260808.json
do
  printf '\n%s\n' "$report"
  if [[ ! -f "$report" ]]; then
    printf 'missing\n'
    continue
  fi
  jq '{passed,safety_passed,coverage_passed,input:{frame_count:.input.frame_count,recorded_duration_sec:.input.recorded_duration_sec,recorded_admittance_mode:.input.recorded_admittance_mode},modes:(.comparisons|with_entries(.value={passed:.value.passed,safety_passed:.value.safety_passed,checks:.value.checks,max_force_release_catchup_m:.value.force_release_transients.max_outward_catchup_m,max_torque_release_catchup_deg:.value.torque_release_transients.max_outward_catchup_deg,max_physical_translation_m:.value.selected_measured_translation_m.max,max_physical_rotation_deg:.value.selected_measured_rotation_deg.max}))}' "$report"
done
