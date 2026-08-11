for report in \
  /tmp/dexteleop_sim_normal_parity_balanced5_both_20260808.json \
  /tmp/dexteleop_sim_normal_parity_moderated8_low_excitation_20260808.json \
  /tmp/dexteleop_sim_normal_parity_rotation_wrench_stress_20260808.json
do
  printf '\n%s\n' "$report"
  if [[ ! -f "$report" ]]; then
    printf 'missing\n'
    continue
  fi
  jq '{passed,safety_passed,coverage_passed,hardware_authorization,production_rotational_active_installer_exercised,admittance_envelope,rotational_envelope,component_release_isolation,input:{frame_count:.input.frame_count,recorded_duration_sec:.input.recorded_duration_sec,recorded_admittance_mode:.input.recorded_admittance_mode},modes:(.comparisons|with_entries(.value={passed:.value.passed,safety_passed:.value.safety_passed,error:.value.error,checks:.value.checks,release_counterfactual_baselines:.value.release_counterfactual_baselines,max_force_release_catchup_m:.value.force_release_transients.max_outward_catchup_m,max_torque_release_catchup_deg:.value.torque_release_transients.max_outward_catchup_deg,max_physical_translation_m:.value.selected_measured_translation_m.max,max_physical_rotation_deg:.value.selected_measured_rotation_deg.max})),active_stops:(.rollouts|to_entries|map(select(.key|endswith("_active")))|map({mode:.key,completed:.value.completed,max_tool_excursion_m:.value.tool_endpoint_excursion_m.max,max_tool_rotation_excursion_deg:.value.tool_rotation_excursion_deg.max,model_errors:.value.model_errors}))}' "$report"
done
