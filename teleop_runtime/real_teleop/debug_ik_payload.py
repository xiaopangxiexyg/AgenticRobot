"""Pure IK, backend, and impedance debug payload serializers.

These functions operate on already captured command/state/metadata mappings.
They do not read hardware, publish UI state, spawn threads, or mutate runtime
control state.  Frame assembly remains in the legacy entrypoint because it
coordinates runtime readers and status seams.
"""

from __future__ import annotations

from collections.abc import Mapping

from robot_motion.runtime import TeleopCommand

from .debug_payload_primitives import (
    _compact_debug_value,
    _pose_compact_debug,
    _pose_delta_compact_debug,
    _pose_matrix_debug,
    _qpos_compact_debug,
    _qpos_debug,
)

_IK_PROFILE_COMPACT_KEYS = (
    "target_already_reached",
    "frame_total_ms",
    "time_budget_ms",
    "time_budget_exceeded",
    "solve_total_ms",
    "solve_qp_ms",
    "solve_integrate_ms",
    "solve_calls",
    "provider_total_ms",
    "provider_call_count",
    "non_provider_ms",
    "fk_ms",
    "singularity_ms",
    "collision_ms",
    "candidate_eval_ms",
    "candidate_eval_count",
    "local_candidate_eval_count",
    "min_distance_total_ms",
    "min_distance_calls",
    "branch_ms",
    "jump_gate_ms",
    "hold_recovery_ms",
)


def _ik_profile_compact_debug(profile: object) -> dict[str, object]:
    if not isinstance(profile, Mapping):
        return {}
    out = {
        key: _compact_debug_value(profile.get(key))
        for key in _IK_PROFILE_COMPACT_KEYS
        if key in profile
    }
    min_phase = profile.get("min_distance_phase_ms")
    if isinstance(min_phase, Mapping):
        out["min_distance_phase_ms"] = {
            str(key): _compact_debug_value(value)
            for key, value in min_phase.items()
        }
    return out


def _execution_step_compact_debug(metadata: object) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        return {}
    keys = (
        "enabled",
        "step_limited",
        "weighted_delta_deg",
        "execution_step_source",
        "left_max_abs_delta_deg",
        "right_max_abs_delta_deg",
        "joint_velocity_limit_profile",
        "joint_velocity_limit_dt_sec",
        "joint_velocity_limit_max_dt_sec",
        "joint_velocity_limit_ratio",
        "joint_velocity_limit_requested_delta_deg",
        "joint_velocity_limit_delta_deg",
        "joint_velocity_limit_max_step_deg",
        "left_max_step_deg",
        "right_max_step_deg",
    )
    return {
        key: _compact_debug_value(metadata.get(key))
        for key in keys
        if key in metadata
    }


def _candidate_selection_compact_debug(metadata: object) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        return {}
    keys = (
        "enabled",
        "status",
        "reason",
        "selected",
        "selected_source",
        "selected_method",
        "selected_score",
        "selected_step_scale",
        "selection_policy",
        "candidate_eval_mode",
        "candidate_count",
        "evaluated_count",
        "collision_safe_count",
        "rejection_count",
        "candidate_eval_elapsed_ms",
        "candidate_eval_pre_selection_elapsed_ms",
        "candidate_eval_branch_budget_ms",
        "candidate_certification_reserve_ms",
        "candidate_post_selection_reserve_ms",
        "candidate_eval_stopped_early",
        "candidate_eval_stop_reason",
        "collision_distance_m",
        "inactive_side_hold_active",
        "expanded_for_elbow_singularity",
    )
    out = {
        key: _compact_debug_value(metadata.get(key))
        for key in keys
        if key in metadata
    }
    for key in ("wrist_singularity_preference", "position_singularity_preference"):
        preference = metadata.get(key)
        if not isinstance(preference, Mapping):
            continue
        out[key] = {
            name: _compact_debug_value(preference.get(name), max_items=8)
            for name in (
                "enabled",
                "threshold_deg",
                "evaluated_feasible_count",
                "fully_non_singular_count",
                "minimum_singular_side_count",
                "minimum_singularity_depth_score",
                "fallback_to_singular",
            )
            if name in preference
        }
        selected_preference = preference.get("selected")
        if isinstance(selected_preference, Mapping):
            out[key]["selected"] = {
                name: _compact_debug_value(selected_preference.get(name), max_items=8)
                for name in (
                    "elbow_singular_side_count",
                    "wrist_singular_side_count",
                    "shoulder_singular_side_count",
                    "singular_side_count",
                    "singularity_depth_score",
                    "any_elbow_singular",
                    "any_wrist_singular",
                    "any_shoulder_singular",
                    "any_singular",
                )
                if name in selected_preference
            }
    certificate = metadata.get("candidate_certificate")
    if isinstance(certificate, Mapping):
        out["candidate_certificate"] = {
            name: _compact_debug_value(certificate.get(name), max_items=4)
            for name in (
                "schema",
                "sequence",
                "selected_branch",
                "selection_status",
                "selection_reason",
                "singular_fallback_claimed",
            )
            if name in certificate
        }
    selected = metadata.get("selected_metadata")
    if isinstance(selected, Mapping):
        selected_keys = (
            "step_scale",
            "step_limited",
            "dual_pose_before",
            "dual_pose_after",
            "dual_pose_score_before",
            "dual_pose_score_after",
            "dual_pose_progress",
            "dual_progress_accept_reason",
            "dual_side_position_progress_m",
            "dual_side_orientation_progress_deg",
            "dual_max_position_progress_m",
            "dual_max_orientation_progress_deg",
            "within_pose_tolerance",
        )
        out["selected_metadata"] = {
            key: _compact_debug_value(selected.get(key), max_items=12)
            for key in selected_keys
            if key in selected
        }
        for singularity_key in ("position_singularity", "wrist_singularity"):
            singularity = selected.get(singularity_key)
            if not isinstance(singularity, Mapping):
                continue
            out["selected_metadata"][singularity_key] = {
                name: _compact_debug_value(singularity.get(name), max_items=8)
                for name in (
                    "elbow_singular_side_count",
                    "wrist_singular_side_count",
                    "shoulder_singular_side_count",
                    "singular_side_count",
                    "singularity_depth_score",
                    "any_elbow_singular",
                    "any_wrist_singular",
                    "any_shoulder_singular",
                    "any_singular",
                )
                if name in singularity
            }
    return out


def _command_metadata_compact_debug(metadata: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        return {}
    selected_keys = (
        "collision_avoidance_status",
        "collision_avoidance_fallback_reason",
        "collision_distance_m",
        "ik_branch_candidate",
        "ik_branch_distance_score_deg",
        "ik_solve_iterations",
        "ik_solve_iteration_mode",
        "ik_solve_iteration_max",
        "ik_solve_iteration_min",
        "ik_damping",
        "ik_solve_iteration_reserve_ms",
        "ik_candidate_certification_reserve_ms",
        "ik_solve_iteration_budget_ms",
        "ik_solve_iteration_budget_exhausted",
        "ik_solve_iteration_stop_reason",
        "ik_solve_pose_error",
        "ik_jump_gate_status",
        "ik_jump_gate_reason",
        "ik_jump_gate_release_ready",
        "ik_jump_gate_hold_frames",
        "ik_jump_gate_metrics",
        "joint_limit_margin",
        "target_pose_smoothing",
        "output_qpos_smoothing",
        "target_activity",
        "scene_profile",
        "collision_pair_count",
        "ik_frame_total_ms",
        "ik_time_budget_ms",
        "ik_time_budget_exceeded",
    )
    out = {
        key: _compact_debug_value(metadata.get(key))
        for key in selected_keys
        if key in metadata
    }
    if "execution_step_metadata" in metadata:
        out["execution_step_metadata"] = _execution_step_compact_debug(
            metadata.get("execution_step_metadata")
        )
    if "ik_candidate_selection" in metadata:
        out["ik_candidate_selection"] = _candidate_selection_compact_debug(
            metadata.get("ik_candidate_selection")
        )
    for key in (
        "target_pose_base",
        "target_pose_tcp_base",
        "target_pose_desk",
        "source_target_pose_base",
        "source_target_pose_desk",
        "source_target_pose_operator",
        "solver_output_endpoint_pose_base",
    ):
        if key in metadata:
            out[key] = _pose_compact_debug(metadata.get(key))
    if "solver_output_qpos" in metadata:
        out["solver_output_qpos"] = _qpos_compact_debug(metadata.get("solver_output_qpos"))
    if "ik_debug_profile" in metadata:
        out["ik_debug_profile"] = _ik_profile_compact_debug(metadata.get("ik_debug_profile"))
    return out


def _teleop_command_metadata_compact_debug(command: TeleopCommand | None) -> dict[str, object]:
    if command is None:
        return {}
    metadata = dict(getattr(command, "metadata", {}) or {})
    selected_keys = (
        "collision_avoidance_status",
        "collision_avoidance_fallback_reason",
        "collision_distance_m",
        "ik_branch_candidate",
        "ik_jump_gate_status",
        "target_pose_smoothing",
        "output_qpos_smoothing",
        "target_activity",
        "scene_profile",
        "collision_pair_count",
        "ik_frame_total_ms",
        "ik_time_budget_ms",
        "ik_time_budget_exceeded",
    )
    out = {
        key: _compact_debug_value(metadata.get(key))
        for key in selected_keys
        if key in metadata
    }
    if "ik_debug_profile" in metadata:
        out["ik_debug_profile"] = _ik_profile_compact_debug(metadata.get("ik_debug_profile"))
    if "ik_candidate_selection" in metadata:
        out["ik_candidate_selection"] = _candidate_selection_compact_debug(
            metadata.get("ik_candidate_selection")
        )
    if "temporary_dual_arm_safety" in metadata:
        out["temporary_dual_arm_safety"] = _compact_debug_value(
            metadata.get("temporary_dual_arm_safety"),
            max_items=24,
        )
    return out


def _command_compact_debug_payload(command) -> dict[str, object] | None:
    if command is None:
        return None
    value_pose = (
        _pose_compact_debug(getattr(command, "value", None))
        if getattr(command, "command_type", "") == "cartesian_pose"
        else None
    )
    value_qpos = (
        _qpos_compact_debug(getattr(command, "value", None))
        if getattr(command, "command_type", "") == "joint_vector"
        else None
    )
    metadata = dict(getattr(command, "metadata", {}) or {})
    return {
        "command_type": getattr(command, "command_type", ""),
        "frame_name": getattr(command, "frame_name", None),
        "value_pose": value_pose,
        "value_qpos": value_qpos,
        "solver_output_qpos": _qpos_compact_debug(metadata.get("solver_output_qpos")),
        "metadata": _command_metadata_compact_debug(metadata),
    }


_IMPEDANCE_RECORD_BASE_KEYS = (
    "step",
    "dt_sec",
    "reference",
    "command_pose",
    "command_joint_deg",
    "pre_follow_smoothing_command_pose",
    "pre_follow_smoothing_command_joint_deg",
    "follow_qpos_smoothing",
    "canfd_follow_stream",
    "measured_pose",
    "measured_joint_deg",
)

_IMPEDANCE_RECORD_BACKEND_TARGET_KEYS = (
    "backend_target_sequence",
    "backend_target_timestamp",
    "backend_target_present",
    "backend_target_joint_hint_present",
    "backend_target_dispatched_this_step",
    "backend_last_dispatched_target_sequence",
    "backend_target_snapshot_wall_time",
    "backend_controller_step_start_wall_time",
    "backend_controller_step_end_wall_time",
    "backend_controller_step_wall_duration_sec",
    "backend_target_age_sec_at_snapshot",
    "backend_target_age_sec_at_step_start",
    "backend_target_age_sec_at_step_end",
)

_IMPEDANCE_RECORD_FORCE_KEYS = (
    "force_source",
    "force_readings",
    "selected_force_wrench",
    "active_wrench",
    "tare_corrected_wrench",
    "filtered_wrench",
    "raw_deadbanded_wrench",
    "admittance_wrench",
    "return_activity_wrench",
)

_IMPEDANCE_RECORD_TARGET_OFFSET_KEYS = (
    "target_offset_unclamped",
    "target_offset_after_return",
    "target_offset_after_offset_limit",
    "target_offset",
)

_IMPEDANCE_RECORD_REALIZATION_KEYS = (
    "requested_offset_before_command_realization",
    "admittance_offset_before_antiwindup",
    "final_command_pose_realized_offset",
    "final_command_pose_realization_source",
    "admittance_offset_state_update_enabled",
    "admittance_offset_antiwindup_would_apply",
    "applied_offset_source",
    "admittance_offset_antiwindup_applied",
    "admittance_offset_antiwindup_skip_reason",
    "admittance_offset_antiwindup_guarded_axes",
)

_IMPEDANCE_RECORD_REALIZATION_TAIL_KEYS = (
    "admittance_offset_antiwindup_guard_reason",
    "admittance_offset_realized_before_guard",
    "applied_offset_realization_linear_error_m",
    "applied_offset_realization_angular_error_rad",
)

_IMPEDANCE_RECORD_RETURN_LIMIT_KEYS = (
    "return_applied_axes",
    "return_paused_axes",
    "linear_return_paused",
    "angular_return_paused",
    "linear_offset_limit_applied",
    "angular_offset_limit_applied",
    "linear_velocity_limit_applied",
    "angular_velocity_limit_applied",
    "angular_axis_velocity_limit_applied",
    "linear_return_velocity_limit_applied",
    "angular_return_velocity_limit_applied",
    "linear_step_velocity_limit_m_s",
    "angular_step_velocity_limit_rad_s",
    "angular_axis_velocity_scales",
    "angular_axis_step_velocity_limits_rad_s",
    "force_norm_n",
    "torque_norm_nm",
    "command_joint_step_deg",
    "allowed_command_joint_step_deg",
    "joint_rate_limited",
)

_IMPEDANCE_RECORD_JOINT_LIMIT_PROFILE_KEYS = (
    "joint_velocity_limit_profile",
    "joint_velocity_limit_dt_sec",
)

_IMPEDANCE_RECORD_COMMON_TAIL_KEYS = (
    "joint_velocity_limit_ratio",
    "joint_safety",
    "reference_to_command_linear_m",
    "reference_to_command_angular_rotvec_rad",
    "admittance_target_linear_norm_m",
    "admittance_target_angular_norm_rad",
    "admittance_offset_linear_norm_m",
    "admittance_offset_angular_norm_rad",
    "commanded_offset_linear_norm_m",
    "commanded_offset_angular_norm_rad",
    "offset_to_command_linear_error_m",
    "offset_to_command_angular_error_rad",
    "offset_to_command_linear_fraction",
    "offset_to_command_angular_fraction",
    "target_to_command_linear_error_m",
    "target_to_command_angular_error_rad",
    "target_to_command_linear_fraction",
    "target_to_command_angular_fraction",
    "reference_to_measured_linear_m",
    "reference_to_measured_angular_rotvec_rad",
    "measured_offset_linear_norm_m",
    "measured_offset_angular_norm_rad",
    "command_to_measured_linear_m",
    "command_to_measured_angular_rotvec_rad",
    "command_to_measured_linear_norm_m",
    "command_to_measured_angular_norm_rad",
    "measured_to_command_linear_m",
    "measured_to_command_angular_rotvec_rad",
    "command_to_measured_joint_deg",
    "measured_to_command_joint_deg",
    "measured_velocity_dt_sec",
    "measured_pose_velocity_linear_m_s",
    "measured_pose_velocity_angular_rad_s",
    "measured_joint_velocity_deg_s",
    "tracking_error_closing_linear_m_s",
    "tracking_error_closing_angular_rad_s",
    "tracking_error_closing_joint_deg_s",
    "ik_method",
    "ik_backoff_scale",
    "ik_reference_backoff_scale",
    "ik_command_step_backoff_scale",
    "ik_command_step_candidate_deg",
    "ik_command_step_allowed_deg",
    "ik_command_step_applied_deg",
    "ik_resolver_tolerance_fraction",
    "ik_timing",
    "rm65_controller_timing",
    "rm65_backend_loop_timing",
    "singularity",
    "joint_limit_margin",
)

_IMPEDANCE_RECORD_POSE_TRACKING_KEYS = (
    "pose_tracking_linear_m",
    "pose_tracking_angular_rad",
)

_IMPEDANCE_RECORD_COMPACT_KEYS = (
    _IMPEDANCE_RECORD_BASE_KEYS
    + _IMPEDANCE_RECORD_BACKEND_TARGET_KEYS
    + _IMPEDANCE_RECORD_FORCE_KEYS
    + _IMPEDANCE_RECORD_TARGET_OFFSET_KEYS
    + ("offset", "applied_offset")
    + _IMPEDANCE_RECORD_REALIZATION_KEYS
    + ("admittance_offset_antiwindup_active_axes",)
    + _IMPEDANCE_RECORD_REALIZATION_TAIL_KEYS
    + ("offset_delta_before_velocity_limit", "offset_step_after_velocity_limit")
    + _IMPEDANCE_RECORD_RETURN_LIMIT_KEYS
    + _IMPEDANCE_RECORD_JOINT_LIMIT_PROFILE_KEYS
    + _IMPEDANCE_RECORD_COMMON_TAIL_KEYS
    + _IMPEDANCE_RECORD_POSE_TRACKING_KEYS
)

_IMPEDANCE_RECORD_DEBUG_KEYS = (
    _IMPEDANCE_RECORD_BASE_KEYS
    + _IMPEDANCE_RECORD_FORCE_KEYS
    + ("offset", "applied_offset")
    + _IMPEDANCE_RECORD_TARGET_OFFSET_KEYS
    + ("offset_delta_before_velocity_limit", "offset_step_after_velocity_limit")
    + _IMPEDANCE_RECORD_REALIZATION_KEYS
    + _IMPEDANCE_RECORD_REALIZATION_TAIL_KEYS
    + _IMPEDANCE_RECORD_RETURN_LIMIT_KEYS
    + (
        "joint_rate_limiter_bypassed",
        "joint_rate_limiter_bypass_reason",
        "joint_rate_limiter_synced_to_prelimited_target",
    )
    + _IMPEDANCE_RECORD_JOINT_LIMIT_PROFILE_KEYS
    + (
        "joint_velocity_limit_requested_delta_deg",
        "joint_velocity_limit_delta_deg",
        "joint_velocity_limit_max_step_deg",
    )
    + _IMPEDANCE_RECORD_COMMON_TAIL_KEYS
    + ("ik_metadata",)
    + _IMPEDANCE_RECORD_POSE_TRACKING_KEYS
)


def _impedance_record_compact_debug(record: object) -> dict[str, object] | None:
    if not isinstance(record, Mapping):
        return None
    keys = _IMPEDANCE_RECORD_COMPACT_KEYS
    out = {}
    for key in keys:
        if key not in record:
            continue
        max_items = 32 if key in {"follow_qpos_smoothing", "canfd_follow_stream"} else 12
        out[key] = _compact_debug_value(record.get(key), max_items=max_items)
    if "ik_metadata" in record:
        out["ik_metadata"] = _backend_ik_metadata_compact_debug(record.get("ik_metadata"))
    return out


def _backend_ik_metadata_compact_debug(metadata: object) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        return {}
    selected_keys = (
        "status",
        "method",
        "source",
        "reason",
        "last_code",
        "previous_ik_status",
        "previous_ik_method",
        "previous_ik_source",
        "relaxation_type",
        "scale",
        "direct_joint_hint_reason",
        "direct_joint_hint_linear_offset_m",
        "direct_joint_hint_angular_offset_rad",
        "direct_joint_hint_max_linear_offset_m",
        "direct_joint_hint_max_angular_offset_rad",
        "direct_joint_hint_offset_ignored",
        "direct_joint_hint_force_guard",
        "direct_joint_hint_force_guard_reset_admittance",
        "ik_status",
        "ik_jump_gate_status",
        "ik_jump_gate_reason",
        "ik_solve_total_ms",
        "ik_backend_solve_ms",
        "ik_provider_total_ms",
        "ik_provider_elapsed_ms",
        "ik_min_distance_total_ms",
        "ik_solve_qp_ms",
        "ik_non_provider_ms",
        "singularity_gate_status",
        "target_singularity_code",
        "target_singularity_label",
        "current_singularity_code",
        "current_singularity_label",
        "target_wrist_center_distance_m",
        "current_wrist_center_distance_m",
        "configured_elbow_threshold_deg",
        "configured_wrist_threshold_deg",
        "configured_shoulder_threshold_m",
        "position_wrist_singularity_fallback_requested",
        "position_wrist_singularity_fallback_allowed",
        "position_wrist_singularity_fallback",
        "position_elbow_singularity_fallback_requested",
        "position_elbow_singularity_fallback_allowed",
        "position_shoulder_singularity_fallback_requested",
        "position_shoulder_singularity_fallback_allowed",
        "position_singularity_fallback_allowed",
    )
    out = {
        key: _compact_debug_value(metadata.get(key))
        for key in selected_keys
        if key in metadata
    }
    if "ik_debug_profile" in metadata:
        out["ik_debug_profile"] = _ik_profile_compact_debug(metadata.get("ik_debug_profile"))
    return out


def _background_streaming_compact_debug(state: object) -> dict[str, object]:
    if not isinstance(state, Mapping):
        return {}
    keys = (
        "enabled",
        "hz",
        "thread_alive",
        "target_active",
        "target_source",
        "target_sequence",
        "target_age_sec",
        "target_max_age_sec",
        "target_stale",
        "continuous_hold",
        "quick_stop_active",
        "quick_stop_reason",
        "position_command_planner",
        "trajectory_policy_mode",
        "trajectory_policy_speed_scale",
        "trajectory_policy_reason",
        "command_control_dt_sec",
        "tick_sequence",
        "send_count",
        "accepted_send_count",
        "failed_send_count",
        "deadline_miss_count",
        "actual_hz",
        "last_call_ms",
        "last_deadline_lag_ms",
        "last_sent_step_deg",
        "stale_count",
        "fault",
        "fault_count",
        "healthy",
        "health_reasons",
    )
    out = {
        key: _compact_debug_value(state.get(key), max_items=8)
        for key in keys
        if key in state
    }
    trajectory_debug = state.get("last_trajectory_debug")
    if isinstance(trajectory_debug, Mapping):
        out["last_trajectory_debug"] = {
            key: _compact_debug_value(trajectory_debug.get(key), max_items=8)
            for key in (
                "planner_mode",
                "trajectory_mode",
                "trajectory_reason",
                "trajectory_quick_stop",
                "trajectory_speed_scale",
                "trajectory_qd_command_rad_s",
                "trajectory_qdd_command_rad_s2",
                "target_velocity_filter_tau_sec",
            )
            if key in trajectory_debug
        }
    return out


def _backend_compact_debug_payload(state: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(state, Mapping):
        return {}
    out: dict[str, object] = {
        "hardware_result_code": state.get("hardware_result_code"),
        "hardware_result_text": state.get("hardware_result_text"),
        "send_result_code": state.get("send_result_code"),
        "send_result_text": state.get("send_result_text"),
        "effective_control_dt_sec": state.get("effective_control_dt_sec"),
        "effective_step_scale": state.get("effective_step_scale"),
        "background_streaming": _background_streaming_compact_debug(
            state.get("background_streaming")
        ),
        "canfd_follow": state.get("canfd_follow"),
        "canfd_trajectory_mode": state.get("canfd_trajectory_mode"),
        "canfd_radio": state.get("canfd_radio"),
        "impedance_enabled": state.get("impedance_enabled"),
        "impedance_loop_state": state.get("impedance_loop_state"),
        "active_tool_name": state.get("active_tool_name"),
        "active_tool_profile_path": state.get("active_tool_profile_path"),
        "active_tool_frame": _compact_debug_value(state.get("active_tool_frame")),
        "profile_tcp": _compact_debug_value(state.get("profile_tcp")),
        "active_impedance_config": _compact_debug_value(
            state.get("active_impedance_config"),
            max_items=80,
        ),
        "last_command_controlled_frame": state.get("last_command_controlled_frame"),
        "last_command_timestamp": state.get("last_command_timestamp"),
        "last_command_sequence": state.get("last_command_sequence"),
        "command_sequence": state.get("command_sequence"),
        "latest_target_tcp_pose6": _compact_debug_value(state.get("latest_target_tcp_pose6")),
        "latest_target_joint_deg": _compact_debug_value(state.get("latest_target_joint_deg")),
        "latest_target_sequence": state.get("latest_target_sequence"),
        "current_qpos": _qpos_compact_debug(state.get("current_qpos")),
        "last_target_qpos": _qpos_compact_debug(state.get("last_target_qpos")),
        "current_tcp_pose": _pose_compact_debug(state.get("current_tcp_pose")),
        "last_target_tcp_pose": _pose_compact_debug(state.get("last_target_tcp_pose")),
        "pose_tracking_error": _pose_delta_compact_debug(
            state.get("current_tcp_pose"),
            state.get("last_target_tcp_pose"),
        ),
        "ik_metadata": _backend_ik_metadata_compact_debug(state.get("ik_metadata")),
        "singularity_metadata": _backend_ik_metadata_compact_debug(
            state.get("singularity_metadata")
        ),
        "last_impedance_record": _impedance_record_compact_debug(
            state.get("last_impedance_record")
        ),
        "recent_impedance_records": [
            item
            for item in (
                _impedance_record_compact_debug(record)
                for record in list(state.get("recent_impedance_records") or [])[-4:]
            )
            if item
        ],
        "last_send_command_timing": _compact_debug_value(state.get("last_send_command_timing")),
        "last_impedance_loop_timing": _compact_debug_value(state.get("last_impedance_loop_timing")),
    }
    return {key: value for key, value in out.items() if value is not None}


def _observation_pose_compact_debug(step_result) -> dict[str, object]:
    channels = getattr(step_result, "channels", {}) or {}
    out: dict[str, object] = {}
    for channel_name in ("left_wrist_pose", "right_wrist_pose"):
        channel = channels.get(channel_name)
        if channel is None:
            continue
        out[channel_name] = {
            "valid": bool(getattr(channel, "valid", False)),
            "timestamp": float(getattr(channel, "timestamp", 0.0)),
            "frame_name": getattr(channel, "frame_name", None),
            "source_name": getattr(channel, "source_name", None),
            "pose": _pose_compact_debug(getattr(channel, "value", None)),
        }
    return out


def _command_debug_payload(command) -> dict[str, object] | None:
    if command is None:
        return None
    metadata = dict(getattr(command, "metadata", {}) or {})
    selected_metadata_keys = (
        "collision_avoidance_status",
        "collision_avoidance_fallback_reason",
        "collision_distance_m",
        "ik_branch_candidate",
        "ik_branch_distance_score_deg",
        "ik_solve_iterations",
        "ik_solve_iteration_mode",
        "ik_solve_iteration_max",
        "ik_solve_iteration_min",
        "ik_damping",
        "ik_solve_iteration_reserve_ms",
        "ik_candidate_certification_reserve_ms",
        "ik_solve_iteration_budget_ms",
        "ik_solve_iteration_budget_exhausted",
        "ik_solve_iteration_stop_reason",
        "ik_solve_pose_error",
        "ik_jump_gate_status",
        "ik_jump_gate_reason",
        "ik_jump_gate_release_ready",
        "ik_jump_gate_hold_frames",
        "ik_jump_gate_metrics",
        "joint_limit_margin",
        "ik_candidate_selection",
        "target_pose_smoothing",
        "output_qpos_smoothing",
        "target_activity",
        "scene_profile",
        "collision_pair_count",
        "ik_frame_total_ms",
        "ik_time_budget_ms",
        "ik_time_budget_exceeded",
        "execution_step_metadata",
        "target_pose_base",
        "target_pose_tcp_base",
        "target_pose_desk",
        "source_target_pose_base",
        "source_target_pose_desk",
        "source_target_pose_operator",
        "solver_output_endpoint_pose_base",
        "solver_output_qpos",
        "ik_debug_profile",
    )
    selected_metadata = {
        key: metadata.get(key)
        for key in selected_metadata_keys
        if key in metadata
    }
    solver_qpos = _qpos_debug(metadata.get("solver_output_qpos"))
    value_pose = (
        _pose_matrix_debug(getattr(command, "value", None))
        if getattr(command, "command_type", "") == "cartesian_pose"
        else None
    )
    value_qpos = (
        _qpos_debug(getattr(command, "value", None))
        if getattr(command, "command_type", "") == "joint_vector"
        else None
    )
    return {
        "command_type": getattr(command, "command_type", ""),
        "frame_name": getattr(command, "frame_name", None),
        "value_pose": value_pose,
        "value_qpos": value_qpos,
        "solver_output_qpos": solver_qpos,
        "metadata": selected_metadata,
    }


def _teleop_command_metadata_debug(command: TeleopCommand | None) -> dict[str, object]:
    if command is None:
        return {}
    metadata = dict(getattr(command, "metadata", {}) or {})
    selected_keys = (
        "collision_avoidance_status",
        "collision_avoidance_fallback_reason",
        "collision_distance_m",
        "ik_branch_candidate",
        "ik_jump_gate_status",
        "ik_candidate_selection",
        "target_pose_smoothing",
        "output_qpos_smoothing",
        "target_activity",
        "scene_profile",
        "collision_pair_count",
        "ik_frame_total_ms",
        "ik_time_budget_ms",
        "ik_time_budget_exceeded",
        "ik_solve_iteration_mode",
        "ik_solve_iteration_max",
        "ik_solve_iteration_min",
        "ik_damping",
        "ik_solve_iteration_reserve_ms",
        "ik_candidate_certification_reserve_ms",
        "ik_solve_iteration_budget_ms",
        "ik_solve_iteration_budget_exhausted",
        "ik_solve_iteration_stop_reason",
        "ik_debug_profile",
        "temporary_dual_arm_safety",
    )
    return {key: metadata.get(key) for key in selected_keys if key in metadata}


def _impedance_record_debug(record: object) -> dict[str, object] | None:
    if not isinstance(record, Mapping):
        return None
    keys = _IMPEDANCE_RECORD_DEBUG_KEYS
    return {key: record.get(key) for key in keys if key in record}


def _backend_debug_payload(state: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(state, Mapping):
        return {}
    return {
        "hardware_result_code": state.get("hardware_result_code"),
        "hardware_result_text": state.get("hardware_result_text"),
        "send_result_code": state.get("send_result_code"),
        "send_result_text": state.get("send_result_text"),
        "effective_control_dt_sec": state.get("effective_control_dt_sec"),
        "effective_step_scale": state.get("effective_step_scale"),
        "background_streaming": state.get("background_streaming"),
        "canfd_follow": state.get("canfd_follow"),
        "canfd_trajectory_mode": state.get("canfd_trajectory_mode"),
        "canfd_radio": state.get("canfd_radio"),
        "impedance_enabled": state.get("impedance_enabled"),
        "impedance_loop_state": state.get("impedance_loop_state"),
        "active_tool_name": state.get("active_tool_name"),
        "active_tool_profile_path": state.get("active_tool_profile_path"),
        "active_tool_frame": state.get("active_tool_frame"),
        "profile_tcp": state.get("profile_tcp"),
        "active_impedance_config": state.get("active_impedance_config"),
        "last_command_controlled_frame": state.get("last_command_controlled_frame"),
        "last_command": _command_debug_payload(state.get("last_command")),
        "last_command_timestamp": state.get("last_command_timestamp"),
        "latest_target_tcp_pose6": state.get("latest_target_tcp_pose6"),
        "latest_target_joint_deg": state.get("latest_target_joint_deg"),
        "current_qpos": _qpos_debug(state.get("current_qpos")),
        "last_target_qpos": _qpos_debug(state.get("last_target_qpos")),
        "current_tcp_pose": _pose_matrix_debug(state.get("current_tcp_pose")),
        "last_target_tcp_pose": _pose_matrix_debug(state.get("last_target_tcp_pose")),
        "ik_metadata": state.get("ik_metadata"),
        "last_impedance_record": _impedance_record_debug(state.get("last_impedance_record")),
        "recent_impedance_records": [
            item
            for item in (
                _impedance_record_debug(record)
                for record in list(state.get("recent_impedance_records") or [])[-4:]
            )
            if item
        ],
        "last_send_command_timing": state.get("last_send_command_timing"),
        "last_impedance_loop_timing": state.get("last_impedance_loop_timing"),
    }


def _observation_pose_debug(step_result) -> dict[str, object]:
    channels = getattr(step_result, "channels", {}) or {}
    out: dict[str, object] = {}
    for channel_name in ("left_wrist_pose", "right_wrist_pose"):
        channel = channels.get(channel_name)
        if channel is None:
            continue
        out[channel_name] = {
            "valid": bool(getattr(channel, "valid", False)),
            "timestamp": float(getattr(channel, "timestamp", 0.0)),
            "frame_name": getattr(channel, "frame_name", None),
            "source_name": getattr(channel, "source_name", None),
            "pose": _pose_matrix_debug(getattr(channel, "value", None)),
        }
    return out

__all__ = [
    "_IK_PROFILE_COMPACT_KEYS",
    "_ik_profile_compact_debug",
    "_execution_step_compact_debug",
    "_candidate_selection_compact_debug",
    "_command_metadata_compact_debug",
    "_teleop_command_metadata_compact_debug",
    "_command_compact_debug_payload",
    "_IMPEDANCE_RECORD_BASE_KEYS",
    "_IMPEDANCE_RECORD_BACKEND_TARGET_KEYS",
    "_IMPEDANCE_RECORD_FORCE_KEYS",
    "_IMPEDANCE_RECORD_TARGET_OFFSET_KEYS",
    "_IMPEDANCE_RECORD_REALIZATION_KEYS",
    "_IMPEDANCE_RECORD_REALIZATION_TAIL_KEYS",
    "_IMPEDANCE_RECORD_RETURN_LIMIT_KEYS",
    "_IMPEDANCE_RECORD_JOINT_LIMIT_PROFILE_KEYS",
    "_IMPEDANCE_RECORD_COMMON_TAIL_KEYS",
    "_IMPEDANCE_RECORD_POSE_TRACKING_KEYS",
    "_IMPEDANCE_RECORD_COMPACT_KEYS",
    "_IMPEDANCE_RECORD_DEBUG_KEYS",
    "_impedance_record_compact_debug",
    "_backend_ik_metadata_compact_debug",
    "_backend_compact_debug_payload",
    "_observation_pose_compact_debug",
    "_command_debug_payload",
    "_teleop_command_metadata_debug",
    "_impedance_record_debug",
    "_backend_debug_payload",
    "_observation_pose_debug",
]
