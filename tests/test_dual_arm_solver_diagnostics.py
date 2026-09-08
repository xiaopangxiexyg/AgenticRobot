from __future__ import annotations

from robot_motion.rm65.dual_arm_target_solver import _side_result_metadata


def test_side_result_metadata_projects_each_side_without_sharing_mutable_maps() -> None:
    scalar_keys = (
        "controller",
        "scene_profile",
        "collision_pair_count",
        "collision_avoidance_mode",
        "collision_avoidance_status",
        "collision_avoidance_fallback_reason",
        "collision_distance_m",
        "ik_framework",
        "ik_objective_mode",
        "ik_branch_candidate",
        "ik_branch_distance_score_deg",
        "ik_solve_iterations",
        "ik_solve_step_mode",
        "ik_solve_iteration_mode",
        "ik_solve_iteration_max",
        "ik_solve_iteration_min",
        "ik_damping",
        "ik_solve_iteration_reserve_ms",
        "ik_candidate_certification_reserve_ms",
        "ik_solve_iteration_budget_ms",
        "ik_solve_iteration_budget_exhausted",
        "ik_solve_iteration_stop_reason",
        "ik_jump_gate_status",
        "ik_jump_gate_reason",
        "ik_jump_gate_release_ready",
        "ik_jump_gate_hold_frames",
        "ik_frame_total_ms",
        "ik_time_budget_ms",
        "ik_time_budget_exceeded",
    )
    output: dict[str, object] = {key: f"shared:{key}" for key in scalar_keys}
    output.update(
        {
            "source_target_pose_base": {"left": "left_base", "right": "right_base"},
            "source_target_pose_desk": {"left": "left_desk", "right": "right_desk"},
            "source_target_pose_operator": {
                "left": "left_operator",
                "right": "right_operator",
            },
            "ik_wrist_coaxial_continuity": {"enabled": True},
            "ik_directional_singularity": {
                "sides": {"left": {"axis": 1}, "right": {"axis": 2}}
            },
            "ik_admitted_velocity_state": {
                "enabled": True,
                "source": "admitted_qref",
                "dt_s": 0.1,
                "left_qdot_rad_s": [1.0],
                "right_qdot_rad_s": [2.0],
                "left_max_abs_qdot_rad_s": 1.0,
                "right_max_abs_qdot_rad_s": 2.0,
            },
            "ik_solve_pose_error": {"left": {"error": 1}, "right": {"error": 2}},
            "ik_current_target_hold": {"enabled": False},
            "ik_jump_gate_metrics": {"left": {"jump": 1}, "right": {"jump": 2}},
            "ik_candidate_selection": {"selected": "candidate"},
            "joint_limit_margin": {"left": {"margin": 1}, "right": {"margin": 2}},
            "execution_step_metadata": {"enabled": True},
            "output_qpos_smoothing": {"left": {"alpha": 0.1}, "right": {"alpha": 0.2}},
            "target_pose_smoothing": {"left": {"tau": 1}, "right": {"tau": 2}},
            "target_activity": {"left": {"active": True}, "right": {"active": False}},
        }
    )

    left = _side_result_metadata(
        output,
        side="left",
        source_command_type="cartesian_pose",
        inactive_jitter_hold=True,
    )
    right = _side_result_metadata(
        output,
        side="right",
        source_command_type=None,
    )

    assert left["source_target_pose_base"] == "left_base"
    assert right["source_target_pose_base"] == "right_base"
    assert left["ik_directional_singularity"] == {"axis": 1}
    assert right["ik_admitted_velocity_state"]["qdot_rad_s"] == [2.0]
    assert left["inactive_jitter_hold"] is True
    assert "inactive_jitter_hold" not in right

    left["ik_candidate_selection"]["selected"] = "mutated"
    assert output["ik_candidate_selection"] == {"selected": "candidate"}
    assert right["ik_candidate_selection"] == {"selected": "candidate"}
