from __future__ import annotations

import pytest

from robot_motion.rm65 import (
    RM65_POSE_LIFT_PROFILE_CHOICES,
    RM65_POSE_LIFT_V1_BASELINE,
    RM65_POSE_LIFT_V1_BASELINE_NAME,
    RM65_POSE_LIFT_V2_CONTINUOUS,
    RM65_POSE_LIFT_V2_CONTINUOUS_NAME,
    RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS,
    RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS_NAME,
    RM65_POSE_LIFT_V4_QP_CONSTRAINED,
    RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
    RM65_POSE_LIFT_V5_SIGMA_ESCAPE,
    RM65_POSE_LIFT_V5_SIGMA_ESCAPE_NAME,
    RM65_POSE_LIFT_V6_STRUCTURAL_BRANCH,
    RM65_POSE_LIFT_V6_STRUCTURAL_BRANCH_NAME,
    RM65_POSE_LIFT_V7_STRUCTURAL_ESCAPE_NAME,
    RM65_POSE_LIFT_V8_NONLINEAR_BEST_CANDIDATE,
    RM65_POSE_LIFT_V8_NONLINEAR_BEST_CANDIDATE_NAME,
    RM65_POSE_LIFT_V9_PREDICTIVE_BRANCH_BARRIER,
    RM65_POSE_LIFT_V9_PREDICTIVE_BRANCH_BARRIER_NAME,
    RM65_POSE_LIFT_V10_DIRECTIONAL_FEASIBILITY_GUARD,
    RM65_POSE_LIFT_V10_DIRECTIONAL_FEASIBILITY_GUARD_NAME,
    RM65PositionCollisionPlannerConfig,
    build_rm65_pose_lift_ik_tuning,
    resolve_rm65_pose_lift_profile,
)


def test_pose_lift_baseline_is_explicit_noop_rollback_profile() -> None:
    profile = resolve_rm65_pose_lift_profile(RM65_POSE_LIFT_V1_BASELINE_NAME)

    assert profile is RM65_POSE_LIFT_V1_BASELINE
    assert profile.equivalent_joint_unwrap is False
    assert profile.wrist_coaxial_canonicalization is False
    assert profile.to_metadata()["fingerprint"] == profile.fingerprint
    assert len(profile.fingerprint) == 64


def test_pose_lift_v2_is_explicit_opt_in_and_profile_names_are_stable() -> None:
    profile = resolve_rm65_pose_lift_profile(RM65_POSE_LIFT_V2_CONTINUOUS_NAME)

    assert profile is RM65_POSE_LIFT_V2_CONTINUOUS
    assert profile.equivalent_joint_unwrap is True
    assert profile.wrist_coaxial_canonicalization is True
    assert RM65_POSE_LIFT_PROFILE_CHOICES == (
        RM65_POSE_LIFT_V1_BASELINE_NAME,
        RM65_POSE_LIFT_V2_CONTINUOUS_NAME,
        RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS_NAME,
        RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
        RM65_POSE_LIFT_V5_SIGMA_ESCAPE_NAME,
        RM65_POSE_LIFT_V6_STRUCTURAL_BRANCH_NAME,
        RM65_POSE_LIFT_V7_STRUCTURAL_ESCAPE_NAME,
        RM65_POSE_LIFT_V8_NONLINEAR_BEST_CANDIDATE_NAME,
        RM65_POSE_LIFT_V9_PREDICTIVE_BRANCH_BARRIER_NAME,
        RM65_POSE_LIFT_V10_DIRECTIONAL_FEASIBILITY_GUARD_NAME,
    )
    with pytest.raises(ValueError, match="unknown RM65 pose-lift profile"):
        resolve_rm65_pose_lift_profile("missing")


def test_pose_lift_v3_adds_bounded_search_without_mutating_frozen_v2_metadata() -> None:
    profile = resolve_rm65_pose_lift_profile(
        RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS_NAME
    )

    assert profile is RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS
    assert profile.wrist_bounded_local_search is True
    assert profile.wrist_soft_threshold_deg == 4.0
    assert profile.wrist_hard_threshold_deg == 3.0
    assert profile.ik_posture_cost == 0.0
    assert RM65_POSE_LIFT_V2_CONTINUOUS.ik_posture_cost is None
    assert RM65PositionCollisionPlannerConfig(
        pose_lift_profile=RM65_POSE_LIFT_V1_BASELINE_NAME
    ).posture_cost == pytest.approx(0.1)
    assert RM65PositionCollisionPlannerConfig(
        pose_lift_profile=RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS_NAME
    ).posture_cost == pytest.approx(0.0)
    assert RM65PositionCollisionPlannerConfig(
        pose_lift_profile=RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS_NAME,
        posture_cost=0.2,
    ).posture_cost == pytest.approx(0.2)
    assert profile.wrist_local_search_scales == (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625)
    assert "wrist_bounded_local_search" in profile.semantic_payload()
    assert "wrist_bounded_local_search" not in (
        RM65_POSE_LIFT_V2_CONTINUOUS.semantic_payload()
    )


def test_pose_lift_v4_is_hardware_default_candidate_with_explicit_v1_rollback() -> None:
    profile = resolve_rm65_pose_lift_profile(
        RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME
    )

    assert profile is RM65_POSE_LIFT_V4_QP_CONSTRAINED
    assert profile.qp_substeps_per_reference == 0
    assert profile.qp_velocity_limit is True
    assert profile.qp_acceleration_limit is True
    assert profile.hardware_dispatch_allowed is True
    assert "validation_scope" not in profile.semantic_payload()
    assert profile.wrist_bounded_local_search is True
    assert RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS.hardware_dispatch_allowed is True
    assert RM65_POSE_LIFT_V3_BOUNDED_CONTINUOUS.qp_substeps_per_reference == 0
    assert RM65PositionCollisionPlannerConfig(
        pose_lift_profile=RM65_POSE_LIFT_V4_QP_CONSTRAINED_NAME,
        solve_min_iterations=2,
        solve_max_iterations=4,
    ).effective_solve_max_iterations == 4


def test_pose_lift_v5_sigma_escape_is_explicit_offline_only_candidate() -> None:
    profile = resolve_rm65_pose_lift_profile(RM65_POSE_LIFT_V5_SIGMA_ESCAPE_NAME)

    assert profile is RM65_POSE_LIFT_V5_SIGMA_ESCAPE
    assert profile.qp_sigma_escape_cost == pytest.approx(0.15)
    assert profile.qp_sigma_escape_guard == pytest.approx(0.04)
    assert profile.qp_sigma_escape_rate_per_sec == pytest.approx(0.03)
    assert profile.hardware_dispatch_allowed is False
    assert profile.semantic_payload()["validation_scope"] == "offline_or_dry_run_only"
    assert "qp_sigma_escape_cost" not in (
        RM65_POSE_LIFT_V4_QP_CONSTRAINED.semantic_payload()
    )


def test_pose_lift_v6_structural_branch_is_explicit_offline_only_candidate() -> None:
    profile = resolve_rm65_pose_lift_profile(
        RM65_POSE_LIFT_V6_STRUCTURAL_BRANCH_NAME
    )

    assert profile is RM65_POSE_LIFT_V6_STRUCTURAL_BRANCH
    assert profile.qp_structural_branch_margin_deg == pytest.approx(0.75)
    assert profile.qp_structural_branch_guard_sigma == pytest.approx(0.04)
    assert profile.qp_structural_branch_release_sigma == pytest.approx(0.05)
    assert profile.hardware_dispatch_allowed is False
    assert profile.semantic_payload()["validation_scope"] == "offline_or_dry_run_only"


def test_pose_lift_v8_nonlinear_best_candidate_is_offline_and_rollback_safe() -> None:
    profile = resolve_rm65_pose_lift_profile(
        RM65_POSE_LIFT_V8_NONLINEAR_BEST_CANDIDATE_NAME
    )

    assert profile is RM65_POSE_LIFT_V8_NONLINEAR_BEST_CANDIDATE
    assert profile.qp_nonlinear_best_so_far is True
    assert profile.qp_nonlinear_best_min_iteration == 2
    assert profile.qp_nonlinear_best_guard_sigma == pytest.approx(0.04)
    assert profile.hardware_dispatch_allowed is False
    assert profile.semantic_payload()["validation_scope"] == "offline_or_dry_run_only"
    assert RM65_POSE_LIFT_V4_QP_CONSTRAINED.qp_nonlinear_best_so_far is False


def test_pose_lift_v9_predictive_barrier_is_offline_and_v6_is_unchanged() -> None:
    profile = resolve_rm65_pose_lift_profile(
        RM65_POSE_LIFT_V9_PREDICTIVE_BRANCH_BARRIER_NAME
    )

    assert profile is RM65_POSE_LIFT_V9_PREDICTIVE_BRANCH_BARRIER
    assert (
        profile.qp_structural_branch_constraint_mode
        == "hybrid_deep_barrier"
    )
    assert profile.qp_structural_branch_barrier_gain == pytest.approx(6.0)
    assert profile.qp_structural_branch_barrier_activation_sigma == pytest.approx(
        0.01
    )
    assert profile.qp_structural_branch_prediction_horizon_sec == pytest.approx(
        0.15
    )
    assert profile.hardware_dispatch_allowed is False
    assert profile.semantic_payload()["validation_scope"] == "offline_or_dry_run_only"
    assert (
        RM65_POSE_LIFT_V6_STRUCTURAL_BRANCH.qp_structural_branch_constraint_mode
        == "hard_margin"
    )
    assert RM65_POSE_LIFT_V6_STRUCTURAL_BRANCH.qp_structural_branch_barrier_gain == 0.0


def test_pose_lift_v10_directional_guard_is_offline_and_v4_is_unchanged() -> None:
    profile = resolve_rm65_pose_lift_profile(
        RM65_POSE_LIFT_V10_DIRECTIONAL_FEASIBILITY_GUARD_NAME
    )

    assert profile is RM65_POSE_LIFT_V10_DIRECTIONAL_FEASIBILITY_GUARD
    assert profile.qp_structural_branch_constraint_mode == "barrier"
    assert profile.qp_structural_branch_margin_deg == pytest.approx(0.25)
    assert profile.directional_task_guard_max_utilization == pytest.approx(0.85)
    assert profile.directional_task_guard_enter_utilization == pytest.approx(1.0)
    assert profile.directional_task_guard_release_utilization == pytest.approx(
        0.75
    )
    assert profile.directional_task_guard_leaving_multiplier == pytest.approx(
        2.0
    )
    assert profile.hardware_dispatch_allowed is False
    payload = profile.semantic_payload()
    assert payload["validation_scope"] == "offline_or_dry_run_only"
    assert payload["directional_task_guard_target_shaping"] == (
        "local_se3_log_singular_direction_projection"
    )
    assert RM65_POSE_LIFT_V4_QP_CONSTRAINED.directional_task_guard_max_utilization == 0.0


def test_pose_lift_ik_objective_is_shared_and_profile_identified() -> None:
    tuning = build_rm65_pose_lift_ik_tuning(
        position_control_profile="rm65_bimanual_position_v1",
        pose_lift_profile=RM65_POSE_LIFT_V2_CONTINUOUS_NAME,
        orientation_weight=0.35,
    )

    assert tuning.ik_objective_mode == (
        "rm65_pose_lift_v2_continuous.ik_objective"
    )
    objective = tuning.objective_weights()
    assert objective.dual_arm_position_cost == pytest.approx(1.0)
    assert objective.dual_arm_orientation_cost == pytest.approx(0.35)
    assert build_rm65_pose_lift_ik_tuning(
        position_control_profile="legacy",
        pose_lift_profile=RM65_POSE_LIFT_V2_CONTINUOUS_NAME,
        orientation_weight=0.35,
    ) is None
