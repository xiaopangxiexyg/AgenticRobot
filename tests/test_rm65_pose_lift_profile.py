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
