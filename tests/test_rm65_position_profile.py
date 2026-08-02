from __future__ import annotations

import numpy as np
import pytest

from robot_motion.joint_limits import load_joint_velocity_limit_profiles
from robot_motion.rm65 import (
    RM65_BIMANUAL_POSITION_PROFILE_SCHEMA,
    RM65_BIMANUAL_POSITION_V1,
    RM65_BIMANUAL_POSITION_V1_NAME,
    RM65_BIMANUAL_POSITION_V1_QREF_PROFILE,
    RM65_BIMANUAL_POSITION_V2_RESPONSIVE,
    RM65_BIMANUAL_POSITION_V2_RESPONSIVE_NAME,
    RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V1,
    RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V1_NAME,
    RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V2,
    RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V2_NAME,
)


def test_formal_position_profile_freezes_validated_pipeline() -> None:
    profile = RM65_BIMANUAL_POSITION_V1

    assert profile.name == RM65_BIMANUAL_POSITION_V1_NAME
    assert profile.schema == RM65_BIMANUAL_POSITION_PROFILE_SCHEMA
    assert profile.main_action_contract == "accepted_qref_fk_pose"
    rates = profile.rates.to_metadata()
    assert rates["sample_hz"] == 15.0
    assert rates["reference_hz"] == 60.0
    assert rates["ik_hz"] == 60.0
    assert rates["command_hz"] == 120.0
    assert profile.ik_execution.orientation_weight == pytest.approx(0.35)
    assert profile.ik_execution.solve_min_iterations == 2
    assert profile.ik_execution.solve_max_iterations == 4
    assert profile.ik_execution.iteration_reserve_ms == pytest.approx(5.0)
    assert profile.ik_execution.candidate_certification_reserve_ms == pytest.approx(3.5)
    assert profile.ik_execution.prefer_non_shoulder_singular_branch is True
    assert profile.ik_execution.require_singularity_candidate_certificate is True
    assert profile.command_execution.target_velocity_filter_tau_sec == pytest.approx(0.03)
    assert profile.command_execution.max_command_step_deg == pytest.approx(0.675)
    assert profile.command_execution.reference_max_age_sec == pytest.approx(0.2)
    assert profile.command_execution.canfd_follow is True
    assert profile.command_execution.canfd_trajectory_mode == 0
    assert profile.command_execution.canfd_radio == 0


def test_main_action_boundary_forbids_reapplying_source_shaping() -> None:
    metadata = RM65_BIMANUAL_POSITION_V1.to_metadata()

    assert metadata["source_shaping"]["application_scope"] == (
        "live_human_source_before_main_action"
    )
    assert metadata["source_shaping"]["replay_or_deploy_reapply"] is False
    assert metadata["action_execution"]["reapply_source_shaping"] is False
    assert len(metadata["sha256"]) == 64


def test_responsive_v2_is_opt_in_and_v1_remains_rollback_profile() -> None:
    v1_source = RM65_BIMANUAL_POSITION_V1.source_shaping
    v2 = RM65_BIMANUAL_POSITION_V2_RESPONSIVE
    v2_source = v2.source_shaping

    assert v2.name == RM65_BIMANUAL_POSITION_V2_RESPONSIVE_NAME
    assert v1_source.post_ik_qpos_normal_fast_path is False
    assert v1_source.wrist_q5_hysteresis_requires_escape_latch is False
    assert set(v1_source.to_metadata()) == {
        "inactive_enter_threshold_scale",
        "post_ik_qpos_smoothing_alpha",
        "post_ik_qpos_smoothing_fast_alpha",
        "post_ik_qpos_smoothing_adaptive",
        "application_scope",
        "replay_or_deploy_reapply",
    }
    assert v2_source.post_ik_qpos_normal_fast_path is True
    assert v2_source.post_ik_qpos_normal_fast_path_alpha == pytest.approx(0.70)
    assert v2_source.post_ik_qpos_normal_fast_path_guard_deg == pytest.approx(4.0)
    assert v2_source.post_ik_qpos_normal_fast_path_quality_grace_sec == pytest.approx(
        0.05
    )
    assert v2_source.wrist_q5_hysteresis_requires_escape_latch is True
    assert v2_source.to_metadata()["post_ik_qpos_normal_fast_path"] is True
    assert v2_source.to_metadata()[
        "post_ik_qpos_normal_fast_path_quality_grace_sec"
    ] == pytest.approx(0.05)
    assert v2.command_execution == RM65_BIMANUAL_POSITION_V1.command_execution
    assert v2.ik_execution == RM65_BIMANUAL_POSITION_V1.ik_execution
    assert v2.to_metadata()["sha256"] != RM65_BIMANUAL_POSITION_V1.to_metadata()[
        "sha256"
    ]


def test_sample_aligned_jerk_profile_is_execution_only_and_reversible() -> None:
    profile = RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V1

    assert profile.name == RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V1_NAME
    assert profile.source_shaping == RM65_BIMANUAL_POSITION_V1.source_shaping
    assert profile.ik_execution == RM65_BIMANUAL_POSITION_V1.ik_execution
    assert profile.rates == RM65_BIMANUAL_POSITION_V1.rates
    assert profile.command_execution.planner == "sample_aligned_jerk_limited"
    assert profile.command_execution.velocity_ramp_sec == pytest.approx(0.05)
    assert profile.command_execution.target_velocity_filter_tau_sec == pytest.approx(
        0.0
    )
    assert profile.command_execution.follow_qpos_alpha == pytest.approx(1.0)
    assert profile.to_metadata()["sha256"] != RM65_BIMANUAL_POSITION_V1.to_metadata()[
        "sha256"
    ]


def test_sample_aligned_jerk_v2_is_atomic_and_v1_remains_exact_rollback() -> None:
    v1 = RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V1
    v2 = RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V2

    assert v2.name == RM65_BIMANUAL_POSITION_SAMPLE_ALIGNED_JERK_V2_NAME
    assert v2.rates == v1.rates
    assert v2.ik_execution == v1.ik_execution
    assert v1.command_execution.sample_aligned_scale_degraded_target is True
    assert v1.command_execution.sample_aligned_jerk_regularization == pytest.approx(
        1e-9
    )
    assert v1.command_execution.sample_aligned_jerk_delta_regularization == 0.0
    assert v1.source_shaping.inactive_execution_release_translation_static_m == (
        pytest.approx(0.03)
    )

    source = v2.source_shaping
    command = v2.command_execution
    assert source.post_ik_qpos_normal_fast_path is True
    assert source.post_ik_qpos_normal_fast_path_alpha == pytest.approx(0.70)
    assert source.inactive_execution_release_translation_static_m == pytest.approx(
        0.0015
    )
    assert source.inactive_execution_release_translation_full_m == pytest.approx(
        0.012
    )
    assert source.inactive_execution_release_rotation_static_deg == pytest.approx(
        0.25
    )
    assert source.inactive_execution_release_rotation_full_deg == pytest.approx(2.0)
    assert command.sample_aligned_scale_degraded_target is False
    assert command.sample_aligned_jerk_regularization == pytest.approx(1e-6)
    assert command.sample_aligned_jerk_delta_regularization == pytest.approx(1e-5)
    assert command.sample_aligned_terminal_velocity_regularization == 0.0
    assert command.sample_aligned_terminal_acceleration_regularization == 0.0
    assert v2.to_metadata()["sha256"] not in {
        v1.to_metadata()["sha256"],
        RM65_BIMANUAL_POSITION_V1.to_metadata()["sha256"],
    }


def test_production_qref_profile_preserves_validated_numeric_envelope() -> None:
    profiles = load_joint_velocity_limit_profiles()

    formal = profiles[RM65_BIMANUAL_POSITION_V1_QREF_PROFILE]
    predecessor = profiles["rm65_position_2x"]
    np.testing.assert_allclose(
        formal.max_velocity_deg_s,
        predecessor.max_velocity_deg_s,
        rtol=0.0,
        atol=0.0,
    )
