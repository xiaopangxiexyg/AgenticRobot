from __future__ import annotations

import numpy as np
import pytest

from robot_motion.joint_limits import load_joint_velocity_limit_profiles
from robot_motion.rm65 import (
    RM65_BIMANUAL_POSITION_PROFILE_SCHEMA,
    RM65_BIMANUAL_POSITION_V1,
    RM65_BIMANUAL_POSITION_V1_NAME,
    RM65_BIMANUAL_POSITION_V1_QREF_PROFILE,
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
