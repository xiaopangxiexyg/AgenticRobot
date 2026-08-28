from __future__ import annotations

import pytest

from sim.sim_real_tests.core import SimRealScenarioResult
from sim.sim_real_tests.rm65 import (
    CONFIG_SCHEMA_VERSION,
    DEFAULT_REFERENCE_FRAME_ID,
    DEFAULT_REFERENCE_FRAME_SPEC_HASH,
    RM65RandomGraspPoseReachabilityConfig,
    load_random_grasp_pose_reachability_config,
    load_random_pose_ik_transition_config,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_scenario_result_is_json_ready() -> None:
    payload = SimRealScenarioResult(
        scenario="example",
        passed=True,
        metrics={"count": 1},
        records=({"step": 1},),
    ).to_dict()

    assert payload == {
        "scenario": "example",
        "passed": True,
        "metrics": {"count": 1},
        "records": [{"step": 1}],
        "failure": "",
    }


def test_default_rm65_random_pose_config_loads() -> None:
    config = load_random_pose_ik_transition_config()

    assert config.sample_count >= 1
    assert config.collision_scene_profile == "dual_arm_hand_proxy_strict"
    assert config.collision_stop_distance_m <= config.collision_hold_distance_m


def test_default_rm65_random_grasp_pose_reachability_config_loads() -> None:
    config = load_random_grasp_pose_reachability_config()

    assert config.sample_count >= 1
    assert config.orientation_sampler == "shoemake_uniform_so3"
    assert config.max_ik_candidates >= 1
    assert config.controlled_frame == "tool_tcp"
    assert config.reference_frame_id == DEFAULT_REFERENCE_FRAME_ID
    assert config.reference_frame_spec_hash == DEFAULT_REFERENCE_FRAME_SPEC_HASH


def test_random_grasp_config_rejects_missing_or_mismatched_schema_version() -> None:
    valid = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "scenario": "rm65_random_grasp_pose_reachability",
    }

    assert RM65RandomGraspPoseReachabilityConfig.from_mapping(valid).sample_count >= 1
    with pytest.raises(ValueError, match="schema_version must be exactly"):
        RM65RandomGraspPoseReachabilityConfig.from_mapping(
            {**valid, "schema_version": "rm65.random_grasp_pose_reachability.v2"}
        )
    with pytest.raises(ValueError, match="schema_version must be exactly"):
        RM65RandomGraspPoseReachabilityConfig.from_mapping(
            {"scenario": "rm65_random_grasp_pose_reachability"}
        )
    with pytest.raises(ValueError, match="scenario must be exactly"):
        RM65RandomGraspPoseReachabilityConfig.from_mapping(
            {**valid, "scenario": "different_scenario"}
        )


def test_canonical_model_base_frame_requires_its_exact_spec_hash() -> None:
    with pytest.raises(ValueError, match="canonical spec hash"):
        RM65RandomGraspPoseReachabilityConfig(
            reference_frame_id=DEFAULT_REFERENCE_FRAME_ID,
            reference_frame_spec_hash="1" * 64,
        )
