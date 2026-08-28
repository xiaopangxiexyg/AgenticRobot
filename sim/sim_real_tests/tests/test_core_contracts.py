from __future__ import annotations

import pytest

from sim.sim_real_tests.core import SimRealScenarioResult
from sim.sim_real_tests.rm65 import (
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
