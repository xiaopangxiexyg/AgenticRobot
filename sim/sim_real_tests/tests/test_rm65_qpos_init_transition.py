from __future__ import annotations

import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("pink")
pytest.importorskip("qpsolvers")

from sim.sim_real_tests.rm65 import (  # noqa: E402
    RM65RandomPoseIkTransitionConfig,
    run_random_pose_ik_qpos_transition,
)


pytestmark = [pytest.mark.integration, pytest.mark.optional]


def test_random_reachable_pose_ik_then_collision_safe_qpos_transition() -> None:
    result = run_random_pose_ik_qpos_transition(
        RM65RandomPoseIkTransitionConfig(seed=17, sample_count=2)
    )

    assert result.passed, result.to_dict()
    assert result.metrics["completed_samples"] == 2
    assert result.metrics["minimum_collision_distance_m"] >= 0.025
    assert all(record["plan_step_count"] > 0 for record in result.records)
