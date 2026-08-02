from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robot_motion.position_control import ResampledMotionReference
from teleop_runtime.position_control.official_all_ik_governor import (
    OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND,
    RM65OfficialAllIKGovernorPlanner,
)


class _CandidateProvider:
    def solve_candidates(self, *, seed_rad, target_endpoint):
        seed = np.asarray(seed_rad, dtype=float).reshape(6)
        candidate = seed.copy()
        candidate[0] = float(np.asarray(target_endpoint)[0, 3])
        return (candidate,)


class _CollisionPlanner:
    collision_pair_count = 4

    def reset(self, current_qpos_by_side):
        return {"reset": True}

    def validate_qpos_reference(self, qpos_by_side, *, source):
        return SimpleNamespace(
            safe=True,
            reason=None,
            collision_distance_m=0.1,
        )

    def admit_qpos_reference(self, qpos_by_side, *, source):
        return self.validate_qpos_reference(qpos_by_side, source=source)

    def observe_endpoint_poses_world(self, qpos_by_side):
        result = {}
        for side in ("left", "right"):
            pose = np.eye(4, dtype=float)
            pose[0, 3] = float(np.asarray(qpos_by_side[side])[0])
            result[side] = pose
        return result


def _poses(x: float) -> dict[str, np.ndarray]:
    result = {}
    for side in ("left", "right"):
        pose = np.eye(4, dtype=float)
        pose[0, 3] = float(x)
        result[side] = pose
    return result


def test_official_all_ik_governor_uses_staged_two_source_horizon() -> None:
    planner = RM65OfficialAllIKGovernorPlanner(
        _CollisionPlanner(),
        endpoint_to_tool_by_side={
            "left": np.eye(4),
            "right": np.eye(4),
        },
        candidate_provider=_CandidateProvider(),
    )
    planner.reset({"left": np.zeros(6), "right": np.zeros(6)})
    for index, x in enumerate((0.0, 0.08, 0.16)):
        planner.stage_source_pose(
            source_index=index,
            source_timestamp=index / 15.0,
            pose_by_side=_poses(x),
        )
    reference = ResampledMotionReference(
        sequence=0,
        timestamp=0.0,
        left_source_index=0,
        right_source_index=0,
        left_source_timestamp=0.0,
        right_source_timestamp=0.0,
        interpolation_alpha=0.0,
        pose_by_side=_poses(0.0),
        qpos_by_side={},
        metadata={},
    )

    result = planner.plan_pose_reference_horizon(
        reference,
        source_timestamp=0.0,
        source="test",
    )

    assert result.valid is True
    assert result.hold is False
    assert result.metadata["planner"] == OFFICIAL_SOURCE_ALL_IK_GOVERNOR_BACKEND
    assert result.metadata["lookahead_source_steps_available"] == 2
    assert result.metadata["segment_collision_checked"] is True
    assert result.metadata["segment_collision_certified"] is False
    assert np.all(np.isfinite(result.q_ref_by_side["left"]))

    next_reference = ResampledMotionReference(
        sequence=1,
        timestamp=1.0 / 120.0,
        left_source_index=0,
        right_source_index=0,
        left_source_timestamp=0.0,
        right_source_timestamp=0.0,
        interpolation_alpha=0.125,
        pose_by_side=_poses(0.01),
        qpos_by_side={},
        metadata={},
    )
    next_result = planner.plan_pose_reference_horizon(
        next_reference,
        source_timestamp=1.0 / 120.0,
        source="test",
    )
    assert next_result.metadata["source_cut_replanned"] is False
    assert (
        next_result.metadata["governor_solver"]
        == "jerk_limited_joint_horizon_qp_v4_reused_batched"
    )
    assert next_result.metadata["governor_solver_tolerance"] == 1e-4
