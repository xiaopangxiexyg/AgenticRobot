from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_DIM
from robot_motion.replay import (
    ReplayTrajectoryGroup,
    RuleBasedBoundaryState,
    RuleBasedRelativeReplayError,
    rebase_replay_action_at_boundary,
)


def _pose(x: float, y: float = 0.0) -> np.ndarray:
    return np.asarray([x, y, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _group() -> ReplayTrajectoryGroup:
    actions = np.zeros((3, ACTION_DIM), dtype=np.float32)
    left_targets = (_pose(1.2), _pose(1.8), _pose(2.5))
    right_targets = (_pose(2.5, 1.0), _pose(3.0, 1.0), _pose(4.0, 1.0))
    for index in range(3):
        actions[index, ACTION_ARM_POSE_SLICES["left"]] = left_targets[index]
        actions[index, ACTION_ARM_POSE_SLICES["right"]] = right_targets[index]
    return ReplayTrajectoryGroup(
        key="next_segment",
        label="next_segment",
        source_path="synthetic",
        source_episode_index=2,
        start_index=0,
        end_index=3,
        frame_count=3,
        action_hz=15.0,
        actions=actions,
        arm_observation_actual_tcp_pose={
            "left": np.stack((_pose(1.0), _pose(1.4), _pose(2.0)), axis=0),
            "right": np.stack((_pose(2.0, 1.0), _pose(2.5, 1.0), _pose(3.5, 1.0)), axis=0),
        },
    )


def _boundary() -> RuleBasedBoundaryState:
    return RuleBasedBoundaryState(
        tcp_pose_by_side={"left": _pose(10.0), "right": _pose(20.0, 1.0)},
        pose_frame_by_side={
            "left": "left_arm_base",
            "right": "right_arm_base",
        },
        source="rule_based_capture_end",
        timestamp=12.5,
    )


def test_rebase_uses_runtime_boundary_as_new_absolute_anchor() -> None:
    action = rebase_replay_action_at_boundary(
        _group(),
        ("left", "right"),
        index=2,
        source_start_index=0,
        boundary=_boundary(),
    )

    np.testing.assert_allclose(action.arm_pose_by_side["left"], _pose(11.5))
    np.testing.assert_allclose(action.arm_pose_by_side["right"], _pose(22.0, 1.0))
    assert action.source == "rule_based_relative_replay"
    assert action.metadata["relative_replay_anchor"] == "runtime_boundary_actual_tcp"
    assert action.metadata["relative_replay_source_start_index"] == 0
    assert action.timestamp == pytest.approx(2.0 / 15.0)


def test_boundary_state_normalizes_matrix_and_rejects_unknown_frames() -> None:
    boundary = RuleBasedBoundaryState(
        tcp_pose_by_side={"left": np.eye(4)},
        pose_frame_by_side={"left": "left_arm_base"},
    )

    np.testing.assert_allclose(boundary.tcp_pose_by_side["left"], _pose(0.0))
    with pytest.raises(ValueError, match="unknown side"):
        RuleBasedBoundaryState(
            tcp_pose_by_side={"left": _pose(0.0)},
            pose_frame_by_side={"left": "left_arm_base", "right": "right_arm_base"},
        )


def test_relative_replay_fails_closed_without_matching_measured_anchor() -> None:
    group = replace(
        _group(),
        arm_observation_actual_tcp_pose={
            "left": _group().arm_observation_actual_tcp_pose["left"]
        },
    )
    with pytest.raises(RuleBasedRelativeReplayError, match="right"):
        rebase_replay_action_at_boundary(
            group,
            ("left", "right"),
            index=1,
            boundary=_boundary(),
        )

    with pytest.raises(RuleBasedRelativeReplayError, match="Frame mismatch"):
        rebase_replay_action_at_boundary(
            _group(),
            ("left",),
            index=1,
            boundary=RuleBasedBoundaryState(
                tcp_pose_by_side={"left": _pose(10.0)},
                pose_frame_by_side={"left": "canonical_desk"},
            ),
        )
