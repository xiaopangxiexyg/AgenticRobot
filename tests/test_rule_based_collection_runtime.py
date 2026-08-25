from __future__ import annotations

import numpy as np
import pytest

from robot_motion.action import RobotAction
from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_DIM
from robot_motion.replay import (
    ReplayTrajectoryGroup,
    RuleBasedCaptureSelection,
    RuleBasedCollectionCursor,
    RuleBasedCollectionPlan,
    RuleBasedCustomAction,
    RuleBasedInitialize,
    RuleBasedReplay,
)
from teleop_runtime.collection.rule_based_adapter import (
    RuleBasedCollectionAdapter,
)


def _group() -> ReplayTrajectoryGroup:
    actions = np.zeros((2, ACTION_DIM), dtype=np.float32)
    for side in ("left", "right"):
        actions[:, ACTION_ARM_POSE_SLICES[side]][:, 6] = 1.0
    qpos = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )
    return ReplayTrajectoryGroup(
        key="source",
        label="source",
        source_path="synthetic",
        source_episode_index=0,
        start_index=0,
        end_index=2,
        frame_count=2,
        action_hz=15.0,
        actions=actions,
        arm_joint_targets={"left": qpos, "right": qpos + 1.0},
    )


def test_adapter_resolves_composed_plan_without_dispatching() -> None:
    custom = RobotAction.bimanual_qpos(left_arm=np.ones(6), source="custom")
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedInitialize(step_id="init", group_key="source"),
            RuleBasedReplay(
                step_id="replay",
                group_key="source",
                start_index=1,
                end_index=2,
                active_sides=("right",),
                capture=RuleBasedCaptureSelection(stream_ids=("action.right",)),
            ),
            RuleBasedCustomAction(step_id="custom", action=custom),
        )
    )

    emissions = list(
        RuleBasedCollectionAdapter(
            RuleBasedCollectionCursor(plan, {"source": _group()})
        )
    )

    assert [item.directive.kind for item in emissions] == [
        "initialize",
        "replay",
        "custom_action",
    ]
    assert emissions[0].command.commands["left_arm"].command_type == "joint_vector"
    assert emissions[0].recordable is False
    assert emissions[1].recordable is True
    assert emissions[1].allows_stream("action.right")
    assert not emissions[1].allows_stream("state.right")
    assert emissions[2].action is custom
    assert emissions[2].command.metadata["action_source"] == "custom"


def test_adapter_reset_restarts_the_finite_cursor() -> None:
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedCustomAction(
                step_id="custom",
                action=RobotAction.bimanual_qpos(left_arm=np.zeros(6)),
            ),
        )
    )
    adapter = RuleBasedCollectionAdapter(RuleBasedCollectionCursor(plan, {}))

    first = next(adapter)
    with pytest.raises(StopIteration):
        next(adapter)
    adapter.reset()

    second = next(adapter)
    assert second.directive.sequence_index == 0
    assert second.action.source == first.action.source
