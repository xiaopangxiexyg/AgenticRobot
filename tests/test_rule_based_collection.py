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

pytestmark = pytest.mark.unit


def _group() -> ReplayTrajectoryGroup:
    actions = np.zeros((3, ACTION_DIM), dtype=np.float32)
    for side in ("left", "right"):
        actions[:, ACTION_ARM_POSE_SLICES[side]][:, 6] = 1.0
    qpos = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        ],
        dtype=np.float32,
    )
    return ReplayTrajectoryGroup(
        key="source",
        label="source",
        source_path="synthetic",
        source_episode_index=4,
        start_index=0,
        end_index=3,
        frame_count=3,
        action_hz=15.0,
        actions=actions,
        arm_joint_targets={"left": qpos, "right": qpos + 1.0},
    )


def test_plan_composes_init_replay_and_custom_action_with_per_step_capture() -> None:
    custom = RobotAction.bimanual_qpos(left_arm=np.ones(6), source="custom_open")
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedInitialize(step_id="init", group_key="source"),
            RuleBasedReplay(
                step_id="replay_head",
                group_key="source",
                start_index=1,
                end_index=3,
                active_sides=("left",),
                capture=RuleBasedCaptureSelection(stream_ids=("action.left",)),
            ),
            RuleBasedCustomAction(step_id="custom", action=custom, repeats=2),
        )
    )

    directives = list(RuleBasedCollectionCursor(plan, {"source": _group()}))

    assert [directive.kind for directive in directives] == [
        "initialize",
        "replay",
        "replay",
        "custom_action",
        "custom_action",
    ]
    assert [directive.sequence_index for directive in directives] == list(range(5))
    assert directives[0].capture.enabled is False
    assert directives[0].qpos_by_side["left"].shape == (6,)
    assert [directive.replay_frame_index for directive in directives[1:3]] == [1, 2]
    assert directives[1].capture.allows("action.left")
    assert not directives[1].capture.allows("state.left")
    assert directives[3].action is custom
    assert directives[3].metadata["custom_action_source"] == "custom_open"


def test_qpos_replay_reuses_existing_qpos_selection() -> None:
    plan = RuleBasedCollectionPlan(
        steps=(
            RuleBasedReplay(
                step_id="qpos",
                group_key="source",
                start_index=0,
                end_index=1,
                active_sides=("right",),
                source="qpos_target",
            ),
        )
    )

    directive = next(RuleBasedCollectionCursor(plan, {"source": _group()}))

    assert directive.action is not None
    np.testing.assert_allclose(
        directive.action.arm_qpos_by_side["right"],
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    )
    assert directive.action.source == "rule_based_replay"


def test_rule_based_plan_fails_closed_for_missing_groups_and_invalid_ranges() -> None:
    with pytest.raises(KeyError, match="Missing trajectory group"):
        RuleBasedCollectionCursor(
            RuleBasedCollectionPlan(
                steps=(RuleBasedInitialize(step_id="init", group_key="missing"),)
            ),
            {},
        )

    with pytest.raises(IndexError, match="exceeds"):
        list(
            RuleBasedCollectionCursor(
                RuleBasedCollectionPlan(
                    steps=(
                        RuleBasedReplay(
                            step_id="too_long",
                            group_key="source",
                            start_index=0,
                            end_index=4,
                        ),
                    )
                ),
                {"source": _group()},
            )
        )


def test_capture_selection_rejects_duplicate_streams() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        RuleBasedCaptureSelection(stream_ids=("state.left", "state.left"))
