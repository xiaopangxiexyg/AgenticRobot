from __future__ import annotations

import numpy as np
import pytest

from robot_motion.relative_action import (
    ACTION_POSE_MODE_ABSOLUTE,
    ACTION_POSE_MODE_RELATIVE,
    RelativeActionError,
    compare_pose_chunks,
    convert_action_and_last_action_chunk,
    convert_arm_pose_chunk,
    convert_last_action_chunk,
    pose_matrix_to_xyzquat,
    relative_pose,
    xyzquat_to_pose_matrix,
)


def _pose(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = [x, y, z]
    return matrix


def _action_rows(values: list[float]) -> np.ndarray:
    rows = np.full((len(values), 14), np.nan, dtype=np.float32)
    for index, value in enumerate(values):
        rows[index, 0:7] = pose_matrix_to_xyzquat(_pose(value), name="left")
        rows[index, 7:14] = pose_matrix_to_xyzquat(_pose(0.0, value), name="right")
    return rows


def test_relative_pose_and_absolute_pose_round_trip() -> None:
    anchor = _pose(1.0, 2.0, 0.0)
    target = _pose(1.2, 2.4, 0.1)
    relative = relative_pose(anchor, target)
    reconstructed = anchor @ relative
    assert reconstructed == pytest.approx(target)


def test_relative_chunk_uses_one_anchor_and_preserves_non_arm_slots() -> None:
    actions = np.full((3, 18), np.nan, dtype=np.float32)
    actions[:, 0:7] = np.stack([pose_matrix_to_xyzquat(_pose(1.0 + i)) for i in range(3)])
    actions[:, 7:14] = np.stack([pose_matrix_to_xyzquat(_pose(0.0, 2.0 + i)) for i in range(3)])
    actions[:, 14:18] = np.asarray([[1, 2, 3, 4]] * 3, dtype=np.float32)
    anchor_by_side = {"left": _pose(1.0), "right": _pose(0.0, 2.0)}

    relative = convert_arm_pose_chunk(
        actions,
        anchors_by_side=anchor_by_side,
        source_mode=ACTION_POSE_MODE_ABSOLUTE,
        target_mode=ACTION_POSE_MODE_RELATIVE,
    )
    assert relative[:, 14:18] == pytest.approx(actions[:, 14:18])
    assert relative[0, 0:3] == pytest.approx([0.0, 0.0, 0.0])
    assert relative[1, 0:3] == pytest.approx([1.0, 0.0, 0.0])
    assert relative[2, 7:10] == pytest.approx([0.0, 2.0, 0.0])

    reconstructed = convert_arm_pose_chunk(
        relative,
        anchors_by_side=anchor_by_side,
        source_mode=ACTION_POSE_MODE_RELATIVE,
        target_mode=ACTION_POSE_MODE_ABSOLUTE,
    )
    metrics = compare_pose_chunks(actions, reconstructed)
    assert metrics["left"].max_translation_error_m < 1e-6
    assert metrics["right"].max_translation_error_m < 1e-6
    assert metrics["left"].max_rotation_error_rad < 1e-6


def test_last_action_uses_current_row_anchor_not_previous_anchor() -> None:
    last_actions = _action_rows([1.0, 1.0])
    current_anchors = {"left": np.stack([_pose(1.0), _pose(2.0)])}
    current_anchors["right"] = np.stack([_pose(0.0, 0.0), _pose(0.0, 1.0)])

    relative = convert_last_action_chunk(
        last_actions,
        current_anchors_by_side=current_anchors,
        source_mode=ACTION_POSE_MODE_ABSOLUTE,
        target_mode=ACTION_POSE_MODE_RELATIVE,
    )
    assert relative[0, 0:3] == pytest.approx([0.0, 0.0, 0.0])
    assert relative[1, 0:3] == pytest.approx([-1.0, 0.0, 0.0])
    assert relative[0, 7:10] == pytest.approx([0.0, 1.0, 0.0])
    assert relative[1, 7:10] == pytest.approx([0.0, 0.0, 0.0])


def test_action_and_last_action_have_distinct_anchor_contracts() -> None:
    actions = _action_rows([1.0, 1.5])
    last_actions = _action_rows([0.5, 1.0])
    converted_action, converted_last = convert_action_and_last_action_chunk(
        actions,
        last_actions,
        action_anchors_by_side={"left": _pose(1.0), "right": _pose(0.0, 0.0)},
        last_action_current_anchors_by_side={
            "left": np.stack([_pose(1.0), _pose(1.5)]),
            "right": np.stack([_pose(0.0), _pose(0.0, 0.5)]),
        },
        source_mode=ACTION_POSE_MODE_ABSOLUTE,
        target_mode=ACTION_POSE_MODE_RELATIVE,
    )
    assert converted_action[1, 0:3] == pytest.approx([0.5, 0.0, 0.0])
    assert converted_last is not None
    assert converted_last[1, 0:3] == pytest.approx([-0.5, 0.0, 0.0])


def test_invalid_anchor_or_action_fails_closed() -> None:
    actions = _action_rows([1.0])
    with pytest.raises(RelativeActionError):
        convert_arm_pose_chunk(
            actions,
            anchors_by_side={"left": np.zeros(7)},
            source_mode=ACTION_POSE_MODE_ABSOLUTE,
            target_mode=ACTION_POSE_MODE_RELATIVE,
        )
    with pytest.raises(RelativeActionError):
        convert_arm_pose_chunk(
            actions,
            anchors_by_side={"left": _pose(0.0)},
            source_mode=ACTION_POSE_MODE_ABSOLUTE,
            target_mode=ACTION_POSE_MODE_RELATIVE,
            active_sides=("left", "right"),
        )


def test_quaternion_output_has_deterministic_nonnegative_w() -> None:
    pose = np.eye(4, dtype=float)
    vector = pose_matrix_to_xyzquat(pose)
    assert vector[6] >= 0.0
    assert xyzquat_to_pose_matrix(vector) == pytest.approx(pose)
