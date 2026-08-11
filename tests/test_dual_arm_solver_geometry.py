from __future__ import annotations

import numpy as np
import pytest

from robot_motion.rm65.dual_arm_solver.geometry import (
    arm_joint_entries,
    arm_joint_indices,
    embed_arm_qpos,
    split_arm_solution,
)


@pytest.mark.parametrize(
    ("solution_layout", "vector_size", "left_start", "right_start"),
    [
        ("dual_arm", 12, 0, 6),
        ("dual_arm", 14, 1, 8),
        ("dual_arm_hand", 24, 0, 18),
        ("dual_arm_hand", 26, 0, 18),
    ],
)
def test_dual_arm_layout_round_trip_and_detached_outputs(
    solution_layout: str,
    vector_size: int,
    left_start: int,
    right_start: int,
) -> None:
    original = np.linspace(-1.0, 1.0, vector_size)
    left = np.linspace(0.1, 0.6, 6)
    right = np.linspace(-0.1, -0.6, 6)

    embedded = embed_arm_qpos(
        solution_layout=solution_layout,
        current_full_q=original,
        left_qpos=left,
        right_qpos=right,
        owner_name="test_solver",
    )

    np.testing.assert_allclose(original, np.linspace(-1.0, 1.0, vector_size))
    np.testing.assert_allclose(embedded[left_start : left_start + 6], left)
    np.testing.assert_allclose(embedded[right_start : right_start + 6], right)
    split_left, split_right = split_arm_solution(
        solution_layout=solution_layout,
        qpos=embedded,
        owner_name="test_solver",
    )
    np.testing.assert_allclose(split_left, left)
    np.testing.assert_allclose(split_right, right)
    split_left[:] = 99.0
    split_right[:] = 99.0
    np.testing.assert_allclose(embedded[left_start : left_start + 6], left)
    np.testing.assert_allclose(embedded[right_start : right_start + 6], right)

    indices = arm_joint_indices(
        solution_layout=solution_layout,
        vector_size=vector_size,
        owner_name="test_solver",
    )
    np.testing.assert_array_equal(indices["left"], np.arange(left_start, left_start + 6))
    np.testing.assert_array_equal(
        indices["right"], np.arange(right_start, right_start + 6)
    )
    assert arm_joint_entries(
        solution_layout=solution_layout,
        vector_size=vector_size,
        owner_name="test_solver",
    ) == tuple(
        [("left", index, left_start + index) for index in range(6)]
        + [("right", index, right_start + index) for index in range(6)]
    )


@pytest.mark.parametrize("vector_size", [0, 6, 13, 15, 23])
def test_dual_arm_layout_rejects_unsupported_vector_sizes(vector_size: int) -> None:
    values = np.zeros(vector_size, dtype=float)
    with pytest.raises(ValueError, match="cannot embed arm qpos"):
        embed_arm_qpos(
            solution_layout="dual_arm",
            current_full_q=values,
            left_qpos=np.zeros(6),
            right_qpos=np.zeros(6),
            owner_name="test_solver",
        )
    with pytest.raises(ValueError, match="unsupported solver output shape"):
        split_arm_solution(
            solution_layout="dual_arm",
            qpos=values,
            owner_name="test_solver",
        )
    with pytest.raises(ValueError, match="cannot resolve arm joint indices"):
        arm_joint_indices(
            solution_layout="dual_arm",
            vector_size=vector_size,
            owner_name="test_solver",
        )
    with pytest.raises(ValueError, match="cannot map arm joint entries"):
        arm_joint_entries(
            solution_layout="dual_arm",
            vector_size=vector_size,
            owner_name="test_solver",
        )
