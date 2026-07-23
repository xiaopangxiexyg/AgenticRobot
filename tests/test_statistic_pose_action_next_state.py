from __future__ import annotations

import numpy as np

from utils.statistic.pose_action_next_state import needed_columns, pose_from_row


def _pose(seed: float) -> list[float]:
    return [seed, seed + 0.1, seed + 0.2, 0.0, 0.0, 0.0, 1.0]


def test_pose_from_v23_packed_vectors_uses_first_fourteen_values() -> None:
    left = _pose(1.0)
    right = _pose(2.0)
    row = {
        "action": left + right + [0.25, 0.75],
        "observation.state": left + right + [0.2, 0.7],
        "observation.last_action": left + right + [0.1, 0.6],
    }

    for source in ("action", "state", "last_action"):
        np.testing.assert_allclose(
            pose_from_row(row, side="left", source=source),
            left,
        )
        np.testing.assert_allclose(
            pose_from_row(row, side="right", source=source),
            right,
        )


def test_pose_from_legacy_v21_state_keeps_38d_compatibility() -> None:
    left = _pose(3.0)
    right = _pose(4.0)
    legacy_state = [0.0] * 38
    legacy_state[6:13] = left
    legacy_state[25:32] = right

    np.testing.assert_allclose(
        pose_from_row(
            {"observation.state": legacy_state},
            side="left",
            source="state",
        ),
        left,
    )
    np.testing.assert_allclose(
        pose_from_row(
            {"observation.state": legacy_state},
            side="right",
            source="state",
        ),
        right,
    )


def test_explicit_v23_pose_columns_take_priority_over_packed_vectors() -> None:
    explicit = _pose(5.0)
    packed = _pose(6.0)
    row = {
        "action.left_arm.pose": explicit,
        "action": packed + packed,
        "observation.arm.left.actual_tcp_pose": explicit,
        "observation.state": packed + packed,
    }

    np.testing.assert_allclose(
        pose_from_row(row, side="left", source="action"),
        explicit,
    )
    np.testing.assert_allclose(
        pose_from_row(row, side="left", source="state"),
        explicit,
    )


def test_needed_columns_includes_v23_combined_and_actual_tcp_columns() -> None:
    available = {
        "action",
        "observation.state",
        "action.arm.pose",
        "observation.arm.pose",
        "observation.arm.left.actual_tcp_pose",
    }

    assert set(needed_columns(available)) == available
