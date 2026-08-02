from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robot_motion.rm65 import select_nearest_collision_safe_rm65_ik_pair


def _q(value: float) -> np.ndarray:
    return np.full(6, value, dtype=float)


def test_source_ik_selector_chooses_nearest_collision_safe_pair() -> None:
    def validate(pair):
        safe = not (
            np.allclose(pair["left"], _q(0.1))
            and np.allclose(pair["right"], _q(0.1))
        )
        return SimpleNamespace(
            safe=safe,
            reason=None if safe else "collision",
        )

    receipt = select_nearest_collision_safe_rm65_ik_pair(
        {
            "left": [_q(0.1), _q(0.2)],
            "right": [_q(0.1), _q(0.3)],
        },
        anchor_qpos_by_side={
            "left": np.zeros(6),
            "right": np.zeros(6),
        },
        validate_pair=validate,
    )

    assert receipt.accepted
    assert dict(receipt.candidate_index_by_side) == {
        "left": 1,
        "right": 0,
    }
    assert receipt.collision_rejection_count == 1
    assert dict(receipt.rejection_reason_counts) == {"collision": 1}
    np.testing.assert_allclose(receipt.qpos_by_side["left"], _q(0.2))
    np.testing.assert_allclose(receipt.qpos_by_side["right"], _q(0.1))


def test_source_ik_selector_uses_enumeration_order_for_exact_ties() -> None:
    receipt = select_nearest_collision_safe_rm65_ik_pair(
        {
            "left": [_q(0.1), _q(-0.1)],
            "right": [_q(0.2), _q(-0.2)],
        },
        anchor_qpos_by_side={
            "left": np.zeros(6),
            "right": np.zeros(6),
        },
        validate_pair=lambda _pair: SimpleNamespace(safe=True, reason=None),
    )

    assert receipt.accepted
    assert dict(receipt.candidate_index_by_side) == {
        "left": 0,
        "right": 0,
    }


def test_source_ik_selector_fails_closed_when_one_side_is_missing() -> None:
    receipt = select_nearest_collision_safe_rm65_ik_pair(
        {
            "left": [_q(0.1)],
            "right": [],
        },
        anchor_qpos_by_side={
            "left": np.zeros(6),
            "right": np.zeros(6),
        },
        validate_pair=lambda _pair: SimpleNamespace(safe=True, reason=None),
    )

    assert not receipt.accepted
    assert receipt.qpos_by_side is None
    assert receipt.reason == "missing_side_candidates"


def test_source_ik_selector_fails_closed_when_all_pairs_are_rejected() -> None:
    receipt = select_nearest_collision_safe_rm65_ik_pair(
        {
            "left": [_q(0.1)],
            "right": [_q(0.2)],
        },
        anchor_qpos_by_side={
            "left": np.zeros(6),
            "right": np.zeros(6),
        },
        validate_pair=lambda _pair: SimpleNamespace(
            safe=False,
            reason="joint_position_limit",
        ),
    )

    assert not receipt.accepted
    assert receipt.reason == "no_collision_safe_pair"
    assert receipt.collision_rejection_count == 1
    assert dict(receipt.rejection_reason_counts) == {
        "joint_position_limit": 1
    }
