from __future__ import annotations

import sys

import numpy as np
import pytest

from hardware.minimal_hand_teleop.inspire_hand_teleop import (
    InspireHandTeleopBridge,
    parse_args,
)
from robot_motion.action.hand_mapping import (
    DEFAULT_INSPIRE_HAND_JOINT_ORDER,
    map_q6_to_dexhand_target,
)


pytestmark = pytest.mark.unit


def test_bridge_default_keeps_identity_joint_order() -> None:
    source = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
    bridge = InspireHandTeleopBridge(hand=None)

    observed = bridge.map_source_to_target(source)
    expected = map_q6_to_dexhand_target(
        source,
        joint_order=(0, 1, 2, 3, 4, 5),
    ).target_q6

    np.testing.assert_array_equal(observed, expected)


def test_bridge_keeps_invalid_shape_error_contract() -> None:
    bridge = InspireHandTeleopBridge(hand=None)

    with pytest.raises(ValueError, match=r"Expected source q6 shape \(6,\), got \(2,\)"):
        bridge.map_source_to_target([0.0, 1.0])


def test_cli_default_keeps_inspire_joint_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["inspire_hand_teleop.py"])

    args = parse_args()

    np.testing.assert_array_equal(
        args.joint_order,
        np.asarray(DEFAULT_INSPIRE_HAND_JOINT_ORDER, dtype=np.int64),
    )


def test_bridge_keeps_mapping_then_ema_order() -> None:
    first_source = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
    second_source = 1.0 - first_source
    bridge = InspireHandTeleopBridge(
        hand=None,
        joint_order=DEFAULT_INSPIRE_HAND_JOINT_ORDER,
        alpha=0.25,
    )

    first = bridge.map_source_to_target(first_source).copy()
    second = bridge.map_source_to_target(second_source)
    second_target = map_q6_to_dexhand_target(
        second_source,
        joint_order=DEFAULT_INSPIRE_HAND_JOINT_ORDER,
    ).target_q6

    np.testing.assert_allclose(second, 0.25 * second_target + 0.75 * first)
