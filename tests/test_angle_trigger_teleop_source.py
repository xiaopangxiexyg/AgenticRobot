from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from teleop_runtime.action import HandActionSpec
from teleop_runtime.retargeters import ScalarToGripperRetargeter, ScalarToGripperRetargeterConfig
from teleop_runtime.schemas import ObservationChannel, TeleopObservation
from teleop_runtime.sources.hand import AngleTriggerSource
from teleop_runtime.debug import run_vive_rm65_real_teleop as real_teleop


class _FakeReader:
    def __init__(self) -> None:
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def read(self) -> dict[str, float]:
        assert self.opened
        return {"left": 0.25, "right": 0.75}


def test_angle_trigger_source_publishes_per_side_scalar_channels() -> None:
    source = AngleTriggerSource(_FakeReader(), rate_hz=100.0)
    source.start()
    try:
        deadline = time.monotonic() + 0.5
        channels = source.get_latest_channels()
        while not channels and time.monotonic() < deadline:
            time.sleep(0.01)
            channels = source.get_latest_channels()
    finally:
        source.close()

    assert channels["left_angle_trigger"].capability == "scalar"
    assert channels["left_angle_trigger"].valid is True
    assert channels["left_angle_trigger"].value == 0.25
    assert channels["right_angle_trigger"].value == 0.75


def test_scalar_retargeter_accepts_angle_trigger_scalar() -> None:
    retargeter = ScalarToGripperRetargeter(
        ScalarToGripperRetargeterConfig(
            side="left",
            input_channel_name="left_angle_trigger",
            source_dim=None,
            alpha=1.0,
            input_min=0.0,
            input_max=1.0,
        )
    )
    observation = TeleopObservation(
        timestamp=1.0,
        channels={
            "left_angle_trigger": ObservationChannel(
                channel_name="left_angle_trigger",
                capability="scalar",
                side="left",
                value=0.25,
                frame_name=None,
                timestamp=1.0,
                source_name="angle_trigger",
                valid=True,
            )
        },
    )

    command = retargeter.retarget(observation)
    gripper = command.commands["left_hand"]
    np.testing.assert_allclose(gripper.value, [0.25, 0, 0, 0, 0, 0])
    assert gripper.metadata["source_dim"] is None
    assert gripper.metadata["hand_action_spec"] == HandActionSpec.dmtac_gripper("left").to_metadata()


def test_gripper_input_selection_prefers_q6_and_falls_back_to_angle_trigger() -> None:
    args = SimpleNamespace(
        hand_backend="dmtac-gripper",
        left_hand_backend=None,
        right_hand_backend=None,
        gripper_angle_trigger=False,
    )
    with patch(
        "hardware.teleop_hand.q6_assignment.scan_q6_devices",
        return_value={"q6_devices": []},
    ):
        real_teleop._resolve_gripper_input_source(args, ("left", "right"))
    assert args.gripper_input_source == "angle_trigger"

    with patch(
        "hardware.teleop_hand.q6_assignment.scan_q6_devices",
        return_value={"q6_devices": [{}, {}]},
    ):
        real_teleop._resolve_gripper_input_source(args, ("left", "right"))
    assert args.gripper_input_source == "q6"

    args.gripper_angle_trigger = True
    real_teleop._resolve_gripper_input_source(args, ("left", "right"))
    assert args.gripper_input_source == "angle_trigger"
