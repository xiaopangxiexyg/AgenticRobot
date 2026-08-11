from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from control_station.sim_hil_profile.human import (
    _build_passive_single_vive_gripper_component_source,
)
from teleop_runtime.calibration import ArmCalibration, TeleopCalibration
from teleop_runtime.configs import PassiveSingleViveGripperSessionConfig
from teleop_runtime.schemas import ObservationChannel


def _right_home_calibration() -> TeleopCalibration:
    identity = np.eye(4)
    return TeleopCalibration(
        T_W_O=identity,
        T_O_D=identity,
        arms={
            "right_arm": ArmCalibration(
                name="right_arm",
                side="right",
                T_D_B=identity,
            )
        },
        metadata={
            "home_ready": True,
            "home_capture": {
                "right_human_home_virtual": identity.tolist(),
                "right_robot_home_desk": identity.tolist(),
            },
        },
    )


def _pose_channel(*, rotation_drift: bool = False) -> ObservationChannel:
    pose = np.eye(4)
    pose[0, 3] = 0.01
    if rotation_drift:
        pose[:3, :3] = np.array(
            [
                [0.99996, -0.00801, 0.0],
                [0.00800, 0.99995, 0.00002],
                [0.0, -0.00002, 1.00001],
            ]
        )
    return ObservationChannel(
        channel_name="right_wrist_pose",
        capability="pose6d",
        side="right",
        value=pose,
        frame_name="vive_world",
        timestamp=time.time(),
        source_name="local_virtual_vive",
        metadata={
            "tracker_name": "tracker_1",
            "tracker_role": "left_wrist",
        },
    )


def _gripper_channel(input_kind: str) -> ObservationChannel:
    if input_kind == "q6":
        value: object = np.array([0.0, 0.0, 0.75, 0.0, 0.0, 0.0])
        channel_name = "right_hand_q6"
        capability = "joint_position"
    else:
        value = 0.75
        channel_name = "right_angle_trigger"
        capability = "scalar"
    return ObservationChannel(
        channel_name=channel_name,
        capability=capability,
        side="right",
        value=value,
        frame_name=None,
        timestamp=time.time(),
        source_name=f"local_virtual_{input_kind}",
        critical=False,
    )


def _install_local_virtual_device_io(
    monkeypatch: pytest.MonkeyPatch,
    source: Any,
    *,
    input_kind: str,
    pose_rotation_drift: bool = False,
) -> dict[str, int]:
    """Replace only transport/device methods on one concrete passive source."""

    session = source.session
    vive_source, hand_source = session.sources
    lifecycle = {
        "vive_start": 0,
        "hand_start": 0,
        "vive_close": 0,
        "hand_close": 0,
    }

    def start_vive() -> None:
        lifecycle["vive_start"] += 1

    def close_vive() -> None:
        lifecycle["vive_close"] += 1

    def start_hand() -> None:
        lifecycle["hand_start"] += 1

    def close_hand() -> None:
        lifecycle["hand_close"] += 1

    def latest_hand_channels() -> dict[str, ObservationChannel]:
        channel = _gripper_channel(input_kind)
        return {channel.channel_name: channel}

    monkeypatch.setattr(vive_source, "start", start_vive)
    monkeypatch.setattr(vive_source, "close", close_vive)
    monkeypatch.setattr(
        vive_source,
        "get_latest_channels_at_or_before",
        lambda _cutoff: {
            "right_wrist_pose": _pose_channel(
                rotation_drift=pose_rotation_drift
            )
        },
    )
    monkeypatch.setattr(hand_source, "start", start_hand)
    monkeypatch.setattr(hand_source, "close", close_hand)
    monkeypatch.setattr(
        hand_source,
        "get_latest_channels",
        latest_hand_channels,
    )
    return lifecycle


@pytest.mark.parametrize("input_kind", ("q6", "angle_trigger"))
def test_concrete_single_vive_gripper_station_adapter_is_locally_exercisable_without_io(
    monkeypatch: pytest.MonkeyPatch,
    input_kind: str,
) -> None:
    """Exercise the production builder while replacing only device I/O.

    The existing Sim-HIL end-to-end test covers the compatible source contract
    through composition and MuJoCo dispatch.  This test closes the other side
    of that seam: concrete VIVE/Q6 or VIVE/angle construction, the lower-owned
    passive TeleopSession, retargeting, and the Station component adapter.
    """

    source, source_spec = _build_passive_single_vive_gripper_component_source(
        calibration=_right_home_calibration(),
        source_config=PassiveSingleViveGripperSessionConfig(
            side="right",
            gripper_input_source=input_kind,
            q6_port="/dev/local-virtual-q6" if input_kind == "q6" else None,
        ),
        target_action_spec_hash_by_component={
            "right_arm": "local-virtual-right-arm-spec",
            "right_gripper": "local-virtual-right-gripper-spec",
        },
        target_units_convention_id=(
            "si_m_rad_xyzw_plus_normalized_hand_v1"
        ),
    )
    session = source.session
    assert session.rig is None
    assert [item.component_id for item in source_spec.components] == [
        "right_arm",
        "right_gripper",
    ]
    assert [item.name for item in session.sources] == [
        "vive_pose",
        "hand_q6_right" if input_kind == "q6" else "angle_trigger",
    ]

    lifecycle = _install_local_virtual_device_io(
        monkeypatch,
        source,
        input_kind=input_kind,
    )

    receipt = source.start()
    source.set_calibration_ready()
    try:
        step = source.step(now=time.monotonic())
    finally:
        source.close()

    assert receipt.rig_absent is True
    assert receipt.authoritative is False
    assert lifecycle == {
        "vive_start": 1,
        "hand_start": 1,
        "vive_close": 1,
        "hand_close": 1,
    }
    assert step.command_dispatched is False
    assert step.authoritative is False
    assert step.emission_valid is True
    assert step.rejection_reason is None
    assert step.emission is not None
    commands = {
        command.effector_name: command
        for command in step.emission.merged_commands
    }
    assert set(commands) == {"right_arm", "right_hand"}
    assert commands["right_arm"].command_type == "cartesian_pose"
    assert commands["right_arm"].frame_name == "right_arm_base"
    arm_pose = np.asarray(commands["right_arm"].value, dtype=float)
    assert arm_pose.shape == (4, 4)
    assert np.all(np.isfinite(arm_pose))
    assert commands["right_hand"].command_type == "joint_vector"
    hand_qpos = np.asarray(commands["right_hand"].value, dtype=float)
    assert hand_qpos.shape == (6,)
    assert np.all(np.isfinite(hand_qpos))
    assert hand_qpos[0] > 0.0


def test_single_vive_angle_trigger_projects_openvr_drift_before_emission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _source_spec = _build_passive_single_vive_gripper_component_source(
        calibration=_right_home_calibration(),
        source_config=PassiveSingleViveGripperSessionConfig(
            side="right",
            gripper_input_source="angle_trigger",
        ),
        target_action_spec_hash_by_component={
            "right_arm": "local-virtual-right-arm-spec",
            "right_gripper": "local-virtual-right-gripper-spec",
        },
        target_units_convention_id=(
            "si_m_rad_xyzw_plus_normalized_hand_v1"
        ),
    )
    _install_local_virtual_device_io(
        monkeypatch,
        source,
        input_kind="angle_trigger",
        pose_rotation_drift=True,
    )

    source.start()
    source.set_calibration_ready()
    try:
        step = source.step(now=time.monotonic())
    finally:
        source.close()

    assert step.emission_valid is True
    assert step.rejection_reason is None
    commands = {
        command.effector_name: command
        for command in step.emission.merged_commands
    }
    arm = commands["right_arm"]
    rotation = np.asarray(arm.value, dtype=float)[:3, :3]
    diagnostics = arm.metadata["input_pose_normalization"]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-10)
    assert diagnostics["rotation_projected"] is True
