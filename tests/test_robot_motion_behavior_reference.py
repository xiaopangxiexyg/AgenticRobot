from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import pytest

import robot_motion.pose as strict_pose_module
import robot_motion.transforms as transforms_module
from robot_motion.diagnostics import (
    FaultEvent,
    FaultSeverity,
    RateTracker,
    TeleopRunState,
    TeleopStateMachine,
)
from robot_motion.action import (
    RobotAction,
    RobotActionResolver,
    encode_robot_action,
)
from robot_motion.action.hand_mapping import map_q6_to_dexhand_target
from robot_motion.frames import (
    ARM_ENDPOINT_FRAME,
    HAND_BASE_TCP_FRAME,
    TOOL_TCP_FRAME,
    convert_pose_between_arm_control_frames,
)
from robot_motion.pose_smoothing import PoseSmoother, PoseSmoothingConfig
from robot_motion.runtime import (
    EffectorCommand,
    ObservationChannel,
    TeleopCommand,
    merge_teleop_commands,
)
from robot_motion.safety import MotionSafetyConfig, SafetyMonitor
from robot_motion.tcp import canonical_tcp_frame, legacy_tcp_mode_requested
from robot_motion.transforms import as_pose_matrix, invert_pose, parse_pose_matrix


def _pose(x: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[0, 3] = float(x)
    return pose


def test_motion_safety_config_exposes_only_enforced_monitor_limits() -> None:
    assert tuple(item.name for item in fields(MotionSafetyConfig)) == (
        "max_translation_jump_m",
        "max_rotation_jump_deg",
        "stale_timeout_sec",
        "invalid_frame_limit",
        "freeze_on_fault",
        "minimum_control_hz",
        "required_channels",
    )
    with pytest.raises(TypeError, match="max_target_speed_mps"):
        MotionSafetyConfig(max_target_speed_mps=1.0)  # type: ignore[call-arg]


def test_strict_pose_checker_and_flexible_pose_parser_reference_sample() -> None:
    pose_matrix = _pose(0.1)
    pose_xyzquat = np.asarray([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    assert strict_pose_module.as_pose_matrix is strict_pose_module.as_strict_pose_matrix
    assert transforms_module.as_pose_matrix is transforms_module.parse_pose_matrix
    np.testing.assert_allclose(strict_pose_module.as_strict_pose_matrix(pose_matrix), pose_matrix)
    np.testing.assert_allclose(parse_pose_matrix(pose_xyzquat), pose_matrix)
    np.testing.assert_allclose(as_pose_matrix(pose_xyzquat), pose_matrix)
    with pytest.raises(ValueError, match="4x4"):
        strict_pose_module.as_strict_pose_matrix(pose_xyzquat)


def test_action_vector_encoding_reference_sample() -> None:
    action = RobotAction.bimanual_ee_pose(
        left_arm=np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        right_hand=np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        timestamp=12.5,
        source="behavior_reference",
    )

    encoding = encode_robot_action(action, fill_value=-1.0)

    expected_vector = np.full((26,), -1.0, dtype=np.float32)
    expected_vector[0:7] = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
    expected_vector[20:26] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    expected_mask = np.zeros((26,), dtype=bool)
    expected_mask[0:7] = True
    expected_mask[20:26] = True

    assert encoding.action_space == "bimanual_ee_pose_v1"
    assert encoding.layout_version == "robot_action_vector.v1"
    np.testing.assert_allclose(encoding.vector, expected_vector)
    np.testing.assert_array_equal(encoding.component_mask, expected_mask)
    assert encoding.effector_mask == {
        "left_arm": True,
        "right_arm": False,
        "left_hand": False,
        "right_hand": True,
    }


def test_action_resolver_cartesian_command_reference_sample() -> None:
    action = RobotAction.bimanual_ee_pose(
        left_arm=np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        timestamp=12.5,
        source="behavior_reference",
    )

    command = RobotActionResolver().resolve(action)
    left = command.commands["left_arm"]

    expected_pose = np.eye(4, dtype=np.float32)
    expected_pose[:3, 3] = [0.1, 0.2, 0.3]
    assert command.timestamp == 12.5
    assert command.metadata["action_space"] == "bimanual_ee_pose_v1"
    assert command.metadata["action_source"] == "behavior_reference"
    assert left.command_type == "cartesian_pose"
    assert left.frame_name == "left_arm_base"
    np.testing.assert_allclose(left.value, expected_pose)
    assert left.metadata["arm_controlled_frame"] == TOOL_TCP_FRAME
    np.testing.assert_allclose(left.metadata["target_pose_base"], expected_pose)
    np.testing.assert_array_equal(
        left.metadata["target_xyzquat"],
        np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )


def test_hand_q6_mapping_reference_sample() -> None:
    result = map_q6_to_dexhand_target(
        np.asarray([-0.5, 0.2, 0.4, 0.6, 0.8, 1.5], dtype=np.float32),
        gain=(1.0, 0.5, 1.0, 0.5, 1.0, 1.0),
        bias=(0.0, 0.1, 0.0, 0.2, 0.0, -0.1),
        output_min=0.0,
        output_max=1.0,
    )

    np.testing.assert_allclose(result.source_q6, [1.0, 0.8, 0.6, 0.4, 0.0, 0.2])
    np.testing.assert_allclose(result.target_q6, [0.0, 0.2, 0.4, 0.5, 1.0, 0.7])


def test_frame_transform_and_tcp_reference_samples() -> None:
    endpoint_pose = np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    hand_base_pose = convert_pose_between_arm_control_frames(
        endpoint_pose,
        from_frame=ARM_ENDPOINT_FRAME,
        to_frame=HAND_BASE_TCP_FRAME,
    )
    endpoint_again = convert_pose_between_arm_control_frames(
        hand_base_pose,
        from_frame=HAND_BASE_TCP_FRAME,
        to_frame=ARM_ENDPOINT_FRAME,
    )

    expected_hand_base_pose = np.eye(4, dtype=float)
    expected_hand_base_pose[:3, 3] = [0.1, 0.2, 0.309]
    np.testing.assert_allclose(hand_base_pose, expected_hand_base_pose)
    np.testing.assert_allclose(endpoint_again, as_pose_matrix(endpoint_pose))

    assert canonical_tcp_frame(None) == TOOL_TCP_FRAME
    assert canonical_tcp_frame("") == TOOL_TCP_FRAME
    assert canonical_tcp_frame("tcp") == TOOL_TCP_FRAME
    assert canonical_tcp_frame("rm_eef") == TOOL_TCP_FRAME
    assert canonical_tcp_frame(HAND_BASE_TCP_FRAME) == HAND_BASE_TCP_FRAME
    assert legacy_tcp_mode_requested("rm_eef") is True
    assert legacy_tcp_mode_requested(TOOL_TCP_FRAME) is False


def test_transform_quaternion_and_inverse_reference_sample() -> None:
    pose_xyzquat = np.asarray(
        [0.1, 0.2, 0.3, 0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)],
        dtype=np.float32,
    )

    pose = as_pose_matrix(pose_xyzquat)
    expected = np.array(
        [
            [0.0, -1.0, 0.0, 0.1],
            [1.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(pose, expected, atol=1e-6)
    np.testing.assert_allclose(invert_pose(pose) @ pose, np.eye(4), atol=1e-6)


def test_pose_smoothing_reference_sequence() -> None:
    smoother = PoseSmoother(
        PoseSmoothingConfig(
            time_constant_sec=1.0,
            nominal_dt_sec=0.1,
            max_dt_sec=0.1,
            reset_after_gap_sec=0.5,
            deadband_translation_m=0.0,
            deadband_rotation_deg=0.0,
            snap_translation_m=0.0,
            snap_rotation_deg=0.0,
        )
    )

    first = smoother.smooth(_pose(0.0), timestamp=0.0)
    second = smoother.smooth(_pose(0.1), timestamp=0.1)

    expected_alpha = 1.0 - math.exp(-0.1)
    assert first.metadata["status"] == "initialized"
    assert second.metadata["status"] == "smoothed"
    assert second.metadata["dt_sec"] == 0.1
    assert second.metadata["raw_dt_sec"] == 0.1
    assert second.metadata["alpha"] == expected_alpha
    np.testing.assert_allclose(second.pose, _pose(0.1 * expected_alpha))


def test_runtime_command_merge_reference_sample() -> None:
    left_command = TeleopCommand(timestamp=1.0, metadata={"source": "left", "shared": "first"})
    right_command = TeleopCommand(timestamp=2.0, metadata={"source": "right", "shared": "second"})
    left_value = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    right_value = np.asarray([4.0, 5.0], dtype=np.float32)
    left_command.add_command(
        EffectorCommand(
            effector_name="left_hand",
            side="left",
            command_type="joint_vector",
            value=left_value,
            metadata={"label": "left"},
        )
    )
    right_command.add_command(
        EffectorCommand(
            effector_name="right_hand",
            side="right",
            command_type="joint_vector",
            value=right_value,
            metadata={"label": "right"},
        )
    )

    merged = merge_teleop_commands([left_command, right_command])

    assert merged.timestamp == 2.0
    assert merged.metadata == {"source": "right", "shared": "second"}
    assert tuple(merged.commands) == ("left_hand", "right_hand")
    np.testing.assert_array_equal(merged.commands["left_hand"].value, left_value)
    np.testing.assert_array_equal(merged.commands["right_hand"].value, right_value)
    assert merged.commands["left_hand"].value is not left_value
    assert merged.commands["right_hand"].metadata == {"label": "right"}


def test_safety_monitor_baseline_reset_reference_sequence() -> None:
    monitor = SafetyMonitor(
        MotionSafetyConfig(
            max_translation_jump_m=0.10,
            max_rotation_jump_deg=45.0,
            invalid_frame_limit=1,
            required_channels=("right_wrist_pose",),
        )
    )

    def channel(*, value: np.ndarray, timestamp: float, valid: bool = True) -> ObservationChannel:
        return ObservationChannel(
            channel_name="right_wrist_pose",
            capability="pose6d",
            side="right",
            value=value,
            frame_name="desk",
            timestamp=timestamp,
            source_name="behavior_reference",
            valid=valid,
            critical=True,
        )

    first = {"right_wrist_pose": channel(value=_pose(), timestamp=0.0)}
    invalid = {"right_wrist_pose": channel(value=_pose(), timestamp=0.1, valid=False)}
    recovered = {"right_wrist_pose": channel(value=_pose(0.30), timestamp=0.2)}
    jumped = {"right_wrist_pose": channel(value=_pose(0.65), timestamp=0.3)}

    assert monitor.evaluate_channels(first, 0.0) == []
    invalid_violations = monitor.evaluate_channels(invalid, 0.1)
    assert [violation.code for violation in invalid_violations] == ["invalid_channel"]
    assert invalid_violations[0].metadata == {"invalid_count": 1}
    assert monitor.evaluate_channels(recovered, 0.2) == []
    jump_violations = monitor.evaluate_channels(jumped, 0.3)
    assert [violation.code for violation in jump_violations] == ["translation_jump"]
    assert np.isclose(jump_violations[0].metadata["translation_jump_m"], 0.35)


def test_safety_monitor_control_rate_reference_sample() -> None:
    monitor = SafetyMonitor(MotionSafetyConfig(minimum_control_hz=15.0))

    assert monitor.evaluate_control_rate(15.0) == []
    violations = monitor.evaluate_control_rate(14.5)
    assert [violation.code for violation in violations] == ["low_control_rate"]
    assert violations[0].metadata == {"control_hz": 14.5}


def test_diagnostics_rate_tracker_reference_sample() -> None:
    tracker = RateTracker(max_samples=3)

    assert tracker.num_samples == 0
    assert tracker.current_hz == 0.0
    assert tracker.tick(0.0) == 0.0
    assert tracker.tick(0.5) == 2.0
    assert tracker.tick(1.0) == 2.0
    assert tracker.tick(2.0) == 4.0 / 3.0
    assert tracker.num_samples == 3


def test_diagnostics_state_machine_reference_sample() -> None:
    state_machine = TeleopStateMachine()

    assert state_machine.state is TeleopRunState.INIT
    assert state_machine.history[0]["state"] == "INIT"
    assert state_machine.history[0]["reason"] == "init"

    state_machine.transition(TeleopRunState.RUNNING, "started")

    assert state_machine.state is TeleopRunState.RUNNING
    assert state_machine.history[-1]["state"] == "RUNNING"
    assert state_machine.history[-1]["reason"] == "started"


def test_diagnostics_fault_event_reference_sample() -> None:
    fault = FaultEvent(
        code="translation_jump",
        message="jump detected",
        severity=FaultSeverity.WARNING,
        timestamp=12.5,
        component="safety",
        channel_name="right_wrist_pose",
        metadata={"translation_jump_m": 0.35},
    )

    assert fault.code == "translation_jump"
    assert fault.message == "jump detected"
    assert fault.severity is FaultSeverity.WARNING
    assert fault.timestamp == 12.5
    assert fault.component == "safety"
    assert fault.channel_name == "right_wrist_pose"
    assert fault.metadata == {"translation_jump_m": 0.35}
