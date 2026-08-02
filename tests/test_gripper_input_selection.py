from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_motion.action import ArmActionSpec
from teleop_runtime.calibration import ArmCalibration, TeleopCalibration
from teleop_runtime.configs import (
    PassiveSingleViveGripperSessionConfig,
    build_passive_single_vive_gripper_session,
)
from teleop_runtime.safety import TeleopSafetyConfig
from teleop_runtime.configs import gripper_input as selection_module
from teleop_runtime.configs.gripper_input import (
    GRIPPER_INPUT_ANGLE_TRIGGER,
    GRIPPER_INPUT_Q6,
    resolve_single_gripper_input,
)
from teleop_runtime.schemas import ObservationChannel, TeleopObservation


def test_auto_gripper_input_prefers_configured_reachable_q6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        selection_module,
        "_probe_configured_q6",
        lambda port: (port == "/dev/q6-right", None),
    )
    monkeypatch.setattr(
        selection_module,
        "_scan_reachable_q6",
        lambda: pytest.fail("scan must not run after a successful configured probe"),
    )

    selection = resolve_single_gripper_input(
        side="right",
        requested_source="auto",
        q6_port="/dev/q6-right",
    )

    assert selection.resolved_source == GRIPPER_INPUT_Q6
    assert selection.q6_port == "/dev/q6-right"
    assert selection.fallback_reason is None


def test_auto_gripper_input_falls_back_when_no_q6_is_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        selection_module,
        "_load_configured_q6_port",
        lambda **_kwargs: (None, "q6_config_missing"),
    )
    monkeypatch.setattr(
        selection_module,
        "_scan_reachable_q6",
        lambda: ([], None),
    )

    selection = resolve_single_gripper_input(
        side="right",
        requested_source="auto",
    )

    assert selection.resolved_source == GRIPPER_INPUT_ANGLE_TRIGGER
    assert selection.q6_port is None
    assert selection.fallback_reason == "q6_config_missing"


def test_explicit_angle_trigger_skips_all_q6_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        selection_module,
        "_load_configured_q6_port",
        lambda **_kwargs: pytest.fail("Q6 config must not be loaded"),
    )
    monkeypatch.setattr(
        selection_module,
        "_scan_reachable_q6",
        lambda: pytest.fail("Q6 scan must not run"),
    )

    selection = resolve_single_gripper_input(
        side="right",
        requested_source="angle_trigger",
    )

    assert selection.resolved_source == GRIPPER_INPUT_ANGLE_TRIGGER
    assert selection.fallback_reason is None


def test_explicit_q6_fails_instead_of_silently_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        selection_module,
        "_load_configured_q6_port",
        lambda **_kwargs: (None, "q6_config_missing"),
    )
    monkeypatch.setattr(
        selection_module,
        "_scan_reachable_q6",
        lambda: ([], None),
    )

    with pytest.raises(RuntimeError, match="Requested Q6 input is unavailable"):
        resolve_single_gripper_input(
            side="right",
            requested_source="q6",
        )


def test_ambiguous_q6_discovery_requires_an_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        selection_module,
        "_load_configured_q6_port",
        lambda **_kwargs: (None, "q6_config_missing"),
    )
    monkeypatch.setattr(
        selection_module,
        "_scan_reachable_q6",
        lambda: (
            [
                {"preferred_port": "/dev/q6-a"},
                {"preferred_port": "/dev/q6-b"},
            ],
            None,
        ),
    )

    with pytest.raises(ValueError, match="no unambiguous right-side assignment"):
        resolve_single_gripper_input(
            side="right",
            requested_source="auto",
        )


class _FakeAngleReader:
    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def read(self) -> dict[str, float]:
        return {"left": 0.0, "right": 0.5}


def _right_calibration() -> TeleopCalibration:
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


def test_passive_single_vive_angle_trigger_uses_common_gripper_retargeter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def load_reader(**kwargs: object) -> _FakeAngleReader:
        calls.append(dict(kwargs))
        return _FakeAngleReader()

    monkeypatch.setattr(
        "hardware.angle_trigger.angle_trigger.load_configured_reader",
        load_reader,
    )

    safety_config = TeleopSafetyConfig(
        max_translation_jump_m=0.45,
        max_rotation_jump_deg=160.0,
        pose_jump_baseline_max_gap_sec=0.25,
    )
    session = build_passive_single_vive_gripper_session(
        calibration=_right_calibration(),
        source_config=PassiveSingleViveGripperSessionConfig(
            side="right",
            gripper_input_source="angle_trigger",
            angle_trigger_config_root=Path("/tmp/angle-trigger-test"),
        ),
        safety_config=safety_config,
    )

    assert session.rig is None
    assert session.safety_monitor.config is safety_config
    assert calls == []
    assert [source.name for source in session.sources] == [
        "vive_pose",
        "angle_trigger",
    ]
    tracker = session.sources[0].tracker_configs["tracker_1"]
    assert tracker.role == "left_wrist"
    assert tracker.channel_name == "right_wrist_pose"
    session.sources[1].start()
    try:
        assert calls == [
            {
                "config_root": Path("/tmp/angle-trigger-test"),
                "baudrate": 115200,
                "active_sides": ("right",),
            }
        ]
    finally:
        session.sources[1].close()
    hand_retargeter = session.retargeters[1]
    assert hand_retargeter.config.input_channel_name == "right_angle_trigger"
    assert hand_retargeter.config.source_dim is None
    assert hand_retargeter.config.effector_name == "right_hand"
    arm_retargeter = session.retargeters[0]
    assert arm_retargeter.config.output_frame_name == "right_arm_base"
    observation = TeleopObservation(
        channels={
            "right_wrist_pose": ObservationChannel(
                channel_name="right_wrist_pose",
                capability="pose6d",
                side="right",
                value=np.eye(4),
                frame_name="vive_world",
                timestamp=1.0,
                source_name="test",
            ),
            "right_angle_trigger": ObservationChannel(
                channel_name="right_angle_trigger",
                capability="scalar",
                side="right",
                value=0.5,
                frame_name=None,
                timestamp=1.0,
                source_name="angle_trigger",
            ),
        },
        timestamp=1.0,
    )
    command = arm_retargeter.build_command_from_target_pose_desk(
        target_pose_desk=np.eye(4),
        observation=observation,
        calibration=session.calibration,
    )
    assert command.commands["right_arm"].metadata["arm_action_spec"] == (
        ArmActionSpec.tool_tcp(
            "right",
            reference_frame="right_arm_base",
        ).to_metadata()
    )
    merged = session.build_command(observation)
    assert set(merged.commands) == {"right_arm", "right_hand"}
    np.testing.assert_allclose(
        merged.commands["right_hand"].value,
        np.asarray([0.5, 0, 0, 0, 0, 0], dtype=np.float32),
    )
