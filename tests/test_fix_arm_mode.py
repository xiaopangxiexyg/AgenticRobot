from __future__ import annotations

import json
import tempfile
import unittest

import numpy as np

from teleop_runtime.calibration import ArmCalibration, TeleopCalibration
from teleop_runtime.diagnostics import FaultEvent, FaultSeverity
from teleop_runtime.gui.calibration_web_ui import CalibrationWebUi
from teleop_runtime.schemas import EffectorCommand, ObservationChannel, TeleopCommand
from teleop_runtime.special_tasks.fix_arm import (
    FixArmCommandController,
    FixArmPoseSnapshotRecorder,
    build_fix_arm_pose_snapshot_payload,
    advance_fix_arm_home_capture,
)


def _command(
    *,
    left_arm: float,
    right_arm: float,
    left_hand: float,
    right_hand: float,
) -> TeleopCommand:
    command = TeleopCommand(timestamp=1.0)
    command.add_command(
        EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=np.full(6, left_arm, dtype=float),
        )
    )
    command.add_command(
        EffectorCommand(
            effector_name="right_arm",
            side="right",
            command_type="joint_vector",
            value=np.full(6, right_arm, dtype=float),
        )
    )
    command.add_command(
        EffectorCommand(
            effector_name="left_hand",
            side="left",
            command_type="joint_vector",
            value=np.full(6, left_hand, dtype=float),
        )
    )
    command.add_command(
        EffectorCommand(
            effector_name="right_hand",
            side="right",
            command_type="joint_vector",
            value=np.full(6, right_hand, dtype=float),
        )
    )
    return command


def _controller(selected_side: str = "left") -> FixArmCommandController:
    return FixArmCommandController(
        active_sides=("left", "right"),
        selected_side=selected_side,
        required_effectors_by_side={
            "left": ("left_arm", "left_hand"),
            "right": ("right_arm", "right_hand"),
        },
    )


def _channel(name: str, side: str, value: float, timestamp: float = 1.0) -> ObservationChannel:
    return ObservationChannel(
        channel_name=name,
        capability="pose6d" if name.endswith("_wrist_pose") else "joint_vector",
        side=side,
        value=np.eye(4) * value if name.endswith("_wrist_pose") else np.full(6, value),
        frame_name="desk" if name.endswith("_wrist_pose") else None,
        timestamp=timestamp,
        source_name="test",
    )


def _channels(
    *,
    left_wrist: float,
    right_wrist: float,
    left_hand: float,
    right_hand: float,
    timestamp: float = 1.0,
) -> dict[str, ObservationChannel]:
    return {
        "left_wrist_pose": _channel("left_wrist_pose", "left", left_wrist, timestamp),
        "right_wrist_pose": _channel("right_wrist_pose", "right", right_wrist, timestamp),
        "left_hand_q6": _channel("left_hand_q6", "left", left_hand, timestamp),
        "right_hand_q6": _channel("right_hand_q6", "right", right_hand, timestamp),
    }


class FixArmCommandControllerTests(unittest.TestCase):
    def test_freeze_reuses_fixed_side_input_channels_and_leaves_other_side_live(self) -> None:
        controller = _controller("left")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )

        transition = controller.freeze_selected(now=1.1)
        self.assertTrue(transition.ok)

        filtered_channels = controller.filter_channels(
            _channels(left_wrist=10.0, right_wrist=20.0, left_hand=30.0, right_hand=40.0),
            now=1.2,
        )

        np.testing.assert_allclose(filtered_channels["left_wrist_pose"].value, np.eye(4) * 1.0)
        np.testing.assert_allclose(filtered_channels["left_hand_q6"].value, np.full(6, 3.0))
        np.testing.assert_allclose(filtered_channels["right_wrist_pose"].value, np.eye(4) * 20.0)
        np.testing.assert_allclose(filtered_channels["right_hand_q6"].value, np.full(6, 40.0))
        self.assertEqual(filtered_channels["left_wrist_pose"].timestamp, 1.2)
        self.assertTrue(filtered_channels["left_wrist_pose"].metadata["fix_arm_frozen"])

    def test_freeze_overrides_pipeline_command_when_present(self) -> None:
        controller = _controller("left")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)

        filtered = controller.filter_command(
            _command(left_arm=10.0, right_arm=20.0, left_hand=30.0, right_hand=40.0),
            None,
            1.2,
        )

        self.assertEqual(
            set(filtered.commands),
            {"left_arm", "right_arm", "left_hand", "right_hand"},
        )
        self.assertEqual(filtered.metadata["fix_arm_active_side"], "left")
        self.assertEqual(filtered.metadata["fix_arm_mode"], "fixed")
        np.testing.assert_allclose(filtered.commands["left_arm"].value, np.full(6, 1.0))
        np.testing.assert_allclose(filtered.commands["left_hand"].value, np.full(6, 3.0))
        np.testing.assert_allclose(filtered.commands["right_arm"].value, np.full(6, 20.0))
        np.testing.assert_allclose(filtered.commands["right_hand"].value, np.full(6, 40.0))
        self.assertTrue(filtered.commands["left_arm"].metadata["fix_arm_active"])
        self.assertEqual(filtered.commands["left_arm"].metadata["fix_arm_side"], "left")
        for effector_command in filtered.commands.values():
            np.testing.assert_equal(np.asarray(effector_command.value).shape, (6,))

    def test_freeze_does_not_inject_fixed_command_when_pipeline_omits_it_by_default(self) -> None:
        controller = _controller("left")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)

        right_only = TeleopCommand(timestamp=1.2)
        right_only.add_command(
            EffectorCommand(
                effector_name="right_arm",
                side="right",
                command_type="joint_vector",
                value=np.full(6, 20.0),
            )
        )

        filtered = controller.filter_command(right_only, None, 1.2)

        self.assertEqual(set(filtered.commands), {"right_arm"})
        np.testing.assert_allclose(filtered.commands["right_arm"].value, np.full(6, 20.0))

    def test_freeze_can_inject_allowed_fixed_effector_for_out_of_band_dispatch(self) -> None:
        controller = _controller("left")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)

        hand_only = controller.filter_command(
            TeleopCommand(timestamp=1.2),
            None,
            1.2,
            include_missing_fixed_effectors=True,
            allowed_effectors=("left_hand",),
        )

        self.assertEqual(set(hand_only.commands), {"left_hand"})
        np.testing.assert_allclose(hand_only.commands["left_hand"].value, np.full(6, 3.0))

    def test_freeze_requires_previous_commands_for_all_required_effectors(self) -> None:
        controller = _controller("right")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        command = TeleopCommand()
        command.add_command(
            EffectorCommand(
                effector_name="right_arm",
                side="right",
                command_type="joint_vector",
                value=np.zeros(6),
            )
        )
        controller.filter_command(command, None, 1.0)

        transition = controller.freeze_selected(now=1.1)

        self.assertFalse(transition.ok)
        self.assertEqual(transition.missing_effectors, ("right_hand",))
        self.assertFalse(controller.fixed)

    def test_fault_filter_ignores_only_fixed_side_tracker_faults(self) -> None:
        controller = _controller("left")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        controller.freeze_selected(now=1.1)
        faults = [
            FaultEvent(
                code="stale_channel",
                message="left stale",
                severity=FaultSeverity.ERROR,
                timestamp=1.2,
                channel_name="left_wrist_pose",
            ),
            FaultEvent(
                code="invalid_channel",
                message="left hand invalid",
                severity=FaultSeverity.ERROR,
                timestamp=1.2,
                channel_name="left_hand_q6",
            ),
            FaultEvent(
                code="invalid_channel",
                message="right invalid",
                severity=FaultSeverity.ERROR,
                timestamp=1.2,
                channel_name="right_wrist_pose",
            ),
            FaultEvent(
                code="low_control_rate",
                message="slow",
                severity=FaultSeverity.WARNING,
                timestamp=1.2,
            ),
        ]

        filtered = controller.filter_faults(faults, {}, 1.2)

        self.assertEqual([fault.message for fault in filtered], ["right invalid", "slow"])

    def test_cancel_active_releases_fixed_side_for_hold(self) -> None:
        controller = _controller("left")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)

        transition = controller.cancel_active(
            now=1.2,
            reason="hold_requested",
            clear_snapshots=True,
        )

        self.assertTrue(transition.ok)
        self.assertFalse(controller.fixed)
        self.assertEqual(controller.fixed_side, "")
        self.assertIn("stopped", controller.status_message)


class FixArmRuntimeHomeTests(unittest.TestCase):
    def test_incremental_home_capture_updates_only_fixed_side_and_releases(self) -> None:
        controller = _controller("left")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)
        self.assertTrue(
            controller.start_recalibration(now=1.2, timeout_sec=5.0, min_samples=2).ok
        )
        calibration = TeleopCalibration(
            arms={
                "left_arm": ArmCalibration(name="left_arm", side="left"),
                "right_arm": ArmCalibration(name="right_arm", side="right"),
            }
        )
        channel_pose = np.eye(4)
        channel = ObservationChannel(
            channel_name="left_wrist_pose",
            capability="pose6d",
            side="left",
            value=channel_pose,
            frame_name="desk",
            timestamp=1.3,
            source_name="test",
        )
        robot_home = np.eye(4)
        robot_home[:3, 3] = [0.1, 0.2, 0.3]

        class Retargeter:
            def __init__(self) -> None:
                self.anchors = None

            def set_home_anchors(self, *, human_pose_desk, robot_pose_desk) -> None:
                self.anchors = (human_pose_desk.copy(), robot_pose_desk.copy())

        retargeter = Retargeter()

        progress = advance_fix_arm_home_capture(
            controller=controller,
            channels={"left_wrist_pose": channel},
            calibration=calibration,
            now=1.3,
            pose_from_channel=lambda item, _calibration: np.asarray(item.value, dtype=float),
            mean_pose=lambda samples: samples[0],
            resolve_robot_home=lambda side: (robot_home.copy(), np.arange(6), "test_home"),
            side_retargeter=lambda side: retargeter,
            tcp_mode="tool_tcp",
        )
        self.assertTrue(progress.active)
        self.assertFalse(progress.completed)

        progress = advance_fix_arm_home_capture(
            controller=controller,
            channels={"left_wrist_pose": channel},
            calibration=calibration,
            now=1.4,
            pose_from_channel=lambda item, _calibration: np.asarray(item.value, dtype=float),
            mean_pose=lambda samples: samples[0],
            resolve_robot_home=lambda side: (robot_home.copy(), np.arange(6), "test_home"),
            side_retargeter=lambda side: retargeter,
            tcp_mode="tool_tcp",
        )

        self.assertTrue(progress.completed)
        self.assertFalse(controller.fixed)
        np.testing.assert_allclose(
            calibration.arms["left_arm"].home_pose_for_tcp_mode(),
            robot_home,
        )
        np.testing.assert_allclose(calibration.captured_home_qpos["left_arm"], np.arange(6))
        self.assertIsNotNone(retargeter.anchors)
        self.assertIsNone(calibration.arms["right_arm"].home_pose_for_tcp_mode())


class _SnapshotRetargeter:
    home_human_pose_desk = np.eye(4)
    home_robot_pose_desk = np.eye(4) * 2.0
    last_target_pose_desk = np.eye(4) * 3.0


class FixArmPoseSnapshotTests(unittest.TestCase):
    def test_snapshot_recorder_writes_json_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = FixArmPoseSnapshotRecorder(tmp_dir)
            command = _command(
                left_arm=1.0,
                right_arm=2.0,
                left_hand=3.0,
                right_hand=4.0,
            )
            channels = _channels(
                left_wrist=1.0,
                right_wrist=2.0,
                left_hand=3.0,
                right_hand=4.0,
                timestamp=9.0,
            )
            calibration = TeleopCalibration(
                arms={
                    "left_arm": ArmCalibration(name="left_arm", side="left"),
                    "right_arm": ArmCalibration(name="right_arm", side="right"),
                }
            )

            payload = build_fix_arm_pose_snapshot_payload(
                timestamp=10.0,
                monotonic_time=20.0,
                step_index=7,
                active_sides=("left", "right"),
                teleop_state="running",
                control_hz=30.0,
                movement_multiple=1.0,
                tcp_mode="tool",
                calibration=calibration,
                raw_channels=channels,
                filtered_channels=channels,
                observation=None,
                step_command=command,
                active_command=command,
                hand_command=None,
                gripper_command=None,
                command_dispatched=True,
                faults=[],
                fix_arm_state={"fix_arm_available": True, "fix_arm_fixed_side": "left"},
                left_retargeter=_SnapshotRetargeter(),
                right_retargeter=_SnapshotRetargeter(),
                left_preview_qpos=np.arange(6),
                right_preview_qpos=np.arange(6) + 10,
                left_hardware_state={"current_qpos": np.arange(6)},
                right_hardware_state={"current_qpos": np.arange(6) + 10},
                left_command_state={"last_target_tcp_pose": np.eye(4)},
                right_command_state={"last_target_tcp_pose": np.eye(4) * 2.0},
                gripper_status={"left": {"sent": False}},
            )
            record = recorder.record(payload, now=10.0)

            self.assertEqual(record.sequence, 1)
            self.assertTrue(record.path.exists())
            self.assertTrue(recorder.index_path.exists())
            saved = json.loads(record.path.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema"], "dexteleop_fix_arm_pose_snapshot_v1")
            self.assertEqual(saved["step_index"], 7)
            self.assertEqual(saved["fix_arm"]["fix_arm_fixed_side"], "left")
            self.assertEqual(
                saved["commands"]["active_command"]["commands"]["right_arm"]["value"],
                [2.0] * 6,
            )
            index_rows = recorder.index_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(index_rows), 1)
            self.assertEqual(json.loads(index_rows[0])["path"], str(record.path))


class FixArmWebUiTests(unittest.TestCase):
    def test_fix_arm_select_event_requires_active_control(self) -> None:
        ui = CalibrationWebUi(open_browser=False)
        self.assertFalse(ui.push_event("fix_arm_select:left"))
        ui.publish(active_controls=["fix_arm_select"])

        self.assertTrue(ui.push_event("fix_arm_select:right"))
        self.assertEqual(ui.consume_events(), ["fix_arm_select:right"])

    def test_pose_snapshot_event_requires_active_control(self) -> None:
        ui = CalibrationWebUi(open_browser=False)
        self.assertFalse(ui.push_event("record_pose_snapshot"))
        ui.publish(active_controls=["record_pose_snapshot"])

        self.assertTrue(ui.push_event("record_pose_snapshot"))
        self.assertEqual(ui.consume_events(), ["record_pose_snapshot"])


if __name__ == "__main__":
    unittest.main()
