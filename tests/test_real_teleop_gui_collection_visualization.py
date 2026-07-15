from __future__ import annotations

from collections import deque
from contextlib import redirect_stderr
from io import StringIO
import json
import sys
from pathlib import Path
import time
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from teleop_runtime.backends.grippers import DMTacGripperBackend, DMTacGripperBackendConfig
from teleop_runtime.debug import run_vive_rm65_real_teleop as real_teleop
from teleop_runtime.gui import calibration_web_ui
from teleop_runtime.schemas import EffectorCommand, ObservationChannel, TeleopCommand
from teleop_runtime.sources.pose import VivePoseSource
from teleop_runtime.tactile_visualizer import DMTacDistributedForceVisualizer


class _FakeTerminal:
    enabled = False
    should_quit = False

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def poll(self) -> None:
        pass

    def consume_calibration_request(self) -> bool:
        return False

    def consume_episode_start_request(self) -> bool:
        return False

    def consume_episode_stop_request(self) -> bool:
        return False


class _FakeUi:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = list(events or [])
        self.published: list[dict[str, object]] = []
        self.messages: list[tuple[str, str]] = []

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def publish(self, **kwargs: object) -> None:
        self.published.append(dict(kwargs))

    def append_message(self, title: str, message: str) -> None:
        self.messages.append((title, message))

    def consume_events(self) -> list[str]:
        events = list(self.events)
        self.events.clear()
        return events


class _BatchedFakeUi(_FakeUi):
    def __init__(self, event_batches: list[list[str]]) -> None:
        super().__init__([])
        self.event_batches = [list(batch) for batch in event_batches]

    def consume_events(self) -> list[str]:
        if not self.event_batches:
            return []
        return self.event_batches.pop(0)


class _FakeCalibration:
    def __init__(self) -> None:
        self.tracker_offsets = {"right_wrist": object()}
        self.metadata: dict[str, object] = {}

    def world_to_desk_pose(self) -> object:
        return object()

    def world_to_desk(self, pose: object) -> object:
        return pose

    def to_dict(self) -> dict[str, object]:
        return {"format": "fake_calibration"}


class _FakeCameraSource:
    def __init__(
        self,
        info: dict[str, object],
        *,
        enable_depth: bool = False,
        frames: list[object] | None = None,
    ) -> None:
        self._info = dict(info)
        self._frames = list(frames or [])
        self._frame_cursor = 0
        self.config = SimpleNamespace(
            serial="CONFIG_SERIAL",
            width=640,
            height=480,
            fps=30,
            color_space="bgr",
            enable_depth=bool(enable_depth),
            depth_width=640,
            depth_height=480,
            depth_fps=30,
        )
        self.name = "fake_camera"

    def device_info(self) -> dict[str, object]:
        return dict(self._info)

    def frame_buffer(self) -> object:
        return self

    def get_latest(self) -> object | None:
        if not self._frames:
            return None
        frame = self._frames[min(self._frame_cursor, len(self._frames) - 1)]
        if self._frame_cursor < len(self._frames) - 1:
            self._frame_cursor += 1
            self._info["frame_count"] = int(self._info.get("frame_count", 0) or 0) + 1
        return frame


def _fake_depth_frame(
    *,
    frame_index: int,
    source_timestamp: float = 100.0,
    depth_frame_number: int = 13,
    depth_camera_timestamp_sec: float = 99.0,
    depth_sensor_timestamp_ms: float = 123.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        frame_index=int(frame_index),
        source_timestamp=float(source_timestamp),
        depth_image=np.ones((4, 4), dtype=np.uint16),
        depth_metadata={
            "depth_scale_m_per_unit": 0.001,
            "frame_number": int(depth_frame_number),
            "camera_timestamp_sec": float(depth_camera_timestamp_sec),
            "sensor_timestamp_ms": float(depth_sensor_timestamp_ms),
        },
    )


class _FakeViveSource:
    def __init__(self) -> None:
        pose = np.eye(4)
        now = time.time()
        self.channels = {
            "left_wrist_pose": ObservationChannel(
                channel_name="left_wrist_pose",
                capability="pose",
                side="left",
                value=pose,
                frame_name="world",
                timestamp=now,
                source_name="vive",
                valid=True,
                metadata={"seq": 1, "tracker_name": "tracker_1", "tracker_role": "left_wrist"},
            ),
            "right_wrist_pose": ObservationChannel(
                channel_name="right_wrist_pose",
                capability="pose",
                side="right",
                value=pose,
                frame_name="world",
                timestamp=now,
                source_name="vive",
                valid=True,
                metadata={"seq": 1, "tracker_name": "tracker_2", "tracker_role": "right_wrist"},
            ),
        }

    def get_latest_channels(self) -> dict[str, ObservationChannel]:
        return dict(self.channels)


def _components() -> SimpleNamespace:
    left_retargeter = SimpleNamespace(config=SimpleNamespace(translation_scale=1.0))
    right_retargeter = SimpleNamespace(config=SimpleNamespace(translation_scale=1.0))
    return SimpleNamespace(
        hand_backend_mode=real_teleop.HAND_BACKEND_DMTAC_GRIPPER,
        active_sides=("right",),
        left_hand_backend=None,
        right_hand_backend=None,
        left_gripper_retargeter=None,
        right_gripper_retargeter=None,
        session=SimpleNamespace(
            sources=[],
            recorder=None,
            calibration=_FakeCalibration(),
        ),
        collision_guard_requested=False,
        dual_arm_retargeter=None,
        left_retargeter=left_retargeter,
        right_retargeter=right_retargeter,
        logical_arm_ip_mapping={},
        loaded_initial_alignment=None,
        gripper_policy={},
    )


def _args(**overrides: object) -> SimpleNamespace:
    payload: dict[str, object] = {
        "host": "127.0.0.1",
        "port": 9999,
        "control_mode": "absolute",
        "hz": 15.0,
        "movement_multiple": 1.0,
        "translation_scale": 1.0,
        "left_right_base_distance": 0.525,
        "collision_scene_profile": "dual_arm_hand_proxy_open",
        "collision_avoidance_mode": "boundary_slide",
        "load_profile": "",
        "load_initial_alignment": "",
        "debug_ik_timing": False,
        "no_record": True,
        "debug_trajectory_recording": False,
        "no_episode_recording": False,
        "camera_visualization_available": True,
        "tactile_visualization_available": True,
        "vis_cam": False,
        "visualize_tactile": False,
        "hand_backend": real_teleop.HAND_BACKEND_DMTAC_GRIPPER,
        "dataset_root": "",
        "dataset_base_root": "",
        "dataset_dir_name": "",
        "dataset_name": "",
        "session_record_root": "",
        "episode_root": "",
        "no_collection": False,
        "collection_warning_hz": 15.0,
        "collection_task_id": "",
        "collection_task_name": "",
        "collection_task_name_zh": "",
        "collection_task_description": "",
        "task_config": "",
        "left_hand_ip": "",
        "right_hand_ip": "",
        "left_hand_port": "",
        "right_hand_port": "",
        "gripper_task_profile": "",
        "left_tracker_name": "tracker_1",
        "right_tracker_name": "tracker_2",
        "aux_tracker_name": "",
        "teleop_hardware_ip": "",
        "teleop_hardware_port": 0,
        "swap_arm_sides": False,
        "swap_operator_sides": False,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


class RealTeleopGuiCollectionVisualizationTests(unittest.TestCase):
    def test_preview_apply_does_not_mutate_recorded_active_command(self) -> None:
        class _MutatingPreviewBackend:
            def __init__(self) -> None:
                self.received: list[EffectorCommand] = []

            def send_command(self, command: EffectorCommand) -> None:
                command.metadata["action_pose_source"] = "preview_mutation"
                command.metadata["target_pose_base"] = np.eye(4)
                self.received.append(command)

        left_preview = _MutatingPreviewBackend()
        right_preview = _MutatingPreviewBackend()
        components = SimpleNamespace(
            left_preview_backend=left_preview,
            right_preview_backend=right_preview,
        )
        command = TeleopCommand()
        left_command = EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="cartesian_pose",
            value=np.eye(4),
            metadata={"action_pose_source": "limited_solver_output_qpos"},
        )
        right_command = EffectorCommand(
            effector_name="right_arm",
            side="right",
            command_type="cartesian_pose",
            value=np.eye(4),
            metadata={"action_pose_source": "limited_solver_output_qpos"},
        )
        command.add_command(left_command)
        command.add_command(right_command)

        real_teleop._apply_active_arm_command_to_preview_real(components, command)

        self.assertEqual(left_command.metadata["action_pose_source"], "limited_solver_output_qpos")
        self.assertEqual(right_command.metadata["action_pose_source"], "limited_solver_output_qpos")
        self.assertNotIn("target_pose_base", left_command.metadata)
        self.assertNotIn("target_pose_base", right_command.metadata)
        self.assertIsNot(left_preview.received[0], left_command)
        self.assertIsNot(right_preview.received[0], right_command)

    def test_parse_args_defaults_load_profile_to_newest_calibration_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_dir = Path(tmp_dir)
            older = profile_dir / "20260701_120000_vive_rm65_real_teleop.json"
            newer = profile_dir / "20260702_090000_vive_rm65_real_teleop.json"
            older.write_text("{}", encoding="utf-8")
            newer.write_text("{}", encoding="utf-8")
            with patch.object(real_teleop, "DEFAULT_CALIBRATION_PROFILE_DIR", profile_dir), patch.object(
                sys,
                "argv",
                [
                    "prog",
                    "--active-sides",
                    "left",
                    "--left-arm-ip",
                    "192.0.2.10",
                    "--no-arm-ip-save-latest",
                    "--no-camera",
                    "--no-video",
                    "--headless",
                    "--no-gui",
                ],
            ):
                args = real_teleop.parse_args()

        self.assertEqual(args.load_profile, str(newer))

    def test_parse_args_errors_when_default_load_profile_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(real_teleop, "DEFAULT_CALIBRATION_PROFILE_DIR", Path(tmp_dir)), patch.object(
                sys,
                "argv",
                [
                    "prog",
                    "--active-sides",
                    "left",
                    "--left-arm-ip",
                    "192.0.2.10",
                    "--no-camera",
                    "--no-video",
                    "--headless",
                    "--no-gui",
                ],
            ):
                with self.assertRaises(SystemExit) as ctx:
                    real_teleop.parse_args()

        self.assertEqual(ctx.exception.code, 2)

    def test_parse_args_accepts_init_arm_available_typo_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_dir = Path(tmp_dir)
            calibration_profile = profile_dir / "20260702_090000_vive_rm65_real_teleop.json"
            init_profile = profile_dir / "init_arm.json"
            calibration_profile.write_text("{}", encoding="utf-8")
            init_profile.write_text("{}", encoding="utf-8")
            with patch.object(real_teleop, "DEFAULT_CALIBRATION_PROFILE_DIR", profile_dir), patch.object(
                sys,
                "argv",
                [
                    "prog",
                    "--active-sides",
                    "both",
                    "--left-arm-ip",
                    "192.0.2.10",
                    "--right-arm-ip",
                    "192.0.2.11",
                    "--no-arm-ip-save-latest",
                    "--no-camera",
                    "--no-video",
                    "--headless",
                    "--no-gui",
                    "--init-arm-availabe",
                    "--init-arm-profile",
                    str(init_profile),
                ],
            ):
                args = real_teleop.parse_args()

        self.assertTrue(args.init_arm_available)
        self.assertEqual(args.init_arm_profile, str(init_profile))

    def test_parse_args_rejects_special_arm_modes_for_single_arm_before_device_setup(self) -> None:
        for mode_flag in ("--fix-arm", "--init-arm-available"):
            stderr = StringIO()
            with self.subTest(mode_flag=mode_flag), patch.object(
                sys,
                "argv",
                [
                    "prog",
                    "--active-sides",
                    "left",
                    mode_flag,
                    "--init-arm-profile",
                    "unused-init-profile.json",
                    "--no-camera",
                    "--no-video",
                    "--headless",
                    "--no-gui",
                ],
            ), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as ctx:
                    real_teleop.parse_args()

            self.assertEqual(ctx.exception.code, 2)
            self.assertIn("requires --active-sides both", stderr.getvalue())

    def test_parse_args_accepts_workspace_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_dir = Path(tmp_dir)
            profile = profile_dir / "20260702_090000_vive_rm65_real_teleop.json"
            profile.write_text("{}", encoding="utf-8")
            with patch.object(real_teleop, "DEFAULT_CALIBRATION_PROFILE_DIR", profile_dir), patch.object(
                sys,
                "argv",
                [
                    "prog",
                    "--active-sides",
                    "left",
                    "--left-arm-ip",
                    "192.0.2.10",
                    "--no-arm-ip-save-latest",
                    "--no-camera",
                    "--no-video",
                    "--headless",
                    "--no-gui",
                    "--arm-workspace-min",
                    "-0.65",
                    "-0.80",
                    "0.00",
                    "--arm-workspace-max=0.80,0.80,0.95",
                    "--left-arm-workspace-max",
                    "0.75",
                    "0.70",
                    "0.90",
                ],
            ):
                args = real_teleop.parse_args()

        self.assertEqual(args.arm_workspace_min, (-0.65, -0.80, 0.00))
        self.assertEqual(args.arm_workspace_max, (0.80, 0.80, 0.95))
        self.assertEqual(args.left_arm_workspace_max, (0.75, 0.70, 0.90))

    def test_workspace_override_applies_side_specific_bounds(self) -> None:
        calibration = real_teleop.build_annotated_dual_rm65_calibration()
        args = SimpleNamespace(
            arm_workspace_min=(-0.65, -0.80, 0.00),
            arm_workspace_max=(0.80, 0.80, 0.95),
            left_arm_workspace_min=None,
            left_arm_workspace_max=(0.75, 0.70, 0.90),
            right_arm_workspace_min=None,
            right_arm_workspace_max=None,
        )

        applied = real_teleop._apply_workspace_overrides(args, calibration)

        self.assertIn("left_arm", applied)
        self.assertIn("right_arm", applied)
        np.testing.assert_allclose(
            calibration.arms["left_arm"].workspace_min,
            np.array([-0.65, -0.80, 0.00]),
        )
        np.testing.assert_allclose(
            calibration.arms["left_arm"].workspace_max,
            np.array([0.75, 0.70, 0.90]),
        )
        np.testing.assert_allclose(
            calibration.arms["right_arm"].workspace_max,
            np.array([0.80, 0.80, 0.95]),
        )
        self.assertTrue(calibration.arms["left_arm"].metadata["workspace_override_runtime"])

    def test_workspace_config_applies_default_bounds(self) -> None:
        calibration = real_teleop.build_annotated_dual_rm65_calibration()
        args = SimpleNamespace(
            arm_workspace_config=str(real_teleop.DEFAULT_ARM_WORKSPACE_CONFIG_PATH),
        )

        applied = real_teleop._apply_workspace_config(args, calibration)

        self.assertIn("left_arm", applied)
        np.testing.assert_allclose(
            calibration.arms["left_arm"].workspace_max,
            np.array([0.80, 0.80, 0.90]),
        )
        np.testing.assert_allclose(
            calibration.arms["right_arm"].workspace_max,
            np.array([0.80, 0.80, 0.90]),
        )

    def test_load_profile_does_not_override_workspace_config(self) -> None:
        loaded = real_teleop.build_annotated_dual_rm65_calibration()
        loaded.arms["left_arm"].workspace_max = np.array([0.60, 0.80, 0.90])
        loaded.arms["right_arm"].workspace_max = np.array([0.60, 0.80, 0.90])
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_path = Path(tmp_dir) / "old_profile.json"
            loaded.save_json(profile_path)

            current = real_teleop.build_annotated_dual_rm65_calibration()
            left_retargeter = SimpleNamespace(clear_home_anchors=lambda: None)
            right_retargeter = SimpleNamespace(clear_home_anchors=lambda: None)
            components = SimpleNamespace(
                session=SimpleNamespace(calibration=current),
                left_retargeter=left_retargeter,
                right_retargeter=right_retargeter,
            )
            args = SimpleNamespace(
                load_profile=str(profile_path),
                arm_workspace_config=str(real_teleop.DEFAULT_ARM_WORKSPACE_CONFIG_PATH),
                arm_workspace_min=None,
                arm_workspace_max=None,
                left_arm_workspace_min=None,
                left_arm_workspace_max=None,
                right_arm_workspace_min=None,
                right_arm_workspace_max=None,
            )

            real_teleop._load_gui_vive_calibration_profile(args, components)

        np.testing.assert_allclose(
            components.session.calibration.arms["left_arm"].workspace_max,
            np.array([0.80, 0.80, 0.90]),
        )
        np.testing.assert_allclose(
            components.session.calibration.arms["right_arm"].workspace_max,
            np.array([0.80, 0.80, 0.90]),
        )

    def test_saved_calibration_profile_payload_omits_workspace_bounds(self) -> None:
        calibration = real_teleop.build_annotated_dual_rm65_calibration()
        payload = real_teleop._calibration_profile_payload_without_workspace(
            calibration,
            workspace_config_path=real_teleop.DEFAULT_ARM_WORKSPACE_CONFIG_PATH,
        )

        for arm_payload in payload["arms"].values():
            self.assertNotIn("workspace_min", arm_payload)
            self.assertNotIn("workspace_max", arm_payload)
        self.assertEqual(
            payload["metadata"]["workspace_policy"],
            "managed_by_teleop_runtime_config",
        )

    def test_ik_debug_log_defaults_to_compact_payload(self) -> None:
        pose = np.eye(4)
        target_pose = np.eye(4)
        target_pose[0, 3] = 0.03
        qpos = np.zeros(6)
        large_init = {"robot_model": list(range(256))}
        profile = {
            "frame_total_ms": 3.2,
            "solve_total_ms": 1.1,
            "provider_call_count": 1,
            "ik_resolver_init": large_init,
            "min_distance_phase_ms": {"final": 0.2},
        }
        command = real_teleop.TeleopCommand(
            timestamp=1.0,
            metadata={
                "collision_avoidance_status": "tracking",
                "ik_debug_profile": profile,
                "temporary_dual_arm_safety": {
                    "enabled": True,
                    "limited": True,
                    "effectors": {
                        "left_arm": {
                            "limited": True,
                            "raw_translation_step_m": 0.04,
                            "applied_translation_step_m": 0.015,
                        }
                    },
                },
            },
        )
        command.add_command(
            real_teleop.EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="cartesian_pose",
                value=pose,
                frame_name="arm_base",
                metadata={
                    "target_pose_base": pose,
                    "solver_output_endpoint_pose_base": pose,
                    "solver_output_qpos": qpos,
                    "ik_debug_profile": profile,
                    "execution_step_metadata": {
                        "enabled": True,
                        "step_limited": True,
                        "joint_velocity_limit_ratio": 2.0,
                        "joint_velocity_limit_requested_delta_deg": list(range(12)),
                    },
                },
            )
        )
        backend_state = {
            "current_qpos": qpos,
            "last_target_qpos": qpos,
            "current_tcp_pose": pose,
            "last_target_tcp_pose": target_pose,
            "ik_metadata": {
                "ik_backend_solve_ms": 4.2,
                "ik_resolver_init": large_init,
            },
            "last_impedance_record": {
                "pose_tracking_linear_m": 0.03,
                "ik_metadata": {
                    "ik_backend_solve_ms": 4.2,
                    "ik_resolver_init": large_init,
                },
            },
        }
        preview = SimpleNamespace(get_state=lambda: {"current_qpos": qpos})
        components = SimpleNamespace(left_preview_backend=preview, right_preview_backend=preview)
        step_result = SimpleNamespace(
            state=SimpleNamespace(value="running"),
            control_hz=12.5,
            faults=[],
            channels={
                "left_wrist_pose": SimpleNamespace(
                    valid=True,
                    timestamp=1.0,
                    frame_name="world",
                    source_name="vive",
                    value=pose,
                )
            },
        )

        frame = real_teleop._build_ik_debug_log_frame(
            components=components,
            step_index=7,
            loop_start=1.0,
            step_result=step_result,
            active_command=command,
            left_command_state=backend_state,
            right_command_state=backend_state,
            left_hw_state=backend_state,
            right_hw_state=backend_state,
            ik_timing_payload={"available": True, "frame_total_ms": 3.2},
            target_step_scale=1.0,
            left_clutch=False,
            right_clutch=False,
        )

        encoded = json.dumps(real_teleop._json_safe(frame), ensure_ascii=False)
        self.assertEqual(frame["log_mode"], "compact")
        self.assertNotIn("ik_resolver_init", encoded)
        self.assertNotIn('"matrix"', encoded)
        self.assertIn('"pose_tracking_error"', encoded)
        self.assertIn('"temporary_dual_arm_safety"', encoded)
        self.assertTrue(
            frame["retargeter_command"]["left"]["metadata"]["execution_step_metadata"]["step_limited"]
        )

    def test_ik_debug_log_full_mode_keeps_historical_payload(self) -> None:
        pose = np.eye(4)
        qpos = np.zeros(6)
        large_init = {"robot_model": [1, 2, 3]}
        command = real_teleop.TeleopCommand(
            metadata={"ik_debug_profile": {"ik_resolver_init": large_init}}
        )
        command.add_command(
            real_teleop.EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="cartesian_pose",
                value=pose,
                frame_name="arm_base",
                metadata={"target_pose_base": pose},
            )
        )
        backend_state = {
            "current_qpos": qpos,
            "last_target_qpos": qpos,
            "current_tcp_pose": pose,
            "last_target_tcp_pose": pose,
            "ik_metadata": {"ik_resolver_init": large_init},
        }
        preview = SimpleNamespace(get_state=lambda: {"current_qpos": qpos})
        components = SimpleNamespace(left_preview_backend=preview, right_preview_backend=preview)
        step_result = SimpleNamespace(
            state=SimpleNamespace(value="running"),
            control_hz=15.0,
            faults=[],
            channels={},
        )

        frame = real_teleop._build_ik_debug_log_frame(
            components=components,
            step_index=1,
            loop_start=1.0,
            step_result=step_result,
            active_command=command,
            left_command_state=backend_state,
            right_command_state=backend_state,
            left_hw_state=backend_state,
            right_hw_state=backend_state,
            ik_timing_payload={},
            target_step_scale=1.0,
            left_clutch=False,
            right_clutch=False,
            full=True,
        )

        encoded = json.dumps(real_teleop._json_safe(frame), ensure_ascii=False)
        self.assertEqual(frame["log_mode"], "full")
        self.assertIn("ik_resolver_init", encoded)
        self.assertIn('"matrix"', encoded)

    def test_ik_debug_log_recorder_stop_flushes_stop_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = real_teleop.IkDebugLogRecorder(root_dir=tmp_dir)
            path = recorder.start(now=1.234)
            recorder.record_frame(
                {
                    "schema": "dexteleop_ik_debug_v1",
                    "event": "frame",
                    "step_index": 1,
                }
            )

            stopped_path = recorder.stop(reason="unit_test", now=2.0)

            self.assertEqual(stopped_path, path)
            lines = path.read_text(encoding="utf-8").splitlines()
            events = [json.loads(line)["event"] for line in lines]
            self.assertIn("start", events)
            self.assertIn("frame", events)
            self.assertEqual(events[-1], "stop")
            self.assertIn("_234", path.name)

    def test_ik_debug_log_recorder_defaults_to_teleop_ik_debug_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "teleop_records" / "ik_debug"
            with patch.object(real_teleop, "DEFAULT_IK_DEBUG_LOG_ROOT", root):
                recorder = real_teleop.IkDebugLogRecorder()
                path = recorder.start(now=1.0)
                recorder.stop(reason="unit_test", now=2.0)

            self.assertEqual(path.parent, root)
            self.assertTrue(path.exists())

    def test_debug_timing_log_recorder_records_slow_and_heartbeat_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = real_teleop.DebugTimingLogRecorder(
                root_dir=tmp_dir,
                heartbeat_sec=1.0,
            )
            path = recorder.start(now=1.234)

            reason = recorder.select_frame_reason(
                now=1.25,
                elapsed_sec=0.01,
                phases=(),
                state="RUNNING",
                fault_codes=(),
                threshold_sec=0.05,
            )
            self.assertEqual(reason, "state_change")
            recorder.record_frame({"schema": "dexteleop_debug_timing_v1", "event": "frame", "reason": reason})

            self.assertIsNone(
                recorder.select_frame_reason(
                    now=1.50,
                    elapsed_sec=0.01,
                    phases=(),
                    state="RUNNING",
                    fault_codes=(),
                    threshold_sec=0.05,
                )
            )

            reason = recorder.select_frame_reason(
                now=1.75,
                elapsed_sec=0.01,
                phases=(("loop.session_run_step", 0.08),),
                state="RUNNING",
                fault_codes=(),
                threshold_sec=0.05,
            )
            self.assertEqual(reason, "slow")
            recorder.record_frame({"schema": "dexteleop_debug_timing_v1", "event": "frame", "reason": reason})

            reason = recorder.select_frame_reason(
                now=2.80,
                elapsed_sec=0.01,
                phases=(),
                state="RUNNING",
                fault_codes=(),
                threshold_sec=0.05,
            )
            self.assertEqual(reason, "heartbeat")
            recorder.record_frame({"schema": "dexteleop_debug_timing_v1", "event": "frame", "reason": reason})

            stopped_path = recorder.stop(reason="unit_test", now=3.0)

            self.assertEqual(stopped_path, path)
            self.assertEqual(path.parent, Path(tmp_dir))
            self.assertIn("debug_timing_", path.name)
            self.assertIn("_234", path.name)
            lines = path.read_text(encoding="utf-8").splitlines()
            rows = [json.loads(line) for line in lines]
            self.assertEqual(rows[0]["event"], "start")
            self.assertEqual(rows[-1]["event"], "stop")
            self.assertEqual([row.get("reason") for row in rows if row.get("event") == "frame"], [
                "state_change",
                "slow",
                "heartbeat",
            ])

    def test_debug_timing_log_frame_keeps_lightweight_backend_timing(self) -> None:
        pose = np.eye(4)
        channel = ObservationChannel(
            channel_name="left_wrist_pose",
            capability="pose",
            side="left",
            value=pose,
            frame_name="world",
            timestamp=9.95,
            source_name="vive",
            valid=True,
            metadata={
                "sender_timestamp": 9.90,
                "receiver_timestamp": 9.94,
                "tracker_name": "tracker_1",
                "tracker_role": "left_wrist",
                "seq": 42,
            },
        )
        step_result = SimpleNamespace(
            timestamp=10.0,
            observation=SimpleNamespace(timestamp=9.98),
            command=SimpleNamespace(timestamp=9.99),
            state=SimpleNamespace(value="RUNNING"),
            control_hz=12.5,
            faults=[
                SimpleNamespace(
                    code="low_control_rate",
                    channel_name=None,
                    component="session",
                    severity=SimpleNamespace(value="warning"),
                    message="slow frame",
                    timestamp=10.0,
                    metadata={"hz": 12.5},
                )
            ],
            channels={"left_wrist_pose": channel},
        )
        backend_state = {
            "hardware_result_code": 0,
            "hardware_result_text": "ok",
            "send_result_code": 0,
            "send_result_text": "ok",
            "effective_step_scale": 1.2,
            "current_tcp_pose": pose,
            "active_tool_name": "DMTAC_M",
            "active_tool_profile_path": "tests/debug/rm65_left_DMTAC_M_angular_debug.json",
            "active_impedance_config": {
                "profile_path": "tests/debug/rm65_left_DMTAC_M_angular_debug.json",
                "mode": "pose",
                "axes": "rxyz",
                "angular_axis_signs": [-1.0, -1.0, -1.0],
                "angular_compliance_rad_per_nm": 0.1,
                "max_angular_offset_rad": 1.2,
                "direct_joint_hint_force_guard_enabled": False,
            },
            "last_send_command_timing": {
                "total_ms": 4.2,
                "send_lock_wait_ms": 1.1,
                "sdk_ms": 2.7,
            },
            "last_impedance_loop_timing": {
                "total_ms": 6.0,
                "lock_wait_ms": 0.4,
            },
            "last_impedance_record": {
                "step": 7,
                "dt_sec": 0.04,
                "active_wrench": [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
                "tare_corrected_wrench": [1.3, 2.2, 3.1, 0.3, 0.2, 0.4],
                "raw_deadbanded_wrench": [1.0, 2.0, 3.0, 0.2, 0.1, 0.3],
                "admittance_wrench": [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
                "return_activity_wrench": [1.3, 2.2, 3.1, 0.3, 0.2, 0.4],
                "target_offset_unclamped": [0.01, 0.0, 0.0, 0.0, 0.02, 0.0],
                "target_offset_after_return": [0.01, 0.0, 0.0, 0.0, 0.02, 0.0],
                "target_offset_after_offset_limit": [0.01, 0.0, 0.0, 0.0, 0.02, 0.0],
                "target_offset": [0.01, 0.0, 0.0, 0.0, 0.02, 0.0],
                "offset": [0.005, 0.0, 0.0, 0.0, 0.01, 0.0],
                "applied_offset": [0.004, 0.0, 0.0, 0.0, 0.008, 0.0],
                "offset_delta_before_velocity_limit": [0.006, 0.0, 0.0, 0.0, 0.012, 0.0],
                "offset_step_after_velocity_limit": [0.001, 0.0, 0.0, 0.0, 0.002, 0.0],
                "return_applied_axes": [],
                "return_paused_axes": [],
                "linear_return_paused": True,
                "angular_return_paused": True,
                "linear_offset_limit_applied": False,
                "angular_offset_limit_applied": False,
                "linear_velocity_limit_applied": False,
                "angular_velocity_limit_applied": False,
                "force_norm_n": 0.1,
                "reference_to_command_linear_m": [0.0, 0.0, 0.0],
                "reference_to_command_angular_rotvec_rad": [0.0, 0.02, 0.0],
                "reference_to_measured_linear_m": [-0.01, 0.0, 0.0],
                "reference_to_measured_angular_rotvec_rad": [0.0, -0.01, 0.0],
                "command_to_measured_linear_m": [-0.01, 0.0, 0.0],
                "command_to_measured_angular_rotvec_rad": [0.0, -0.03, 0.0],
                "measured_to_command_linear_m": [0.01, 0.0, 0.0],
                "measured_to_command_angular_rotvec_rad": [0.0, 0.03, 0.0],
                "command_to_measured_joint_deg": [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                "measured_to_command_joint_deg": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                "measured_velocity_dt_sec": 0.05,
                "measured_pose_velocity_linear_m_s": [0.1, 0.0, 0.0],
                "measured_pose_velocity_angular_rad_s": [0.0, 0.2, 0.0],
                "measured_joint_velocity_deg_s": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "tracking_error_closing_linear_m_s": 0.1,
                "tracking_error_closing_angular_rad_s": 0.2,
                "tracking_error_closing_joint_deg_s": 2.0,
                "ik_metadata": {
                    "status": "pass",
                    "method": "dual_arm_joint_hint",
                    "direct_joint_hint_reason": "offset_threshold",
                    "direct_joint_hint_offset_ignored": True,
                    "direct_joint_hint_force_guard": {"active": False},
                },
                "rm65_controller_timing": {"step_total_ms": 2.3, "sdk_read_ms": 0.8},
                "rm65_backend_loop_timing": {"total_ms": 6.0},
                "ik_timing": {"solve_total_ms": 1.5},
            },
            "recent_impedance_records": [
                {
                    "step": 6,
                    "dt_sec": 0.04,
                    "force_source": "work",
                    "force_readings": {
                        "raw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "work": [0.0, 0.0, 0.0, 0.2, 0.1, 0.3],
                        "tool": [0.0, 0.0, 0.0, 0.2, 0.1, 0.3],
                    },
                    "selected_force_wrench": [0.0, 0.0, 0.0, 0.2, 0.1, 0.3],
                    "tare_corrected_wrench": [0.0, 0.0, 0.0, 0.2, 0.1, 0.3],
                    "filtered_wrench": [0.0, 0.0, 0.0, 0.18, 0.09, 0.27],
                    "raw_deadbanded_wrench": [0.0, 0.0, 0.0, 0.15, 0.06, 0.24],
                    "return_activity_wrench": [0.0, 0.0, 0.0, 0.18, 0.09, 0.27],
                    "target_offset_unclamped": [0.0, 0.0, 0.0, 0.015, 0.006, 0.024],
                    "target_offset_after_return": [0.0, 0.0, 0.0, 0.015, 0.006, 0.024],
                    "target_offset_after_offset_limit": [0.0, 0.0, 0.0, 0.015, 0.006, 0.024],
                    "target_offset": [0.0, 0.0, 0.0, 0.015, 0.006, 0.024],
                    "offset": [0.0, 0.0, 0.0, 0.01, 0.004, 0.02],
                    "applied_offset": [0.0, 0.0, 0.0, 0.01, 0.004, 0.02],
                    "offset_delta_before_velocity_limit": [0.0, 0.0, 0.0, 0.005, 0.002, 0.004],
                    "offset_step_after_velocity_limit": [0.0, 0.0, 0.0, 0.005, 0.002, 0.004],
                    "return_applied_axes": [],
                    "return_paused_axes": [],
                    "linear_return_paused": False,
                    "angular_return_paused": True,
                    "linear_offset_limit_applied": False,
                    "angular_offset_limit_applied": False,
                    "linear_velocity_limit_applied": False,
                    "angular_velocity_limit_applied": False,
                }
            ],
        }

        frame = real_teleop._build_debug_timing_log_frame(
            components=SimpleNamespace(
                active_sides=("left", "right"),
                session=SimpleNamespace(last_fault_reason="main_loop: right_arm impedance fault"),
            ),
            step_index=11,
            loop_start=10.0,
            elapsed_sec=0.09,
            phases=(("loop.session_run_step", 0.08), ("loop.status_emit", 0.002)),
            reason="slow",
            step_result=step_result,
            active_command=None,
            left_command_state=backend_state,
            right_command_state={},
            left_hw_state=backend_state,
            right_hw_state={},
            ik_timing_payload={"available": True, "frame_total_ms": 3.0},
            target_step_scale=1.2,
            left_clutch=False,
            right_clutch=True,
            camera_sources={},
            preview_camera_sources={},
            camera_previewer=None,
            hand_input_status={"left": "ok"},
            gripper_status={"left": "control_disabled"},
            recording_status={"active": True, "frame_count": 3},
        )

        encoded = json.dumps(real_teleop._json_safe(frame), ensure_ascii=False)
        self.assertEqual(frame["schema"], "dexteleop_debug_timing_v1")
        self.assertEqual(frame["reason"], "slow")
        self.assertEqual(
            frame["session_fault_reason"],
            "main_loop: right_arm impedance fault",
        )
        self.assertEqual(
            frame["backend"]["left_command"]["active_tool_profile_path"],
            "tests/debug/rm65_left_DMTAC_M_angular_debug.json",
        )
        self.assertEqual(
            frame["backend"]["left_command"]["active_impedance_config"]["axes"],
            "rxyz",
        )
        self.assertEqual(
            frame["backend"]["left_command"]["active_impedance_config"][
                "angular_axis_signs"
            ],
            [-1.0, -1.0, -1.0],
        )
        self.assertIn("send_lock_wait_ms", frame["backend"]["left_command"]["last_send_command_timing"])
        self.assertIn(
            "rm65_controller_timing",
            frame["backend"]["left_command"]["last_impedance_record"],
        )
        self.assertIn(
            "active_wrench",
            frame["backend"]["left_command"]["last_impedance_record"],
        )
        self.assertIn(
            "raw_deadbanded_wrench",
            frame["backend"]["left_command"]["last_impedance_record"],
        )
        self.assertIn(
            "tare_corrected_wrench",
            frame["backend"]["left_command"]["last_impedance_record"],
        )
        self.assertEqual(
            frame["backend"]["left_command"]["recent_impedance_records"][0]["force_source"],
            "work",
        )
        self.assertIn(
            "filtered_wrench",
            frame["backend"]["left_command"]["recent_impedance_records"][0],
        )
        self.assertIn(
            "force_readings",
            frame["backend"]["left_command"]["recent_impedance_records"][0],
        )
        self.assertIn(
            "return_applied_axes",
            frame["backend"]["left_command"]["recent_impedance_records"][0],
        )
        self.assertIn(
            "return_paused_axes",
            frame["backend"]["left_command"]["recent_impedance_records"][0],
        )
        self.assertIn(
            "admittance_wrench",
            frame["backend"]["left_command"]["last_impedance_record"],
        )
        self.assertIn(
            "return_activity_wrench",
            frame["backend"]["left_command"]["last_impedance_record"],
        )
        self.assertIn(
            "target_offset",
            frame["backend"]["left_command"]["last_impedance_record"],
        )
        self.assertIn(
            "target_offset_unclamped",
            frame["backend"]["left_command"]["last_impedance_record"],
        )
        self.assertIn(
            "target_offset_after_return",
            frame["backend"]["left_command"]["last_impedance_record"],
        )
        self.assertIn(
            "offset_step_after_velocity_limit",
            frame["backend"]["left_command"]["last_impedance_record"],
        )
        self.assertIn(
            "angular_return_paused",
            frame["backend"]["left_command"]["last_impedance_record"],
        )
        self.assertIn(
            "offset",
            frame["backend"]["left_command"]["last_impedance_record"],
        )
        self.assertIn(
            "applied_offset",
            frame["backend"]["left_command"]["last_impedance_record"],
        )
        self.assertEqual(
            frame["backend"]["left_command"]["last_impedance_record"][
                "measured_to_command_linear_m"
            ],
            [0.01, 0.0, 0.0],
        )
        self.assertEqual(
            frame["backend"]["left_command"]["last_impedance_record"][
                "tracking_error_closing_linear_m_s"
            ],
            0.1,
        )
        self.assertEqual(
            frame["backend"]["left_command"]["last_impedance_record"]["ik_metadata"]["status"],
            "pass",
        )
        self.assertEqual(
            frame["backend"]["left_command"]["last_impedance_record"]["ik_metadata"][
                "direct_joint_hint_reason"
            ],
            "offset_threshold",
        )
        self.assertEqual(frame["channels"]["left_wrist_pose"]["metadata"]["seq"], 42)
        self.assertNotIn('"matrix"', encoded)
        self.assertNotIn("current_tcp_pose", encoded)

    def test_gui_ik_debug_toggle_creates_log_before_teleop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ui = _FakeUi(["d"])
            control = real_teleop.RealTeleopControlInput(
                terminal=_FakeTerminal(),
                ui=ui,
                args=_args(),
                components=_components(),
            )
            recorder = real_teleop.IkDebugLogRecorder(root_dir=tmp_dir)

            control.poll()
            real_teleop._handle_ik_debug_log_toggle_request(
                args=_args(),
                components=_components(),
                clutch_input=control,
                ik_debug_log_recorder=recorder,
                control_ui=ui,
                ik_debug_profile_base_enabled=False,
                now=1.25,
            )

            self.assertTrue(recorder.active)
            self.assertTrue(ui.published[-1]["debug_ik_logging"])
            path = Path(ui.published[-1]["ik_debug_log_path"])
            self.assertEqual(path.parent, Path(tmp_dir))
            self.assertTrue(path.exists())
            self.assertIn("_250", path.name)
            self.assertTrue(any("已开始 IK 记录" in message for _, message in ui.messages))
            recorder.stop(reason="unit_test", now=2.0)

    def test_running_gui_exposes_episode_and_visualization_controls(self) -> None:
        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=_components(),
        )

        control.publish_teleop_running()

        running_publish = next(
            published
            for published in reversed(ui.published)
            if published.get("stage") == "RM65 真机遥操运行中"
        )
        active_controls = set(running_publish["active_controls"])
        self.assertIn("episode_start", active_controls)
        self.assertIn("episode_stop", active_controls)
        self.assertIn("toggle_camera_vis", active_controls)
        self.assertIn("toggle_tactile_vis", active_controls)
        self.assertIn("d", active_controls)
        self.assertNotIn("g", active_controls)
        self.assertNotIn("init_arm_toggle", active_controls)
        self.assertNotIn("movement_multiple_down", active_controls)
        self.assertNotIn("movement_multiple_up", active_controls)
        self.assertNotIn("movement_multiple_reset", active_controls)
        self.assertNotIn("movement_multiple_set", active_controls)
        self.assertEqual(running_publish["movement_multiple"], 1.0)
        self.assertNotIn("gripper_toggle", active_controls)
        self.assertNotIn("gripper_open", active_controls)
        self.assertNotIn("gripper_half", active_controls)
        self.assertNotIn("gripper_close", active_controls)
        self.assertNotIn("record_pose_snapshot", active_controls)
        self.assertFalse(running_publish["pose_snapshot_available"])

    def test_fix_arm_running_gui_exposes_pose_snapshot_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ui = _FakeUi()
            components = _components()
            components.fix_arm_controller = SimpleNamespace(
                status_payload=lambda: {
                    "fix_arm_available": True,
                    "fix_arm_selected_side": "left",
                    "fix_arm_fixed_side": "",
                    "fix_arm_active": False,
                    "fix_arm_recalibrating": False,
                    "fix_arm_status": "",
                    "fix_arm_sample_count": None,
                    "fix_arm_min_samples": None,
                }
            )
            control = real_teleop.RealTeleopControlInput(
                terminal=_FakeTerminal(),
                ui=ui,
                args=_args(),
                components=components,
                pose_snapshot_recorder=real_teleop.FixArmPoseSnapshotRecorder(tmp_dir),
            )

            control.publish_teleop_running()

            running_publish = next(
                published
                for published in reversed(ui.published)
                if published.get("stage") == "RM65 真机遥操运行中"
            )
            active_controls = set(running_publish["active_controls"])
            self.assertIn("c", active_controls)
            self.assertIn("fix_arm_toggle", active_controls)
            self.assertIn("record_pose_snapshot", active_controls)
            self.assertTrue(running_publish["pose_snapshot_available"])
            self.assertEqual(running_publish["pose_snapshot_dir"], tmp_dir)

    def test_init_arm_running_gui_exposes_g_without_stealing_d(self) -> None:
        ui = _FakeUi()
        components = _components()
        components.init_arm_controller = SimpleNamespace(
            status_payload=lambda: {
                "init_arm_available": True,
                "init_arm_selected_side": "right",
                "init_arm_active_side": "",
                "init_arm_active": False,
                "init_arm_recalibrating": False,
                "init_arm_status": "ready",
                "init_arm_sample_count": None,
                "init_arm_min_samples": None,
                "init_arm_profile_path": "profile.json",
            }
        )
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=components,
        )

        control.publish_teleop_running()

        running_publish = next(
            published
            for published in reversed(ui.published)
            if published.get("stage") == "RM65 真机遥操运行中"
        )
        active_controls = set(running_publish["active_controls"])
        self.assertIn("d", active_controls)
        self.assertIn("g", active_controls)
        self.assertIn("init_arm_toggle", active_controls)
        self.assertIn("init_arm_select", active_controls)
        self.assertNotIn("c", active_controls)
        self.assertFalse(running_publish["fix_arm_available"])
        self.assertTrue(running_publish["init_arm_available"])
        self.assertEqual(running_publish["init_arm_selected_side"], "right")

    def test_pose_snapshot_request_is_blocked_while_episode_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ui = _FakeUi(["record_pose_snapshot"])
            components = _components()
            components.fix_arm_controller = SimpleNamespace(
                status_payload=lambda: {
                    "fix_arm_available": True,
                    "fix_arm_selected_side": "left",
                    "fix_arm_fixed_side": "",
                    "fix_arm_active": False,
                    "fix_arm_recalibrating": False,
                    "fix_arm_status": "",
                    "fix_arm_sample_count": None,
                    "fix_arm_min_samples": None,
                }
            )
            components.init_arm_controller = SimpleNamespace(
                status_payload=lambda: {
                    "init_arm_available": True,
                    "init_arm_selected_side": "right",
                    "init_arm_active_side": "",
                    "init_arm_active": False,
                    "init_arm_recalibrating": False,
                    "init_arm_status": "ready",
                    "init_arm_sample_count": None,
                    "init_arm_min_samples": None,
                    "init_arm_profile_path": "profile.json",
                }
            )
            control = real_teleop.RealTeleopControlInput(
                terminal=_FakeTerminal(),
                ui=ui,
                args=_args(),
                components=components,
                pose_snapshot_recorder=real_teleop.FixArmPoseSnapshotRecorder(tmp_dir),
            )
            control.publish_teleop_running()
            control.set_episode_recording_state(active=True)

            control.poll()

            self.assertFalse(control.consume_pose_snapshot_request())
            self.assertIn("互斥", ui.published[-1]["pose_snapshot_status"])

    def test_episode_countdown_controls_exclude_pose_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ui = _FakeUi()
            components = _components()
            components.fix_arm_controller = SimpleNamespace(
                status_payload=lambda: {
                    "fix_arm_available": True,
                    "fix_arm_selected_side": "left",
                    "fix_arm_fixed_side": "",
                    "fix_arm_active": False,
                    "fix_arm_recalibrating": False,
                    "fix_arm_status": "",
                    "fix_arm_sample_count": None,
                    "fix_arm_min_samples": None,
                }
            )
            components.init_arm_controller = SimpleNamespace(
                status_payload=lambda: {
                    "init_arm_available": True,
                    "init_arm_selected_side": "right",
                    "init_arm_active_side": "",
                    "init_arm_active": False,
                    "init_arm_recalibrating": False,
                    "init_arm_status": "ready",
                    "init_arm_sample_count": None,
                    "init_arm_min_samples": None,
                    "init_arm_profile_path": "profile.json",
                }
            )
            control = real_teleop.RealTeleopControlInput(
                terminal=_FakeTerminal(),
                ui=ui,
                args=_args(),
                components=components,
                pose_snapshot_recorder=real_teleop.FixArmPoseSnapshotRecorder(tmp_dir),
            )

            control.publish_teleop_running()
            control.set_episode_start_countdown_state(remaining_sec=2.0)

            active_controls = set(ui.published[-1]["active_controls"])
            self.assertIn("episode_stop", active_controls)
            self.assertNotIn("episode_start", active_controls)
            self.assertNotIn("record_pose_snapshot", active_controls)
            self.assertNotIn("fix_arm_toggle", active_controls)
            self.assertNotIn("init_arm_toggle", active_controls)

    def test_fix_arm_c_resets_fixed_side_safety_baseline(self) -> None:
        class FakeSafetyMonitor:
            def __init__(self) -> None:
                self.reset_channels: list[tuple[str, ...]] = []

            def request_pose_baseline_reset_many(self, channels) -> None:
                self.reset_channels.append(tuple(channels))

        class FakeFixArmController:
            recapturing = False
            fixed = False

            def freeze_selected(self, *, now: float):
                return SimpleNamespace(
                    ok=True,
                    side="left",
                    mode="fixed",
                    message="left fixed",
                    missing_effectors=(),
                    missing_channels=(),
                )

            def status_payload(self) -> dict[str, object]:
                return {
                    "fix_arm_available": True,
                    "fix_arm_selected_side": "left",
                    "fix_arm_fixed_side": "left",
                    "fix_arm_active": True,
                    "fix_arm_recalibrating": False,
                    "fix_arm_status": "left fixed",
                    "fix_arm_sample_count": None,
                    "fix_arm_min_samples": None,
                }

        safety_monitor = FakeSafetyMonitor()
        dual_arm_retargeter = SimpleNamespace(reset_count=0)

        def reset_target_pose_smoothing() -> None:
            dual_arm_retargeter.reset_count += 1

        dual_arm_retargeter.reset_target_pose_smoothing = reset_target_pose_smoothing
        ui = _FakeUi()
        components = _components()
        components.fix_arm_controller = FakeFixArmController()
        components.session.safety_monitor = safety_monitor
        components.dual_arm_retargeter = dual_arm_retargeter
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=components,
        )
        control.publish_teleop_running()

        handled = control._handle_fix_arm_c_event(now=10.0)

        self.assertTrue(handled)
        self.assertEqual(safety_monitor.reset_channels, [("left_wrist_pose",)])
        self.assertEqual(dual_arm_retargeter.reset_count, 1)
        self.assertFalse(control.consume_calibration_request())

    def test_fix_arm_c_failure_is_consumed_without_runtime_home_request(self) -> None:
        class FakeFixArmController:
            recapturing = False
            fixed = False
            selected_side = "left"
            fixed_side = ""

            def freeze_selected(self, *, now: float):
                raise RuntimeError("freeze boom")

            def status_payload(self) -> dict[str, object]:
                return {
                    "fix_arm_available": True,
                    "fix_arm_selected_side": "left",
                    "fix_arm_fixed_side": "",
                    "fix_arm_active": False,
                    "fix_arm_recalibrating": False,
                    "fix_arm_status": "ready",
                    "fix_arm_sample_count": None,
                    "fix_arm_min_samples": None,
                }

        ui = _FakeUi(["c"])
        components = _components()
        components.fix_arm_controller = FakeFixArmController()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=components,
        )
        control.publish_teleop_running()

        control.poll()

        self.assertFalse(control.consume_calibration_request())
        self.assertTrue(
            any(
                title == "FixArm" and "freeze boom" in message
                for title, message in ui.messages
            )
        )

    def test_stale_c_event_without_running_fix_arm_does_not_request_runtime_home(self) -> None:
        ui = _FakeUi(["c"])
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=_components(),
        )

        control.poll()

        self.assertFalse(control.consume_calibration_request())
        self.assertTrue(
            any("C 仅用于运行中的 fix-arm" in message for _title, message in ui.messages)
        )

    def test_hold_cancel_helper_stops_fix_and_init_modes(self) -> None:
        class FakeController:
            def __init__(self, *, side: str, active_attr: str) -> None:
                self.side = side
                setattr(self, active_attr, True)
                self.cancel_calls: list[dict[str, object]] = []

            def cancel_active(self, **kwargs):
                self.cancel_calls.append(dict(kwargs))
                self.fixed = False
                self.active = False
                return SimpleNamespace(ok=True, side=self.side)

        class FakeRetargeter:
            def __init__(self) -> None:
                self.reset_count = 0

            def reset_target_pose_smoothing(self) -> None:
                self.reset_count += 1

        fix_controller = FakeController(side="left", active_attr="fixed")
        init_controller = FakeController(side="right", active_attr="active")
        retargeter = FakeRetargeter()
        components = SimpleNamespace(
            fix_arm_controller=fix_controller,
            init_arm_controller=init_controller,
            dual_arm_retargeter=retargeter,
        )

        messages = real_teleop._cancel_special_arm_modes_for_hold(
            components,
            now=1.0,
            reason="hold_requested",
        )

        self.assertEqual(messages, ["左臂 fix-arm 已停止", "右臂 init-arm 已停止"])
        self.assertFalse(fix_controller.fixed)
        self.assertFalse(init_controller.active)
        self.assertTrue(fix_controller.cancel_calls[0]["clear_snapshots"])
        self.assertEqual(init_controller.cancel_calls[0]["reason"], "hold_requested")
        self.assertEqual(retargeter.reset_count, 1)

    def test_hold_gui_allows_stopping_active_ik_debug_log(self) -> None:
        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=_components(),
        )
        control.publish_teleop_running()
        control.set_ik_debug_logging_state(active=True, path="/tmp/ik_debug.jsonl")

        control.request_hold()

        hold_publish = next(
            published
            for published in reversed(ui.published)
            if published.get("stage") == "Hold / 停臂保持"
        )
        self.assertIn("d", hold_publish["active_controls"])
        self.assertTrue(hold_publish["debug_ik_logging"])

    def test_ready_gui_exposes_movement_multiple_controls(self) -> None:
        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=_components(),
        )

        control.start()

        ready_publish = next(
            published
            for published in reversed(ui.published)
            if published.get("stage") == "RM65 真机遥操准备"
        )
        active_controls = set(ready_publish["active_controls"])
        self.assertIn("d", active_controls)
        self.assertIn("movement_multiple_down", active_controls)
        self.assertIn("movement_multiple_up", active_controls)
        self.assertIn("movement_multiple_reset", active_controls)
        self.assertIn("movement_multiple_set", active_controls)
        self.assertEqual(ready_publish["movement_multiple"], 1.0)
        self.assertEqual(ready_publish["capture_stage"], "runtime_home")
        self.assertEqual(ready_publish["capture_phase"], "ready")
        self.assertFalse(ready_publish["teleop_running"])

    def test_gui_events_feed_episode_and_visualization_requests(self) -> None:
        ui = _FakeUi(
            [
                "episode_start",
                "episode_stop",
                "toggle_camera_vis",
                "toggle_tactile_vis",
            ]
        )
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=_components(),
        )

        control.poll()

        self.assertTrue(control.consume_episode_start_request())
        self.assertTrue(control.consume_episode_stop_request())
        self.assertTrue(control.consume_camera_visualization_toggle_request())
        self.assertTrue(control.consume_tactile_visualization_toggle_request())

    def test_gui_movement_multiple_event_updates_all_arm_retargeters(self) -> None:
        ui = _FakeUi(["movement_multiple_set:2.5"])
        components = _components()
        components.dual_arm_retargeter = SimpleNamespace(
            config=SimpleNamespace(translation_scale=1.0),
            left_pose_retargeter=components.left_retargeter,
            right_pose_retargeter=components.right_retargeter,
        )
        args = _args()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=args,
            components=components,
        )

        control.poll()

        self.assertEqual(args.movement_multiple, 2.5)
        self.assertEqual(args.translation_scale, 2.5)
        self.assertEqual(components.left_retargeter.config.translation_scale, 2.5)
        self.assertEqual(components.right_retargeter.config.translation_scale, 2.5)
        self.assertEqual(components.dual_arm_retargeter.config.translation_scale, 2.5)
        movement_publish = next(
            item for item in reversed(ui.published) if "movement_multiple" in item
        )
        self.assertEqual(movement_publish["movement_multiple"], 2.5)

    def test_gui_movement_multiple_button_steps_and_resets(self) -> None:
        ui = _FakeUi(["movement_multiple_up", "movement_multiple_down", "movement_multiple_reset"])
        args = _args(movement_multiple=1.2, translation_scale=1.2)
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=args,
            components=_components(),
        )

        control.poll()

        self.assertAlmostEqual(args.movement_multiple, 1.0)
        self.assertAlmostEqual(args.translation_scale, 1.0)

    def test_gui_movement_multiple_change_is_rejected_while_running(self) -> None:
        ui = _FakeUi(["movement_multiple_set:2.5"])
        args = _args(movement_multiple=1.0, translation_scale=1.0)
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=args,
            components=_components(),
        )
        control.publish_teleop_running()

        control.poll()

        self.assertEqual(args.movement_multiple, 1.0)
        self.assertTrue(any("运行中不直接修改" in message for _, message in ui.messages))

    def test_web_ui_accepts_movement_multiple_set_payload_when_control_active(self) -> None:
        ui = calibration_web_ui.CalibrationWebUi(open_browser=False)
        ui.publish(active_controls=["movement_multiple_set"])

        self.assertTrue(ui.push_event("movement_multiple_set:2.5"))
        self.assertEqual(ui.consume_events(), ["movement_multiple_set:2.5"])

    def test_web_ui_accepts_gripper_force_set_payload_when_control_active(self) -> None:
        ui = calibration_web_ui.CalibrationWebUi(open_browser=False)
        ui.publish(active_controls=["gripper_force_set"])

        event = 'gripper_force_set:{"left":0.31,"right":0.64}'
        self.assertTrue(ui.push_event(event))
        self.assertEqual(ui.consume_events(), [event])

    def test_gui_gripper_force_set_updates_active_sides_and_rejects_collection(self) -> None:
        components = _components()
        components.active_sides = ("left", "right")
        components.left_hand_backend = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="left",
                connect_hardware=False,
                connect_tactile=False,
            )
        )
        components.right_hand_backend = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="right",
                connect_hardware=False,
                connect_tactile=False,
            )
        )
        ui = _FakeUi(['gripper_force_set:{"left":0.31,"right":0.64}'])
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=components,
        )

        control.poll()

        self.assertAlmostEqual(components.left_hand_backend.config.torque, 0.31)
        self.assertAlmostEqual(components.right_hand_backend.config.torque, 0.64)
        self.assertEqual(
            control._manual_gripper_force_by_side,
            {"left": 0.31, "right": 0.64},
        )

        control._episode_countdown_ui_active = True
        ui.events.append('gripper_force_set:{"left":0.10,"right":0.20}')
        control.poll()

        self.assertAlmostEqual(components.left_hand_backend.config.torque, 0.31)
        self.assertAlmostEqual(components.right_hand_backend.config.torque, 0.64)
        self.assertTrue(any("数采录制、暂停、倒计时或保存中" in message for _, message in ui.messages))

    def test_manual_gripper_force_is_restored_to_task_policy_before_collection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            gripper_profile = Path(tmp_dir) / "gripper_profile.json"
            gripper_profile.write_text(
                json.dumps(
                    {
                        "gripper": {"force": 0.20},
                        "tasks": {
                            "firm_grasp": {
                                "gripper": {
                                    "left": {"force": 0.31},
                                    "right": {"force": 0.47},
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            components = _components()
            components.active_sides = ("left", "right")
            components.left_hand_backend = DMTacGripperBackend(
                DMTacGripperBackendConfig(
                    side="left",
                    connect_hardware=False,
                    connect_tactile=False,
                )
            )
            components.right_hand_backend = DMTacGripperBackend(
                DMTacGripperBackendConfig(
                    side="right",
                    connect_hardware=False,
                    connect_tactile=False,
                )
            )
            control = real_teleop.RealTeleopControlInput(
                terminal=_FakeTerminal(),
                ui=_FakeUi(),
                args=_args(
                    gripper_task="firm_grasp",
                    gripper_task_profile=str(gripper_profile),
                ),
                components=components,
            )

            control._set_gripper_force_from_gui('{"left":0.80,"right":0.90}')
            control._restore_gripper_task_policy_for_collection()

        self.assertAlmostEqual(components.left_hand_backend.config.torque, 0.31)
        self.assertAlmostEqual(components.right_hand_backend.config.torque, 0.47)
        self.assertEqual(control._manual_gripper_force_by_side, {})

    def test_dmtac_runtime_torque_update_applies_to_connected_device(self) -> None:
        class _FakeDevice:
            gripper = object()

            def __init__(self) -> None:
                self.torque_values: list[float] = []

            def set_torque(self, value: float) -> None:
                self.torque_values.append(float(value))

        backend = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="right",
                connect_hardware=True,
                connect_tactile=False,
            )
        )
        device = _FakeDevice()
        backend.device = device

        self.assertAlmostEqual(backend.set_torque(0.52), 0.52)

        self.assertEqual(device.torque_values, [0.52])
        self.assertAlmostEqual(backend.config.torque, 0.52)

        class _FailingDevice:
            gripper = object()

            @staticmethod
            def set_torque(_value: float) -> None:
                raise RuntimeError("serial write failed")

        backend.device = _FailingDevice()
        with self.assertRaisesRegex(RuntimeError, "serial write failed"):
            backend.set_torque(0.80)
        self.assertAlmostEqual(backend.config.torque, 0.52)

        backend.device = device
        backend.config.send_commands = False
        backend.set_torque(0.61)
        self.assertEqual(device.torque_values, [0.52])
        self.assertAlmostEqual(backend.config.torque, 0.61)

    def test_web_ui_accepts_episode_start_payload_when_control_active(self) -> None:
        ui = calibration_web_ui.CalibrationWebUi(open_browser=False)
        ui.publish(active_controls=["episode_start"])

        event = 'episode_start:{"task_key":"soft_grasp"}'
        self.assertTrue(ui.push_event(event))
        self.assertEqual(ui.consume_events(), [event])

    def test_web_ui_accepts_task_select_payload_when_control_active(self) -> None:
        ui = calibration_web_ui.CalibrationWebUi(open_browser=False)
        ui.publish(active_controls=["task_select"])

        event = 'task_select:{"task_key":"soft_grasp"}'
        self.assertTrue(ui.push_event(event))
        self.assertEqual(ui.consume_events(), [event])

    def test_http_task_select_payload_is_encoded_for_runtime(self) -> None:
        payload = {
            "event": "task_select",
            "dataset_base_root": "/tmp/datasets",
            "task_type": "general",
            "task_key": "soft_grasp",
            "task_english_name": "Soft Grasp",
            "task_chinese_name": "轻柔抓取",
            "round_trip_phase": "in",
        }

        self.assertEqual(
            calibration_web_ui._encode_http_event_payload(payload),
            'task_select:{"dataset_base_root":"/tmp/datasets","task_type":"general","task_key":"soft_grasp","task_english_name":"Soft Grasp","task_chinese_name":"轻柔抓取","round_trip_phase":"in"}',
        )
        self.assertEqual(
            calibration_web_ui._encode_http_event_payload({"event": "fix_arm_toggle"}),
            "fix_arm_toggle",
        )

    def test_web_ui_accepts_fix_arm_toggle_when_control_active(self) -> None:
        ui = calibration_web_ui.CalibrationWebUi(open_browser=False)
        ui.publish(active_controls=["fix_arm_toggle"])

        self.assertTrue(ui.push_event("fix_arm_toggle"))
        self.assertEqual(ui.consume_events(), ["fix_arm_toggle"])
        self.assertTrue(
            any(
                item.get("tag") == "WebGUIEvent"
                and "accepted: fix_arm_toggle" in item.get("message", "")
                for item in ui.snapshot()["messages"]
            )
        )

    def test_web_ui_accepts_init_arm_toggle_and_side_when_control_active(self) -> None:
        ui = calibration_web_ui.CalibrationWebUi(open_browser=False)
        ui.publish(active_controls=["init_arm_toggle", "init_arm_select"])

        self.assertTrue(ui.push_event("init_arm_toggle"))
        self.assertTrue(ui.push_event("init_arm_select:right"))
        self.assertEqual(ui.consume_events(), ["init_arm_toggle", "init_arm_select:right"])
        self.assertTrue(
            any(
                item.get("tag") == "WebGUIEvent"
                and "accepted: init_arm_toggle" in item.get("message", "")
                for item in ui.snapshot()["messages"]
            )
        )

    def test_gui_episode_start_payload_updates_task_path_and_gripper_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_root = Path(tmp_dir) / "datasets"
            task_config = Path(tmp_dir) / "tasks.json"
            gripper_profile = Path(tmp_dir) / "gripper_profile.json"
            task_config.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "key": "stack_blocks",
                                "english_name": "Stack Blocks",
                                "chinese_name": "叠方块",
                                "description": "stack blocks demo",
                                "gripper_task": "firm_grasp",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            gripper_profile.write_text(
                json.dumps(
                    {
                        "gripper": {"speed": 1.0, "force": 0.2, "max_action": 0.95},
                        "tasks": {
                            "firm_grasp": {
                                "gripper": {
                                    "force": 0.4,
                                    "left": {"force": 0.31},
                                    "right": {"force": 0.47},
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            ui = _FakeUi(
                [
                    "episode_start:"
                    + json.dumps(
                        {
                            "dataset_base_root": str(base_root),
                            "task_key": "stack_blocks",
                            "task_english_name": "",
                            "task_chinese_name": "",
                        },
                        ensure_ascii=False,
                    )
                ]
            )
            components = _components()
            components.active_sides = ("left", "right")
            components.left_hand_backend = DMTacGripperBackend(
                DMTacGripperBackendConfig(
                    side="left",
                    connect_hardware=False,
                    connect_tactile=False,
                )
            )
            components.right_hand_backend = DMTacGripperBackend(
                DMTacGripperBackendConfig(
                    side="right",
                    connect_hardware=False,
                    connect_tactile=False,
                )
            )
            args = _args(
                dataset_root=str(base_root),
                task_config=str(task_config),
                gripper_task_profile=str(gripper_profile),
            )
            real_teleop._apply_formal_dataset_layout(args)
            control = real_teleop.RealTeleopControlInput(
                terminal=_FakeTerminal(),
                ui=ui,
                args=args,
                components=components,
            )

            with patch.object(real_teleop, "datetime") as fake_datetime:
                fake_datetime.now.return_value.strftime.return_value = "2026-07-02"
                control.poll()

        self.assertTrue(control.consume_episode_start_request())
        self.assertEqual(args.collection_task_id, "stack_blocks")
        self.assertEqual(args.collection_task_name, "Stack Blocks")
        self.assertEqual(args.collection_task_name_zh, "叠方块")
        self.assertEqual(args.collection_task_description, "stack blocks demo")
        self.assertEqual(args.gripper_task, "firm_grasp")
        self.assertEqual(Path(args.dataset_base_root), base_root)
        self.assertEqual(Path(args.dataset_root), base_root / "2026-07-02_Stack_Blocks")
        self.assertEqual(args.dataset_name, "2026-07-02_Stack_Blocks")
        self.assertAlmostEqual(components.left_hand_backend.config.speed, 1.0)
        self.assertAlmostEqual(components.right_hand_backend.config.speed, 1.0)
        self.assertAlmostEqual(components.left_hand_backend.config.torque, 0.31)
        self.assertAlmostEqual(components.right_hand_backend.config.torque, 0.47)
        task_publish = next(
            item for item in reversed(ui.published) if item.get("collection_task_id") == "stack_blocks"
        )
        self.assertEqual(task_publish["collection_task_name_zh"], "叠方块")

    def test_gui_task_select_payload_updates_policy_without_starting_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_root = Path(tmp_dir) / "datasets"
            task_config = Path(tmp_dir) / "tasks.json"
            task_config.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "key": "soft_pick",
                                "english_name": "Soft Pick",
                                "chinese_name": "轻柔抓取",
                                "description": "fragile object task",
                                "gripper_task": "soft_grasp",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            ui = _FakeUi(
                [
                    "task_select:"
                    + json.dumps(
                        {
                            "dataset_base_root": str(base_root),
                            "task_key": "soft_pick",
                            "task_english_name": "",
                            "task_chinese_name": "",
                        },
                        ensure_ascii=False,
                    )
                ]
            )
            args = _args(dataset_root=str(base_root), task_config=str(task_config))
            real_teleop._apply_formal_dataset_layout(args)
            control = real_teleop.RealTeleopControlInput(
                terminal=_FakeTerminal(),
                ui=ui,
                args=args,
                components=_components(),
            )

            with patch.object(real_teleop, "datetime") as fake_datetime:
                fake_datetime.now.return_value.strftime.return_value = "2026-07-02"
                control.poll()

        self.assertFalse(control.consume_episode_start_request())
        self.assertEqual(args.collection_task_id, "soft_pick")
        self.assertEqual(args.collection_task_name, "Soft Pick")
        self.assertEqual(args.gripper_task, "soft_grasp")
        self.assertEqual(Path(args.dataset_root), base_root / "2026-07-02_Soft_Pick")

    def test_gui_round_trip_task_payload_starts_in_and_can_toggle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_root = Path(tmp_dir) / "datasets"
            task_config = Path(tmp_dir) / "tasks.json"
            task_config.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "key": "drawer_push_pull",
                                "task_type": "round_trip",
                                "english_name": "Drawer Push Pull",
                                "chinese_name": "抽屉推拉",
                                "description": "drawer push pull demo",
                                "gripper_task": "firm_grasp",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            ui = _FakeUi(
                [
                    "episode_start:"
                    + json.dumps(
                        {
                            "dataset_base_root": str(base_root),
                            "task_type": "round_trip",
                            "task_key": "drawer_push_pull",
                            "task_english_name": "",
                            "task_chinese_name": "",
                            "round_trip_phase": "in",
                        },
                        ensure_ascii=False,
                    )
                ]
            )
            args = _args(
                dataset_root=str(base_root),
                task_config=str(task_config),
                gripper_task="",
            )
            real_teleop._apply_formal_dataset_layout(args)
            control = real_teleop.RealTeleopControlInput(
                terminal=_FakeTerminal(),
                ui=ui,
                args=args,
                components=_components(),
            )

            with patch.object(real_teleop, "datetime") as fake_datetime:
                fake_datetime.now.return_value.strftime.return_value = "2026-07-04"
                control.poll()
                self.assertTrue(control.consume_episode_start_request())
                self.assertEqual(args.collection_round_trip_phase, "in")
                self.assertEqual(args.collection_task_id, "drawer_push_pull_in")
                self.assertEqual(args.collection_task_name, "Drawer Push Pull_in")
                self.assertEqual(Path(args.dataset_root), base_root / "2026-07-04_Drawer_Push_Pull_in")

                ui.events = ["round_trip_toggle"]
                control.poll()

        self.assertEqual(args.collection_task_type, "round_trip")
        self.assertEqual(args.collection_round_trip_phase, "out")
        self.assertEqual(args.collection_task_id, "drawer_push_pull_out")
        self.assertEqual(args.collection_task_name, "Drawer Push Pull_out")
        self.assertEqual(args.collection_round_trip_base_id, "drawer_push_pull")
        self.assertEqual(args.gripper_task, "firm_grasp")
        self.assertEqual(Path(args.dataset_root), base_root / "2026-07-04_Drawer_Push_Pull_out")
        self.assertTrue(any("来回任务已手动切换到 _out" in message for _, message in ui.messages))

    def test_gui_vive_calibration_button_calibrates_coordinate_profile(self) -> None:
        ui = _FakeUi(["calibrate_vive_coordinate_profile"])
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=_components(),
        )

        with patch.object(
            real_teleop,
            "_calibrate_gui_vive_coordinate_profile",
            return_value=Path("/tmp/vive_profile.json"),
        ) as quick_calibrate:
            control.poll()
            requested = control.consume_calibration_request()

        self.assertFalse(requested)
        quick_calibrate.assert_called_once_with(control.args, control.components, ui)
        calibration_publish = next(item for item in ui.published if "control_hint" in item)
        self.assertIn("坐标系已保存", calibration_publish["control_hint"])
        self.assertFalse(calibration_publish["home_ready"])

    def test_gui_vive_calibration_hand_selection_defaults_to_right(self) -> None:
        ui = _FakeUi(["n"])
        components = _components()
        components.active_sides = ("left", "right")

        selected_side, channel_name = real_teleop._select_gui_vive_coordinate_sweep_hand(
            ui=ui,
            args=_args(virtual_desk_tracker_role="auto"),
            components=components,
            source=_FakeViveSource(),
        )

        self.assertEqual(selected_side, "right")
        self.assertEqual(channel_name, "right_wrist_pose")
        self.assertTrue(any("默认右手" in message for _, message in ui.messages))
        publish = next(item for item in reversed(ui.published) if item.get("stage") == "VIVE 标定手选择")
        self.assertEqual(publish["selected_side"], "right")

    def test_gui_vive_calibration_hand_selection_accepts_left(self) -> None:
        ui = _FakeUi(["1", "n"])
        components = _components()
        components.active_sides = ("left", "right")

        selected_side, channel_name = real_teleop._select_gui_vive_coordinate_sweep_hand(
            ui=ui,
            args=_args(virtual_desk_tracker_role="auto"),
            components=components,
            source=_FakeViveSource(),
        )

        self.assertEqual(selected_side, "left")
        self.assertEqual(channel_name, "left_wrist_pose")
        publish = next(item for item in reversed(ui.published) if item.get("stage") == "VIVE 标定手选择")
        self.assertEqual(publish["selected_side"], "left")

    def test_gui_vive_axis_capture_waits_for_next_before_starting(self) -> None:
        ui = _BatchedFakeUi([[], ["n"]])
        components = _components()
        components.active_sides = ("left", "right")

        poses = real_teleop._capture_single_vive_channel_from_gui(
            ui=ui,
            args=_args(capture_prep_sec=0.0, hz=1000.0),
            components=components,
            source=_FakeViveSource(),
            stage_name="VirtualDeskX",
            instructions="X",
            timed_flow_hint="hint",
            direction_hint="后 -> 前",
            channel_name="right_wrist_pose",
            min_samples=1,
            capture_duration_sec=0.0,
            validator=lambda captured: None,
        )

        self.assertEqual(len(poses), 1)
        ready_publish = next(item for item in ui.published if item.get("capture_phase") == "ready")
        self.assertIn("n", ready_publish["active_controls"])
        self.assertNotIn("c", ready_publish["active_controls"])
        self.assertIn("下一步", ready_publish["control_hint"])

    def test_gui_vive_wrist_offset_capture_waits_for_next_before_starting(self) -> None:
        ui = _BatchedFakeUi([[], ["n"]])
        components = _components()
        components.active_sides = ("left", "right")

        left_poses, right_poses = real_teleop._capture_dual_wrist_offsets_from_gui(
            ui=ui,
            args=_args(
                capture_extra_wait_sec=0.0,
                hold_capture_sec=0.0,
                hz=1000.0,
                min_hold_samples=1,
                wrist_capture_prep_sec=0.0,
            ),
            components=components,
            source=_FakeViveSource(),
        )

        self.assertEqual(len(left_poses), 1)
        self.assertEqual(len(right_poses), 1)
        ready_publish = next(
            item for item in ui.published
            if item.get("capture_stage") == "WristOffset" and item.get("capture_phase") == "ready"
        )
        self.assertIn("n", ready_publish["active_controls"])
        self.assertNotIn("c", ready_publish["active_controls"])
        self.assertIn("下一步", ready_publish["control_hint"])

    def test_gui_does_not_expose_skip_runtime_home_button(self) -> None:
        self.assertNotIn("skip_runtime_home", calibration_web_ui._HTML_PAGE)
        self.assertNotIn("跳过 Home", calibration_web_ui._HTML_PAGE)

    def test_gui_script_avoids_newer_js_features_that_leave_static_page_stuck(self) -> None:
        self.assertNotIn("replaceAll(", calibration_web_ui._HTML_PAGE)
        self.assertNotIn("??", calibration_web_ui._HTML_PAGE)

    def test_gui_has_legacy_state_poll_before_modern_script(self) -> None:
        html = calibration_web_ui._HTML_PAGE
        legacy_index = html.index("__calibrationWebUiLegacyPollActive")
        modern_index = html.index("window.__calibrationWebUiModernScriptLoaded = true;")
        self.assertLess(legacy_index, modern_index)
        legacy_script = html[html.rfind("<script>", 0, legacy_index):modern_index]
        self.assertIn("XMLHttpRequest", legacy_script)
        self.assertNotIn("async function", legacy_script)
        self.assertNotIn("=>", legacy_script)
        self.assertNotIn("const ", legacy_script)
        self.assertNotIn("let ", legacy_script)

    def test_gui_legacy_poll_updates_raw_vive_and_ik_log_button(self) -> None:
        html = calibration_web_ui._HTML_PAGE
        legacy_index = html.index("__calibrationWebUiLegacyPollActive")
        modern_index = html.index("window.__calibrationWebUiModernScriptLoaded = true;")
        legacy_script = html[html.rfind("<script>", 0, legacy_index):modern_index]

        self.assertIn("renderRawPosesLegacy(state)", legacy_script)
        self.assertIn("renderEpisodeStateLegacy(state)", legacy_script)
        self.assertIn("renderIkDebugLogButtonLegacy(state)", legacy_script)
        self.assertIn("leftRawPose", legacy_script)
        self.assertIn("rightRawPose", legacy_script)
        self.assertIn("IK 记录切换中...", legacy_script)

    def test_gui_serves_external_fallback_script_for_inline_script_blocking(self) -> None:
        self.assertIn('src="/calibration-web-ui-fallback.js"', calibration_web_ui._HTML_PAGE)
        fallback_script = calibration_web_ui._FALLBACK_SCRIPT
        self.assertIn("__calibrationWebUiExternalFallbackActive", fallback_script)
        self.assertIn("XMLHttpRequest", fallback_script)
        self.assertNotIn("async function", fallback_script)
        self.assertNotIn("=>", fallback_script)
        self.assertNotIn("const ", fallback_script)
        self.assertNotIn("let ", fallback_script)
        self.assertIn("renderEpisodeStateFallback(state)", fallback_script)

        def code_strings(code: object) -> list[str]:
            found: list[str] = []
            for const in getattr(code, "co_consts", ()):
                if isinstance(const, str):
                    found.append(const)
                elif hasattr(const, "co_consts"):
                    found.extend(code_strings(const))
            return found

        start_consts = "\n".join(code_strings(calibration_web_ui.CalibrationWebUi.start.__code__))
        self.assertIn("/calibration-web-ui-fallback.js", start_consts)

    def test_gui_state_poll_keeps_retrying_after_modern_fetch_failure(self) -> None:
        html = calibration_web_ui._HTML_PAGE
        legacy_index = html.index("__calibrationWebUiLegacyPollActive")
        modern_index = html.index("window.__calibrationWebUiModernScriptLoaded = true;")
        legacy_script = html[html.rfind("<script>", 0, legacy_index):modern_index]

        self.assertNotIn("clearInterval(legacyTimer)", legacy_script)
        self.assertIn("showRefreshWarning(error);", html)
        self.assertIn("window.__calibrationWebUiModernRefreshOk = false;", html)
        self.assertNotIn("showShutdown(\"无法继续连接标定服务", html)

    def test_gui_modern_ik_log_button_has_pending_feedback(self) -> None:
        html = calibration_web_ui._HTML_PAGE

        self.assertIn("ikDebugLogTogglePending", html)
        self.assertIn("updateIkDebugLogButton(state)", html)
        self.assertIn("IK 记录切换中...", html)
        self.assertIn('ikDebugLogButtonEl.disabled = !activeControls.has("d") || ikDebugLogTogglePending', html)

    def test_gui_initial_html_is_rendered_from_backend_snapshot(self) -> None:
        html = calibration_web_ui._render_initial_html(
            {
                "stage": "RM65 真机遥操准备",
                "selected_side": "right",
                "frame_viz_mode": "calibrated",
                "virtual_desk_post_yaw_deg": 180.0,
                "control_hint": "请选择采集 Home。",
                "capture_stage": "runtime_home",
                "capture_phase": "ready",
                "raw_tracker_poses": {
                    "left": {
                        "enabled": True,
                        "valid": True,
                        "tracker_name": "tracker_1",
                        "tracker_role": "left_wrist",
                        "seq": 7,
                        "age_sec": 0.12,
                        "status_label": "有效",
                        "status_level": "ok",
                        "xyz_m": [1.0, 2.0, 3.0],
                        "rpy_deg": [4.0, 5.0, 6.0],
                    }
                },
            }
        )

        self.assertNotIn("__INITIAL_", html)
        self.assertIn("RM65 真机遥操准备", html)
        self.assertIn("准备就绪", html)
        self.assertIn("请选择采集 Home。", html)
        self.assertIn("tracker_1", html)
        self.assertIn("网页脚本尚未接管实时刷新", html)

    def test_gui_raw_tracker_pose_has_stale_warning_styles(self) -> None:
        html = calibration_web_ui._HTML_PAGE

        self.assertIn(".raw-pose-card.stale", html)
        self.assertIn("leftRawCard", html)
        self.assertIn("setRawPoseCardClass", html)
        self.assertIn("status_label", html)

    def test_gui_state_json_payload_sanitizes_non_finite_values(self) -> None:
        ui = calibration_web_ui.CalibrationWebUi(open_browser=False)
        ui.publish(
            virtual_desk_post_yaw_deg=float("nan"),
            raw_tracker_poses={
                "left": {
                    "valid": True,
                    "xyz_m": np.array([1.0, np.nan, np.inf]),
                    "rpy_deg": [np.float32("-inf"), 2.0, 3.0],
                }
            },
            ik_timing={"solve_ms": np.float64("inf")},
            gripper_status={"right": {"q6_values": np.array([0.1, np.nan])}},
        )

        payload = calibration_web_ui._json_response_payload(ui.snapshot())
        text = payload.decode("utf-8")

        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        parsed = json.loads(text)
        self.assertIsNone(parsed["virtual_desk_post_yaw_deg"])
        self.assertEqual(parsed["raw_tracker_poses"]["left"]["xyz_m"], [1.0, None, None])
        self.assertEqual(parsed["raw_tracker_poses"]["left"]["rpy_deg"], [None, 2.0, 3.0])
        self.assertIsNone(parsed["ik_timing"]["solve_ms"])
        self.assertEqual(parsed["gripper_status"]["right"]["q6_values"], [0.1, None])

    def test_single_right_gui_tracker_swap_remaps_to_left_tracker(self) -> None:
        args = _args()
        components = _components()
        source = VivePoseSource(
            tracker_configs=real_teleop._build_vive_tracker_configs_for_args(
                args,
                components.active_sides,
            )
        )
        components.session.sources = [source]
        ui = _FakeUi(events=["w"])
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=args,
            components=components,
        )

        control.poll()

        self.assertTrue(args.swap_operator_sides)
        configs = list(source.tracker_configs.values())
        self.assertEqual(len(configs), 1)
        config = configs[0]
        self.assertEqual(config.channel_name, "right_wrist_pose")
        self.assertEqual(config.side, "right")
        self.assertEqual(config.name, "tracker_1")
        self.assertEqual(config.role, "left_wrist")
        self.assertIn("logical_right<-left_tracker", ui.messages[-1][1])

    def test_single_right_hold_tracker_swap_is_allowed_before_rehome(self) -> None:
        args = _args()
        components = _components()
        source = VivePoseSource(
            tracker_configs=real_teleop._build_vive_tracker_configs_for_args(
                args,
                components.active_sides,
            )
        )
        components.session.sources = [source]
        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=args,
            components=components,
        )
        control._teleop_started = True
        control._tracker_swap_locked = True

        control.request_hold()
        self.assertIn("w", ui.published[-1]["active_controls"])

        ui.events.append("w")
        control.poll()

        self.assertTrue(args.swap_operator_sides)
        configs = list(source.tracker_configs.values())
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].channel_name, "right_wrist_pose")
        self.assertEqual(configs[0].name, "tracker_1")
        self.assertEqual(configs[0].role, "left_wrist")
        status_publish = next(
            item for item in reversed(ui.published) if "teleop_running" in item
        )
        self.assertFalse(status_publish["teleop_running"])
        self.assertFalse(status_publish["home_ready"])
        self.assertIn("重新点击", ui.messages[-1][1])

    def test_gripper_runtime_status_signature_includes_q6_and_send_details(self) -> None:
        backend = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="right",
                connect_hardware=False,
                connect_tactile=False,
            )
        )
        components = _components()
        components.right_hand_backend = backend
        components.right_gripper_retargeter = real_teleop.ScalarToGripperRetargeter(
            real_teleop.ScalarToGripperRetargeterConfig(
                side="right",
                input_channel_name="right_hand_q6",
                effector_name="right_hand",
                source_dim=2,
                output_dim=0,
            )
        )
        channel = ObservationChannel(
            channel_name="right_hand_q6",
            capability="q6",
            side="right",
            value=np.array([0.0, 0.1, 0.62, 0.0, 0.0, 0.0]),
            frame_name=None,
            timestamp=9.5,
            source_name="q6_right",
            valid=True,
        )

        status = real_teleop._gripper_runtime_status_signature(
            components,
            channels={"right_hand_q6": channel},
            command_status={
                "right": {
                    "status": "sent",
                    "sent": True,
                    "source_value": 0.62,
                    "target": 0.62,
                }
            },
            now=10.0,
        )

        self.assertEqual(len(status), 1)
        self.assertIn("right=", status[0])
        self.assertIn("input=q6", status[0])
        self.assertIn("valid=ok", status[0])
        self.assertIn("dim=2", status[0])
        self.assertIn("src=0.620", status[0])
        self.assertIn("target=0.650", status[0])
        self.assertIn("send=sent/1", status[0])
        self.assertIn("send_target=0.620", status[0])

    def test_dmtac_tactile_visualizer_builder_uses_force_window(self) -> None:
        vis = real_teleop._build_tactile_visualizer(
            _args(visualize_tactile=True),
            ("right",),
        )

        self.assertIsInstance(vis, DMTacDistributedForceVisualizer)
        vis.close()

    def test_dmtac_tactile_visualization_requires_distributed_force_backend(self) -> None:
        backend = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="right",
                connect_hardware=False,
                connect_tactile=True,
            )
        )
        components = _components()
        components.right_hand_backend = backend

        self.assertFalse(real_teleop._tactile_visualization_available(_args(), components))

        backend = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="right",
                connect_hardware=False,
                connect_tactile=True,
                tactile_fields=("distributed_force",),
            )
        )
        components.right_hand_backend = backend

        self.assertTrue(real_teleop._tactile_visualization_available(_args(), components))

    def test_session_record_root_sets_timestamped_episode_root_without_session_recorder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(session_record_root=tmp_dir)
            components = SimpleNamespace(session=SimpleNamespace(recorder=None))

            with patch.object(real_teleop.time, "strftime", return_value="20260630_120000"):
                session_root = real_teleop._session_record_root_dir(args)
                episode_root = real_teleop._episode_root_dir(args, components)

            expected = Path(tmp_dir) / "20260630_120000_vive_rm65_real_teleop"
            self.assertEqual(session_root, expected)
            self.assertEqual(episode_root, expected / "trajectories")

    def test_episode_root_overrides_dataset_root_for_episode_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(
                session_record_root=str(Path(tmp_dir) / "session_base"),
                episode_root=str(Path(tmp_dir) / "episodes_exact"),
            )
            components = SimpleNamespace(session=SimpleNamespace(recorder=None))

            self.assertEqual(
                real_teleop._episode_root_dir(args, components),
                Path(tmp_dir) / "episodes_exact",
            )

    def test_debug_episode_recording_defaults_off_for_formal_dataset(self) -> None:
        args = _args(no_collection=False, episode_root="", debug_trajectory_recording=False)

        self.assertFalse(real_teleop._debug_episode_recording_enabled(args))

    def test_debug_episode_recording_defaults_on_without_formal_collection(self) -> None:
        args = _args(no_collection=True, episode_root="", debug_trajectory_recording=False)

        self.assertTrue(real_teleop._debug_episode_recording_enabled(args))

    def test_episode_root_explicitly_enables_debug_episode_recording(self) -> None:
        args = _args(no_collection=False, episode_root="/tmp/debug_episodes")

        self.assertTrue(real_teleop._debug_episode_recording_enabled(args))

    def test_episode_metadata_records_resolved_episode_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(session_record_root=tmp_dir, no_collection=True)
            components = _components()

            with patch.object(real_teleop.time, "strftime", return_value="20260630_120000"):
                metadata = real_teleop._build_episode_metadata(
                    args,
                    components,
                    start_reason="manual_key_start",
                )

            expected = Path(tmp_dir) / "20260630_120000_vive_rm65_real_teleop" / "trajectories"
            self.assertEqual(metadata["episode_root"], str(expected))
            self.assertFalse(metadata["session_record_enabled"])
            self.assertTrue(metadata["debug_episode_recording_enabled"])
            self.assertEqual(metadata["start_reason"], "manual_key_start")

    def test_formal_dataset_metadata_does_not_claim_unused_debug_episode_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(session_record_root=tmp_dir, no_collection=False)
            components = _components()

            metadata = real_teleop._build_episode_metadata(
                args,
                components,
                start_reason="manual_key_start",
            )

            self.assertIsNone(metadata["episode_root"])
            self.assertFalse(metadata["session_record_enabled"])
            self.assertFalse(metadata["debug_episode_recording_enabled"])

    def test_episode_metadata_records_debug_timing_configuration(self) -> None:
        args = _args(
            debug_timing=True,
            debug_timing_threshold_sec=0.04,
            debug_timing_report_interval_sec=0.5,
        )
        metadata = real_teleop._build_episode_metadata(
            args,
            _components(),
            start_reason="manual_key_start",
        )

        self.assertTrue(metadata["debug_timing"])
        self.assertTrue(metadata["debug_timing_legacy_all"])
        self.assertEqual(
            metadata["debug_timing_modules"],
            ["console", "log", "rm65", "ik", "watchdog"],
        )
        self.assertTrue(metadata["debug_timing_console"])
        self.assertTrue(metadata["debug_timing_light_log"])
        self.assertTrue(metadata["rm65_backend_debug_timing"])
        self.assertTrue(metadata["debug_timing_ik_profile"])
        self.assertTrue(metadata["debug_timing_watchdog"])
        self.assertEqual(metadata["debug_timing_threshold_sec"], 0.04)
        self.assertEqual(metadata["debug_timing_report_interval_sec"], 0.5)

    def test_episode_metadata_records_specific_debug_timing_modules(self) -> None:
        args = _args(
            debug_timing_log=True,
            debug_timing_threshold_sec=0.066,
            debug_timing_report_interval_sec=2.0,
        )
        metadata = real_teleop._build_episode_metadata(
            args,
            _components(),
            start_reason="manual_key_start",
        )

        self.assertTrue(metadata["debug_timing"])
        self.assertFalse(metadata["debug_timing_legacy_all"])
        self.assertEqual(metadata["debug_timing_modules"], ["log"])
        self.assertFalse(metadata["debug_timing_console"])
        self.assertTrue(metadata["debug_timing_light_log"])
        self.assertFalse(metadata["rm65_backend_debug_timing"])
        self.assertFalse(metadata["debug_timing_ik_profile"])
        self.assertFalse(metadata["debug_timing_watchdog"])
        self.assertEqual(metadata["debug_timing_threshold_sec"], 0.066)
        self.assertEqual(metadata["debug_timing_report_interval_sec"], 2.0)

    def test_debug_timing_ik_and_rm65_enable_timing_log_by_default(self) -> None:
        args = _args(
            debug_timing_ik=True,
            debug_timing_rm65=True,
        )
        metadata = real_teleop._debug_timing_metadata(args)

        self.assertTrue(metadata["debug_timing"])
        self.assertFalse(metadata["debug_timing_legacy_all"])
        self.assertEqual(metadata["debug_timing_modules"], ["log", "rm65", "ik"])
        self.assertTrue(metadata["debug_timing_light_log"])
        self.assertTrue(metadata["rm65_backend_debug_timing"])
        self.assertTrue(metadata["debug_timing_ik_profile"])

    def test_debug_timing_auto_log_can_be_disabled_for_narrow_modules(self) -> None:
        args = _args(
            debug_timing_ik=True,
            debug_timing_rm65=True,
            debug_timing_auto_log=False,
        )
        metadata = real_teleop._debug_timing_metadata(args)

        self.assertTrue(metadata["debug_timing"])
        self.assertEqual(metadata["debug_timing_modules"], ["rm65", "ik"])
        self.assertFalse(metadata["debug_timing_light_log"])
        self.assertTrue(metadata["rm65_backend_debug_timing"])
        self.assertTrue(metadata["debug_timing_ik_profile"])

    def test_formal_dataset_layout_uses_dataset_root_and_task_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(
                dataset_root=tmp_dir,
                collection_task_name="Pick Chips",
                collection_task_id="chips_v1",
                collection_task_description="pick chips demo",
                dataset_name="chips_dataset",
                gripper_task="",
            )

            with patch.object(real_teleop, "datetime") as fake_datetime:
                fake_datetime.now.return_value.strftime.return_value = "2026-06-30"
                real_teleop._apply_formal_dataset_layout(args)

            self.assertEqual(Path(args.dataset_base_root), Path(tmp_dir))
            self.assertEqual(Path(args.dataset_root), Path(tmp_dir) / "2026-06-30_Pick_Chips")
            self.assertEqual(args.collection_task_id, "chips_v1")
            self.assertEqual(args.dataset_name, "chips_dataset")

    def test_round_trip_dataset_layout_defaults_to_in_and_toggles_to_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(
                dataset_root=tmp_dir,
                collection_task_type="round_trip",
                collection_task_name="Drawer Push Pull",
                collection_task_id="drawer_push_pull",
                collection_task_description="drawer demo",
                gripper_task="firm_grasp",
            )

            with patch.object(real_teleop, "datetime") as fake_datetime:
                fake_datetime.now.return_value.strftime.return_value = "2026-07-03"
                real_teleop._apply_formal_dataset_layout(args)

                self.assertEqual(args.collection_round_trip_phase, "in")
                self.assertEqual(args.collection_task_id, "drawer_push_pull_in")
                self.assertEqual(args.collection_task_name, "Drawer Push Pull_in")
                self.assertEqual(Path(args.dataset_root), Path(tmp_dir) / "2026-07-03_Drawer_Push_Pull_in")

                self.assertTrue(real_teleop._toggle_collection_round_trip_phase(args))

            self.assertEqual(args.collection_round_trip_phase, "out")
            self.assertEqual(args.collection_task_id, "drawer_push_pull_out")
            self.assertEqual(args.collection_task_name, "Drawer Push Pull_out")
            self.assertEqual(Path(args.dataset_root), Path(tmp_dir) / "2026-07-03_Drawer_Push_Pull_out")

    def test_preview_defaults_to_all_configured_cameras(self) -> None:
        cameras = {
            "front": object(),
            "left_wrist": object(),
            "right_wrist": object(),
        }

        selected = real_teleop._select_preview_camera_sources(cameras, "")

        self.assertEqual(set(selected), set(cameras))

    def test_camera_status_marks_waiting_and_stale_cameras(self) -> None:
        fresh = _FakeCameraSource(
            {
                "camera_key": "front",
                "serial": "123",
                "frame_count": 4,
                "last_frame_age_sec": 0.05,
                "stale_after_sec": 1.0,
                "timeout_count": 0,
                "consecutive_timeout_count": 0,
                "restart_count": 0,
            }
        )
        waiting = _FakeCameraSource(
            {
                "camera_key": "left_wrist",
                "serial": "456",
                "frame_count": 0,
                "last_runtime_error": "Frame did not arrive",
                "stale_after_sec": 1.0,
            }
        )
        stale = _FakeCameraSource(
            {
                "camera_key": "right_wrist",
                "serial": "789",
                "frame_count": 9,
                "last_frame_age_sec": 2.5,
                "stale_after_sec": 1.0,
            }
        )

        status = real_teleop._build_camera_status(
            {"front": fresh, "left_wrist": waiting, "right_wrist": stale},
            preview_camera_sources={"front": fresh},
            camera_previewer=object(),
            now=10.0,
        )

        self.assertEqual(status["front"]["status_level"], "ok")
        self.assertTrue(status["front"]["preview_active"])
        self.assertEqual(status["left_wrist"]["status_level"], "waiting")
        self.assertIn("Frame did not arrive", status["left_wrist"]["status_message"])
        self.assertEqual(status["right_wrist"]["status_level"], "error")
        self.assertEqual(status["right_wrist"]["status_label"], "数据停滞")

    def test_wrist_depth_status_detects_repeated_depth_metadata(self) -> None:
        repeated_frames = [
            _fake_depth_frame(frame_index=1, source_timestamp=100.0),
            _fake_depth_frame(frame_index=2, source_timestamp=100.1),
            _fake_depth_frame(frame_index=3, source_timestamp=100.2),
            _fake_depth_frame(frame_index=4, source_timestamp=100.3),
        ]
        left_wrist = _FakeCameraSource(
            {
                "camera_key": "left_wrist",
                "serial": "456",
                "frame_count": 1,
                "last_frame_age_sec": 0.01,
                "stale_after_sec": 1.0,
                "depth_enabled": True,
                "depth_scale_m_per_unit": 0.001,
            },
            enable_depth=True,
            frames=repeated_frames,
        )
        monitor = real_teleop.WristDepthFreshnessMonitor(stale_repeat_threshold=2)

        first = real_teleop._build_camera_status(
            {"left_wrist": left_wrist},
            now=100.0,
            depth_monitor=monitor,
        )
        self.assertEqual(first["left_wrist"]["depth_status_level"], "waiting")

        real_teleop._build_camera_status(
            {"left_wrist": left_wrist},
            now=100.1,
            depth_monitor=monitor,
        )
        stale = real_teleop._build_camera_status(
            {"left_wrist": left_wrist},
            now=100.2,
            depth_monitor=monitor,
        )

        self.assertEqual(stale["left_wrist"]["status_level"], "error")
        self.assertEqual(stale["left_wrist"]["depth_status_label"], "Wrist Depth 停滞")
        self.assertGreaterEqual(stale["left_wrist"]["depth_signature_repeat_count"], 2)

    def test_wrist_depth_status_marks_missing_expected_wrist_cameras(self) -> None:
        status = real_teleop._with_expected_wrist_depth_camera_statuses(
            {"front": {"status_level": "ok"}},
            args=_args(wrist_depth_mask=True),
            now=100.0,
        )

        self.assertEqual(status["left_wrist"]["status_level"], "error")
        self.assertEqual(status["right_wrist"]["depth_status_level"], "error")
        self.assertIn("front", status)

    def test_camera_startup_self_check_publishes_failed_depth_payload(self) -> None:
        repeated_frames = [
            _fake_depth_frame(frame_index=index, source_timestamp=100.0 + index * 0.01)
            for index in range(1, 8)
        ]
        left_wrist = _FakeCameraSource(
            {
                "camera_key": "left_wrist",
                "serial": "456",
                "frame_count": 1,
                "last_frame_age_sec": 0.01,
                "stale_after_sec": 1.0,
                "depth_enabled": True,
                "depth_scale_m_per_unit": 0.001,
            },
            enable_depth=True,
            frames=repeated_frames,
        )
        ui = _FakeUi()
        clutch_input = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(wrist_depth_mask=True),
            components=_components(),
        )
        clutch_input.start()

        payload = real_teleop._run_camera_startup_self_check(
            camera_sources={"left_wrist": left_wrist},
            clutch_input=clutch_input,
            duration_sec=0.03,
            interval_sec=0.01,
        )

        self.assertEqual(payload["status"], "failed")
        self.assertIn("left_wrist", payload["failed_keys"])
        published_checks = [
            item["camera_self_check"]
            for item in ui.published
            if "camera_self_check" in item
        ]
        self.assertTrue(published_checks)
        self.assertEqual(published_checks[-1]["status"], "failed")

    def test_camera_startup_self_check_exits_early_after_consecutive_fresh_frames(self) -> None:
        class _FreshCameraSource:
            def __init__(self) -> None:
                self.frame_count = 0
                self.config = SimpleNamespace(
                    serial="fresh-camera",
                    width=640,
                    height=480,
                    fps=15,
                    color_space="bgr",
                    enable_depth=False,
                    depth_width=640,
                    depth_height=480,
                    depth_fps=15,
                )
                self.name = "fresh_camera"

            def device_info(self) -> dict[str, object]:
                self.frame_count += 1
                return {
                    "camera_key": "front",
                    "serial": "fresh-camera",
                    "frame_count": self.frame_count,
                    "last_frame_age_sec": 0.01,
                    "stale_after_sec": 1.0,
                }

        clutch_input = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=_FakeUi(),
            args=_args(),
            components=_components(),
        )
        payload = real_teleop._run_camera_startup_self_check(
            camera_sources={"front": _FreshCameraSource()},
            clutch_input=clutch_input,
            duration_sec=1.0,
            interval_sec=0.01,
        )

        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["completion_reason"], "fresh_frames")
        self.assertEqual(payload["fresh_frame_streaks"], {"front": 3})
        self.assertLess(float(payload["elapsed_sec"]), 1.0)

    def test_camera_startup_self_check_fails_empty_sources_when_wrist_depth_required(self) -> None:
        ui = _FakeUi()
        clutch_input = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(wrist_depth_mask=True),
            components=_components(),
        )

        payload = real_teleop._run_camera_startup_self_check(
            camera_sources={},
            clutch_input=clutch_input,
            duration_sec=0.0,
        )

        self.assertEqual(payload["status"], "failed")
        self.assertIn("left_wrist", payload["failed_keys"])
        self.assertEqual(ui.published[-1]["camera_status"]["left_wrist"]["status_level"], "error")

    def test_tactile_camera_status_highlights_partial_ready_and_worker_error(self) -> None:
        status = real_teleop._build_tactile_camera_status(
            {
                "left": {
                    "configured": True,
                    "connect_tactile": True,
                    "is_connected": True,
                    "tactile_devices": ["dev0", "dev1"],
                    "tactile_ready": {"dev0": True, "dev1": False},
                    "tactile_frame_ids": {"dev0": 11},
                    "tactile_timestamp": 99.5,
                    "tactile_read_hz": 15.0,
                    "tactile_process_read": True,
                    "tactile_worker_alive": True,
                    "tactile_worker_receiver_alive": True,
                },
                "right": {
                    "configured": True,
                    "connect_tactile": True,
                    "is_connected": True,
                    "tactile_devices": ["dev0"],
                    "tactile_ready": {"dev0": True},
                    "tactile_frame_ids": {"dev0": 12},
                    "tactile_timestamp": 99.9,
                    "tactile_process_read": True,
                    "tactile_worker_alive": False,
                    "tactile_worker_receiver_alive": True,
                },
            },
            now=100.0,
        )

        self.assertEqual(status["left"]["status_level"], "warn")
        self.assertEqual(status["left"]["ready_count"], 1)
        self.assertEqual(status["left"]["ready_total"], 2)
        self.assertEqual(status["right"]["status_level"], "error")
        self.assertEqual(status["right"]["status_label"], "worker 异常")

    def test_web_ui_snapshot_includes_camera_status_fields(self) -> None:
        ui = calibration_web_ui.CalibrationWebUi(open_browser=False)
        ui.publish(
            camera_status={"front": {"status_level": "ok"}},
            camera_self_check={"status": "passed", "status_level": "ok"},
            tactile_camera_status={"left": {"status_level": "warn"}},
        )

        snapshot = ui.snapshot()

        self.assertEqual(snapshot["camera_status"]["front"]["status_level"], "ok")
        self.assertEqual(snapshot["camera_self_check"]["status"], "passed")
        self.assertEqual(snapshot["tactile_camera_status"]["left"]["status_level"], "warn")

    def test_collection_frequency_warning_trips_below_15hz(self) -> None:
        recorder = real_teleop.RealTeleopDatasetRecorder.__new__(
            real_teleop.RealTeleopDatasetRecorder
        )
        current_time = time.monotonic()
        recorder.active_episode_index = 0
        recorder.sample_builder = SimpleNamespace(
            frame_index=4,
            last_camera_rejection=None,
            last_camera_frame_age_sec_by_key={},
        )
        recorder.frequency_warning_hz = 15.0
        recorder.frequency_window_sec = 3.0
        recorder.frequency_min_elapsed_sec = 2.0
        recorder._episode_start_monotonic = current_time - 3.0
        recorder._record_timestamps = deque(
            [current_time - 3.0, current_time - 2.0, current_time - 1.0, current_time]
        )

        status = recorder.collection_frequency_status()

        self.assertLess(status["sample_hz"], 15.0)
        self.assertTrue(status["warning"]["active"])
        self.assertIn("15.0 Hz", status["warning"]["message"])

    def test_dataset_recorder_pause_blocks_samples_and_resumes_same_episode(self) -> None:
        recorder = real_teleop.RealTeleopDatasetRecorder.__new__(
            real_teleop.RealTeleopDatasetRecorder
        )
        current_time = time.monotonic()
        recorder.active_episode_index = 12
        recorder.sample_builder = SimpleNamespace(
            frame_index=4,
            include_debug_metadata=False,
            include_action_debug_metadata=False,
            last_camera_rejection=None,
            last_camera_frame_age_sec_by_key={},
        )
        recorder.frequency_warning_hz = 15.0
        recorder.frequency_window_sec = 3.0
        recorder.frequency_min_elapsed_sec = 2.0
        recorder._episode_start_monotonic = current_time - 3.0
        recorder._record_timestamps = deque(
            [current_time - 3.0, current_time - 2.0, current_time - 1.0, current_time]
        )
        recorder._pause_started_at = None
        recorder._pause_started_monotonic = None
        recorder._pause_reason = None
        recorder._pause_started_frame_index = None
        recorder._pause_started_step_index = None
        recorder._pause_started_last_state = None
        recorder._pause_segments = []
        recorder._paused_accumulated_sec = 0.0
        recorder.writer = SimpleNamespace(tactile_frame_counts=lambda _episode_index: {})
        recorder.camera_sources = {}
        recorder.dataset_root = Path("/tmp/dataset")

        paused = recorder.pause_episode(
            reason="hold_requested",
            now=time.time(),
            step_index=42,
            last_state="RUNNING",
        )

        self.assertTrue(paused)
        self.assertTrue(recorder.is_paused)
        self.assertFalse(recorder.can_record_samples)
        self.assertIsNone(recorder.record_step(SimpleNamespace()))
        self.assertEqual(recorder.frame_count, 4)
        paused_status = recorder.collection_frequency_status()
        self.assertTrue(paused_status["paused"])
        self.assertFalse(paused_status["warning"]["active"])

        resumed = recorder.resume_episode(
            reason="runtime_running_warmup_ready",
            now=time.time(),
            step_index=45,
        )

        self.assertTrue(resumed)
        self.assertFalse(recorder.is_paused)
        self.assertTrue(recorder.can_record_samples)
        self.assertEqual(len(recorder.pause_summary()["recording_pause_segments"]), 1)
        self.assertEqual(
            recorder.pause_summary()["recording_pause_segments"][0]["start_frame_index"],
            4,
        )

        class _Observer:
            def handle_step(self, _step_result, **_kwargs):
                recorder.sample_builder.frame_index += 1
                return "recorded"

        recorder.writer_observer = _Observer()
        self.assertEqual(recorder.record_step(SimpleNamespace()), "recorded")
        self.assertEqual(recorder.frame_count, 5)
        self.assertFalse(recorder.sample_builder.include_action_debug_metadata)

        seen_action_debug: list[bool] = []
        seen_action_metadata: list[dict[str, object]] = []

        class _DebugObserver:
            def handle_step(self, _step_result, **_kwargs):
                seen_action_debug.append(bool(recorder.sample_builder.include_action_debug_metadata))
                seen_action_metadata.append(dict(_kwargs.get("metadata_extra") or {}))
                recorder.sample_builder.frame_index += 1
                return "debug-recorded"

        recorder.writer_observer = _DebugObserver()
        self.assertEqual(
            recorder.record_step(
                SimpleNamespace(),
                include_action_debug_metadata=True,
            ),
            "debug-recorded",
        )
        self.assertEqual(seen_action_debug, [True])
        self.assertIn("action_debug", seen_action_metadata[0])
        self.assertEqual(
            seen_action_metadata[0]["action_debug"]["schema"],
            "dexteleop_action_debug_v1",
        )
        self.assertFalse(recorder.sample_builder.include_action_debug_metadata)

    def test_dataset_recorder_action_debug_metadata_reads_command_backend(self) -> None:
        current_pose = np.eye(4)
        target_pose = np.eye(4)
        target_pose[0, 3] = 0.03

        class _Backend:
            def __init__(self, state: dict[str, object]) -> None:
                self.state = dict(state)

            def get_state(self) -> dict[str, object]:
                return dict(self.state)

        command_backend = _Backend(
            {
                "arm_state_source": "pose_impedance_command",
                "last_command_timestamp": 9.97,
                "last_command_sequence": 12,
                "command_sequence": 12,
                "latest_target_sequence": 12,
                "current_tcp_pose": current_pose,
                "last_target_tcp_pose": target_pose,
                "latest_target_tcp_pose6": [0.03, 0.0, 0.0, 0.0, 0.0, 0.0],
                "latest_target_joint_deg": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "last_impedance_record": {
                    "step": 7,
                    "reference": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "command_pose": [0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "measured_pose": [0.015, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "command_joint_deg": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    "measured_joint_deg": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
                    "backend_target_sequence": 12,
                    "backend_target_timestamp": 9.97,
                    "backend_target_age_sec_at_step_start": 0.015,
                    "backend_controller_step_start_wall_time": 9.985,
                    "backend_controller_step_end_wall_time": 9.989,
                },
                "effective_control_dt_sec": 0.025,
                "effective_step_scale": 1.0,
            }
        )
        observation_backend = _Backend(
            {
                "arm_state_source": "hardware_state_observation",
                "last_command_timestamp": None,
                "current_tcp_pose": current_pose,
                "last_target_tcp_pose": None,
            }
        )
        recorder = real_teleop.RealTeleopDatasetRecorder.__new__(
            real_teleop.RealTeleopDatasetRecorder
        )
        recorder.components = SimpleNamespace(
            active_sides=("left",),
            left_command_backend=command_backend,
            right_command_backend=None,
            left_state_backend=observation_backend,
            right_state_backend=None,
        )
        command = TeleopCommand(timestamp=9.94)
        command.add_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.zeros(6),
                frame_name="joint",
                metadata={"action_pose_source": "unit_test"},
            )
        )
        step_result = SimpleNamespace(
            timestamp=10.0,
            command=command,
            command_dispatched=True,
        )

        metadata = recorder._action_debug_metadata(step_result, record_wall_time=10.02)

        left_command = metadata["command_backend_by_side"]["left"]
        self.assertTrue(left_command["available"])
        self.assertEqual(left_command["arm_state_source"], "pose_impedance_command")
        self.assertAlmostEqual(left_command["last_command_timestamp"], 9.97)
        self.assertEqual(left_command["last_command_sequence"], 12)
        self.assertEqual(left_command["latest_target_sequence"], 12)
        self.assertAlmostEqual(left_command["last_command_age_sec_at_sample"], 0.03)
        self.assertAlmostEqual(left_command["sample_to_last_command_dt_sec"], -0.03)
        self.assertAlmostEqual(left_command["last_command_age_sec_at_record"], 0.05)
        self.assertAlmostEqual(left_command["target_current_translation_m"], 0.03)
        self.assertEqual(left_command["backend_state_compact"]["latest_target_tcp_pose6"][0], 0.03)
        impedance_record = left_command["last_impedance_record"]
        self.assertEqual(impedance_record["backend_target_sequence"], 12)
        self.assertEqual(impedance_record["command_pose"][0], 0.02)
        self.assertEqual(impedance_record["measured_pose"][0], 0.015)
        self.assertEqual(
            left_command["backend_state_compact"]["last_impedance_record"]["command_pose"][0],
            0.02,
        )
        left_observation = metadata["observation_backend_by_side"]["left"]
        self.assertTrue(left_observation["available"])
        self.assertEqual(left_observation["arm_state_source"], "hardware_state_observation")
        self.assertNotIn("same_as_command_backend", left_observation)
        self.assertEqual(metadata["command_by_side"]["left"]["command_type"], "joint_vector")
        self.assertEqual(metadata["command_effectors"], ["left_arm"])

    def test_episode_recording_state_publishes_frequency_warning(self) -> None:
        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=_components(),
        )

        control.set_episode_recording_state(
            active=True,
            episode_dir="/tmp/demo",
            frame_count=7,
            frequency_status={
                "sample_hz": 12.25,
                "target_hz": 15.0,
                "warning": {"active": True, "message": "WARNING: slow"},
            },
        )

        published = ui.published[-1]
        self.assertEqual(published["episode_sample_hz"], 12.25)
        self.assertEqual(published["episode_target_hz"], 15.0)
        self.assertTrue(published["episode_frequency_warning"]["active"])

    def test_episode_recording_state_keeps_fix_init_available_but_blocks_config_controls(self) -> None:
        ui = _FakeUi()
        components = _components()
        components.fix_arm_controller = SimpleNamespace(
            status_payload=lambda: {
                "fix_arm_available": True,
                "fix_arm_selected_side": "left",
                "fix_arm_fixed_side": "",
                "fix_arm_active": False,
                "fix_arm_recalibrating": False,
                "fix_arm_status": "",
                "fix_arm_sample_count": None,
                "fix_arm_min_samples": None,
            }
        )
        components.init_arm_controller = SimpleNamespace(
            status_payload=lambda: {
                "init_arm_available": True,
                "init_arm_selected_side": "right",
                "init_arm_active_side": "",
                "init_arm_active": False,
                "init_arm_recalibrating": False,
                "init_arm_status": "ready",
                "init_arm_sample_count": None,
                "init_arm_min_samples": None,
                "init_arm_profile_path": "profile.json",
            }
        )
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=components,
        )
        control.publish_teleop_running()

        control.set_episode_recording_state(
            active=True,
            episode_dir="/tmp/demo",
            frame_count=7,
            frequency_status={
                "sample_hz": 15.0,
                "target_hz": 15.0,
                "warning": {"active": False, "message": ""},
            },
        )

        active_controls = set(ui.published[-1]["active_controls"])
        self.assertIn("episode_stop", active_controls)
        self.assertIn("h", active_controls)
        self.assertIn("q", active_controls)
        self.assertIn("d", active_controls)
        self.assertIn("c", active_controls)
        self.assertIn("fix_arm_toggle", active_controls)
        self.assertIn("fix_arm_select", active_controls)
        self.assertIn("g", active_controls)
        self.assertIn("init_arm_toggle", active_controls)
        self.assertIn("init_arm_select", active_controls)
        self.assertIn("toggle_camera_vis", active_controls)
        self.assertIn("toggle_tactile_vis", active_controls)
        self.assertNotIn("episode_start", active_controls)
        self.assertNotIn("round_trip_toggle", active_controls)
        self.assertNotIn("record_pose_snapshot", active_controls)
        self.assertNotIn("movement_multiple_set", active_controls)
        self.assertNotIn("save_vive_calibration_profile", active_controls)

    def test_episode_finalizing_state_blocks_restart_controls(self) -> None:
        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=_components(),
        )
        control.publish_teleop_running()

        control.set_episode_recording_state(
            active=False,
            episode_dir="/tmp/demo",
            frame_count=7,
            event_status="finalizing",
            event_message="正在后台保存。",
        )

        active_controls = set(ui.published[-1]["active_controls"])
        self.assertNotIn("episode_start", active_controls)
        self.assertNotIn("episode_stop", active_controls)
        self.assertNotIn("round_trip_toggle", active_controls)
        self.assertNotIn("movement_multiple_set", active_controls)
        self.assertIn("h", active_controls)
        self.assertIn("q", active_controls)
        self.assertIn("d", active_controls)

    def test_episode_finalize_completion_restores_hold_controls(self) -> None:
        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=_components(),
        )
        control.publish_teleop_running()

        control.set_episode_recording_state(
            active=False,
            episode_dir="/tmp/demo",
            frame_count=7,
            event_status="finalizing",
            event_message="正在后台保存。",
        )
        self.assertIn("h", ui.published[-1]["active_controls"])

        control.set_episode_finalize_state(
            status="saved",
            message="保存完成。",
            episode_index=0,
            frame_count=7,
        )

        active_controls = set(ui.published[-1]["active_controls"])
        self.assertIn("h", active_controls)
        self.assertIn("episode_start", active_controls)

    def test_episode_paused_state_keeps_stop_available_in_hold(self) -> None:
        ui = _FakeUi()
        components = _components()
        components.fix_arm_controller = SimpleNamespace(
            status_payload=lambda: {
                "fix_arm_available": True,
                "fix_arm_selected_side": "left",
                "fix_arm_fixed_side": "",
                "fix_arm_active": False,
                "fix_arm_recalibrating": False,
                "fix_arm_status": "",
                "fix_arm_sample_count": None,
                "fix_arm_min_samples": None,
            }
        )
        components.init_arm_controller = SimpleNamespace(
            status_payload=lambda: {
                "init_arm_available": True,
                "init_arm_selected_side": "right",
                "init_arm_active_side": "",
                "init_arm_active": False,
                "init_arm_recalibrating": False,
                "init_arm_status": "ready",
                "init_arm_sample_count": None,
                "init_arm_min_samples": None,
                "init_arm_profile_path": "profile.json",
            }
        )
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=components,
        )
        control._teleop_started = True

        control.set_episode_recording_state(
            active=True,
            episode_dir="/tmp/demo",
            frame_count=7,
            frequency_status={
                "sample_hz": None,
                "target_hz": 15.0,
                "paused": True,
                "pause_reason": "hold_requested",
                "warning": {"active": False, "message": ""},
            },
            event_status="paused",
            event_message="数采已暂停：hold_requested。",
        )
        control.request_hold()

        published = ui.published[-1]
        self.assertTrue(ui.published[-2]["episode_recording"])
        self.assertEqual(ui.published[-2]["episode_event_status"], "paused")
        self.assertIn("episode_stop", published["active_controls"])
        self.assertIn("p", published["active_controls"])
        self.assertNotIn("c", published["active_controls"])
        self.assertNotIn("episode_start", published["active_controls"])
        self.assertNotIn("round_trip_toggle", published["active_controls"])
        self.assertNotIn("movement_multiple_set", published["active_controls"])
        self.assertNotIn("calibrate_vive_coordinate_profile", published["active_controls"])
        self.assertNotIn("fix_arm_toggle", published["active_controls"])
        self.assertNotIn("init_arm_toggle", published["active_controls"])

    def test_episode_hold_can_be_requested_again_after_resume(self) -> None:
        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=_components(),
        )
        control.publish_teleop_running()
        control.set_episode_recording_state(active=True)

        control.request_hold()
        self.assertTrue(control.consume_hold_request())
        self.assertIn("episode_stop", ui.published[-1]["active_controls"])
        self.assertIn("p", ui.published[-1]["active_controls"])
        self.assertNotIn("h", ui.published[-1]["active_controls"])

        control.publish_teleop_running()
        control.set_episode_recording_state(active=True)
        self.assertIn("h", ui.published[-1]["active_controls"])
        self.assertNotIn("p", ui.published[-1]["active_controls"])

        control.request_hold()
        self.assertTrue(control.consume_hold_request())
        self.assertIn("episode_stop", ui.published[-1]["active_controls"])
        self.assertIn("p", ui.published[-1]["active_controls"])
        self.assertNotIn("h", ui.published[-1]["active_controls"])

    def test_episode_start_countdown_state_is_visible_and_blocks_restart(self) -> None:
        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=_components(),
        )

        control.set_episode_start_countdown_state(remaining_sec=2.4)

        published = ui.published[-1]
        self.assertEqual(published["episode_event_status"], "preparing")
        self.assertFalse(published["episode_recording"])
        self.assertIn("2.4 秒", published["episode_event_message"])
        self.assertNotIn("episode_start", published["active_controls"])
        self.assertIn("episode_stop", published["active_controls"])
        self.assertIn("h", published["active_controls"])

    def test_episode_start_countdown_runs_in_parallel_with_warmup(self) -> None:
        deadline = 13.0

        self.assertAlmostEqual(real_teleop._episode_start_countdown_remaining(10.0, deadline), 3.0)
        self.assertFalse(
            real_teleop._episode_start_countdown_ready_to_record(
                now=12.9,
                deadline=deadline,
                warmup_ready=True,
            )
        )
        self.assertFalse(
            real_teleop._episode_start_countdown_ready_to_record(
                now=14.0,
                deadline=deadline,
                warmup_ready=False,
            )
        )
        self.assertTrue(
            real_teleop._episode_start_countdown_ready_to_record(
                now=14.0,
                deadline=deadline,
                warmup_ready=True,
            )
        )

    def test_gui_episode_banner_has_preparing_state(self) -> None:
        html = calibration_web_ui._HTML_PAGE
        self.assertIn(".episode-banner.preparing", html)
        self.assertIn("3 秒后开始数采", html)

    def test_gui_c_shortcut_only_sends_fix_arm_not_runtime_home_or_quit(self) -> None:
        html_and_fallback = calibration_web_ui._HTML_PAGE + calibration_web_ui._FALLBACK_SCRIPT

        self.assertIn('id="shortcutCLabel"', calibration_web_ui._HTML_PAGE)
        self.assertIn('id="shortcutGLabel"', calibration_web_ui._HTML_PAGE)
        self.assertIn('data-event="fix_arm_toggle"', calibration_web_ui._HTML_PAGE)
        self.assertIn('data-event="init_arm_toggle"', calibration_web_ui._HTML_PAGE)
        self.assertIn("固定 ", calibration_web_ui._HTML_PAGE)
        self.assertIn("归位", calibration_web_ui._HTML_PAGE)
        self.assertNotIn("固定/急停", html_and_fallback)
        self.assertIn("无 fix-arm 时不可用", calibration_web_ui._HTML_PAGE)
        self.assertNotIn('? "c" : "q"', html_and_fallback)
        self.assertNotIn(": \"q\";", html_and_fallback)
        self.assertIn('eventName = "fix_arm_toggle";', html_and_fallback)
        self.assertIn('eventName = "init_arm_toggle";', html_and_fallback)
        self.assertNotIn('eventName = "c";', html_and_fallback)
        self.assertIn("C 仅用于运行中的 fix-arm", html_and_fallback)
        self.assertIn("固定臂不可用", html_and_fallback)
        self.assertIn("当前不可用", html_and_fallback)

    def test_gui_keeps_fix_arm_c_enabled_during_collection(self) -> None:
        html = calibration_web_ui._HTML_PAGE

        self.assertIn("state.capturing &&", html)
        self.assertIn('eventName === "c"', html)
        self.assertIn("state.fix_arm_available && state.teleop_running", html)
        self.assertIn("fix_arm_toggle", html)

    def test_impedance_guard_holds_all_rig_backends_before_fault(self) -> None:
        class FakeRig:
            def __init__(self) -> None:
                self.holds: list[str] = []

            def hold_motion(self, reason: str = "manual hold") -> None:
                self.holds.append(reason)

        class FakeSession:
            def __init__(self, rig: FakeRig) -> None:
                self.rig = rig
                self.state_machine = SimpleNamespace(state=real_teleop.TeleopRunState.RUNNING)
                self.faults: list[str] = []

            def fault(self, reason: str) -> None:
                self.faults.append(reason)
                self.state_machine.state = real_teleop.TeleopRunState.FAULT

        rig = FakeRig()
        session = FakeSession(rig)
        components = SimpleNamespace(session=session)
        issue = "right_arm pose impedance loop stopped before command dispatch"
        expected_reason = f"main_loop: {issue}"

        with patch.object(real_teleop, "_impedance_force_control_issue", return_value=issue), patch.object(
            real_teleop,
            "_log",
        ):
            ok = real_teleop._guard_impedance_force_control(
                SimpleNamespace(),
                components,
                context="main_loop",
                now=1.0,
            )

        self.assertFalse(ok)
        self.assertEqual(rig.holds, [f"control fault: {expected_reason}"])
        self.assertEqual(session.faults, [expected_reason])
        self.assertEqual(session.state_machine.state, real_teleop.TeleopRunState.FAULT)

    def test_impedance_guard_faults_even_if_hold_motion_reports_failure(self) -> None:
        class FailingRig:
            def __init__(self) -> None:
                self.holds: list[str] = []

            def hold_motion(self, reason: str = "manual hold") -> None:
                self.holds.append(reason)
                raise RuntimeError("simulated hold failure")

        class FakeSession:
            def __init__(self, rig: FailingRig) -> None:
                self.rig = rig
                self.state_machine = SimpleNamespace(state=real_teleop.TeleopRunState.RUNNING)
                self.faults: list[str] = []

            def fault(self, reason: str) -> None:
                self.faults.append(reason)
                self.state_machine.state = real_teleop.TeleopRunState.FAULT

        rig = FailingRig()
        session = FakeSession(rig)
        components = SimpleNamespace(session=session)
        issue = "left_arm pose impedance is disabled before command dispatch"
        expected_reason = f"main_loop: {issue}"

        with patch.object(real_teleop, "_impedance_force_control_issue", return_value=issue), patch.object(
            real_teleop,
            "_log",
        ) as log:
            ok = real_teleop._guard_impedance_force_control(
                SimpleNamespace(),
                components,
                context="main_loop",
                now=1.0,
            )

        self.assertFalse(ok)
        self.assertEqual(rig.holds, [f"control fault: {expected_reason}"])
        self.assertEqual(session.faults, [expected_reason])
        self.assertTrue(
            any("hold_motion failed" in str(call.args[1]) for call in log.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
