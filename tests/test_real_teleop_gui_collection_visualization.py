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
from control_station.web import calibration_web_ui
from teleop_runtime.schemas import EffectorCommand, ObservationChannel, TeleopCommand
from teleop_runtime.sources.pose import VivePoseSource
from teleop_runtime.tactile_visualizer import DMTacDistributedForceVisualizer
from teleop_runtime.tactile_visualizer import dmtac_visualization_snapshot_from_backends
from dataset_contracts.tactile_calibration import save_tactile_calibration
from collection_runtime.progress.profiles import (
    bind_dataset_progress_profile,
    load_managed_task_profile,
)


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
    def test_tracker_side_warning_requires_explicit_confirmation_before_home(self) -> None:
        calibration = real_teleop.TeleopCalibration()
        post_yaw = np.eye(4)
        post_yaw[:3, :3] = [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        calibration.set_world_to_desk(post_yaw)
        calibration.metadata["virtual_desk_post_yaw_deg"] = 180.0
        components = _components()
        components.active_sides = ("left", "right")
        components.session.calibration = calibration
        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(virtual_desk_post_yaw_deg=180.0),
            components=components,
        )
        now = time.time()
        channels = {
            "left_wrist_pose": ObservationChannel(
                channel_name="left_wrist_pose",
                capability="pose",
                side="left",
                value=np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, -0.2],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                frame_name="vive_world",
                timestamp=now,
                source_name="vive",
            ),
            "right_wrist_pose": ObservationChannel(
                channel_name="right_wrist_pose",
                capability="pose",
                side="right",
                value=np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.2],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                frame_name="vive_world",
                timestamp=now,
                source_name="vive",
            ),
        }

        control.publish_raw_tracker_pose_state(channels=channels, force=True)
        control._request_calibration_with_tracker_side_guard(source="gui")

        self.assertFalse(control._calibration_requested)
        self.assertTrue(control._tracker_side_confirmation_pending)
        confirmation_publish = ui.published[-1]
        self.assertIn(
            "tracker_side_confirm_continue",
            confirmation_publish["active_controls"],
        )
        self.assertIn(
            "tracker_side_confirm_cancel",
            confirmation_publish["active_controls"],
        )
        self.assertTrue(
            confirmation_publish["tracker_side_check"]["confirmation_required"]
        )

        control._confirm_tracker_side_warning()

        self.assertTrue(control.consume_calibration_request())
        self.assertFalse(control._tracker_side_confirmation_pending)
        self.assertTrue(control._tracker_side_initial_check_completed)

    def test_tracker_side_warning_cancel_keeps_tracker_swap_available(self) -> None:
        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=_components(),
        )
        control._tracker_side_check = {
            "status": "ambiguous",
            "warning_required": True,
            "message": "too close",
        }

        control._request_calibration_with_tracker_side_guard(source="gui")
        control._cancel_tracker_side_warning()

        self.assertFalse(control.consume_calibration_request())
        self.assertFalse(control._tracker_side_confirmation_pending)
        self.assertIn("w", control._current_active_controls())

    def test_tracker_side_confirmation_survives_transient_tracker_dropout(self) -> None:
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=_FakeUi(),
            args=_args(),
            components=_components(),
        )
        control._tracker_side_check = {
            "status": "swapped",
            "warning_required": True,
            "message": "swapped",
        }
        control._request_calibration_with_tracker_side_guard(source="gui")

        control._update_tracker_side_check(
            {
                "status": "unavailable",
                "warning_required": False,
                "message": "temporary dropout",
            }
        )

        self.assertTrue(control._tracker_side_confirmation_pending)
        self.assertTrue(control._tracker_side_check["confirmation_required"])
        self.assertEqual(control._tracker_side_check["live_status"], "unavailable")

    def test_tracker_side_home_request_waits_for_fresh_check_then_continues(self) -> None:
        components = _components()
        components.active_sides = ("left", "right")
        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=components,
        )

        control._request_calibration_with_tracker_side_guard(source="gui")

        self.assertTrue(control._tracker_side_request_pending)
        self.assertFalse(control._calibration_requested)
        self.assertNotIn("p", control._current_active_controls())

        control._update_tracker_side_check(
            {
                "status": "correct",
                "warning_required": False,
                "message": "correct",
            }
        )

        self.assertFalse(control._tracker_side_request_pending)
        self.assertTrue(control._calibration_requested)
        self.assertEqual(ui.published[-1]["active_controls"], ["q"])

    def test_terminal_cannot_bypass_web_tracker_side_confirmation(self) -> None:
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=_FakeUi(),
            args=_args(),
            components=_components(),
        )
        control._tracker_side_check = {
            "status": "swapped",
            "warning_required": True,
            "message": "swapped",
        }

        control._request_calibration_with_tracker_side_guard(source="gui")
        control._request_calibration_with_tracker_side_guard(source="terminal")

        self.assertTrue(control._tracker_side_confirmation_pending)
        self.assertFalse(control._calibration_requested)

    def test_two_pedal_fast_home_supersedes_pending_tracker_confirmation(self) -> None:
        components = _components()
        components.active_sides = ("left", "right")
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=None,
            args=_args(special_arm_control_mode="foot-pad"),
            components=components,
            foot_pad_input=SimpleNamespace(foot_pad_count=2, is_online=True),
        )
        control._teleop_started = True
        control._hold_active = True
        control._tracker_side_check = {
            "status": "swapped",
            "warning_required": True,
            "message": "swapped",
        }
        control._request_calibration_with_tracker_side_guard(source="gui")
        self.assertTrue(control._tracker_side_confirmation_pending)
        self.assertNotIn("two_pedal_b", control._current_active_controls())

        control._handle_two_pedal_b_event(now=10.0)

        self.assertTrue(control._tracker_side_confirmation_pending)
        self.assertEqual(
            control._tracker_side_confirmation_request_kind,
            "two_pedal_fast",
        )
        self.assertEqual(
            control._tracker_side_check["confirmation_result"], "pending"
        )

        control._confirm_tracker_side_warning()

        self.assertTrue(control._calibration_requested)
        self.assertTrue(control._two_pedal_fast_home_requested)
        self.assertFalse(control._tracker_side_guard_request_active)
        self.assertTrue(control._tracker_side_initial_check_completed)
        self.assertEqual(
            control.consume_two_pedal_fast_home_target(),
            (True, None),
        )

    def test_explicit_home_does_not_override_queued_two_pedal_fast_home(self) -> None:
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=None,
            args=_args(special_arm_control_mode="foot-pad"),
            components=_components(),
            foot_pad_input=SimpleNamespace(foot_pad_count=2, is_online=True),
        )
        control._teleop_started = True
        control._hold_active = True
        control._tracker_side_check = {
            "status": "swapped",
            "warning_required": True,
            "message": "swapped",
        }

        control._handle_two_pedal_b_event(now=10.0)
        control._request_calibration_with_tracker_side_guard(source="gui")

        self.assertFalse(control._calibration_requested)
        self.assertTrue(control._tracker_side_confirmation_pending)
        self.assertEqual(
            control._tracker_side_confirmation_request_kind,
            "two_pedal_fast",
        )

    def test_two_pedal_request_replaces_queued_normal_home_kind(self) -> None:
        components = _components()
        components.active_sides = ("left", "right")
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=None,
            args=_args(special_arm_control_mode="foot-pad"),
            components=components,
            foot_pad_input=SimpleNamespace(foot_pad_count=2, is_online=True),
        )
        control._teleop_started = True
        control._hold_active = True
        control._tracker_side_check = {
            "status": "correct",
            "warning_required": False,
            "message": "correct",
        }

        control._request_calibration_with_tracker_side_guard(source="gui")
        self.assertFalse(control._two_pedal_fast_home_requested)

        control._handle_two_pedal_b_event(now=10.0)

        self.assertTrue(control._calibration_requested)
        self.assertTrue(control._two_pedal_fast_home_requested)
        self.assertEqual(
            control.consume_two_pedal_fast_home_target(now=10.0),
            (True, None),
        )

    def test_two_pedal_b_releases_single_arm_fix_without_hold_or_collection_pause(self) -> None:
        class FixController:
            fixed = True
            fixed_sides = ("right",)

            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def start_two_pedal_transition(self, **kwargs: object):
                self.calls.append(dict(kwargs))
                return SimpleNamespace(ok=True, message="pending")

        controller = FixController()
        components = _components()
        components.fix_arm_controller = controller
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=None,
            args=_args(special_arm_control_mode="foot-pad"),
            components=components,
            foot_pad_input=SimpleNamespace(foot_pad_count=2, is_online=True),
        )
        control._teleop_started = True
        control._hold_active = False
        control._episode_recording_ui_active = True
        control.request_hold = lambda **_kwargs: self.fail(
            "single-arm to bimanual B must not request global Hold"
        )

        control._handle_two_pedal_b_event(now=10.0)

        self.assertEqual(
            controller.calls,
            [
                {
                    "fixed_side": None,
                    "now": 10.0,
                    "timeout_sec": 1.0,
                    "min_samples": 1,
                }
            ],
        )
        self.assertFalse(control._hold_active)
        self.assertFalse(control._calibration_requested)
        self.assertTrue(control._episode_recording_ui_active)
        self.assertTrue(control._two_pedal_transition_active)

    def test_two_pedal_mode_switch_is_rejected_while_init_is_active(self) -> None:
        components = _components()
        components.init_arm_controller = SimpleNamespace(
            active=True,
            qpos_active=True,
            moving=True,
            recapturing=False,
        )
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=_FakeUi(),
            args=_args(special_arm_control_mode="foot-pad"),
            components=components,
            foot_pad_input=SimpleNamespace(foot_pad_count=2, is_online=True),
        )
        control._teleop_started = True
        control._hold_active = True

        control._handle_two_pedal_b_event(now=10.0)

        self.assertFalse(control._calibration_requested)
        self.assertFalse(control._two_pedal_fast_home_requested)
        self.assertNotIn("two_pedal_b", control._current_active_controls())
        self.assertIn("two_pedal_c", control._current_active_controls())
        self.assertEqual(control._two_pedal_alert_level, "red")
        self.assertIn("请先按 C", control._two_pedal_alert_message)

    def test_current_single_arm_request_cancels_pending_gui_transition_state(self) -> None:
        class FixController:
            fixed = True
            fixed_sides = ("left",)
            two_pedal_transition_pending = False

            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def start_two_pedal_transition(self, **kwargs: object):
                self.calls.append(dict(kwargs))
                return SimpleNamespace(
                    ok=True,
                    message="older transition cancelled; left remains fixed",
                )

        controller = FixController()
        components = _components()
        components.fix_arm_controller = controller
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=None,
            args=_args(special_arm_control_mode="foot-pad"),
            components=components,
            foot_pad_input=SimpleNamespace(foot_pad_count=2, is_online=True),
        )
        control._teleop_started = True
        control._two_pedal_transition_active = True
        control._two_pedal_transition_target_fixed_side = "right"

        control._request_two_pedal_fixed_side(
            "left",
            now=10.0,
            source="pedal_2",
        )

        self.assertEqual(len(controller.calls), 1)
        self.assertFalse(control._two_pedal_transition_active)
        self.assertIsNone(control._two_pedal_transition_target_fixed_side)

    def test_two_pedal_transition_cleanup_does_not_depend_on_web_ui(self) -> None:
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=None,
            args=_args(special_arm_control_mode="foot-pad"),
            components=_components(),
            foot_pad_input=SimpleNamespace(foot_pad_count=2, is_online=True),
        )
        control._two_pedal_transition_active = True
        control._two_pedal_transition_target_fixed_side = "right"
        control._two_pedal_alert_level = "orange"

        control.publish_fix_arm_home_capture_progress(
            SimpleNamespace(
                active=True,
                completed=True,
                failed=False,
                side="left",
                message="completed",
            )
        )

        self.assertFalse(control._two_pedal_transition_active)
        self.assertIsNone(control._two_pedal_transition_target_fixed_side)
        self.assertEqual(control._two_pedal_alert_level, "")

    def test_normal_home_is_not_misclassified_as_two_pedal_fast_home(self) -> None:
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=None,
            args=_args(special_arm_control_mode="foot-pad"),
            components=_components(),
            foot_pad_input=SimpleNamespace(foot_pad_count=2, is_online=True),
        )
        control._teleop_started = True
        control._hold_active = True
        control._tracker_side_check = {
            "status": "correct",
            "warning_required": False,
            "message": "correct",
        }

        control._request_calibration_with_tracker_side_guard(source="gui")

        self.assertTrue(control._calibration_requested)
        self.assertFalse(control._two_pedal_fast_home_requested)
        self.assertEqual(
            control.consume_two_pedal_fast_home_target(),
            (False, None),
        )
        self.assertTrue(control.consume_calibration_request())

    def test_two_pedal_fresh_tracker_wait_times_out_after_one_second(self) -> None:
        components = _components()
        components.active_sides = ("left", "right")
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=None,
            args=_args(special_arm_control_mode="foot-pad"),
            components=components,
            foot_pad_input=SimpleNamespace(foot_pad_count=2, is_online=True),
        )
        control._teleop_started = True
        control._hold_active = True
        control._tracker_side_check = {
            "status": "unavailable",
            "warning_required": False,
            "message": "stale",
        }

        control._handle_two_pedal_b_event(now=10.0)
        self.assertTrue(control._tracker_side_request_pending)

        control._poll_tracker_side_request_timeout(now=11.0)

        self.assertFalse(control._tracker_side_request_pending)
        self.assertFalse(control._calibration_requested)
        self.assertEqual(control._two_pedal_alert_level, "red")
        self.assertEqual(control._two_pedal_alert_expires_at, 14.0)

    def test_normal_start_fresh_tracker_wait_times_out_to_not_started_state(self) -> None:
        components = _components()
        components.active_sides = ("left", "right")
        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(calibration_timeout=10.0),
            components=components,
        )
        control._tracker_side_check = {
            "status": "unavailable",
            "warning_required": False,
            "message": "stale",
        }

        control._request_calibration_with_tracker_side_guard(
            source="gui",
            now=10.0,
        )

        self.assertTrue(control._tracker_side_request_pending)
        self.assertEqual(control._tracker_side_request_deadline, 11.0)
        self.assertFalse(control._calibration_requested)

        control._poll_tracker_side_request_timeout(now=11.0)

        self.assertFalse(control._tracker_side_request_pending)
        self.assertFalse(control._calibration_requested)
        self.assertFalse(control._teleop_started)
        self.assertFalse(control._hold_active)
        self.assertEqual(
            control._tracker_side_check["confirmation_result"],
            "runtime_home_tracker_timeout",
        )
        self.assertEqual(ui.published[-1]["stage"], "RM65 真机遥操准备")
        self.assertIn("p", ui.published[-1]["active_controls"])
        self.assertTrue(
            any("已回到未遥操状态" in message for _, message in ui.messages)
        )

    def test_normal_resume_fresh_tracker_wait_times_out_to_existing_hold(self) -> None:
        components = _components()
        components.active_sides = ("left", "right")
        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(calibration_timeout=10.0),
            components=components,
        )
        control._teleop_started = True
        control._hold_active = True
        control._tracker_side_check = {
            "status": "unavailable",
            "warning_required": False,
            "message": "stale",
        }

        control._request_calibration_with_tracker_side_guard(
            source="gui",
            now=20.0,
        )
        control._poll_tracker_side_request_timeout(now=21.0)

        self.assertTrue(control._teleop_started)
        self.assertTrue(control._hold_active)
        self.assertFalse(control._calibration_requested)
        self.assertEqual(ui.published[-1]["stage"], "Hold / 停臂保持")
        self.assertIn("p", ui.published[-1]["active_controls"])
        self.assertTrue(
            any("机械臂保持 Hold" in message for _, message in ui.messages)
        )

    def test_two_pedal_tracker_wait_and_home_share_one_second_budget(self) -> None:
        components = _components()
        components.active_sides = ("left", "right")
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=None,
            args=_args(special_arm_control_mode="foot-pad"),
            components=components,
            foot_pad_input=SimpleNamespace(foot_pad_count=2, is_online=True),
        )
        control._teleop_started = True
        control._hold_active = True
        control._tracker_side_check = {
            "status": "unavailable",
            "warning_required": False,
            "message": "stale",
        }

        control._handle_two_pedal_b_event(now=10.0)
        control._update_tracker_side_check(
            {
                "status": "correct",
                "warning_required": False,
                "message": "correct",
            }
        )

        self.assertEqual(
            control.consume_two_pedal_fast_home_target(now=10.6),
            (True, None),
        )
        self.assertAlmostEqual(
            control.two_pedal_fast_home_timeout_sec(),
            0.4,
            places=6,
        )

    def test_locked_dual_arm_tracker_warning_does_not_offer_disabled_swap(self) -> None:
        components = _components()
        components.active_sides = ("left", "right")
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=_FakeUi(),
            args=_args(),
            components=components,
        )
        control._tracker_swap_locked = True
        control._tracker_side_check = {
            "status": "swapped",
            "warning_required": True,
            "message": "swapped",
        }

        control._request_calibration_with_tracker_side_guard(source="gui")

        confirmation = str(
            control._tracker_side_check["confirmation_message"]
        )
        self.assertIn("当前双臂运行映射已锁定", confirmation)
        self.assertNotIn("点击“交换左右 Tracker”", confirmation)
        self.assertFalse(control._tracker_side_check["swap_available"])

    def test_tracker_side_is_rechecked_immediately_before_home_capture(self) -> None:
        components = _components()
        components.active_sides = ("left", "right")
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=_FakeUi(),
            args=_args(),
            components=components,
        )
        control._tracker_side_check = {
            "status": "correct",
            "warning_required": False,
            "message": "correct",
        }
        control._request_calibration_with_tracker_side_guard(source="gui")
        self.assertTrue(control._tracker_side_guard_request_active)

        control._tracker_side_check = {
            "status": "swapped",
            "warning_required": True,
            "message": "crossed during countdown",
        }

        self.assertFalse(control._final_tracker_side_home_admission())
        self.assertFalse(control._tracker_side_guard_request_active)

    def test_explicit_tracker_side_confirmation_completes_gui_session_check(self) -> None:
        components = _components()
        components.active_sides = ("left", "right")
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=_FakeUi(),
            args=_args(),
            components=components,
        )
        control._tracker_side_check = {
            "status": "swapped",
            "warning_required": True,
            "message": "swapped",
        }
        control._request_calibration_with_tracker_side_guard(source="gui")
        control._confirm_tracker_side_warning()

        self.assertTrue(control._final_tracker_side_home_admission())
        self.assertFalse(control._tracker_side_override_granted)
        self.assertFalse(control._tracker_side_guard_request_active)
        self.assertTrue(control._tracker_side_initial_check_completed)
        self.assertFalse(control._tracker_side_check["monitoring_active"])

    def test_tracker_side_check_runs_only_for_first_home_in_gui_session(self) -> None:
        components = _components()
        components.active_sides = ("left", "right")
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=_FakeUi(),
            args=_args(),
            components=components,
        )
        control._tracker_side_check = {
            "status": "correct",
            "warning_required": False,
            "message": "correct",
        }

        control._request_calibration_with_tracker_side_guard(source="gui")
        self.assertTrue(control._final_tracker_side_home_admission())
        self.assertTrue(control._tracker_side_initial_check_completed)
        self.assertTrue(control.consume_calibration_request())

        control._tracker_side_check.update(
            {
                "status": "swapped",
                "warning_required": True,
                "message": "must be ignored after first admission",
            }
        )
        control._request_calibration_with_tracker_side_guard(source="gui")

        self.assertTrue(control._calibration_requested)
        self.assertFalse(control._tracker_side_confirmation_pending)
        self.assertFalse(control._tracker_side_guard_request_active)

    def test_tracker_side_evaluator_stops_after_first_gui_session_check(self) -> None:
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=_FakeUi(),
            args=_args(),
            components=_components(),
        )
        control._complete_initial_tracker_side_check(result="check_passed")

        with patch(
            "teleop_runtime.gui.real_teleop_control.evaluate_tracker_side_assignment"
        ) as evaluator:
            control.publish_raw_tracker_pose_state(
                channels={},
                now=10.0,
                force=True,
            )

        evaluator.assert_not_called()

    def test_q6_dmtac_command_contract_survives_missing_startup_device(self) -> None:
        self.assertTrue(
            real_teleop._dmtac_command_contract_configured(
                gripper_input_source="q6",
                angle_trigger_source=None,
            )
        )
        self.assertTrue(
            real_teleop._dmtac_command_contract_configured(
                gripper_input_source="angle_trigger",
                angle_trigger_source=None,
            )
        )
        self.assertFalse(
            real_teleop._dmtac_command_contract_configured(
                gripper_input_source="none",
                angle_trigger_source=None,
            )
        )

    def test_gui_keeps_only_tracker_swap_from_legacy_view_and_pose_tuning_controls(self) -> None:
        html = calibration_web_ui._HTML_PAGE

        self.assertIn("Tracker 输入", html)
        self.assertIn('data-event="w">交换左右 Tracker</button>', html)
        self.assertNotIn("左右 / 视图", html)
        self.assertNotIn("姿态微调", html)
        for event in ("1", "2", "t", "v", "f", "x", "j", "l", "i", "k", "u", "o"):
            self.assertNotIn(f'data-event="{event}"', html)

    def test_rendered_page_keeps_modern_javascript_syntax_intact(self) -> None:
        page = calibration_web_ui._render_initial_html(
            calibration_web_ui.CalibrationWebUiState().__dict__
        )

        self.assertNotIn('.join("\n")', page)
        self.assertIn('.join("\\n")', page)
        self.assertIn(
            'buttons[i].textContent = state.camera_visualization_active',
            page,
        )
        self.assertIn(
            'buttons[i].textContent = state.tactile_visualization_active',
            page,
        )
        self.assertIn(
            'initSection.hidden = !Boolean(state.init_arm_available)',
            page,
        )

    def test_left_status_pills_show_foot_pad_online_count_or_unavailable(self) -> None:
        html = calibration_web_ui._HTML_PAGE

        self.assertIn('state.foot_pad_requested || state.foot_pad_available', html)
        self.assertIn('online ? `${count} 个已连接` : "未连接 / 未就绪"', html)
        self.assertIn('online ? "ok" : "warn"', html)
        self.assertIn(".pill.ok", html)
        self.assertIn(".pill.warn", html)

    def test_foot_pad_topology_does_not_hide_fix_or_init_gui_controls(self) -> None:
        html = calibration_web_ui._HTML_PAGE

        self.assertIn("captureButton.hidden = false;", html)
        self.assertIn("initArmToggleButtonEl.hidden = false;", html)
        self.assertIn(
            "fixArmToggleButtonEl.parentElement.hidden = false;",
            html,
        )
        self.assertIn("initArmMoveToQposButtonEl.hidden = annotationOnly;", html)
        self.assertIn("initArmMoveBothToQposButtonEl.hidden = annotationOnly;", html)
        self.assertNotIn(
            "initArmMoveToQposButtonEl.hidden = annotationOnly || twoPedal;",
            html,
        )
        self.assertNotIn(
            "initArmMoveBothToQposButtonEl.hidden = annotationOnly || twoPedal;",
            html,
        )

    def test_gripper_force_copy_explains_temporary_normalized_command(self) -> None:
        html = calibration_web_ui._HTML_PAGE

        self.assertIn("临时夹爪驱动力（归一化 0–1）", html)
        self.assertIn("不是牛顿值", html)
        self.assertIn("开始数采前会恢复任务配置", html)
        self.assertIn("保存为新数据集永久力度", html)
        self.assertIn("gripper_force_persist", html)

    def test_pre_home_wait_services_visualization_callbacks_before_quit(self) -> None:
        class Clutch:
            enabled = True
            should_quit = False

            def __init__(self) -> None:
                self.startup_enabled: list[bool] = []
                self.poll_count = 0

            def set_startup_qpos_maintenance_enabled(self, enabled):
                self.startup_enabled.append(bool(enabled))

            def poll(self):
                self.poll_count += 1
                self.should_quit = self.poll_count >= 2

            @staticmethod
            def consume_calibration_request():
                return False

            @staticmethod
            def publish_raw_tracker_pose_state(**_kwargs):
                return None

        clutch = Clutch()
        request_times: list[float] = []
        update_times: list[float] = []
        runtime_status_updates: list[bool] = []

        class StateBackend:
            def __init__(self) -> None:
                self.refresh_count = 0

            def refresh_state_from_hardware(self):
                self.refresh_count += 1
                raise RuntimeError("idle startup must not read RM65 state")

        left_state = StateBackend()
        right_state = StateBackend()
        components = SimpleNamespace(
            active_sides=("left", "right"),
            arm_state_read_executor=None,
            left_command_backend=None,
            right_command_backend=None,
            left_state_backend=left_state,
            right_state_backend=right_state,
            left_preview_backend=SimpleNamespace(),
            right_preview_backend=SimpleNamespace(),
            session=SimpleNamespace(poll_sources=lambda now: ({}, [])),
        )
        with patch.object(real_teleop, "_handle_ik_debug_log_toggle_request"), patch.object(
            real_teleop,
            "_advance_startup_gui_qpos_maintenance",
        ), patch.object(real_teleop.time, "sleep"):
            result = real_teleop._wait_for_real_teleop_start_request(
                components,
                clutch,
                arm_mode="real",
                args=_args(arm_control_mode="position"),
                control_ui=None,
                ik_debug_log_recorder=SimpleNamespace(),
                ik_debug_profile_base_enabled=False,
                visualization_request_handler=request_times.append,
                visualization_update_handler=update_times.append,
                runtime_status_update_handler=lambda: runtime_status_updates.append(True),
            )

        self.assertIsNone(result)
        self.assertEqual(len(request_times), 2)
        self.assertEqual(len(update_times), 1)
        self.assertEqual(len(runtime_status_updates), 2)
        self.assertEqual(clutch.startup_enabled, [True, False])
        self.assertEqual(left_state.refresh_count, 0)
        self.assertEqual(right_state.refresh_count, 0)

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
        self.assertEqual(args.init_arm_left_qpos_speed_scale, 5.0)
        self.assertEqual(args.init_arm_right_qpos_speed_scale, 5.0)

    def test_parse_args_two_pedals_do_not_disable_requested_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_dir = Path(tmp_dir)
            calibration_profile = profile_dir / "20260702_090000_vive_rm65_real_teleop.json"
            calibration_profile.write_text("{}", encoding="utf-8")
            missing_init_profile = profile_dir / "missing-init-qpos.json"
            with patch.object(real_teleop, "DEFAULT_CALIBRATION_PROFILE_DIR", profile_dir), patch.object(
                real_teleop, "configured_foot_pad_count", return_value=2
            ), patch.object(
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
                    "--init-arm-available",
                    "--init-arm-profile",
                    str(missing_init_profile),
                ],
            ):
                args = real_teleop.parse_args()

        self.assertTrue(args.init_arm_available)
        self.assertTrue(args.init_arm_requested)
        self.assertEqual(args.init_arm_disabled_reason, "")
        self.assertEqual(args.special_arm_control_mode, "foot-pad")

    def test_missing_saved_two_pedal_device_restores_manual_init_arm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            init_profile = Path(tmp_dir) / "init_arm.json"
            init_profile.write_text("{}", encoding="utf-8")
            args = SimpleNamespace(
                init_arm_available=False,
                init_arm_requested=True,
                init_arm_disabled_reason="two_pedal_layout",
                init_arm_profile=str(init_profile),
            )

            restored, message = real_teleop._restore_init_arm_after_foot_pad_fallback(
                args,
                live_foot_pad_count=None,
            )

        self.assertTrue(restored)
        self.assertTrue(args.init_arm_available)
        self.assertEqual(args.init_arm_disabled_reason, "")
        self.assertIn("已恢复", message)

    def test_offline_foot_pad_restores_requested_init_from_stale_disabled_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            init_profile = Path(tmp_dir) / "init_arm.json"
            init_profile.write_text("{}", encoding="utf-8")
            args = SimpleNamespace(
                init_arm_available=False,
                init_arm_requested=True,
                init_arm_disabled_reason="",
                init_arm_profile=str(init_profile),
            )

            restored, message = real_teleop._restore_init_arm_after_foot_pad_fallback(
                args,
                live_foot_pad_count=None,
            )

        self.assertTrue(restored)
        self.assertTrue(args.init_arm_available)
        self.assertEqual(args.init_arm_disabled_reason, "")
        self.assertIn("已恢复", message)

    def test_live_two_pedal_device_does_not_block_requested_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            init_profile = Path(tmp_dir) / "init_arm.json"
            init_profile.write_text("{}", encoding="utf-8")
            args = SimpleNamespace(
                init_arm_available=False,
                init_arm_requested=True,
                init_arm_disabled_reason="two_pedal_layout",
                init_arm_profile=str(init_profile),
            )

            restored, message = real_teleop._restore_init_arm_after_foot_pad_fallback(
                args,
                live_foot_pad_count=2,
            )

        self.assertTrue(restored)
        self.assertTrue(args.init_arm_available)
        self.assertIn("已恢复", message)

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

    def test_single_arm_defaults_cannot_be_restored_as_requested_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_dir = Path(tmp_dir)
            calibration_profile = (
                profile_dir / "20260702_090000_vive_rm65_real_teleop.json"
            )
            calibration_profile.write_text("{}", encoding="utf-8")
            with patch.object(
                real_teleop,
                "DEFAULT_CALIBRATION_PROFILE_DIR",
                profile_dir,
            ), patch.object(
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

        self.assertFalse(args.fix_arm)
        self.assertFalse(args.init_arm_available)
        self.assertFalse(args.init_arm_requested)
        restored, _message = real_teleop._restore_init_arm_after_foot_pad_fallback(
            args,
            live_foot_pad_count=None,
        )
        self.assertFalse(restored)
        self.assertFalse(args.init_arm_available)

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

    def test_device_retry_is_restored_only_after_hold_is_consumed(self) -> None:
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
        self.assertNotIn("device_retry", running_publish["active_controls"])

        control.request_hold()
        self.assertNotIn("device_retry", ui.published[-1]["active_controls"])
        admitted, message = control.device_retry_admission()
        self.assertFalse(admitted)
        self.assertIn("Hold 正在生效", message)

        control.set_visualization_state(camera_active=True)
        self.assertNotIn("device_retry", ui.published[-1]["active_controls"])

        self.assertTrue(control.consume_hold_request())
        control.set_visualization_state(camera_active=True)
        self.assertIn("device_retry", ui.published[-1]["active_controls"])

    def test_stale_device_retry_event_is_rejected_by_running_backend_state(self) -> None:
        ui = _FakeUi(["device_retry"])
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=_components(),
        )
        control.publish_teleop_running()

        control.poll()

        self.assertFalse(control.consume_device_retry_request())
        self.assertTrue(
            any(
                title == "Device" and "运行中请先进入 Hold" in message
                for title, message in ui.messages
            )
        )

    def test_device_retry_backend_gate_rechecks_state_after_event_was_queued(self) -> None:
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=_FakeUi(),
            args=_args(),
            components=_components(),
        )
        admitted, _message = control.device_retry_admission()
        self.assertTrue(admitted)

        control._teleop_started = True
        control._hold_active = False
        admitted, message = control.device_retry_admission()

        self.assertFalse(admitted)
        self.assertIn("运行中请先进入 Hold", message)

    def test_device_retry_backend_gate_rejects_exact_qpos_activity(self) -> None:
        components = _components()
        components.init_arm_controller = SimpleNamespace(qpos_active=True)
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=_FakeUi(),
            args=_args(),
            components=components,
        )

        admitted, message = control.device_retry_admission()

        self.assertFalse(admitted)
        self.assertIn("精确 Init qpos", message)

    def test_pending_hold_rejects_movement_change_until_hardware_hold_runs(self) -> None:
        args = _args(movement_multiple=1.0)
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=_FakeUi(),
            args=args,
            components=_components(),
        )
        control.publish_teleop_running()
        control.request_hold()

        self.assertIsNone(
            control._set_movement_multiple_value(1.5, source="test")
        )
        self.assertEqual(args.movement_multiple, 1.0)

        self.assertTrue(control.consume_hold_request())
        self.assertEqual(
            control._set_movement_multiple_value(1.5, source="test"),
            1.5,
        )
        self.assertEqual(args.movement_multiple, 1.5)

    def test_hold_event_is_a_barrier_for_later_events_in_same_gui_batch(self) -> None:
        ui = _FakeUi(["device_retry", "p", "h", "movement_multiple_up"])
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(movement_multiple=1.0),
            components=_components(),
        )
        control.publish_teleop_running()

        control.poll()

        self.assertTrue(control.consume_hold_request())
        self.assertFalse(control.consume_device_retry_request())
        self.assertFalse(control.consume_calibration_request())
        self.assertEqual(control.args.movement_multiple, 1.0)

    def test_quit_event_discards_all_other_events_in_same_gui_batch(self) -> None:
        ui = _FakeUi(["device_retry", "p", "q", "movement_multiple_up"])
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(movement_multiple=1.0),
            components=_components(),
        )
        control.publish_teleop_running()

        control.poll()

        self.assertTrue(control.should_quit)
        self.assertFalse(control.consume_device_retry_request())
        self.assertFalse(control.consume_calibration_request())
        self.assertEqual(control.args.movement_multiple, 1.0)

    def test_init_emergency_stop_discards_other_events_in_same_gui_batch(self) -> None:
        ui = _FakeUi(
            ["device_retry", "p", "init_qpos_emergency_stop", "movement_multiple_up"]
        )
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(movement_multiple=1.0),
            components=_components(),
        )
        control.publish_teleop_running()

        control.poll()

        self.assertTrue(control.consume_init_qpos_emergency_stop_request())
        self.assertFalse(control.consume_device_retry_request())
        self.assertFalse(control.consume_calibration_request())
        self.assertEqual(control.args.movement_multiple, 1.0)

    def test_visualization_refresh_cannot_reenable_controls_during_recording(self) -> None:
        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=_components(),
        )
        control.publish_teleop_running()
        control.set_episode_recording_state(active=True)

        control.set_visualization_state(camera_active=True)

        active_controls = set(ui.published[-1]["active_controls"])
        self.assertIn("episode_stop", active_controls)
        self.assertNotIn("episode_start", active_controls)
        self.assertNotIn("device_retry", active_controls)
        self.assertNotIn("movement_multiple_set", active_controls)

    def test_rejected_episode_start_restores_one_shot_controls(self) -> None:
        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=_components(),
        )
        control.publish_teleop_running()

        control._publish_episode_start_rejection("test rejection")

        self.assertIn("episode_start", ui.published[-1]["active_controls"])
        self.assertIn("test rejection", ui.published[-1]["control_hint"])

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
        control._teleop_started = True

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

    def test_profile_hold_cancel_can_preserve_last_live_fix_snapshot(self) -> None:
        class FakeFixController:
            fixed = True

            def __init__(self) -> None:
                self.cancel_calls: list[dict[str, object]] = []

            def cancel_active(self, **kwargs):
                self.cancel_calls.append(dict(kwargs))
                self.fixed = False
                return SimpleNamespace(ok=True, side="left")

        fix_controller = FakeFixController()
        components = SimpleNamespace(
            fix_arm_controller=fix_controller,
            init_arm_controller=None,
            dual_arm_retargeter=None,
            left_command_backend=None,
            right_command_backend=None,
        )

        messages = real_teleop._cancel_special_arm_modes_for_hold(
            components,
            now=1.0,
            reason="progress_arm_profile_transition",
            clear_fix_snapshots=False,
        )

        self.assertEqual(messages, ["左臂 fix-arm 已停止"])
        self.assertFalse(
            fix_controller.cancel_calls[0]["clear_snapshots"]
        )

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
        self.assertIn("device_retry", active_controls)
        self.assertEqual(ready_publish["movement_multiple"], 1.0)
        self.assertEqual(ready_publish["capture_stage"], "runtime_home")
        self.assertEqual(ready_publish["capture_phase"], "ready")
        self.assertFalse(ready_publish["teleop_running"])

    def test_debug_episode_only_mode_does_not_expose_formal_dataset_delete(self) -> None:
        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(no_collection=True, no_episode_recording=False),
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
        self.assertNotIn("episode_delete", active_controls)

    def test_gui_events_feed_episode_and_visualization_requests(self) -> None:
        ui = _FakeUi(
            [
                "episode_start",
                "episode_stop",
                "toggle_camera_vis",
                "toggle_tactile_vis",
                "enable_tactile_solving",
            ]
        )
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=_components(),
        )
        control._teleop_started = True

        control.poll()

        self.assertTrue(control.consume_episode_start_request())
        self.assertTrue(control.consume_episode_stop_request())
        self.assertTrue(control.consume_camera_visualization_toggle_request())
        self.assertTrue(control.consume_tactile_visualization_toggle_request())
        self.assertTrue(control.consume_tactile_solving_enable_request())

    def test_progress_gui_without_profile_does_not_block_episode_start(self) -> None:
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=_FakeUi(),
            args=_args(
                collection_progress_gui=True,
                collection_progress_profile=None,
            ),
            components=_components(),
        )
        control._teleop_started = True

        admitted, message = control.episode_start_admission()

        self.assertTrue(admitted, message)
        self.assertEqual(message, "")

    def test_task_selection_adopts_target_binding_without_old_profile_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_root = Path(tmp_dir) / "datasets"
            task_config = Path(tmp_dir) / "tasks.json"
            task_config.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "key": "bound_task",
                                "english_name": "Bound Task",
                                "chinese_name": "已绑定任务",
                                "gripper_task": "firm_grasp",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            target_root = base_root / "2026-07-02_Bound_Task"
            target_profile = load_managed_task_profile(
                "example_three_stage_pick_place_progress"
            ).progress_profile
            bind_dataset_progress_profile(target_root, target_profile)
            ui = _FakeUi()
            args = _args(
                dataset_root=str(base_root),
                dataset_base_root=str(base_root),
                task_config=str(task_config),
                collection_progress_profile=load_managed_task_profile(
                    "dmtac_rm65_bimanual_position_v1_progress"
                ).progress_profile,
            )
            real_teleop._apply_formal_dataset_layout(args)
            control = real_teleop.RealTeleopControlInput(
                terminal=_FakeTerminal(),
                ui=ui,
                args=args,
                components=_components(),
            )
            control._teleop_started = True

            with patch.object(real_teleop, "datetime") as fake_datetime:
                fake_datetime.now.return_value.strftime.return_value = "2026-07-02"
                applied = control._apply_episode_start_payload(
                    json.dumps(
                        {
                            "dataset_base_root": str(base_root),
                            "task_key": "bound_task",
                            "task_english_name": "",
                            "task_chinese_name": "",
                        },
                        ensure_ascii=False,
                    )
                )

        self.assertTrue(applied)
        self.assertEqual(args.collection_task_id, "bound_task")
        self.assertEqual(
            args.collection_progress_profile.profile_id,
            target_profile.profile_id,
        )

    def test_task_selection_defaults_new_blank_dataset_to_four_stage_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_root = Path(tmp_dir) / "datasets"
            args = _args(
                dataset_root=str(base_root),
                dataset_base_root=str(base_root),
                collection_progress_gui=True,
                collection_progress_profile=None,
            )
            real_teleop._apply_formal_dataset_layout(args)
            control = real_teleop.RealTeleopControlInput(
                terminal=_FakeTerminal(),
                ui=_FakeUi(),
                args=args,
                components=_components(),
            )

            with patch.object(real_teleop, "datetime") as fake_datetime:
                fake_datetime.now.return_value.strftime.return_value = "2026-08-03"
                applied = control._apply_episode_start_payload(
                    json.dumps(
                        {
                            "dataset_base_root": str(base_root),
                            "task_key": "writing",
                            "task_english_name": "Writing",
                            "task_chinese_name": "书写",
                        },
                        ensure_ascii=False,
                    )
                )

        self.assertTrue(applied)
        self.assertEqual(
            args.collection_progress_profile.profile_id,
            "dmtac_rm65_bimanual_position_writing_four_stage_progress",
        )

    def test_stale_task_select_event_cannot_mutate_active_episode_contract(self) -> None:
        ui = _FakeUi()
        args = _args(
            collection_task_name="original_task",
            collection_task_id="original_task",
            dataset_root="/tmp/original_dataset",
        )
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=args,
            components=_components(),
        )
        control._episode_recording_ui_active = True

        applied = control._apply_episode_start_payload(
            json.dumps(
                {
                    "task_key": "replacement_task",
                    "task_english_name": "Replacement Task",
                    "dataset_base_root": "/tmp/replacement_dataset",
                    "ignored_camera_keys": [],
                    "ignored_tactile_keys": [],
                }
            )
        )

        self.assertFalse(applied)
        self.assertEqual(args.collection_task_name, "original_task")
        self.assertEqual(args.collection_task_id, "original_task")
        self.assertEqual(args.dataset_root, "/tmp/original_dataset")
        self.assertTrue(
            any("不能切换任务" in message for _title, message in ui.messages)
        )

    def test_gui_tactile_solving_control_is_only_available_before_teleop_or_in_hold(self) -> None:
        args = _args(
            tactile_solving_available=True,
            tactile_solving_active=False,
            tactile_visualization_available=False,
        )
        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(), ui=ui, args=args, components=_components()
        )

        control.start()
        ready = next(
            item for item in reversed(ui.published) if "active_controls" in item
        )
        self.assertIn("enable_tactile_solving", ready["active_controls"])

        control.publish_teleop_running()
        running = next(
            item for item in reversed(ui.published) if "active_controls" in item
        )
        self.assertNotIn("enable_tactile_solving", running["active_controls"])

        control.request_hold(reason="unit_test")
        pending_hold = next(
            item for item in reversed(ui.published) if "active_controls" in item
        )
        self.assertNotIn("enable_tactile_solving", pending_hold["active_controls"])

        self.assertTrue(control.consume_hold_request())
        control.set_visualization_state(status="hold settled")
        hold = next(
            item for item in reversed(ui.published) if "active_controls" in item
        )
        self.assertIn("enable_tactile_solving", hold["active_controls"])

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

        self.assertAlmostEqual(args.movement_multiple, 1.35)
        self.assertAlmostEqual(args.translation_scale, 1.35)

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
        ui.publish(active_controls=["gripper_force_set", "gripper_force_persist"])

        event = 'gripper_force_set:{"left":0.31,"right":0.64}'
        self.assertTrue(ui.push_event(event))
        self.assertEqual(ui.consume_events(), [event])
        persist_event = 'gripper_force_persist:{"left":0.31,"right":0.64}'
        self.assertTrue(ui.push_event(persist_event))
        self.assertEqual(ui.consume_events(), [persist_event])

    def test_web_ui_accepts_init_qpos_speed_payload_when_control_active(self) -> None:
        ui = calibration_web_ui.CalibrationWebUi(open_browser=False)
        ui.publish(active_controls=["init_arm_qpos_speed_set"])

        event = 'init_arm_qpos_speed_set:{"left":4.0,"right":6.0}'
        self.assertTrue(ui.push_event(event))
        self.assertEqual(ui.consume_events(), [event])

    def test_gui_init_qpos_speed_updates_both_sides_atomically(self) -> None:
        ui = _FakeUi(['init_arm_qpos_speed_set:{"left":4.0,"right":6.0}'])
        args = _args()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=args,
            components=_components(),
        )

        control.poll()

        self.assertEqual(args.init_arm_left_qpos_speed_scale, 4.0)
        self.assertEqual(args.init_arm_right_qpos_speed_scale, 6.0)
        state = next(
            item
            for item in reversed(ui.published)
            if "init_arm_qpos_speed_scale_by_side" in item
        )
        self.assertEqual(
            state["init_arm_qpos_speed_scale_by_side"],
            {"left": 4.0, "right": 6.0},
        )
        self.assertTrue(any("统一使用该设置" in message for _, message in ui.messages))

    def test_gui_init_qpos_speed_rejects_partial_invalid_update(self) -> None:
        ui = _FakeUi(['init_arm_qpos_speed_set:{"left":4.0,"right":20.0}'])
        args = _args()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=args,
            components=_components(),
        )

        control.poll()

        self.assertFalse(hasattr(args, "init_arm_left_qpos_speed_scale"))
        self.assertFalse(hasattr(args, "init_arm_right_qpos_speed_scale"))
        self.assertEqual(
            control._init_arm_qpos_speed_scale_by_side,
            {"left": 5.0, "right": 5.0},
        )
        self.assertTrue(any("倍率无效" in message for _, message in ui.messages))

    def test_init_qpos_emergency_stop_is_operation_stop_not_process_quit(self) -> None:
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=_FakeUi(["init_qpos_emergency_stop"]),
            args=_args(),
            components=_components(),
        )

        control.poll()

        assert control.should_quit is False
        assert control.consume_init_qpos_emergency_stop_request() is True
        assert any("不退出遥操程序" in message for _, message in control.ui.messages)

    def test_init_qpos_emergency_stop_holds_without_latching_arm_backend(self) -> None:
        class _CommandBackend:
            def __init__(self) -> None:
                self.emergency_stop_calls = 0

            def emergency_stop(self) -> None:
                self.emergency_stop_calls += 1

        class _Controller:
            qpos_active = True
            qpos_operation_id = "op-1"

            def __init__(self) -> None:
                self.cancelled = False

            def cancel_active(self, **_kwargs):
                self.cancelled = True
                return SimpleNamespace(ok=True, message="cancelled")

        class _Session:
            def __init__(self) -> None:
                self.hold_messages: list[str] = []

            def hold_hardware(self, message: str) -> None:
                self.hold_messages.append(message)

        controller = _Controller()
        session = _Session()
        left_backend = _CommandBackend()
        right_backend = _CommandBackend()
        components = SimpleNamespace(
            active_sides=("left", "right"),
            init_arm_controller=controller,
            left_command_backend=left_backend,
            right_command_backend=right_backend,
            session=session,
        )
        control = SimpleNamespace(
            _teleop_started=False,
            _cancel_init_arm_qpos_sequence=lambda **_kwargs: None,
            _publish_init_arm_state=lambda **_kwargs: None,
        )

        real_teleop._handle_init_qpos_emergency_stop(components, control, now=1.0)

        assert controller.cancelled is True
        assert session.hold_messages
        assert left_backend.emergency_stop_calls == 0
        assert right_backend.emergency_stop_calls == 0

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

    def test_gui_gripper_force_persist_binds_new_dataset_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            components = _components()
            components.active_sides = ("right",)
            components.right_hand_backend = DMTacGripperBackend(
                DMTacGripperBackendConfig(
                    side="right",
                    connect_hardware=False,
                    connect_tactile=False,
                )
            )
            args = _args(dataset_root=str(Path(tmp_dir) / "new_dataset"))
            control = real_teleop.RealTeleopControlInput(
                terminal=_FakeTerminal(),
                ui=_FakeUi(),
                args=args,
                components=components,
            )

            control._persist_gripper_force_from_gui('{"right":0.73}')

            contract = real_teleop.read_gripper_force_contract(args.dataset_root)
            assert contract is not None
            assert contract.force_by_side_dict == {"right": 0.73}
            assert args._gripper_force_dataset_contract == contract.to_dict()
            assert components.right_hand_backend.config.torque == 0.73

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
        self.assertNotIn("task_select", ui.snapshot()["active_controls"])

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

    def test_http_episode_start_preserves_dataset_capture_contract_selection(self) -> None:
        encoded = calibration_web_ui._encode_http_event_payload(
            {
                "event": "episode_start",
                "dataset_base_root": "/tmp/datasets",
                "ignored_camera_keys": ["right_wrist"],
                "ignored_tactile_keys": ["left"],
            }
        )
        event, raw_payload = encoded.split(":", 1)
        payload = json.loads(raw_payload)

        self.assertEqual(event, "episode_start")
        self.assertEqual(payload["ignored_camera_keys"], ["right_wrist"])
        self.assertEqual(payload["ignored_tactile_keys"], ["left"])

    def test_web_ui_accepts_init_arm_toggle_and_side_when_control_active(self) -> None:
        ui = calibration_web_ui.CalibrationWebUi(open_browser=False)
        ui.publish(
            active_controls=[
                "init_arm_toggle",
                "init_arm_select",
                "init_arm_move_to_qpos",
                "init_arm_move_both_to_qpos",
            ]
        )

        self.assertTrue(ui.push_event("init_arm_toggle"))
        self.assertTrue(ui.push_event("init_arm_select:right"))
        self.assertTrue(ui.push_event("init_arm_move_to_qpos"))
        self.assertTrue(ui.push_event("init_arm_move_both_to_qpos"))
        self.assertEqual(
            ui.consume_events(),
            [
                "init_arm_toggle",
                "init_arm_select:right",
                "init_arm_move_to_qpos",
                "init_arm_move_both_to_qpos",
            ],
        )
        self.assertTrue(
            any(
                item.get("tag") == "WebGUIEvent"
                and "accepted: init_arm_toggle" in item.get("message", "")
                for item in ui.snapshot()["messages"]
            )
        )

    def test_web_ui_publishes_exact_qpos_progress_fields(self) -> None:
        ui = calibration_web_ui.CalibrationWebUi(open_browser=False)

        ui.publish(
            init_arm_qpos_active=True,
            init_arm_qpos_phase="qpos_executing",
            init_arm_qpos_step_index=12,
            init_arm_qpos_step_count=40,
            init_arm_qpos_max_error_deg=0.25,
            init_arm_sequence_active=True,
            init_arm_sequence_side="left",
            init_arm_sequence_phase="left_qpos",
            init_arm_sequence_status="running left",
            fix_arm_fixed_sides=["left", "right"],
            fix_hold_active=True,
        )

        snapshot = ui.snapshot()
        self.assertTrue(snapshot["init_arm_qpos_active"])
        self.assertEqual(snapshot["init_arm_qpos_phase"], "qpos_executing")
        self.assertEqual(snapshot["init_arm_qpos_step_index"], 12)
        self.assertEqual(snapshot["init_arm_qpos_step_count"], 40)
        self.assertEqual(snapshot["init_arm_qpos_max_error_deg"], 0.25)
        self.assertTrue(snapshot["init_arm_sequence_active"])
        self.assertEqual(snapshot["init_arm_sequence_side"], "left")
        self.assertEqual(snapshot["init_arm_sequence_phase"], "left_qpos")
        self.assertEqual(snapshot["init_arm_sequence_status"], "running left")
        self.assertEqual(snapshot["fix_arm_fixed_sides"], ["left", "right"])
        self.assertTrue(snapshot["fix_hold_active"])

    def test_fix_hold_disables_gripper_and_blocks_new_episode_start(self) -> None:
        ui = _FakeUi(["episode_start"])
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=_components(),
        )
        control._teleop_started = True
        control._gripper_control_enabled = True

        control.set_fix_hold_active(True, message="both sides fixed-like")
        control.poll()

        self.assertTrue(control.fix_hold_active())
        self.assertFalse(control.gripper_control_enabled())
        self.assertFalse(control.consume_episode_start_request())
        self.assertTrue(any("Fix-Hold" in message for _tag, message in ui.messages))

        control.set_fix_hold_active(False, message="one side released")
        self.assertFalse(control.fix_hold_active())
        self.assertTrue(control._gripper_control_enabled)

    def test_terminal_episode_start_cannot_bypass_fix_hold_admission(self) -> None:
        class StartTerminal(_FakeTerminal):
            def __init__(self) -> None:
                self.requested = True

            def consume_episode_start_request(self) -> bool:
                requested = self.requested
                self.requested = False
                return requested

        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=StartTerminal(),
            ui=ui,
            args=_args(),
            components=_components(),
        )
        control._teleop_started = True
        control.set_fix_hold_active(True, message="both sides fixed-like")

        self.assertFalse(control.consume_episode_start_request())
        self.assertTrue(any("Fix-Hold" in message for _tag, message in ui.messages))

    def test_gui_exact_qpos_event_uses_dedicated_planner_not_legacy_g_flow(self) -> None:
        ui = _FakeUi(["init_arm_move_to_qpos"])
        components = _components()
        controller = SimpleNamespace(
            selected_side="left",
            active=False,
            qpos_active=False,
            status_payload=lambda: {
                "init_arm_available": True,
                "init_arm_selected_side": "left",
                "init_arm_active_side": "",
                "init_arm_active": False,
                "init_arm_recalibrating": False,
                "init_arm_status": "ready",
            },
        )
        components.init_arm_controller = controller
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(arm_control_mode="position"),
            components=components,
        )
        control._teleop_started = True
        transition = SimpleNamespace(ok=True, side="left", message="planning")

        with patch.object(real_teleop, "_start_gui_qpos_init", return_value=transition) as start:
            control.poll()

        start.assert_called_once()
        self.assertTrue(any("后台路径规划" in message for _tag, message in ui.messages))

    def test_gui_exact_qpos_event_is_admitted_before_home_only_with_startup_executor(self) -> None:
        components = _components()
        components.collision_guard_requested = True
        components.dual_arm_retargeter = object()
        components.active_sides = ("left", "right")
        components.init_arm_controller = SimpleNamespace(
            selected_side="left",
            active=False,
            qpos_active=False,
            status_payload=lambda: {"init_arm_available": True},
        )
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=_FakeUi(),
            args=_args(arm_control_mode="position"),
            components=components,
        )
        transition = SimpleNamespace(ok=True, side="left", message="planning")

        with patch.object(real_teleop, "_start_gui_qpos_init", return_value=transition) as start:
            self.assertFalse(control._handle_init_arm_move_to_qpos_event(now=1.0))
            control.set_startup_qpos_maintenance_enabled(True)
            self.assertTrue(control._handle_init_arm_move_to_qpos_event(now=2.0))

        start.assert_called_once()
        self.assertFalse(control._teleop_started)
        self.assertFalse(control._episode_recording_ui_active)

    def test_gui_exact_qpos_event_is_admitted_from_regular_hold(self) -> None:
        components = _components()
        components.init_arm_controller = SimpleNamespace(
            selected_side="left",
            active=False,
            qpos_active=False,
            status_payload=lambda: {"init_arm_available": True},
        )
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=_FakeUi(),
            args=_args(arm_control_mode="position"),
            components=components,
        )
        control._teleop_started = True
        control._hold_active = True
        transition = SimpleNamespace(ok=True, side="left", message="planning")

        with patch.object(
            real_teleop,
            "_start_gui_qpos_init",
            return_value=transition,
        ) as start:
            self.assertTrue(control._handle_init_arm_move_to_qpos_event(now=2.0))

        start.assert_called_once()
        self.assertTrue(control.hold_active())

    def test_startup_qpos_maintenance_dispatches_without_building_teleop_action(self) -> None:
        class Controller:
            qpos_active = True

            @staticmethod
            def qpos_seed_qpos_by_side():
                return {"left": np.zeros(6), "right": np.ones(6)}

            @staticmethod
            def consume_qpos_terminal_event():
                return None

        dispatched: list[TeleopCommand] = []
        filter_calls: list[tuple[object, float]] = []

        def command_filter(command, observation, now):
            filter_calls.append((observation, now))
            command.metadata["shared_filter_applied"] = True
            return command

        session = SimpleNamespace(
            command_filter=command_filter,
            evaluate_command_guard=lambda command, observation, now: None,
            dispatch_command=lambda command: dispatched.append(command),
        )
        components = SimpleNamespace(
            init_arm_controller=Controller(),
            session=session,
        )
        clutch = SimpleNamespace()

        self.assertTrue(
            real_teleop._advance_startup_gui_qpos_maintenance(
                args=_args(arm_control_mode="position"),
                components=components,
                clutch_input=clutch,
                now=3.0,
            )
        )

        self.assertEqual(filter_calls, [(None, 3.0)])
        self.assertEqual(len(dispatched), 1)
        self.assertEqual(set(dispatched[0].commands), {"left_arm", "right_arm"})
        self.assertTrue(dispatched[0].metadata["startup_qpos_maintenance"])
        self.assertFalse(dispatched[0].metadata["dataset_recording_eligible"])

    def test_startup_qpos_completion_releases_init_without_home_or_collection(self) -> None:
        class Controller:
            active = True
            qpos_operation_id = "startup-operation"

            def __init__(self) -> None:
                self.cancel_reasons: list[str] = []

            def cancel_active(self, *, now, reason):
                del now
                self.cancel_reasons.append(reason)
                self.active = False
                return SimpleNamespace(ok=True, message="released")

        controller = Controller()
        hold_reasons: list[str] = []
        published: list[str] = []
        components = SimpleNamespace(
            init_arm_controller=controller,
            left_command_backend=None,
            right_command_backend=None,
            session=SimpleNamespace(
                hold_hardware=lambda reason: hold_reasons.append(reason),
            ),
        )
        clutch = SimpleNamespace(
            _init_qpos_sequence_active=False,
            handle_init_arm_qpos_terminal=lambda terminal, now: None,
            _cancel_init_arm_qpos_sequence=lambda reason: None,
            _publish_init_arm_state=lambda **kwargs: published.append(kwargs["message"]),
        )

        real_teleop._finish_startup_gui_qpos_terminal(
            components=components,
            clutch_input=clutch,
            terminal={"status": "completed", "side": "left", "message": "arrived"},
            now=4.0,
        )

        self.assertEqual(
            controller.cancel_reasons,
            ["startup_qpos_completed_without_home_capture"],
        )
        self.assertEqual(hold_reasons, [])
        self.assertIn("未采集 tracker Home", published[-1])

    def test_gui_dual_exact_qpos_runs_left_then_right_without_home_then_holds(self) -> None:
        class InitController:
            def __init__(self) -> None:
                self.selected_side = "right"
                self.active_side = ""
                self.active = False
                self.qpos_active = False
                self.recapturing = False
                self.qpos_operation_id = ""
                self.selected_history: list[str] = []
                self.cancel_history: list[str] = []

            def select_side(self, side: str):
                if self.active:
                    return SimpleNamespace(ok=False, side=self.active_side, message="active")
                self.selected_side = side
                self.selected_history.append(side)
                return SimpleNamespace(ok=True, side=side, message="selected")

            def cancel_active(self, **_kwargs):
                side = self.active_side
                self.cancel_history.append(side)
                self.active = False
                self.active_side = ""
                self.qpos_active = False
                self.qpos_operation_id = ""
                return SimpleNamespace(ok=True, side=side, message="released")

            def status_payload(self):
                return {
                    "init_arm_available": True,
                    "init_arm_selected_side": self.selected_side,
                    "init_arm_active_side": self.active_side,
                    "init_arm_active": self.active,
                    "init_arm_recalibrating": self.recapturing,
                    "init_arm_status": "ready",
                }

        ui = _FakeUi()
        components = _components()
        controller = InitController()
        components.init_arm_controller = controller
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(arm_control_mode="position"),
            components=components,
        )
        control._teleop_started = True

        started_sides: list[str] = []

        def start_selected(_args, _components, *, now):
            del now
            side = controller.selected_side
            started_sides.append(side)
            controller.active = True
            controller.active_side = side
            controller.qpos_active = True
            controller.qpos_operation_id = "operation-" + side
            return SimpleNamespace(ok=True, side=side, message="planning")

        with patch.object(real_teleop, "_start_gui_qpos_init", side_effect=start_selected):
            self.assertTrue(control._handle_init_arm_move_both_to_qpos_event(now=1.0))
            self.assertEqual(started_sides, ["left"])

            control.handle_init_arm_qpos_terminal(
                {"status": "completed", "side": "left", "message": "arrived"},
                now=2.0,
            )
            self.assertEqual(controller.cancel_history, ["left"])
            self.assertEqual(started_sides, ["left", "right"])
            self.assertFalse(controller.recapturing)

            control.handle_init_arm_qpos_terminal(
                {"status": "completed", "side": "right", "message": "arrived"},
                now=3.0,
            )

        self.assertFalse(control._init_qpos_sequence_active)
        self.assertEqual(control._init_qpos_sequence_phase, "completed_hold_requested")
        self.assertEqual(controller.selected_side, "right")
        self.assertTrue(control.hold_active())
        self.assertTrue(control.consume_hold_request())
        self.assertIn("未自动重采 tracker Home", control._init_qpos_sequence_status)

    def test_gui_dual_exact_qpos_terminal_failure_aborts_before_right_and_holds(self) -> None:
        ui = _FakeUi()
        components = _components()
        components.init_arm_controller = SimpleNamespace(
            active=True,
            active_side="left",
            selected_side="left",
            qpos_active=True,
            status_payload=lambda: {"init_arm_available": True},
        )
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(arm_control_mode="position"),
            components=components,
        )
        control._teleop_started = True
        control._init_qpos_sequence_active = True
        control._init_qpos_sequence_side = "left"
        control._init_qpos_sequence_phase = "left_qpos"

        with patch.object(control, "_start_init_arm_sequence_side") as start_next:
            control.handle_init_arm_qpos_terminal(
                {
                    "status": "failed",
                    "side": "left",
                    "message": "measured qpos did not arrive",
                },
                now=3.0,
            )

        start_next.assert_not_called()
        self.assertTrue(control.hold_active())
        self.assertFalse(control._init_qpos_sequence_active)
        self.assertIn("measured qpos did not arrive", control._init_qpos_sequence_status)

    def test_gui_exact_qpos_event_is_allowed_while_episode_is_active(self) -> None:
        ui = _FakeUi(["init_arm_move_to_qpos"])
        components = _components()
        components.init_arm_controller = SimpleNamespace(
            selected_side="left",
            active=False,
            qpos_active=False,
            status_payload=lambda: {"init_arm_available": True},
        )
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(arm_control_mode="position"),
            components=components,
        )
        control._teleop_started = True
        control._episode_recording_ui_active = True

        transition = SimpleNamespace(ok=True, side="left", message="planning")
        with patch.object(
            real_teleop,
            "_start_gui_qpos_init",
            return_value=transition,
        ) as start:
            control.poll()

        start.assert_called_once()
        self.assertTrue(any("后台路径规划" in message for _tag, message in ui.messages))

    def test_gui_exact_qpos_snapshot_requires_fresh_successful_dual_feedback(self) -> None:
        now = time.time()

        class StateBackend:
            def __init__(self, qpos):
                self.qpos = np.asarray(qpos, dtype=float)

            def refresh_state_from_hardware(self):
                return {
                    "hardware_read_success": True,
                    "hardware_measurement_valid": True,
                    "hardware_arm_err": 0,
                    "hardware_sys_err": 0,
                    "hardware_measurement_timestamp": now,
                    "hardware_qpos": self.qpos.copy(),
                }

        class PreviewBackend:
            @staticmethod
            def normalize_local_qpos_observation(qpos, *, label):
                del label
                return np.asarray(qpos, dtype=float).copy(), {"status": "exact"}

        preview = PreviewBackend()
        components = SimpleNamespace(
            active_sides=("left", "right"),
            left_state_backend=StateBackend(np.zeros(6)),
            right_state_backend=StateBackend(np.ones(6)),
            left_preview_backend=preview,
            right_preview_backend=preview,
            arm_state_read_executor=None,
        )

        snapshot = real_teleop._fresh_gui_init_qpos_by_side(components)

        np.testing.assert_allclose(snapshot["left"], np.zeros(6))
        np.testing.assert_allclose(snapshot["right"], np.ones(6))

    def test_gui_exact_qpos_snapshot_bounds_tiny_feedback_overshoot_before_seed(self) -> None:
        now = time.time()

        class StateBackend:
            def __init__(self, qpos):
                self.qpos = np.asarray(qpos, dtype=float)

            def refresh_state_from_hardware(self):
                return {
                    "hardware_read_success": True,
                    "hardware_measurement_valid": True,
                    "hardware_arm_err": 0,
                    "hardware_sys_err": 0,
                    "hardware_measurement_timestamp": now,
                    "hardware_qpos": self.qpos.copy(),
                }

        right_qpos = np.zeros(6)
        right_qpos[2] = 2.355008
        left_preview = real_teleop.RM65ArmBackend(
            real_teleop.RM65ArmBackendConfig(
                arm_ip="192.0.2.1",
                side="left",
                connect_hardware=False,
                initial_qpos=np.zeros(6),
                singularity_protection_enabled=False,
            )
        )
        right_preview = real_teleop.RM65ArmBackend(
            real_teleop.RM65ArmBackendConfig(
                arm_ip="192.0.2.2",
                side="right",
                connect_hardware=False,
                initial_qpos=np.zeros(6),
                singularity_protection_enabled=False,
            )
        )
        components = SimpleNamespace(
            active_sides=("left", "right"),
            left_state_backend=StateBackend(np.zeros(6)),
            right_state_backend=StateBackend(right_qpos),
            left_preview_backend=left_preview,
            right_preview_backend=right_preview,
            arm_state_read_executor=None,
        )

        snapshot = real_teleop._fresh_gui_init_qpos_by_side(components)

        self.assertAlmostEqual(snapshot["right"][2], 2.355, places=9)
        right_preview._validate_final_joint_target(
            snapshot["right"],
            label="GUI Init qpos seed",
        )
        with self.assertRaisesRegex(ValueError, "absolute limits"):
            right_preview._validate_final_joint_target(
                right_qpos,
                label="raw hardware feedback",
            )

    def test_gui_exact_qpos_snapshot_requires_command_safe_preview_normalizer(self) -> None:
        now = time.time()
        state = {
            "hardware_read_success": True,
            "hardware_measurement_valid": True,
            "hardware_arm_err": 0,
            "hardware_sys_err": 0,
            "hardware_measurement_timestamp": now,
            "hardware_qpos": np.zeros(6),
        }
        backend = SimpleNamespace(
            refresh_state_from_hardware=lambda: dict(state)
        )
        components = SimpleNamespace(
            active_sides=("left", "right"),
            left_state_backend=backend,
            right_state_backend=backend,
            left_preview_backend=None,
            right_preview_backend=None,
            arm_state_read_executor=None,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "command-safe observation normalization",
        ):
            real_teleop._fresh_gui_init_qpos_by_side(components)

    def test_gui_exact_qpos_snapshot_rejects_stale_feedback(self) -> None:
        stale_state = {
            "hardware_read_success": True,
            "hardware_measurement_valid": True,
            "hardware_arm_err": 0,
            "hardware_sys_err": 0,
            "hardware_measurement_timestamp": time.time() - 2.0,
            "hardware_qpos": np.zeros(6),
        }
        backend = SimpleNamespace(
            refresh_state_from_hardware=lambda: dict(stale_state)
        )
        components = SimpleNamespace(
            active_sides=("left", "right"),
            left_state_backend=backend,
            right_state_backend=backend,
            arm_state_read_executor=None,
        )

        with self.assertRaisesRegex(RuntimeError, "feedback is stale"):
            real_teleop._fresh_gui_init_qpos_by_side(components, max_age_sec=0.5)

    def test_impedance_exact_qpos_admission_arms_both_backends(self) -> None:
        class FakeController:
            selected_side = "left"
            qpos_operation_id = "operation-impedance"

            def __init__(self) -> None:
                target = SimpleNamespace(arm_qpos=np.ones(6))
                self.targets_by_side = {"left": target, "right": target}

            def start_qpos_planning(self, **_kwargs):
                return real_teleop.InitArmTransition(
                    ok=True,
                    message="planning",
                    side="left",
                    mode="qpos_planning",
                )

        class FakeImpedanceBackend:
            def __init__(self) -> None:
                self.begin_calls: list[str] = []

            def begin_exact_qpos_transit(self, operation_id: str) -> None:
                self.begin_calls.append(operation_id)

        left = FakeImpedanceBackend()
        right = FakeImpedanceBackend()
        components = SimpleNamespace(
            init_arm_controller=FakeController(),
            fix_arm_controller=None,
            collision_guard_requested=True,
            dual_arm_retargeter=object(),
            left_command_backend=left,
            right_command_backend=right,
        )
        qpos = {"left": np.zeros(6), "right": np.zeros(6)}

        with patch.object(
            real_teleop,
            "RM65ImpedanceArmBackend",
            FakeImpedanceBackend,
        ), patch.object(
            real_teleop,
            "_fresh_gui_init_qpos_by_side",
            return_value=qpos,
        ):
            transition = real_teleop._start_gui_qpos_init(
                _args(arm_control_mode="impedance"),
                components,
                now=1.0,
            )

        self.assertTrue(transition.ok)
        self.assertEqual(left.begin_calls, ["operation-impedance"])
        self.assertEqual(right.begin_calls, ["operation-impedance"])

    def test_impedance_exact_qpos_rejects_before_force_control_initialization(self) -> None:
        class FakeController:
            selected_side = "left"

            def __init__(self) -> None:
                target = SimpleNamespace(arm_qpos=np.ones(6))
                self.targets_by_side = {"left": target, "right": target}
                self.plan_calls = 0

            def start_qpos_planning(self, **_kwargs):
                self.plan_calls += 1
                return real_teleop.InitArmTransition(ok=True, message="planning", side="left")

        class FakeImpedanceBackend:
            config = SimpleNamespace(connect_hardware=True)

            def get_state(self):
                return {
                    "impedance_enabled": False,
                    "impedance_loop_state": "not_started",
                    "last_impedance_record": None,
                }

        controller = FakeController()
        components = SimpleNamespace(
            active_sides=("left", "right"),
            init_arm_controller=controller,
            fix_arm_controller=None,
            collision_guard_requested=True,
            dual_arm_retargeter=object(),
            left_command_backend=FakeImpedanceBackend(),
            right_command_backend=FakeImpedanceBackend(),
        )

        with patch.object(
            real_teleop,
            "RM65ImpedanceArmBackend",
            FakeImpedanceBackend,
        ):
            transition = real_teleop._start_gui_qpos_init(
                _args(arm_control_mode="impedance"),
                components,
                now=1.0,
            )

        self.assertFalse(transition.ok)
        self.assertIn("力控尚未初始化", transition.message)
        self.assertIn("开始遥操", transition.message)
        self.assertEqual(controller.plan_calls, 0)

    def test_exact_qpos_plan_scales_selected_side_from_shared_slow_baseline(self) -> None:
        class FakeController:
            selected_side = "right"

            def __init__(self) -> None:
                target = SimpleNamespace(arm_qpos=np.ones(6))
                self.targets_by_side = {"left": target, "right": target}
                self.plan_builder = None

            def start_qpos_planning(self, **kwargs):
                self.plan_builder = kwargs["plan_builder"]
                return real_teleop.InitArmTransition(
                    ok=True,
                    message="planning",
                    side=self.selected_side,
                )

        controller = FakeController()
        components = SimpleNamespace(
            active_sides=("left", "right"),
            init_arm_controller=controller,
            fix_arm_controller=None,
            collision_guard_requested=True,
            dual_arm_retargeter=object(),
        )
        qpos = {"left": np.zeros(6), "right": np.zeros(6)}
        args = _args(
            arm_control_mode="position",
            init_arm_left_qpos_speed_scale=2.0,
            init_arm_right_qpos_speed_scale=6.0,
        )
        fake_profile = SimpleNamespace(
            max_velocity_deg_s=(6.0, 6.0, 7.0, 8.0, 10.0, 10.0)
        )

        with patch.object(
            real_teleop,
            "_fresh_gui_init_qpos_by_side",
            return_value=qpos,
        ), patch.object(
            real_teleop,
            "load_joint_velocity_limit_profiles",
            return_value={"rm65_init_slow": fake_profile},
        ):
            transition = real_teleop._start_gui_qpos_init(
                args,
                components,
                now=1.0,
            )

        self.assertTrue(transition.ok)
        self.assertIsNotNone(controller.plan_builder)
        with patch(
            "teleop_runtime.replay.arm_init.qpos_execute.build_rm65_qpos_init_plan",
            return_value=object(),
        ) as build_plan:
            controller.plan_builder()

        expected = np.deg2rad(np.asarray(fake_profile.max_velocity_deg_s) * 6.0)
        np.testing.assert_allclose(
            build_plan.call_args.kwargs["qpos_init_max_angvel_rad_s"],
            expected,
        )
        self.assertEqual(
            build_plan.call_args.kwargs["joint_velocity_profile"],
            "rm65_init_slow",
        )

    def test_impedance_exact_qpos_admission_rolls_back_first_arm(self) -> None:
        class FakeController:
            selected_side = "left"
            qpos_operation_id = "operation-rollback"

            def __init__(self) -> None:
                target = SimpleNamespace(arm_qpos=np.ones(6))
                self.targets_by_side = {"left": target, "right": target}
                self.cancel_calls: list[dict[str, object]] = []

            def start_qpos_planning(self, **_kwargs):
                return real_teleop.InitArmTransition(ok=True, message="planning", side="left")

            def cancel_active(self, **kwargs):
                self.cancel_calls.append(dict(kwargs))

        class FakeImpedanceBackend:
            def __init__(self, *, fail: bool = False) -> None:
                self.fail = fail
                self.begin_calls: list[str] = []
                self.end_calls: list[str] = []

            def begin_exact_qpos_transit(self, operation_id: str) -> None:
                self.begin_calls.append(operation_id)
                if self.fail:
                    raise RuntimeError("force preflight rejected")

            def end_exact_qpos_transit(self, operation_id: str) -> None:
                self.end_calls.append(operation_id)

        controller = FakeController()
        left = FakeImpedanceBackend()
        right = FakeImpedanceBackend(fail=True)
        components = SimpleNamespace(
            init_arm_controller=controller,
            fix_arm_controller=None,
            collision_guard_requested=True,
            dual_arm_retargeter=object(),
            left_command_backend=left,
            right_command_backend=right,
        )
        qpos = {"left": np.zeros(6), "right": np.zeros(6)}

        with patch.object(
            real_teleop,
            "RM65ImpedanceArmBackend",
            FakeImpedanceBackend,
        ), patch.object(
            real_teleop,
            "_fresh_gui_init_qpos_by_side",
            return_value=qpos,
        ):
            with self.assertRaisesRegex(RuntimeError, "force preflight rejected"):
                real_teleop._start_gui_qpos_init(
                    _args(arm_control_mode="impedance"),
                    components,
                    now=2.0,
                )

        self.assertEqual(left.end_calls, ["operation-rollback"])
        self.assertEqual(len(controller.cancel_calls), 1)

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
            control._teleop_started = True

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

    def test_gui_episode_start_contract_failure_rolls_back_and_does_not_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_root = Path(tmp_dir) / "datasets"
            args = _args(dataset_root=str(base_root))
            args.ignore_camera = []
            args.ignore_tactile = []
            args.configured_camera_keys = ("left_wrist", "right_wrist")
            real_teleop._apply_formal_dataset_layout(args)
            original = {
                "dataset_root": args.dataset_root,
                "ignore_camera": list(args.ignore_camera),
                "ignore_tactile": list(args.ignore_tactile),
            }
            ui = _FakeUi(
                [
                    "episode_start:"
                    + json.dumps(
                        {
                            "dataset_base_root": str(base_root),
                            "task_key": "soft_pick",
                            "task_english_name": "Soft Pick",
                            "ignored_camera_keys": ["right_wrist"],
                            "ignored_tactile_keys": ["left"],
                        }
                    )
                ]
            )
            control = real_teleop.RealTeleopControlInput(
                terminal=_FakeTerminal(),
                ui=ui,
                args=args,
                components=_components(),
            )
            control._teleop_started = True

            with patch.object(
                real_teleop,
                "bind_capture_modality_contract",
                side_effect=RuntimeError("contract mismatch"),
            ):
                control.poll()

            self.assertFalse(control.consume_episode_start_request())
            self.assertEqual(args.dataset_root, original["dataset_root"])
            self.assertEqual(args.ignore_camera, original["ignore_camera"])
            self.assertEqual(args.ignore_tactile, original["ignore_tactile"])
            self.assertTrue(
                any(
                    "已恢复原配置" in message
                    for _title, message in ui.messages
                )
            )

    def test_gui_episode_start_rejects_unknown_camera_ignore_key(self) -> None:
        args = _args()
        args.configured_camera_keys = ("left_wrist", "right_wrist")
        ui = _FakeUi(
            [
                "episode_start:"
                + json.dumps(
                    {
                        "ignored_camera_keys": ["front"],
                        "ignored_tactile_keys": [],
                    }
                )
            ]
        )
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=args,
            components=_components(),
        )
        control._teleop_started = True

        control.poll()

        self.assertFalse(control.consume_episode_start_request())
        self.assertEqual(getattr(args, "ignore_camera", []), [])

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
            control._teleop_started = True

            with patch.object(real_teleop, "datetime") as fake_datetime:
                fake_datetime.now.return_value.strftime.return_value = "2026-07-02"
                control.poll()

        self.assertFalse(control.consume_episode_start_request())
        self.assertEqual(args.collection_task_id, "soft_pick")
        self.assertEqual(args.collection_task_name, "Soft Pick")
        self.assertEqual(args.gripper_task, "soft_grasp")
        self.assertEqual(Path(args.dataset_root), base_root / "2026-07-02_Soft_Pick")

    def test_gui_can_rename_existing_task_into_a_new_dataset_folder(self) -> None:
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
                            "task_english_name": "Soft Pick Recovery",
                            "task_chinese_name": "",
                        },
                        ensure_ascii=False,
                    )
                ]
            )
            args = _args(dataset_root=str(base_root), task_config=str(task_config))
            real_teleop._apply_formal_dataset_layout(args)
            original_root = Path(args.dataset_root)
            control = real_teleop.RealTeleopControlInput(
                terminal=_FakeTerminal(),
                ui=ui,
                args=args,
                components=_components(),
            )
            control._teleop_started = True

            with patch.object(real_teleop, "datetime") as fake_datetime:
                fake_datetime.now.return_value.strftime.return_value = "2026-07-02"
                control.poll()

        self.assertFalse(control.consume_episode_start_request())
        self.assertEqual(args.collection_task_id, "soft_pick")
        self.assertEqual(args.collection_task_name, "Soft Pick Recovery")
        self.assertEqual(args.collection_task_name_zh, "轻柔抓取")
        self.assertNotEqual(Path(args.dataset_root), original_root)
        self.assertEqual(
            Path(args.dataset_root),
            base_root / "2026-07-02_Soft_Pick_Recovery",
        )

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
            control._teleop_started = True

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
        self.assertIn("showRefreshWarning(error, refreshPhase);", html)
        self.assertIn('render: "页面渲染失败"', html)
        self.assertIn("window.__calibrationWebUiModernRefreshOk = false;", html)
        self.assertIn(
            "if (window.__calibrationWebUiModernRefreshOk) return;\n"
            "        var eventName = this.getAttribute(\"data-event\");",
            html,
        )
        self.assertIn(
            "button.addEventListener(\"click\", async () => {\n"
            "    if (!window.__calibrationWebUiModernRefreshOk) return;",
            html,
        )
        self.assertNotIn("showShutdown(\"无法继续连接标定服务", html)

    def test_gui_modern_ik_log_button_has_pending_feedback(self) -> None:
        html = calibration_web_ui._HTML_PAGE

        self.assertIn("ikDebugLogTogglePending", html)
        self.assertIn("updateIkDebugLogButton(state)", html)
        self.assertIn("IK 记录切换中...", html)
        self.assertIn('ikDebugLogButtonEl.disabled = !activeControls.has("d") || ikDebugLogTogglePending', html)

    def test_umi_keeps_idle_dataset_directory_picker_enabled(self) -> None:
        html = calibration_web_ui._HTML_PAGE

        self.assertIn(
            "|| datasetConfigurationSubmissionPending",
            html,
        )
        self.assertIn(
            'const datasetConfigurationAvailable = activeControls.has("task_select");',
            html,
        )
        self.assertIn("datasetBaseRootInputEl.disabled = pathLocked;", html)
        self.assertIn("datasetChooseRootButtonEl.disabled = pathLocked;", html)
        self.assertIn("const taskLocked = Boolean(pathLocked || runtimeConfigured);", html)
        self.assertIn(
            "roundTripToggleButtonEl.disabled = taskLocked || !roundTripActive",
            html,
        )
        self.assertNotIn(
            "roundTripToggleButtonEl.disabled = locked || !roundTripActive",
            html,
        )
        self.assertIn("|| (!runtimeConfigured && !englishName)", html)
        self.assertIn('? "应用文件夹 / 切换数据集"', html)
        self.assertIn("let datasetBaseRootOverride = null;", html)
        self.assertIn("datasetBaseRootOverride = String(result.path);", html)

    def test_dataset_configuration_has_explicit_draft_apply_flow(self) -> None:
        html = calibration_web_ui._HTML_PAGE

        self.assertIn(
            'id="datasetConfigurationApplyButton" class="secondary" '
            'data-event="task_select"',
            html,
        )
        self.assertIn("let taskEnglishNameOverride = null;", html)
        self.assertIn("let taskChineseNameOverride = null;", html)
        self.assertIn("markDatasetConfigurationDirty", html)
        self.assertIn("let datasetConfigurationSubmittedRoot = null;", html)
        self.assertIn("const submittedRootMatches = (", html)
        self.assertIn(
            'eventName === "episode_start" || eventName === "task_select"',
            html,
        )
        self.assertNotIn(
            'await sendEvent("task_select", collectEpisodeStartPayload());',
            html,
        )

    def test_episode_delete_controls_are_greyed_while_episode_is_busy(self) -> None:
        html = calibration_web_ui._HTML_PAGE

        self.assertIn(
            'id="episodeDeleteIndexInput" type="number" min="0" step="1" '
            'inputmode="numeric" placeholder="例如 3" disabled',
            html,
        )
        self.assertIn(
            'id="episodeDeleteButton" class="secondary" '
            'data-event="episode_delete" disabled',
            html,
        )
        self.assertIn('activeControls.has("episode_delete")', html)
        self.assertIn("!state.episode_recording", html)
        self.assertIn("episodeStatus !== \"delete_queued\"", html)
        self.assertIn(
            "episodeDeleteIndexInputEl.disabled = !episodeDeleteAllowed;",
            html,
        )

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
            _args(visualize_tactile=True, skip_renderer=True),
            ("right",),
        )

        self.assertIsInstance(vis, DMTacDistributedForceVisualizer)
        self.assertEqual(vis.config.fixed_ref, 1.0)
        vis.close()

    def test_dmtac_tactile_visualizer_builder_uses_configured_common_scale(self) -> None:
        vis = real_teleop._build_tactile_visualizer(
            _args(
                visualize_tactile=True,
                skip_renderer=True,
                dmtac_tactile_vis_fixed_ref=2.5,
            ),
            ("left", "right"),
        )

        self.assertIsInstance(vis, DMTacDistributedForceVisualizer)
        self.assertEqual(vis.config.fixed_ref, 2.5)
        self.assertEqual(vis._robust_ref(np.asarray([0.001, 50.0]), vis.config.fixed_ref), 2.5)
        vis.close()

    def test_dmtac_tactile_visualization_requires_calibrated_force_backend(self) -> None:
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
                tactile_calibration_policy="required",
            )
        )
        components.right_hand_backend = backend

        self.assertTrue(real_teleop._tactile_visualization_available(_args(), components))

    def test_dmtac_visualization_filters_non_force_fields_without_copying(self) -> None:
        raw = np.zeros((480, 640), dtype=np.uint8)
        force = np.ones((8, 10, 3), dtype=np.float32)

        class _Backend:
            @staticmethod
            def get_state(**kwargs: object) -> dict[str, object]:
                self.assertFalse(kwargs.get("copy_tactile_arrays", True))
                return {
                    "tactile_dict": {"sensor:raw": raw, "sensor:distributed_force": force},
                    "tactile_field_by_key": {
                        "sensor:raw": "raw",
                        "sensor:distributed_force": "distributed_force",
                    },
                    "tactile_fields": ("raw", "distributed_force"),
                    "tactile_timestamp": 10.0,
                    "tactile_read_hz": 15.0,
                    "tactile_process_read": True,
                    "tactile_worker_alive": True,
                    "telemetry_error": None,
                }

        snapshot, health = dmtac_visualization_snapshot_from_backends(
            ("right",), lambda _side: _Backend(), now=10.01
        )

        self.assertEqual(set(snapshot["right"] or {}), {"sensor:distributed_force"})
        self.assertEqual(health["right"]["status"], "ok")

    def test_dmtac_gui_solving_requires_and_applies_fixed_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_tactile_calibration(
                tmp_dir,
                "SENSOR_TEST",
                np.zeros((480, 640), dtype=np.uint8),
            )
            backend = DMTacGripperBackend(
                DMTacGripperBackendConfig(
                    side="right",
                    connect_hardware=False,
                    connect_tactile=True,
                    tactile_ids=("SENSOR_TEST",),
                    tactile_fields=("raw",),
                    tactile_calibration_dir=tmp_dir,
                )
            )
            with patch.object(backend, "_stop_tactile_worker_for_reconfigure") as stop:
                with patch.object(backend, "_start_tactile_worker") as start:
                    self.assertTrue(backend.enable_tactile_solving())

        stop.assert_called_once()
        start.assert_called_once()
        self.assertEqual(backend.config.tactile_fields, ("raw", "distributed_force"))
        self.assertEqual(backend.config.tactile_compute_profile, "dist-only")
        self.assertEqual(backend.config.tactile_calibration_policy, "required")

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
            self.assertNotIn("capture_modality_contract", metadata)

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

    def test_camera_retry_self_check_replaces_snapshot_without_nested_input_poll(self) -> None:
        class _FreshCameraSource:
            def __init__(self) -> None:
                self.frame_count = 0
                self.config = SimpleNamespace(
                    serial="reconnected-camera",
                    width=640,
                    height=480,
                    fps=15,
                    color_space="bgr",
                    enable_depth=False,
                    depth_width=640,
                    depth_height=480,
                    depth_fps=15,
                )
                self.name = "reconnected_camera"

            def device_info(self) -> dict[str, object]:
                self.frame_count += 1
                return {
                    "camera_key": "front",
                    "serial": "reconnected-camera",
                    "frame_count": self.frame_count,
                    "last_frame_age_sec": 0.01,
                    "stale_after_sec": 1.0,
                }

        ui = _FakeUi()
        clutch_input = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=_args(),
            components=_components(),
        )
        with patch.object(
            clutch_input,
            "poll",
            side_effect=AssertionError("retry callback must not recursively poll input"),
        ):
            payload = real_teleop._run_camera_startup_self_check(
                camera_sources={"front": _FreshCameraSource()},
                clutch_input=clutch_input,
                duration_sec=1.0,
                interval_sec=0.01,
                poll_input=False,
            )

        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["completion_reason"], "fresh_frames")
        published_checks = [
            item["camera_self_check"]
            for item in ui.published
            if "camera_self_check" in item
        ]
        self.assertEqual(published_checks[-1]["status"], "passed")

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

    def test_collection_frequency_warning_ignores_hidden_rounding_jitter(self) -> None:
        recorder = real_teleop.RealTeleopDatasetRecorder.__new__(
            real_teleop.RealTeleopDatasetRecorder
        )
        current_time = time.monotonic()
        recorder.active_episode_index = 0
        recorder.sample_builder = SimpleNamespace(
            frame_index=46,
            last_camera_rejection=None,
            last_camera_frame_age_sec_by_key={},
        )
        recorder.frequency_warning_hz = 15.0
        recorder.frequency_window_sec = 3.1
        recorder.frequency_min_elapsed_sec = 2.0
        recorder._episode_start_monotonic = current_time - 3.1
        period_sec = 1.0 / 14.96
        recorder._record_timestamps = deque(
            current_time - period_sec * offset
            for offset in reversed(range(46))
        )

        status = recorder.collection_frequency_status()

        self.assertAlmostEqual(status["sample_hz"], 14.96, places=2)
        self.assertFalse(status["warning"]["active"])
        self.assertEqual(
            status["warning"]["comparison_threshold_hz"],
            14.95,
        )

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
        recorder._sample_modality_reason = "camera_frame_stale:front:0.102755"
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
        self.assertFalse(
            recorder.pause_episode(
                reason="fix_hold:both_sides_fixed_like",
                now=time.time(),
                step_index=43,
                last_state="RUNNING",
            )
        )
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
        self.assertEqual(recorder.collection_modality_reason, "")
        self.assertEqual(recorder.continuity_segment_index, 1)
        self.assertEqual(
            recorder.pause_summary()["collection_continuity_segment_count"],
            2,
        )
        self.assertEqual(len(recorder.pause_summary()["recording_pause_segments"]), 1)
        self.assertEqual(
            recorder.pause_summary()["recording_pause_segments"][0]["start_frame_index"],
            4,
        )
        self.assertEqual(
            recorder.pause_summary()["recording_pause_segments"][0]["reasons"],
            ["hold_requested", "fix_hold:both_sides_fixed_like"],
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
        self.assertNotIn("toggle_camera_vis", active_controls)
        self.assertNotIn("toggle_tactile_vis", active_controls)
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
        self.assertIn("task_select", active_controls)

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

    def test_gui_foot_pad_test_completion_is_persistent_for_connection(self) -> None:
        html_and_fallback = (
            calibration_web_ui._HTML_PAGE + calibration_web_ui._FALLBACK_SCRIPT
        )

        self.assertIn("foot_pad_test_completed", html_and_fallback)
        self.assertIn("本次连接的踏板测试已完成", html_and_fallback)
        self.assertIn("重新测试踏板", html_and_fallback)
        self.assertIn(".foot-pad-test-banner.completed", calibration_web_ui._HTML_PAGE)

    def test_gui_abc_shortcuts_follow_backend_active_controls(self) -> None:
        html_and_fallback = (
            calibration_web_ui._HTML_PAGE + calibration_web_ui._FALLBACK_SCRIPT
        )

        self.assertIn('activeControls.has("episode_start")', html_and_fallback)
        self.assertIn('activeControls.has("two_pedal_b")', html_and_fallback)
        self.assertIn('activeControls.has("two_pedal_c")', html_and_fallback)
        self.assertIn('legacyActiveControls["two_pedal_b"]', html_and_fallback)
        self.assertIn('fallbackActiveControls["two_pedal_c"]', html_and_fallback)
        self.assertIn("5 踏板 3/4/5：推进/失败/撤销；Init 使用 G/GUI", html_and_fallback)
        self.assertIn("B/C 派发与双踏板专用语义一致", html_and_fallback)
        self.assertIn('code === "keya"', html_and_fallback)
        self.assertIn('code === "keyc"', html_and_fallback)

    def test_gui_c_shortcut_only_sends_fix_arm_not_runtime_home_or_quit(self) -> None:
        html_and_fallback = calibration_web_ui._HTML_PAGE + calibration_web_ui._FALLBACK_SCRIPT

        self.assertIn('id="shortcutCLabel"', calibration_web_ui._HTML_PAGE)
        self.assertIn('id="shortcutGLabel"', calibration_web_ui._HTML_PAGE)
        self.assertIn('data-event="fix_arm_toggle"', calibration_web_ui._HTML_PAGE)
        self.assertIn('data-event="init_arm_toggle"', calibration_web_ui._HTML_PAGE)
        self.assertIn(
            'data-event="init_arm_move_to_qpos"',
            calibration_web_ui._HTML_PAGE,
        )
        self.assertIn(
            'data-event="init_arm_move_both_to_qpos"',
            calibration_web_ui._HTML_PAGE,
        )
        self.assertIn("机械臂回到初始化位姿", calibration_web_ui._HTML_PAGE)
        self.assertIn("精确归位所选臂：左臂", calibration_web_ui._HTML_PAGE)
        self.assertIn("双臂顺序精确归位：左臂 → 右臂", calibration_web_ui._HTML_PAGE)
        self.assertIn("过程中不会修改 tracker Home", html_and_fallback)
        self.assertIn("启动前执行时到位后停留在启动前 Hold", html_and_fallback)
        self.assertLess(
            calibration_web_ui._HTML_PAGE.index('id="initArmSection"'),
            calibration_web_ui._HTML_PAGE.index('id="footPadTestControls"'),
        )
        self.assertGreater(
            calibration_web_ui._HTML_PAGE.index('id="datasetModalityIgnoreOptions"'),
            calibration_web_ui._HTML_PAGE.index('id="poseSnapshotSection"'),
        )
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

    def test_impedance_force_collision_uses_runtime_collision_callback(self) -> None:
        class FakeSession:
            def __init__(self) -> None:
                self.state_machine = SimpleNamespace(state=real_teleop.TeleopRunState.RUNNING)
                self.faults: list[str] = []

            def fault(self, reason: str) -> None:
                self.faults.append(reason)

        session = FakeSession()
        components = SimpleNamespace(session=session)
        issue = "left_arm pose impedance fault before command dispatch: RuntimeError: left: force 55.0000N > hard_stop_force_n"
        collisions: list[str] = []

        with patch.object(real_teleop, "_impedance_force_control_issue", return_value=issue), patch.object(
            real_teleop,
            "_log",
        ) as log, patch.object(real_teleop, "_hold_motion_after_impedance_guard_issue") as hold:
            ok = real_teleop._guard_impedance_force_control(
                SimpleNamespace(),
                components,
                context="main_loop",
                now=1.0,
                collision_handler=collisions.append,
            )

        self.assertFalse(ok)
        self.assertEqual(collisions, [f"main_loop: {issue}"])
        self.assertEqual(session.faults, [])
        hold.assert_not_called()
        log.assert_not_called()


if __name__ == "__main__":
    unittest.main()
