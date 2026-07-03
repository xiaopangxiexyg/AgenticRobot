from __future__ import annotations

import json
from collections import deque
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
from teleop_runtime.schemas import ObservationChannel
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


class _FakeCalibration:
    def __init__(self) -> None:
        self.tracker_offsets = {"right_wrist": object()}
        self.metadata: dict[str, object] = {}

    def world_to_desk_pose(self) -> object:
        return object()

    def to_dict(self) -> dict[str, object]:
        return {"format": "fake_calibration"}


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
        self.assertNotIn("movement_multiple_down", active_controls)
        self.assertNotIn("movement_multiple_up", active_controls)
        self.assertNotIn("movement_multiple_reset", active_controls)
        self.assertNotIn("movement_multiple_set", active_controls)
        self.assertEqual(running_publish["movement_multiple"], 1.0)
        self.assertNotIn("gripper_toggle", active_controls)
        self.assertNotIn("gripper_open", active_controls)
        self.assertNotIn("gripper_half", active_controls)
        self.assertNotIn("gripper_close", active_controls)

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

    def test_web_ui_accepts_episode_start_payload_when_control_active(self) -> None:
        ui = calibration_web_ui.CalibrationWebUi(open_browser=False)
        ui.publish(active_controls=["episode_start"])

        event = 'episode_start:{"task_key":"soft_grasp"}'
        self.assertTrue(ui.push_event(event))
        self.assertEqual(ui.consume_events(), [event])

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
                output_dim=1,
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
        self.assertIn("q6=ok", status[0])
        self.assertIn("dim=2", status[0])
        self.assertIn("src=0.620", status[0])
        self.assertIn("target=0.620", status[0])
        self.assertIn("send=sent/1", status[0])

    def test_dmtac_tactile_visualizer_builder_uses_force_window(self) -> None:
        vis = real_teleop._build_tactile_visualizer(
            _args(visualize_tactile=True),
            ("right",),
        )

        self.assertIsInstance(vis, DMTacDistributedForceVisualizer)
        vis.close()

    def test_dmtac_tactile_visualization_requires_tactile_backend(self) -> None:
        backend = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="right",
                connect_hardware=False,
                connect_tactile=True,
            )
        )
        components = _components()
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

    def test_episode_metadata_records_resolved_episode_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = _args(session_record_root=tmp_dir)
            components = _components()

            with patch.object(real_teleop.time, "strftime", return_value="20260630_120000"):
                metadata = real_teleop._build_episode_metadata(
                    args,
                    components,
                    start_reason="manual_key_start",
                )

            expected = Path(tmp_dir) / "20260630_120000_vive_rm65_real_teleop" / "trajectories"
            self.assertEqual(metadata["episode_root"], str(expected))
            self.assertEqual(metadata["start_reason"], "manual_key_start")

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

    def test_preview_defaults_to_all_configured_cameras(self) -> None:
        cameras = {
            "front": object(),
            "left_wrist": object(),
            "right_wrist": object(),
        }

        selected = real_teleop._select_preview_camera_sources(cameras, "")

        self.assertEqual(set(selected), set(cameras))

    def test_collection_frequency_warning_trips_below_15hz(self) -> None:
        recorder = real_teleop.RealTeleopDatasetRecorder.__new__(
            real_teleop.RealTeleopDatasetRecorder
        )
        current_time = time.monotonic()
        recorder.active_episode_index = 0
        recorder.sample_builder = SimpleNamespace(frame_index=4)
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


if __name__ == "__main__":
    unittest.main()
