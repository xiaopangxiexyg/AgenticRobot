from __future__ import annotations

import json
import argparse
import os
import queue
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from teleop_runtime import (
    DMTacGripperBackendConfig,
    build_annotated_dual_rm65_calibration,
    build_bimanual_session,
)
from teleop_runtime.action import robot_action_from_teleop_command
from teleop_runtime.backends.grippers import DMTacGripperBackend
from teleop_runtime.deploy import run_policy as deploy_run_policy
from teleop_runtime.debug import run_vive_rm65_real_teleop as real_teleop
from teleop_runtime.collection import run_collect
from teleop_runtime.collection.sample_assembler import CollectorSample
from teleop_runtime.collection.schema_v21 import collector_sample_to_row
from teleop_runtime.retargeters import ScalarToGripperRetargeter, ScalarToGripperRetargeterConfig
from teleop_runtime.schemas import EffectorCommand, ObservationChannel, TeleopObservation


class DMTacGripperBimanualConfigTest(unittest.TestCase):
    def test_dmtac_serial_dev_id_resolves_to_sdk_index_by_default(self) -> None:
        from hardware.DMTAC_gripper.portable_dm_tac import core as dmtac_core

        previous_cache = dmtac_core._SDK_DISCOVERY_CACHE
        dmtac_core._SDK_DISCOVERY_CACHE = (
            time.monotonic(),
            [{"device_index": 4, "serial": "M26190173"}],
        )
        try:
            with patch.dict(os.environ, {"DMTAC_RESOLVE_SERIAL_TO_INDEX": "1"}):
                self.assertEqual(dmtac_core._resolve_device_id_for_sensor("M26190173"), 4)
                self.assertEqual(dmtac_core._resolve_device_id_for_sensor(4), 4)
            with patch.dict(os.environ, {"DMTAC_RESOLVE_SERIAL_TO_INDEX": "0"}):
                self.assertEqual(
                    dmtac_core._resolve_device_id_for_sensor("M26190173"),
                    "M26190173",
                )
        finally:
            dmtac_core._SDK_DISCOVERY_CACHE = previous_cache

    def test_bimanual_builder_accepts_two_dmtac_grippers(self) -> None:
        left_config = DMTacGripperBackendConfig(
            side="left",
            gripper_port="/dev/null",
            connect_hardware=False,
            connect_tactile=False,
            tactile_ids=("L_LEFT_INNER", "L_LEFT_OUTER"),
        )
        right_config = DMTacGripperBackendConfig(
            side="right",
            gripper_port="/dev/null",
            connect_hardware=False,
            connect_tactile=False,
            tactile_ids=("L_RIGHT_INNER", "L_RIGHT_OUTER"),
        )

        session = build_bimanual_session(
            calibration=build_annotated_dual_rm65_calibration(),
            left_hand_serial_port="/dev/null",
            right_hand_serial_port="/dev/null",
            left_hand_backend_config=left_config,
            right_hand_backend_config=right_config,
            enable_dual_arm_collision_avoidance=False,
        )

        self.assertIsInstance(session.rig.backends["left_hand"], DMTacGripperBackend)
        self.assertIsInstance(session.rig.backends["right_hand"], DMTacGripperBackend)
        self.assertEqual(
            session.rig.backends["left_hand"].config.tactile_ids,
            ("L_LEFT_INNER", "L_LEFT_OUTER"),
        )
        self.assertEqual(
            session.rig.backends["right_hand"].config.tactile_ids,
            ("L_RIGHT_INNER", "L_RIGHT_OUTER"),
        )
        self.assertEqual(
            sum(isinstance(retargeter, ScalarToGripperRetargeter) for retargeter in session.retargeters),
            2,
        )

    def test_scalar_gripper_retargeter_marks_only_gripper_slot_active(self) -> None:
        retargeter = ScalarToGripperRetargeter(
            config=next(
                retargeter.config
                for retargeter in build_bimanual_session(
                    calibration=build_annotated_dual_rm65_calibration(),
                    left_hand_serial_port="/dev/null",
                    left_hand_backend_config=DMTacGripperBackendConfig(
                        side="left",
                        gripper_port="/dev/null",
                        connect_hardware=False,
                        connect_tactile=False,
                    ),
                    enable_dual_arm_collision_avoidance=False,
                ).retargeters
                if isinstance(retargeter, ScalarToGripperRetargeter)
            )
        )
        observation = TeleopObservation(
            channels={
                "left_hand_q6": ObservationChannel(
                    channel_name="left_hand_q6",
                    capability="joint_vector",
                    side="left",
                    value=np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0], dtype=np.float32),
                    frame_name=None,
                    timestamp=1.0,
                    source_name="unit",
                )
            },
            timestamp=1.0,
        )

        command = retargeter.retarget(observation)
        action = robot_action_from_teleop_command(command)

        np.testing.assert_allclose(
            command.commands["left_hand"].value,
            np.array([0.1875, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        spec = action.metadata["hand_action_specs_by_side"]["left"]
        self.assertEqual(spec["hand_model"], "dmtac_gripper")
        self.assertEqual(spec["active_mask"], [True, False, False, False, False, False])
        self.assertEqual(spec["slot_map"], {"gripper_close": 0})

    def test_scalar_gripper_retargeter_accepts_095_output_max(self) -> None:
        retargeter = ScalarToGripperRetargeter(
            ScalarToGripperRetargeterConfig(
                side="right",
                input_channel_name="right_hand_q6",
                effector_name="right_hand",
                source_dim=2,
                output_max=0.95,
            )
        )
        observation = TeleopObservation(
            channels={
                "right_hand_q6": ObservationChannel(
                    channel_name="right_hand_q6",
                    capability="joint_vector",
                    side="right",
                    value=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    frame_name=None,
                    timestamp=1.0,
                    source_name="unit",
                )
            },
            timestamp=1.0,
        )

        command = retargeter.retarget(observation)

        np.testing.assert_allclose(
            command.commands["right_hand"].value,
            np.array([0.95, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

    def test_scalar_gripper_retargeter_maps_q6_01_to_09_to_full_gripper_range(self) -> None:
        retargeter = ScalarToGripperRetargeter(
            ScalarToGripperRetargeterConfig(
                side="right",
                input_channel_name="right_hand_q6",
                effector_name="right_hand",
                source_dim=2,
                output_max=0.95,
            )
        )

        def command_for(value: float) -> np.ndarray:
            observation = TeleopObservation(
                channels={
                    "right_hand_q6": ObservationChannel(
                        channel_name="right_hand_q6",
                        capability="joint_vector",
                        side="right",
                        value=np.array([0.0, 0.0, value, 0.0, 0.0, 0.0], dtype=np.float32),
                        frame_name=None,
                        timestamp=1.0,
                        source_name="unit",
                    )
                },
                timestamp=1.0,
            )
            return retargeter.retarget(observation).commands["right_hand"].value

        np.testing.assert_allclose(
            command_for(0.1),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            command_for(0.5),
            np.array([0.475, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            command_for(0.9),
            np.array([0.95, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

    def test_run_collect_builds_dmtac_gripper_retargeter_from_q6(self) -> None:
        args = SimpleNamespace(
            hand_backend="dmtac-gripper",
            left_hand_port="/dev/q6-left",
            right_hand_port="",
            left_gripper_port="/dev/gripper-left",
            right_gripper_port="",
            left_gripper_id=1,
            right_gripper_id=1,
            dmtac_default_config=False,
            with_tactile=True,
            dmtac_tactile=None,
            gripper_task_profile="",
            gripper_source_dim=None,
            gripper_output_dim=None,
            gripper_invert=None,
            gripper_alpha=None,
            gripper_speed=0.12,
            gripper_torque=0.34,
            gripper_max_action=0.8,
            gripper_send_threshold=0.01,
            gripper_command_hz=12.0,
            gripper_continuous_grasp=True,
            gripper_telemetry_hz=20.0,
        )

        retargeters = run_collect._build_hand_retargeters(args, ("left",))
        configs = run_collect._dmtac_backend_configs(args, ("left",))

        self.assertIsInstance(retargeters["left"], ScalarToGripperRetargeter)
        self.assertEqual(retargeters["left"].config.source_dim, 2)
        self.assertEqual(retargeters["left"].config.output_dim, 0)
        self.assertFalse(retargeters["left"].config.invert)
        self.assertAlmostEqual(retargeters["left"].config.input_min, 0.1)
        self.assertAlmostEqual(retargeters["left"].config.input_max, 0.9)
        self.assertEqual(configs["left"].gripper_port, "/dev/gripper-left")
        self.assertEqual(configs["left"].command_dim, 0)
        self.assertFalse(configs["left"].connect_tactile)
        self.assertAlmostEqual(configs["left"].speed, 0.12)
        self.assertAlmostEqual(configs["left"].torque, 0.34)
        self.assertAlmostEqual(configs["left"].max_action, 0.8)
        self.assertAlmostEqual(configs["left"].command_hz, 12.0)
        self.assertTrue(configs["left"].continuous_grasp)
        self.assertFalse(configs["left"].tactile_parallel_read)
        self.assertTrue(configs["left"].tactile_global_read_lock)
        self.assertEqual(configs["left"].tactile_compute_profile, "dist-only")

    def test_run_collect_can_disable_dmtac_parallel_tactile_read(self) -> None:
        config = DMTacGripperBackendConfig(side="left", connect_hardware=False, connect_tactile=True)
        policy = {
            "speed": 0.2,
            "torque": 0.2,
            "max_action": 0.9,
            "output_dim": 0,
            "send_threshold": 0.002,
            "command_hz": 20.0,
            "continuous_grasp": False,
            "telemetry_hz": 10.0,
        }
        args = SimpleNamespace(
            dmtac_tactile_timeout_ms=None,
            dmtac_tactile_wait_for_new=None,
            dmtac_tactile_read_hz=2.5,
            dmtac_tactile_process_read=True,
            dmtac_tactile_worker_cpus=(6, 7),
            dmtac_tactile_worker_nice=15,
            dmtac_tactile_worker_queue_size=4,
            dmtac_tactile_fields=("raw",),
            dmtac_tactile_compute_profile="legacy",
            dmtac_tactile_parallel_read=None,
            dmtac_tactile_sequential_read=True,
            dmtac_tactile_global_read_lock=None,
            dmtac_tactile_max_fps=30,
            dmtac_tactile_warmup_frames=3,
            dmtac_tactile_denoise=True,
            dmtac_tactile_flat_vector=True,
            dmtac_tactile=True,
        )

        updated = run_collect._apply_gripper_policy_to_dmtac_config(config, args=args, policy=policy)

        self.assertFalse(updated.tactile_parallel_read)
        self.assertEqual(updated.tactile_fields, ("raw",))
        self.assertEqual(updated.tactile_compute_profile, "legacy")
        self.assertAlmostEqual(updated.tactile_read_hz, 2.5)
        self.assertTrue(updated.tactile_process_read)
        self.assertEqual(updated.tactile_process_cpu_affinity, (6, 7))
        self.assertEqual(updated.tactile_process_nice, 15)
        self.assertEqual(updated.tactile_process_queue_size, 4)
        self.assertEqual(updated.tactile_max_fps, 30)
        self.assertEqual(updated.tactile_warmup_frames, 3)
        self.assertTrue(updated.tactile_denoise)
        self.assertTrue(updated.tactile_store_flat_vector)

    def test_dmtac_runtime_tactile_defaults_match_assignment_gui_load(self) -> None:
        config = DMTacGripperBackendConfig(
            side="left",
            connect_hardware=False,
            connect_tactile=True,
        )

        self.assertFalse(config.tactile_denoise)
        self.assertEqual(config.tactile_warmup_frames, 0)
        self.assertFalse(config.tactile_store_flat_vector)
        self.assertFalse(config.tactile_parallel_read)
        self.assertTrue(config.tactile_global_read_lock)
        self.assertAlmostEqual(config.tactile_read_hz, 1.0)
        self.assertTrue(config.tactile_process_read)
        self.assertEqual(config.tactile_process_nice, 12)
        self.assertEqual(config.tactile_process_cpu_affinity, ())
        self.assertEqual(config.tactile_process_queue_size, 2)
        self.assertEqual(config.tactile_max_fps, 30)
        self.assertEqual(config.tactile_fields, ("raw",))

    def test_gripper_task_profile_overrides_force_for_selected_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_path = Path(tmp_dir) / "gripper_tasks.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "tasks": {
                            "soft_grasp": {
                                "gripper": {
                                    "force": 0.17,
                                    "speed": 0.09,
                                    "max_close": 0.55,
                                    "output_dim": 0,
                                    "continuous": True,
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                task_id="soft_grasp",
                task_name="",
                gripper_task_profile=str(profile_path),
                gripper_source_dim=None,
                gripper_output_dim=None,
                gripper_invert=None,
                gripper_alpha=None,
                gripper_speed=None,
                gripper_torque=None,
                gripper_max_action=None,
                gripper_send_threshold=None,
                gripper_command_hz=None,
                gripper_continuous_grasp=None,
                gripper_telemetry_hz=None,
            )

            policy = run_collect._resolved_gripper_policy(args)

        self.assertAlmostEqual(policy["torque"], 0.17)
        self.assertAlmostEqual(policy["speed"], 0.09)
        self.assertAlmostEqual(policy["max_action"], 0.55)
        self.assertEqual(policy["output_dim"], 0)
        self.assertTrue(policy["continuous_grasp"])

    def test_real_teleop_gripper_task_selects_profile_force_and_speed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_path = Path(tmp_dir) / "gripper_tasks.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "gripper": {"force": 0.2, "speed": 0.2, "max_action": 0.77},
                        "tasks": {
                            "soft_grasp": {"gripper": {"force": 0.12, "speed": 0.10}},
                            "firm_grasp": {"gripper": {"force": 0.36, "speed": 0.22}},
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                gripper_task_profile=str(profile_path),
                gripper_task="firm_grasp",
                task_id="soft_grasp",
                task_name="",
                task="",
                gripper_source_dim=None,
                gripper_output_dim=None,
                gripper_invert=None,
                gripper_alpha=None,
                gripper_speed=None,
                gripper_torque=None,
                gripper_max_action=None,
                gripper_send_threshold=None,
                gripper_command_hz=None,
                gripper_continuous_grasp=None,
                gripper_telemetry_hz=None,
            )

            policy = real_teleop._resolved_gripper_policy(args)

        self.assertEqual(args._resolved_gripper_task, "firm_grasp")
        self.assertAlmostEqual(policy["torque"], 0.36)
        self.assertAlmostEqual(policy["speed"], 0.22)
        self.assertAlmostEqual(policy["max_action"], 0.77)

    def test_real_teleop_gripper_task_profile_supports_left_right_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_path = Path(tmp_dir) / "gripper_tasks.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "gripper": {
                            "speed": 1.0,
                            "force": 0.2,
                            "max_action": 0.95,
                            "input_min": 0.12,
                            "input_max": 0.88,
                            "left": {"force": 0.21},
                            "right": {"force": 0.22},
                        },
                        "tasks": {
                            "firm_grasp": {
                                "gripper": {
                                    "force": 0.4,
                                    "input_min": 0.2,
                                    "input_max": 0.8,
                                    "left": {"force": 0.31, "q6_min": 0.15, "q6_max": 0.85},
                                    "right": {"force": 0.47},
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                gripper_task_profile=str(profile_path),
                gripper_task="firm_grasp",
                task_id="",
                task_name="",
                task="",
                gripper_source_dim=None,
                gripper_output_dim=None,
                gripper_invert=None,
                gripper_alpha=None,
                gripper_speed=None,
                gripper_torque=None,
                gripper_max_action=None,
                gripper_send_threshold=None,
                gripper_command_hz=None,
                gripper_continuous_grasp=None,
                gripper_telemetry_hz=None,
            )

            base_policy = real_teleop._resolved_gripper_policy(args)
            left_policy = real_teleop._resolved_gripper_policy_for_side(args, "left")
            right_policy = real_teleop._resolved_gripper_policy_for_side(args, "right")

        self.assertAlmostEqual(base_policy["speed"], 1.0)
        self.assertAlmostEqual(base_policy["torque"], 0.4)
        self.assertAlmostEqual(base_policy["input_min"], 0.2)
        self.assertAlmostEqual(base_policy["input_max"], 0.8)
        self.assertAlmostEqual(left_policy["speed"], 1.0)
        self.assertAlmostEqual(right_policy["speed"], 1.0)
        self.assertAlmostEqual(left_policy["torque"], 0.31)
        self.assertAlmostEqual(right_policy["torque"], 0.47)
        self.assertAlmostEqual(left_policy["input_min"], 0.15)
        self.assertAlmostEqual(left_policy["input_max"], 0.85)
        self.assertAlmostEqual(right_policy["input_min"], 0.2)
        self.assertAlmostEqual(right_policy["input_max"], 0.8)

    def test_real_teleop_example_gripper_profile_builds_dmtac_config(self) -> None:
        args = SimpleNamespace(
            hand_backend="dmtac-gripper",
            gripper_task_profile="teleop_runtime/configs/gripper_task_profile.example.json",
            gripper_task="",
            task_id="",
            task_name="",
            task="",
            gripper_source_dim=None,
            gripper_output_dim=None,
            gripper_invert=None,
            gripper_alpha=None,
            gripper_speed=None,
            gripper_torque=None,
            gripper_max_action=None,
            gripper_send_threshold=None,
            gripper_command_hz=None,
            gripper_continuous_grasp=None,
            gripper_telemetry_hz=None,
            dmtac_default_config=True,
            dmtac_config="latest",
            dmtac_validate_current_usb=False,
            dmtac_allow_missing=False,
            dmtac_assume_yes_for_extra=True,
            dmtac_tactile_backend="cpu",
            dmtac_tactile=None,
            dmtac_tactile_timeout_ms=None,
            dmtac_tactile_wait_for_new=None,
            dmtac_tactile_compute_profile="dist-only",
            dmtac_tactile_sequential_read=False,
        )
        assignment = {
            "assignments": {
                "right": {
                    "gripper": {
                        "port": "/dev/serial/by-path/right-gripper",
                        "gripper_id": 1,
                        "baudrate": 115200,
                        "timeout": 0.03,
                    },
                    "tactile": {"inner": None, "outer": None},
                }
            }
        }

        with patch.object(real_teleop, "load_dmtac_hardware_assignment", return_value=assignment):
            configs = real_teleop._dmtac_backend_configs(args, ("right",))

        self.assertIn("right", configs)
        self.assertAlmostEqual(configs["right"].max_action, 1.0)
        self.assertAlmostEqual(configs["right"].send_threshold, 0.002)
        for task_name in ("soft_grasp", "firm_grasp"):
            task_args = argparse.Namespace(**vars(args))
            task_args.gripper_task = task_name
            real_teleop._clear_gripper_policy_cache(task_args)
            policy = real_teleop._resolved_gripper_policy(task_args)
            self.assertAlmostEqual(policy["max_action"], 1.0)
            self.assertAlmostEqual(policy["send_threshold"], 0.002)

    def test_real_teleop_keeps_gripper_gui_visible_without_backend(self) -> None:
        components = SimpleNamespace(
            hand_backend_mode=real_teleop.HAND_BACKEND_DMTAC_GRIPPER,
            active_sides=("right",),
            left_hand_backend=None,
            right_hand_backend=None,
            left_gripper_retargeter=None,
            right_gripper_retargeter=None,
            session=SimpleNamespace(sources=[], calibration=None),
            gripper_policy={},
        )

        self.assertTrue(real_teleop._gripper_gui_available(components))

    def test_real_teleop_running_gui_exposes_hold(self) -> None:
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
            def __init__(self) -> None:
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
                return []

        components = SimpleNamespace(
            hand_backend_mode=real_teleop.HAND_BACKEND_DMTAC_GRIPPER,
            active_sides=("right",),
            left_hand_backend=None,
            right_hand_backend=None,
            left_gripper_retargeter=None,
            right_gripper_retargeter=None,
            session=SimpleNamespace(sources=[], calibration=None),
            gripper_policy={},
        )
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=_FakeUi(),
            args=SimpleNamespace(host="127.0.0.1", port=8765, load_profile="", debug_ik_timing=False),
            components=components,
        )

        control.publish_teleop_running()

        running_publish = next(
            published
            for published in reversed(control.ui.published)
            if published.get("stage") == "RM65 真机遥操运行中"
        )
        self.assertIn("h", running_publish["active_controls"])

    def test_real_teleop_running_auto_enables_q6_gripper_control(self) -> None:
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
            def __init__(self) -> None:
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
                return []

        components = SimpleNamespace(
            hand_backend_mode=real_teleop.HAND_BACKEND_DMTAC_GRIPPER,
            active_sides=("right",),
            left_hand_backend=None,
            right_hand_backend=None,
            left_gripper_retargeter=None,
            right_gripper_retargeter=SimpleNamespace(
                config=SimpleNamespace(
                    source_dim=2,
                    output_dim=0,
                    invert=False,
                    alpha=1.0,
                    output_min=0.0,
                    output_max=0.95,
                )
            ),
            session=SimpleNamespace(sources=[], calibration=None),
            gripper_policy={"source_dim": 2, "output_dim": 0, "max_action": 0.95},
        )
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=_FakeUi(),
            args=SimpleNamespace(host="127.0.0.1", port=8765, load_profile="", debug_ik_timing=False),
            components=components,
        )

        control.publish_teleop_running()

        running_publish = next(
            published
            for published in reversed(control.ui.published)
            if published.get("stage") == "RM65 真机遥操运行中"
        )
        self.assertTrue(control.gripper_control_enabled())
        self.assertTrue(running_publish["gripper_control_enabled"])

    def test_real_teleop_hold_event_requests_hold_and_pauses_gripper(self) -> None:
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
            def __init__(self) -> None:
                self.events = ["h"]
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

        components = SimpleNamespace(
            hand_backend_mode=real_teleop.HAND_BACKEND_DMTAC_GRIPPER,
            active_sides=("right",),
            left_hand_backend=None,
            right_hand_backend=None,
            left_gripper_retargeter=None,
            right_gripper_retargeter=SimpleNamespace(
                config=SimpleNamespace(
                    source_dim=2,
                    output_dim=0,
                    invert=False,
                    alpha=1.0,
                    output_min=0.0,
                    output_max=0.95,
                )
            ),
            session=SimpleNamespace(sources=[], calibration=None),
            gripper_policy={"source_dim": 2, "output_dim": 0, "max_action": 0.95},
        )
        ui = _FakeUi()
        control = real_teleop.RealTeleopControlInput(
            terminal=_FakeTerminal(),
            ui=ui,
            args=SimpleNamespace(host="127.0.0.1", port=8765, load_profile="", debug_ik_timing=False),
            components=components,
        )
        control.publish_teleop_running()
        control._gripper_control_enabled = True

        control.poll()

        hold_publish = next(
            published
            for published in reversed(ui.published)
            if published.get("stage") == "Hold / 停臂保持"
        )
        self.assertTrue(control.consume_hold_request())
        self.assertFalse(control.gripper_control_enabled())
        active_controls = set(hold_publish["active_controls"])
        self.assertTrue({"p", "c", "q"}.issubset(active_controls))
        self.assertFalse(hold_publish["gripper_control_enabled"])

    def test_dmtac_backend_rejects_nonzero_command_dim(self) -> None:
        with self.assertRaisesRegex(ValueError, "slot 0"):
            DMTacGripperBackendConfig(
                side="left",
                gripper_port="/dev/null",
                connect_hardware=False,
                connect_tactile=False,
                command_dim=2,
                max_action=0.8,
            )

    def test_dmtac_backend_accepts_095_max_action(self) -> None:
        config = DMTacGripperBackendConfig(
            side="right",
            connect_hardware=False,
            connect_tactile=False,
            max_action=0.95,
        )

        self.assertAlmostEqual(config.max_action, 0.95)

    def test_dmtac_backend_rate_limits_q6_commands_but_not_fixed_commands(self) -> None:
        backend = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="left",
                connect_hardware=False,
                connect_tactile=False,
                command_hz=1.0,
                send_threshold=0.0,
            )
        )
        first = EffectorCommand(
            effector_name="left_hand",
            side="left",
            command_type="joint_vector",
            value=np.asarray([0.0, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        second = EffectorCommand(
            effector_name="left_hand",
            side="left",
            command_type="joint_vector",
            value=np.asarray([0.0, 0.4, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

        self.assertTrue(backend.send_command(first))
        self.assertFalse(backend.send_command(second))
        self.assertEqual(backend.get_state()["gripper_last_send_status"], "rate_limited")
        self.assertTrue(backend.send_fixed_joint_pos(0.0))

    def test_dmtac_gripper_telemetry_skips_recent_command_window(self) -> None:
        class _FakeDevice:
            gripper = object()
            tactile = None

            def __init__(self) -> None:
                self.reads = 0

            def read_gripper_state(self):
                self.reads += 1
                raise AssertionError("recent command telemetry should skip gripper state read")

        fake = _FakeDevice()
        backend = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="left",
                connect_hardware=True,
                connect_tactile=False,
                gripper_telemetry_command_grace_s=1.0,
            )
        )
        backend.device = fake
        backend._last_sent_command_timestamp = time.time()

        backend._read_telemetry_once()
        state = backend.get_state(include_tactile_arrays=False)

        self.assertEqual(fake.reads, 0)
        self.assertTrue(state["gripper_read_skipped"])
        self.assertEqual(state["gripper_read_skip_reason"], "recent_command")

    def test_dmtac_backend_reuses_static_motion_parameters_between_commands(self) -> None:
        class _FakeDevice:
            gripper = object()
            tactile = None

            def __init__(self) -> None:
                self.moves: list[dict[str, object]] = []

            def move(self, joint_pos: float, **kwargs: object) -> None:
                self.moves.append({"joint_pos": float(joint_pos), **dict(kwargs)})

        fake = _FakeDevice()
        backend = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="left",
                connect_hardware=True,
                connect_tactile=False,
                speed=1.0,
                torque=0.2,
                continuous_grasp=False,
            )
        )
        backend.device = fake
        backend._applied_speed = 1.0
        backend._applied_torque = 0.2
        backend._applied_continuous_grasp = False

        self.assertTrue(backend.send_fixed_joint_pos(np.asarray([0.2], dtype=np.float32)))
        self.assertIsNone(fake.moves[-1]["speed"])
        self.assertIsNone(fake.moves[-1]["torque"])
        self.assertIsNone(fake.moves[-1]["continuous"])

        backend.config.torque = 0.4
        backend.config.continuous_grasp = True
        self.assertTrue(backend.send_fixed_joint_pos(np.asarray([0.3], dtype=np.float32)))
        self.assertIsNone(fake.moves[-1]["speed"])
        self.assertAlmostEqual(fake.moves[-1]["torque"], 0.4)
        self.assertTrue(fake.moves[-1]["continuous"])

    def test_deploy_builds_dmtac_gripper_backend_from_assignment(self) -> None:
        parser = argparse.ArgumentParser()
        deploy_run_policy.add_args(parser)
        args = parser.parse_args(
            [
                "--no-arms",
                "--hand-backend",
                "dmtac-gripper",
                "--active-sides",
                "left",
                "--dmtac-no-validate-current-usb",
                "--gripper-output-dim",
                "0",
                "--gripper-speed",
                "0.11",
                "--gripper-force",
                "0.22",
            ]
        )
        active_sides = deploy_run_policy.validate_args(args)
        assignment = {
            "assignments": {
                "left": {
                    "gripper": {
                        "port": "/dev/serial/by-id/left-gripper",
                        "gripper_id": 1,
                        "baudrate": 115200,
                        "timeout": 0.03,
                    },
                    "tactile": {"inner": None, "outer": None},
                },
                "right": {"gripper": None, "tactile": {"inner": None, "outer": None}},
            }
        }

        with patch.object(deploy_run_policy, "load_dmtac_hardware_assignment", return_value=assignment):
            backends = deploy_run_policy.build_hand_backends(args, active_sides)
            specs = deploy_run_policy.hand_action_specs_for_backends(args, backends, active_sides)

        self.assertIsInstance(backends["left"], DMTacGripperBackend)
        self.assertEqual(backends["left"].config.gripper_port, "/dev/serial/by-id/left-gripper")
        self.assertEqual(backends["left"].config.command_dim, 0)
        self.assertAlmostEqual(backends["left"].config.speed, 0.11)
        self.assertAlmostEqual(backends["left"].config.torque, 0.22)
        self.assertEqual(specs["left"]["hand_model"], "dmtac_gripper")
        self.assertEqual(specs["left"]["active_mask"], [True, False, False, False, False, False])

    def test_deploy_gripper_task_profile_binds_force_and_speed_to_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_path = Path(tmp_dir) / "gripper_tasks.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "gripper": {"force": 0.2, "speed": 0.2},
                        "tasks": {
                            "soft_grasp": {"gripper": {"force": 0.10, "speed": 0.08}},
                            "firm_grasp": {"gripper": {"force": 0.42, "speed": 0.24}},
                        },
                    }
                ),
                encoding="utf-8",
            )
            parser = argparse.ArgumentParser()
            deploy_run_policy.add_args(parser)
            args = parser.parse_args(
                [
                    "--no-arms",
                    "--hand-backend",
                    "dmtac-gripper",
                    "--active-sides",
                    "left",
                    "--dmtac-no-validate-current-usb",
                    "--gripper-task-profile",
                    str(profile_path),
                    "--task-id",
                    "soft_grasp",
                    "--gripper-task",
                    "firm_grasp",
                ]
            )
            active_sides = deploy_run_policy.validate_args(args)
            assignment = {
                "assignments": {
                    "left": {
                        "gripper": {
                            "port": "/dev/serial/by-id/left-gripper",
                            "gripper_id": 1,
                            "baudrate": 115200,
                            "timeout": 0.03,
                        },
                        "tactile": {"inner": None, "outer": None},
                    }
                }
            }

            with patch.object(deploy_run_policy, "load_dmtac_hardware_assignment", return_value=assignment):
                backends = deploy_run_policy.build_hand_backends(args, active_sides)

        self.assertEqual(args._resolved_gripper_task, "firm_grasp")
        self.assertAlmostEqual(backends["left"].config.torque, 0.42)
        self.assertAlmostEqual(backends["left"].config.speed, 0.24)

    def test_dmtac_tactile_telemetry_does_not_block_gripper_command_lock(self) -> None:
        class _FakeGripperState:
            joint_pos = 0.0
            timestamp = 1.0

            def to_dict(self) -> dict[str, float]:
                return {"joint_pos": self.joint_pos, "timestamp": self.timestamp}

        class _FakeDevice:
            gripper = object()
            tactile = object()

            def __init__(self) -> None:
                self.tactile_read_started = threading.Event()
                self.release_tactile_read = threading.Event()
                self.moves: list[float] = []

            def read_gripper_state(self) -> _FakeGripperState:
                return _FakeGripperState()

            def read_tactile(self, **_kwargs):
                self.tactile_read_started.set()
                self.release_tactile_read.wait(timeout=1.0)
                return {}

            def move(self, joint_pos: float, **_kwargs) -> None:
                self.moves.append(float(joint_pos))

        fake = _FakeDevice()
        backend = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="left",
                connect_hardware=True,
                connect_tactile=True,
                tactile_wait_for_new=False,
                tactile_process_read=False,
            )
        )
        backend.device = fake
        thread = threading.Thread(target=backend._read_telemetry_once)
        thread.start()
        self.assertTrue(fake.tactile_read_started.wait(timeout=0.2))

        started_at = time.perf_counter()
        backend.send_fixed_joint_pos(np.asarray([0.5], dtype=np.float32))
        elapsed_s = time.perf_counter() - started_at

        fake.release_tactile_read.set()
        thread.join(timeout=1.0)
        self.assertLess(elapsed_s, 0.05)
        self.assertEqual(fake.moves, [0.5])

    def test_dmtac_backend_state_can_skip_tactile_arrays_and_flat_vector(self) -> None:
        class _FakeFrame:
            fid = 42

            def __init__(self) -> None:
                self.distributed_force = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
                self.timings_ms = {"distributed_force": 12.5, "getDevStatus": 1.25}

        class _FakeDevice:
            gripper = None
            tactile = object()

            def read_tactile(self, **_kwargs):
                return {"M_TEST": _FakeFrame()}

        backend = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="left",
                connect_hardware=False,
                connect_tactile=True,
                tactile_process_read=False,
                tactile_fields=("distributed_force",),
            )
        )
        backend.device = _FakeDevice()

        backend._read_telemetry_once()

        self.assertIsNone(backend._latest_tactile)
        light_state = backend.get_state(include_tactile_arrays=False)
        self.assertIsNone(light_state["tactile"])
        self.assertIsNone(light_state["tactile_dict"])
        self.assertEqual(light_state["tactile_devices"], ("M_TEST",))
        self.assertEqual(light_state["tactile_shapes"], {"M_TEST": (2, 3, 3)})
        self.assertEqual(light_state["tactile_frame_ids"], {"M_TEST": 42})
        self.assertEqual(light_state["tactile_timings_ms"]["M_TEST"]["distributed_force"], 12.5)

        dict_only_state = backend.get_state(include_tactile_vector=False)
        self.assertIsNone(dict_only_state["tactile"])
        self.assertIn("M_TEST", dict_only_state["tactile_dict"])
        self.assertIsNot(
            dict_only_state["tactile_dict"]["M_TEST"],
            backend._latest_tactile_dict["M_TEST"],
        )

        borrowed_state = backend.get_state(
            include_tactile_vector=False,
            copy_tactile_arrays=False,
        )
        self.assertIs(
            borrowed_state["tactile_dict"]["M_TEST"],
            backend._latest_tactile_dict["M_TEST"],
        )

        default_state = backend.get_state()
        self.assertIsNone(default_state["tactile"])

        backend.config.tactile_store_flat_vector = True
        compatible_state = backend.get_state()
        np.testing.assert_allclose(
            compatible_state["tactile"],
            np.arange(18, dtype=np.float32),
        )

    def test_dmtac_backend_can_read_raw_tactile_images_without_force_array(self) -> None:
        class _RawImage:
            def __init__(self) -> None:
                self.img = np.arange(12, dtype=np.uint8).reshape(3, 4)
                self.serial = "M_RAW"

        class _FakeFrame:
            fid = 7

            def __init__(self) -> None:
                self.raw = _RawImage()
                self.fids = {"raw": 7}
                self.timings_ms = {"raw": 1.5, "getDevStatus": 0.25}

        class _FakeDevice:
            gripper = None
            tactile = object()

            def read_tactile(self, **_kwargs):
                return {"M_RAW": _FakeFrame()}

        backend = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="left",
                connect_hardware=False,
                connect_tactile=True,
                tactile_process_read=False,
                tactile_fields=("raw",),
                tactile_compute_profile="requested",
            )
        )
        backend.device = _FakeDevice()

        backend._read_telemetry_once()
        state = backend.get_state(include_tactile_vector=False)

        self.assertEqual(state["tactile_fields"], ("raw",))
        self.assertEqual(state["tactile_raw_shape_per_sensor"], (3, 4))
        self.assertEqual(state["tactile_shapes"], {"M_RAW": (3, 4)})
        self.assertEqual(state["tactile_frame_ids"], {"M_RAW": 7})
        self.assertEqual(state["tactile_field_by_key"], {"M_RAW": "raw"})
        self.assertEqual(state["tactile_dict"]["M_RAW"].dtype, np.dtype("uint8"))
        np.testing.assert_array_equal(
            state["tactile_dict"]["M_RAW"],
            np.arange(12, dtype=np.uint8).reshape(3, 4),
        )
        self.assertIsNone(state["tactile"])

    def test_dmtac_process_worker_queue_is_drained_by_get_state(self) -> None:
        class _FakeQueue:
            def __init__(self) -> None:
                self.items = [
                    {
                        "type": "frame",
                        "timestamp": 123.0,
                        "tactile_dict": {
                            "M_TEST": np.ones((2, 3, 3), dtype=np.float32),
                        },
                        "frame_ids": {"M_TEST": 7},
                        "shapes": {"M_TEST": (2, 3, 3)},
                        "timings_ms": {"M_TEST": {"distributed_force": 4.5}},
                        "devices": ["M_TEST"],
                        "version": 3,
                        "read_elapsed_s": 0.012,
                        "gate_wait_s": 0.001,
                        "error": None,
                    }
                ]

            def get_nowait(self):
                if not self.items:
                    raise queue.Empty()
                return self.items.pop(0)

        backend = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="left",
                connect_hardware=False,
                connect_tactile=True,
                tactile_process_read=True,
            )
        )
        backend._tactile_worker_queue = _FakeQueue()

        state = backend.get_state(include_tactile_arrays=False)

        self.assertEqual(state["tactile_version"], 3)
        self.assertEqual(state["tactile_devices"], ("M_TEST",))
        self.assertEqual(state["tactile_frame_ids"], {"M_TEST": 7})
        self.assertEqual(state["tactile_process_queue_size"], 2)
        self.assertAlmostEqual(state["tactile_read_elapsed_s"], 0.012)

    def test_dmtac_tactile_worker_payload_tracks_parent_and_name(self) -> None:
        backend = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="right",
                connect_hardware=False,
                connect_tactile=True,
                tactile_process_read=True,
            )
        )

        payload = backend._tactile_worker_config_payload()

        self.assertEqual(payload["parent_pid"], os.getpid())
        self.assertEqual(payload["worker_name"], "dmtac_right_tac")

    def test_dmtac_tactile_worker_parent_watchdog_sets_stop_event(self) -> None:
        from teleop_runtime.backends.grippers import dmtac as dmtac_backend

        stop_event = threading.Event()
        dmtac_backend._start_parent_watchdog(
            stop_event,
            999999999,
            poll_s=0.01,
        )

        self.assertTrue(stop_event.wait(timeout=0.5))

    def test_dmtac_process_worker_queue_is_drained_by_receiver_thread(self) -> None:
        class _FakeQueue:
            def __init__(self) -> None:
                self.items = [
                    {
                        "type": "frame",
                        "timestamp": 123.0,
                        "tactile_dict": {
                            "M_TEST": np.ones((2, 3, 3), dtype=np.float32),
                        },
                        "frame_ids": {"M_TEST": 7},
                        "shapes": {"M_TEST": (2, 3, 3)},
                        "timings_ms": {"M_TEST": {"distributed_force": 4.5}},
                        "devices": ["M_TEST"],
                        "version": 3,
                        "read_elapsed_s": 0.012,
                        "gate_wait_s": 0.001,
                        "error": None,
                    }
                ]
                self.lock = threading.Lock()

            def get(self, timeout=None):
                del timeout
                with self.lock:
                    if not self.items:
                        raise queue.Empty()
                    return self.items.pop(0)

            def get_nowait(self):
                return self.get(timeout=0.0)

        backend = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="left",
                connect_hardware=False,
                connect_tactile=True,
                tactile_process_read=True,
            )
        )
        backend._tactile_worker_queue = _FakeQueue()
        backend._start_tactile_worker_receiver()
        try:
            deadline = time.time() + 1.0
            while time.time() < deadline:
                state = backend.get_state(include_tactile_arrays=False)
                if state["tactile_version"] == 3:
                    break
                time.sleep(0.01)

            state = backend.get_state(include_tactile_arrays=False)

            self.assertTrue(state["tactile_worker_receiver_alive"])
            self.assertEqual(state["tactile_version"], 3)
            self.assertEqual(state["tactile_devices"], ("M_TEST",))
            self.assertEqual(state["tactile_frame_ids"], {"M_TEST": 7})
            self.assertAlmostEqual(state["tactile_read_elapsed_s"], 0.012)
        finally:
            backend.close()

    def test_dmtac_tactile_read_hz_skips_immediate_re_reads(self) -> None:
        class _FakeDevice:
            gripper = None
            tactile = object()

            def __init__(self) -> None:
                self.read_count = 0

            def read_tactile(self, **_kwargs):
                self.read_count += 1
                return {}

        fake = _FakeDevice()
        backend = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="left",
                connect_hardware=False,
                connect_tactile=True,
                tactile_read_hz=1.0,
                tactile_process_read=False,
            )
        )
        backend.device = fake

        backend._read_telemetry_once()
        first_state = backend.get_state()
        backend._read_telemetry_once()
        skipped_state = backend.get_state()

        self.assertEqual(fake.read_count, 1)
        self.assertFalse(first_state["tactile_read_skipped"])
        self.assertTrue(skipped_state["tactile_read_skipped"])
        self.assertAlmostEqual(skipped_state["tactile_read_hz"], 1.0)
        self.assertIsNone(skipped_state["tactile_read_elapsed_s"])

        assert backend._next_tactile_read_at is not None
        backend._next_tactile_read_at -= 1.1
        backend._read_telemetry_once()
        self.assertEqual(fake.read_count, 2)

    def test_dmtac_empty_tactile_read_does_not_advance_version(self) -> None:
        class _FakeDevice:
            gripper = None
            tactile = object()

            def read_tactile(self, **_kwargs):
                return {}

        backend = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="left",
                connect_hardware=False,
                connect_tactile=True,
                tactile_process_read=False,
            )
        )
        backend.device = _FakeDevice()

        backend._read_telemetry_once()
        state = backend.get_state()

        self.assertEqual(state["tactile_version"], 0)
        self.assertIsNone(state["tactile_timestamp"])
        self.assertEqual(state["tactile_devices"], ())
        self.assertEqual(state["tactile_frame_ids"], {})

    def test_dmtac_tactile_reads_are_serialized_across_backend_instances_by_default(self) -> None:
        class _FakeDevice:
            gripper = None
            tactile = object()

            def __init__(self) -> None:
                self.tactile_read_started = threading.Event()
                self.release_tactile_read = threading.Event()

            def read_tactile(self, **_kwargs):
                self.tactile_read_started.set()
                self.release_tactile_read.wait(timeout=1.0)
                return {}

        left_device = _FakeDevice()
        right_device = _FakeDevice()
        left = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="left",
                connect_hardware=False,
                connect_tactile=True,
                tactile_process_read=False,
            )
        )
        right = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="right",
                connect_hardware=False,
                connect_tactile=True,
                tactile_process_read=False,
            )
        )
        left.device = left_device
        right.device = right_device

        left_thread = threading.Thread(target=left._read_telemetry_once)
        right_thread = threading.Thread(target=right._read_telemetry_once)
        left_thread.start()
        self.assertTrue(left_device.tactile_read_started.wait(timeout=0.2))

        right_thread.start()
        time.sleep(0.05)
        self.assertFalse(right_device.tactile_read_started.is_set())

        left_device.release_tactile_read.set()
        self.assertTrue(right_device.tactile_read_started.wait(timeout=1.0))
        right_device.release_tactile_read.set()
        left_thread.join(timeout=1.0)
        right_thread.join(timeout=1.0)

        self.assertFalse(left_thread.is_alive())
        self.assertFalse(right_thread.is_alive())
        self.assertIsNotNone(right.get_state()["tactile_gate_wait_s"])

    def test_dmtac_backend_close_waits_for_active_tactile_read(self) -> None:
        class _FakeDevice:
            gripper = None
            tactile = object()

            def __init__(self) -> None:
                self.tactile_read_started = threading.Event()
                self.release_tactile_read = threading.Event()
                self.close_called = threading.Event()

            def read_tactile(self, **_kwargs):
                self.tactile_read_started.set()
                self.release_tactile_read.wait(timeout=1.0)
                return {}

            def close(self) -> None:
                self.close_called.set()

        fake = _FakeDevice()
        backend = DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side="left",
                connect_hardware=False,
                connect_tactile=True,
                tactile_wait_for_new=False,
                tactile_process_read=False,
            )
        )
        backend.device = fake
        read_thread = threading.Thread(target=backend._read_telemetry_once)
        read_thread.start()
        self.assertTrue(fake.tactile_read_started.wait(timeout=0.2))

        close_thread = threading.Thread(target=backend.close)
        close_thread.start()
        time.sleep(0.05)

        self.assertFalse(fake.close_called.is_set())
        fake.release_tactile_read.set()
        close_thread.join(timeout=1.0)
        read_thread.join(timeout=1.0)

        self.assertFalse(close_thread.is_alive())
        self.assertFalse(read_thread.is_alive())
        self.assertTrue(fake.close_called.is_set())
        self.assertIsNone(backend.device)

    def test_schema_records_gripper_action_and_measured_state_separately(self) -> None:
        action = np.full(26, np.nan, dtype=np.float32)
        action[14:20] = np.asarray([0.7, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        sample = CollectorSample(
            index=0,
            episode_index=0,
            frame_index=0,
            timestamp=1.0,
            session_state="running",
            control_hz=40.0,
            action=action,
            observation_state=np.full(38, np.nan, dtype=np.float32),
            observation_last_action=np.full(26, np.nan, dtype=np.float32),
            arm_state_source="unit",
            active_sides=("left",),
            hand_states={
                "left": {
                    "joint_pos": np.asarray([0.4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    "measured_joint_pos": np.asarray([0.4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    "commanded_joint_pos": np.asarray([0.7, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    "last_sent_command": np.asarray([0.7, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    "gripper_state_scalar": 0.4,
                    "gripper_measured_scalar": 0.4,
                    "gripper_commanded_scalar": 0.7,
                },
                "right": {},
            },
            metadata={
                "command_metadata_by_effector": {
                    "left_hand": {
                        "source_value": 0.3,
                        "source_dim": 1,
                        "output_dim": 0,
                        "gripper_slot": 0,
                        "gripper_action_dim": 1,
                        "invert": False,
                    }
                }
            },
        )

        row, _camera_expectations = collector_sample_to_row(sample)

        self.assertAlmostEqual(row["action.gripper.left.target"], 0.7, places=6)
        self.assertAlmostEqual(row["observation.gripper.left.measured"], 0.4, places=6)
        self.assertAlmostEqual(row["observation.gripper.left.commanded"], 0.7, places=6)
        self.assertEqual(row["action.gripper.left.source_dim"], 1)
        self.assertEqual(row["action.gripper.left.slot_index"], 0)
        self.assertEqual(row["action.gripper.left.action_dim"], 1)
        self.assertFalse(row["action.gripper.left.invert"])


if __name__ == "__main__":
    unittest.main()
