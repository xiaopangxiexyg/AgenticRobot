from __future__ import annotations

import json
import sys
import tempfile
import time
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation as R

from teleop_runtime.backends.hands.inspire_tac import InspireTacHandBackend, InspireTacHandBackendConfig
from teleop_runtime.calibration import TeleopCalibration
from teleop_runtime.configs import build_annotated_dual_rm65_calibration
from teleop_runtime.debug.run_vive_sapien_virtual import (
    _build_optional_hand_command,
    _preview_tcp_pose_desk,
    _resolve_robot_home_for_side,
    _send_optional_hand_command,
    _update_optional_hand_preview_from_channels,
    build_components,
)
from teleop_runtime.debug.run_vive_rm65_real_teleop import (
    _build_components as build_real_components,
)
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.hand_tcp_alignment import tracker_hand_to_rm65_tcp_rotation
from teleop_runtime.hand_render import inspire_q6_to_render_qpos
from teleop_runtime.initial_alignment import AlignmentSideState, TeleopInitialAlignment
from teleop_runtime.retargeters.pose_to_arm import PoseToArmRetargeter, PoseToArmRetargeterConfig
from teleop_runtime.retargeters.joints_to_dexhand import (
    JointsToDexHandRetargeter,
    JointsToDexHandRetargeterConfig,
)
from teleop_runtime.safety import SafetyMonitor, TeleopSafetyConfig
from teleop_runtime.schemas import ObservationChannel
from teleop_runtime.schemas import TeleopObservation
from teleop_runtime.session import TeleopSession, TeleopSessionConfig
from teleop_runtime.sources.hand import HandQ6Source
from teleop_runtime.tcp import HAND_BASE_TCP_FRAME
from teleop_runtime.tracker_hand import (
    DEFAULT_CANONICAL_HAND_RPY_DEG,
    canonical_tracker_to_hand_rotation,
)
from teleop_runtime.transforms import identity_pose


def _pose(tx: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    yaw = np.deg2rad(float(yaw_deg))
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    pose = identity_pose()
    pose[:3, :3] = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    pose[0, 3] = float(tx)
    return pose


class StubPoseSource:
    name = "stub_pose_source"

    def __init__(self):
        self.left_valid = True
        self.right_valid = True
        self.left_pose = _pose()
        self.right_pose = _pose()
        self.timestamp = 0.0

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def expected_channel_names(self) -> tuple[str, ...]:
        return ("left_wrist_pose", "right_wrist_pose")

    def get_latest_channels(self) -> dict[str, ObservationChannel]:
        return {
            "left_wrist_pose": ObservationChannel(
                channel_name="left_wrist_pose",
                capability="pose6d",
                side="left",
                value=self.left_pose,
                frame_name="desk",
                timestamp=self.timestamp,
                source_name=self.name,
                valid=self.left_valid,
                critical=True,
            ),
            "right_wrist_pose": ObservationChannel(
                channel_name="right_wrist_pose",
                capability="pose6d",
                side="right",
                value=self.right_pose,
                frame_name="desk",
                timestamp=self.timestamp,
                source_name=self.name,
                valid=self.right_valid,
                critical=True,
            ),
        }


class TeleopRecoveryTests(unittest.TestCase):
    def test_absolute_home_anchor_preserves_full_pose_when_tracker_stays_at_home(self) -> None:
        retargeter = PoseToArmRetargeter(
            PoseToArmRetargeterConfig(
                side="left",
                input_channel_name="left_wrist_pose",
                control_mode="absolute",
            )
        )
        human_home = _pose(tx=0.12, yaw_deg=35.0)
        robot_home = _pose(tx=-0.28, yaw_deg=-70.0)
        retargeter.set_home_anchors(
            human_pose_desk=human_home,
            robot_pose_desk=robot_home,
        )

        target = retargeter._apply_absolute_mapping(human_home)
        np.testing.assert_allclose(target, robot_home, atol=1e-9)

    def test_delta_clutch_reference_keeps_tracker_orientation_at_home(self) -> None:
        retargeter = PoseToArmRetargeter(
            PoseToArmRetargeterConfig(
                side="left",
                input_channel_name="left_wrist_pose",
                control_mode="delta_clutch",
            )
        )
        human_home = _pose(tx=0.12, yaw_deg=35.0)
        robot_home = _pose(tx=-0.28, yaw_deg=-70.0)
        retargeter.set_home_anchors(
            human_pose_desk=human_home,
            robot_pose_desk=robot_home,
        )

        reference = retargeter._apply_clutch_reference_mapping(human_home)
        np.testing.assert_allclose(reference[:3, 3], robot_home[:3, 3], atol=1e-9)
        np.testing.assert_allclose(reference[:3, :3], human_home[:3, :3], atol=1e-9)

    def test_inspire_hand_backend_clips_read_joint_pos_to_unit_range(self) -> None:
        class FakeHand:
            def read_joint_pos(self):
                return np.asarray([-0.2, 0.1, 0.5, 1.2, 0.8, 2.0], dtype=float)

        backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
            )
        )
        backend.hand = FakeHand()
        q6 = backend.read_joint_pos_now()
        np.testing.assert_allclose(q6, np.asarray([0.0, 0.1, 0.5, 1.0, 0.8, 1.0], dtype=float))
        np.testing.assert_allclose(backend.get_state()["joint_pos"], q6)

    def test_inspire_preview_hand_backend_tracks_last_command_without_hardware(self) -> None:
        backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
            )
        )
        backend.send_command(
            SimpleNamespace(
                effector_name="left_hand",
                command_type="joint_vector",
                value=np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=float),
            )
        )
        np.testing.assert_allclose(
            backend.get_state()["joint_pos"],
            np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=float),
        )

    def test_inspire_hand_backend_connection_failure_is_fail_open(self) -> None:
        backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip="192.0.2.1",
                side="left",
                connect_hardware=True,
            )
        )
        fake_modbus_module = types.ModuleType("hardware.inspire_tac.inspire_tac_modbus")

        class FailingModbus:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("hand offline")

        fake_modbus_module.InspireTacModbus = FailingModbus
        with patch.dict(sys.modules, {"hardware.inspire_tac.inspire_tac_modbus": fake_modbus_module}):
            backend.start()

        self.assertIsNone(backend.hand)
        self.assertIn("hand offline", backend.get_state()["telemetry_error"])

    def test_optional_hand_command_is_built_and_sent_without_entering_arm_rig(self) -> None:
        left_hand_backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
            )
        )
        components = SimpleNamespace(
            session=SimpleNamespace(
                calibration=None,
                config=SimpleNamespace(tcp_mode=HAND_BASE_TCP_FRAME),
            ),
            left_hand_retargeter=JointsToDexHandRetargeter(
                JointsToDexHandRetargeterConfig(
                    side="left",
                    input_channel_name="left_hand_q6",
                    effector_name="left_hand",
                )
            ),
            right_hand_retargeter=None,
            left_hand_backend=left_hand_backend,
            right_hand_backend=None,
        )
        channel = ObservationChannel(
            channel_name="left_hand_q6",
            capability="joint_vector",
            side="left",
            value=np.asarray([0.0, 0.0, 0.25, 0.5, 0.75, 1.0], dtype=float),
            frame_name=None,
            timestamp=time.time(),
            source_name="unit_test",
            valid=True,
            critical=False,
        )
        step_result = SimpleNamespace(
            observation=TeleopObservation(
                channels={"left_hand_q6": channel},
                timestamp=channel.timestamp,
                metadata={"tcp_mode": HAND_BASE_TCP_FRAME},
            ),
            channels={"left_hand_q6": channel},
            timestamp=channel.timestamp,
        )
        hand_command = _build_optional_hand_command(components, step_result)
        self.assertIsNotNone(hand_command)
        self.assertIn("left_hand", hand_command.commands)

        _send_optional_hand_command(components, hand_command)
        np.testing.assert_allclose(
            left_hand_backend.get_state()["joint_pos"],
            np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 1.0], dtype=float),
            atol=1e-6,
        )

    def test_optional_hand_preview_updates_from_polled_channels_before_main_loop(self) -> None:
        left_hand_backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
            )
        )
        components = SimpleNamespace(
            session=SimpleNamespace(
                calibration=None,
                config=SimpleNamespace(tcp_mode=HAND_BASE_TCP_FRAME),
            ),
            left_hand_retargeter=JointsToDexHandRetargeter(
                JointsToDexHandRetargeterConfig(
                    side="left",
                    input_channel_name="left_hand_q6",
                    effector_name="left_hand",
                )
            ),
            right_hand_retargeter=None,
            left_hand_backend=left_hand_backend,
            right_hand_backend=None,
        )
        channel = ObservationChannel(
            channel_name="left_hand_q6",
            capability="joint_vector",
            side="left",
            value=np.asarray([0.0, 0.0, 0.25, 0.5, 0.75, 1.0], dtype=float),
            frame_name=None,
            timestamp=time.time(),
            source_name="unit_test",
            valid=True,
            critical=False,
        )
        hand_command = _update_optional_hand_preview_from_channels(
            components,
            {"left_hand_q6": channel},
            timestamp=channel.timestamp,
        )
        self.assertIsNotNone(hand_command)
        self.assertIn("left_hand", hand_command.commands)
        np.testing.assert_allclose(
            left_hand_backend.get_state()["joint_pos"],
            np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 1.0], dtype=float),
            atol=1e-6,
        )

    def test_optional_hand_source_startup_failure_is_fail_open(self) -> None:
        source = HandQ6Source.from_single_hand(
            port="/dev/fake_left_hand",
            side="left",
            critical=False,
            strict_expected=False,
        )
        with patch(
            "hardware.teleop_hand.hand_q6_reader.SingleHandQ6Reader",
            side_effect=RuntimeError("serial unavailable"),
        ):
            source.start()
        try:
            channels = source.get_latest_channels()
            self.assertIn("left_hand_q6", channels)
            self.assertFalse(channels["left_hand_q6"].valid)
            self.assertIn("serial unavailable", channels["left_hand_q6"].metadata["read_error"])
        finally:
            source.close()

    def test_required_hand_source_startup_failure_still_raises(self) -> None:
        source = HandQ6Source.from_single_hand(
            port="/dev/fake_left_hand",
            side="left",
            critical=True,
            strict_expected=True,
        )
        with patch(
            "hardware.teleop_hand.hand_q6_reader.SingleHandQ6Reader",
            side_effect=RuntimeError("serial unavailable"),
        ):
            with self.assertRaises(RuntimeError):
                source.start()

    def test_inspire_q6_render_mapping_expands_to_hand_q12(self) -> None:
        q12 = inspire_q6_to_render_qpos([1.0, 0.5, 1.0, 0.0, 0.25, 0.75])
        expected = np.asarray(
            [
                0.0,
                0.275,
                0.165,
                0.22,
                0.0,
                0.0,
                1.6,
                1.68,
                1.2,
                1.26,
                0.4,
                0.472,
            ],
            dtype=float,
        )
        self.assertEqual(q12.shape, (12,))
        np.testing.assert_allclose(q12, expected, atol=1e-6)

    def test_initial_alignment_round_trip_normalizes_tcp_frame(self) -> None:
        pose = _pose(tx=0.12, yaw_deg=25.0)
        alignment = TeleopInitialAlignment(
            tcp_frame="rm_eef",
            sides={
                "left": AlignmentSideState(
                    side="left",
                    arm_qpos=np.linspace(0.1, 0.6, 6),
                    tcp_pose_base=pose,
                    tcp_pose_desk=pose,
                    hand_qpos_q6=np.linspace(0.0, 1.0, 6),
                    metadata={"connected": True},
                ),
                "right": AlignmentSideState(
                    side="right",
                    arm_qpos=np.linspace(-0.1, -0.6, 6),
                    tcp_pose_base=identity_pose(),
                    tcp_pose_desk=identity_pose(),
                    metadata={"connected": False},
                ),
            },
            metadata={"source": "unit_test"},
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = f"{tmp_dir}/alignment.json"
            alignment.save_json(path)
            loaded = TeleopInitialAlignment.load_json(path)

        self.assertEqual(loaded.tcp_frame, HAND_BASE_TCP_FRAME)
        self.assertEqual(loaded.metadata["source"], "unit_test")
        np.testing.assert_allclose(loaded.get_side("left").arm_qpos, np.linspace(0.1, 0.6, 6))
        np.testing.assert_allclose(loaded.get_side("left").tcp_pose_desk, pose)
        np.testing.assert_allclose(loaded.get_side("left").hand_qpos_q6, np.linspace(0.0, 1.0, 6))
        np.testing.assert_allclose(loaded.get_side("right").arm_qpos, np.linspace(-0.1, -0.6, 6))

    def test_virtual_components_load_initial_alignment_into_preview_backends(self) -> None:
        payload = {
            "format": "teleop_initial_alignment",
            "format_version": 1,
            "tcp_frame": "hand_base",
            "sides": {
                "left": {
                    "side": "left",
                    "arm_qpos": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    "tcp_pose_base": identity_pose().tolist(),
                    "tcp_pose_desk": identity_pose().tolist(),
                    "hand_qpos_q6": None,
                    "metadata": {},
                },
                "right": {
                    "side": "right",
                    "arm_qpos": [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
                    "tcp_pose_base": identity_pose().tolist(),
                    "tcp_pose_desk": identity_pose().tolist(),
                    "hand_qpos_q6": None,
                    "metadata": {},
                },
            },
            "metadata": {},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            alignment_path = f"{tmp_dir}/alignment.json"
            with open(alignment_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            args = SimpleNamespace(
                hz=30.0,
                arm_mode="virtual",
                disable_dual_arm_collision_avoidance=True,
                ik_tuning_json="",
                load_initial_alignment=alignment_path,
                load_profile="",
                left_right_base_distance=0.525,
                left_tracker_name="tracker_1",
                right_tracker_name="tracker_2",
                aux_tracker_name="tracker_3",
                control_mode="absolute",
                translation_scale=1.0,
                collision_scene_profile="dual_arm_hand_proxy_open",
                collision_avoidance_mode="boundary_slide",
                recovery_healthy_frames=8,
                no_record=True,
                host="0.0.0.0",
                port=9999,
                tcp_mode="hand_base",
                left_arm_ip="",
                right_arm_ip="",
                skip_renderer=True,
                headless=True,
            )
            components = build_components(args)
            try:
                self.assertIsNone(components.left_hand_backend)
                self.assertIsNone(components.right_hand_backend)
                self.assertIsNone(components.left_hand_retargeter)
                self.assertIsNone(components.right_hand_retargeter)
                np.testing.assert_allclose(
                    components.left_preview_backend.get_state()["current_qpos"],
                    np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=float),
                )
                np.testing.assert_allclose(
                    components.right_preview_backend.get_state()["current_qpos"],
                    np.asarray([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6], dtype=float),
                )
            finally:
                components.session.close()
                components.left_preview_backend.close()
                components.right_preview_backend.close()

    def test_virtual_components_enable_optional_hand_preview_when_ports_are_provided(self) -> None:
        args = SimpleNamespace(
            hz=30.0,
            arm_mode="virtual",
            disable_dual_arm_collision_avoidance=True,
            ik_tuning_json="",
            load_initial_alignment="",
            load_profile="",
            left_right_base_distance=0.525,
            left_tracker_name="tracker_1",
            right_tracker_name="tracker_2",
            aux_tracker_name="tracker_3",
            control_mode="absolute",
            translation_scale=1.0,
            collision_scene_profile="dual_arm_hand_proxy_open",
            collision_avoidance_mode="boundary_slide",
            recovery_healthy_frames=8,
            no_record=True,
            host="0.0.0.0",
            port=9999,
            tcp_mode="hand_base",
            left_arm_ip="",
            right_arm_ip="",
            skip_renderer=True,
            headless=True,
            left_hand_port="/dev/fake_left_hand",
            right_hand_port="",
            hand_reader_baudrate=115200,
            hand_reader_timeout=0.05,
            show_hand_raw_lines=False,
            hand_joint_order=(5, 4, 3, 2, 0, 1),
            hand_invert=True,
            hand_command_alpha=1.0,
            hand_send_threshold=0.002,
        )
        components = build_components(args)
        try:
            self.assertIsNotNone(components.left_hand_backend)
            self.assertIsNone(components.right_hand_backend)
            self.assertIsNotNone(components.left_hand_retargeter)
            self.assertIsNone(components.right_hand_retargeter)
            self.assertEqual(len(components.session.sources), 3)
        finally:
            components.session.close()
            components.left_preview_backend.close()
            components.right_preview_backend.close()

    def test_virtual_components_single_right_arm_use_single_arm_retargeter(self) -> None:
        args = SimpleNamespace(
            active_sides=("right",),
            hz=30.0,
            arm_mode="virtual",
            disable_dual_arm_collision_avoidance=False,
            ik_tuning_json="",
            load_initial_alignment="",
            load_profile="",
            left_right_base_distance=0.525,
            left_tracker_name="tracker_1",
            right_tracker_name="tracker_2",
            aux_tracker_name="",
            control_mode="absolute",
            translation_scale=1.0,
            collision_scene_profile="dual_arm_hand_proxy_open",
            collision_avoidance_mode="boundary_slide",
            arm_max_angvel=(0.35, 0.35, 0.35, 0.5, 0.5, 0.5),
            recovery_healthy_frames=8,
            no_record=True,
            host="0.0.0.0",
            port=9999,
            tcp_mode="hand_base",
            left_arm_ip="",
            right_arm_ip="",
            skip_renderer=True,
            headless=True,
            left_hand_port="",
            right_hand_port="",
            hand_reader_baudrate=115200,
            hand_reader_timeout=0.05,
            show_hand_raw_lines=False,
            hand_joint_order=(5, 4, 3, 2, 0, 1),
            hand_invert=True,
            hand_command_alpha=1.0,
            hand_send_threshold=0.002,
        )
        components = build_components(args)
        try:
            self.assertEqual(components.active_sides, ("right",))
            self.assertIsNone(components.dual_arm_retargeter)
            self.assertEqual([retargeter.name for retargeter in components.session.retargeters], ["pose_to_arm_right"])
            self.assertEqual(
                components.session.safety_monitor.config.required_channels,
                ("right_wrist_pose",),
            )
        finally:
            components.session.close()
            components.left_preview_backend.close()
            components.right_preview_backend.close()

    def test_real_components_single_left_arm_use_single_arm_retargeter(self) -> None:
        args = SimpleNamespace(
            active_sides=("left",),
            hz=30.0,
            disable_dual_arm_collision_avoidance=False,
            ik_tuning_json="",
            load_initial_alignment="",
            load_profile="",
            left_right_base_distance=0.525,
            left_tracker_name="tracker_1",
            right_tracker_name="tracker_2",
            aux_tracker_name="",
            control_mode="absolute",
            translation_scale=1.0,
            collision_scene_profile="dual_arm_hand_proxy_open",
            collision_avoidance_mode="boundary_slide",
            recovery_healthy_frames=8,
            no_record=True,
            host="0.0.0.0",
            port=9999,
            tcp_mode="hand_base",
            left_arm_ip="192.168.1.18",
            right_arm_ip="",
            left_hand_ip="",
            right_hand_ip="",
            left_hand_port="",
            right_hand_port="",
            hand_reader_baudrate=115200,
            hand_reader_timeout=0.05,
            show_hand_raw_lines=False,
            hand_joint_order=(5, 4, 3, 2, 0, 1),
            hand_invert=True,
            hand_command_alpha=1.0,
            hand_send_threshold=0.002,
            hand_telemetry_hz=30.0,
            hand_tactile_regions="",
            max_fingertip_force=800,
            teleop_hardware_ip="",
            teleop_hardware_port=0,
            swap_arm_sides=False,
            arm_max_angvel=(0.35, 0.35, 0.35, 0.5, 0.5, 0.5),
            minimum_control_hz=15.0,
            status_sec=1.0,
            skip_renderer=True,
            headless=True,
            keyboard_clutch=False,
            rm65_hardware_read_retries=0,
            rm65_hardware_read_retry_delay_sec=0.02,
            arm_io_threads=2,
            arm_step_dt_scaling=True,
            arm_step_dt_alpha=0.25,
            arm_step_dt_max_scale=2.0,
            arm_step_dt_stall_reset_sec=0.25,
            max_start_joint_error_deg=5.0,
            max_start_weighted_error_deg=12.0,
            ignore_start_mismatch=False,
        )
        components = build_real_components(args)
        try:
            self.assertEqual(components.active_sides, ("left",))
            self.assertIsNone(components.dual_arm_retargeter)
            self.assertIsNotNone(components.left_command_backend)
            self.assertIsNone(components.right_command_backend)
            self.assertEqual(set(components.session.rig.backends), {"left_arm"})
            self.assertEqual([retargeter.name for retargeter in components.session.retargeters], ["pose_to_arm_left"])
            self.assertEqual(
                components.session.safety_monitor.config.required_channels,
                ("left_wrist_pose",),
            )
        finally:
            components.session.close()
            components.left_preview_backend.close()
            components.right_preview_backend.close()
            if components.left_state_backend is not None:
                components.left_state_backend.close()
            if components.right_state_backend is not None:
                components.right_state_backend.close()

    def test_robot_home_prefers_initial_alignment_pose_and_qpos(self) -> None:
        left_pose = _pose(tx=0.33, yaw_deg=10.0)
        right_pose = _pose(tx=-0.27, yaw_deg=-15.0)
        payload = {
            "format": "teleop_initial_alignment",
            "format_version": 1,
            "tcp_frame": "hand_base",
            "sides": {
                "left": {
                    "side": "left",
                    "arm_qpos": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                    "tcp_pose_base": identity_pose().tolist(),
                    "tcp_pose_desk": left_pose.tolist(),
                    "hand_qpos_q6": None,
                    "metadata": {},
                },
                "right": {
                    "side": "right",
                    "arm_qpos": [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1],
                    "tcp_pose_base": identity_pose().tolist(),
                    "tcp_pose_desk": right_pose.tolist(),
                    "hand_qpos_q6": None,
                    "metadata": {},
                },
            },
            "metadata": {},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            alignment_path = f"{tmp_dir}/alignment.json"
            with open(alignment_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            args = SimpleNamespace(
                hz=30.0,
                arm_mode="virtual",
                disable_dual_arm_collision_avoidance=True,
                ik_tuning_json="",
                load_initial_alignment=alignment_path,
                load_profile="",
                left_right_base_distance=0.525,
                left_tracker_name="tracker_1",
                right_tracker_name="tracker_2",
                aux_tracker_name="tracker_3",
                control_mode="absolute",
                translation_scale=1.0,
                collision_scene_profile="dual_arm_hand_proxy_open",
                collision_avoidance_mode="boundary_slide",
                recovery_healthy_frames=8,
                no_record=True,
                host="0.0.0.0",
                port=9999,
                tcp_mode="hand_base",
                left_arm_ip="",
                right_arm_ip="",
                skip_renderer=True,
                headless=True,
            )
            components = build_components(args)
            try:
                left_robot_home_desk, left_qpos, left_source = _resolve_robot_home_for_side(
                    components,
                    side="left",
                )
                right_robot_home_desk, right_qpos, right_source = _resolve_robot_home_for_side(
                    components,
                    side="right",
                )
                np.testing.assert_allclose(left_robot_home_desk, left_pose)
                np.testing.assert_allclose(right_robot_home_desk, right_pose)
                np.testing.assert_allclose(left_qpos, np.asarray([0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=float))
                np.testing.assert_allclose(right_qpos, np.asarray([-0.6, -0.5, -0.4, -0.3, -0.2, -0.1], dtype=float))
                self.assertEqual(left_source, "initial_alignment_pose_desk")
                self.assertEqual(right_source, "initial_alignment_pose_desk")
            finally:
                components.session.close()
                components.left_preview_backend.close()
                components.right_preview_backend.close()

    def test_dual_arm_target_context_uses_desk_pose_with_base_translation(self) -> None:
        args = SimpleNamespace(
            hz=30.0,
            arm_mode="virtual",
            disable_dual_arm_collision_avoidance=False,
            ik_tuning_json="teleop_runtime/configs/rm65_ik_tuning.json",
            load_initial_alignment=(
                "teleop_records/initial_alignment_states/"
                "20260418_154512_rm65_initial_alignment.json"
            ),
            load_profile=(
                "teleop_records/calibration_profiles/"
                "20260416_231341_vive_rm65_virtual_desk.json"
            ),
            left_right_base_distance=0.525,
            left_tracker_name="tracker_1",
            right_tracker_name="tracker_2",
            aux_tracker_name="tracker_3",
            control_mode="absolute",
            translation_scale=1.0,
            collision_scene_profile="dual_arm_hand_proxy_open",
            collision_avoidance_mode="boundary_slide",
            recovery_healthy_frames=8,
            no_record=True,
            host="0.0.0.0",
            port=9999,
            tcp_mode="hand_base",
            left_arm_ip="",
            right_arm_ip="",
            skip_renderer=True,
            headless=True,
            keyboard_clutch=False,
        )
        components = build_components(args)
        try:
            left_human_home = identity_pose()
            left_human_home[0, 3] = -0.25
            left_human_home[1, 3] = -0.15
            left_human_home[2, 3] = 0.32
            right_human_home = identity_pose()
            right_human_home[0, 3] = -0.26
            right_human_home[1, 3] = 0.16
            right_human_home[2, 3] = 0.31

            left_robot_home_desk, _, _ = _resolve_robot_home_for_side(components, side="left")
            right_robot_home_desk, _, _ = _resolve_robot_home_for_side(components, side="right")
            components.left_retargeter.set_home_anchors(
                human_pose_desk=left_human_home,
                robot_pose_desk=left_robot_home_desk,
            )
            components.right_retargeter.set_home_anchors(
                human_pose_desk=right_human_home,
                robot_pose_desk=right_robot_home_desk,
            )

            timestamp = time.time()
            observation = TeleopObservation(
                channels={
                    "left_wrist_pose": ObservationChannel(
                        channel_name="left_wrist_pose",
                        capability="pose6d",
                        side="left",
                        value=left_human_home,
                        frame_name="desk",
                        timestamp=timestamp,
                        source_name="unit_test",
                        valid=True,
                        critical=True,
                    ),
                    "right_wrist_pose": ObservationChannel(
                        channel_name="right_wrist_pose",
                        capability="pose6d",
                        side="right",
                        value=right_human_home,
                        frame_name="desk",
                        timestamp=timestamp,
                        source_name="unit_test",
                        valid=True,
                        critical=True,
                    ),
                },
                timestamp=timestamp,
                metadata={"tcp_mode": HAND_BASE_TCP_FRAME},
            )
            context = components.dual_arm_retargeter.build_target_context(
                observation,
                components.session.calibration,
            )
            current_targets = components.dual_arm_retargeter.current_world_targets()

            np.testing.assert_allclose(
                context.left_target[:3, 3],
                current_targets["left"][:3, 3],
                atol=2e-2,
            )
            np.testing.assert_allclose(
                context.right_target[:3, 3],
                current_targets["right"][:3, 3],
                atol=2e-2,
            )
            left_q_before = components.left_preview_backend.get_state()["current_qpos"].copy()
            right_q_before = components.right_preview_backend.get_state()["current_qpos"].copy()
            command = components.session.build_command(observation)
            np.testing.assert_allclose(command.commands["left_arm"].value, left_q_before, atol=1e-6)
            np.testing.assert_allclose(command.commands["right_arm"].value, right_q_before, atol=1e-6)
        finally:
            components.session.close()
            components.left_preview_backend.close()
            components.right_preview_backend.close()

    def test_dual_arm_single_command_converges_to_requested_rotation_axis(self) -> None:
        args = SimpleNamespace(
            hz=30.0,
            arm_mode="virtual",
            disable_dual_arm_collision_avoidance=False,
            ik_tuning_json="teleop_runtime/configs/rm65_ik_tuning.json",
            load_initial_alignment=(
                "teleop_records/initial_alignment_states/"
                "20260418_154512_rm65_initial_alignment.json"
            ),
            load_profile=(
                "teleop_records/calibration_profiles/"
                "20260416_231341_vive_rm65_virtual_desk.json"
            ),
            left_right_base_distance=0.525,
            left_tracker_name="tracker_1",
            right_tracker_name="tracker_2",
            aux_tracker_name="tracker_3",
            control_mode="absolute",
            translation_scale=1.0,
            collision_scene_profile="dual_arm_hand_proxy_open",
            collision_avoidance_mode="boundary_slide",
            recovery_healthy_frames=8,
            no_record=True,
            host="0.0.0.0",
            port=9999,
            tcp_mode="hand_base",
            left_arm_ip="",
            right_arm_ip="",
            skip_renderer=True,
            headless=True,
            keyboard_clutch=False,
        )
        components = build_components(args)
        try:
            home_rot = R.from_euler("xyz", [20.0, -35.0, 40.0], degrees=True).as_matrix()
            left_human_home = identity_pose()
            left_human_home[:3, :3] = home_rot
            left_human_home[:3, 3] = np.asarray([-0.25, -0.15, 0.32], dtype=float)
            right_human_home = identity_pose()
            right_human_home[:3, :3] = home_rot
            right_human_home[:3, 3] = np.asarray([-0.26, 0.16, 0.31], dtype=float)
            left_robot_home_desk, _, _ = _resolve_robot_home_for_side(components, side="left")
            right_robot_home_desk, _, _ = _resolve_robot_home_for_side(components, side="right")
            components.left_retargeter.set_home_anchors(
                human_pose_desk=left_human_home,
                robot_pose_desk=left_robot_home_desk,
            )
            components.right_retargeter.set_home_anchors(
                human_pose_desk=right_human_home,
                robot_pose_desk=right_robot_home_desk,
            )

            delta_rot = R.from_euler("xyz", [15.0, 0.0, 0.0], degrees=True).as_matrix()
            left_current = left_human_home.copy()
            left_current[:3, :3] = left_human_home[:3, :3] @ delta_rot
            right_current = right_human_home.copy()
            right_current[:3, :3] = right_human_home[:3, :3] @ delta_rot
            timestamp = time.time()
            observation = TeleopObservation(
                channels={
                    "left_wrist_pose": ObservationChannel(
                        channel_name="left_wrist_pose",
                        capability="pose6d",
                        side="left",
                        value=left_current,
                        frame_name="desk",
                        timestamp=timestamp,
                        source_name="unit_test",
                        valid=True,
                        critical=True,
                    ),
                    "right_wrist_pose": ObservationChannel(
                        channel_name="right_wrist_pose",
                        capability="pose6d",
                        side="right",
                        value=right_current,
                        frame_name="desk",
                        timestamp=timestamp,
                        source_name="unit_test",
                        valid=True,
                        critical=True,
                    ),
                },
                timestamp=timestamp,
                metadata={"tcp_mode": HAND_BASE_TCP_FRAME},
            )
            command = components.session.build_command(observation)
            components.left_preview_backend.send_command(command.commands["left_arm"])
            components.right_preview_backend.send_command(command.commands["right_arm"])
            achieved = _preview_tcp_pose_desk(components, side="left")
            achieved_delta_deg = R.from_matrix(
                left_robot_home_desk[:3, :3].T @ achieved[:3, :3]
            ).as_euler("xyz", degrees=True)
            expected_delta_deg = R.from_matrix(
                tracker_hand_to_rm65_tcp_rotation().T
                @ delta_rot
                @ tracker_hand_to_rm65_tcp_rotation()
            ).as_euler("xyz", degrees=True)
            np.testing.assert_allclose(
                achieved_delta_deg,
                expected_delta_deg,
                atol=1.0,
            )
            self.assertGreaterEqual(command.metadata.get("ik_solve_iterations", 0), 1)
        finally:
            components.session.close()
            components.left_preview_backend.close()
            components.right_preview_backend.close()

    def test_legacy_tcp_mode_normalizes_to_hand_base(self) -> None:
        config = TeleopSessionConfig(record_enabled=False, tcp_mode="rm_eef")
        self.assertEqual(config.tcp_mode, HAND_BASE_TCP_FRAME)

    def test_legacy_calibration_payload_loads_as_hand_base_only(self) -> None:
        pose = _pose(tx=0.12, yaw_deg=15.0)
        calibration = TeleopCalibration.from_dict(
            {
                "arms": {
                    "left_arm": {
                        "name": "left_arm",
                        "side": "left",
                        "T_D_B": identity_pose().tolist(),
                        "workspace_min": None,
                        "workspace_max": None,
                        "home_qpos": None,
                        "home_pose_debug": None,
                        "home_pose_fine": pose.tolist(),
                        "tcp_debug_frame": "rm_eef",
                        "tcp_fine_frame": "hand_base",
                        "metadata": {},
                    }
                }
            }
        )
        self.assertEqual(calibration.target_frame_name("left", "rm_eef"), HAND_BASE_TCP_FRAME)
        np.testing.assert_allclose(
            calibration.get_arm_for_side("left").home_pose_for_tcp_mode("rm_eef"),
            pose,
        )
        serialized_arm = calibration.to_dict()["arms"]["left_arm"]
        self.assertEqual(serialized_arm["tcp_frame"], HAND_BASE_TCP_FRAME)
        self.assertNotIn("tcp_debug_frame", serialized_arm)
        self.assertNotIn("tcp_fine_frame", serialized_arm)

    def test_legacy_tracker_offset_profile_is_preserved(self) -> None:
        calibration = TeleopCalibration.from_dict(
            {
                "T_W_O": identity_pose().tolist(),
                "T_O_D": identity_pose().tolist(),
                "tracker_offsets": {
                    "left_wrist": identity_pose().tolist(),
                },
                "metadata": {
                    "calibration_mode": "vive_rm65_virtual_desk",
                    "virtual_desk_sweep_convention_version": 2,
                    "virtual_desk_post_yaw_deg": 0.0,
                    "tracker_offset_calibration": {
                        "left_wrist": {
                            "canonical_hand_rpy_deg": [0.0, 0.0, 0.0],
                            "coarse_offset": identity_pose().tolist(),
                            "fine_tune_rpy_deg": [1.0, 2.0, 3.0],
                            "final_offset": identity_pose().tolist(),
                        }
                    },
                },
            }
        )
        np.testing.assert_allclose(
            calibration.get_tracker_offset("left_wrist"),
            identity_pose(),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            calibration.get_tracker_offset("left_wrist")[:3, :3],
            canonical_tracker_to_hand_rotation(),
            atol=1e-9,
        )
        tracker_info = calibration.metadata["tracker_offset_calibration"]["left_wrist"]
        np.testing.assert_allclose(
            np.asarray(tracker_info["canonical_hand_rpy_deg"], dtype=float),
            DEFAULT_CANONICAL_HAND_RPY_DEG,
            atol=1e-9,
        )
        self.assertNotIn("compatibility_auto_corrected", tracker_info)
        self.assertNotIn("tracker_offset_compatibility_upgrades", calibration.metadata)

    def test_virtual_desk_profile_defaults_scene_alignment_to_180_yaw(self) -> None:
        calibration = TeleopCalibration.from_dict(
            {
                "T_W_O": identity_pose().tolist(),
                "T_O_D": identity_pose().tolist(),
                "tracker_offsets": {
                    "left_wrist": identity_pose().tolist(),
                },
                "metadata": {
                    "calibration_mode": "vive_rm65_virtual_desk",
                    "virtual_desk": {
                        "T_W_V": identity_pose().tolist(),
                        "x_axis_world": [1.0, 0.0, 0.0],
                        "y_axis_world": [0.0, 1.0, 0.0],
                        "z_axis_world": [0.0, 0.0, 1.0],
                    },
                    "tracker_offset_calibration": {
                        "left_wrist": {
                            "final_offset": identity_pose().tolist(),
                            "coarse_offset": identity_pose().tolist(),
                            "fine_tune_rpy_deg": [1.0, 0.0, 0.0],
                        }
                    },
                },
            }
        )
        expected = identity_pose()
        expected[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        np.testing.assert_allclose(calibration.world_to_desk_pose(), expected, atol=1e-9)
        np.testing.assert_allclose(calibration.get_tracker_offset("left_wrist"), expected, atol=1e-9)
        self.assertTrue(bool(calibration.metadata["virtual_desk_scene_alignment_defaulted"]))
        self.assertEqual(calibration.metadata["virtual_desk_post_yaw_deg"], 180.0)
        self.assertEqual(calibration.metadata["virtual_desk_sweep_convention_version"], 2)

    def test_pose_to_arm_applies_tracker_to_tcp_rotation_offset(self) -> None:
        calibration = build_annotated_dual_rm65_calibration()
        retargeter = PoseToArmRetargeter(
            PoseToArmRetargeterConfig(
                side="left",
                input_channel_name="left_wrist_pose",
                effector_name="left_arm",
                control_mode="absolute",
                input_local_rotation_offset=tracker_hand_to_rm65_tcp_rotation(),
            )
        )
        observation = TeleopObservation(
            channels={
                "left_wrist_pose": ObservationChannel(
                    channel_name="left_wrist_pose",
                    capability="pose",
                    side="left",
                    value=identity_pose(),
                    frame_name="desk",
                    timestamp=0.0,
                    source_name="test",
                    metadata={"tracker_role": "left_wrist"},
                )
            },
            timestamp=0.0,
        )
        command = retargeter.retarget(observation, calibration)
        target_pose_desk = np.asarray(
            command.commands["left_arm"].metadata["target_pose_desk"],
            dtype=float,
        )
        np.testing.assert_allclose(
            target_pose_desk[:3, :3],
            tracker_hand_to_rm65_tcp_rotation(),
            atol=1e-9,
        )

    def test_session_auto_recovery_enters_recovering_after_healthy_frames(self) -> None:
        source = StubPoseSource()
        session = TeleopSession(
            config=TeleopSessionConfig(
                record_enabled=False,
                allow_hardware_after_debug_pass=False,
                recovery_healthy_frame_count=2,
            ),
            sources=[source],
            retargeters=[],
            safety_config=TeleopSafetyConfig(
                freeze_on_fault=True,
                required_channels=("left_wrist_pose", "right_wrist_pose"),
                invalid_frame_limit=1,
            ),
        )
        try:
            session.start()
            session.set_calibration_ready()
            session.mark_debug_passed()
            session.arm_hardware()

            source.timestamp = 0.0
            step = session.run_step(now=0.0)
            self.assertEqual(step.state, TeleopRunState.ARMED)

            source.timestamp = 0.1
            source.right_valid = False
            step = session.run_step(now=0.1)
            self.assertEqual(step.state, TeleopRunState.FROZEN)
            self.assertEqual([fault.code for fault in step.faults], ["invalid_channel"])

            source.timestamp = 0.2
            source.right_valid = True
            source.right_pose = _pose(tx=0.30, yaw_deg=95.0)
            step = session.run_step(now=0.2)
            self.assertEqual(step.state, TeleopRunState.FROZEN)
            self.assertEqual(step.faults, [])

            source.timestamp = 0.3
            step = session.run_step(now=0.3)
            self.assertEqual(step.state, TeleopRunState.RECOVERING)
            self.assertIsNotNone(step.observation)

            session.finish_recovery()
            source.timestamp = 0.4
            step = session.run_step(now=0.4)
            self.assertEqual(step.state, TeleopRunState.ARMED)
        finally:
            session.close()

    def test_safety_monitor_resets_pose_baseline_after_invalid_channel(self) -> None:
        monitor = SafetyMonitor(
            TeleopSafetyConfig(
                max_translation_jump_m=0.10,
                max_rotation_jump_deg=45.0,
                invalid_frame_limit=1,
                required_channels=("right_wrist_pose",),
            )
        )

        valid0 = {
            "right_wrist_pose": ObservationChannel(
                channel_name="right_wrist_pose",
                capability="pose6d",
                side="right",
                value=_pose(),
                frame_name="desk",
                timestamp=0.0,
                source_name="stub",
                valid=True,
                critical=True,
            )
        }
        self.assertEqual(monitor.evaluate_channels(valid0, 0.0), [])

        invalid = {
            "right_wrist_pose": ObservationChannel(
                channel_name="right_wrist_pose",
                capability="pose6d",
                side="right",
                value=_pose(),
                frame_name="desk",
                timestamp=0.1,
                source_name="stub",
                valid=False,
                critical=True,
            )
        }
        violations = monitor.evaluate_channels(invalid, 0.1)
        self.assertEqual([violation.code for violation in violations], ["invalid_channel"])

        recovered = {
            "right_wrist_pose": ObservationChannel(
                channel_name="right_wrist_pose",
                capability="pose6d",
                side="right",
                value=_pose(tx=0.30, yaw_deg=95.0),
                frame_name="desk",
                timestamp=0.2,
                source_name="stub",
                valid=True,
                critical=True,
            )
        }
        self.assertEqual(monitor.evaluate_channels(recovered, 0.2), [])

        jumped = {
            "right_wrist_pose": ObservationChannel(
                channel_name="right_wrist_pose",
                capability="pose6d",
                side="right",
                value=_pose(tx=0.65, yaw_deg=155.0),
                frame_name="desk",
                timestamp=0.3,
                source_name="stub",
                valid=True,
                critical=True,
            )
        }
        violations = monitor.evaluate_channels(jumped, 0.3)
        self.assertEqual(
            [violation.code for violation in violations],
            ["translation_jump", "rotation_jump"],
        )


if __name__ == "__main__":
    unittest.main()
