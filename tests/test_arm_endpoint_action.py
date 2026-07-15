from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

import numpy as np

from robot_motion.runtime import BackendDispatchError

from hardware.rm65.calibration_profiles import is_position_endpoint_profile, load_profile
from teleop_runtime.arm_frames import (
    ARM_ENDPOINT_FRAME,
    EMPTY_TOOL_FRAME,
    TOOL_TCP_FRAME,
    canonical_arm_controlled_frame,
)
from teleop_runtime.backends.arms import (
    RM65ArmBackend,
    RM65ArmBackendConfig,
    RM65ImpedanceArmBackend,
    RM65ImpedanceArmBackendConfig,
)
from teleop_runtime.calibration import ArmCalibration, TeleopCalibration
from teleop_runtime.defaults import DEFAULT_RM65_POSITION_ARM_CONTROL_HZ
from teleop_runtime.retargeters.pose_to_arm import PoseToArmRetargeter, PoseToArmRetargeterConfig
from teleop_runtime.schemas import EffectorCommand, ObservationChannel, TeleopObservation


class _SingularityFakeArm:
    def __init__(self):
        self.sent_joint_deg: list[list[float]] = []
        self.follow_values: list[bool] = []
        self.movej_options: list[dict[str, int | bool]] = []
        self.thresholds: tuple[float, float, float] | None = None
        self.stop_calls = 0
        self.slow_stop_calls = 0

    def Algo_Kin_Set_Singularity_Thresholds(self, limit_qe, limit_qw, limit_d):
        self.thresholds = (float(limit_qe), float(limit_qw), float(limit_d))

    def Algo_Kin_Get_Singularity_Thresholds(self):
        return self.thresholds or (3.0, 3.0, 0.05)

    def Algo_Kin_Singularity_Analyse(self, joint_deg):
        joint = np.asarray(joint_deg, dtype=float).reshape(6)
        if abs(float(joint[2])) < 3.0:
            return -2, 0.42
        return 0, 0.42

    def Movej_CANFD(self, joint_deg, follow=False, trajectory_mode=0, radio=0):
        self.sent_joint_deg.append([float(value) for value in joint_deg])
        self.follow_values.append(bool(follow))
        self.movej_options.append(
            {
                "follow": bool(follow),
                "trajectory_mode": int(trajectory_mode),
                "radio": int(radio),
            }
        )
        return 0

    def Arm_Stop(self):
        self.stop_calls += 1
        return 0

    def Arm_Slow_Stop(self):
        self.slow_stop_calls += 1
        return 0


class _PositionTransportFakeArm:
    def __init__(
        self,
        *,
        movej_results: list[object] | None = None,
        movej_exception: Exception | None = None,
        measured_joint_deg: np.ndarray | None = None,
    ) -> None:
        self.movej_results = list(movej_results or [0])
        self.movej_exception = movej_exception
        self.measured_joint_deg = np.asarray(
            np.zeros(6) if measured_joint_deg is None else measured_joint_deg,
            dtype=float,
        ).reshape(6)
        self.sent_joint_deg: list[np.ndarray] = []
        self.slow_stop_calls = 0
        self.uninit_calls = 0
        self.socket_close_calls = 0

    def Movej_CANFD(self, joint_deg, follow=False, trajectory_mode=0, radio=0):
        del follow, trajectory_mode, radio
        self.sent_joint_deg.append(np.asarray(joint_deg, dtype=float).reshape(6).copy())
        if self.movej_exception is not None:
            raise self.movej_exception
        if not self.movej_results:
            return 0
        if len(self.movej_results) == 1:
            return self.movej_results[0]
        return self.movej_results.pop(0)

    def Get_Current_Arm_State(self):
        return 0, self.measured_joint_deg.tolist(), [0.0] * 6, 0, 0

    def Arm_Slow_Stop(self):
        self.slow_stop_calls += 1
        return 0

    def RM_API_UnInit(self):
        self.uninit_calls += 1

    def Arm_Socket_Close(self):
        self.socket_close_calls += 1


class _ToolSyncFakeArm:
    def __init__(self):
        self.current_tool = "empty"
        self.tool_pose = [0.0] * 6
        self.frames: dict[str, tuple[list[float], float, list[float]]] = {}
        self.updated: list[tuple[str, list[float], float, list[float]]] = []
        self.created: list[tuple[str, list[float], float, list[float]]] = []
        self.changed: list[str] = []
        self.stop_calls = 0
        self.slow_stop_calls = 0

    def Arm_Socket_State(self):
        return 0

    def Get_Current_Tool_Frame(self):
        return 0, {"name": self.current_tool, "pose": list(self.tool_pose)}

    def Update_Tool_Frame(self, name, pose, payload, x, y, z):
        values = (str(name), [float(value) for value in pose], float(payload), [float(x), float(y), float(z)])
        self.updated.append(values)
        self.frames[str(name)] = (values[1], values[2], values[3])
        return 0

    def Manual_Set_Tool_Frame(self, name, pose, payload, x, y, z):
        values = (str(name), [float(value) for value in pose], float(payload), [float(x), float(y), float(z)])
        self.created.append(values)
        self.frames[str(name)] = (values[1], values[2], values[3])
        return 0

    def Change_Tool_Frame(self, name):
        self.changed.append(str(name))
        frame = self.frames.get(str(name))
        if frame is not None:
            self.current_tool = str(name)
            self.tool_pose = list(frame[0])
        return 0

    def Arm_Slow_Stop(self):
        self.slow_stop_calls += 1
        return 0

    def Arm_Stop(self):
        self.stop_calls += 1
        return 0


class _StopMotionTrapController:
    def __init__(self, robot):
        self.robot = robot
        self.stop_motion_calls = 0
        self.close_calls = 0

    def stop_motion(self):
        self.stop_motion_calls += 1
        raise AssertionError("emergency_stop must not use controller.stop_motion")

    def close(self):
        self.close_calls += 1


def _dmtac_impedance_profile() -> dict[str, object]:
    return {
        "data_type": "rm65_calibration_profile",
        "name": "DMTAC_L",
        "arm": "right",
        "tool_name": "DMTAC_L",
        "tcp": {
            "pose_flange_m_rad": [0.02, 0.0, 0.18, 0.0, 0.0, 0.0],
            "confirmed": True,
        },
        "payload": {
            "mass_kg": 0.657,
            "com_tool_m": [0.011, -0.006, 0.104],
        },
        "impedance": {
            "pose": {
                "linear_compliance_m_per_n": 0.005,
            },
        },
    }


class ArmEndpointActionTest(unittest.TestCase):
    def test_rm65_position_backend_prefers_arm_link6_for_endpoint_commands(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
            )
        )

        self.assertEqual(backend._candidate_tcp_frames(ARM_ENDPOINT_FRAME)[0], "arm_link6")

    def test_rm65_position_backend_defaults_to_robot_motion_ik_framework(self) -> None:
        config = RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
        )

        self.assertEqual(config.ik_framework, "robot_motion")

    def test_rm65_position_backend_defaults_to_canfd_follow_150hz(self) -> None:
        config = RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
        )

        self.assertTrue(config.canfd_follow)
        self.assertEqual(config.background_streaming_hz, DEFAULT_RM65_POSITION_ARM_CONTROL_HZ)

    def test_rm65_position_backend_sends_canfd_follow_from_config(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                initial_qpos=np.zeros(6),
                max_angvel=(100.0,) * 6,
            )
        )
        fake_arm = _SingularityFakeArm()
        backend.arm = fake_arm

        backend.send_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.deg2rad(np.asarray([1.0, 2.0, 20.0, 4.0, 5.0, 6.0])),
            )
        )

        self.assertEqual(fake_arm.follow_values[-1], True)

        backend.config.canfd_follow = False
        backend.send_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.deg2rad(np.asarray([2.0, 3.0, 21.0, 5.0, 6.0, 7.0])),
            )
        )

        self.assertEqual(fake_arm.follow_values[-1], False)

    def test_rm65_position_backend_rejects_legacy_ik_framework(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported RM65 IK framework"):
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                ik_framework="legacy",
            )

    def test_rm65_position_backend_robot_motion_framework_uses_shared_builder(self) -> None:
        sentinel_solver = object()
        with mock.patch(
            "teleop_runtime.backends.arms.rm65.build_rm65_single_arm_ik_solver",
            return_value=sentinel_solver,
        ) as build_solver:
            backend = RM65ArmBackend(
                RM65ArmBackendConfig(
                    arm_ip="0.0.0.0",
                    side="left",
                    connect_hardware=False,
                )
            )
            solver = backend._solver_for_tcp_frame(TOOL_TCP_FRAME)

        self.assertIs(solver, sentinel_solver)
        build_solver.assert_called_once()
        self.assertEqual(build_solver.call_args.args[0].controlled_frame, TOOL_TCP_FRAME)

    def test_rm65_position_reset_motion_state_uses_live_qpos_and_clears_targets(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                initial_qpos=np.zeros(6),
            )
        )
        stale_target = np.asarray([0.5, 0.4, 0.3, 0.2, 0.1, -0.1], dtype=float)
        backend._stream_target_qpos = stale_target.copy()
        backend._last_target_qpos = stale_target.copy()
        backend._last_command = EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=stale_target,
        )

        live_qpos = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=float)
        backend.reset_motion_state(live_qpos)
        state = backend.get_state()

        np.testing.assert_allclose(state["current_qpos"], live_qpos)
        self.assertIsNone(state["last_target_qpos"])
        self.assertIsNone(state["last_command"])
        self.assertIsNone(backend._stream_target_qpos)

    def test_rm65_hold_motion_clears_background_stream_target(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                initial_qpos=np.zeros(6),
            )
        )
        backend._stream_target_qpos = np.ones(6)
        backend._last_target_qpos = np.ones(6)

        backend.hold_motion("unit test")
        state = backend.get_state()

        self.assertIsNone(backend._stream_target_qpos)
        self.assertIsNone(state["last_target_qpos"])

    def test_rm65_joint_vector_prelimited_solver_qpos_bypasses_backend_step_clip(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                dry_run_fast=False,
                control_dt=0.025,
                max_angvel=(0.1,) * 6,
                initial_qpos=np.zeros(6),
            )
        )
        target = np.asarray([0.2, -0.2, 0.1, -0.1, 0.05, -0.05], dtype=float)
        command = EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=target,
            metadata={
                "action_pose_source": "limited_solver_output_qpos",
                "execution_step_metadata": {"enabled": True},
            },
        )

        backend.send_command(command)

        np.testing.assert_allclose(backend.get_state()["current_qpos"], target)
        np.testing.assert_allclose(command.value, target)
        np.testing.assert_allclose(command.metadata["solver_output_qpos"], target)
        np.testing.assert_allclose(command.metadata["backend_final_target_qpos"], target)
        self.assertEqual(command.metadata["action_pose_source"], "non_impedance_final_qpos")
        try:
            __import__("pinocchio")
        except ModuleNotFoundError:
            self.assertFalse(command.metadata["target_pose_available"])
            self.assertEqual(command.metadata["target_pose_error_type"], "ModuleNotFoundError")
            self.assertIn("pinocchio", command.metadata["target_pose_error"])
        else:
            self.assertTrue(command.metadata["target_pose_available"])
            self.assertIn("target_pose_base", command.metadata)
            self.assertIn("target_pose_tcp_base", command.metadata)

    def test_rm65_background_stream_interpolates_to_15hz_target_without_step_clip(self) -> None:
        start = np.zeros(6, dtype=float)
        target = np.asarray([0.2, -0.2, 0.1, -0.1, 0.05, -0.05], dtype=float)

        halfway = RM65ArmBackend._interpolate_stream_qpos(
            start_qpos=start,
            target_qpos=target,
            segment_start_time=10.0,
            segment_duration_sec=1.0 / 15.0,
            now=10.0 + 0.5 / 15.0,
        )
        done = RM65ArmBackend._interpolate_stream_qpos(
            start_qpos=start,
            target_qpos=target,
            segment_start_time=10.0,
            segment_duration_sec=1.0 / 15.0,
            now=10.0 + 1.0 / 15.0,
        )

        np.testing.assert_allclose(halfway, target * 0.5)
        np.testing.assert_allclose(done, target)

    def test_rm65_background_stream_records_low_rate_warning(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                background_streaming_enabled=True,
                background_streaming_hz=120.0,
                background_streaming_warning_ratio=0.9,
                initial_qpos=np.zeros(6),
            )
        )

        backend._record_background_stream_send(
            qpos_rad=np.zeros(6),
            sent_at_perf=100.0,
            call_ms=0.2,
            deadline_lag_ms=0.0,
        )
        backend._record_background_stream_send(
            qpos_rad=np.ones(6) * np.deg2rad(1.0),
            sent_at_perf=100.02,
            call_ms=0.3,
            deadline_lag_ms=10.0,
        )

        stream = backend.get_state()["background_streaming"]

        self.assertEqual(stream["send_count"], 2)
        self.assertAlmostEqual(stream["actual_hz"], 50.0)
        self.assertEqual(stream["low_rate_count"], 1)
        self.assertEqual(stream["deadline_miss_count"], 1)
        self.assertTrue(stream["warning"]["active"])
        self.assertIn("below target", stream["warning"]["message"])
        self.assertAlmostEqual(stream["warning_threshold_hz"], 108.0)
        self.assertAlmostEqual(stream["last_sent_step_deg"], 1.0)

    def test_rm65_position_movej_canfd_passes_follow_trajectory_options(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                canfd_follow=True,
                canfd_trajectory_mode=2,
                canfd_radio=30,
                initial_qpos=np.zeros(6),
            )
        )
        fake_arm = _SingularityFakeArm()
        backend.arm = fake_arm

        backend._send_joint_target(np.zeros(6))

        self.assertEqual(
            fake_arm.movej_options[-1],
            {"follow": True, "trajectory_mode": 2, "radio": 30},
        )

    def test_rm65_position_transport_keeps_command_and_measured_state_separate(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                initial_qpos=np.zeros(6),
                singularity_protection_enabled=False,
            )
        )
        fake_arm = _PositionTransportFakeArm(measured_joint_deg=np.full(6, 10.0))
        backend.arm = fake_arm
        accepted = np.asarray([0.1, -0.1, 0.2, -0.2, 0.15, -0.15], dtype=float)

        backend.send_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=accepted,
            )
        )
        backend.refresh_state_from_hardware(hardware_read_retries=0)
        state = backend.get_state()

        np.testing.assert_allclose(state["last_attempted_qpos"], accepted)
        np.testing.assert_allclose(state["last_accepted_qpos"], accepted)
        np.testing.assert_allclose(state["current_qpos"], accepted)
        np.testing.assert_allclose(state["measured_qpos"], np.deg2rad(np.full(6, 10.0)))
        np.testing.assert_allclose(state["hardware_qpos"], state["measured_qpos"])
        self.assertEqual(state["last_attempted_sequence"], 1)
        self.assertEqual(state["last_accepted_sequence"], 1)
        self.assertEqual(state["measured_sequence"], 1)

        backend.config.background_streaming_enabled = True
        next_target = accepted + 0.05
        backend.send_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=next_target,
            )
        )
        np.testing.assert_allclose(backend._stream_segment_start_qpos, accepted)
        planner_tick = backend._stream_planner.step(
            monotonic_time=backend._stream_target_monotonic,
        )
        self.assertEqual(
            planner_tick.reference_sequence,
            backend._stream_target_sequence,
        )
        np.testing.assert_allclose(
            planner_tick.q_cmd_by_side["left"],
            accepted,
        )

    def test_rm65_position_transport_nonzero_sdk_result_does_not_advance_accepted_state(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                initial_qpos=np.zeros(6),
                singularity_protection_enabled=False,
            )
        )
        backend.arm = _PositionTransportFakeArm(movej_results=[7])
        target = np.asarray([0.1, -0.1, 0.2, -0.2, 0.15, -0.15], dtype=float)

        with self.assertRaises(BackendDispatchError) as caught:
            backend.send_command(
                EffectorCommand(
                    effector_name="left_arm",
                    side="left",
                    command_type="joint_vector",
                    value=target,
                )
            )
        state = backend.get_state()

        self.assertEqual(caught.exception.receipt.result_code, 7)
        self.assertFalse(caught.exception.receipt.accepted)
        np.testing.assert_allclose(state["last_attempted_qpos"], target)
        np.testing.assert_allclose(state["last_accepted_qpos"], np.zeros(6))
        np.testing.assert_allclose(state["current_qpos"], np.zeros(6))
        self.assertEqual(state["last_attempted_sequence"], 1)
        self.assertEqual(state["last_accepted_sequence"], 0)
        self.assertEqual(state["send_result_code"], 7)

    def test_rm65_position_transport_rejects_nonfinite_and_absolute_limit_targets(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                initial_qpos=np.zeros(6),
                singularity_protection_enabled=False,
            )
        )
        fake_arm = _PositionTransportFakeArm()
        backend.arm = fake_arm

        for target in (
            np.asarray([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.asarray([0.0, 1.7, 0.0, 0.0, 0.0, 0.0]),
        ):
            with self.assertRaisesRegex(ValueError, "non-finite|absolute limits"):
                backend.send_command(
                    EffectorCommand(
                        effector_name="left_arm",
                        side="left",
                        command_type="joint_vector",
                        value=target,
                    )
                )

        self.assertEqual(fake_arm.sent_joint_deg, [])

    def test_rm65_background_stream_latches_sdk_exception_without_accepting_target(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                initial_qpos=np.zeros(6),
                singularity_protection_enabled=False,
                background_streaming_enabled=True,
                background_streaming_hz=150.0,
                background_streaming_target_max_age_sec=1.0,
            )
        )
        backend.arm = _PositionTransportFakeArm(movej_exception=RuntimeError("sdk exploded"))
        backend.send_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.full(6, 0.1),
            )
        )

        backend._background_stream_loop()
        state = backend.get_state()
        stream = state["background_streaming"]

        self.assertIn("sdk exploded", stream["fault"])
        self.assertEqual(stream["fault_count"], 1)
        self.assertEqual(stream["attempted_send_count"], 1)
        self.assertEqual(stream["accepted_send_count"], 0)
        self.assertEqual(stream["failed_send_count"], 1)
        self.assertFalse(stream["target_active"])
        self.assertFalse(stream["healthy"])
        np.testing.assert_allclose(state["current_qpos"], np.zeros(6))

    def test_rm65_close_stops_an_arm_that_received_a_motion_command_before_disconnect(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                initial_qpos=np.zeros(6),
                singularity_protection_enabled=False,
                background_streaming_enabled=False,
            )
        )
        fake_arm = _PositionTransportFakeArm()
        backend.arm = fake_arm
        backend.send_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.full(6, 0.01),
            )
        )

        backend.close()

        self.assertEqual(fake_arm.slow_stop_calls, 1)
        self.assertEqual(fake_arm.uninit_calls, 1)
        self.assertEqual(fake_arm.socket_close_calls, 1)

    def test_rm65_background_stream_expires_stale_target_before_sdk_send(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                initial_qpos=np.zeros(6),
                singularity_protection_enabled=False,
                background_streaming_enabled=True,
                background_streaming_target_max_age_sec=0.1,
            )
        )
        fake_arm = _PositionTransportFakeArm()
        backend.arm = fake_arm
        backend.send_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.full(6, 0.1),
            )
        )
        with backend._state_lock:
            backend._stream_target_monotonic = 10.0

        self.assertTrue(backend._expire_stale_stream_target(now_monotonic=10.2))
        stream = backend.get_state()["background_streaming"]

        self.assertFalse(stream["target_active"])
        self.assertEqual(stream["stale_count"], 1)
        self.assertAlmostEqual(stream["last_stale_age_sec"], 0.2)
        self.assertEqual(fake_arm.sent_joint_deg, [])

    def test_rm65_hold_is_a_hard_boundary_for_background_sends(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                initial_qpos=np.zeros(6),
                singularity_protection_enabled=False,
                background_streaming_enabled=True,
                background_streaming_hz=200.0,
                background_streaming_target_max_age_sec=1.0,
            )
        )
        fake_arm = _PositionTransportFakeArm()
        backend.arm = fake_arm
        backend._stream_thread = threading.Thread(target=backend._background_stream_loop, daemon=True)
        backend._stream_thread.start()
        backend.send_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.full(6, 0.1),
            )
        )
        deadline = time.monotonic() + 0.2
        while not fake_arm.sent_joint_deg and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertTrue(fake_arm.sent_joint_deg)

        backend.hold_motion("test")
        send_count_after_hold = len(fake_arm.sent_joint_deg)
        time.sleep(0.02)

        self.assertEqual(len(fake_arm.sent_joint_deg), send_count_after_hold)
        backend.close()

    def test_rm65_close_reports_background_thread_join_timeout(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                initial_qpos=np.zeros(6),
                background_streaming_enabled=True,
                background_streaming_shutdown_timeout_sec=0.01,
            )
        )
        release = threading.Event()
        backend._stream_thread = threading.Thread(target=release.wait, daemon=True)
        backend._stream_thread.start()

        with self.assertRaisesRegex(RuntimeError, "did not stop"):
            backend.close()
        stream = backend.get_state()["background_streaming"]
        self.assertTrue(stream["join_timed_out"])
        self.assertIn("shutdown timeout", stream["fault"])

        release.set()
        backend._stream_thread.join(timeout=0.2)
        backend.close()

    def test_empty_frame_aliases_to_arm_endpoint(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
            )
        )

        self.assertEqual(canonical_arm_controlled_frame(EMPTY_TOOL_FRAME), ARM_ENDPOINT_FRAME)
        self.assertEqual(backend._candidate_tcp_frames(EMPTY_TOOL_FRAME)[0], "arm_link6")

    def test_rm65_position_backend_treats_empty_tool_profile_as_builtin_endpoint(self) -> None:
        with mock.patch("hardware.rm65.calibration_profiles.load_profile") as load_profile:
            config = RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="right",
                connect_hardware=False,
                tool_profile=EMPTY_TOOL_FRAME,
            )
        load_profile.assert_not_called()
        backend = RM65ArmBackend(config)

        state = backend.get_state()

        self.assertEqual(state["tool_tcp_source"], "builtin_empty_arm_endpoint")
        np.testing.assert_allclose(state["tool_tcp_pose_flange_m_rad"], np.zeros(6))
        self.assertFalse(bool(state["active_tool_profile_path"]))

    def test_position_endpoint_profile_is_position_only_endpoint_tcp(self) -> None:
        profile = load_profile("position_endpoint", arm="left")

        self.assertTrue(is_position_endpoint_profile(profile))
        self.assertEqual(profile["control_interface"]["mode"], "position")
        self.assertEqual(profile["control_interface"]["controlled_frame"], ARM_ENDPOINT_FRAME)
        self.assertFalse(profile["control_interface"]["uses_force_control"])
        self.assertFalse(profile["control_interface"]["uses_payload_compensation"])
        np.testing.assert_allclose(profile["tcp"]["pose_flange_m_rad"], np.zeros(6))
        self.assertEqual(profile["payload"], {})
        self.assertEqual(profile["impedance"], {})

    def test_rm65_position_backend_loads_position_endpoint_profile_without_auto_profile(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                tool_profile="position_endpoint",
            )
        )

        state = backend.get_state()

        self.assertEqual(state["tool_tcp_source"], "position_endpoint_profile")
        np.testing.assert_allclose(state["tool_tcp_pose_flange_m_rad"], np.zeros(6))
        self.assertFalse(backend.config.auto_load_tool_profile)

    def test_impedance_backend_rejects_position_endpoint_profile(self) -> None:
        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                tool_profile="position_endpoint",
            )
        )
        profile = load_profile("position_endpoint", arm="left")

        with self.assertRaisesRegex(RuntimeError, "position-only profile"):
            backend._validate_profile(profile, active_tool_name=None)

    def test_impedance_backend_auto_syncs_profile_tool_frame_before_strict_validation(self) -> None:
        fake_arm = _ToolSyncFakeArm()

        with (
            mock.patch(
                "teleop_runtime.backends.arms.rm65_impedance._load_rm65_sdk",
                return_value=(lambda _sdk, _ip: fake_arm, object),
            ),
            mock.patch(
                "teleop_runtime.backends.arms.rm65_impedance.load_profile",
                return_value=_dmtac_impedance_profile(),
            ),
        ):
            backend = RM65ImpedanceArmBackend(
                RM65ImpedanceArmBackendConfig(
                    arm_ip="0.0.0.0",
                    side="right",
                    connect_hardware=True,
                    tool_profile="DMTAC_L",
                )
            )
            backend.start()

        self.assertEqual(fake_arm.changed, ["DMTAC_L"])
        self.assertEqual(fake_arm.current_tool, "DMTAC_L")
        self.assertEqual(fake_arm.updated[0][0], "DMTAC_L")
        np.testing.assert_allclose(fake_arm.updated[0][1], [0.02, 0.0, 0.18, 0.0, 0.0, 0.0])
        state = backend.get_state()
        self.assertEqual(state["active_tool_name"], "DMTAC_L")
        self.assertEqual(state["active_impedance_config"]["tool_name"], "DMTAC_L")
        self.assertEqual(state["active_impedance_config"]["mode"], "pose")
        self.assertEqual(state["active_impedance_config"]["linear_compliance_m_per_n"], 0.005)
        self.assertIn(
            "linear_compliance_m_per_n",
            state["active_impedance_config"]["profile_section_keys"],
        )

    def test_impedance_backend_preserves_solver_output_qpos_hint(self) -> None:
        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="0.0.0.0",
                side="right",
                connect_hardware=False,
                tool_profile="DMTAC_L",
            )
        )
        joint_deg = [-80.0, 8.0, 100.0, 40.0, 5.0, -110.0]
        command = EffectorCommand(
            effector_name="right_arm",
            side="right",
            command_type="cartesian_pose",
            value=np.eye(4),
            metadata={"solver_output_qpos": np.deg2rad(joint_deg)},
        )

        backend.send_command(command)

        np.testing.assert_allclose(backend._latest_target_joint_deg, joint_deg)

    def test_impedance_backend_debug_timing_records_send_command_lock_wait(self) -> None:
        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="0.0.0.0",
                side="right",
                connect_hardware=False,
                tool_profile="DMTAC_L",
                debug_timing=True,
            )
        )
        command = EffectorCommand(
            effector_name="right_arm",
            side="right",
            command_type="cartesian_pose",
            value=np.eye(4),
        )

        backend.send_command(command)
        timing = backend.get_state()["last_send_command_timing"]

        self.assertTrue(timing["enabled"])
        self.assertGreaterEqual(timing["send_lock_wait_ms"], 0.0)
        phase_names = [phase["name"] for phase in timing["phases"]]
        self.assertIn("send_lock_wait", phase_names)
        self.assertIn("state_update", phase_names)

    def test_rm65_position_backend_emergency_stop_requests_arm_stop(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
            )
        )
        fake_arm = _SingularityFakeArm()
        backend.arm = fake_arm

        backend.emergency_stop()

        self.assertEqual(fake_arm.stop_calls, 1)
        self.assertEqual(fake_arm.slow_stop_calls, 0)
        self.assertEqual(backend.get_state()["send_result_text"], "emergency_stop:Arm_Stop:0")
        self.assertTrue(bool(backend.get_state()["emergency_stopped"]))

    def test_impedance_backend_emergency_stop_requests_arm_stop(self) -> None:
        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="0.0.0.0",
                side="right",
                connect_hardware=False,
                tool_profile="DMTAC_L",
            )
        )
        fake_arm = _ToolSyncFakeArm()
        backend.arm = fake_arm

        backend.emergency_stop()

        self.assertEqual(fake_arm.stop_calls, 1)
        self.assertEqual(fake_arm.slow_stop_calls, 0)
        self.assertEqual(backend.get_state()["send_result_text"], "emergency_stop:Arm_Stop:0")
        self.assertTrue(bool(backend.get_state()["emergency_stopped"]))

    def test_rm65_position_backend_rejects_commands_after_emergency_stop(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                initial_qpos=(0.0, 0.0, 0.1, 0.0, 0.0, 0.0),
            )
        )
        fake_arm = _SingularityFakeArm()
        backend.arm = fake_arm
        backend.emergency_stop()
        command = EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=np.asarray([0.1, 0.0, 0.1, 0.0, 0.0, 0.0], dtype=float),
        )

        with self.assertRaisesRegex(RuntimeError, "emergency_stop is latched"):
            backend.send_command(command)

        self.assertEqual(fake_arm.sent_joint_deg, [])

    def test_impedance_backend_rejects_commands_after_emergency_stop(self) -> None:
        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="0.0.0.0",
                side="right",
                connect_hardware=False,
                tool_profile="DMTAC_L",
            )
        )
        fake_arm = _ToolSyncFakeArm()
        backend.arm = fake_arm
        backend.emergency_stop()
        command = EffectorCommand(
            effector_name="right_arm",
            side="right",
            command_type="cartesian_pose",
            value=np.eye(4),
        )

        with self.assertRaisesRegex(RuntimeError, "emergency_stop is latched"):
            backend.send_command(command)

        self.assertIsNone(backend.get_state()["last_command"])

    def test_impedance_emergency_stop_bypasses_controller_slow_stop(self) -> None:
        fake_arm = _ToolSyncFakeArm()
        controller = _StopMotionTrapController(fake_arm)
        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="0.0.0.0",
                side="right",
                connect_hardware=False,
                tool_profile="DMTAC_L",
            )
        )
        backend._controller = controller

        backend.emergency_stop()
        backend.close()

        self.assertEqual(fake_arm.stop_calls, 1)
        self.assertEqual(fake_arm.slow_stop_calls, 0)
        self.assertEqual(controller.stop_motion_calls, 0)
        self.assertEqual(controller.close_calls, 1)

    def test_impedance_backend_disabled_error_reports_last_fault(self) -> None:
        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="192.168.0.2",
                side="right",
                connect_hardware=True,
                tool_profile="DMTAC_L",
            )
        )
        backend._last_fault = "RuntimeError: right: force 41.0000N > hard_stop_force_n"
        command = EffectorCommand(
            effector_name="right_arm",
            side="right",
            command_type="cartesian_pose",
            value=np.eye(4),
        )

        with self.assertRaisesRegex(RuntimeError, "last_fault=.*hard_stop_force_n"):
            backend.send_command(command)

    def test_rm65_position_backend_skips_auto_profile_for_active_empty_tool(self) -> None:
        class _FakeArm:
            def Get_Current_Tool_Frame(self):
                return 0, {"name": EMPTY_TOOL_FRAME}

        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="right",
                connect_hardware=False,
            )
        )
        backend.arm = _FakeArm()
        with mock.patch("hardware.rm65.calibration_profiles.load_profile") as load_profile:
            backend._auto_load_active_tool_profile()

        load_profile.assert_not_called()
        self.assertEqual(backend.get_state()["active_tool_name"], EMPTY_TOOL_FRAME)
        self.assertEqual(backend.get_state()["tool_tcp_source"], "builtin_empty_arm_endpoint")

    def test_pose_retargeter_outputs_tool_tcp_pose(self) -> None:
        calibration = TeleopCalibration(
            arms={
                "left_arm": ArmCalibration(
                    name="left_arm",
                    side="left",
                    T_D_B=np.eye(4, dtype=float),
                )
            }
        )
        observation = TeleopObservation(
            channels={
                "left_wrist_pose": ObservationChannel(
                    channel_name="left_wrist_pose",
                    capability="pose",
                    side="left",
                    value=np.eye(4, dtype=float),
                    frame_name="desk",
                    timestamp=1.0,
                    source_name="test",
                )
            },
            timestamp=1.0,
        )
        retargeter = PoseToArmRetargeter(
            PoseToArmRetargeterConfig(
                side="left",
                input_channel_name="left_wrist_pose",
                effector_name="left_arm",
            )
        )

        command = retargeter.build_command_from_target_pose_desk(
            target_pose_desk=np.eye(4, dtype=float),
            observation=observation,
            calibration=calibration,
        )

        left = command.commands["left_arm"]
        self.assertEqual(left.metadata["arm_controlled_frame"], TOOL_TCP_FRAME)
        self.assertAlmostEqual(float(left.value[2, 3]), 0.0, places=6)
        self.assertAlmostEqual(float(left.metadata["target_pose_desk"][2, 3]), 0.0, places=6)
        self.assertAlmostEqual(
            float(left.metadata["target_pose_desk_retarget_frame"][2, 3]),
            0.0,
            places=6,
        )

    def test_rm65_backend_holds_when_command_enters_singularity(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                control_dt=1.0,
                max_angvel=(10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
                initial_qpos=(0.0, 0.0, np.deg2rad(4.0), 0.0, 0.0, 0.0),
            )
        )
        fake_arm = _SingularityFakeArm()
        backend.arm = fake_arm
        command = EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=np.asarray([0.0, 0.0, np.deg2rad(1.0), 0.0, 0.0, 0.0], dtype=float),
        )

        backend.send_command(command)

        self.assertEqual(fake_arm.thresholds, (3.0, 3.0, 0.05))
        self.assertEqual(backend.get_state()["singularity_metadata"]["singularity_gate_status"], "hold_entry")
        self.assertAlmostEqual(fake_arm.sent_joint_deg[-1][2], 4.0, places=6)

    def test_rm65_backend_slows_qpos_init_when_command_enters_singularity(self) -> None:
        for source in ("replay_qpos_init", "replay_qpos_init_final_hold", "deploy_fixed_init_qpos"):
            with self.subTest(source=source):
                backend = RM65ArmBackend(
                    RM65ArmBackendConfig(
                        arm_ip="0.0.0.0",
                        side="left",
                        connect_hardware=False,
                        control_dt=1.0,
                        max_angvel=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
                        singularity_qpos_init_slowdown=0.2,
                        initial_qpos=(0.0, 0.0, np.deg2rad(4.0), 0.0, 0.0, 0.0),
                    )
                )
                fake_arm = _SingularityFakeArm()
                backend.arm = fake_arm
                command = EffectorCommand(
                    effector_name="left_arm",
                    side="left",
                    command_type="joint_vector",
                    value=np.asarray([0.0, 0.0, np.deg2rad(1.0), 0.0, 0.0, 0.0], dtype=float),
                    metadata={"source": source},
                )

                backend.send_command(command)

                metadata = backend.get_state()["singularity_metadata"]
                self.assertEqual(
                    metadata["singularity_gate_status"],
                    "slow_qpos_init_entering_singularity",
                )
                self.assertEqual(metadata["source"], source)
                self.assertAlmostEqual(float(metadata["singularity_qpos_init_slowdown"]), 0.2)
                self.assertLess(fake_arm.sent_joint_deg[-1][2], 4.0)
                self.assertGreater(fake_arm.sent_joint_deg[-1][2], 1.0)
                self.assertAlmostEqual(fake_arm.sent_joint_deg[-1][2], np.rad2deg(np.deg2rad(4.0) - 0.02))

    def test_rm65_backend_holds_unchecked_replay_qpos_trajectory_entering_singularity(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                control_dt=1.0,
                max_angvel=(10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
                initial_qpos=(0.0, 0.0, np.deg2rad(4.0), 0.0, 0.0, 0.0),
            )
        )
        fake_arm = _SingularityFakeArm()
        backend.arm = fake_arm
        command = EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=np.asarray([0.0, 0.0, np.deg2rad(1.0), 0.0, 0.0, 0.0], dtype=float),
            metadata={"source": "replay_qpos_trajectory"},
        )

        backend.send_command(command)

        metadata = backend.get_state()["singularity_metadata"]
        self.assertEqual(metadata["singularity_gate_status"], "hold_entry")
        self.assertEqual(metadata["source"], "replay_qpos_trajectory")
        self.assertAlmostEqual(fake_arm.sent_joint_deg[-1][2], 4.0, places=6)

    def test_rm65_backend_accepts_only_prechecked_replay_singularity_slow_target(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                control_dt=1.0,
                max_angvel=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
                singularity_replay_slowdown=0.2,
                initial_qpos=(0.0, 0.0, np.deg2rad(4.0), 0.0, 0.0, 0.0),
            )
        )
        fake_arm = _SingularityFakeArm()
        backend.arm = fake_arm
        requested_target = np.asarray(
            [0.0, 0.0, np.deg2rad(1.0), 0.0, 0.0, 0.0], dtype=float
        )

        slow_target, preparation = backend.prepare_replay_singularity_target(
            requested_target,
            source="replay_collision_qpos_reference_75hz",
        )

        self.assertTrue(preparation["applied"])
        self.assertAlmostEqual(
            slow_target[2],
            np.deg2rad(4.0) - 0.02,
        )
        command = EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=slow_target,
            metadata={
                "source": "replay_collision_qpos_reference_75hz",
                "singularity_replay_slowdown_preapplied": True,
                "singularity_replay_slowdown_collision_rechecked": True,
                "max_joint_step_rad": preparation["max_joint_step_rad"],
            },
        )

        backend.send_command(command)

        metadata = backend.get_state()["singularity_metadata"]
        self.assertEqual(metadata["singularity_gate_status"], "slow_replay_entering_singularity")
        self.assertTrue(metadata["singularity_replay_slowdown_preapplied"])
        self.assertAlmostEqual(fake_arm.sent_joint_deg[-1][2], np.rad2deg(slow_target[2]))

    def test_rm65_backend_uses_robot_action_source_for_qpos_init_slowdown(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                control_dt=1.0,
                max_angvel=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
                singularity_qpos_init_slowdown=0.2,
                initial_qpos=(0.0, 0.0, np.deg2rad(4.0), 0.0, 0.0, 0.0),
            )
        )
        fake_arm = _SingularityFakeArm()
        backend.arm = fake_arm
        command = EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=np.asarray([0.0, 0.0, np.deg2rad(1.0), 0.0, 0.0, 0.0], dtype=float),
            metadata={
                "source": "robot_action",
                "action_source": "deploy_fixed_init_qpos",
            },
        )

        backend.send_command(command)

        metadata = backend.get_state()["singularity_metadata"]
        self.assertEqual(metadata["singularity_gate_status"], "slow_qpos_init_entering_singularity")
        self.assertEqual(metadata["source"], "deploy_fixed_init_qpos")
        self.assertLess(fake_arm.sent_joint_deg[-1][2], 4.0)
        self.assertGreater(fake_arm.sent_joint_deg[-1][2], 1.0)

    def test_rm65_backend_allows_motion_when_current_is_already_singular(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="left",
                connect_hardware=False,
                control_dt=1.0,
                max_angvel=(10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
                initial_qpos=(0.0, 0.0, np.deg2rad(2.0), 0.0, 0.0, 0.0),
            )
        )
        fake_arm = _SingularityFakeArm()
        backend.arm = fake_arm
        command = EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=np.asarray([0.0, 0.0, np.deg2rad(1.0), 0.0, 0.0, 0.0], dtype=float),
        )

        backend.send_command(command)

        self.assertEqual(
            backend.get_state()["singularity_metadata"]["singularity_gate_status"],
            "allow_current_singular",
        )
        self.assertAlmostEqual(fake_arm.sent_joint_deg[-1][2], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
