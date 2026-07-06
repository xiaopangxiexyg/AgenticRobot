from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

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
from teleop_runtime.retargeters.pose_to_arm import PoseToArmRetargeter, PoseToArmRetargeterConfig
from teleop_runtime.schemas import EffectorCommand, ObservationChannel, TeleopObservation


class _SingularityFakeArm:
    def __init__(self):
        self.sent_joint_deg: list[list[float]] = []
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

    def Movej_CANFD(self, joint_deg, follow=False):
        self.sent_joint_deg.append([float(value) for value in joint_deg])
        return 0

    def Arm_Stop(self):
        self.stop_calls += 1
        return 0

    def Arm_Slow_Stop(self):
        self.slow_stop_calls += 1
        return 0


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
        self.assertEqual(backend.get_state()["active_tool_name"], "DMTAC_L")

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
