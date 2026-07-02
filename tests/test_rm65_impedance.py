from __future__ import annotations

import time
import unittest

import numpy as np

from hardware.rm65.force_compensation import (
    admittance_wrench_for_arm,
    apply_pose_offset,
    apply_pose_offset_work_rotvec,
)
from hardware.rm65.impedance import RM65ImpedanceConfig, RM65JointImpedanceController, RM65PoseImpedanceController


class _FakeRobot:
    def __init__(
        self,
        *,
        tool_name: str,
        tool_pose: list[float],
        current_joint: list[float],
        current_pose: list[float],
        model_fk_pose: list[float] | None = None,
        ik_success_pose: list[float] | None = None,
        ik_retry_pose: list[float] | None = None,
        ik_success_joint: list[float] | None = None,
        ik_retry_joint: list[float] | None = None,
        ik_quaternion_success_pose: list[float] | None = None,
        ik_quaternion_success_joint: list[float] | None = None,
        ik_extra_successes: list[tuple[list[float], list[float]]] | None = None,
        ik_all_solutions: list[list[float]] | None = None,
        force_sequence: list[list[float]] | None = None,
        position_limit_code: int = 0,
        self_collision_code: int = 0,
    ):
        self.tool_name = tool_name
        self.tool_pose = list(tool_pose)
        self.current_joint = list(current_joint)
        self.current_pose = list(current_pose)
        self.model_fk_pose = list(current_pose if model_fk_pose is None else model_fk_pose)
        self.ik_success_pose = list(self.model_fk_pose if ik_success_pose is None else ik_success_pose)
        self.ik_retry_pose = list(self.ik_success_pose if ik_retry_pose is None else ik_retry_pose)
        self.ik_success_joint = list(current_joint if ik_success_joint is None else ik_success_joint)
        self.ik_retry_joint = list(current_joint if ik_retry_joint is None else ik_retry_joint)
        self.ik_quaternion_success_pose = (
            None if ik_quaternion_success_pose is None else list(ik_quaternion_success_pose)
        )
        self.ik_quaternion_success_joint = (
            list(current_joint)
            if ik_quaternion_success_joint is None
            else list(ik_quaternion_success_joint)
        )
        self.ik_extra_successes = [
            (list(success_pose), list(success_joint))
            for success_pose, success_joint in (ik_extra_successes or [])
        ]
        self.ik_all_solutions = None if ik_all_solutions is None else [list(row) for row in ik_all_solutions]
        self.force_sequence = [list(force) for force in (force_sequence or [[0.0] * 6])]
        self.position_limit_code = int(position_limit_code)
        self.self_collision_code = int(self_collision_code)
        self.force_index = 0
        self.ik_calls: list[list[float]] = []
        self.movep_calls: list[list[float]] = []
        self.movej_calls: list[list[float]] = []

    def Arm_Socket_State(self) -> int:
        return 0

    def Arm_Socket_Close(self) -> None:
        pass

    def Get_Current_Tool_Frame(self):
        return 0, {"name": self.tool_name, "pose": self.tool_pose}

    def Get_Current_Arm_State(self, retry: int = 0):
        del retry
        return 0, list(self.current_joint), list(self.current_pose), 0, 0

    def Get_Force_Data(self):
        index = min(self.force_index, len(self.force_sequence) - 1)
        force = list(self.force_sequence[index])
        self.force_index += 1
        return 0, list(force), list(force), list(force), list(force)

    def Algo_Forward_Kinematics(self, joint, flag=1):
        del joint, flag
        return list(self.model_fk_pose)

    def Algo_Inverse_Kinematics(self, q_in, pose, flag=1):
        seed = [float(value) for value in q_in[:6]]
        self.ik_calls.append(seed)
        if int(flag) == 0 and self.ik_quaternion_success_pose is not None:
            if all(
                abs(float(a) - float(b)) <= 1e-6
                for a, b in zip(pose[:7], self.ik_quaternion_success_pose[:7])
            ):
                return 0, list(self.ik_quaternion_success_joint)
        for success_pose, success_joint in self.ik_extra_successes:
            if all(abs(float(a) - float(b)) <= 1e-6 for a, b in zip(pose[:6], success_pose[:6])):
                return 0, list(success_joint)
        if all(abs(float(a) - float(b)) <= 1e-6 for a, b in zip(seed, self.current_joint[:6])) and all(
            abs(float(a) - float(b)) <= 1e-6 for a, b in zip(pose[:6], self.ik_retry_pose[:6])
        ):
            return 0, list(self.ik_retry_joint)
        if all(abs(float(a) - float(b)) <= 1e-6 for a, b in zip(pose[:6], self.ik_success_pose[:6])):
            return 0, list(self.ik_success_joint)
        return 1, [0.0] * 6

    def Algo_Inverse_Kinematics_All(self, q_in, pose, flag=1):
        del q_in, pose, flag

        class Result:
            pass

        result = Result()
        if self.ik_all_solutions is None:
            result.result = 1
            result.num = 0
            result.q_solve = []
            return result
        result.result = 0
        result.num = len(self.ik_all_solutions)
        result.q_solve = [list(row[:6]) + [0.0, 0.0] for row in self.ik_all_solutions]
        return result

    def Algo_IKine_Check_Joint_Position_Limit(self, joint):
        del joint
        return self.position_limit_code

    def Algo_Safety_Robot_Self_Collision_Detection(self, joint):
        del joint
        return self.self_collision_code

    def Movep_CANFD(self, pose, follow=False, trajectory_mode=0, radio=0):
        del follow, trajectory_mode, radio
        self.movep_calls.append(list(pose))
        return 0

    def Movej_CANFD(self, joint, follow=False, trajectory_mode=0, radio=0):
        del follow, trajectory_mode, radio
        self.movej_calls.append(list(joint))
        return 0

    def Arm_Slow_Stop(self) -> int:
        return 0


class RM65ImpedanceTest(unittest.TestCase):
    def test_right_arm_admittance_wrench_swaps_rx_ry(self) -> None:
        wrench = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        self.assertEqual(admittance_wrench_for_arm(wrench, "left"), wrench)
        self.assertEqual(admittance_wrench_for_arm(wrench, "right"), [1.0, 2.0, 3.0, 5.0, 4.0, 6.0])

    def test_apply_pose_offset_uses_direct_euler_channel_offsets(self) -> None:
        pose = [-0.165333, 0.196294, 0.637266, 1.494, 0.449, -1.619]
        offset = [0.0, 0.0, 0.0, 0.0, 0.001, 0.0]

        shifted = apply_pose_offset(pose, offset)

        self.assertAlmostEqual(shifted[3], pose[3], places=9)
        self.assertAlmostEqual(shifted[4], pose[4] + 0.001, places=9)
        self.assertAlmostEqual(shifted[5], pose[5], places=9)

    def test_work_rotvec_offset_still_exhibits_pose_dependent_axis_coupling(self) -> None:
        pose = [-0.165333, 0.196294, 0.637266, 1.494, 0.449, -1.619]
        offset = [0.0, 0.0, 0.0, 0.0, 0.001, 0.0]

        shifted = apply_pose_offset_work_rotvec(pose, offset)

        self.assertLess(shifted[4] - pose[4], 2e-4)
        self.assertGreater(abs(shifted[3] - pose[3]), 1e-4)
        self.assertGreater(abs(shifted[5] - pose[5]), 1e-4)

    def test_saturation_thresholds_quantify_aggressive_compliance(self) -> None:
        config = RM65ImpedanceConfig(
            tool_name="DMTAC_L",
            linear_compliance_m_per_n=0.2,
            angular_compliance_rad_per_nm=0.4,
            max_linear_offset_m=0.05,
            max_angular_offset_rad=0.04,
            force_deadband_n=0.3,
            torque_deadband_nm=0.03,
        )

        thresholds = config.saturation_thresholds()

        self.assertAlmostEqual(thresholds["linear_deadbanded_force_n_for_max_offset"], 0.25)
        self.assertAlmostEqual(thresholds["linear_filtered_force_n_for_max_offset"], 0.55)
        self.assertAlmostEqual(thresholds["angular_deadbanded_torque_nm_for_max_offset"], 0.1)
        self.assertAlmostEqual(thresholds["angular_filtered_torque_nm_for_max_offset"], 0.13)

    def test_pose_controller_preflight_reports_state_pose_vs_local_fk_mismatch(self) -> None:
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=[-88.66, 6.76, 44.49, 91.72, 85.81, 12.80],
            current_pose=[-0.165333, 0.196294, 0.637266, 1.494, 0.449, -1.619],
            model_fk_pose=[-0.155333, 0.196294, 0.637266, 1.494, 0.449, -1.619],
            ik_success_pose=[-0.155333, 0.196294, 0.637266, 1.494, 0.449, -1.618],
            ik_retry_pose=[-0.155333, 0.196294, 0.637266, 1.494, 0.449, -1.617],
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        with self.assertRaises(RuntimeError) as ctx:
            controller.start()
        text = str(ctx.exception)

        self.assertIn("state_vs_model_fk_delta=linear:0.010000m", text)
        self.assertIn("hint=if zero-offset current pose is inconsistent with local IK", text)
        self.assertIn("hint=zero-offset preflight failed before the impedance loop started", text)

    def test_pose_controller_reports_offset_saturation_context_when_ik_fails(self) -> None:
        current_pose = [-0.165333, 0.196294, 0.637266, 1.494, 0.449, -1.619]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=[-88.66, 6.76, 44.49, 91.72, 85.81, 12.80],
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=[-0.165333, 0.196294, 0.637266, 1.494, 0.449, -1.618],
            ik_retry_pose=current_pose,
            force_sequence=[
                [0.0] * 6,
                [0.9, 0.0, 0.0, 0.13, 0.0, 0.0],
            ],
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                lowpass_alpha=1.0,
                linear_compliance_m_per_n=0.2,
                angular_compliance_rad_per_nm=0.4,
                max_linear_offset_m=0.05,
                max_angular_offset_rad=0.04,
                force_deadband_n=0.3,
                torque_deadband_nm=0.03,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )
        controller.start()
        controller._last_step_time = time.perf_counter() - (1.0 / controller.config.control_hz)

        with self.assertRaises(RuntimeError) as ctx:
            controller.step()

        text = str(ctx.exception)
        self.assertIn("step=1 tool=DMTAC_L", text)
        self.assertIn("target_offset=[0.050000, 0.000000, 0.000000", text)
        self.assertIn("-0.040000, 0.000000]", text)
        self.assertRegex(text, r"offset_norms=linear:0\.01000[0-9]m angular:0\.01000[0-9]rad")
        self.assertIn("deadbanded_wrench=[0.60000, 0.00000, 0.00000, 0.00000, 0.10000, 0.00000]", text)
        self.assertIn("raw_deadbanded_wrench=[0.60000, 0.00000, 0.00000, 0.10000, 0.00000, 0.00000]", text)
        self.assertIn("linear target offset is saturating the configured limit", text)
        self.assertIn("filtered force above about 0.550N saturates max_linear_offset_m", text)

    def test_pose_controller_rejects_large_j123_ik_branch_jump(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.03, 0.0, 0.35, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=target_pose,
            ik_retry_pose=current_pose,
            ik_success_joint=[95.0, 0.0, 20.0, 0.0, 0.0, 0.0],
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                lowpass_alpha=1.0,
                max_command_joint_step_deg=180.0,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        controller._last_step_time = time.perf_counter() - 0.2
        with self.assertRaises(RuntimeError) as ctx:
            controller.step(target_pose)

        text = str(ctx.exception)
        self.assertIn("IK branch jump guard tripped on J1-J3", text)
        self.assertIn("max_delta=95.000deg", text)
        self.assertIn("candidate_joint_deg=[95.000, 0.000, 20.000", text)

    def test_pose_controller_uses_reference_backoff_for_unsolved_target(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.04, 0.0, 0.35, 0.0, 0.0, 0.0]
        halfway_pose = [0.02, 0.0, 0.35, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=halfway_pose,
            ik_retry_pose=current_pose,
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                lowpass_alpha=1.0,
                ik_reference_backoff_min_scale=0.25,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        controller._last_step_time = time.perf_counter() - 0.2
        record = controller.step(target_pose)

        self.assertEqual(record["step"], 1)
        self.assertEqual(record["command_pose"], halfway_pose)
        self.assertEqual(record["ik_reference_backoff_scale"], 0.5)
        self.assertEqual(record["ik_method"], "euler")

    def test_pose_controller_can_hold_last_safe_when_runtime_ik_fails(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.04, 0.0, 0.35, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=current_pose,
            ik_retry_pose=current_pose,
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                lowpass_alpha=1.0,
                hold_last_safe_on_ik_failure=True,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        controller._last_step_time = time.perf_counter() - 0.2
        record = controller.step(target_pose)

        self.assertEqual(record["command_pose"], current_pose)
        self.assertEqual(record["command_joint_deg"], current_joint)
        self.assertEqual(record["command_joint_step_deg"], 0.0)
        self.assertEqual(record["ik_metadata"]["status"], "hold_last_safe")
        self.assertEqual(record["ik_metadata"]["reason"], "ik_return_code")
        self.assertEqual(record["ik_metadata"]["previous_ik_status"], "smoothing_required")
        self.assertEqual(record["ik_metadata"]["failed_ik_metadata"]["status"], "smoothing_required")
        self.assertNotIn("failed_ik_metadata", record["ik_metadata"]["failed_ik_metadata"])
        self.assertEqual(robot.movej_calls[-1], current_joint)

    def test_hold_last_safe_preserves_failed_ik_metadata_without_recursive_nesting(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                lowpass_alpha=1.0,
                hold_last_safe_on_ik_failure=True,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        controller._previous_command_pose = list(current_pose)
        controller._previous_command_joint = list(current_joint)
        controller._latest_wrench_context = {
            "ik_metadata": {
                "status": "smoothing_required",
                "method": "pinocchio_iterative",
                "source": "generic_ik_resolver",
                "best_rejected": {"reject_reason": "fk_pose_error"},
                "local_step_status": "failed_falling_back_to_full_resolve",
                "ik_debug_profile": {"solve_total_ms": 42.0},
                "failed_ik_metadata": {"status": "older_hold"},
            }
        }

        held = controller._hold_last_safe_command_after_ik_failure(
            reason="generic_resolver_return_code",
            message="generic resolver failed",
            target_pose=current_pose,
            seed_joint=current_joint,
            code=-100,
        )

        self.assertEqual(held, (current_pose, current_joint))
        metadata = controller._latest_wrench_context["ik_metadata"]
        self.assertEqual(metadata["status"], "hold_last_safe")
        self.assertEqual(metadata["previous_ik_method"], "pinocchio_iterative")
        self.assertEqual(metadata["best_rejected"]["reject_reason"], "fk_pose_error")
        self.assertEqual(metadata["failed_ik_metadata"]["best_rejected"]["reject_reason"], "fk_pose_error")
        self.assertNotIn("failed_ik_metadata", metadata["failed_ik_metadata"])

    def test_pose_controller_uses_preferred_joint_hint_without_runtime_sdk_ik(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.02, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_joint = [1.0, 2.0, 20.0, 3.0, 4.0, 5.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=current_pose,
            ik_retry_pose=current_pose,
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                lowpass_alpha=1.0,
                max_command_joint_velocity_deg_s=1000.0,
                joint_rate_limiter_enabled=False,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        preflight_ik_calls = len(robot.ik_calls)
        controller._last_step_time = time.perf_counter() - 0.2
        record = controller.step(target_pose, target_joint_deg=target_joint)

        self.assertEqual(record["command_pose"], target_pose)
        self.assertEqual(record["command_joint_deg"], target_joint)
        self.assertEqual(record["ik_method"], "dual_arm_joint_hint")
        self.assertEqual(record["ik_metadata"]["source"], "retargeter_solver_output_qpos")
        self.assertEqual(robot.movej_calls[-1], target_joint)
        self.assertEqual(len(robot.ik_calls), preflight_ik_calls)

    def test_pose_controller_uses_preferred_joint_hint_for_tiny_force_offset(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.02, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_joint = [1.0, 2.0, 20.0, 3.0, 4.0, 5.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=current_pose,
            ik_retry_pose=current_pose,
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                lowpass_alpha=1.0,
                max_command_joint_velocity_deg_s=1000.0,
                joint_rate_limiter_enabled=False,
                ik_resolver_enabled=False,
                direct_joint_hint_max_linear_offset_m=1e-3,
                direct_joint_hint_max_angular_offset_rad=2e-3,
            ),
            robot=robot,
        )

        controller.start()
        preflight_ik_calls = len(robot.ik_calls)
        controller._latest_wrench_context = {}
        command_pose, command_joint = controller._command_from_reference(
            target_pose,
            [5e-4, 0.0, 0.0, 0.0, 1e-3, 0.0],
            preferred_joint=target_joint,
        )

        self.assertEqual(command_pose, target_pose)
        self.assertEqual(command_joint, target_joint)
        self.assertEqual(controller._latest_wrench_context["ik_method"], ["dual_arm_joint_hint"])
        self.assertTrue(
            controller._latest_wrench_context["ik_metadata"]["direct_joint_hint_offset_ignored"]
        )
        self.assertEqual(len(robot.ik_calls), preflight_ik_calls)

    def test_pose_controller_backs_off_when_ik_joint_step_exceeds_limit(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.04, 0.0, 0.35, 0.0, 0.0, 0.0]
        scaled_pose = [0.03, 0.0, 0.35, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=target_pose,
            ik_success_joint=[0.0, 0.0, 20.0, 20.75, 0.0, 0.0],
            ik_retry_pose=current_pose,
            ik_extra_successes=[
                (scaled_pose, [0.0, 0.0, 20.0, 15.0, 0.0, 0.0]),
            ],
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                lowpass_alpha=1.0,
                max_command_joint_step_deg=20.0,
                max_command_joint_velocity_deg_s=1000.0,
                joint_rate_limiter_enabled=False,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        controller._last_step_time = time.perf_counter() - 0.2
        record = controller.step(target_pose)

        self.assertEqual(record["command_pose"], scaled_pose)
        self.assertEqual(record["command_joint_deg"], [0.0, 0.0, 20.0, 15.0, 0.0, 0.0])
        self.assertEqual(record["command_joint_step_deg"], 15.0)
        self.assertEqual(record["ik_command_step_backoff_scale"], 0.75)
        self.assertEqual(record["ik_command_step_candidate_deg"], 20.75)
        self.assertEqual(robot.movej_calls[-1], [0.0, 0.0, 20.0, 15.0, 0.0, 0.0])
        self.assertEqual(robot.movep_calls, [])

    def test_pose_controller_rate_limits_large_runtime_ik_joint_step(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.04, 0.0, 0.35, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=target_pose,
            ik_success_joint=[0.0, 0.0, 20.0, 30.0, 0.0, 0.0],
            ik_retry_pose=current_pose,
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                lowpass_alpha=1.0,
                max_command_joint_velocity_deg_s=1000.0,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        controller._last_step_time = time.perf_counter() - 0.2
        record = controller.step(target_pose)

        self.assertTrue(record["joint_rate_limited"])
        self.assertEqual(record["joint_velocity_limit_profile"], "rm65")
        self.assertTrue(np.allclose(record["command_joint_deg"], [0.0, 0.0, 20.0, 15.0, 0.0, 0.0], atol=1e-2))
        self.assertAlmostEqual(record["command_joint_step_deg"], 15.0, places=2)
        self.assertAlmostEqual(record["joint_velocity_limit_observed_dt_sec"], 0.2, places=2)
        self.assertAlmostEqual(record["joint_velocity_limit_effective_dt_sec"], 0.2, places=2)
        self.assertEqual(record["joint_velocity_limit_max_dt_sec"], 1.0)
        self.assertEqual(record["joint_safety"]["status_text"], "position_limit=ok self_collision=ok")
        self.assertTrue(np.allclose(robot.movej_calls[-1], [0.0, 0.0, 20.0, 15.0, 0.0, 0.0], atol=1e-2))

    def test_pose_controller_caps_runtime_rate_limit_dt_at_profile_max(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.04, 0.0, 0.35, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=target_pose,
            ik_success_joint=[0.0, 0.0, 20.0, 80.0, 0.0, 0.0],
            ik_retry_pose=current_pose,
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                lowpass_alpha=1.0,
                max_command_joint_velocity_deg_s=1000.0,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        controller._last_step_time = time.perf_counter() - 2.0
        record = controller.step(target_pose)

        self.assertTrue(record["joint_rate_limited"])
        self.assertTrue(np.allclose(record["command_joint_deg"], [0.0, 0.0, 20.0, 75.0, 0.0, 0.0], atol=1e-6))
        self.assertAlmostEqual(record["joint_velocity_limit_observed_dt_sec"], 2.0, places=2)
        self.assertEqual(record["joint_velocity_limit_effective_dt_sec"], 1.0)
        self.assertEqual(record["joint_velocity_limit_max_dt_sec"], 1.0)

    def test_pose_controller_resolver_selects_smallest_safe_all_solution_branch(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.01, 0.0, 0.35, 0.0, 0.0, 0.0]
        safe_joint = [5.0, 0.0, 20.0, 2.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=target_pose,
            ik_success_joint=[100.0, 0.0, 20.0, 0.0, 0.0, 0.0],
            ik_retry_pose=current_pose,
            ik_all_solutions=[
                [100.0, 0.0, 20.0, 0.0, 0.0, 0.0],
                safe_joint,
            ],
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                lowpass_alpha=1.0,
                max_command_joint_step_deg=180.0,
                max_command_joint_velocity_deg_s=1000.0,
                joint_rate_limiter_enabled=False,
                ik_resolver_include_pinocchio=False,
                ik_resolver_candidate_provider_names=["rm65_sdk"],
            ),
            robot=robot,
        )

        controller.start()
        controller._last_step_time = time.perf_counter() - 0.2
        record = controller.step(target_pose)

        self.assertEqual(record["command_joint_deg"], safe_joint)
        self.assertEqual(record["ik_method"], "euler_all")
        self.assertEqual(record["ik_metadata"]["source"], "generic_ik_resolver")
        self.assertEqual(record["ik_metadata"]["candidate_source"], "rm_sdk_all")
        self.assertEqual(record["ik_timing"]["source"], "generic_ik_resolver")
        self.assertIn("command", record["singularity"])
        self.assertIn("command", record["joint_limit_margin"])
        self.assertEqual(robot.movej_calls[-1], safe_joint)

    def test_pose_controller_sends_validated_joint_by_default(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.01, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_joint = [1.0, 2.0, 20.0, 3.0, 4.0, 5.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=target_pose,
            ik_success_joint=target_joint,
            ik_retry_pose=current_pose,
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                lowpass_alpha=1.0,
                max_command_joint_velocity_deg_s=1000.0,
                joint_rate_limiter_enabled=False,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        controller._last_step_time = time.perf_counter() - 0.2
        record = controller.step(target_pose)

        self.assertEqual(record["command_joint_deg"], target_joint)
        self.assertEqual(robot.movej_calls[-1], target_joint)
        self.assertEqual(robot.movep_calls, [])

    def test_pose_controller_can_opt_into_pose_transport(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.01, 0.0, 0.35, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=target_pose,
            ik_success_joint=[1.0, 0.0, 20.0, 0.0, 0.0, 0.0],
            ik_retry_pose=current_pose,
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                lowpass_alpha=1.0,
                pose_command_transport="pose",
                joint_rate_limiter_enabled=False,
                max_command_joint_velocity_deg_s=1000.0,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        controller._last_step_time = time.perf_counter() - 0.2
        controller.step(target_pose)

        self.assertEqual(robot.movep_calls[-1], target_pose)
        self.assertEqual(robot.movej_calls, [])

    def test_joint_controller_retries_ik_with_current_state_seed(self) -> None:
        current_joint = [-88.66, 6.76, 44.49, 91.72, 85.81, 12.80]
        current_pose = [-0.165333, 0.196294, 0.637266, 1.494, 0.449, -1.619]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=[-0.160578, 0.258421, 0.642874, 1.497779, 0.377615, -1.661674],
            ik_retry_pose=current_pose,
        )
        controller = RM65JointImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                lowpass_alpha=1.0,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        controller._reference.current = list(current_joint)
        controller._reference.start = list(current_joint)
        controller._reference.target = list(current_joint)
        controller._last_step_time = time.perf_counter() - (1.0 / controller.config.control_hz)
        controller._ik_seed = [-90.0, 10.0, 40.0, 90.0, 86.0, 10.0]

        record = controller.step()

        self.assertEqual(record["step"], 1)
        self.assertGreaterEqual(len(robot.ik_calls), 2)
        self.assertEqual(robot.ik_calls[0], [-90.0, 10.0, 40.0, 90.0, 86.0, 10.0])
        self.assertTrue(any(call == current_joint for call in robot.ik_calls[1:]))
        self.assertEqual(controller._ik_seed, current_joint)


if __name__ == "__main__":
    unittest.main()
