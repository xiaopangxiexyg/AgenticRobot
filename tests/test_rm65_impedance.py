from __future__ import annotations

import time
import unittest

from hardware.rm65.force_compensation import apply_pose_offset, apply_pose_offset_work_rotvec
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
        force_sequence: list[list[float]] | None = None,
    ):
        self.tool_name = tool_name
        self.tool_pose = list(tool_pose)
        self.current_joint = list(current_joint)
        self.current_pose = list(current_pose)
        self.model_fk_pose = list(current_pose if model_fk_pose is None else model_fk_pose)
        self.ik_success_pose = list(self.model_fk_pose if ik_success_pose is None else ik_success_pose)
        self.ik_retry_pose = list(self.ik_success_pose if ik_retry_pose is None else ik_retry_pose)
        self.force_sequence = [list(force) for force in (force_sequence or [[0.0] * 6])]
        self.force_index = 0
        self.ik_calls: list[list[float]] = []

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
        del flag
        seed = [float(value) for value in q_in[:6]]
        self.ik_calls.append(seed)
        if all(abs(float(a) - float(b)) <= 1e-6 for a, b in zip(seed, self.current_joint[:6])) and all(
            abs(float(a) - float(b)) <= 1e-6 for a, b in zip(pose[:6], self.ik_retry_pose[:6])
        ):
            return 0, list(self.current_joint)
        if all(abs(float(a) - float(b)) <= 1e-6 for a, b in zip(pose[:6], self.ik_success_pose[:6])):
            return 0, list(self.current_joint)
        return 1, [0.0] * 6

    def Movep_CANFD(self, pose, follow=False, trajectory_mode=0, radio=0):
        del pose, follow, trajectory_mode, radio
        return 0

    def Movej_CANFD(self, joint, follow=False, trajectory_mode=0, radio=0):
        del joint, follow, trajectory_mode, radio
        return 0

    def Arm_Slow_Stop(self) -> int:
        return 0


class RM65ImpedanceTest(unittest.TestCase):
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
            config=RM65ImpedanceConfig(tool_name="DMTAC_L", tare_sec=0.0),
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
            ),
            robot=robot,
        )
        controller.start()
        controller._last_step_time = time.perf_counter() - (1.0 / controller.config.control_hz)

        with self.assertRaises(RuntimeError) as ctx:
            controller.step()

        text = str(ctx.exception)
        self.assertIn("step=1 tool=DMTAC_L", text)
        self.assertIn("target_offset=[0.050000, 0.000000, 0.000000, -0.040000", text)
        self.assertIn("offset_norms=linear:0.010004m angular:0.010004rad", text)
        self.assertIn("deadbanded_wrench=[0.60000, 0.00000, 0.00000, 0.10000, 0.00000, 0.00000]", text)
        self.assertIn("linear target offset is saturating the configured limit", text)
        self.assertIn("filtered force above about 0.550N saturates max_linear_offset_m", text)

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
            config=RM65ImpedanceConfig(tool_name="DMTAC_L", tare_sec=0.0, lowpass_alpha=1.0),
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
        self.assertEqual(robot.ik_calls[1], current_joint)
        self.assertEqual(controller._ik_seed, current_joint)


if __name__ == "__main__":
    unittest.main()
