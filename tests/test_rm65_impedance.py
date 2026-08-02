from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from hardware.rm65.calibration_profiles import load_profile
from hardware.rm65.force_compensation import (
    PoseAdmittance,
    admittance_wrench_for_arm,
    apply_pose_offset,
    apply_pose_offset_work_rotvec,
)
from hardware.rm65.impedance import RM65ImpedanceConfig, RM65JointImpedanceController, RM65PoseImpedanceController
from teleop_runtime.defaults import (
    DEFAULT_RM65_ARM_CONTROL_HZ,
    DEFAULT_RM65_POSITION_ARM_CONTROL_HZ,
)
from teleop_runtime.backends.arms.rm65_impedance import (
    RM65ImpedanceArmBackend,
    RM65ImpedanceArmBackendConfig,
)
from teleop_runtime.schemas import EffectorCommand
from robot_motion.types import IKResolveRequest, IKResolveResult, IKResolveStatus


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
        self.fk_call_count = 0
        self.position_limit_call_count = 0
        self.self_collision_call_count = 0
        self.ik_calls: list[list[float]] = []
        self.movep_calls: list[list[float]] = []
        self.movep_options: list[dict[str, object]] = []
        self.movej_calls: list[list[float]] = []
        self.movej_options: list[dict[str, object]] = []
        self.movej_call_times: list[float] = []
        self.movej_delay_sec = 0.0

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
        self.fk_call_count += 1
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
        self.position_limit_call_count += 1
        return self.position_limit_code

    def Algo_Safety_Robot_Self_Collision_Detection(self, joint):
        del joint
        self.self_collision_call_count += 1
        return self.self_collision_code

    def Movep_CANFD(self, pose, follow=False, trajectory_mode=0, radio=0):
        self.movep_calls.append(list(pose))
        self.movep_options.append(
            {
                "follow": bool(follow),
                "trajectory_mode": int(trajectory_mode),
                "radio": int(radio),
            }
        )
        return 0

    def Movej_CANFD(self, joint, follow=False, trajectory_mode=0, radio=0):
        self.movej_call_times.append(time.perf_counter())
        if self.movej_delay_sec > 0.0:
            time.sleep(self.movej_delay_sec)
        self.movej_calls.append(list(joint))
        self.movej_options.append(
            {
                "follow": bool(follow),
                "trajectory_mode": int(trajectory_mode),
                "radio": int(radio),
            }
        )
        return 0

    def Arm_Slow_Stop(self) -> int:
        return 0


class RM65ImpedanceTest(unittest.TestCase):
    def test_impedance_backend_manual_rearm_clears_force_stop_latch(self) -> None:
        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="127.0.0.1",
                side="left",
                connect_hardware=False,
            )
        )

        class FakeController:
            def __init__(self) -> None:
                self.stop_calls = 0
                self.reset_qpos = None

            def stop_motion(self) -> None:
                self.stop_calls += 1

            def reset_motion_state(self, qpos) -> None:
                self.reset_qpos = np.asarray(qpos, dtype=float).copy()

        controller = FakeController()
        backend.arm = object()  # type: ignore[assignment]
        backend._controller = controller  # type: ignore[assignment]
        backend._emergency_stopped = True
        backend._last_fault = "RuntimeError: left: force 55.0000N > hard_stop_force_n"

        result = backend.manual_rearm_after_emergency_stop(np.zeros(6))

        self.assertTrue(result["rearmed"])
        self.assertFalse(backend.is_emergency_stopped())
        self.assertIsNone(backend.get_state()["fault"])
        self.assertEqual(controller.stop_calls, 1)
        np.testing.assert_allclose(controller.reset_qpos, np.zeros(6))

    def test_impedance_backend_exposes_profile_tcp_for_replay_preflight(self) -> None:
        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="127.0.0.1",
                side="left",
                connect_hardware=False,
            )
        )
        backend._profile = {
            "tcp": {
                "pose_flange_m_rad": [0.0, 0.0, 0.14, 0.0, 0.0, 0.0]
            }
        }

        np.testing.assert_allclose(
            backend._active_tcp_pose_flange_m_rad(),
            np.asarray([0.0, 0.0, 0.14, 0.0, 0.0, 0.0]),
        )

    def test_config_from_profile_accepts_follow_alias(self) -> None:
        profile = {
            "tool_name": "DMTAC_M",
            "impedance": {
                "pose": {
                    "follow": True,
                    "follow_qpos_alpha": 0.8,
                    "canfd_follow_stream_hz": 120.0,
                },
            },
        }

        config = RM65ImpedanceConfig.from_profile(profile, mode="pose")
        self.assertTrue(config.canfd_follow)
        self.assertEqual(config.follow_qpos_alpha, 0.8)
        self.assertEqual(config.canfd_follow_stream_hz, 120.0)
        self.assertEqual(config.trajectory_mode, 0)
        self.assertEqual(config.radio, 0)

        override = RM65ImpedanceConfig.from_profile(profile, mode="pose", follow=False)
        self.assertFalse(override.canfd_follow)

    def test_config_defaults_to_no_follow_with_current_position_stream_when_enabled(self) -> None:
        config = RM65ImpedanceConfig(tool_name="DMTAC_M")

        self.assertFalse(config.canfd_follow)
        self.assertEqual(config.control_hz, DEFAULT_RM65_ARM_CONTROL_HZ)
        self.assertEqual(config.canfd_follow_stream_hz, DEFAULT_RM65_POSITION_ARM_CONTROL_HZ)
        self.assertFalse(config.deployment_full_ik_preempts_local)
        backend_config = RM65ImpedanceArmBackendConfig(arm_ip="127.0.0.1", side="left")
        self.assertEqual(backend_config.actual_send_dt_max_scale, 1.0)
        self.assertFalse(backend_config.scale_outer_execution_step_by_actual_dt)

    def test_policy_inference_soft_hold_freezes_measured_joint_and_releases_on_fresh_target(self) -> None:
        current_joint = [0.0] * 6
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        hold_joint = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
                ik_resolver_enabled=False,
                joint_rate_limiter_enabled=False,
            ),
            robot=robot,
        )
        controller.start()
        controller._previous_command_joint = [20.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        controller._previous_command_pose = list(current_pose)

        held = controller.step(
            soft_hold_joint_deg=hold_joint,
            soft_hold_reason="test_inference_gap",
        )

        self.assertTrue(held["policy_inference_soft_hold"])
        self.assertEqual(held["policy_inference_soft_hold_reason"], "test_inference_gap")
        np.testing.assert_allclose(held["command_joint_deg"], hold_joint)
        self.assertEqual(held["applied_offset"], [0.0] * 6)
        self.assertEqual(held["ik_method"], "policy_inference_soft_hold")

        # A stale Cartesian reference must not leak through a later hold tick.
        controller._reference.set_target([0.08, 0.0, 0.35, 0.0, 0.0, 0.0])
        held_again = controller.step(
            soft_hold_joint_deg=hold_joint,
            soft_hold_reason="test_inference_gap",
        )
        self.assertTrue(held_again["policy_inference_soft_hold"])
        np.testing.assert_allclose(held_again["command_joint_deg"], hold_joint)

        released = controller.step(current_pose, target_joint_deg=hold_joint)
        self.assertFalse(released["policy_inference_soft_hold"])
        self.assertFalse(controller._soft_hold_active)

    def test_dmtac_impedance_profiles_disable_direct_joint_hint_offsets(self) -> None:
        for profile_name in (
            "hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json",
            "hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json",
        ):
            profile = load_profile(profile_name)
            for mode in ("pose", "joint"):
                config = RM65ImpedanceConfig.from_profile(profile, mode=mode)
                self.assertAlmostEqual(config.direct_joint_hint_max_linear_offset_m, 0.0)
                self.assertAlmostEqual(config.direct_joint_hint_max_angular_offset_rad, 0.0)

    def test_hard_deadline_rejects_blocking_rm65_sdk_candidate_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-blocking candidate providers"):
            RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                ik_hard_deadline_enabled=True,
                ik_resolver_candidate_provider_names=["rm65_sdk"],
            )
        with self.assertRaisesRegex(ValueError, "non-blocking candidate providers"):
            RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                ik_hard_deadline_enabled=True,
                ik_resolver_candidate_provider_names=[],
            )

        config = RM65ImpedanceConfig(
            tool_name="DMTAC_L",
            ik_hard_deadline_enabled=False,
            ik_resolver_candidate_provider_names=["rm65_sdk"],
        )
        self.assertEqual(config.ik_resolver_candidate_provider_names, ["rm65_sdk"])

    def test_config_from_profile_allows_canfd_trajectory_override(self) -> None:
        profile = {
            "tool_name": "DMTAC_M",
            "impedance": {
                "pose": {
                    "trajectory_mode": 0,
                    "radio": 0,
                },
            },
        }

        config = RM65ImpedanceConfig.from_profile(
            profile,
            mode="pose",
            trajectory_mode=1,
            radio=50,
        )

        self.assertEqual(config.trajectory_mode, 1)
        self.assertEqual(config.radio, 50)

    def test_config_rejects_out_of_range_canfd_radio(self) -> None:
        with self.assertRaisesRegex(ValueError, "radio must be in \\[0, 100\\]"):
            RM65ImpedanceConfig(tool_name="DMTAC_M", radio=500)

    def test_impedance_normalizes_tiny_observed_joint_limit_overshoot_only_for_state_anchor(self) -> None:
        current_joint = [0.0, 0.0, float(np.rad2deg(2.355008)), 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
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
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()

        upper_deg = float(np.rad2deg(2.355))
        self.assertAlmostEqual(controller._previous_command_joint[2], upper_deg, places=6)
        normalization = controller._last_observed_joint_normalization
        self.assertEqual(normalization["status"], "clipped")
        self.assertEqual(normalization["normalized_joint_indices"], [2])
        self.assertGreater(normalization["raw_joint_deg"][2], upper_deg)
        self.assertAlmostEqual(normalization["normalized_joint_deg"][2], upper_deg, places=6)

        too_far = list(current_joint)
        too_far[2] = upper_deg + 0.02
        with self.assertRaisesRegex(ValueError, "observed-state tolerance"):
            controller._normalize_observed_joint_deg(too_far, label="test feedback")

    def test_force_joint_operating_margin_rejects_further_entry_and_allows_recovery(self) -> None:
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        upper_deg = float(np.rad2deg(2.355))
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=[0.0, 0.0, upper_deg - 1.0, 0.0, 0.0, 0.0],
            current_pose=current_pose,
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                joint_operating_margin_deg=3.0,
                joint_rate_limiter_enabled=False,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )
        controller._previous_command_joint = [0.0, 0.0, upper_deg - 1.0, 0.0, 0.0, 0.0]
        controller._latest_wrench_context = {"ik_metadata": {}}

        with self.assertRaisesRegex(RuntimeError, "joint operating margin rejected"):
            controller._apply_joint_operating_margin_guard(
                command_joint=[0.0, 0.0, upper_deg - 0.5, 0.0, 0.0, 0.0]
            )
        rejected = controller._latest_wrench_context["joint_operating_margin"]
        self.assertEqual(rejected["status"], "rejected")
        self.assertTrue(rejected["reject"])

        controller._apply_joint_operating_margin_guard(
            command_joint=[0.0, 0.0, upper_deg - 1.5, 0.0, 0.0, 0.0]
        )
        recovery = controller._latest_wrench_context["joint_operating_margin"]
        self.assertEqual(recovery["status"], "recovery_only")
        self.assertFalse(recovery["reject"])

    def test_pose_controller_hard_deadline_holds_last_safe_not_new_preferred_hint(self) -> None:
        class DeadlineResolver:
            profile = SimpleNamespace(provider_time_budget_ms=None)

            def resolve_step(self, request, **_kwargs):
                return IKResolveResult(
                    status=IKResolveStatus.RELAXED,
                    qpos=np.asarray(request.previous_safe_qpos, dtype=float),
                    requested_target_pose=np.asarray(request.target_pose, dtype=float),
                    effective_target_pose=np.asarray(request.reference_pose, dtype=float),
                    tolerance_fraction=1.0,
                    candidate_source="hard_deadline_hold_reference",
                    message="ik_hard_deadline_hold_reference",
                    metadata={
                        "hard_deadline_enabled": True,
                        "hard_deadline_budget_ms": 25.0,
                        "hard_deadline_expired": True,
                        "hard_deadline_phase": "local_first",
                    },
                )

        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        preferred_joint = [10.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
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
                hold_last_safe_on_ik_failure=True,
                ik_resolver_enabled=False,
                direct_joint_hint_max_linear_offset_m=0.003,
            ),
            robot=robot,
        )
        controller.start()
        controller._ik_resolver = DeadlineResolver()
        controller._latest_wrench_context = {
            "tare_corrected": [0.0] * 6,
            "ik_metadata": {},
        }

        command_pose, command_joint = controller._command_from_reference(
            current_pose,
            [0.004, 0.0, 0.0, 0.0, 0.0, 0.0],
            preferred_joint=preferred_joint,
        )

        self.assertEqual(command_pose, current_pose)
        self.assertEqual(command_joint, current_joint)
        metadata = controller._latest_wrench_context["ik_metadata"]
        self.assertEqual(metadata["reason"], "ik_hard_deadline")
        self.assertTrue(metadata["ik_hard_deadline_expired"])
        self.assertEqual(metadata["failed_ik_metadata"]["candidate_source"], "hard_deadline_hold_reference")

    def test_right_arm_admittance_wrench_keeps_raw_tx_ty_order(self) -> None:
        wrench = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        self.assertEqual(admittance_wrench_for_arm(wrench, "left"), wrench)
        self.assertEqual(admittance_wrench_for_arm(wrench, "right"), wrench)

    def test_apply_pose_offset_uses_work_frame_rotation_vector_offsets(self) -> None:
        pose = [-0.165333, 0.196294, 0.637266, 1.494, 0.449, -1.619]
        offset = [0.0, 0.0, 0.0, 0.0, 0.001, 0.0]

        shifted = apply_pose_offset(pose, offset)
        work_rotvec_shifted = apply_pose_offset_work_rotvec(pose, offset)

        for actual, expected in zip(shifted, work_rotvec_shifted):
            self.assertAlmostEqual(actual, expected, places=9)
        self.assertLess(shifted[4] - pose[4], 2e-4)
        self.assertGreater(abs(shifted[3] - pose[3]), 1e-4)
        self.assertGreater(abs(shifted[5] - pose[5]), 1e-4)

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

    def test_return_deadbands_default_to_active_deadbands(self) -> None:
        config = RM65ImpedanceConfig(
            tool_name="DMTAC_L",
            force_deadband_n=0.42,
            torque_deadband_nm=0.07,
        )

        self.assertIsNone(config.return_force_deadband_n)
        self.assertIsNone(config.return_torque_deadband_nm)
        self.assertAlmostEqual(config.effective_return_force_deadband_n, 0.42)
        self.assertAlmostEqual(config.effective_return_torque_deadband_nm, 0.07)

    def test_pose_admittance_angular_axis_velocity_scales_limit_rz_only(self) -> None:
        admittance = PoseAdmittance(
            enabled_axes=[False, False, False, True, True, True],
            angular_force_sign=1.0,
            angular_axis_signs=[1.0, 1.0, 1.0],
            angular_axis_velocity_scales=[1.0, 1.0, 0.2],
            angular_compliance_rad_per_nm=1.0,
            max_angular_velocity_rad_s=1.0,
            max_angular_offset_rad=2.0,
            return_rate_s=0.0,
        )

        update = admittance.update([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 0.1)

        self.assertTrue(update["angular_axis_velocity_limit_applied"])
        self.assertTrue(update["angular_velocity_limit_applied"])
        self.assertEqual(update["angular_axis_velocity_scales"], [1.0, 1.0, 0.2])
        self.assertEqual(update["angular_axis_step_velocity_limits_rad_s"], [1.0, 1.0, 0.2])
        self.assertAlmostEqual(update["offset"][5], 0.02)
        self.assertAlmostEqual(update["velocity"][5], 0.2)

    def test_impedance_config_local_step_position_worsen_override_reaches_resolver_profile(self) -> None:
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=[0.0] * 6,
            current_pose=[0.0, 0.0, 0.35, 0.0, 0.0, 0.0],
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                ik_resolver_include_pinocchio=False,
                ik_resolver_local_step_max_position_worsen_m=0.025,
                ik_resolver_local_step_component_progress_max_position_worsen_m=0.03,
            ),
            robot=robot,
        )

        controller._initialize_ik_resolver()

        self.assertAlmostEqual(
            controller._ik_resolver_init_metadata["local_step_max_position_worsen_m"],
            0.025,
        )
        self.assertAlmostEqual(
            controller._ik_resolver_init_metadata[
                "local_step_component_progress_max_position_worsen_m"
            ],
            0.03,
        )

    def test_pose_admittance_compliance_releases_by_return_rate_after_deadband(self) -> None:
        admittance = PoseAdmittance(
            enabled_axes=[False, False, False, True, True, True],
            angular_force_sign=1.0,
            angular_axis_signs=[1.0, 1.0, 1.0],
            angular_compliance_rad_per_nm=0.1,
            max_angular_velocity_rad_s=1.0,
            max_angular_offset_rad=1.2,
            return_rate_s=0.35,
            offset=[0.0, 0.0, 0.0, 0.0, 0.0, 0.6],
        )

        update = admittance.update([0.0] * 6, 0.025)

        expected = 0.6 * (1.0 - 0.35 * 0.025)
        self.assertEqual(update["target_offset_unclamped"][3:], [0.0, 0.0, 0.0])
        self.assertEqual(update["return_applied_axes"], ["rx", "ry", "rz"])
        self.assertEqual(update["target_offset_after_return"][:5], [0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(update["target_offset"][5], expected)
        self.assertAlmostEqual(update["target_offset_after_return"][5], expected)
        self.assertAlmostEqual(update["target_offset_after_offset_limit"][5], expected)
        self.assertAlmostEqual(update["offset"][5], expected)
        self.assertAlmostEqual(update["offset_step_after_velocity_limit"][5], expected - 0.6)
        self.assertAlmostEqual(update["velocity"][5], (expected - 0.6) / 0.025)

    def test_pose_admittance_compliance_can_hold_after_deadband(self) -> None:
        admittance = PoseAdmittance(
            enabled_axes=[False, False, False, True, True, True],
            angular_force_sign=1.0,
            angular_axis_signs=[1.0, 1.0, 1.0],
            angular_compliance_rad_per_nm=0.1,
            max_angular_velocity_rad_s=1.0,
            max_angular_offset_rad=1.2,
            return_rate_s=0.0,
            offset=[0.0, 0.0, 0.0, 0.2, -0.1, 0.3],
        )

        update = admittance.update([0.0] * 6, 0.025)

        self.assertEqual(update["return_applied_axes"], ["rx", "ry", "rz"])
        self.assertEqual(update["target_offset"][3:], [0.2, -0.1, 0.3])
        self.assertEqual(update["offset"][3:], [0.2, -0.1, 0.3])
        self.assertEqual(update["velocity"][3:], [0.0, 0.0, 0.0])

    def test_pose_admittance_compliance_force_resumes_target_tracking(self) -> None:
        admittance = PoseAdmittance(
            enabled_axes=[False, False, False, True, True, True],
            angular_force_sign=1.0,
            angular_axis_signs=[1.0, 1.0, 1.0],
            angular_compliance_rad_per_nm=0.1,
            max_angular_velocity_rad_s=1.0,
            max_angular_offset_rad=1.2,
            return_rate_s=0.35,
            offset=[0.0, 0.0, 0.0, 0.2, -0.1, 0.6],
        )

        update = admittance.update([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 0.025)

        self.assertEqual(update["return_applied_axes"], ["rx", "ry"])
        self.assertEqual(update["return_paused_axes"], [])
        self.assertFalse(update["linear_return_paused"])
        self.assertFalse(update["angular_return_paused"])
        self.assertEqual(update["target_offset_unclamped"][3:], [0.0, 0.0, 0.1])
        self.assertAlmostEqual(update["target_offset_after_return"][3], 0.2 * (1.0 - 0.35 * 0.025))
        self.assertAlmostEqual(update["target_offset_after_return"][4], -0.1 * (1.0 - 0.35 * 0.025))
        self.assertAlmostEqual(update["target_offset_after_return"][5], 0.1)
        self.assertAlmostEqual(update["target_offset"][3], 0.2 * (1.0 - 0.35 * 0.025))
        self.assertAlmostEqual(update["target_offset"][4], -0.1 * (1.0 - 0.35 * 0.025))
        self.assertAlmostEqual(update["target_offset"][5], 0.1)
        self.assertLess(update["offset"][3], 0.2)
        self.assertGreater(update["offset"][4], -0.1)
        self.assertAlmostEqual(update["offset"][5], 0.575, places=6)
        self.assertAlmostEqual(update["offset_delta_before_velocity_limit"][5], -0.5)
        self.assertAlmostEqual(update["offset_step_after_velocity_limit"][5], -0.025, places=6)
        self.assertTrue(update["angular_velocity_limit_applied"])
        self.assertAlmostEqual(update["velocity"][5], -1.0, delta=1e-5)

    def test_pose_admittance_compliance_caps_unloading_toward_zero(self) -> None:
        admittance = PoseAdmittance(
            enabled_axes=[False, False, False, True, True, True],
            angular_force_sign=1.0,
            angular_axis_signs=[1.0, 1.0, 1.0],
            angular_compliance_rad_per_nm=0.1,
            max_angular_velocity_rad_s=1.0,
            max_angular_return_velocity_rad_s=0.05,
            max_angular_offset_rad=1.2,
            return_rate_s=0.35,
            offset=[0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        )

        update = admittance.update([0.0] * 6, 0.1)

        expected_target = 0.2 * (1.0 - 0.35 * 0.1)
        self.assertEqual(update["return_applied_axes"], ["rx", "ry", "rz"])
        self.assertAlmostEqual(update["target_offset"][5], expected_target)
        self.assertAlmostEqual(update["offset_delta_before_velocity_limit"][5], expected_target - 0.2)
        self.assertAlmostEqual(update["offset_step_after_velocity_limit"][5], -0.005)
        self.assertAlmostEqual(update["offset"][5], 0.195)
        self.assertAlmostEqual(update["velocity"][5], -0.05)
        self.assertTrue(update["angular_velocity_limit_applied"])
        self.assertTrue(update["angular_return_velocity_limit_applied"])
        self.assertAlmostEqual(update["angular_step_velocity_limit_rad_s"], 0.05)

    def test_pose_admittance_compliance_does_not_cap_larger_force_reversal(self) -> None:
        admittance = PoseAdmittance(
            enabled_axes=[False, False, False, True, True, True],
            angular_force_sign=1.0,
            angular_axis_signs=[1.0, 1.0, 1.0],
            angular_compliance_rad_per_nm=0.1,
            max_angular_velocity_rad_s=1.0,
            max_angular_return_velocity_rad_s=0.05,
            max_angular_offset_rad=1.2,
            return_rate_s=0.35,
            offset=[0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        )

        update = admittance.update([0.0, 0.0, 0.0, 0.0, 0.0, -3.0], 0.1)

        self.assertAlmostEqual(update["target_offset"][5], -0.3)
        self.assertAlmostEqual(update["offset_delta_before_velocity_limit"][5], -0.5)
        self.assertAlmostEqual(update["offset_step_after_velocity_limit"][5], -0.1)
        self.assertAlmostEqual(update["offset"][5], 0.1)
        self.assertTrue(update["angular_velocity_limit_applied"])
        self.assertFalse(update["angular_return_velocity_limit_applied"])
        self.assertAlmostEqual(update["angular_step_velocity_limit_rad_s"], 1.0)

    def test_pose_admittance_compliance_does_not_cap_active_force_reversal_toward_zero(self) -> None:
        admittance = PoseAdmittance(
            enabled_axes=[False, False, False, True, True, True],
            angular_force_sign=1.0,
            angular_axis_signs=[1.0, 1.0, 1.0],
            angular_compliance_rad_per_nm=0.1,
            max_angular_velocity_rad_s=1.0,
            max_angular_return_velocity_rad_s=0.05,
            max_angular_offset_rad=1.2,
            return_rate_s=0.35,
            offset=[0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
        )

        update = admittance.update([0.0, 0.0, 0.0, 0.0, 0.0, -1.0], 0.1)

        self.assertEqual(update["return_applied_axes"], ["rx", "ry"])
        self.assertAlmostEqual(update["target_offset"][5], -0.1)
        self.assertAlmostEqual(update["offset_delta_before_velocity_limit"][5], -0.4)
        self.assertAlmostEqual(update["offset_step_after_velocity_limit"][5], -0.1)
        self.assertAlmostEqual(update["offset"][5], 0.2)
        self.assertTrue(update["angular_velocity_limit_applied"])
        self.assertFalse(update["angular_return_velocity_limit_applied"])
        self.assertAlmostEqual(update["angular_step_velocity_limit_rad_s"], 1.0)

    def test_pose_admittance_compliance_pauses_return_from_filtered_activity(self) -> None:
        admittance = PoseAdmittance(
            enabled_axes=[False, False, False, True, True, True],
            angular_force_sign=1.0,
            angular_axis_signs=[1.0, 1.0, 1.0],
            angular_compliance_rad_per_nm=0.1,
            max_angular_velocity_rad_s=1.0,
            max_angular_offset_rad=1.2,
            return_rate_s=0.35,
            offset=[0.0, 0.0, 0.0, 0.2, -0.1, 0.6],
        )

        update = admittance.update(
            [0.0] * 6,
            0.025,
            return_activity_wrench=[0.0, 0.0, 0.0, 0.0, 0.0, 0.04],
            return_torque_deadband_nm=0.03,
        )

        self.assertEqual(update["return_applied_axes"], ["rx", "ry"])
        self.assertEqual(update["return_paused_axes"], ["rz"])
        self.assertTrue(update["angular_return_paused"])
        self.assertEqual(update["target_offset_unclamped"][3:], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(update["target_offset_after_return"][3], 0.2 * (1.0 - 0.35 * 0.025))
        self.assertAlmostEqual(update["target_offset_after_return"][4], -0.1 * (1.0 - 0.35 * 0.025))
        self.assertAlmostEqual(update["target_offset_after_return"][5], 0.6)
        self.assertLess(update["offset"][3], 0.2)
        self.assertGreater(update["offset"][4], -0.1)
        self.assertAlmostEqual(update["offset"][5], 0.6)

    def test_pose_admittance_compliance_return_pause_is_per_axis(self) -> None:
        admittance = PoseAdmittance(
            enabled_axes=[True, True, True, False, False, False],
            force_sign=1.0,
            linear_compliance_m_per_n=0.004,
            max_linear_velocity_m_s=1.0,
            max_linear_offset_m=1.0,
            return_rate_s=1.0,
            offset=[0.02, -0.02, 0.01, 0.0, 0.0, 0.0],
        )

        update = admittance.update(
            [0.0] * 6,
            0.025,
            return_activity_wrench=[0.3, 0.3, 0.3, 0.0, 0.0, 0.0],
            return_force_deadband_n=0.45,
        )

        self.assertEqual(update["return_paused_axes"], [])
        self.assertEqual(update["return_applied_axes"], ["x", "y", "z"])
        self.assertFalse(update["linear_return_paused"])
        self.assertLess(abs(update["offset"][0]), 0.02)
        self.assertLess(abs(update["offset"][1]), 0.02)
        self.assertLess(abs(update["offset"][2]), 0.01)

    def test_impedance_backend_control_dt_overrides_profile_control_hz(self) -> None:
        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="127.0.0.1",
                side="left",
                connect_hardware=False,
                control_dt=1.0 / 30.0,
                action_hz=30.0,
            )
        )
        config = backend._build_impedance_config(
            {
                "tool_name": "DMTAC_M",
                "impedance": {
                    "pose": {
                        "mode": "pose",
                        "control_hz": 40.0,
                        "action_hz": 20.0,
                    }
                },
            }
        )

        self.assertAlmostEqual(config.control_hz, 30.0)
        self.assertAlmostEqual(config.action_hz, 30.0)
        self.assertIsNotNone(backend._active_impedance_config)
        self.assertAlmostEqual(backend._active_impedance_config["requested_control_hz"], 30.0)
        self.assertIn("control_hz", backend._active_impedance_config["override_keys"])

    def test_impedance_backend_caps_requested_control_hz_to_rm65_limit(self) -> None:
        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="127.0.0.1",
                side="left",
                connect_hardware=False,
                control_dt=1.0 / 50.0,
                action_hz=50.0,
            )
        )
        config = backend._build_impedance_config(
            {
                "tool_name": "DMTAC_M",
                "impedance": {
                    "pose": {
                        "mode": "pose",
                        "control_hz": 40.0,
                        "action_hz": 20.0,
                    }
                },
            }
        )

        self.assertAlmostEqual(config.control_hz, 40.0)
        self.assertAlmostEqual(config.action_hz, 40.0)
        self.assertIsNotNone(backend._active_impedance_config)
        self.assertAlmostEqual(backend._active_impedance_config["requested_control_hz"], 50.0)
        self.assertAlmostEqual(backend._active_impedance_config["effective_control_hz"], 40.0)
        self.assertTrue(backend._active_impedance_config["control_hz_clamped"])

    def test_impedance_backend_caps_control_hz_override_to_rm65_limit(self) -> None:
        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="127.0.0.1",
                side="left",
                connect_hardware=False,
                control_dt=1.0 / 30.0,
                action_hz=30.0,
                impedance_overrides={
                    "control_hz": 50.0,
                    "action_hz": 50.0,
                    "return_torque_deadband_nm": 0.005,
                },
            )
        )
        config = backend._build_impedance_config(
            {
                "tool_name": "DMTAC_M",
                "impedance": {
                    "pose": {
                        "mode": "pose",
                        "control_hz": 40.0,
                        "action_hz": 20.0,
                    }
                },
            }
        )

        self.assertAlmostEqual(config.control_hz, 40.0)
        self.assertAlmostEqual(config.action_hz, 40.0)
        self.assertAlmostEqual(config.return_torque_deadband_nm, 0.005)
        self.assertEqual(config.trajectory_mode, 0)
        self.assertEqual(config.radio, 0)
        self.assertAlmostEqual(backend._active_impedance_config["requested_control_hz"], 50.0)
        self.assertAlmostEqual(backend._active_impedance_config["effective_control_hz"], 40.0)
        self.assertAlmostEqual(backend._active_impedance_config["return_torque_deadband_nm"], 0.005)
        self.assertAlmostEqual(backend._active_impedance_config["effective_return_torque_deadband_nm"], 0.005)
        self.assertEqual(backend._active_impedance_config["trajectory_mode"], 0)
        self.assertEqual(backend._active_impedance_config["radio"], 0)
        self.assertTrue(backend._active_impedance_config["control_hz_clamped"])

    def test_impedance_backend_canfd_follow_decimates_state_read_by_default(self) -> None:
        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="127.0.0.1",
                side="left",
                connect_hardware=False,
                control_dt=1.0 / 40.0,
                action_hz=20.0,
                impedance_overrides={"canfd_follow": True},
            )
        )
        config = backend._build_impedance_config(
            {
                "tool_name": "DMTAC_M",
                "impedance": {
                    "pose": {
                        "mode": "pose",
                        "control_hz": 40.0,
                        "action_hz": 20.0,
                        "state_read_every": 1,
                    }
                },
            }
        )

        self.assertEqual(config.state_read_every, 4)
        self.assertIsNotNone(backend._active_impedance_config)
        self.assertEqual(backend._active_impedance_config["state_read_every"], 4)

    def test_impedance_backend_canfd_follow_keeps_explicit_state_read_every(self) -> None:
        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="127.0.0.1",
                side="left",
                connect_hardware=False,
                control_dt=1.0 / 40.0,
                action_hz=20.0,
                impedance_overrides={"canfd_follow": True, "state_read_every": 2},
            )
        )
        config = backend._build_impedance_config(
            {
                "tool_name": "DMTAC_M",
                "impedance": {
                    "pose": {
                        "mode": "pose",
                        "control_hz": 40.0,
                        "action_hz": 20.0,
                        "state_read_every": 1,
                    }
                },
            }
        )

        self.assertEqual(config.state_read_every, 2)
        self.assertIsNotNone(backend._active_impedance_config)
        self.assertEqual(backend._active_impedance_config["state_read_every"], 2)

    def test_impedance_backend_passthrough_position_uses_action_rate_only(self) -> None:
        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="127.0.0.1",
                side="left",
                connect_hardware=False,
                control_dt=1.0 / 40.0,
                action_hz=15.0,
                impedance_overrides={
                    "canfd_follow": True,
                    "passthrough_position_control": True,
                },
            )
        )
        config = backend._build_impedance_config(
            {
                "tool_name": "DMTAC_M",
                "impedance": {
                    "pose": {
                        "mode": "pose",
                        "control_hz": 40.0,
                        "action_hz": 20.0,
                        "state_read_every": 4,
                        "joint_rate_limiter_enabled": True,
                        "max_command_joint_step_deg": 1.0,
                    }
                },
            }
        )

        self.assertAlmostEqual(config.control_hz, 15.0)
        self.assertAlmostEqual(config.action_hz, 15.0)
        self.assertTrue(config.passthrough_position_control)
        self.assertEqual(config.state_read_every, 1)
        self.assertFalse(config.joint_rate_limiter_enabled)
        self.assertGreater(config.max_command_joint_step_deg, 1.0e6)
        self.assertAlmostEqual(config.follow_qpos_alpha, 1.0)
        self.assertIsNotNone(backend._active_impedance_config)
        self.assertAlmostEqual(backend._active_impedance_config["effective_control_hz"], 15.0)
        self.assertTrue(backend._active_impedance_config["passthrough_position_control"])

    def test_pose_controller_passthrough_position_skips_force_reads(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.01, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_joint = [3.0, 1.0, 22.0, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=target_pose,
            ik_success_pose=current_pose,
            ik_success_joint=target_joint,
            force_sequence=[[9.0, 8.0, 7.0, 6.0, 5.0, 4.0]],
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=1.0,
                action_hz=15.0,
                control_hz=15.0,
                canfd_follow=True,
                canfd_follow_stream_hz=0.0,
                passthrough_position_control=True,
                joint_rate_limiter_enabled=False,
                ik_resolver_enabled=False,
                direct_joint_hint_max_linear_offset_m=0.0,
                direct_joint_hint_max_angular_offset_rad=0.0,
            ),
            robot=robot,
        )

        controller.start()
        self.assertEqual(robot.force_index, 0)
        fk_calls_after_start = robot.fk_call_count
        position_limit_calls_after_start = robot.position_limit_call_count
        self_collision_calls_after_start = robot.self_collision_call_count
        controller._last_step_time = time.perf_counter() - (1.0 / 15.0)
        record = controller.step(target_pose, target_joint_deg=target_joint)

        self.assertEqual(robot.force_index, 0)
        self.assertEqual(robot.fk_call_count, fk_calls_after_start)
        self.assertEqual(robot.position_limit_call_count, position_limit_calls_after_start)
        self.assertEqual(robot.self_collision_call_count, self_collision_calls_after_start)
        self.assertTrue(record["passthrough_position_control"])
        self.assertTrue(record["force_admittance_skipped"])
        np.testing.assert_allclose(record["selected_force_wrench"], [0.0] * 6)
        self.assertEqual(record["ik_metadata"]["method"], "passthrough_position_joint_hint")
        self.assertTrue(record["ik_metadata"]["safety_check_skipped"])
        self.assertTrue(record["ik_metadata"]["fk_skipped"])
        self.assertNotIn("singularity", record)
        self.assertEqual(record["joint_limit_margin"]["command"]["reason"], "passthrough_position_control")
        self.assertTrue(robot.movej_options[-1]["follow"])
        np.testing.assert_allclose(robot.movej_calls[-1], target_joint)

    def test_pose_controller_passthrough_position_requires_solver_qpos(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.01, 0.0, 0.35, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=current_pose,
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                action_hz=15.0,
                control_hz=15.0,
                canfd_follow=True,
                canfd_follow_stream_hz=0.0,
                passthrough_position_control=True,
                joint_rate_limiter_enabled=False,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        ik_calls_after_start = len(robot.ik_calls)
        with self.assertRaisesRegex(RuntimeError, "requires 15Hz solver_output_qpos"):
            controller.step(target_pose, target_joint_deg=None)

        self.assertEqual(len(robot.ik_calls), ik_calls_after_start)
        self.assertEqual(robot.movej_calls, [])

    def test_impedance_backend_dispatches_target_only_when_sequence_changes(self) -> None:
        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="127.0.0.1",
                side="left",
                connect_hardware=False,
                control_dt=1.0 / 40.0,
                action_hz=15.0,
            )
        )

        class FakeController:
            def __init__(self) -> None:
                self.config = SimpleNamespace(control_hz=500.0)
                self.calls: list[tuple[list[float] | None, list[float] | None]] = []

            def step(self, target, *, target_joint_deg=None):
                target_row = None if target is None else list(target)
                joint_row = None if target_joint_deg is None else list(target_joint_deg)
                self.calls.append((target_row, joint_row))
                if len(self.calls) >= 3:
                    backend._loop_stop_event.set()
                return {"fake_step": len(self.calls)}

            def stop_motion(self) -> None:
                pass

        controller = FakeController()
        backend._controller = controller  # type: ignore[assignment]
        with backend._state_lock:
            backend._latest_target_pose6 = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
            backend._latest_target_joint_deg = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            backend._latest_target_sequence = 7
            backend._last_command_timestamp = time.time()

        backend._impedance_loop()

        self.assertGreaterEqual(len(controller.calls), 3)
        self.assertEqual(controller.calls[0][0], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
        self.assertEqual(controller.calls[0][1], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertIsNone(controller.calls[1][0])
        self.assertIsNone(controller.calls[1][1])
        self.assertIsNone(controller.calls[2][0])
        self.assertIsNone(controller.calls[2][1])
        self.assertIsNotNone(backend._last_step_record)
        self.assertFalse(backend._last_step_record["backend_target_dispatched_this_step"])
        self.assertEqual(backend._last_step_record["backend_last_dispatched_target_sequence"], 7)

    def test_impedance_backend_routes_inference_soft_hold_to_controller(self) -> None:
        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="127.0.0.1",
                side="left",
                connect_hardware=False,
                control_dt=1.0 / 40.0,
            )
        )

        class FakeController:
            def __init__(self) -> None:
                self.config = SimpleNamespace(control_hz=500.0)
                self.calls: list[dict[str, object]] = []

            def step(
                self,
                target,
                *,
                target_joint_deg=None,
                soft_hold_joint_deg=None,
                soft_hold_reason="",
            ):
                self.calls.append(
                    {
                        "target": target,
                        "target_joint_deg": target_joint_deg,
                        "soft_hold_joint_deg": soft_hold_joint_deg,
                        "soft_hold_reason": soft_hold_reason,
                    }
                )
                if len(self.calls) >= 2:
                    backend._loop_stop_event.set()
                return {"fake_step": len(self.calls)}

        controller = FakeController()
        backend._controller = controller  # type: ignore[assignment]
        with backend._state_lock:
            backend._latest_target_pose6 = [0.1] * 6
            backend._latest_target_joint_deg = [1.0] * 6
            backend._latest_target_sequence = 9
            backend._policy_inference_soft_hold = True
            backend._policy_inference_soft_hold_joint_deg = [2.0] * 6
            backend._policy_inference_soft_hold_reason = "test_inference_gap"

        backend._impedance_loop()

        self.assertEqual(controller.calls[0]["target"], None)
        self.assertEqual(controller.calls[0]["target_joint_deg"], None)
        self.assertEqual(controller.calls[0]["soft_hold_joint_deg"], [2.0] * 6)
        self.assertEqual(controller.calls[0]["soft_hold_reason"], "test_inference_gap")
        self.assertTrue(backend._last_step_record["backend_policy_inference_soft_hold"])

    def test_impedance_backend_hard_stop_callback_uses_latched_emergency_stop(self) -> None:
        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="127.0.0.1",
                side="left",
                connect_hardware=False,
                control_dt=1.0 / 40.0,
            )
        )

        class FakeArm:
            def __init__(self) -> None:
                self.stop_calls = 0
                self.slow_stop_calls = 0

            def Arm_Stop(self) -> int:
                self.stop_calls += 1
                return 0

            def Arm_Slow_Stop(self) -> int:
                self.slow_stop_calls += 1
                return 0

        class FakeController:
            def __init__(self) -> None:
                self.config = SimpleNamespace(control_hz=40.0)
                self.stop_motion_calls = 0

            def step(self, target, *, target_joint_deg=None):
                del target, target_joint_deg
                raise RuntimeError(
                    "left: force 60.1000N > hard_stop_force_n"
                )

            def stop_motion(self) -> None:
                self.stop_motion_calls += 1

        arm = FakeArm()
        controller = FakeController()
        backend.arm = arm
        backend._controller = controller  # type: ignore[assignment]
        callback_calls = []

        def hard_stop_callback(side: str, message: str) -> None:
            callback_calls.append((side, message))
            backend.emergency_stop()

        backend.set_force_hard_stop_callback(hard_stop_callback)
        backend._impedance_loop()

        self.assertEqual(len(callback_calls), 1)
        self.assertEqual(callback_calls[0][0], "left")
        self.assertTrue(backend.is_emergency_stopped())
        self.assertEqual(arm.stop_calls, 1)
        self.assertEqual(arm.slow_stop_calls, 0)
        self.assertEqual(controller.stop_motion_calls, 0)
        self.assertIn("hard_stop_force_n", backend.get_state()["fault"])

    def test_pose_controller_reuses_same_joint_hint_without_resetting_interpolator(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_joint = [0.0, 3.0, 23.0, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=current_pose,
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                action_hz=10.0,
                control_hz=40.0,
                ik_resolver_enabled=False,
                direct_joint_hint_max_linear_offset_m=0.0,
                direct_joint_hint_max_angular_offset_rad=0.0,
            ),
            robot=robot,
        )
        controller.start()

        first = controller._advance_preferred_joint(target_joint, 0.025)
        second = controller._advance_preferred_joint(target_joint, 0.025)

        expected_first = [0.0, 0.75, 20.75, 0.0, 0.0, 0.0]
        expected_second = [0.0, 1.5, 21.5, 0.0, 0.0, 0.0]
        np.testing.assert_allclose(first, expected_first)
        np.testing.assert_allclose(second, expected_second)

    def test_pose_controller_follow_advances_existing_joint_hint_without_new_target(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_joint = [0.0, 4.0, 24.0, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=current_pose,
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                action_hz=10.0,
                control_hz=40.0,
                canfd_follow=True,
                follow_qpos_alpha=1.0,
                canfd_follow_stream_hz=0.0,
                joint_rate_limiter_enabled=False,
                ik_resolver_enabled=False,
                direct_joint_hint_max_linear_offset_m=0.0,
                direct_joint_hint_max_angular_offset_rad=0.0,
            ),
            robot=robot,
        )
        controller.start()

        first = controller._advance_preferred_joint(target_joint, 0.025)
        second = controller._advance_preferred_joint(None, 0.025)

        np.testing.assert_allclose(first, [0.0, 1.0, 21.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(second, [0.0, 2.0, 22.0, 0.0, 0.0, 0.0])

    def test_pose_controller_fresh_cartesian_target_without_hint_clears_stale_joint_hint(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        stale_target_joint = [0.0, 4.0, 24.0, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=current_pose,
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                action_hz=10.0,
                control_hz=40.0,
                joint_rate_limiter_enabled=False,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )
        controller.start()
        self.assertIsNotNone(controller._advance_preferred_joint(stale_target_joint, 0.025))

        # A new Cartesian pose with no matching solver qpos must use that pose
        # for IK, not FK of the old interpolating joint hint.
        controller._last_step_time = time.perf_counter() - 0.025
        record = controller.step(current_pose, target_joint_deg=None)

        self.assertIsNone(controller._joint_hint_target)
        self.assertIsNone(controller._advance_preferred_joint(None, 0.025))
        self.assertEqual(
            record["joint_hint_invalidation"]["reason"],
            "fresh_cartesian_target_without_solver_qpos",
        )
        self.assertNotEqual(record["ik_method"], "dual_arm_joint_hint")

    def test_pose_controller_non_follow_advances_existing_joint_hint_without_new_target(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_joint = [0.0, 4.0, 24.0, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=current_pose,
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                action_hz=10.0,
                control_hz=40.0,
                joint_rate_limiter_enabled=False,
                ik_resolver_enabled=False,
                direct_joint_hint_max_linear_offset_m=0.0,
                direct_joint_hint_max_angular_offset_rad=0.0,
            ),
            robot=robot,
        )
        controller.start()

        first = controller._advance_preferred_joint(target_joint, 0.025)
        second = controller._advance_preferred_joint(None, 0.025)

        np.testing.assert_allclose(first, [0.0, 1.0, 21.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(second, [0.0, 2.0, 22.0, 0.0, 0.0, 0.0])

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
        self.assertIn("-0.040000, 0.000000, 0.000000]", text)
        self.assertRegex(text, r"offset_norms=linear:0\.01000[0-9]m angular:0\.01000[0-9]rad")
        self.assertIn("deadbanded_wrench=[0.60000, 0.00000, 0.00000, 0.10000, 0.00000, 0.00000]", text)
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

    def test_pose_controller_reset_motion_state_reanchors_reference_to_current_qpos(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        reset_joint_deg = [1.0, 2.0, 21.0, 4.0, 5.0, 6.0]
        reset_pose = [0.03, -0.02, 0.36, 0.01, 0.02, -0.03]
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
                joint_rate_limiter_enabled=False,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        controller._reference.set_target([0.2, 0.0, 0.35, 0.0, 0.0, 0.0])
        controller._joint_hint_reference.set_target([10.0, 0.0, 20.0, 0.0, 0.0, 0.0])
        controller._latest_wrench_context = {"stale": True}
        controller._step_index = 7
        robot.model_fk_pose = reset_pose

        controller.reset_motion_state(np.deg2rad(reset_joint_deg))

        np.testing.assert_allclose(controller._previous_command_joint, reset_joint_deg)
        np.testing.assert_allclose(controller._ik_seed, reset_joint_deg)
        self.assertEqual(controller._previous_command_pose, reset_pose)
        self.assertEqual(controller._reference.current, reset_pose)
        self.assertEqual(controller._reference.target, reset_pose)
        np.testing.assert_allclose(controller._joint_hint_reference.current, reset_joint_deg)
        np.testing.assert_allclose(controller._joint_hint_reference.target, reset_joint_deg)
        self.assertIsNone(controller._latest_wrench_context)
        self.assertEqual(controller._step_index, 0)

    def test_impedance_backend_reset_motion_state_clears_targets_and_resets_controller(self) -> None:
        class FakeController:
            def __init__(self) -> None:
                self.reset_qpos: np.ndarray | None = None

            def reset_motion_state(self, qpos) -> None:
                self.reset_qpos = np.asarray(qpos, dtype=float).copy()

        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="192.0.2.10",
                side="right",
                connect_hardware=False,
            )
        )
        controller = FakeController()
        backend._controller = controller
        backend._latest_target_pose6 = [0.1] * 6
        backend._latest_target_joint_deg = [1.0] * 6
        backend._last_command = object()
        backend._last_command_timestamp = 123.0
        backend._last_step_record = {"old": True}
        backend._last_send_time = 456.0
        qpos = np.linspace(0.0, 0.5, 6)

        backend.reset_motion_state(qpos)

        self.assertIsNone(backend._latest_target_pose6)
        self.assertIsNone(backend._latest_target_joint_deg)
        self.assertIsNone(backend._last_command)
        self.assertIsNone(backend._last_command_timestamp)
        self.assertIsNone(backend._last_step_record)
        self.assertIsNone(backend._last_send_time)
        self.assertEqual(backend._last_send_result_text, "reset_motion_state")
        np.testing.assert_allclose(controller.reset_qpos, qpos)

    def test_impedance_backend_warm_start_initializes_controller_without_enabling_loop(self) -> None:
        class FakeController:
            def __init__(self) -> None:
                self._started = False
                self.start_count = 0

            def start(self) -> None:
                self.start_count += 1
                self._started = True

        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="192.0.2.10",
                side="right",
                connect_hardware=True,
            )
        )
        controller = FakeController()
        backend.arm = object()
        backend._controller = controller

        self.assertFalse(backend.force_control_warm_started())
        backend.warm_start_force_control()

        state = backend.get_state()
        self.assertTrue(backend.force_control_warm_started())
        self.assertFalse(state["impedance_enabled"])
        self.assertEqual(state["impedance_loop_state"], "not_started")
        self.assertEqual(state["send_result_text"], "warm_started")

        backend.warm_start_force_control()
        self.assertEqual(controller.start_count, 1)

    def test_impedance_backend_exact_qpos_lease_rejects_unrelated_commands(self) -> None:
        class FakeController:
            def __init__(self) -> None:
                self.begin_calls: list[str] = []
                self.end_calls: list[str] = []

            def begin_exact_qpos_transit(self, operation_id: str) -> None:
                self.begin_calls.append(operation_id)

            def end_exact_qpos_transit(self, operation_id: str) -> None:
                self.end_calls.append(operation_id)

        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="192.0.2.10",
                side="right",
                connect_hardware=False,
            )
        )
        controller = FakeController()
        backend._controller = controller
        backend._enabled = True
        backend.begin_exact_qpos_transit("operation-lease")

        ordinary = EffectorCommand(
            effector_name="right_arm",
            side="right",
            command_type="cartesian_pose",
            value=np.eye(4),
            metadata={"solver_output_qpos": [0.0] * 6},
        )
        with self.assertRaisesRegex(RuntimeError, "only accepts commands"):
            backend.send_command(ordinary)

        exact = ordinary.copy()
        exact.metadata.update(
            {
                "init_arm_qpos_active": True,
                "init_arm_qpos_operation_id": "operation-lease",
            }
        )
        backend.send_command(exact)

        self.assertEqual(controller.begin_calls, ["operation-lease"])
        self.assertTrue(backend.exact_qpos_transit_active)
        self.assertTrue(backend.end_exact_qpos_transit("operation-lease"))
        self.assertEqual(controller.end_calls, ["operation-lease"])
        self.assertFalse(backend.exact_qpos_transit_active)

    def test_impedance_backend_exact_qpos_preflight_rejects_contact(self) -> None:
        class FakeController:
            def __init__(self) -> None:
                self.begin_calls: list[str] = []

            def begin_exact_qpos_transit(self, operation_id: str) -> None:
                self.begin_calls.append(operation_id)

        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="192.0.2.10",
                side="right",
                connect_hardware=True,
            )
        )
        controller = FakeController()
        backend._controller = controller
        backend._enabled = True
        backend._last_step_record = {
            "backend_controller_step_end_wall_time": time.time(),
            "force_norm_n": 21.0,
            "torque_norm_nm": 0.0,
        }

        with self.assertRaisesRegex(RuntimeError, "under contact"):
            backend.begin_exact_qpos_transit("operation-contact")

        self.assertEqual(controller.begin_calls, [])
        self.assertFalse(backend.exact_qpos_transit_active)

    def test_impedance_backend_send_command_does_not_wait_for_step_lock(self) -> None:
        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="192.0.2.10",
                side="right",
                connect_hardware=False,
                debug_timing=True,
            )
        )
        command = EffectorCommand(
            effector_name="right_arm",
            side="right",
            command_type="cartesian_pose",
            value=np.eye(4),
            metadata={"solver_output_qpos": [0.0] * 6},
        )
        done = threading.Event()
        errors: list[BaseException] = []

        def send_in_thread() -> None:
            try:
                backend.send_command(command)
            except BaseException as exc:  # noqa: BLE001 - propagate from worker.
                errors.append(exc)
            finally:
                done.set()

        backend._send_lock.acquire()
        try:
            thread = threading.Thread(target=send_in_thread)
            start = time.perf_counter()
            thread.start()
            completed_while_locked = done.wait(timeout=0.1)
            elapsed = time.perf_counter() - start
        finally:
            backend._send_lock.release()
        thread.join(timeout=1.0)

        if errors:
            raise errors[0]
        self.assertTrue(completed_while_locked, "send_command waited for the impedance step lock")
        self.assertLess(elapsed, 0.1)
        self.assertEqual(backend._latest_target_pose6, [0.0] * 6)
        self.assertEqual(backend._latest_target_joint_deg, [0.0] * 6)
        state = backend.get_state()
        timing = state["last_send_command_timing"]
        self.assertIsNotNone(timing)
        self.assertEqual(timing["send_lock_wait_ms"], 0.0)
        self.assertIn("send_lock_wait", [phase["name"] for phase in timing["phases"]])
        self.assertEqual(state["last_command_controlled_frame"], "tool_tcp")
        self.assertEqual(state["latest_target_tcp_pose6"], [0.0] * 6)
        self.assertEqual(state["latest_target_joint_deg"], [0.0] * 6)
        self.assertIsNotNone(state["last_command"])

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

        self.assertEqual(record["command_pose"], current_pose)
        self.assertEqual(record["command_joint_deg"], target_joint)
        self.assertEqual(record["ik_method"], "dual_arm_joint_hint")
        self.assertEqual(record["ik_metadata"]["source"], "retargeter_solver_output_qpos")
        self.assertEqual(robot.movej_calls[-1], target_joint)
        self.assertEqual(len(robot.ik_calls), preflight_ik_calls)

    def test_exact_qpos_transit_suspends_compliance_but_keeps_force_monitoring(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_joint = [1.0, 2.0, 20.0, 3.0, 4.0, 5.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            force_sequence=[[0.0] * 6, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                lowpass_alpha=1.0,
                force_deadband_n=0.0,
                linear_compliance_m_per_n=0.004,
                max_linear_velocity_m_s=1.0,
                max_command_joint_velocity_deg_s=1000.0,
                joint_rate_limiter_enabled=False,
                ik_resolver_enabled=False,
                direct_joint_hint_max_linear_offset_m=0.0,
                direct_joint_hint_max_angular_offset_rad=0.0,
                direct_joint_hint_force_guard_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        controller.begin_exact_qpos_transit("operation-1")
        controller._last_step_time = time.perf_counter() - 0.2
        record = controller.step(current_pose, target_joint_deg=target_joint)

        self.assertTrue(record["exact_qpos_transit_active"])
        self.assertEqual(record["exact_qpos_transit_operation_id"], "operation-1")
        self.assertEqual(record["force_norm_n"], 10.0)
        self.assertEqual(record["command_joint_deg"], target_joint)
        self.assertEqual(record["applied_offset"], [0.0] * 6)
        self.assertEqual(record["offset"], [0.0] * 6)
        self.assertEqual(
            record["ik_metadata"]["direct_joint_hint_reason"],
            "exact_qpos_transit",
        )
        self.assertTrue(controller.end_exact_qpos_transit("operation-1"))
        self.assertFalse(controller.exact_qpos_transit_active)

    def test_exact_qpos_transit_does_not_bypass_force_hard_stop(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            force_sequence=[[0.0] * 6, [60.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                lowpass_alpha=1.0,
                force_deadband_n=0.0,
                max_command_joint_velocity_deg_s=1000.0,
                joint_rate_limiter_enabled=False,
                ik_resolver_enabled=False,
                hard_stop_force_n=50.0,
            ),
            robot=robot,
        )

        controller.start()
        controller.begin_exact_qpos_transit("operation-2")
        with self.assertRaisesRegex(RuntimeError, "hard_stop_force_n"):
            controller.step(current_pose, target_joint_deg=current_joint)

    def test_pose_controller_rate_limits_anomalous_prelimited_joint_hint(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.02, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_joint = [22.0, 0.0, 20.0, 0.0, 0.0, 0.0]
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
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        preflight_ik_calls = len(robot.ik_calls)
        controller._last_step_time = time.perf_counter() - 0.2
        record = controller.step(target_pose, target_joint_deg=target_joint)

        self.assertEqual(record["command_pose"], current_pose)
        expected_joint = [2.125, 0.0, 20.0, 0.0, 0.0, 0.0]
        np.testing.assert_allclose(record["command_joint_deg"], expected_joint, atol=1e-3)
        self.assertTrue(record["joint_rate_limited"])
        self.assertIsNone(record["joint_rate_limiter_bypassed"])
        self.assertEqual(record["joint_safety"]["status_text"], "position_limit=ok self_collision=ok")
        np.testing.assert_allclose(robot.movej_calls[-1], expected_joint, atol=1e-3)
        self.assertEqual(len(robot.ik_calls), preflight_ik_calls)
        self.assertIsNotNone(controller._joint_rate_limiter)
        np.testing.assert_allclose(
            np.rad2deg(controller._joint_rate_limiter.command_qpos),
            expected_joint,
            atol=1e-3,
        )

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

        self.assertEqual(command_pose, current_pose)
        self.assertEqual(command_joint, target_joint)
        self.assertEqual(controller._latest_wrench_context["ik_method"], ["dual_arm_joint_hint"])
        self.assertTrue(
            controller._latest_wrench_context["ik_metadata"]["direct_joint_hint_offset_ignored"]
        )
        self.assertEqual(len(robot.ik_calls), preflight_ik_calls)

    def test_pose_controller_force_guard_prefers_joint_hint_over_backend_ik(self) -> None:
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
            force_sequence=[[0.0] * 6, [30.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                lowpass_alpha=1.0,
                force_deadband_n=0.0,
                linear_compliance_m_per_n=0.004,
                max_linear_velocity_m_s=1.0,
                max_command_joint_velocity_deg_s=1000.0,
                joint_rate_limiter_enabled=False,
                ik_resolver_enabled=False,
                direct_joint_hint_max_linear_offset_m=1e-3,
                direct_joint_hint_force_guard_force_n=20.0,
            ),
            robot=robot,
        )

        controller.start()
        preflight_ik_calls = len(robot.ik_calls)
        controller._last_step_time = time.perf_counter() - 0.2
        record = controller.step(target_pose, target_joint_deg=target_joint)

        self.assertEqual(record["command_joint_deg"], target_joint)
        self.assertEqual(record["ik_method"], "dual_arm_joint_hint")
        self.assertEqual(len(robot.ik_calls), preflight_ik_calls)
        metadata = record["ik_metadata"]
        self.assertEqual(metadata["direct_joint_hint_reason"], "force_guard")
        self.assertTrue(metadata["direct_joint_hint_force_guard"]["active"])
        # A delayed force frame still integrates only one nominal 40 Hz
        # admittance step; it must not catch up with a larger Cartesian offset.
        self.assertAlmostEqual(record["requested_offset_before_command_realization"][0], 0.025)
        self.assertAlmostEqual(record["admittance_observed_dt_sec"], 0.2, places=2)
        self.assertEqual(record["admittance_effective_dt_sec"], 0.025)
        self.assertEqual(record["admittance_dt_policy"], "nominal_control_period")
        self.assertEqual(record["applied_offset"], [0.0] * 6)
        self.assertEqual(record["offset"], [0.0] * 6)
        self.assertFalse(record["admittance_offset_antiwindup_applied"])
        self.assertEqual(record["admittance_offset_antiwindup_skip_reason"], "direct_joint_hint")

    def test_pose_controller_can_pause_return_below_active_deadband(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        held_offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
        held_pose = apply_pose_offset(current_pose, held_offset)
        held_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 2.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=held_pose,
            ik_success_joint=held_joint,
            ik_retry_pose=current_pose,
            force_sequence=[
                [0.0] * 6,
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.01],
            ],
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                axes="rz",
                lowpass_alpha=1.0,
                torque_deadband_nm=0.03,
                return_torque_deadband_nm=0.005,
                angular_compliance_rad_per_nm=0.1,
                max_angular_velocity_rad_s=1.0,
                max_angular_offset_rad=1.2,
                max_command_joint_velocity_deg_s=1000.0,
                joint_rate_limiter_enabled=False,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        controller._admittance.offset = list(held_offset)
        controller._last_step_time = time.perf_counter() - 0.2
        record = controller.step(current_pose)

        self.assertEqual(record["active_wrench"], [0.0] * 6)
        self.assertAlmostEqual(record["filtered_wrench"][5], 0.01)
        self.assertAlmostEqual(record["return_activity_wrench"][5], 0.01)
        self.assertAlmostEqual(record["return_torque_deadband_nm"], 0.005)
        self.assertEqual(record["return_applied_axes"], [])
        self.assertEqual(record["return_paused_axes"], ["rz"])
        self.assertAlmostEqual(record["offset"][5], 0.2)
        self.assertEqual(record["command_joint_deg"], held_joint)

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
        self.assertTrue(np.allclose(record["command_joint_deg"], [0.0, 0.0, 20.0, 2.625, 0.0, 0.0], atol=1e-2))
        self.assertAlmostEqual(record["command_joint_step_deg"], 2.625, places=2)
        self.assertAlmostEqual(record["allowed_command_joint_step_deg"], 3.0, places=2)
        self.assertAlmostEqual(record["joint_velocity_limit_observed_dt_sec"], 0.2, places=2)
        self.assertAlmostEqual(record["joint_velocity_limit_effective_dt_sec"], 0.025, places=3)
        self.assertEqual(record["joint_velocity_limit_max_dt_sec"], 0.025)
        self.assertEqual(record["joint_velocity_limit_dt_policy"], "nominal_control_period")
        self.assertEqual(record["joint_safety"]["status_text"], "position_limit=ok self_collision=ok")
        self.assertTrue(np.allclose(robot.movej_calls[-1], [0.0, 0.0, 20.0, 2.625, 0.0, 0.0], atol=1e-2))

    def test_pose_controller_absolute_step_cap_is_final_upper_bound(self) -> None:
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
                max_command_joint_step_deg=1.0,
                max_command_joint_velocity_deg_s=1000.0,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        controller._last_step_time = time.perf_counter() - 0.2
        record = controller.step(target_pose)

        self.assertAlmostEqual(record["allowed_command_joint_step_deg"], 1.0, places=6)
        self.assertAlmostEqual(record["command_joint_step_deg"], 1.0, places=6)
        self.assertTrue(record["final_command_step_guard_applied"])
        np.testing.assert_allclose(robot.movej_calls[-1], [0.0, 0.0, 20.0, 1.0, 0.0, 0.0])

    def test_pose_controller_step_policy_holds_when_prelimited_hint_cannot_be_bounded(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_joint = [0.0, 0.0, 20.0, 37.23, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_retry_pose=current_pose,
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
        record = controller.step(current_pose, target_joint_deg=target_joint)

        expected_joint = current_joint
        self.assertIsNone(record["joint_rate_limited"])
        self.assertFalse(record["final_command_step_guard_applied"])
        self.assertEqual(record["command_joint_step_deg"], 0.0)
        self.assertEqual(record["allowed_command_joint_step_deg"], 20.0)
        self.assertTrue(np.allclose(record["command_joint_deg"], expected_joint, atol=1e-6))
        self.assertTrue(np.allclose(robot.movej_calls[-1], expected_joint, atol=1e-6))

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
            ik_success_joint=[0.0, 0.0, 20.0, 140.0, 0.0, 0.0],
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
        self.assertTrue(np.allclose(record["command_joint_deg"], [0.0, 0.0, 20.0, 2.625, 0.0, 0.0], atol=1e-6))
        self.assertAlmostEqual(record["joint_velocity_limit_observed_dt_sec"], 2.0, places=2)
        self.assertEqual(record["joint_velocity_limit_effective_dt_sec"], 0.025)
        self.assertEqual(record["joint_velocity_limit_max_dt_sec"], 0.025)
        self.assertEqual(record["joint_velocity_limit_dt_policy"], "nominal_control_period")

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
                ik_hard_deadline_enabled=False,
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
        self.assertNotIn("singularity", record)
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

    def test_pose_controller_debug_timing_records_sdk_phases(self) -> None:
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
                debug_timing=True,
            ),
            robot=robot,
        )

        controller.start()
        controller._last_step_time = time.perf_counter() - 0.2
        record = controller.step(target_pose)

        timing = record["rm65_controller_timing"]
        self.assertTrue(timing["enabled"])
        self.assertGreaterEqual(timing["total_ms"], 0.0)
        phase_names = [phase["name"] for phase in timing["phases"]]
        for expected in (
            "read_force",
            "command_from_reference",
            "send",
            "state_read_after_send",
        ):
            self.assertIn(expected, phase_names)

    def test_pose_controller_records_tracking_direction_and_closing_speed(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        measured_joint = [0.2, 0.0, 20.0, 0.0, 0.0, 0.0]
        measured_pose = [0.01, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.03, 0.0, 0.35, 0.0, 0.0, 0.0]
        command_joint = [1.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=target_pose,
            ik_success_joint=command_joint,
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
        robot.current_joint = list(measured_joint)
        robot.current_pose = list(measured_pose)
        controller._last_step_time = time.perf_counter() - 0.2
        record = controller.step(target_pose)

        np.testing.assert_allclose(record["measured_to_command_linear_m"], [0.02, 0.0, 0.0])
        self.assertEqual(record["force_source"], "work")
        self.assertIn("work", record["force_readings"])
        np.testing.assert_allclose(record["selected_force_wrench"], [0.0] * 6)
        np.testing.assert_allclose(record["filtered_wrench"], [0.0] * 6)
        np.testing.assert_allclose(record["command_to_measured_linear_m"], [-0.02, 0.0, 0.0])
        np.testing.assert_allclose(record["reference_to_command_linear_m"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(record["reference_to_measured_linear_m"], [-0.02, 0.0, 0.0])
        np.testing.assert_allclose(
            record["measured_to_command_joint_deg"],
            [0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.assertGreater(record["tracking_error_closing_linear_m_s"], 0.0)
        self.assertGreater(record["tracking_error_closing_joint_deg_s"], 0.0)

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

    def test_pose_controller_movej_canfd_defaults_to_low_follow_without_final_smoothing(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.01, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_joint = [10.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=target_pose,
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
                joint_rate_limiter_enabled=False,
                max_command_joint_step_deg=1000.0,
                max_command_joint_velocity_deg_s=1000.0,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        controller._last_step_time = time.perf_counter() - 0.2
        record = controller.step(target_pose)

        np.testing.assert_allclose(robot.movej_calls[-1], target_joint)
        self.assertFalse(robot.movej_options[-1]["follow"])
        self.assertFalse(record["follow_qpos_smoothing"]["enabled"])
        self.assertFalse(record["follow_qpos_smoothing"]["applied"])
        np.testing.assert_allclose(record["command_joint_deg"], target_joint)

    def test_pose_controller_canfd_follow_does_not_smooth_joint_hint_before_interpolation(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.01, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_joint = [10.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=target_pose,
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
                canfd_follow=True,
                follow_qpos_alpha=0.9,
                canfd_follow_stream_hz=0.0,
                joint_rate_limiter_enabled=False,
                max_command_joint_step_deg=1000.0,
                max_command_joint_velocity_deg_s=1000.0,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        controller._last_step_time = time.perf_counter() - 0.2
        record = controller.step(target_pose, target_joint_deg=target_joint)

        np.testing.assert_allclose(robot.movej_calls[-1], target_joint)
        self.assertTrue(robot.movej_options[-1]["follow"])
        self.assertFalse(record["follow_qpos_smoothing"]["enabled"])
        self.assertFalse(record["follow_qpos_smoothing"]["applied"])
        self.assertEqual(record["follow_qpos_smoothing"]["alpha"], 0.9)
        self.assertEqual(record["follow_qpos_smoothing"]["stage"], "pre_interpolation_target")
        self.assertEqual(record["follow_qpos_smoothing"]["final_command_stage"], "passthrough")
        self.assertTrue(record["follow_qpos_smoothing"]["final_command_realized_from_joint"])
        np.testing.assert_allclose(record["follow_qpos_smoothing"]["raw_target_joint_deg"], target_joint)
        np.testing.assert_allclose(record["follow_qpos_smoothing"]["output_target_joint_deg"], target_joint)
        np.testing.assert_allclose(
            record["pre_follow_smoothing_command_joint_deg"],
            target_joint,
        )
        np.testing.assert_allclose(record["command_joint_deg"], target_joint)

    def test_pose_controller_canfd_follow_rate_limits_prelimited_joint_hint_target(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.01, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_joint = [10.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=target_pose,
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
                canfd_follow=True,
                follow_qpos_alpha=1.0,
                canfd_follow_stream_hz=0.0,
                max_command_joint_velocity_deg_s=1000.0,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        controller._last_step_time = time.perf_counter() - 0.2
        record = controller.step(target_pose, target_joint_deg=target_joint)

        np.testing.assert_allclose(robot.movej_calls[-1], [2.125, 0.0, 20.0, 0.0, 0.0, 0.0])
        self.assertTrue(robot.movej_options[-1]["follow"])
        smoothing = record["follow_qpos_smoothing"]
        self.assertFalse(smoothing["enabled"])
        self.assertFalse(smoothing["target_rate_limiter_enabled"])
        self.assertTrue(smoothing["target_rate_limiter_bypassed"])
        self.assertEqual(
            smoothing["target_rate_limiter_bypass_reason"],
            "prelimited_15hz_solver_qpos",
        )
        np.testing.assert_allclose(smoothing["output_target_joint_deg"], target_joint)
        np.testing.assert_allclose(
            record["pre_follow_smoothing_command_joint_deg"],
            [2.125, 0.0, 20.0, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(record["command_joint_deg"], [2.125, 0.0, 20.0, 0.0, 0.0, 0.0])

    def test_pose_controller_canfd_follow_streams_final_movej_qpos(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.01, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_joint = [10.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=target_pose,
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
                canfd_follow=True,
                follow_qpos_alpha=0.9,
                canfd_follow_stream_hz=200.0,
                joint_rate_limiter_enabled=False,
                max_command_joint_step_deg=1000.0,
                max_command_joint_velocity_deg_s=1000.0,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        try:
            controller.start()
            controller._last_step_time = time.perf_counter() - 0.2
            record = controller.step(target_pose, target_joint_deg=target_joint)
            deadline = time.perf_counter() + 0.2
            while time.perf_counter() < deadline:
                if any(np.allclose(call, target_joint, atol=1e-6) for call in robot.movej_calls):
                    break
                time.sleep(0.005)
        finally:
            controller.close()

        self.assertTrue(record["canfd_follow_stream"]["enabled"])
        self.assertTrue(record["canfd_follow_stream"]["thread_alive"])
        self.assertTrue(record["canfd_follow_stream"]["target_updated"])
        self.assertEqual(record["canfd_follow_stream"]["updated_target_sequence"], 1)
        self.assertEqual(record["follow_qpos_smoothing"]["stage"], "pre_interpolation_target")
        self.assertEqual(record["follow_qpos_smoothing"]["final_command_stage"], "passthrough")
        self.assertGreaterEqual(len(robot.movej_calls), 1)
        self.assertTrue(all(option["follow"] for option in robot.movej_options))
        self.assertTrue(any(np.allclose(call, target_joint, atol=1e-6) for call in robot.movej_calls))

    def test_canfd_follow_stream_drops_missed_deadlines_instead_of_bursting(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_pose = [0.01, 0.0, 0.35, 0.0, 0.0, 0.0]
        target_joint = [10.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=target_pose,
            ik_success_pose=target_pose,
            ik_success_joint=target_joint,
            ik_retry_pose=current_pose,
        )
        robot.movej_delay_sec = 0.015
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                lowpass_alpha=1.0,
                canfd_follow=True,
                follow_qpos_alpha=1.0,
                canfd_follow_stream_hz=100.0,
                joint_rate_limiter_enabled=False,
                max_command_joint_step_deg=1000.0,
                max_command_joint_velocity_deg_s=1000.0,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        try:
            controller.start()
            controller._last_step_time = time.perf_counter() - 0.2
            controller.step(target_pose, target_joint_deg=target_joint)
            deadline = time.perf_counter() + 0.15
            while time.perf_counter() < deadline and len(robot.movej_call_times) < 3:
                time.sleep(0.005)
            stream_snapshot = controller._follow_stream_snapshot()
        finally:
            controller.close()

        intervals = [
            second - first
            for first, second in zip(robot.movej_call_times, robot.movej_call_times[1:])
        ]
        self.assertGreaterEqual(len(intervals), 2)
        self.assertGreater(min(intervals), 0.020)
        self.assertIsNotNone(stream_snapshot)
        assert stream_snapshot is not None
        self.assertIsNotNone(stream_snapshot["actual_hz"])
        self.assertLess(stream_snapshot["actual_hz"], stream_snapshot["warning_threshold_hz"])
        self.assertGreaterEqual(stream_snapshot["low_rate_count"], 1)
        self.assertTrue(stream_snapshot["warning"]["active"])
        self.assertIn("below target", stream_snapshot["warning"]["message"])

    def test_canfd_follow_stream_interpolates_over_action_period(self) -> None:
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                control_hz=40.0,
                action_hz=15.0,
                canfd_follow=True,
                canfd_follow_stream_hz=120.0,
            ),
            robot=None,
        )
        controller._follow_stream_target_joint = [15.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        controller._follow_stream_segment_start_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        controller._follow_stream_segment_start_time = 100.0
        controller._follow_stream_target_sequence = 3

        joint, sequence, alpha = controller._follow_stream_command_at(100.025)

        self.assertEqual(sequence, 3)
        self.assertAlmostEqual(alpha, 0.375, places=6)
        np.testing.assert_allclose(joint, [5.625, 0.0, 20.0, 0.0, 0.0, 0.0], atol=1e-9)

    def test_canfd_follow_stop_retains_and_faults_live_worker(self) -> None:
        class StuckThread:
            def __init__(self) -> None:
                self.join_timeouts: list[float | None] = []

            def is_alive(self) -> bool:
                return True

            def join(self, timeout: float | None = None) -> None:
                self.join_timeouts.append(timeout)

        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(tool_name="DMTAC_L"),
            robot=None,
        )
        worker = StuckThread()
        controller._follow_stream_thread = worker

        with self.assertRaisesRegex(RuntimeError, "did not stop within 1.000s"):
            controller._stop_follow_stream()

        self.assertIs(controller._follow_stream_thread, worker)
        self.assertTrue(controller.follow_stream_worker_alive())
        self.assertEqual(worker.join_timeouts, [1.0])
        self.assertIn("did not stop", str(controller._follow_stream_fault))

    def test_backend_close_refuses_socket_close_with_live_follow_worker(self) -> None:
        class LiveWorkerController:
            def stop_motion(self) -> None:
                pass

            def close(self) -> None:
                pass

            def follow_stream_worker_alive(self) -> bool:
                return True

        class RecordingArm:
            def __init__(self) -> None:
                self.close_count = 0

            def Arm_Socket_Close(self) -> None:
                self.close_count += 1

        backend = RM65ImpedanceArmBackend(
            RM65ImpedanceArmBackendConfig(
                arm_ip="127.0.0.1",
                side="right",
                connect_hardware=False,
            )
        )
        arm = RecordingArm()
        backend._controller = LiveWorkerController()
        backend.arm = arm

        with self.assertRaisesRegex(RuntimeError, "refuses to close the RM65 socket"):
            backend.close()

        self.assertEqual(arm.close_count, 0)
        self.assertIs(backend.arm, arm)

    def test_pose_controller_records_offset_command_realization_metrics(self) -> None:
        current_joint = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
        current_pose = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        offset = [0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
        command_pose = apply_pose_offset(current_pose, offset)
        robot = _FakeRobot(
            tool_name="DMTAC_L",
            tool_pose=[0.0] * 6,
            current_joint=current_joint,
            current_pose=current_pose,
            model_fk_pose=current_pose,
            ik_success_pose=command_pose,
            ik_success_joint=[1.0, 0.0, 20.0, 0.0, 0.0, 0.0],
            ik_retry_pose=current_pose,
            force_sequence=[
                [0.0] * 6,
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                lowpass_alpha=1.0,
                force_deadband_n=0.0,
                linear_compliance_m_per_n=0.01,
                max_linear_velocity_m_s=1.0,
                max_command_joint_velocity_deg_s=1000.0,
                joint_rate_limiter_enabled=False,
                ik_resolver_enabled=False,
            ),
            robot=robot,
        )

        controller.start()
        controller._last_step_time = time.perf_counter() - 0.2
        record = controller.step(current_pose)

        np.testing.assert_allclose(record["applied_offset"], offset)
        self.assertAlmostEqual(record["admittance_offset_linear_norm_m"], 0.02)
        self.assertAlmostEqual(record["commanded_offset_linear_norm_m"], 0.02)
        self.assertAlmostEqual(record["offset_to_command_linear_error_m"], 0.0)
        self.assertAlmostEqual(record["offset_to_command_linear_fraction"], 1.0)
        self.assertAlmostEqual(record["target_to_command_linear_fraction"], 1.0)
        self.assertAlmostEqual(record["measured_offset_linear_norm_m"], 0.0)
        self.assertAlmostEqual(record["command_to_measured_linear_norm_m"], 0.02)

    def test_pose_controller_syncs_admittance_to_relaxed_command_offset(self) -> None:
        reference = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        requested_offset = [0.05, 0.0, 0.0, 0.0, 0.0, 0.4]
        realized_offset = [0.03, 0.0, 0.0, 0.0, 0.0, 0.8]
        clamped_realized_offset = [0.03, 0.0, 0.0, 0.0, 0.0, 0.4]
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                joint_rate_limiter_enabled=False,
                max_command_joint_velocity_deg_s=1000.0,
                max_linear_offset_m=0.05,
                max_angular_offset_rad=0.6,
            ),
            robot=None,
        )
        controller._ik_resolver = object()
        controller._ik_seed = [0.0] * 6
        controller._previous_command_joint = [0.0] * 6
        controller._previous_command_pose = list(reference)
        controller._current_step_dt = 0.1
        controller._admittance = PoseAdmittance(
            enabled_axes=[True] * 6,
            max_angular_offset_rad=1.0,
            offset=list(requested_offset),
        )
        controller._latest_wrench_context = {
            "offset": list(requested_offset),
            "active_wrench": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "return_applied_axes": ["ry"],
        }

        def relaxed_solver(**kwargs):
            return (
                apply_pose_offset(kwargs["reference_pose"], realized_offset),
                [0.0] * 6,
                [0.0] * 6,
                1.0,
                list(kwargs["offset"]),
                0,
            )

        controller._command_from_pose_with_resolver = relaxed_solver  # type: ignore[method-assign]

        command_pose, command_joint = controller._command_from_reference(reference, requested_offset)

        self.assertEqual(command_joint, [0.0] * 6)
        np.testing.assert_allclose(command_pose, apply_pose_offset(reference, realized_offset))
        np.testing.assert_allclose(controller._admittance.offset, clamped_realized_offset, atol=1e-9)
        np.testing.assert_allclose(
            controller._latest_wrench_context["applied_offset"],
            clamped_realized_offset,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            controller._latest_wrench_context["requested_offset_before_command_realization"],
            requested_offset,
            atol=1e-9,
        )
        self.assertTrue(controller._latest_wrench_context["admittance_offset_antiwindup_applied"])
        self.assertEqual(controller._latest_wrench_context["admittance_offset_antiwindup_skip_reason"], "")
        self.assertEqual(
            controller._latest_wrench_context["admittance_offset_antiwindup_active_axes"],
            ["x"],
        )
        self.assertEqual(controller._latest_wrench_context["applied_offset_source"], "final_command_pose")
        self.assertGreater(
            controller._latest_wrench_context["applied_offset_realization_linear_error_m"],
            0.01,
        )
        self.assertAlmostEqual(
            controller._latest_wrench_context["applied_offset_realization_angular_error_rad"],
            0.0,
        )

    def test_pose_controller_antiwindup_syncs_only_active_axes(self) -> None:
        reference = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        requested_offset = [0.02, 0.0, 0.0, 0.0, 0.0, 0.2]
        realized_offset = [-0.04, 0.03, 0.0, 0.0, 0.0, 0.3]
        expected_offset = [0.02, 0.0, 0.0, 0.0, 0.0, 0.3]
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                joint_rate_limiter_enabled=False,
                max_command_joint_velocity_deg_s=1000.0,
                max_linear_offset_m=0.08,
                max_angular_offset_rad=0.6,
            ),
            robot=None,
        )
        controller._admittance = PoseAdmittance(
            enabled_axes=[True] * 6,
            max_angular_offset_rad=1.0,
            offset=list(requested_offset),
        )
        controller._latest_wrench_context = {
            "active_wrench": [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            "target_offset_after_offset_limit": list(requested_offset),
        }

        realized = controller._sync_admittance_to_realized_pose_offset(
            reference_pose=reference,
            command_pose=apply_pose_offset(reference, realized_offset),
            requested_offset=requested_offset,
        )

        np.testing.assert_allclose(realized, expected_offset, atol=1e-9)
        np.testing.assert_allclose(controller._admittance.offset, expected_offset, atol=1e-9)
        np.testing.assert_allclose(
            controller._latest_wrench_context["applied_offset"],
            expected_offset,
            atol=1e-9,
        )
        self.assertTrue(controller._latest_wrench_context["admittance_offset_antiwindup_applied"])
        self.assertEqual(
            controller._latest_wrench_context["admittance_offset_antiwindup_active_axes"],
            ["rz"],
        )
        self.assertEqual(
            controller._latest_wrench_context["admittance_offset_antiwindup_guarded_axes"],
            [],
        )
        self.assertEqual(controller._latest_wrench_context["applied_offset_source"], "final_command_pose")

    def test_pose_controller_does_not_sync_reference_tracking_lag_without_force(self) -> None:
        reference = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        requested_offset = [0.0] * 6
        lagged_pose_offset = [-0.01, 0.0, 0.0, 0.0, 0.0, -0.08]
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                joint_rate_limiter_enabled=False,
                max_command_joint_velocity_deg_s=1000.0,
            ),
            robot=None,
        )
        controller._ik_resolver = object()
        controller._ik_seed = [0.0] * 6
        controller._previous_command_joint = [0.0] * 6
        controller._previous_command_pose = list(reference)
        controller._current_step_dt = 0.1
        controller._admittance = PoseAdmittance(
            enabled_axes=[True] * 6,
            max_angular_offset_rad=1.0,
            offset=list(requested_offset),
        )
        controller._latest_wrench_context = {
            "offset": list(requested_offset),
            "active_wrench": [0.0] * 6,
        }

        def lagged_solver(**kwargs):
            return (
                apply_pose_offset(kwargs["reference_pose"], lagged_pose_offset),
                [0.0] * 6,
                [0.0] * 6,
                1.0,
                list(kwargs["offset"]),
                0,
            )

        controller._command_from_pose_with_resolver = lagged_solver  # type: ignore[method-assign]

        controller._command_from_reference(reference, requested_offset)

        np.testing.assert_allclose(controller._admittance.offset, requested_offset, atol=1e-9)
        np.testing.assert_allclose(controller._latest_wrench_context["applied_offset"], requested_offset, atol=1e-9)
        self.assertFalse(controller._latest_wrench_context["admittance_offset_antiwindup_applied"])
        self.assertEqual(
            controller._latest_wrench_context["admittance_offset_antiwindup_skip_reason"],
            "active_wrench_zero",
        )

    def test_pose_controller_does_not_sync_admittance_while_returning_to_neutral(self) -> None:
        reference = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        requested_offset = [0.04, 0.0, 0.0, 0.0, 0.0, 0.2]
        lagged_pose_offset = [0.01, 0.0, 0.0, 0.0, 0.0, 0.04]
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                joint_rate_limiter_enabled=False,
                max_command_joint_velocity_deg_s=1000.0,
            ),
            robot=None,
        )
        controller._ik_resolver = object()
        controller._ik_seed = [0.0] * 6
        controller._previous_command_joint = [0.0] * 6
        controller._previous_command_pose = list(reference)
        controller._current_step_dt = 0.1
        controller._admittance = PoseAdmittance(
            enabled_axes=[True] * 6,
            max_angular_offset_rad=1.0,
            offset=list(requested_offset),
        )
        controller._latest_wrench_context = {
            "offset": list(requested_offset),
            "active_wrench": [0.0] * 6,
            "linear_return_velocity_limit_applied": True,
        }

        def lagged_solver(**kwargs):
            return (
                apply_pose_offset(kwargs["reference_pose"], lagged_pose_offset),
                [0.0] * 6,
                [0.0] * 6,
                1.0,
                list(kwargs["offset"]),
                0,
            )

        controller._command_from_pose_with_resolver = lagged_solver  # type: ignore[method-assign]

        controller._command_from_reference(reference, requested_offset)

        np.testing.assert_allclose(controller._admittance.offset, requested_offset, atol=1e-9)
        np.testing.assert_allclose(controller._latest_wrench_context["applied_offset"], requested_offset, atol=1e-9)
        self.assertFalse(controller._latest_wrench_context["admittance_offset_antiwindup_applied"])
        self.assertEqual(
            controller._latest_wrench_context["admittance_offset_antiwindup_skip_reason"],
            "returning_to_neutral",
        )

    def test_pose_controller_guards_antiwindup_against_force_opposing_linear_realization(self) -> None:
        reference = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        requested_offset = [0.0, -0.01, 0.0, 0.0, 0.0, 0.0]
        realized_pose_offset = [0.0, -0.025, 0.0, 0.0, 0.0, 0.0]
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                joint_rate_limiter_enabled=False,
                max_command_joint_velocity_deg_s=1000.0,
            ),
            robot=None,
        )
        controller._admittance = PoseAdmittance(
            enabled_axes=[True] * 6,
            max_angular_offset_rad=1.0,
            offset=list(requested_offset),
        )
        controller._latest_wrench_context = {
            "active_wrench": [0.0, 12.0, 0.0, 0.0, 0.0, 0.0],
            "target_offset_after_offset_limit": [0.0, 0.048, 0.0, 0.0, 0.0, 0.0],
        }

        realized = controller._sync_admittance_to_realized_pose_offset(
            reference_pose=reference,
            command_pose=apply_pose_offset(reference, realized_pose_offset),
            requested_offset=requested_offset,
        )

        self.assertAlmostEqual(realized[1], 0.0)
        self.assertAlmostEqual(controller._admittance.offset[1], 0.0)
        self.assertTrue(controller._latest_wrench_context["admittance_offset_antiwindup_applied"])
        self.assertEqual(
            controller._latest_wrench_context["applied_offset_source"],
            "guarded_final_command_pose",
        )
        self.assertEqual(
            controller._latest_wrench_context["admittance_offset_antiwindup_guarded_axes"],
            ["y"],
        )
        self.assertAlmostEqual(
            controller._latest_wrench_context["admittance_offset_realized_before_guard"][1],
            -0.025,
        )

    def test_pose_controller_final_joint_transport_realization_syncs_admittance(self) -> None:
        reference = [0.0, 0.0, 0.35, 0.0, 0.0, 0.0]
        requested_offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        realized_pose_offset = [0.025, 0.0, 0.0, 0.0, 0.0, 0.12]
        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                tare_sec=0.0,
                canfd_follow=True,
                joint_rate_limiter_enabled=False,
                max_command_joint_velocity_deg_s=1000.0,
                max_linear_offset_m=0.08,
                max_angular_offset_rad=0.6,
            ),
            robot=None,
        )
        controller._admittance = PoseAdmittance(
            enabled_axes=[True] * 6,
            max_angular_offset_rad=1.0,
            offset=list(requested_offset),
        )
        controller._latest_wrench_context = {
            "active_wrench": [4.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            "target_offset_after_offset_limit": [0.02, 0.0, 0.0, 0.0, 0.0, 0.1],
            "applied_offset": list(requested_offset),
            "offset": list(requested_offset),
            "applied_offset_source": "direct_joint_hint",
        }

        realized = controller._sync_admittance_to_realized_pose_offset(
            reference_pose=reference,
            command_pose=apply_pose_offset(reference, realized_pose_offset),
            requested_offset=requested_offset,
            realization_source="final_joint_transport_fk",
        )

        np.testing.assert_allclose(realized, realized_pose_offset, atol=1e-9)
        np.testing.assert_allclose(controller._admittance.offset, realized_pose_offset, atol=1e-9)
        np.testing.assert_allclose(
            controller._latest_wrench_context["applied_offset"],
            realized_pose_offset,
            atol=1e-9,
        )
        self.assertEqual(controller._latest_wrench_context["applied_offset_source"], "final_command_pose")
        self.assertTrue(controller._latest_wrench_context["admittance_offset_antiwindup_applied"])
        self.assertTrue(controller._latest_wrench_context["admittance_offset_antiwindup_would_apply"])
        self.assertTrue(controller._latest_wrench_context["admittance_offset_state_update_enabled"])
        self.assertEqual(controller._latest_wrench_context["admittance_offset_antiwindup_skip_reason"], "")
        self.assertEqual(
            controller._latest_wrench_context["final_command_pose_realization_source"],
            "final_joint_transport_fk",
        )

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

    def test_deployment_full_ik_worker_runs_at_15hz_without_sdk_calls(self) -> None:
        class RecordingFullResolver:
            def __init__(self) -> None:
                self.call_times: list[float] = []

            def resolve(self, request):
                self.call_times.append(time.perf_counter())
                return IKResolveResult(
                    status=IKResolveStatus.PASS,
                    qpos=np.zeros(6),
                    requested_target_pose=np.asarray(request.target_pose, dtype=float),
                    effective_target_pose=np.asarray(request.target_pose, dtype=float),
                    tolerance_fraction=1.0,
                    candidate_source="pinocchio",
                    message="ok",
                )

        controller = RM65PoseImpedanceController(
            arm="right",
            config=RM65ImpedanceConfig(
                tool_name="DMTAC_L",
                action_hz=15.0,
                deployment_dual_rate_ik_enabled=True,
                deployment_full_ik_hz=15.0,
                deployment_full_ik_deadline_ms=60.0,
                deployment_local_ik_deadline_ms=6.0,
                ik_resolver_candidate_provider_names=["rm65_pinocchio"],
            ),
            robot=None,
        )
        resolver = RecordingFullResolver()
        controller._deployment_full_ik_resolver = resolver
        controller._deployment_local_ik_resolver = object()  # required only to start the worker
        request = IKResolveRequest(
            target_pose=np.eye(4),
            reference_qpos=np.zeros(6),
            reference_pose=np.eye(4),
            metadata={"provider_name_allowlist": ["rm65_pinocchio"]},
        )
        with controller._deployment_dual_rate_lock:
            controller._deployment_full_ik_stats["enabled"] = True
            controller._deployment_full_ik_snapshot = {
                "request": request,
                "generation": 0,
                "sequence": 1,
            }

        controller._start_deployment_dual_rate_ik()
        time.sleep(0.16)
        controller._stop_deployment_dual_rate_ik()

        self.assertGreaterEqual(len(resolver.call_times), 2)
        self.assertEqual(controller._deployment_full_ik_stats["deadline_miss_count"], 0)
        self.assertGreaterEqual(controller._deployment_full_ik_stats["completed_count"], 2)
        self.assertIsNotNone(controller._deployment_full_ik_result)
        spacing = resolver.call_times[1] - resolver.call_times[0]
        self.assertGreater(spacing, 0.04)
        self.assertLess(spacing, 0.10)


if __name__ == "__main__":
    unittest.main()
