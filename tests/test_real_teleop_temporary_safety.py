from __future__ import annotations

import unittest

import numpy as np

from teleop_runtime.debug.run_vive_rm65_real_teleop import (
    TemporaryDualArmSafetyConfig,
    TemporaryDualArmSafetyLimiter,
)
from teleop_runtime.schemas import EffectorCommand, TeleopCommand


def _arm_pose(x: float) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[0, 3] = float(x)
    return pose


def _left_arm_command(x: float) -> TeleopCommand:
    return TeleopCommand(
        commands={
            "left_arm": EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="cartesian_pose",
                value=_arm_pose(x),
            )
        }
    )


def _left_arm_joint_command(qpos: np.ndarray) -> TeleopCommand:
    pose = _arm_pose(0.25)
    return TeleopCommand(
        commands={
            "left_arm": EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.asarray(qpos, dtype=float),
                metadata={
                    "target_pose_base": pose.copy(),
                    "target_pose_tcp_base": pose.copy(),
                    "solver_output_qpos": np.asarray(qpos, dtype=float).copy(),
                    "action_pose_source": "limited_solver_output_qpos",
                },
            )
        }
    )


class TemporaryDualArmSafetyLimiterTest(unittest.TestCase):
    def test_sync_from_recovery_command_prevents_stale_limited_step(self) -> None:
        limiter = TemporaryDualArmSafetyLimiter(
            TemporaryDualArmSafetyConfig(
                enabled=True,
                active_sides=("left", "right"),
                max_xyz_step_m=0.1,
                max_rot_step_deg=15.0,
            )
        )
        limiter.filter_command(_left_arm_command(0.3), None, 0.0)
        limiter.sync_from_command(_left_arm_command(0.0))

        filtered = limiter.filter_command(_left_arm_command(0.005), None, 0.1)
        filtered_pose = np.asarray(filtered.commands["left_arm"].value, dtype=float)

        np.testing.assert_allclose(filtered_pose[:3, 3], [0.005, 0.0, 0.0], atol=1e-12)
        summary = filtered.metadata["temporary_dual_arm_safety"]["effectors"]["left_arm"]
        self.assertFalse(summary["limited"])

    def test_limited_cartesian_command_invalidates_stale_solver_qpos_hint(self) -> None:
        limiter = TemporaryDualArmSafetyLimiter(
            TemporaryDualArmSafetyConfig(
                enabled=True,
                active_sides=("left", "right"),
                max_xyz_step_m=0.1,
                max_rot_step_deg=15.0,
            )
        )
        limiter.filter_command(_left_arm_command(0.0), None, 0.0)
        command = _left_arm_command(0.3)
        arm_command = command.commands["left_arm"]
        arm_command.metadata.update(
            {
                "target_pose_base": _arm_pose(0.3),
                "target_pose_tcp_base": _arm_pose(0.3),
                "solver_output_qpos": np.arange(6, dtype=float),
                "action_pose_source": "limited_solver_output_qpos",
            }
        )

        filtered = limiter.filter_command(command, None, 0.1)
        filtered_arm = filtered.commands["left_arm"]
        filtered_pose = np.asarray(filtered_arm.value, dtype=float)

        np.testing.assert_allclose(filtered_pose[:3, 3], [0.1, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(filtered_arm.metadata["target_pose_base"], dtype=float),
            filtered_pose,
            atol=1e-12,
        )
        self.assertNotIn("solver_output_qpos", filtered_arm.metadata)
        self.assertEqual(
            filtered_arm.metadata["action_pose_source"],
            "temporary_safety_limited_cartesian_pose",
        )

    def test_limited_joint_command_invalidates_stale_target_pose_until_backend_fk(self) -> None:
        limiter = TemporaryDualArmSafetyLimiter(
            TemporaryDualArmSafetyConfig(
                enabled=True,
                active_sides=("left", "right"),
                max_joint_step_deg=10.0,
            )
        )
        limiter.filter_command(_left_arm_joint_command(np.zeros(6)), None, 0.0)

        filtered = limiter.filter_command(
            _left_arm_joint_command(np.deg2rad(np.full(6, 30.0))),
            None,
            0.1,
        )
        filtered_arm = filtered.commands["left_arm"]
        expected = np.deg2rad(np.full(6, 10.0))

        np.testing.assert_allclose(filtered_arm.value, expected, atol=1e-12)
        np.testing.assert_allclose(filtered_arm.metadata["solver_output_qpos"], expected, atol=1e-12)
        self.assertNotIn("target_pose_base", filtered_arm.metadata)
        self.assertNotIn("target_pose_tcp_base", filtered_arm.metadata)
        self.assertTrue(filtered_arm.metadata["temporary_dual_arm_safety_action_pose_pending_fk"])
        self.assertEqual(filtered_arm.metadata["action_pose_source"], "limited_solver_output_qpos")


if __name__ == "__main__":
    unittest.main()
