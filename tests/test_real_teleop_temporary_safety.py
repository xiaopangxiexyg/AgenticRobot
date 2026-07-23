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

    def test_evaluation_records_exact_pre_filter_anchor_without_exposing_mutable_state(self) -> None:
        limiter = TemporaryDualArmSafetyLimiter(
            TemporaryDualArmSafetyConfig(
                enabled=True,
                active_sides=("left", "right"),
                max_xyz_step_m=0.1,
                max_rot_step_deg=15.0,
                evidence_enabled=True,
            )
        )
        limiter.filter_command(_left_arm_command(0.0), None, 10.0)
        first = limiter.last_evaluation
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first.filter_sequence, 0)
        self.assertTrue(first.stages[0].initialized)
        self.assertIsNone(first.stages[0].previous_anchor_values)

        filtered = limiter.filter_command(_left_arm_command(0.3), None, 10.1)
        second = limiter.last_evaluation
        self.assertIsNotNone(second)
        assert second is not None
        stage = second.stages[0]
        self.assertEqual(second.filter_sequence, 1)
        self.assertEqual(stage.previous_anchor_shape, (4, 4))
        previous = np.asarray(stage.previous_anchor_values).reshape(stage.previous_anchor_shape)
        output = np.asarray(stage.output_values).reshape(stage.output_shape)
        np.testing.assert_allclose(previous[:3, 3], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(output[:3, 3], [0.1, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(filtered.commands["left_arm"].value, dtype=float),
            output,
            atol=1e-12,
        )

        # A later compatibility sync may advance the limiter, but cannot mutate
        # the immutable evidence captured for the prior invocation.
        limiter.sync_from_command(_left_arm_command(-0.2))
        previous_after_sync = np.asarray(stage.previous_anchor_values).reshape(
            stage.previous_anchor_shape
        )
        np.testing.assert_allclose(
            previous_after_sync[:3, 3], [0.0, 0.0, 0.0], atol=1e-12
        )

    def test_disabled_limiter_records_not_applied_evidence_without_changing_command(self) -> None:
        limiter = TemporaryDualArmSafetyLimiter(
            TemporaryDualArmSafetyConfig(enabled=False, evidence_enabled=True)
        )
        command = _left_arm_command(0.3)

        filtered = limiter.filter_command(command, None, 5.0)

        self.assertIs(filtered, command)
        evidence = limiter.last_evaluation
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertFalse(evidence.enabled)
        self.assertEqual(evidence.stages, ())

    def test_evidence_is_default_off_on_the_legacy_control_path(self) -> None:
        limiter = TemporaryDualArmSafetyLimiter(
            TemporaryDualArmSafetyConfig(
                enabled=True,
                active_sides=("left", "right"),
            )
        )

        limiter.filter_command(_left_arm_command(0.1), None, 5.0)

        self.assertIsNone(limiter.last_evaluation)


if __name__ == "__main__":
    unittest.main()
