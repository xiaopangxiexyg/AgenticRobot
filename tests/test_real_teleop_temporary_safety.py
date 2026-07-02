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


if __name__ == "__main__":
    unittest.main()
