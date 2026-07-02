from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from rm65_inspire_tac.ik_tuning import hard_branch_jump_exceeds
from teleop_runtime.backends.arms.rm65 import RM65ArmBackend, RM65ArmBackendConfig


def _yaw_pose(yaw_deg: float) -> np.ndarray:
    yaw = np.deg2rad(float(yaw_deg))
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return pose


class RM65IKTuningTest(unittest.TestCase):
    def test_hard_branch_jump_exceeds_for_j123_over_limit(self) -> None:
        exceeded, deltas = hard_branch_jump_exceeds(
            reference_q=np.zeros(6),
            candidate_q=np.deg2rad([91.0, 0.0, 0.0, 45.0, 0.0, 0.0]),
            joint_indices=(0, 1, 2),
            max_delta_deg=(90.0, 90.0, 90.0),
        )

        self.assertTrue(exceeded)
        self.assertEqual(deltas.tolist(), [91.0, 0.0, 0.0])

    def test_hard_branch_jump_ignores_non_root_wrist_large_jump(self) -> None:
        exceeded, deltas = hard_branch_jump_exceeds(
            reference_q=np.zeros(6),
            candidate_q=np.deg2rad([0.0, 0.0, 89.0, 120.0, 120.0, 120.0]),
            joint_indices=(0, 1, 2),
            max_delta_deg=(90.0, 90.0, 90.0),
        )

        self.assertFalse(exceeded)
        self.assertEqual(deltas.tolist(), [0.0, 0.0, 89.0])

    def test_rm65_jump_gate_holds_moderate_pose_large_joint_jump(self) -> None:
        backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side="right",
                connect_hardware=False,
                initial_qpos=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
        )
        backend._last_target_pose = np.eye(4, dtype=float)
        current_qpos = np.zeros(6, dtype=float)
        candidate_qpos = np.deg2rad([0.0, 0.0, 23.0, 0.0, 0.0, 0.0])

        result = backend._apply_jump_gate(
            solver=SimpleNamespace(last_branch_selection="unit_test"),
            target_pose=_yaw_pose(6.0),
            current_qpos=current_qpos,
            candidate_qpos=candidate_qpos,
        )

        np.testing.assert_allclose(result, current_qpos, atol=1e-12)
        metadata = backend._last_ik_metadata
        self.assertEqual(metadata["ik_jump_gate_status"], "hold_entered")
        self.assertEqual(metadata["ik_jump_gate_reason"], "moderate_pose_large_q_jump")


if __name__ == "__main__":
    unittest.main()
