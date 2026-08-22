from __future__ import annotations

import math
import unittest

import numpy as np

from hardware.quest_dev.features.visualization import transforms


class TransformTests(unittest.TestCase):
    def test_pose_round_trip(self) -> None:
        pose = {
            "position": [0.1, -0.2, 0.3],
            "orientation_quat_xyzw": [
                0.0,
                0.0,
                math.sin(math.pi / 4),
                math.cos(math.pi / 4),
            ],
        }
        matrix = transforms.pose_payload_to_matrix(pose)
        reconstructed = transforms.pose_payload_to_matrix(
            transforms.matrix_to_pose_payload(matrix)
        )
        np.testing.assert_allclose(reconstructed, matrix, atol=1e-8)

    def test_rigid_inverse(self) -> None:
        matrix = transforms.pose_payload_to_matrix(
            {
                "position": [1.0, 2.0, 3.0],
                "orientation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        )
        np.testing.assert_allclose(
            matrix @ transforms.rigid_inverse(matrix), np.eye(4), atol=1e-8
        )

    def test_column_major_translation_indices(self) -> None:
        matrix = np.eye(4)
        matrix[:3, 3] = [1.0, 2.0, 3.0]
        flat = transforms.matrix_to_column_major(matrix)
        self.assertEqual(flat[12:15], [1.0, 2.0, 3.0])

    def test_zero_quaternion_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-zero"):
            transforms.pose_payload_to_matrix(
                {
                    "position": [0, 0, 0],
                    "orientation_quat_xyzw": [0, 0, 0, 0],
                }
            )

    def test_half_turn_quaternions_round_trip(self) -> None:
        for quaternion in ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]):
            rotation = transforms.quaternion_xyzw_to_matrix(quaternion)
            reconstructed = transforms.quaternion_xyzw_to_matrix(
                transforms.matrix_to_quaternion_xyzw(rotation)
            )
            np.testing.assert_allclose(reconstructed, rotation, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
