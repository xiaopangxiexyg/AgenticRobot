from __future__ import annotations

import math
import unittest

import numpy as np

from hardware.quest_dev.features.visualization import alignment
from hardware.quest_dev.features.visualization.transforms import (
    quaternion_xyzw_to_matrix,
    transform_points,
)


class AlignmentTests(unittest.TestCase):
    def test_recovers_known_rigid_transform(self) -> None:
        robot = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.1, 0.1, 0.15],
            ]
        )
        expected = np.eye(4)
        expected[:3, :3] = quaternion_xyzw_to_matrix(
            [
                0.0,
                0.0,
                math.sin(math.pi / 8),
                math.cos(math.pi / 8),
            ]
        )
        expected[:3, 3] = [0.5, -0.25, 0.8]
        quest = transform_points(expected, robot)

        result = alignment.fit_quest_from_robot(robot, quest)
        np.testing.assert_allclose(result.quest_from_robot, expected, atol=1e-9)
        self.assertLess(result.max_error_m, 1e-9)
        self.assertTrue(
            alignment.alignment_acceptable(
                result,
                max_rms_error_m=0.001,
                max_point_error_m=0.002,
            )
        )

    def test_collinear_points_are_rejected(self) -> None:
        points = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        with self.assertRaisesRegex(ValueError, "collinear"):
            alignment.fit_quest_from_robot(points, points)


if __name__ == "__main__":
    unittest.main()
