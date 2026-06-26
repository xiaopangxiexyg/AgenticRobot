from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.handeye.calibration import CalibrationResult, CalibrationSample, load_result, save_result
from utils.handeye.transforms import (
    average_transforms,
    compose_transforms,
    invert_transform,
    make_transform,
    pose6_xyz_rpy_to_matrix,
    rotation_matrix_from_euler_xyz,
    transform_distance,
)


class HandeyeTransformTest(unittest.TestCase):
    def test_pose6_xyz_rpy_uses_meters_and_radians(self) -> None:
        pose = [0.1, -0.2, 0.3, 0.0, 0.0, np.pi / 2.0]
        matrix = pose6_xyz_rpy_to_matrix(pose)
        np.testing.assert_allclose(matrix[:3, 3], [0.1, -0.2, 0.3])
        np.testing.assert_allclose(matrix[:3, :3] @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-8)

    def test_inverse_and_compose_round_trip(self) -> None:
        transform = make_transform(
            rotation_matrix_from_euler_xyz(0.2, -0.1, 0.3),
            [0.4, 0.5, -0.2],
        )
        identity = compose_transforms(transform, invert_transform(transform))
        np.testing.assert_allclose(identity, np.eye(4), atol=1e-8)

    def test_average_transforms_keeps_center_pose(self) -> None:
        first = make_transform(np.eye(3), [0.0, 0.0, 0.0])
        second = make_transform(np.eye(3), [0.2, 0.0, 0.0])
        averaged = average_transforms([first, second])
        np.testing.assert_allclose(averaged[:3, 3], [0.1, 0.0, 0.0], atol=1e-8)
        trans_m, rot_deg = transform_distance(averaged, make_transform(np.eye(3), [0.1, 0.0, 0.0]))
        self.assertLess(trans_m, 1e-9)
        self.assertLess(rot_deg, 1e-6)


class HandeyeSerializationTest(unittest.TestCase):
    def test_result_json_npz_and_legacy_npy_load(self) -> None:
        sample = CalibrationSample(
            T_gripper_to_base=make_transform(np.eye(3), [0.1, 0.0, 0.0]),
            T_target_to_camera=make_transform(np.eye(3), [0.0, 0.0, 0.5]),
        )
        result = CalibrationResult(
            T_camera_to_base=make_transform(np.eye(3), [0.2, 0.3, 0.4]),
            T_target_to_gripper=make_transform(np.eye(3), [0.0, 0.0, 0.1]),
            method="TEST",
            metrics={"sample_count": 1},
            samples=[sample],
            created_at_utc="2026-06-20T00:00:00Z",
            metadata={"camera_serial": "serial"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_result(result, Path(tmpdir), name="unit", save_legacy_npy=True)
            for path in paths.values():
                loaded = load_result(path)
                np.testing.assert_allclose(loaded.T_camera_to_base, result.T_camera_to_base)
            self.assertIn("json", paths)
            self.assertIn("npz", paths)
            self.assertIn("legacy_npy", paths)


if __name__ == "__main__":
    unittest.main()
