from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.hand_eye.calibration import (
    CalibrationResult,
    CalibrationSample,
    evaluate_eye_in_hand,
    load_result,
    sample_plan_summary,
    save_result,
    solve_eye_in_hand,
)
from utils.hand_eye.transforms import (
    compose_transforms,
    invert_transform,
    make_transform,
    rotation_matrix_from_euler_xyz,
    transform_distance,
)


class HandEyeEyeInHandTest(unittest.TestCase):
    def test_synthetic_eye_in_hand_solution_recovers_camera_to_gripper(self) -> None:
        try:
            import cv2  # noqa: F401
        except Exception:
            self.skipTest("OpenCV is not available")

        T_camera_to_gripper = make_transform(
            rotation_matrix_from_euler_xyz(0.16, -0.22, 0.31),
            [0.045, -0.028, 0.082],
        )
        T_target_to_base = make_transform(
            rotation_matrix_from_euler_xyz(-0.08, 0.13, -0.18),
            [0.54, -0.11, 0.36],
        )
        samples = []
        for index in range(18):
            angle = float(index) * 0.37
            T_gripper_to_base = make_transform(
                rotation_matrix_from_euler_xyz(
                    0.25 * np.sin(angle),
                    -0.20 * np.cos(angle * 0.7),
                    -0.55 + 0.08 * index,
                ),
                [
                    0.30 + 0.045 * np.cos(angle),
                    -0.10 + 0.060 * np.sin(angle * 0.9),
                    0.42 + 0.030 * np.sin(angle * 1.3),
                ],
            )
            T_target_to_camera = compose_transforms(
                invert_transform(T_camera_to_gripper),
                invert_transform(T_gripper_to_base),
                T_target_to_base,
            )
            samples.append(
                CalibrationSample(
                    T_gripper_to_base=T_gripper_to_base,
                    T_target_to_camera=T_target_to_camera,
                    metadata={"sample_index": index + 1},
                )
            )

        result = solve_eye_in_hand(samples, method="AUTO")
        trans_m, rot_deg = transform_distance(result.T_camera_to_gripper, T_camera_to_gripper)
        self.assertLess(trans_m, 1e-6)
        self.assertLess(rot_deg, 1e-4)
        metrics = evaluate_eye_in_hand(samples, result.T_camera_to_gripper)
        self.assertLess(metrics["target_in_base_translation_rms_m"], 1e-6)
        self.assertLess(metrics["target_in_base_rotation_rms_deg"], 1e-4)

    def test_auto_reject_outliers_records_history(self) -> None:
        try:
            import cv2  # noqa: F401
        except Exception:
            self.skipTest("OpenCV is not available")

        samples = _synthetic_samples(25)
        result = solve_eye_in_hand(
            samples,
            method="TSAI",
            refine=False,
            auto_reject_outliers=True,
            outlier_reject_rounds=1,
            outlier_reject_fraction=0.10,
            outlier_min_samples=20,
        )
        self.assertEqual(len(result.samples), 22)
        self.assertIn("outlier_rejection", result.metrics)
        self.assertEqual(result.metrics["outlier_rejection"][0]["input_samples"], 25)
        self.assertEqual(result.metrics["outlier_rejection"][0]["kept_samples"], 22)

    def test_sample_plan_summary_counts_distance_grid_and_wrist_span(self) -> None:
        samples = _synthetic_samples(3)
        samples[0].metadata.update({"target_distance_m": 0.35, "target_center_norm": [0.1, 0.1]})
        samples[1].metadata.update({"target_distance_m": 0.55, "target_center_norm": [0.5, 0.5]})
        samples[2].metadata.update({"target_distance_m": 0.90, "target_center_norm": [0.9, 0.9]})
        summary = sample_plan_summary(samples)
        self.assertEqual(summary["distance_bin_counts"], {"near": 1, "middle": 1, "far": 1})
        self.assertEqual(summary["image_grid_counts"]["top_left"], 1)
        self.assertEqual(summary["image_grid_counts"]["center"], 1)
        self.assertEqual(summary["image_grid_counts"]["bottom_right"], 1)

    def test_result_json_npz_and_npy_load(self) -> None:
        result = CalibrationResult(
            T_camera_to_gripper=make_transform(
                rotation_matrix_from_euler_xyz(0.1, 0.2, -0.1),
                [0.03, 0.04, 0.05],
            ),
            T_target_to_base=make_transform(np.eye(3), [0.5, 0.0, 0.2]),
            method="TEST",
            metrics={"sample_count": 0},
            samples=[],
            created_at_utc="2026-07-03T00:00:00Z",
            metadata={"camera_serial": "serial"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_result(result, Path(tmpdir), name="unit", save_legacy_npy=True)
            loaded_json = load_result(paths["json"])
            loaded_npz = load_result(paths["npz"])
            loaded_g2c = load_result(paths["legacy_npy"])
            loaded_c2g = load_result(paths["camera_to_gripper_npy"])
            for loaded in (loaded_json, loaded_npz, loaded_g2c, loaded_c2g):
                np.testing.assert_allclose(
                    loaded.T_camera_to_gripper,
                    result.T_camera_to_gripper,
                    atol=1e-9,
                )
                np.testing.assert_allclose(
                    loaded.T_gripper_to_camera,
                    result.T_gripper_to_camera,
                    atol=1e-9,
                )


def _synthetic_samples(count: int) -> list[CalibrationSample]:
    T_camera_to_gripper = make_transform(
        rotation_matrix_from_euler_xyz(0.16, -0.22, 0.31),
        [0.045, -0.028, 0.082],
    )
    T_target_to_base = make_transform(
        rotation_matrix_from_euler_xyz(-0.08, 0.13, -0.18),
        [0.54, -0.11, 0.36],
    )
    samples = []
    for index in range(count):
        angle = float(index) * 0.37
        pose = [
            0.30 + 0.045 * np.cos(angle),
            -0.10 + 0.060 * np.sin(angle * 0.9),
            0.42 + 0.030 * np.sin(angle * 1.3),
            0.25 * np.sin(angle),
            -0.20 * np.cos(angle * 0.7),
            -0.55 + 0.08 * index,
        ]
        T_gripper_to_base = make_transform(
            rotation_matrix_from_euler_xyz(*pose[3:6]),
            pose[:3],
        )
        T_target_to_camera = compose_transforms(
            invert_transform(T_camera_to_gripper),
            invert_transform(T_gripper_to_base),
            T_target_to_base,
        )
        samples.append(
            CalibrationSample(
                T_gripper_to_base=T_gripper_to_base,
                T_target_to_camera=T_target_to_camera,
                robot_pose_xyz_rpy=pose,
                metadata={"sample_index": index + 1},
            )
        )
    return samples


if __name__ == "__main__":
    unittest.main()
