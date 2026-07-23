from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
from utils.hand_eye.camera import RealSenseCamera, RealSenseConfig, _set_sensor_option


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


class HandEyeRealSenseSettingsTest(unittest.TestCase):
    def test_startup_no_frames_resets_only_selected_serial_and_retries(self) -> None:
        selected = _FakeHardwareDevice("LEFT123")
        other = _FakeHardwareDevice("RIGHT456")
        fake_rs = _FakeRealSenseStartup(
            devices=[other, selected],
            warmup_errors_by_pipeline=[
                [
                    RuntimeError("Frame didn't arrive within 1000"),
                    RuntimeError("Frame didn't arrive within 1000"),
                ],
                [],
            ],
        )
        config = RealSenseConfig(
            serial="LEFT123",
            resolution=(640, 480),
            fps=15,
            enable_depth=False,
            startup_warmup_frames=2,
            startup_hardware_reset_on_failure=True,
            startup_hardware_reset_retry_attempts=1,
            startup_hardware_reset_sleep_sec=0.0,
        )

        with patch("utils.hand_eye.camera._require_pyrealsense2", return_value=fake_rs):
            camera = RealSenseCamera(config).start()

        self.assertEqual(selected.hardware_reset_calls, 1)
        self.assertEqual(other.hardware_reset_calls, 0)
        self.assertEqual(len(fake_rs.pipelines), 2)
        self.assertEqual(fake_rs.pipelines[0].stop_calls, 1)
        self.assertEqual(fake_rs.pipelines[1].wait_calls, [1000, 1000])
        self.assertEqual(camera.settings_report["startup.hardware_reset_recovery_count"], 1)
        camera.stop()

    def test_fixed_realsense_options_restore_previous_values_on_stop(self) -> None:
        sensor = _FakeSensor(
            {
                "enable_auto_exposure": 1.0,
                "exposure": 33000.0,
            }
        )
        snapshot = []
        report = {}

        _set_sensor_option(
            _FakeRS,
            sensor,
            "enable_auto_exposure",
            False,
            report,
            prefix="color",
            snapshot=snapshot,
        )
        _set_sensor_option(
            _FakeRS,
            sensor,
            "exposure",
            8000.0,
            report,
            prefix="color",
            snapshot=snapshot,
        )

        self.assertEqual(report["color.enable_auto_exposure.previous"], 1.0)
        self.assertEqual(report["color.exposure.previous"], 33000.0)
        self.assertEqual(sensor.values["enable_auto_exposure"], 0.0)
        self.assertEqual(sensor.values["exposure"], 8000.0)

        camera = object.__new__(RealSenseCamera)
        camera.config = RealSenseConfig(restore_settings_on_stop=True)
        pipeline = _FakePipeline()
        camera._pipeline = pipeline
        camera._align = object()
        camera._profile = object()
        camera._settings_snapshot = snapshot
        camera.settings_report = {}

        camera.stop()

        self.assertTrue(camera._pipeline is None)
        self.assertTrue(pipeline.stopped)
        self.assertTrue(camera.settings_report["restore.enabled"])
        self.assertEqual(camera.settings_report["restore.count"], 2)
        self.assertEqual(sensor.values["exposure"], 33000.0)
        self.assertEqual(sensor.values["enable_auto_exposure"], 1.0)
        self.assertEqual(
            sensor.set_history,
            [
                ("enable_auto_exposure", 0.0),
                ("exposure", 8000.0),
                ("exposure", 33000.0),
                ("enable_auto_exposure", 1.0),
            ],
        )

    def test_realsense_option_restore_can_be_disabled(self) -> None:
        sensor = _FakeSensor({"exposure": 33000.0})
        snapshot = []
        _set_sensor_option(
            _FakeRS,
            sensor,
            "exposure",
            8000.0,
            {},
            prefix="color",
            snapshot=snapshot,
        )
        camera = object.__new__(RealSenseCamera)
        camera.config = RealSenseConfig(restore_settings_on_stop=False)
        camera._pipeline = _FakePipeline()
        camera._align = object()
        camera._profile = object()
        camera._settings_snapshot = snapshot
        camera.settings_report = {}

        camera.stop()

        self.assertEqual(sensor.values["exposure"], 8000.0)
        self.assertEqual(camera.settings_report["restore.enabled"], False)


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


class _FakeRS:
    class option:
        enable_auto_exposure = "enable_auto_exposure"
        exposure = "exposure"


class _FakeSensor:
    def __init__(self, values: dict[str, float]):
        self.values = dict(values)
        self.set_history: list[tuple[str, float]] = []

    def supports(self, option: str) -> bool:
        return option in self.values

    def get_option(self, option: str) -> float:
        return self.values[option]

    def set_option(self, option: str, value: float) -> None:
        self.values[option] = float(value)
        self.set_history.append((option, float(value)))


class _FakePipeline:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _FakeHardwareDevice:
    def __init__(self, serial: str) -> None:
        self.serial = serial
        self.hardware_reset_calls = 0
        self.sensors = []

    def supports(self, field: object) -> bool:
        return field == "serial_number"

    def get_info(self, field: object) -> str:
        if field != "serial_number":
            raise KeyError(field)
        return self.serial

    def hardware_reset(self) -> None:
        self.hardware_reset_calls += 1


class _FakeStartupProfile:
    def __init__(self, device: _FakeHardwareDevice) -> None:
        self._device = device

    def get_device(self) -> _FakeHardwareDevice:
        return self._device


class _FakeStartupConfig:
    def __init__(self) -> None:
        self.serial: str | None = None

    def enable_device(self, serial: str) -> None:
        self.serial = serial

    def enable_stream(self, *args: object) -> None:
        return None


class _FakeStartupPipeline:
    def __init__(
        self,
        owner: "_FakeRealSenseStartup",
        warmup_errors: list[RuntimeError],
    ) -> None:
        self.owner = owner
        self.warmup_errors = list(warmup_errors)
        self.stop_calls = 0
        self.wait_calls: list[int] = []

    def start(self, config: _FakeStartupConfig) -> _FakeStartupProfile:
        serial = config.serial
        device = next(device for device in self.owner.devices if device.serial == serial)
        return _FakeStartupProfile(device)

    def wait_for_frames(self, timeout_ms: int) -> object:
        self.wait_calls.append(timeout_ms)
        if self.warmup_errors:
            raise self.warmup_errors.pop(0)
        return object()

    def stop(self) -> None:
        self.stop_calls += 1


class _FakeRealSenseStartup:
    class stream:
        color = "color"
        depth = "depth"

    class format:
        bgr8 = "bgr8"
        yuyv = "yuyv"
        z16 = "z16"

    class camera_info:
        serial_number = "serial_number"

    def __init__(
        self,
        *,
        devices: list[_FakeHardwareDevice],
        warmup_errors_by_pipeline: list[list[RuntimeError]],
    ) -> None:
        self.devices = list(devices)
        self.warmup_errors_by_pipeline = list(warmup_errors_by_pipeline)
        self.pipelines: list[_FakeStartupPipeline] = []

    def pipeline(self) -> _FakeStartupPipeline:
        errors = self.warmup_errors_by_pipeline[len(self.pipelines)]
        pipeline = _FakeStartupPipeline(self, errors)
        self.pipelines.append(pipeline)
        return pipeline

    def config(self) -> _FakeStartupConfig:
        return _FakeStartupConfig()

    def context(self):
        return type("Context", (), {"devices": self.devices})()


if __name__ == "__main__":
    unittest.main()
