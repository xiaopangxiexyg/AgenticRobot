from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

import utils.calibration_common.camera as common_camera
import utils.calibration_common.calibration as common_calibration
import utils.calibration_common.capture as common_capture
import utils.calibration_common.interactive as common_interactive
import utils.calibration_common.robot as common_robot
import utils.calibration_common.transforms as common_transforms
import utils.hand_eye.camera as hand_camera
import utils.hand_eye.calibration as hand_calibration
import utils.hand_eye.capture as hand_capture
import utils.hand_eye.robot as hand_robot
import utils.hand_eye.run_calibrate_handeye as hand_runner
import utils.hand_eye.transforms as hand_transforms
import utils.head_eye.camera as head_camera
import utils.head_eye.calibration as head_calibration
import utils.head_eye.capture as head_capture
import utils.head_eye.robot as head_robot
import utils.head_eye.run_calibrate_handeye as head_runner
import utils.head_eye.transforms as head_transforms
from utils.hand_eye.run_calibrate_handeye import (
    build_parser as build_wrist_calibration_parser,
)
from utils.hand_eye.visualize_handeye import (
    build_parser as build_wrist_visualizer_parser,
)
from utils.head_eye.calibration import (
    CalibrationResult,
    CalibrationSample,
    load_result,
    load_samples,
    load_samples_with_metadata,
    save_result,
    save_samples,
)
from utils.head_eye.camera import (
    VideoStreamProfileInfo,
    _choose_compatible_video_profile_pair,
)
from utils.head_eye.dual_arm import build_parser as build_dual_arm_parser
from utils.head_eye.run_calibrate_handeye import (
    build_parser as build_calibration_parser,
)
from utils.head_eye.transforms import (
    average_transforms,
    compose_transforms,
    invert_transform,
    make_transform,
    pose6_xyz_rpy_to_matrix,
    rotation_matrix_from_euler_xyz,
    transform_distance,
)
from utils.head_eye.visualize_handeye import build_parser as build_visualizer_parser


class CalibrationCommonOwnershipTest(unittest.TestCase):
    def test_calibration_namespaces_share_data_and_stateless_helpers(self) -> None:
        payload = {
            "T_gripper_to_base": np.eye(4).tolist(),
            "T_target_to_camera": np.eye(4).tolist(),
            "robot_pose_xyz_rpy": [0, 1, 2, 3, 4, 5],
            "robot_joint_deg": [6, 7, 8, 9, 10, 11],
            "target_reprojection_rmse_px": 0.25,
            "target_corner_count": 12,
            "timestamp_utc": "2026-09-22T00:00:00Z",
            "metadata": {"source": "unit"},
        }
        for sample_type, module_name in (
            (hand_calibration.CalibrationSample, "utils.hand_eye.calibration"),
            (head_calibration.CalibrationSample, "utils.head_eye.calibration"),
        ):
            self.assertIs(sample_type.__mro__[1], common_calibration.CalibrationSample)
            self.assertEqual(sample_type.__module__, module_name)
            sample = sample_type.from_dict(payload)
            self.assertIs(type(sample), sample_type)
            self.assertEqual(sample.to_dict(), payload)

        for compatibility_name, shared_name in (
            ("_float_or_none", "float_or_none"),
            ("_jsonable", "jsonable"),
            ("_list_or_none", "list_or_none"),
            ("_max_or_zero", "max_or_zero"),
            ("_now_utc", "now_utc"),
            ("_pose_span", "pose_span"),
            ("_rms", "rms"),
            ("_timestamp_for_filename", "timestamp_for_filename"),
        ):
            shared = getattr(common_calibration, shared_name)
            self.assertIs(getattr(hand_calibration, compatibility_name), shared)
            self.assertIs(getattr(head_calibration, compatibility_name), shared)

    def test_interactive_namespaces_export_one_mechanics_implementation(self) -> None:
        self.assertIs(hand_runner.AutoRecordState, common_interactive.AutoRecordState)
        self.assertIs(head_runner.AutoRecordState, common_interactive.AutoRecordState)
        self.assertIs(hand_runner._draw_session_overlay, common_interactive.draw_session_overlay)
        self.assertIs(head_runner._draw_session_overlay, common_interactive.draw_session_overlay)

    def test_camera_namespaces_export_one_leaf_helper_implementation(self) -> None:
        names = (
            "_aruco_dictionary",
            "_detector_parameters",
            "_make_aruco_detector",
            "_put_status",
            "_requested_profile_text",
            "_require_aruco",
            "_require_cv2",
            "_require_pyrealsense2",
            "_safe_device_info",
        )
        shared_names = (
            "aruco_dictionary",
            "detector_parameters",
            "make_aruco_detector",
            "put_status",
            "requested_profile_text",
            "require_aruco",
            "require_cv2",
            "require_pyrealsense2",
            "safe_device_info",
        )
        for compatibility_name, shared_name in zip(names, shared_names, strict=True):
            shared = getattr(common_camera, shared_name)
            self.assertIs(getattr(hand_camera, compatibility_name), shared)
            self.assertIs(getattr(head_camera, compatibility_name), shared)

    def test_capture_namespaces_export_one_stateless_helper_implementation(self) -> None:
        for compatibility_name, shared_name in (
            ("_max_distance", "max_distance"),
            ("_mean_vector", "mean_vector"),
            ("_most_common", "most_common"),
            ("_now_utc", "now_utc"),
        ):
            shared = getattr(common_capture, shared_name)
            self.assertIs(getattr(hand_capture, compatibility_name), shared)
            self.assertIs(getattr(head_capture, compatibility_name), shared)

    def test_transform_namespaces_export_one_implementation(self) -> None:
        names = (
            "as_transform",
            "average_quaternion_xyzw",
            "average_transforms",
            "compose_transforms",
            "invert_transform",
            "make_transform",
            "matrix_to_quaternion_xyzw",
            "pose6_xyz_rpy_to_matrix",
            "project_to_rotation",
            "quaternion_xyzw_to_matrix",
            "rotation_angle_deg",
            "rotation_matrix_from_euler_xyz",
            "transform_distance",
            "transform_to_list",
        )
        for name in names:
            shared = getattr(common_transforms, name)
            self.assertIs(getattr(hand_transforms, name), shared)
            self.assertIs(getattr(head_transforms, name), shared)

    def test_robot_namespaces_keep_compatibility_types(self) -> None:
        self.assertIs(hand_robot.RM65PoseReader.__mro__[1], common_robot.RM65PoseReader)
        self.assertIs(head_robot.RM65PoseReader.__mro__[1], common_robot.RM65PoseReader)
        self.assertIs(hand_robot.RM65PoseReader.sample_type, hand_robot.RobotPoseSample)
        self.assertIs(head_robot.RM65PoseReader.sample_type, head_robot.RobotPoseSample)
        self.assertEqual(hand_robot.RobotPoseSample.__module__, "utils.hand_eye.robot")
        self.assertEqual(head_robot.RobotPoseSample.__module__, "utils.head_eye.robot")
        self.assertIs(hand_robot._finite_float_list, common_robot._finite_float_list)
        self.assertIs(head_robot._finite_float_list, common_robot._finite_float_list)

        values = {
            "T_gripper_to_base": np.eye(4),
            "pose_xyz_rpy": [0.0] * 6,
            "joint_deg": [0.0] * 6,
            "timestamp_sec": 1.0,
            "arm_err": 0,
            "sys_err": 0,
        }
        self.assertIsInstance(hand_robot.RobotPoseSample(**values), common_robot.RobotPoseSample)
        self.assertIsInstance(head_robot.RobotPoseSample(**values), common_robot.RobotPoseSample)

    def test_shared_reader_returns_each_namespace_sample_without_hardware(self) -> None:
        class FakeArm:
            def Get_Current_Arm_State(self, *, retry: int):
                self.retry = retry
                return 0, [1, 2, 3, 4, 5, 6], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0], 0, 0

        for reader_type, sample_type in (
            (hand_robot.RM65PoseReader, hand_robot.RobotPoseSample),
            (head_robot.RM65PoseReader, head_robot.RobotPoseSample),
        ):
            reader = object.__new__(reader_type)
            reader.retry = 3
            reader.current_tool = "offline-test-tool"
            reader._arm = FakeArm()

            sample = reader.read()

            self.assertIs(type(sample), sample_type)
            self.assertEqual(sample.tool_frame, "offline-test-tool")
            self.assertEqual(reader._arm.retry, 3)
            np.testing.assert_allclose(
                sample.T_gripper_to_base,
                common_transforms.pose6_xyz_rpy_to_matrix(sample.pose_xyz_rpy),
            )


class HeadEyePackageTest(unittest.TestCase):
    def test_cli_modules_import_from_canonical_package(self) -> None:
        for build_parser in (
            build_calibration_parser,
            build_dual_arm_parser,
            build_visualizer_parser,
            build_wrist_calibration_parser,
            build_wrist_visualizer_parser,
        ):
            self.assertIsNotNone(build_parser())

    def test_fixed_camera_profile_fallback_defaults_to_compatible_opt_in(self) -> None:
        args = build_calibration_parser().parse_args([])
        self.assertTrue(args.auto_select_supported_profile)
        self.assertFalse(args.show_profiles)

        profiles = [
            VideoStreamProfileInfo("color", "bgr8", 640, 480, 15),
            VideoStreamProfileInfo("depth", "z16", 640, 480, 15),
            VideoStreamProfileInfo("color", "bgr8", 1280, 720, 30),
            VideoStreamProfileInfo("depth", "z16", 1280, 720, 30),
        ]
        pair = _choose_compatible_video_profile_pair(
            profiles,
            requested_width=1280,
            requested_height=720,
            requested_fps=15,
        )
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertEqual((pair[0].width, pair[0].height, pair[0].fps), (1280, 720, 30))
        self.assertEqual(pair[0].describe(), "color 1280x720@30 bgr8")

    @pytest.mark.integration
    def test_module_and_direct_script_help_entrypoints(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        commands = [
            [sys.executable, "-m", "utils.head_eye", "--help"],
            [sys.executable, "-m", "utils.head_eye.dual_arm", "--help"],
            [sys.executable, "-m", "utils.head_eye.visualize_handeye", "--help"],
            [sys.executable, str(repo_root / "utils/head_eye/run_calibrate_handeye.py"), "--help"],
            [sys.executable, str(repo_root / "utils/head_eye/dual_arm.py"), "--help"],
            [sys.executable, str(repo_root / "utils/head_eye/visualize_handeye.py"), "--help"],
            [sys.executable, "-m", "utils.hand_eye", "--help"],
            [sys.executable, "-m", "utils.hand_eye.visualize_handeye", "--help"],
            [sys.executable, str(repo_root / "utils/hand_eye/run_calibrate_handeye.py"), "--help"],
            [sys.executable, str(repo_root / "utils/hand_eye/visualize_handeye.py"), "--help"],
        ]
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        for command in commands:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"command failed: {command!r}\n{completed.stdout}{completed.stderr}",
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
            metadata={
                "camera_serial": "serial",
                "realsense_settings_report": {
                    "fallback_used": True,
                    "fallback_color_profile": "color 640x480@15 bgr8",
                },
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_result(result, Path(tmpdir), name="unit", save_legacy_npy=True)
            for path in paths.values():
                loaded = load_result(path)
                np.testing.assert_allclose(loaded.T_camera_to_base, result.T_camera_to_base)
            self.assertTrue(load_result(paths["json"]).metadata["realsense_settings_report"]["fallback_used"])
            self.assertIn("json", paths)
            self.assertIn("npz", paths)
            self.assertIn("legacy_npy", paths)

            samples_path = save_samples(
                Path(tmpdir) / "unit_samples.json",
                [sample],
                metadata=result.metadata,
            )
            self.assertEqual(len(load_samples(samples_path)), 1)
            _, sample_metadata = load_samples_with_metadata(samples_path)
            self.assertEqual(sample_metadata["realsense_settings_report"], result.metadata["realsense_settings_report"])


if __name__ == "__main__":
    unittest.main()
