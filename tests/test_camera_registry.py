from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from teleop_runtime.configs.camera_paths import (
    DEFAULT_REALSENSE_CAMERA_REGISTRY_PATH,
    DEFAULT_REALSENSE_DEVICE_CONFIG_PATH,
    DEFAULT_REALSENSE_FRONT_DEVICE_CONFIG_PATH,
)
from sim.mujoco.config import default_camera_registry_path
from teleop_runtime.cameras.registry import (
    build_camera_registry_snapshot,
    camera_matrix_from_intrinsics,
    load_camera_registry,
    normalize_camera_registry,
    pinhole_fov_from_intrinsics,
)
from utils.dataset_postprocess.wrist_cam_adjust.geometry import resolve_repo_relative_path


class CameraRegistryTest(unittest.TestCase):
    def test_default_registry_keeps_extrinsics_by_frame_with_placements(self) -> None:
        registry = load_camera_registry(DEFAULT_REALSENSE_CAMERA_REGISTRY_PATH)

        front_frames = registry["cameras"]["front"]["extrinsics"]["frames"]
        self.assertIn("head_camera_frame", front_frames)
        placements = front_frames["head_camera_frame"]["placements"]
        expected_dual_arm_urdf = (
            "assets/rm65_inspire_gripper_v3/"
            "rm65_dual_arm_wrist_cameras_v3_head_camera_4b2_cad_with_legacy_housing.urdf"
        )
        legacy_dual_arm_urdf = (
            "assets/rm65_inspire_gripper_v3/"
            "rm65_dual_arm_wrist_cameras_v3_head_camera.urdf"
        )
        self.assertEqual(placements[0]["frame"], "head_camera_frame")
        self.assertEqual(
            {placement["urdf"] for placement in placements},
            {legacy_dual_arm_urdf, expected_dual_arm_urdf},
        )

        left_frames = registry["cameras"]["left_wrist"]["extrinsics"]["frames"]
        self.assertIn("left_camera_rgb", left_frames)
        left_wrist_placements = left_frames["left_camera_rgb"]["placements"]
        self.assertEqual(len(left_wrist_placements), 2)
        self.assertIn(
            expected_dual_arm_urdf,
            {placement["urdf"] for placement in left_wrist_placements},
        )

        right_frames = registry["cameras"]["right_wrist"]["extrinsics"]["frames"]
        right_wrist_placements = right_frames["right_camera_rgb"]["placements"]
        self.assertEqual(len(right_wrist_placements), 2)
        self.assertIn(
            expected_dual_arm_urdf,
            {placement["urdf"] for placement in right_wrist_placements},
        )

    def test_canonical_camera_configs_exist_without_root_cameras_package(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.assertTrue(DEFAULT_REALSENSE_DEVICE_CONFIG_PATH.exists())
        self.assertTrue(DEFAULT_REALSENSE_FRONT_DEVICE_CONFIG_PATH.exists())
        self.assertTrue(DEFAULT_REALSENSE_CAMERA_REGISTRY_PATH.exists())
        self.assertEqual(
            DEFAULT_REALSENSE_DEVICE_CONFIG_PATH.resolve(),
            (project_root / "teleop_runtime/configs/devices/realsense.json").resolve(),
        )
        self.assertEqual(
            DEFAULT_REALSENSE_FRONT_DEVICE_CONFIG_PATH.resolve(),
            (
                project_root
                / "teleop_runtime/configs/devices/realsense_front.json"
            ).resolve(),
        )
        self.assertEqual(
            DEFAULT_REALSENSE_CAMERA_REGISTRY_PATH.resolve(),
            (
                project_root
                / "robot_motion/configs/rm65_camera_registry.json"
            ).resolve(),
        )
        self.assertFalse((project_root / "cameras").exists())

    def test_real2sim_default_camera_registry_uses_shared_robot_motion_config(self) -> None:
        self.assertEqual(default_camera_registry_path(), DEFAULT_REALSENSE_CAMERA_REGISTRY_PATH)

    def test_historical_registry_path_alias_resolves_to_shared_config(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            resolve_repo_relative_path(
                "cameras/configs/rm65/cameras/realsense_registry.json",
                repo_root=project_root,
            ),
            DEFAULT_REALSENSE_CAMERA_REGISTRY_PATH.resolve(),
        )
        self.assertEqual(
            resolve_repo_relative_path(
                "cameras/configs/rm65/devices/realsense.json",
                repo_root=project_root,
            ),
            DEFAULT_REALSENSE_DEVICE_CONFIG_PATH.resolve(),
        )
        self.assertEqual(
            resolve_repo_relative_path(
                "teleop_runtime/configs/cameras/realsense_registry.json",
                repo_root=project_root,
            ),
            DEFAULT_REALSENSE_CAMERA_REGISTRY_PATH.resolve(),
        )

    def test_default_realsense_profiles_are_15hz(self) -> None:
        devices = json.loads(DEFAULT_REALSENSE_DEVICE_CONFIG_PATH.read_text(encoding="utf-8"))
        registry = load_camera_registry(DEFAULT_REALSENSE_CAMERA_REGISTRY_PATH)

        for camera_key in ("front", "left_wrist", "right_wrist"):
            self.assertEqual(devices[camera_key]["fps"], 15)
            self.assertEqual(registry["cameras"][camera_key]["profile"]["fps"], 15)

    def test_runtime_intrinsics_overlay_registry_without_losing_urdf_placements(self) -> None:
        registry = normalize_camera_registry(
            {
                "cameras": {
                    "front": {
                        "serial": "123",
                        "profile": {"width": 640, "height": 480, "fps": 30},
                        "extrinsics": {
                            "frames": {
                                "head_camera_frame": {
                                    "placements": [
                                        {"urdf": "a.urdf", "frame": "head_camera_frame"},
                                        {"urdf": "b.urdf", "frame": "head_camera_frame"},
                                    ]
                                }
                            }
                        },
                    }
                }
            }
        )
        snapshot = build_camera_registry_snapshot(
            registry,
            device_info_by_key={
                "front": {
                    "camera_key": "front",
                    "serial_number": "123",
                    "firmware_version": "1.2.3",
                    "frame_count": 999,
                    "intrinsics": {
                        "width": 640,
                        "height": 480,
                        "fx": 610.0,
                        "fy": 611.0,
                        "ppx": 319.5,
                        "ppy": 239.5,
                        "coeffs": [0.1, 0.0, 0.0, 0.0, 0.0],
                    },
                }
            },
        )

        front = snapshot["cameras"]["front"]
        self.assertEqual(front["intrinsics"]["source"], "runtime_device_info")
        self.assertEqual(front["intrinsics"]["cx"], 319.5)
        self.assertEqual(front["intrinsics"]["cy"], 239.5)
        self.assertNotIn("frame_count", front["runtime_device"])
        self.assertEqual(
            len(front["extrinsics"]["frames"]["head_camera_frame"]["placements"]),
            2,
        )

    def test_legacy_single_urdf_extrinsics_normalize_to_placements(self) -> None:
        registry = normalize_camera_registry(
            {
                "cameras": {
                    "front": {
                        "serial": "123",
                        "extrinsics": {
                            "camera_frame": "head_camera_frame",
                            "urdf": "robot.urdf",
                        },
                    }
                }
            }
        )

        frame = registry["cameras"]["front"]["extrinsics"]["frames"]["head_camera_frame"]
        self.assertEqual(frame["placements"], [{"urdf": "robot.urdf", "frame": "head_camera_frame"}])

    def test_camera_matrix_and_fov_helpers(self) -> None:
        intrinsics = {
            "width": 640,
            "height": 480,
            "fx": 320.0,
            "fy": 240.0,
            "cx": 319.5,
            "cy": 239.5,
        }

        self.assertEqual(
            camera_matrix_from_intrinsics(intrinsics),
            [[320.0, 0.0, 319.5], [0.0, 240.0, 239.5], [0.0, 0.0, 1.0]],
        )
        fov = pinhole_fov_from_intrinsics(intrinsics)
        self.assertAlmostEqual(fov["fovx_rad"], math.pi / 2.0)
        self.assertAlmostEqual(fov["fovy_rad"], math.pi / 2.0)


if __name__ == "__main__":
    unittest.main()
