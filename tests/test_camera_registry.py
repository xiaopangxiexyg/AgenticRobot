from __future__ import annotations

import math
import unittest
from pathlib import Path

from teleop_runtime.cameras.registry import (
    build_camera_registry_snapshot,
    camera_matrix_from_intrinsics,
    load_camera_registry,
    normalize_camera_registry,
    pinhole_fov_from_intrinsics,
)


class CameraRegistryTest(unittest.TestCase):
    def test_default_registry_keeps_extrinsics_by_frame_with_placements(self) -> None:
        registry = load_camera_registry(
            Path("teleop_runtime/configs/cameras/realsense_registry.json")
        )

        front_frames = registry["cameras"]["front"]["extrinsics"]["frames"]
        self.assertIn("head_camera_frame", front_frames)
        placements = front_frames["head_camera_frame"]["placements"]
        self.assertEqual(placements[0]["frame"], "head_camera_frame")
        self.assertIn("rm65_dual_arm_gripper_v1_head_camera.urdf", placements[0]["urdf"])

        left_frames = registry["cameras"]["left_wrist"]["extrinsics"]["frames"]
        self.assertIn("left_camera_rgb", left_frames)
        self.assertGreaterEqual(len(left_frames["left_camera_rgb"]["placements"]), 2)

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
