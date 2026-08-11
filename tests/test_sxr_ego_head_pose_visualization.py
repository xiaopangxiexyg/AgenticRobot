from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import pytest

from utils.portable_vis.sxr_ego_head_pose.visualize_head_extrinsics import (
    CAMERA_OPTICAL_TO_CALIBRATION_ROTATION,
    build_parser,
    decode_head_pose_message,
    display_transforms,
    load_camera_extrinsics,
)


def test_decode_head_pose_supports_explicit_quaternion_order() -> None:
    payload = struct.pack("<q7f", 123, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9)
    xyzw = decode_head_pose_message(payload, quaternion_order="xyzw")
    wxyz = decode_head_pose_message(payload, quaternion_order="wxyz")

    assert xyzw.timestamp_ns == 123
    np.testing.assert_allclose(xyzw.position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        wxyz.quaternion,
        np.array([0.2, 0.3, 0.9, 0.1]) / np.linalg.norm([0.2, 0.3, 0.9, 0.1]),
    )


def test_first_frame_mode_recenters_head_pose() -> None:
    first = decode_head_pose_message(
        struct.pack("<q7f", 100, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0)
    )
    second = decode_head_pose_message(
        struct.pack("<q7f", 200, 4.1, 5.2, 6.3, 0.0, 0.0, 0.0, 1.0)
    )
    transforms = display_transforms(
        [first, second],
        pose_direction="world_from_imu",
        frame_mode="first",
    )

    np.testing.assert_allclose(transforms[0], np.eye(4), atol=1e-12)
    np.testing.assert_allclose(transforms[1][:3, 3], [0.1, 0.2, 0.3], atol=1e-6)


@pytest.mark.integration
def test_loads_all_selected_camera_extrinsics(tmp_path: Path) -> None:
    identity = np.eye(4).tolist()
    calibration = {
        "observation": {
            "images": {
                group: {
                    "extrinsics": {
                        "T_ic_cam0_to_imu0": identity,
                        "T_ic_cam1_to_imu0": identity,
                    }
                }
                for group in ("rgb", "tracking", "ctrl")
            }
        }
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")

    cameras = load_camera_extrinsics(path, camera_groups=["rgb", "ctrl"])

    assert [camera.name for camera in cameras] == [
        "rgb/cam0",
        "rgb/cam1",
        "ctrl/cam0",
        "ctrl/cam1",
    ]
    np.testing.assert_allclose(
        cameras[0].T_optical_camera_to_imu[:3, :3],
        CAMERA_OPTICAL_TO_CALIBRATION_ROTATION,
    )


def test_cli_defaults_to_safe_offline_relative_replay() -> None:
    args = build_parser().parse_args(["episode", "--dry-run"])
    assert args.frame_mode == "first"
    assert args.quaternion_order == "xyzw"
    assert args.pose_direction == "world_from_imu"
    assert args.speed == 1.0
    assert args.dry_run is True
