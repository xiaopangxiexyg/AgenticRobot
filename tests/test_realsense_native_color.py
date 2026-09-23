from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from hardware.realsense.color import color_image_to_bgr
from teleop_runtime.sources.camera.realsense_rgb import RealSenseRGBCameraConfig
import utils.calibration_common.camera as common_camera
import utils.hand_eye.camera as hand_camera
import utils.head_eye.camera as head_camera
from utils.hand_eye.camera import RealSenseConfig, _color_image_to_bgr
from utils.head_eye.camera import RealSenseConfig as HeadEyeRealSenseConfig


class RealSenseNativeColorTest(unittest.TestCase):
    def test_calibration_profile_selection_preserves_resolution_and_format_priority(self) -> None:
        for profile_type, module_name in (
            (hand_camera.VideoStreamProfileInfo, "utils.hand_eye.camera"),
            (head_camera.VideoStreamProfileInfo, "utils.head_eye.camera"),
        ):
            self.assertIs(profile_type.__mro__[1], common_camera.VideoStreamProfileInfo)
            self.assertEqual(profile_type.__module__, module_name)

        dimensions = (
            (640, 480, 15, "yuyv"),
            (1280, 720, 30, "bgr8"),
            (1280, 720, 30, "yuyv"),
        )
        hand_profiles = [
            hand_camera.VideoStreamProfileInfo("color", fmt, width, height, fps)
            for width, height, fps, fmt in dimensions
        ]
        head_profiles = [
            head_camera.VideoStreamProfileInfo("color", fmt, width, height, fps)
            for width, height, fps, fmt in dimensions
        ]
        selected_head = head_camera._choose_video_profile(
            head_profiles,
            stream="color",
            preferred_formats=("yuyv", "bgr8"),
            requested_width=1280,
            requested_height=720,
            requested_fps=15,
        )

        rs = SimpleNamespace(
            stream=SimpleNamespace(color=object()),
            format=SimpleNamespace(yuyv=object(), bgr8=object()),
            camera_info=SimpleNamespace(serial_number=object()),
        )
        device = SimpleNamespace(supports=lambda _: True, get_info=lambda _: "serial")
        with patch.object(hand_camera, "available_video_profiles", return_value=hand_profiles):
            selected_hand = hand_camera._select_video_profile(
                rs,
                device,
                stream=rs.stream.color,
                preferred_formats=(rs.format.yuyv, rs.format.bgr8),
                requested_width=1280,
                requested_height=720,
                requested_fps=15,
            )

        for selected in (selected_hand, selected_head):
            self.assertIsNotNone(selected)
            assert selected is not None
            self.assertEqual(selected.describe(), "color 1280x720@30 yuyv")

    def test_config_defaults_to_native_yuyv(self) -> None:
        self.assertEqual(RealSenseConfig().color_format, "yuyv")
        self.assertEqual(RealSenseConfig(color_format="BGR8").color_format, "bgr8")
        self.assertEqual(HeadEyeRealSenseConfig().color_format, "yuyv")
        self.assertEqual(RealSenseRGBCameraConfig(serial="camera").color_format, "yuyv")

    def test_config_rejects_unknown_color_format(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported RealSense color_format"):
            RealSenseConfig(color_format="mjpeg")

    @pytest.mark.optional
    def test_yuyv_payload_is_converted_to_bgr(self) -> None:
        try:
            import cv2  # type: ignore
        except ImportError:
            self.skipTest("OpenCV is not available")

        payload = np.array(
            [
                [[82, 90], [82, 240]],
                [[145, 54], [145, 34]],
            ],
            dtype=np.uint8,
        )
        converted = _color_image_to_bgr(payload, "yuyv")
        shared_converted = color_image_to_bgr(payload, "yuyv")

        self.assertEqual(converted.shape, (2, 2, 3))
        self.assertEqual(converted.dtype, np.uint8)
        np.testing.assert_array_equal(converted, cv2.cvtColor(payload, cv2.COLOR_YUV2BGR_YUY2))
        np.testing.assert_array_equal(converted, shared_converted)

    @pytest.mark.optional
    def test_packed_uint16_yuyv_payload_is_converted_to_bgr(self) -> None:
        try:
            import cv2  # type: ignore
        except ImportError:
            self.skipTest("OpenCV is not available")

        payload_u8 = np.array(
            [
                [[82, 90], [82, 240]],
                [[145, 54], [145, 34]],
            ],
            dtype=np.uint8,
        )
        payload_u16 = payload_u8.view(np.uint16).reshape(2, 2)

        converted = _color_image_to_bgr(payload_u16, "yuyv")

        np.testing.assert_array_equal(converted, cv2.cvtColor(payload_u8, cv2.COLOR_YUV2BGR_YUY2))

    def test_bgr_payload_remains_bgr_and_is_copied(self) -> None:
        payload = np.zeros((2, 3, 3), dtype=np.uint8)
        payload[0, 0] = [1, 2, 3]

        converted = _color_image_to_bgr(payload, "bgr8")
        payload[0, 0] = [9, 9, 9]

        np.testing.assert_array_equal(converted[0, 0], [1, 2, 3])
        self.assertTrue(converted.flags.c_contiguous)


if __name__ == "__main__":
    unittest.main()
