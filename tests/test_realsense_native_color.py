from __future__ import annotations

import unittest
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from hardware.realsense.color import color_image_to_bgr
from teleop_runtime.sources.camera.realsense_rgb import RealSenseRGBCameraConfig
import utils.calibration_common.camera as common_camera
import utils.hand_eye.camera as hand_camera
import utils.head_eye.camera as head_camera
from utils.hand_eye.camera import RealSenseConfig, _color_image_to_bgr
from utils.head_eye.camera import RealSenseConfig as HeadEyeRealSenseConfig


@pytest.mark.parametrize("module", [head_camera, hand_camera])
def test_calibration_close_retains_pending_native_operation(module):
    release, entered = threading.Event(), threading.Event()
    calls = []
    camera = object.__new__(module.RealSenseCamera)
    camera._init_camera_lifecycle()
    camera.config = module.RealSenseConfig(pipeline_stop_timeout_sec=.02)
    camera.settings_report = {}
    camera._settings_snapshot = []
    camera._active_color_format = camera.config.color_format
    def stop():
        calls.append("stop")
        entered.set()
        release.wait(2)
    camera._pipeline = SimpleNamespace(stop=stop)
    try:
        with pytest.raises(RuntimeError, match="did not finish"):
            camera.stop()
        assert entered.is_set()
        with pytest.raises(RuntimeError, match="did not finish"):
            camera.stop()
        with pytest.raises(RuntimeError, match="cleanup is not complete"):
            camera.start()
        assert calls == ["stop"]
        assert camera._native_pipeline_stop.worker.is_alive()
    finally:
        release.set()
        assert camera._native_pipeline_stop.wait(1) is None
        camera.stop()
    assert calls == ["stop"]


@pytest.mark.parametrize("module", [head_camera, hand_camera])
def test_calibration_failed_start_cannot_fallback_or_reset_before_cleanup(module, monkeypatch):
    release = threading.Event()
    calls = []
    class Pipeline:
        def start(self, config):
            calls.append("start")
            raise RuntimeError("Couldn't resolve requests")
        def stop(self):
            calls.append("stop")
            release.wait(2)
    rs = SimpleNamespace(
        pipeline=lambda: Pipeline(), config=lambda: SimpleNamespace(
            enable_device=lambda *args: None, enable_stream=lambda *args: None,
        ), stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(yuyv="yuyv", z16="z16"),
    )
    monkeypatch.setattr(module, "_require_pyrealsense2", lambda: rs)
    kwargs = {"startup_hardware_reset_on_failure": True} if module is hand_camera else {}
    camera = module.RealSenseCamera(module.RealSenseConfig(serial="unit", pipeline_stop_timeout_sec=.02, **kwargs))
    monkeypatch.setattr(camera, "_fallback_stream_config", lambda _: calls.append("fallback"))
    if module is hand_camera:
        monkeypatch.setattr(camera, "_hardware_reset_device", lambda _: calls.append("reset"))
    try:
        with pytest.raises(RuntimeError, match="cleanup"):
            camera.start()
        assert calls == ["start", "stop"]
    finally:
        release.set()
        assert camera._native_pipeline_stop.wait(1) is None
        camera.stop()


def test_calibration_settings_restore_is_inside_the_close_budget(monkeypatch):
    release, entered = threading.Event(), threading.Event()
    camera = object.__new__(hand_camera.RealSenseCamera)
    camera._init_camera_lifecycle()
    camera.config = hand_camera.RealSenseConfig(pipeline_stop_timeout_sec=.02)
    camera.settings_report = {}
    calls = []
    camera._pipeline = SimpleNamespace(stop=lambda: calls.append("stop"))
    def restore():
        entered.set()
        release.wait(2)
        calls.append("restore")
    monkeypatch.setattr(camera, "_before_pipeline_stop", restore)
    try:
        with pytest.raises(RuntimeError, match="did not finish"):
            camera.stop()
        assert entered.is_set() and calls == []
        diagnostic = camera.settings_report["pipeline.stop"]
        assert diagnostic["phase"] == "before_stop"
        assert diagnostic["sdk_call_sec"] is None
        assert diagnostic["before_stop_sec"] > 0
    finally:
        release.set()
        assert camera._native_pipeline_stop.wait(1) is None
        camera.stop()
    assert calls == ["restore", "stop"]


@pytest.mark.parametrize("module", [head_camera, hand_camera])
def test_calibration_cancelled_lock_wait_never_opens_pipeline(module, monkeypatch):
    from hardware.realsense.lifecycle import SDK_LIFECYCLE_LOCK
    calls = []
    rs = SimpleNamespace(
        pipeline=lambda: calls.append("pipeline"), config=lambda: SimpleNamespace(
            enable_device=lambda *args: None, enable_stream=lambda *args: None,
        ), stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(yuyv="yuyv", z16="z16"),
    )
    monkeypatch.setattr(module, "_require_pyrealsense2", lambda: rs)
    monkeypatch.setattr(module, "CALIBRATION_CAMERA_STARTUP_TIMEOUT_SEC", .02)
    camera = module.RealSenseCamera(module.RealSenseConfig(serial="unit", auto_select_supported_profile=False))
    with SDK_LIFECYCLE_LOCK:
        with pytest.raises(TimeoutError):
            camera.__enter__()
        with pytest.raises(RuntimeError, match="startup worker is still active"):
            camera.stop()
        assert calls == []
    camera._startup_worker.join(1)
    assert not camera._startup_worker.is_alive()
    assert calls == []
    camera.stop()


@pytest.mark.parametrize("module", [head_camera, hand_camera])
@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_calibration_rejects_invalid_pipeline_stop_budget(module, timeout):
    with pytest.raises(ValueError, match="pipeline_stop_timeout_sec"):
        module.RealSenseConfig(pipeline_stop_timeout_sec=timeout)


def test_calibration_reset_cancelled_while_waiting_for_sdk_lock(monkeypatch):
    from hardware.realsense.lifecycle import SDK_LIFECYCLE_LOCK

    camera = object.__new__(hand_camera.RealSenseCamera)
    camera._init_camera_lifecycle()
    camera.config = hand_camera.RealSenseConfig(serial="unit", startup_hardware_reset_sleep_sec=0)
    camera._rs = object()
    waiting = threading.Event()
    calls, errors = [], []
    monkeypatch.setattr(camera, "_check_pending_stop", waiting.set)
    monkeypatch.setattr(hand_camera, "_device_from_serial", lambda *args: SimpleNamespace(
        hardware_reset=lambda: calls.append("reset"),
    ))
    monkeypatch.setattr(camera, "_wait_for_device_reconnect", lambda _: calls.append("reconnect"))

    def reset():
        try:
            camera._hardware_reset_device("unit")
        except RuntimeError as exc:
            errors.append(str(exc))

    with SDK_LIFECYCLE_LOCK:
        worker = threading.Thread(target=reset)
        worker.start()
        assert waiting.wait(1)
        camera._startup_timeout_event.set()
        assert calls == []
    worker.join(1)
    assert not worker.is_alive()
    assert calls == []
    assert len(errors) == 1 and "no reset was issued" in errors[0]


class RealSenseNativeColorTest(unittest.TestCase):
    def test_calibration_profile_enumeration_preserves_module_types_and_device_errors(self) -> None:
        color, depth, infrared = object(), object(), object()
        yuyv, z16 = object(), object()
        rs = SimpleNamespace(
            stream=SimpleNamespace(color=color, depth=depth, infrared=infrared),
            format=SimpleNamespace(yuyv=yuyv, z16=z16),
            camera_info=SimpleNamespace(serial_number=object()),
        )

        def profile(stream: object, fmt: object, width: int):
            video = SimpleNamespace(width=lambda: width, height=lambda: 480)
            return SimpleNamespace(
                as_video_stream_profile=lambda: video,
                stream_type=lambda: stream,
                format=lambda: fmt,
                fps=lambda: 15,
            )

        invalid = SimpleNamespace(as_video_stream_profile=lambda: None)
        sensor = SimpleNamespace(
            get_stream_profiles=lambda: [
                profile(depth, z16, 1280),
                invalid,
                profile(color, yuyv, 640),
                profile(infrared, z16, 320),
            ]
        )
        failing_sensor = SimpleNamespace(
            get_stream_profiles=Mock(side_effect=RuntimeError("profile query failed"))
        )
        device = SimpleNamespace(
            sensors=[failing_sensor, sensor],
            supports=lambda _: True,
            get_info=lambda _: "selected-serial",
        )
        rs.context = lambda: SimpleNamespace(devices=[device])

        for module in (hand_camera, head_camera):
            with self.subTest(module=module.__name__), patch.object(
                module, "_require_pyrealsense2", return_value=rs
            ):
                profiles = module.available_video_profiles("selected-serial")
                self.assertEqual(len(profiles), 3)
                self.assertTrue(all(type(item) is module.VideoStreamProfileInfo for item in profiles))
                self.assertEqual(
                    profiles,
                    sorted(
                        profiles,
                        key=lambda item: (
                            item.stream, item.width, item.height, item.fps, item.format_name
                        ),
                    ),
                )
                by_width = {item.width: item for item in profiles}
                self.assertEqual(by_width[640].describe(), "color 640x480@15 yuyv")
                self.assertEqual(by_width[1280].describe(), "depth 1280x480@15 z16")
                infrared_profile = by_width[320]
                if module is head_camera:
                    self.assertEqual(infrared_profile.stream, "infrared")
                else:
                    self.assertNotEqual(infrared_profile.stream, "infrared")
                with self.assertRaisesRegex(ValueError, "missing-serial.*selected-serial"):
                    module.available_video_profiles("missing-serial")

        rs.context = lambda: SimpleNamespace(devices=[])
        for module in (hand_camera, head_camera):
            with self.subTest(empty_module=module.__name__), patch.object(
                module, "_require_pyrealsense2", return_value=rs
            ):
                with self.assertRaisesRegex(RuntimeError, "no RealSense devices detected"):
                    module.available_video_profiles()

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
