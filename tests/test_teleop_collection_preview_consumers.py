from __future__ import annotations

import unittest

import numpy as np

from teleop_runtime.collection.camera_preview import CameraPreviewConfig, CameraPreviewer
from teleop_runtime.collection.preview_rate import PreviewRateLimiter
from teleop_runtime.collection.video_frame import VideoFrame
from teleop_runtime.tactile_visualizer import tactile_by_side_from_hand_states


class _FakeCv2:
    WINDOW_NORMAL = 16
    WINDOW_AUTOSIZE = 1
    WINDOW_FREERATIO = 256
    WND_PROP_ASPECT_RATIO = 2
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self):
        self.imshow_calls = []
        self.named_windows = []
        self.window_properties = []

    def namedWindow(self, name, flag):
        self.named_windows.append((name, flag))

    def setWindowProperty(self, name, prop, value):
        self.window_properties.append((name, prop, value))

    def imshow(self, name, image):
        self.imshow_calls.append((name, np.asarray(image).copy()))

    def waitKey(self, delay):
        return -1

    def putText(self, image, *args, **kwargs):
        return image


class _CountingCameraSource:
    def __init__(self):
        self.get_latest_frame_calls = 0

    def get_latest_frame(self):
        self.get_latest_frame_calls += 1
        return None

    def device_info(self):
        return {"frame_count": 1}


class _BufferedCameraSource(_CountingCameraSource):
    def __init__(self, frame):
        super().__init__()
        from teleop_runtime.collection.video_frame import NearestFrameBuffer

        self._buffer = NearestFrameBuffer()
        self._buffer.push(frame)

    def frame_buffer(self):
        return self._buffer


class TeleopCollectionPreviewConsumersTest(unittest.TestCase):
    def test_preview_rate_limiter_skips_until_interval_elapses(self) -> None:
        limiter = PreviewRateLimiter(10.0)

        self.assertTrue(limiter.should_update(1.00))
        self.assertFalse(limiter.should_update(1.05))
        self.assertTrue(limiter.should_update(1.10))
        limiter.reset()
        self.assertTrue(limiter.should_update(1.11))

    def test_preview_rate_limiter_rejects_non_positive_rate(self) -> None:
        with self.assertRaises(ValueError):
            PreviewRateLimiter(0.0)

    def test_camera_preview_frame_matches_do_not_read_camera_source(self) -> None:
        previewer = CameraPreviewer(CameraPreviewConfig(preflight_highgui=False))
        previewer._cv2 = _FakeCv2()
        source = _CountingCameraSource()
        frame = VideoFrame(
            image=np.ones((4, 5), dtype=np.uint8),
            source_timestamp=1.0,
        )

        keys = previewer.update_frame_matches(
            {"front": frame},
            camera_sources={"front": source},
        )

        self.assertEqual(keys, [])
        self.assertEqual(source.get_latest_frame_calls, 0)
        self.assertEqual(len(previewer._cv2.imshow_calls), 1)
        self.assertEqual(
            previewer._cv2.named_windows,
            [("realsense:front", _FakeCv2.WINDOW_NORMAL | _FakeCv2.WINDOW_FREERATIO)],
        )
        self.assertEqual(
            previewer._cv2.window_properties,
            [
                (
                    "realsense:front",
                    _FakeCv2.WND_PROP_ASPECT_RATIO,
                    _FakeCv2.WINDOW_FREERATIO,
                )
            ],
        )
        self.assertEqual(previewer._cv2.imshow_calls[0][0], "realsense:front")

    def test_tactile_preview_can_use_sample_hand_states(self) -> None:
        tactile = np.arange(6, dtype=np.float32)
        result = tactile_by_side_from_hand_states(
            ("left", "right"),
            {"left": {"tactile": tactile}, "right": {}},
        )

        np.testing.assert_allclose(result["left"], tactile)
        self.assertIsNone(result["right"])

    def test_camera_preview_uses_immutable_buffer_without_copying_depth(self) -> None:
        previewer = CameraPreviewer(CameraPreviewConfig(preflight_highgui=False))
        previewer._cv2 = _FakeCv2()
        frame = VideoFrame(image=np.ones((4, 5), dtype=np.uint8), source_timestamp=1.0)
        source = _BufferedCameraSource(frame)

        previewer.update({"front": source})

        self.assertEqual(source.get_latest_frame_calls, 0)

    def test_camera_preview_marks_age_based_stale_frame_as_placeholder(self) -> None:
        previewer = CameraPreviewer(CameraPreviewConfig(preflight_highgui=False))
        previewer._cv2 = _FakeCv2()
        frame = VideoFrame(image=np.ones((4, 5), dtype=np.uint8), source_timestamp=1.0)
        source = _BufferedCameraSource(frame)
        source.device_info = lambda: {
            "frame_count": 1,
            "last_frame_age_sec": 2.1,
            "stale_after_sec": 2.0,
        }

        previewer.update({"front": source})

        shown = previewer._cv2.imshow_calls[-1][1]
        self.assertEqual(shown.shape[:2], (480, 640))


if __name__ == "__main__":
    unittest.main()
