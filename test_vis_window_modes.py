from __future__ import annotations

import unittest

import numpy as np

from inspire_data_collect.run_collect import add_args as add_collect_args
from inspire_data_collect.deploy.run_policy import add_args as add_deploy_args
from inspire_data_collect.utils.camera_preview import CameraPreviewConfig, CameraPreviewer
from inspire_data_collect.utils.tactile_visualizer import TactileVisualizerConfig, TactileUVVisualizer


class _FakeCv2:
    WINDOW_NORMAL = 16
    WINDOW_AUTOSIZE = 1
    WINDOW_FREERATIO = 256
    INTER_LINEAR = 1

    def __init__(self):
        self.named_windows: list[tuple[str, int]] = []
        self.resized_windows: list[tuple[str, int, int]] = []
        self.resize_calls: list[tuple[tuple[int, ...], tuple[int, int], int]] = []

    def namedWindow(self, name: str, flag: int) -> None:
        self.named_windows.append((name, int(flag)))

    def resizeWindow(self, name: str, width: int, height: int) -> None:
        self.resized_windows.append((name, int(width), int(height)))

    def resize(self, image: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
        width, height = size
        self.resize_calls.append((tuple(image.shape), (int(width), int(height)), int(interpolation)))
        channels = () if image.ndim == 2 else (image.shape[2],)
        return np.zeros((height, width, *channels), dtype=image.dtype)


class VisWindowModeTests(unittest.TestCase):
    def test_collect_parser_exposes_camera_preview_window_mode(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()
        add_collect_args(parser)
        args = parser.parse_args(
            [
                "--task-name",
                "x",
                "--vis-cam-window-mode",
                "autosize",
                "--vis-cam-render-scale",
                "3",
                "--tactile-vis-render-scale",
                "4",
            ]
        )
        self.assertEqual(args.vis_cam_window_mode, "autosize")
        self.assertEqual(args.vis_cam_render_scale, 3.0)
        self.assertEqual(args.tactile_vis_render_scale, 4.0)

    def test_deploy_parser_exposes_camera_preview_window_mode(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()
        add_deploy_args(parser)
        args = parser.parse_args(
            [
                "--vis-cam-window-mode",
                "autosize",
                "--vis-cam-render-scale",
                "3",
                "--tactile-vis-render-scale",
                "4",
            ]
        )
        self.assertEqual(args.vis_cam_window_mode, "autosize")
        self.assertEqual(args.vis_cam_render_scale, 3.0)
        self.assertEqual(args.tactile_vis_render_scale, 4.0)

    def test_camera_preview_uses_freeratio_resizable_window(self) -> None:
        previewer = CameraPreviewer(CameraPreviewConfig(resizable_window=True))
        previewer._cv2 = _FakeCv2()

        previewer._ensure_window("preview")

        self.assertEqual(
            previewer._cv2.named_windows,
            [("preview", _FakeCv2.WINDOW_NORMAL | _FakeCv2.WINDOW_FREERATIO)],
        )
        self.assertEqual(previewer._cv2.resized_windows, [("preview", 640, 480)])

    def test_camera_preview_render_scale_upscales_backing_image(self) -> None:
        previewer = CameraPreviewer(
            CameraPreviewConfig(resizable_window=True, render_scale=2.5)
        )
        previewer._cv2 = _FakeCv2()
        image = np.ones((100, 200, 3), dtype=np.uint8)

        scaled = previewer._scaled_preview_image(image)

        self.assertEqual(scaled.shape, (250, 500, 3))
        self.assertEqual(
            previewer._cv2.resize_calls,
            [((100, 200, 3), (500, 250), _FakeCv2.INTER_LINEAR)],
        )

    def test_tactile_visualizer_uses_freeratio_resizable_window(self) -> None:
        visualizer = object.__new__(TactileUVVisualizer)
        visualizer.config = TactileVisualizerConfig(resizable_window=True)
        visualizer._cv2 = _FakeCv2()
        visualizer._window_created = False

        visualizer._ensure_window(__import__("numpy").zeros((100, 200, 3), dtype="uint8"))

        self.assertEqual(
            visualizer._cv2.named_windows,
            [
                (
                    visualizer.config.window_name,
                    _FakeCv2.WINDOW_NORMAL | _FakeCv2.WINDOW_FREERATIO,
                )
            ],
        )
        self.assertEqual(
            visualizer._cv2.resized_windows,
            [(visualizer.config.window_name, 200, 100)],
        )

    def test_tactile_render_scale_upscales_backing_canvas(self) -> None:
        visualizer = object.__new__(TactileUVVisualizer)
        visualizer.config = TactileVisualizerConfig(resizable_window=True, render_scale=2.5)
        visualizer._cv2 = _FakeCv2()
        canvas = np.ones((100, 200, 3), dtype=np.uint8)

        scaled = visualizer._scaled_canvas(canvas)

        self.assertEqual(scaled.shape, (250, 500, 3))
        self.assertEqual(
            visualizer._cv2.resize_calls,
            [((100, 200, 3), (500, 250), _FakeCv2.INTER_LINEAR)],
        )


if __name__ == "__main__":
    unittest.main()
