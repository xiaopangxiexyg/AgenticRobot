from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CameraPreviewConfig:
    window_prefix: str = "realsense"


class CameraPreviewer:
    def __init__(self, config: CameraPreviewConfig | None = None):
        self.config = config if config is not None else CameraPreviewConfig()
        self._cv2 = None
        self._window_names: set[str] = set()
        self._closed = False

    def _require_cv2(self) -> Any:
        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:
                raise ImportError("OpenCV (cv2) is required for --vis_cam.") from exc
            self._cv2 = cv2
        return self._cv2

    def _window_name(self, camera_key: str) -> str:
        return f"{self.config.window_prefix}:{camera_key}"

    def _ensure_window(self, name: str) -> None:
        if name in self._window_names:
            return
        cv2 = self._require_cv2()
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        self._window_names.add(name)

    def _display_image(self, frame: Any) -> np.ndarray:
        cv2 = self._require_cv2()
        image = np.asarray(frame.image)
        if image.ndim == 2:
            return image
        if image.ndim != 3:
            raise ValueError(
                f"Camera preview expects HxW or HxWxC image, got shape={image.shape}."
            )
        color_space = str(getattr(frame, "metadata", {}).get("color_space", "rgb")).lower()
        if image.shape[2] == 4:
            code = cv2.COLOR_RGBA2BGR if color_space == "rgb" else cv2.COLOR_BGRA2BGR
            return cv2.cvtColor(image, code)
        if image.shape[2] == 3 and color_space == "rgb":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def update(self, camera_sources: Mapping[str, Any]) -> None:
        if self._closed:
            return
        any_window = False
        for camera_key, source in camera_sources.items():
            frame = source.get_latest_frame()
            if frame is None:
                continue
            name = self._window_name(str(camera_key))
            self._ensure_window(name)
            self._require_cv2().imshow(name, self._display_image(frame))
            any_window = True
        if any_window:
            self._require_cv2().waitKey(1)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._cv2 is None:
            return
        for name in tuple(self._window_names):
            try:
                self._cv2.destroyWindow(name)
            except Exception:
                pass
        self._window_names.clear()
        try:
            self._cv2.waitKey(1)
        except Exception:
            pass
