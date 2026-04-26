from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .opencv_highgui import probe_opencv_highgui


@dataclass(frozen=True)
class CameraPreviewConfig:
    window_prefix: str = "realsense"
    preflight_highgui: bool = True
    preflight_timeout_sec: float = 3.0


class CameraPreviewer:
    def __init__(self, config: CameraPreviewConfig | None = None):
        self.config = config if config is not None else CameraPreviewConfig()
        self._cv2 = None
        self._window_names: set[str] = set()
        self._closed = False
        self._preflight_done = False
        self._disabled_reason: str | None = None

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def _require_cv2(self) -> Any:
        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:
                raise ImportError("OpenCV (cv2) is required for --vis_cam.") from exc
            self._cv2 = cv2
        return self._cv2

    def ensure_available(self) -> bool:
        if self._closed:
            return False
        if self._disabled_reason is not None:
            return False
        if self._preflight_done:
            return True
        self._preflight_done = True
        if self.config.preflight_highgui:
            result = probe_opencv_highgui(
                timeout_sec=float(self.config.preflight_timeout_sec),
                window_name=f"{self.config.window_prefix}_probe",
            )
            if not result.ok:
                self._disabled_reason = result.reason
                return False
        self._require_cv2()
        return True

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

    def _camera_dimensions(self, source: Any) -> tuple[int, int]:
        config = getattr(source, "config", None)
        width = int(getattr(config, "width", 640) or 640)
        height = int(getattr(config, "height", 480) or 480)
        if width <= 0 or height <= 0:
            return 640, 480
        return width, height

    def _camera_status(self, source: Any) -> dict[str, Any]:
        device_info = getattr(source, "device_info", None)
        if not callable(device_info):
            return {}
        try:
            return dict(device_info())
        except Exception as exc:  # noqa: BLE001
            return {"last_error": f"device_info_failed:{type(exc).__name__}:{exc}"}

    @staticmethod
    def _short_text(value: object, *, max_chars: int = 120) -> str:
        text = str(value)
        if len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 3)] + "..."

    def _placeholder_image(
        self,
        *,
        camera_key: str,
        source: Any,
        extra_status: str | None = None,
    ) -> np.ndarray:
        cv2 = self._require_cv2()
        width, height = self._camera_dimensions(source)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        status = self._camera_status(source)
        lines = [
            f"{self.config.window_prefix}:{camera_key}",
            "waiting for RealSense frame",
            f"frames={status.get('frame_count', 0)}",
        ]
        last_error = status.get("last_error")
        if last_error:
            lines.append(f"last_error={self._short_text(last_error)}")
        if extra_status:
            lines.append(self._short_text(extra_status))

        y = max(28, min(54, height // 10))
        step = max(24, min(42, height // 16))
        scale = max(0.5, min(1.0, width / 1280.0))
        for line in lines:
            cv2.putText(
                image,
                line,
                (24, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (230, 230, 230),
                max(1, int(round(scale * 2))),
                cv2.LINE_AA,
            )
            y += step
        return image

    def update(self, camera_sources: Mapping[str, Any]) -> None:
        if self._closed or not self.ensure_available():
            return
        any_window = False
        for camera_key, source in camera_sources.items():
            frame = None
            extra_status = None
            try:
                frame = source.get_latest_frame()
            except Exception as exc:  # noqa: BLE001
                extra_status = f"get_latest_frame_failed:{type(exc).__name__}:{exc}"
            if frame is None:
                image = self._placeholder_image(
                    camera_key=str(camera_key),
                    source=source,
                    extra_status=extra_status,
                )
            else:
                image = self._display_image(frame)
            name = self._window_name(str(camera_key))
            self._ensure_window(name)
            self._require_cv2().imshow(name, image)
            any_window = True
        if any_window:
            self._require_cv2().waitKey(1)

    def status_snapshot(self, camera_sources: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            str(camera_key): self._camera_status(source)
            for camera_key, source in camera_sources.items()
        }

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
