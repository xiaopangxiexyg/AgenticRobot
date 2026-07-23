from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..opencv_highgui import probe_opencv_highgui


@dataclass(frozen=True)
class CameraPreviewConfig:
    window_prefix: str = "realsense"
    preflight_highgui: bool = True
    preflight_timeout_sec: float = 3.0
    resizable_window: bool = True


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
        flag = cv2.WINDOW_AUTOSIZE
        if self.config.resizable_window:
            flag = cv2.WINDOW_NORMAL | int(getattr(cv2, "WINDOW_FREERATIO", 0))
        cv2.namedWindow(name, flag)
        if self.config.resizable_window:
            set_property = getattr(cv2, "setWindowProperty", None)
            aspect_property = getattr(cv2, "WND_PROP_ASPECT_RATIO", None)
            free_ratio = getattr(cv2, "WINDOW_FREERATIO", None)
            if callable(set_property) and aspect_property is not None and free_ratio is not None:
                try:
                    set_property(name, aspect_property, free_ratio)
                except Exception:
                    pass
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
    def _is_stale(status: Mapping[str, Any]) -> bool:
        try:
            age = float(status.get("last_frame_age_sec"))
            stale_after = float(status.get("stale_after_sec"))
        except (TypeError, ValueError):
            age = stale_after = -1.0
        if age >= 0.0 and stale_after > 0.0 and age >= stale_after:
            return True
        error = str(status.get("last_error") or "").lower()
        return bool(error and ("stale" in error or "disconnect" in error or "restart" in error))

    @staticmethod
    def _latest_preview_frame(source: Any) -> Any | None:
        """Use the immutable frame buffer so preview does not copy depth."""

        frame_buffer = getattr(source, "frame_buffer", None)
        if callable(frame_buffer):
            try:
                get_latest = getattr(frame_buffer(), "get_latest", None)
                if callable(get_latest):
                    return get_latest()
            except Exception:  # noqa: BLE001 - fall back to compatibility API.
                pass
        return source.get_latest_frame()

    def _closed_window_keys(self) -> list[str]:
        cv2 = self._require_cv2()
        get_property = getattr(cv2, "getWindowProperty", None)
        visible_property = getattr(cv2, "WND_PROP_VISIBLE", None)
        if not callable(get_property) or visible_property is None:
            return []
        for name in tuple(self._window_names):
            try:
                if float(get_property(name, visible_property)) < 1.0:
                    return ["q"]
            except Exception:  # noqa: BLE001 - unsupported by some HighGUI builds.
                continue
        return []

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
        image = np.full((height, width, 3), (36, 30, 20), dtype=np.uint8)
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
                (80, 210, 255),
                max(1, int(round(scale * 2))),
                cv2.LINE_AA,
            )
            y += step
        return image

    def update(self, camera_sources: Mapping[str, Any]) -> list[str]:
        if self._closed or not self.ensure_available():
            return []
        any_window = False
        for camera_key, source in camera_sources.items():
            frame = None
            extra_status = None
            status = self._camera_status(source)
            last_error = status.get("last_error")
            stale = self._is_stale(status)
            try:
                frame = self._latest_preview_frame(source)
            except Exception as exc:  # noqa: BLE001
                extra_status = f"get_latest_frame_failed:{type(exc).__name__}:{exc}"
            if stale:
                extra_status = self._short_text(last_error)
            if frame is None or stale:
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
            closed_keys = self._closed_window_keys()
            if closed_keys:
                return closed_keys
            key_code = self._require_cv2().waitKey(1)
            if key_code >= 0:
                key = key_code & 0xFF
                if key == 3:
                    raise KeyboardInterrupt
                if 0 < key < 128:
                    return [chr(key).lower()]
        return []

    def update_frame_matches(
        self,
        camera_frames: Mapping[str, Any],
        *,
        camera_sources: Mapping[str, Any] | None = None,
    ) -> list[str]:
        if self._closed or not self.ensure_available():
            return []
        any_window = False
        sources = {} if camera_sources is None else dict(camera_sources)
        for camera_key, frame_or_match in camera_frames.items():
            key = str(camera_key)
            frame = getattr(frame_or_match, "frame", frame_or_match)
            source = sources.get(key)
            if frame is None:
                if source is None:
                    continue
                image = self._placeholder_image(
                    camera_key=key,
                    source=source,
                    extra_status="no matched frame",
                )
            else:
                image = self._display_image(frame)
            name = self._window_name(key)
            self._ensure_window(name)
            self._require_cv2().imshow(name, image)
            any_window = True
        if any_window:
            closed_keys = self._closed_window_keys()
            if closed_keys:
                return closed_keys
            key_code = self._require_cv2().waitKey(1)
            if key_code >= 0:
                key = key_code & 0xFF
                if key == 3:
                    raise KeyboardInterrupt
                if 0 < key < 128:
                    return [chr(key).lower()]
        return []

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
