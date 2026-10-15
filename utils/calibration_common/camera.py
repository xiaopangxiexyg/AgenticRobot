"""Leaf camera helpers shared by calibration tools."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, TypeVar

import numpy as np

from hardware.realsense.lifecycle import (
    SDK_LIFECYCLE_LOCK,
    NativePipelineStop,
    clear_native_tracebacks,
)


@dataclass(frozen=True)
class VideoStreamProfileInfo:
    """A video profile advertised by a calibration camera."""

    stream: str
    format_name: str
    width: int
    height: int
    fps: int

    def describe(self) -> str:
        return f"{self.stream} {self.width}x{self.height}@{self.fps} {self.format_name}"


_ProfileInfo = TypeVar("_ProfileInfo", bound=VideoStreamProfileInfo)


def device_from_serial(rs: Any, serial: str | None):
    ctx = rs.context()
    devices = list(ctx.devices)
    if not devices:
        raise RuntimeError("no RealSense devices detected.")
    if serial:
        for device in devices:
            if safe_device_info(rs, device, rs.camera_info.serial_number) == str(serial):
                return device
        available = ", ".join(
            safe_device_info(rs, device, rs.camera_info.serial_number) for device in devices
        )
        raise ValueError(f"RealSense serial {serial!r} not found; available: {available}")
    return devices[0]


def enum_name(namespace: Any, value: Any, names: Sequence[str]) -> str:
    for name in names:
        if getattr(namespace, name, None) == value:
            return name
    return str(value).split(".")[-1]


def available_video_profiles(
    rs: Any,
    device: Any,
    *,
    profile_type: type[_ProfileInfo],
    stream_names: Sequence[str],
    format_names: Sequence[str],
) -> list[_ProfileInfo]:
    profiles: list[_ProfileInfo] = []
    try:
        sensors = list(device.sensors)
    except Exception:
        sensors = []
    for sensor in sensors:
        try:
            stream_profiles = sensor.get_stream_profiles()
        except Exception:
            continue
        for profile in stream_profiles:
            try:
                video = profile.as_video_stream_profile()
                info = profile_type(
                    stream=enum_name(rs.stream, profile.stream_type(), stream_names),
                    format_name=enum_name(rs.format, profile.format(), format_names),
                    width=int(video.width()),
                    height=int(video.height()),
                    fps=int(profile.fps()),
                )
            except Exception:
                continue
            profiles.append(info)
    profiles.sort(
        key=lambda item: (item.stream, item.width, item.height, item.fps, item.format_name)
    )
    return profiles


def video_profile_score(
    profile: VideoStreamProfileInfo,
    *,
    requested_width: int,
    requested_height: int,
    requested_fps: int,
    format_rank: Mapping[str, int],
) -> tuple[int, int, int, int, int, int]:
    """Rank exact resolution, then rate, area, dimensions, and format."""

    width = int(requested_width)
    height = int(requested_height)
    return (
        0 if (profile.width, profile.height) == (width, height) else 1,
        abs(profile.fps - int(requested_fps)),
        abs(profile.width * profile.height - width * height),
        abs(profile.width - width),
        abs(profile.height - height),
        format_rank.get(profile.format_name, len(format_rank)),
    )
CALIBRATION_CAMERA_STARTUP_TIMEOUT_SEC = 30.0


def start_camera_bounded(
    start: Callable[[], Any],
    *,
    timeout_sec: float,
    on_timeout: Callable[[], None] | None = None,
    on_worker: Callable[[threading.Thread], None] | None = None,
) -> Any:
    """Bound CLI camera admission without retrying a stuck native SDK call.

    On timeout the owning command must exit. The daemon worker is not joined or
    stopped because librealsense may still be in an uninterruptible USB wait.
    """

    duration = float(timeout_sec)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("timeout_sec must be finite and positive")
    result: list[Any] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            result.append(start())
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, name="calibration_camera_start", daemon=True)
    if on_worker is not None:
        on_worker(worker)
    worker.start()
    worker.join(timeout=duration)
    if worker.is_alive():
        if on_timeout is not None:
            on_timeout()
        raise TimeoutError(
            f"Calibration camera startup exceeded {duration:.3f}s; "
            "native SDK worker may still be blocked; exit this process without retry"
        )
    if errors:
        raise errors[0]
    return result[0]


def require_pyrealsense2():
    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on calibration env
        raise ImportError(
            "pyrealsense2 is required for RealSense capture. "
            "Install it in the calibration environment or use --load-samples."
        ) from exc
    return rs


def require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on calibration env
        raise ImportError(
            "OpenCV is required for target detection. Install opencv-contrib-python."
        ) from exc
    return cv2


def require_aruco(cv2: Any):
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise ImportError("cv2.aruco is missing; install opencv-contrib-python.")
    return aruco


def requested_profile_text(
    width: int,
    height: int,
    fps: int,
    *,
    color_format: str = "yuyv",
    enable_depth: bool,
) -> str:
    streams = [f"color {int(width)}x{int(height)}@{int(fps)} {color_format}"]
    if enable_depth:
        streams.append(f"depth {int(width)}x{int(height)}@{int(fps)} z16")
    return ", ".join(streams)


def safe_device_info(rs: Any, device: Any, field: Any) -> str:
    try:
        if device.supports(field):
            return str(device.get_info(field))
    except Exception:
        pass
    return ""


class CalibrationCameraLifecycle:
    """Shared native ownership for head/wrist calibration camera clients."""

    def _init_camera_lifecycle(self) -> None:
        self._camera_lifecycle_lock = threading.RLock()
        self._native_pipeline_stop: NativePipelineStop | None = None
        self._pipeline = None
        self._profile = None
        self._align = None
        self._startup_timeout_event = threading.Event()
        self._startup_worker: threading.Thread | None = None

    def _remember_startup_worker(self, worker: threading.Thread) -> None:
        if self._startup_worker is not None and self._startup_worker.is_alive():
            raise RuntimeError("Calibration camera startup worker is still active")
        self._startup_worker = worker

    def _check_pending_stop(self) -> None:
        if self._native_pipeline_stop is not None:
            error = self._native_pipeline_stop.wait(0)
            if error is not None:
                raise RuntimeError(f"Calibration camera native cleanup is not complete: {error}") from error

    def _start_owned_pipeline(self, config: Any) -> Any:
        self._check_pending_stop()
        with SDK_LIFECYCLE_LOCK:
            if self._startup_timeout_event.is_set():
                raise RuntimeError("Calibration camera startup timed out; no pipeline was opened")
            self._native_pipeline_stop = None
            self._pipeline = self._rs.pipeline()
            try:
                self._profile = self._pipeline.start(config)
            except BaseException as exc:  # noqa: BLE001 - release partial startup on interruption too.
                clear_native_tracebacks(exc)
                error = exc
            else:
                return self._profile
        self._cleanup_failed_start(error, allow_not_started=True)
        raise error

    def _cleanup_failed_start(self, error: BaseException, *, allow_not_started: bool = False) -> None:
        clear_native_tracebacks(error)
        try:
            self._stop_owned_pipeline(allow_not_started=allow_not_started)
        except BaseException as cleanup_error:  # noqa: BLE001 - keep operation and cleanup evidence.
            raise RuntimeError(f"{error}; camera cleanup failed: {cleanup_error}") from error

    def _stop_owned_pipeline(self, *, allow_not_started: bool = False,
                             before_stop: Callable[[], None] | None = None) -> None:
        self._align = None
        self._profile = None
        if self._pipeline is not None:
            self._check_pending_stop()
            self._native_pipeline_stop = NativePipelineStop(
                self._pipeline, allow_not_started=allow_not_started, before_stop=before_stop,
            )
            self._pipeline = None
            self._native_pipeline_stop.start()
        if self._native_pipeline_stop is not None:
            error = self._native_pipeline_stop.wait(self.config.pipeline_stop_timeout_sec)
            self.settings_report["pipeline.stop"] = self._native_pipeline_stop.snapshot()
            if error is not None:
                raise RuntimeError(f"Calibration camera pipeline cleanup failed: {error}") from error

    def start(self):
        with self._camera_lifecycle_lock:
            self._check_pending_stop()
            if self._startup_timeout_event.is_set():
                raise RuntimeError("Calibration camera startup timed out; reconstruct the camera")
            try:
                return self._start_camera()
            except BaseException as exc:  # noqa: BLE001 - release partial startup on interruption too.
                clear_native_tracebacks(exc)
                failure = exc
            self._cleanup_failed_start(failure)
            raise failure

    def stop(self) -> None:
        if (self._startup_worker is not None and self._startup_worker.is_alive()
                and self._startup_worker is not threading.current_thread()):
            self._startup_timeout_event.set()
            raise RuntimeError("Calibration camera startup worker is still active; cleanup is not complete")
        if not self._camera_lifecycle_lock.acquire(blocking=False):
            self._startup_timeout_event.set()
            raise RuntimeError("Calibration camera startup is still active; cleanup is not complete")
        try:
            self._stop_owned_pipeline(before_stop=self._before_pipeline_stop)
            self._active_color_format = self.config.color_format
        finally:
            self._camera_lifecycle_lock.release()

    def _before_pipeline_stop(self) -> None:
        pass


def aruco_dictionary(aruco: Any, dictionary: str):
    name = str(dictionary)
    if hasattr(aruco, name):
        return aruco.getPredefinedDictionary(getattr(aruco, name))
    try:
        return aruco.getPredefinedDictionary(int(name))
    except Exception as exc:
        raise ValueError(f"unknown ArUco dictionary {dictionary!r}.") from exc


def detector_parameters(aruco: Any):
    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()
    if hasattr(parameters, "cornerRefinementMethod") and hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    if hasattr(parameters, "cornerRefinementWinSize"):
        parameters.cornerRefinementWinSize = 5
    if hasattr(parameters, "cornerRefinementMaxIterations"):
        parameters.cornerRefinementMaxIterations = 50
    if hasattr(parameters, "cornerRefinementMinAccuracy"):
        parameters.cornerRefinementMinAccuracy = 0.01
    return parameters


def make_aruco_detector(aruco: Any, dictionary: Any, parameters: Any):
    if hasattr(aruco, "ArucoDetector"):
        return aruco.ArucoDetector(dictionary, parameters)
    return None


def put_status(cv2: Any, image: np.ndarray, text: str, *, ok: bool) -> None:
    color = (40, 180, 40) if ok else (40, 40, 220)
    cv2.rectangle(image, (8, 8), (8 + min(760, max(260, 11 * len(text))), 42), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


__all__ = [
    "CALIBRATION_CAMERA_STARTUP_TIMEOUT_SEC",
    "CalibrationCameraLifecycle",
    "VideoStreamProfileInfo",
    "aruco_dictionary",
    "detector_parameters",
    "make_aruco_detector",
    "put_status",
    "requested_profile_text",
    "require_aruco",
    "require_cv2",
    "require_pyrealsense2",
    "safe_device_info",
    "start_camera_bounded",
    "video_profile_score",
]
