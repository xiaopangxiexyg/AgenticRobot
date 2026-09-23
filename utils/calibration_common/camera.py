"""Leaf camera helpers shared by calibration tools."""

from __future__ import annotations

import math
import threading
from typing import Any, Callable

import numpy as np


CALIBRATION_CAMERA_STARTUP_TIMEOUT_SEC = 30.0


def start_camera_bounded(
    start: Callable[[], Any],
    *,
    timeout_sec: float,
    on_timeout: Callable[[], None] | None = None,
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


def stop_pipeline_quietly(pipeline: Any | None) -> None:
    """Release a failed SDK pipeline without masking its original error."""

    if pipeline is None:
        return
    try:
        pipeline.stop()
    except Exception:
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
    "stop_pipeline_quietly",
]
