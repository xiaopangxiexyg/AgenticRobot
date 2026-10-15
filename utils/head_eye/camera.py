from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from utils.calibration_common.camera import (
    CALIBRATION_CAMERA_STARTUP_TIMEOUT_SEC,
    CalibrationCameraLifecycle,
    VideoStreamProfileInfo as _CommonVideoStreamProfileInfo,
    available_video_profiles as _collect_video_profiles,
    aruco_dictionary as _aruco_dictionary,
    device_from_serial as _device_from_serial,
    detector_parameters as _detector_parameters,
    make_aruco_detector as _make_aruco_detector,
    put_status as _put_status,
    requested_profile_text as _requested_profile_text,
    require_aruco as _require_aruco,
    require_cv2 as _require_cv2,
    require_pyrealsense2 as _require_pyrealsense2,
    safe_device_info as _safe_device_info,
    start_camera_bounded,
    video_profile_score as _video_profile_score,
)
from hardware.realsense.color import (
    color_format_value as _color_format_value,
    color_image_to_bgr as _color_image_to_bgr,
    normalize_color_format as _normalize_color_format,
)

from .transforms import make_transform


DEFAULT_CAMERA_SERIALS = [
    "239722072965",
    "247122073084",
    "243522072934",
    "247122073147",
    "017322074878",
]
CAMERA_IDS = DEFAULT_CAMERA_SERIALS


@dataclass(frozen=True)
class RealSenseDevice:
    index: int
    name: str
    serial: str
    product_line: str | None = None
    firmware_version: str | None = None


@dataclass(frozen=True)
class VideoStreamProfileInfo(_CommonVideoStreamProfileInfo):
    """Preserve the fixed camera's historical profile type path."""


@dataclass
class CameraFrame:
    color_bgr: np.ndarray
    depth_z16: np.ndarray | None
    intrinsics_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    serial: str | None
    timestamp_sec: float
    depth_scale_m: float | None = None


@dataclass(frozen=True)
class RealSenseConfig:
    serial: str | None = None
    camera_index: int | None = None
    resolution: tuple[int, int] = (1280, 720)
    fps: int = 15
    # Request native YUYV transport; CameraFrame.color_bgr remains BGR.
    color_format: str = "yuyv"
    enable_depth: bool = True
    # Only used after the explicitly requested profile fails to start.
    auto_select_supported_profile: bool = True
    startup_wait_timeout_ms: int = 1000
    startup_frame_grace_sec: float = 5.0
    pipeline_stop_timeout_sec: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "color_format", _normalize_color_format(self.color_format))
        if int(self.startup_wait_timeout_ms) <= 0:
            raise ValueError("startup_wait_timeout_ms must be positive")
        if float(self.startup_frame_grace_sec) <= 0.0:
            raise ValueError("startup_frame_grace_sec must be positive")
        if not math.isfinite(self.pipeline_stop_timeout_sec) or self.pipeline_stop_timeout_sec <= 0:
            raise ValueError("pipeline_stop_timeout_sec must be finite and positive")


class RealSenseCamera(CalibrationCameraLifecycle):
    def __init__(self, config: RealSenseConfig):
        self.config = config
        self._rs = _require_pyrealsense2()
        self._init_camera_lifecycle()
        self._active_color_format = self.config.color_format
        self.serial: str | None = None
        self.depth_scale_m: float | None = None
        # Saved by the command-line tool with new calibration samples/results.
        self.settings_report: dict[str, Any] = {}

    @staticmethod
    def list_devices() -> list[RealSenseDevice]:
        rs = _require_pyrealsense2()
        ctx = rs.context()
        devices: list[RealSenseDevice] = []
        for index, device in enumerate(ctx.devices):
            devices.append(
                RealSenseDevice(
                    index=index,
                    name=_safe_device_info(rs, device, rs.camera_info.name),
                    serial=_safe_device_info(rs, device, rs.camera_info.serial_number),
                    product_line=_safe_device_info(rs, device, rs.camera_info.product_line),
                    firmware_version=_safe_device_info(rs, device, rs.camera_info.firmware_version),
                )
            )
        return devices

    def _start_camera(self) -> "RealSenseCamera":
        if self._startup_timeout_event.is_set():
            raise RuntimeError("Calibration camera startup timed out; reconstruct the camera")
        if self._pipeline is not None:
            return self
        serial = self.config.serial
        if serial is None and self.config.camera_index is not None:
            serial = _serial_from_index(self.config.camera_index)
        self.serial = serial

        rs = self._rs
        config = rs.config()
        color_format = _color_format_value(rs, self.config.color_format)
        if serial:
            config.enable_device(str(serial))
        width, height = self.config.resolution
        if self.config.enable_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, int(self.config.fps))
        config.enable_stream(rs.stream.color, width, height, color_format, int(self.config.fps))
        try:
            profile = self._start_owned_pipeline(config)
            selected_profiles = {
                "requested": _requested_profile_text(
                    width,
                    height,
                    int(self.config.fps),
                    color_format=self.config.color_format,
                    enable_depth=self.config.enable_depth,
                ),
                "fallback_used": False,
            }
        except Exception as exc:
            self._check_pending_stop()
            if not self.config.auto_select_supported_profile:
                raise RuntimeError(self._profile_error_message(exc, serial)) from exc
            fallback_config, selected_profiles, active_color_format = self._fallback_stream_config(
                serial
            )
            try:
                profile = self._start_owned_pipeline(fallback_config)
            except Exception as fallback_exc:
                raise RuntimeError(self._profile_error_message(fallback_exc, serial)) from fallback_exc
        if self._startup_timeout_event.is_set():
            raise RuntimeError("Calibration camera startup timed out")
        self._wait_for_first_frameset(self._pipeline)
        if self._startup_timeout_event.is_set():
            raise RuntimeError("Calibration camera startup timed out")
        self._align = rs.align(rs.stream.color) if self.config.enable_depth else None
        self._active_color_format = (
            self.config.color_format
            if not selected_profiles.get("fallback_used")
            else active_color_format
        )
        self.settings_report = selected_profiles
        self.settings_report["color.format"] = self._active_color_format
        if self.serial is None:
            try:
                device = profile.get_device()
                self.serial = device.get_info(rs.camera_info.serial_number)
            except Exception:
                self.serial = None
        if self.config.enable_depth:
            try:
                self.depth_scale_m = float(profile.get_device().first_depth_sensor().get_depth_scale())
            except Exception:
                self.depth_scale_m = None
        if self._startup_timeout_event.is_set():
            self.stop()
            raise RuntimeError("Calibration camera startup timed out")
        return self

    def _wait_for_first_frameset(self, pipeline: Any) -> None:
        deadline = time.monotonic() + float(self.config.startup_frame_grace_sec)
        attempts = 0
        last_error = "no frameset"
        while True:
            remaining_sec = deadline - time.monotonic()
            if remaining_sec <= 0.0:
                break
            timeout_ms = max(
                1,
                min(int(self.config.startup_wait_timeout_ms), int(remaining_sec * 1000)),
            )
            attempts += 1
            try:
                frames = pipeline.wait_for_frames(timeout_ms)
                if frames.get_color_frame() and (
                    not self.config.enable_depth or frames.get_depth_frame()
                ):
                    return
                last_error = "incomplete color/depth frameset"
            except RuntimeError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(
            "RealSense calibration camera first-frame warmup failed: "
            f"serial={self.serial!r} attempts={attempts} last_error={last_error}"
        )

    def _fallback_stream_config(
        self,
        serial: str | None,
    ) -> tuple[Any, dict[str, Any], str]:
        """Build a compatible profile only after the requested one was rejected.

        The fixed-camera calibration always needs matched color/depth streams:
        depth is aligned to color before use, so deliberately avoid combining
        independently "closest" stream profiles here.
        """
        rs = self._rs
        width, height = self.config.resolution
        fps = int(self.config.fps)
        device = _device_from_serial(rs, serial)
        profiles = _available_video_profiles(rs, device)
        depth_profile: VideoStreamProfileInfo | None = None
        if self.config.enable_depth:
            pair = _choose_compatible_video_profile_pair(
                profiles,
                requested_width=width,
                requested_height=height,
                requested_fps=fps,
                preferred_color_formats=(
                    self.config.color_format,
                    "yuyv",
                    "bgr8",
                    "rgb8",
                    "rgba8",
                    "bgra8",
                    "y8",
                ),
            )
            if pair is None:
                raise RuntimeError(
                    self._profile_error_message("no compatible color/depth profile pair", serial)
                )
            color_profile, depth_profile = pair
        else:
            color_profile = _choose_video_profile(
                profiles,
                stream="color",
                preferred_formats=(
                    self.config.color_format,
                    "yuyv",
                    "bgr8",
                    "rgb8",
                    "rgba8",
                    "bgra8",
                    "y8",
                ),
                requested_width=width,
                requested_height=height,
                requested_fps=fps,
            )
            if color_profile is None:
                raise RuntimeError(self._profile_error_message("no supported color profile", serial))

        config = rs.config()
        if serial:
            config.enable_device(str(serial))
        config.enable_stream(
            rs.stream.color,
            color_profile.width,
            color_profile.height,
            getattr(rs.format, color_profile.format_name),
            color_profile.fps,
        )
        report: dict[str, Any] = {
            "requested": _requested_profile_text(
                width,
                height,
                fps,
                color_format=self.config.color_format,
                enable_depth=self.config.enable_depth,
            ),
            "fallback_used": True,
            "fallback_color_profile": color_profile.describe(),
        }
        if depth_profile is not None:
            config.enable_stream(
                rs.stream.depth,
                depth_profile.width,
                depth_profile.height,
                getattr(rs.format, depth_profile.format_name),
                depth_profile.fps,
            )
            report["fallback_depth_profile"] = depth_profile.describe()
        return config, report, color_profile.format_name

    def _profile_error_message(self, exc: object, serial: str | None) -> str:
        width, height = self.config.resolution
        lines = [
            "RealSense could not start the requested stream profile.",
            (
                f"requested camera={serial or 'auto'} "
                f"color={width}x{height}@{int(self.config.fps)} "
                f"{self.config.color_format} "
                f"depth={bool(self.config.enable_depth)}"
            ),
            f"error: {exc}",
        ]
        try:
            profiles = available_video_profiles(serial)
        except Exception as profile_exc:
            profiles = []
            lines.append(f"available profile query failed: {profile_exc}")
        if profiles:
            lines.append("available profiles:")
            for item in profiles[:80]:
                lines.append(f"  - {item.describe()}")
            if len(profiles) > 80:
                lines.append(f"  ... {len(profiles) - 80} more")
        lines.append(
            "Try --resolution 640 480 --fps 15, or run "
            "--list-cameras --show-profiles."
        )
        return "\n".join(lines)

    def read(self, *, timeout_ms: int = 5000) -> CameraFrame:
        if self._pipeline is None:
            raise RuntimeError("RealSenseCamera.start() must be called before read().")
        frames = self._pipeline.wait_for_frames(timeout_ms)
        if self._align is not None:
            frames = self._align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("RealSense color frame is not available.")
        depth_frame = frames.get_depth_frame() if self.config.enable_depth else None
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        intrinsics_matrix = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        color_image = _color_image_to_bgr(
            np.asanyarray(color_frame.get_data()),
            self._active_color_format,
        )
        depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
        timestamp_ms = float(color_frame.get_timestamp() or 0.0)
        timestamp_sec = timestamp_ms / 1000.0 if timestamp_ms > 0.0 else time.time()
        return CameraFrame(
            color_bgr=color_image,
            depth_z16=depth_image,
            intrinsics_matrix=intrinsics_matrix,
            distortion_coeffs=np.asarray(intr.coeffs, dtype=np.float64),
            serial=self.serial,
            timestamp_sec=timestamp_sec,
            depth_scale_m=self.depth_scale_m,
        )

    def __enter__(self) -> "RealSenseCamera":
        return start_camera_bounded(
            self.start,
            timeout_sec=CALIBRATION_CAMERA_STARTUP_TIMEOUT_SEC,
            on_timeout=self._startup_timeout_event.set,
            on_worker=self._remember_startup_worker,
        )

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


@dataclass(frozen=True)
class TargetConfig:
    target_type: str = "aruco"
    dictionary: str = "DICT_6X6_100"
    marker_length_m: float = 0.10
    marker_id: int | None = None
    charuco_squares_x: int = 5
    charuco_squares_y: int = 7
    charuco_square_length_m: float = 0.035
    charuco_marker_length_m: float = 0.026
    min_charuco_corners: int = 12
    axis_length_m: float = 0.05


@dataclass
class TargetDetection:
    T_target_to_camera: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    corner_count: int
    marker_ids: list[int]
    reprojection_rmse_px: float | None = None


class TargetDetector:
    def __init__(self, config: TargetConfig):
        self.config = config
        self._cv2 = _require_cv2()
        self._aruco = _require_aruco(self._cv2)
        self._dictionary = _aruco_dictionary(self._aruco, config.dictionary)
        self._parameters = _detector_parameters(self._aruco)
        self._detector = _make_aruco_detector(self._aruco, self._dictionary, self._parameters)
        self._charuco_board = self._make_charuco_board() if config.target_type == "charuco" else None

    def detect(
        self,
        color_bgr: np.ndarray,
        intrinsics_matrix: np.ndarray,
        distortion_coeffs: np.ndarray,
        *,
        draw: bool = True,
    ) -> tuple[TargetDetection | None, np.ndarray]:
        cv2 = self._cv2
        aruco = self._aruco
        image = np.asarray(color_bgr).copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, _ = self._detect_markers(gray)
        if ids is None or len(corners) == 0:
            if draw:
                _put_status(cv2, image, "target: not detected", ok=False)
            return None, image
        if draw:
            aruco.drawDetectedMarkers(image, corners, ids)
        if self.config.target_type == "charuco":
            detection = self._detect_charuco(gray, corners, ids, intrinsics_matrix, distortion_coeffs)
        else:
            detection = self._detect_aruco(corners, ids, intrinsics_matrix, distortion_coeffs)
        if detection is not None and draw:
            self._draw_axes(image, intrinsics_matrix, distortion_coeffs, detection)
            reproj = detection.reprojection_rmse_px
            suffix = "" if reproj is None else f", reproj={reproj:.2f}px"
            _put_status(cv2, image, f"target: ok corners={detection.corner_count}{suffix}", ok=True)
        elif draw:
            reason = "need one marker" if self.config.target_type == "aruco" else "need more charuco corners"
            _put_status(cv2, image, f"target: {reason}", ok=False)
        return detection, image

    def _detect_markers(self, gray: np.ndarray):
        if self._detector is not None:
            return self._detector.detectMarkers(gray)
        return self._aruco.detectMarkers(
            gray,
            self._dictionary,
            parameters=self._parameters,
        )

    def _detect_aruco(
        self,
        corners: Sequence[np.ndarray],
        ids: np.ndarray,
        intrinsics_matrix: np.ndarray,
        distortion_coeffs: np.ndarray,
    ) -> TargetDetection | None:
        ids_flat = [int(value) for value in np.asarray(ids).reshape(-1)]
        selected_index: int | None = None
        if self.config.marker_id is not None:
            for index, marker_id in enumerate(ids_flat):
                if marker_id == int(self.config.marker_id):
                    selected_index = index
                    break
        elif len(ids_flat) == 1:
            selected_index = 0
        if selected_index is None:
            return None
        selected_corners = [np.asarray(corners[selected_index], dtype=np.float64)]
        rvecs, tvecs, _ = self._aruco.estimatePoseSingleMarkers(
            selected_corners,
            float(self.config.marker_length_m),
            intrinsics_matrix,
            distortion_coeffs,
        )
        rvec = np.asarray(rvecs[0], dtype=np.float64).reshape(3)
        tvec = np.asarray(tvecs[0], dtype=np.float64).reshape(3)
        rotation, _ = self._cv2.Rodrigues(rvec)
        rmse = _marker_reprojection_rmse(
            self._cv2,
            selected_corners[0].reshape(4, 2),
            rvec,
            tvec,
            float(self.config.marker_length_m),
            intrinsics_matrix,
            distortion_coeffs,
        )
        return TargetDetection(
            T_target_to_camera=make_transform(rotation, tvec),
            rvec=rvec,
            tvec=tvec,
            corner_count=4,
            marker_ids=[ids_flat[selected_index]],
            reprojection_rmse_px=rmse,
        )

    def _detect_charuco(
        self,
        gray: np.ndarray,
        corners: Sequence[np.ndarray],
        ids: np.ndarray,
        intrinsics_matrix: np.ndarray,
        distortion_coeffs: np.ndarray,
    ) -> TargetDetection | None:
        if self._charuco_board is None:
            return None
        aruco = self._aruco
        cv2 = self._cv2
        try:
            count, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners,
                ids,
                gray,
                self._charuco_board,
                cameraMatrix=intrinsics_matrix,
                distCoeffs=distortion_coeffs,
            )
        except TypeError:
            count, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners,
                ids,
                gray,
                self._charuco_board,
                intrinsics_matrix,
                distortion_coeffs,
            )
        if (
            charuco_ids is None
            or charuco_corners is None
            or int(count) < int(self.config.min_charuco_corners)
        ):
            return None
        try:
            ok, rvec, tvec = aruco.estimatePoseCharucoBoard(
                charuco_corners,
                charuco_ids,
                self._charuco_board,
                intrinsics_matrix,
                distortion_coeffs,
                None,
                None,
            )
        except TypeError:
            rvec = np.zeros((3, 1), dtype=np.float64)
            tvec = np.zeros((3, 1), dtype=np.float64)
            ok = aruco.estimatePoseCharucoBoard(
                charuco_corners,
                charuco_ids,
                self._charuco_board,
                intrinsics_matrix,
                distortion_coeffs,
                rvec,
                tvec,
            )
        if not ok:
            return None
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        rotation, _ = cv2.Rodrigues(rvec)
        rmse = _charuco_reprojection_rmse(
            cv2,
            self._charuco_board,
            charuco_corners,
            charuco_ids,
            rvec,
            tvec,
            intrinsics_matrix,
            distortion_coeffs,
        )
        return TargetDetection(
            T_target_to_camera=make_transform(rotation, tvec),
            rvec=rvec,
            tvec=tvec,
            corner_count=int(count),
            marker_ids=[int(value) for value in np.asarray(ids).reshape(-1)],
            reprojection_rmse_px=rmse,
        )

    def _make_charuco_board(self):
        aruco = self._aruco
        size = (int(self.config.charuco_squares_x), int(self.config.charuco_squares_y))
        square_length = float(self.config.charuco_square_length_m)
        marker_length = float(self.config.charuco_marker_length_m)
        if hasattr(aruco, "CharucoBoard"):
            return aruco.CharucoBoard(size, square_length, marker_length, self._dictionary)
        if hasattr(aruco, "CharucoBoard_create"):
            return aruco.CharucoBoard_create(
                size[0],
                size[1],
                square_length,
                marker_length,
                self._dictionary,
            )
        raise RuntimeError("this OpenCV build does not expose CharucoBoard.")

    def _draw_axes(
        self,
        image: np.ndarray,
        intrinsics_matrix: np.ndarray,
        distortion_coeffs: np.ndarray,
        detection: TargetDetection,
    ) -> None:
        if hasattr(self._cv2, "drawFrameAxes"):
            self._cv2.drawFrameAxes(
                image,
                intrinsics_matrix,
                distortion_coeffs,
                detection.rvec.reshape(3, 1),
                detection.tvec.reshape(3, 1),
                float(self.config.axis_length_m),
            )


def available_video_profiles(serial: str | None = None) -> list[VideoStreamProfileInfo]:
    """Return the native video profiles advertised by one RealSense device."""
    rs = _require_pyrealsense2()
    device = _device_from_serial(rs, serial)
    return _available_video_profiles(rs, device)


def _available_video_profiles(rs: Any, device: Any) -> list[VideoStreamProfileInfo]:
    return _collect_video_profiles(
        rs,
        device,
        profile_type=VideoStreamProfileInfo,
        stream_names=("color", "depth", "infrared"),
        format_names=("bgr8", "bgra8", "rgb8", "rgba8", "uyvy", "y8", "yuyv", "z16"),
    )


def _choose_compatible_video_profile_pair(
    profiles: Sequence[VideoStreamProfileInfo],
    *,
    requested_width: int,
    requested_height: int,
    requested_fps: int,
    preferred_color_formats: Sequence[str] = ("yuyv", "bgr8"),
) -> tuple[VideoStreamProfileInfo, VideoStreamProfileInfo] | None:
    format_rank = {
        str(name).strip().lower(): index
        for index, name in enumerate(preferred_color_formats)
    }
    colors = [
        item
        for item in profiles
        if item.stream == "color" and item.format_name in format_rank
    ]
    depths = [
        item
        for item in profiles
        if item.stream == "depth" and item.format_name == "z16"
    ]
    pairs = [
        (color, depth)
        for color in colors
        for depth in depths
        if (color.width, color.height, color.fps)
        == (depth.width, depth.height, depth.fps)
    ]
    if not pairs:
        return None

    return min(
        pairs,
        key=lambda pair: _video_profile_score(
            pair[0],
            requested_width=requested_width,
            requested_height=requested_height,
            requested_fps=requested_fps,
            format_rank=format_rank,
        ),
    )


def _choose_video_profile(
    profiles: Sequence[VideoStreamProfileInfo],
    *,
    stream: str,
    preferred_formats: Sequence[str],
    requested_width: int,
    requested_height: int,
    requested_fps: int,
) -> VideoStreamProfileInfo | None:
    format_names = {str(item) for item in preferred_formats}
    candidates = [
        item
        for item in profiles
        if item.stream == str(stream) and item.format_name in format_names
    ]
    if not candidates:
        return None

    format_rank = {
        str(name).strip().lower(): index for index, name in enumerate(preferred_formats)
    }
    return min(
        candidates,
        key=lambda item: _video_profile_score(
            item,
            requested_width=requested_width,
            requested_height=requested_height,
            requested_fps=requested_fps,
            format_rank=format_rank,
        ),
    )


def _serial_from_index(index: int) -> str:
    devices = RealSenseCamera.list_devices()
    if 0 <= int(index) < len(devices):
        return devices[int(index)].serial
    if 0 <= int(index) < len(DEFAULT_CAMERA_SERIALS):
        return DEFAULT_CAMERA_SERIALS[int(index)]
    raise ValueError(f"camera index {index} is not available.")


def _marker_reprojection_rmse(
    cv2: Any,
    image_corners: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    marker_length_m: float,
    intrinsics_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
) -> float:
    half = float(marker_length_m) * 0.5
    object_points = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )
    projected, _ = cv2.projectPoints(
        object_points,
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        intrinsics_matrix,
        distortion_coeffs,
    )
    diff = projected.reshape(-1, 2) - np.asarray(image_corners, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

def _charuco_reprojection_rmse(
    cv2: Any,
    board: Any,
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    intrinsics_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
) -> float | None:
    object_corners = None
    if hasattr(board, "getChessboardCorners"):
        object_corners = board.getChessboardCorners()
    elif hasattr(board, "chessboardCorners"):
        object_corners = board.chessboardCorners
    if object_corners is None:
        return None
    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    object_points = np.asarray(object_corners, dtype=np.float64)[ids]
    image_points = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
    projected, _ = cv2.projectPoints(
        object_points,
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        intrinsics_matrix,
        distortion_coeffs,
    )
    diff = projected.reshape(-1, 2) - image_points
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
