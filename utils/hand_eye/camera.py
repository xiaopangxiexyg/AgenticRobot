from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from utils.calibration_common.camera import (
    VideoStreamProfileInfo as _CommonVideoStreamProfileInfo,
    available_video_profiles as _available_video_profiles,
    aruco_dictionary as _aruco_dictionary,
    device_from_serial as _device_from_serial,
    detector_parameters as _detector_parameters,
    enum_name as _enum_name,
    make_aruco_detector as _make_aruco_detector,
    put_status as _put_status,
    requested_profile_text as _requested_profile_text,
    require_aruco as _require_aruco,
    require_cv2 as _require_cv2,
    require_pyrealsense2 as _require_pyrealsense2,
    safe_device_info as _safe_device_info,
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
_PROFILE_STREAM_NAMES = ("color", "depth")
_PROFILE_FORMAT_NAMES = ("bgr8", "rgb8", "z16", "y8", "yuyv", "uyvy", "rgba8", "bgra8")


@dataclass(frozen=True)
class RealSenseDevice:
    index: int
    name: str
    serial: str
    product_line: str | None = None
    firmware_version: str | None = None


@dataclass(frozen=True)
class VideoStreamProfileInfo(_CommonVideoStreamProfileInfo):
    """Preserve the wrist camera's historical profile type path."""


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
    # RealSense color sensors expose YUYV natively. CameraFrame.color_bgr
    # remains BGR after read(), regardless of this transport format.
    color_format: str = "yuyv"
    enable_depth: bool = True
    color_auto_exposure: bool | None = False
    color_exposure_us: float | None = 8000.0
    color_gain: float | None = 16.0
    color_auto_white_balance: bool | None = False
    color_white_balance: float | None = 4600.0
    color_focus: float | None = None
    depth_emitter_enabled: bool | None = False
    depth_laser_power: float | None = 0.0
    auto_select_supported_profile: bool = True
    restore_settings_on_stop: bool = True
    startup_warmup_frames: int = 0
    startup_wait_timeout_ms: int = 1000
    startup_hardware_reset_on_failure: bool = False
    startup_hardware_reset_retry_attempts: int = 1
    startup_hardware_reset_sleep_sec: float = 2.0
    startup_hardware_reset_reconnect_timeout_sec: float = 6.0
    startup_hardware_reset_reconnect_poll_sec: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(self, "color_format", _normalize_color_format(self.color_format))
        if int(self.startup_warmup_frames) < 0:
            raise ValueError("startup_warmup_frames must be >= 0")
        if int(self.startup_wait_timeout_ms) <= 0:
            raise ValueError("startup_wait_timeout_ms must be positive")
        if int(self.startup_hardware_reset_retry_attempts) < 0:
            raise ValueError("startup_hardware_reset_retry_attempts must be >= 0")
        if float(self.startup_hardware_reset_sleep_sec) < 0.0:
            raise ValueError("startup_hardware_reset_sleep_sec must be >= 0")
        if float(self.startup_hardware_reset_reconnect_timeout_sec) < 0.0:
            raise ValueError("startup_hardware_reset_reconnect_timeout_sec must be >= 0")
        if float(self.startup_hardware_reset_reconnect_poll_sec) <= 0.0:
            raise ValueError("startup_hardware_reset_reconnect_poll_sec must be positive")


@dataclass(frozen=True)
class _SensorOptionSnapshot:
    sensor: Any
    option: Any
    key: str
    value: float


class RealSenseCamera:
    def __init__(self, config: RealSenseConfig):
        self.config = config
        self._rs = _require_pyrealsense2()
        self._pipeline = None
        self._align = None
        self._profile = None
        self.serial: str | None = None
        self.depth_scale_m: float | None = None
        self.settings_report: dict[str, Any] = {}
        self._settings_snapshot: list[_SensorOptionSnapshot] = []
        self._active_color_format = self.config.color_format

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

    def start(self) -> "RealSenseCamera":
        if self._pipeline is not None:
            return self
        serial = self.config.serial
        if serial is None and self.config.camera_index is not None:
            serial = _serial_from_index(self.config.camera_index)
        self.serial = serial

        attempts = 1
        if self.config.startup_hardware_reset_on_failure:
            attempts += int(self.config.startup_hardware_reset_retry_attempts)
        last_error: Exception | None = None
        recovery_count = 0
        for attempt_index in range(attempts):
            try:
                self._start_once(serial)
                self._warmup()
                self.settings_report["startup.warmup_frames"] = int(
                    self.config.startup_warmup_frames
                )
                self.settings_report["startup.hardware_reset_recovery_count"] = recovery_count
                return self
            except Exception as exc:
                last_error = exc
                self.stop()
                should_retry = (
                    attempt_index + 1 < attempts
                    and serial is not None
                    and _is_startup_connection_error(str(exc))
                )
                if not should_retry:
                    raise
                if not self._hardware_reset_device(str(serial)):
                    raise RuntimeError(
                        "RealSense startup recovery failed: unable to reset and reconnect "
                        f"serial {serial!r} after: {exc}"
                    ) from exc
                recovery_count += 1
        if last_error is not None:  # pragma: no cover - loop always returns or raises
            raise last_error
        raise RuntimeError("RealSense startup failed without an exception")  # pragma: no cover

    def _start_once(self, serial: str | None) -> None:
        if self._pipeline is not None:
            raise RuntimeError("RealSense pipeline is already running")

        rs = self._rs
        pipeline = rs.pipeline()
        config = rs.config()
        color_format = _color_format_value(rs, self.config.color_format)
        if serial:
            config.enable_device(str(serial))
        width, height = self.config.resolution
        if self.config.enable_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, int(self.config.fps))
        config.enable_stream(rs.stream.color, width, height, color_format, int(self.config.fps))
        try:
            profile = pipeline.start(config)
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
            _stop_pipeline_quietly(pipeline)
            if not self.config.auto_select_supported_profile:
                raise RuntimeError(self._profile_error_message(exc, serial)) from exc
            fallback_config, selected_profiles, active_color_format = self._fallback_stream_config(
                serial
            )
            pipeline = rs.pipeline()
            try:
                profile = pipeline.start(fallback_config)
            except Exception as fallback_exc:
                _stop_pipeline_quietly(pipeline)
                raise RuntimeError(self._profile_error_message(fallback_exc, serial)) from fallback_exc
        self._pipeline = pipeline
        self._profile = profile
        self._active_color_format = (
            self.config.color_format
            if not selected_profiles.get("fallback_used")
            else active_color_format
        )
        self._align = rs.align(rs.stream.color) if self.config.enable_depth else None
        self.settings_report = self._apply_device_settings(profile)
        self.settings_report.update(selected_profiles)
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

    def _warmup(self) -> None:
        warmup_frames = int(self.config.startup_warmup_frames)
        if warmup_frames <= 0:
            return
        if self._pipeline is None:
            raise RuntimeError("RealSense pipeline is not running during startup warmup")
        timeout_ms = int(self.config.startup_wait_timeout_ms)
        success_count = 0
        last_error: Exception | None = None
        for _ in range(warmup_frames):
            try:
                self._pipeline.wait_for_frames(timeout_ms)
                success_count += 1
            except Exception as exc:
                last_error = exc
        if success_count <= 0:
            detail = "no frame returned"
            if last_error is not None:
                detail = f"{type(last_error).__name__}: {last_error}"
            raise RuntimeError(
                "RealSense startup warmup received no frames: "
                f"serial={self.serial!r}, frames={warmup_frames}, "
                f"timeout_ms={timeout_ms}, last_error={detail}"
            ) from last_error

    def _hardware_reset_device(self, serial: str) -> bool:
        try:
            device = _device_from_serial(self._rs, serial)
            device.hardware_reset()
            reset_sleep_sec = float(self.config.startup_hardware_reset_sleep_sec)
            if reset_sleep_sec > 0.0:
                time.sleep(reset_sleep_sec)
            return self._wait_for_device_reconnect(serial)
        except Exception:
            return False

    def _wait_for_device_reconnect(self, serial: str) -> bool:
        timeout_sec = float(self.config.startup_hardware_reset_reconnect_timeout_sec)
        deadline = time.monotonic() + timeout_sec
        while True:
            try:
                _device_from_serial(self._rs, serial)
                return True
            except (RuntimeError, ValueError):
                if time.monotonic() >= deadline:
                    return False
                time.sleep(float(self.config.startup_hardware_reset_reconnect_poll_sec))

    def _fallback_stream_config(
        self,
        serial: str | None,
    ) -> tuple[Any, dict[str, Any], str]:
        rs = self._rs
        width, height = self.config.resolution
        fps = int(self.config.fps)
        device = _device_from_serial(rs, serial)
        preferred_color_formats = [
            getattr(rs.format, name)
            for name in (
                self.config.color_format,
                "yuyv",
                "bgr8",
                "rgb8",
                "rgba8",
                "bgra8",
                "y8",
            )
            if getattr(rs.format, name, None) is not None
        ]
        color_profile = _select_video_profile(
            rs,
            device,
            stream=rs.stream.color,
            preferred_formats=preferred_color_formats,
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
        report = {
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
        if self.config.enable_depth:
            depth_profile = _select_video_profile(
                rs,
                device,
                stream=rs.stream.depth,
                preferred_formats=[rs.format.z16],
                requested_width=color_profile.width,
                requested_height=color_profile.height,
                requested_fps=color_profile.fps,
            )
            if depth_profile is None:
                raise RuntimeError(self._profile_error_message("no supported depth profile", serial))
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
            f"requested camera={serial or 'auto'} "
            f"color={width}x{height}@{int(self.config.fps)} "
            f"{self.config.color_format} depth={bool(self.config.enable_depth)}",
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
            "Try --resolution 640 480 --fps 30, or set color_format='yuyv' "
            "and run --list-cameras --show-profiles."
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

    def stop(self) -> None:
        pipeline = self._pipeline
        try:
            if pipeline is not None and self.config.restore_settings_on_stop:
                restore_report = self._restore_device_settings()
                if restore_report:
                    self.settings_report.update(restore_report)
            elif pipeline is not None and self._settings_snapshot:
                self.settings_report["restore.enabled"] = False
        finally:
            self._settings_snapshot = []
            try:
                if pipeline is not None:
                    pipeline.stop()
            finally:
                self._pipeline = None
                self._align = None
                self._profile = None
                self._active_color_format = self.config.color_format

    def __enter__(self) -> "RealSenseCamera":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _apply_device_settings(self, profile: Any) -> dict[str, Any]:
        rs = self._rs
        report: dict[str, Any] = {}
        try:
            device = profile.get_device()
        except Exception as exc:
            return {"error": f"unable to access RealSense device: {exc}"}

        color_sensor = _sensor_for_stream(rs, device, rs.stream.color)
        if color_sensor is not None:
            _set_sensor_option(
                rs,
                color_sensor,
                "enable_auto_exposure",
                self.config.color_auto_exposure,
                report,
                prefix="color",
                snapshot=self._settings_snapshot,
            )
            _set_sensor_option(
                rs,
                color_sensor,
                "exposure",
                self.config.color_exposure_us,
                report,
                prefix="color",
                snapshot=self._settings_snapshot,
            )
            _set_sensor_option(
                rs,
                color_sensor,
                "gain",
                self.config.color_gain,
                report,
                prefix="color",
                snapshot=self._settings_snapshot,
            )
            _set_sensor_option(
                rs,
                color_sensor,
                "enable_auto_white_balance",
                self.config.color_auto_white_balance,
                report,
                prefix="color",
                snapshot=self._settings_snapshot,
            )
            _set_sensor_option(
                rs,
                color_sensor,
                "white_balance",
                self.config.color_white_balance,
                report,
                prefix="color",
                snapshot=self._settings_snapshot,
            )
            _set_sensor_option(
                rs,
                color_sensor,
                "focus",
                self.config.color_focus,
                report,
                prefix="color",
                snapshot=self._settings_snapshot,
            )
        else:
            report["color.sensor"] = "not_found"

        depth_sensor = _sensor_for_stream(rs, device, rs.stream.depth)
        if depth_sensor is None:
            try:
                depth_sensor = device.first_depth_sensor()
            except Exception:
                depth_sensor = None
        if depth_sensor is not None:
            _set_sensor_option(
                rs,
                depth_sensor,
                "emitter_enabled",
                None if self.config.depth_emitter_enabled is None else int(bool(self.config.depth_emitter_enabled)),
                report,
                prefix="depth",
                snapshot=self._settings_snapshot,
            )
            _set_sensor_option(
                rs,
                depth_sensor,
                "laser_power",
                self.config.depth_laser_power,
                report,
                prefix="depth",
                snapshot=self._settings_snapshot,
            )
        else:
            report["depth.sensor"] = "not_found"
        return report

    def _restore_device_settings(self) -> dict[str, Any]:
        report: dict[str, Any] = {"restore.enabled": True}
        if not self._settings_snapshot:
            report["restore.count"] = 0
            return report
        restored = 0
        failed = 0
        for item in reversed(self._settings_snapshot):
            restore_key = f"restore.{item.key}"
            try:
                item.sensor.set_option(item.option, float(item.value))
                restored += 1
            except Exception as exc:
                report[restore_key] = f"set_failed: {exc}"
                failed += 1
                continue
            try:
                report[restore_key] = float(item.sensor.get_option(item.option))
            except Exception:
                report[restore_key] = float(item.value)
        report["restore.count"] = restored
        if failed:
            report["restore.failed_count"] = failed
        return report


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
    # A ChArUco board is also an ArUco marker board. When interpolation of
    # inner corners is weak, retain the same fixed board frame by estimating
    # its pose from the detected board markers.
    allow_aruco_board_fallback: bool = True


@dataclass
class TargetDetection:
    T_target_to_camera: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    corner_count: int
    marker_ids: list[int]
    reprojection_rmse_px: float | None = None
    target_center_px: tuple[float, float] | None = None
    image_size_px: tuple[int, int] | None = None
    object_points: np.ndarray | None = None
    image_points: np.ndarray | None = None


class TargetDetector:
    def __init__(self, config: TargetConfig):
        self.config = config
        self._cv2 = _require_cv2()
        self._aruco = _require_aruco(self._cv2)
        self._dictionary = _aruco_dictionary(self._aruco, config.dictionary)
        self._parameters = _detector_parameters(self._aruco)
        self._detector = _make_aruco_detector(self._aruco, self._dictionary, self._parameters)
        self._charuco_board = self._make_charuco_board() if config.target_type == "charuco" else None
        self._last_status = "target: not detected"
        self._last_charuco_corner_count = 0
        self._last_charuco_failure: str | None = None
        self._last_charuco_corners: np.ndarray | None = None
        self._last_charuco_ids: np.ndarray | None = None
        self._last_target_mode = config.target_type
        self._aruco_board = (
            self._make_aruco_board()
            if config.target_type == "charuco" and config.allow_aruco_board_fallback
            else None
        )

    @property
    def last_status(self) -> str:
        """Human-readable diagnostic for the most recently processed frame."""

        return self._last_status

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
        self._last_charuco_corner_count = 0
        self._last_charuco_failure = None
        self._last_charuco_corners = None
        self._last_charuco_ids = None
        self._last_target_mode = self.config.target_type
        corners, ids, _ = self._detect_markers(gray)
        if ids is None or len(corners) == 0:
            self._last_status = "target: not detected (markers=0)"
            if draw:
                _put_status(cv2, image, self._last_status, ok=False)
            return None, image
        marker_count = len(corners)
        if draw:
            aruco.drawDetectedMarkers(image, corners, ids)
        if self.config.target_type == "charuco":
            detection = self._detect_charuco(
                gray,
                corners,
                ids,
                intrinsics_matrix,
                distortion_coeffs,
            )
            if detection is None and self._aruco_board is not None:
                detection = self._detect_aruco_board(
                    corners,
                    ids,
                    intrinsics_matrix,
                    distortion_coeffs,
                    image_shape=gray.shape[:2],
                )
                if detection is not None:
                    self._last_target_mode = "aruco-board"
            if draw and self._last_charuco_corners is not None and self._last_charuco_ids is not None:
                try:
                    aruco.drawDetectedCornersCharuco(
                        image,
                        self._last_charuco_corners,
                        self._last_charuco_ids,
                    )
                except (AttributeError, cv2.error):
                    pass
        else:
            detection = self._detect_aruco(
                corners,
                ids,
                intrinsics_matrix,
                distortion_coeffs,
                image_shape=gray.shape[:2],
            )
        if detection is not None:
            reproj = detection.reprojection_rmse_px
            suffix = "" if reproj is None else f", reproj={reproj:.2f}px"
            if self.config.target_type == "charuco":
                max_corners = max(
                    0,
                    (int(self.config.charuco_squares_x) - 1)
                    * (int(self.config.charuco_squares_y) - 1),
                )
                if self._last_target_mode == "aruco-board":
                    self._last_status = (
                        f"target: ok aruco-board markers={len(detection.marker_ids)} "
                        f"charuco={self._last_charuco_corner_count}/"
                        f"{int(self.config.min_charuco_corners)} "
                        f"(board_max={max_corners}){suffix}"
                    )
                else:
                    self._last_status = (
                        f"target: ok markers={marker_count} "
                        f"charuco={detection.corner_count}/{max_corners}{suffix}"
                    )
            else:
                self._last_status = f"target: ok corners={detection.corner_count}{suffix}"
            if draw:
                self._draw_axes(image, intrinsics_matrix, distortion_coeffs, detection)
                _put_status(cv2, image, self._last_status, ok=True)
        else:
            if self.config.target_type == "aruco":
                self._last_status = f"target: need one marker (markers={marker_count})"
            else:
                reason = self._last_charuco_failure or (
                    f"charuco corners={self._last_charuco_corner_count}/"
                    f"{int(self.config.min_charuco_corners)}"
                )
                self._last_status = f"target: {reason} markers={marker_count}"
            if draw:
                _put_status(cv2, image, self._last_status, ok=False)
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
        image_shape: tuple[int, int],
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
        object_points = _marker_object_points(float(self.config.marker_length_m))
        image_points = selected_corners[0].reshape(4, 2)
        rmse = _marker_reprojection_rmse(
            self._cv2,
            image_points,
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
            target_center_px=_mean_image_point(image_points),
            image_size_px=(int(image_shape[1]), int(image_shape[0])),
            object_points=object_points,
            image_points=image_points,
        )

    def _make_aruco_board(self):
        if self._charuco_board is None or not hasattr(self._aruco, "Board"):
            return None
        try:
            return self._aruco.Board(
                self._charuco_board.getObjPoints(),
                self._dictionary,
                self._charuco_board.getIds(),
            )
        except (AttributeError, TypeError, ValueError):
            return None

    def _detect_aruco_board(
        self,
        corners: Sequence[np.ndarray],
        ids: np.ndarray,
        intrinsics_matrix: np.ndarray,
        distortion_coeffs: np.ndarray,
        image_shape: tuple[int, int],
    ) -> TargetDetection | None:
        """Estimate the fixed ChArUco board frame from its ArUco markers."""

        if self._aruco_board is None:
            return None
        cv2 = self._cv2
        aruco = self._aruco
        board_ids = [
            int(value) for value in np.asarray(self._charuco_board.getIds()).reshape(-1)
        ]
        board_id_to_index = {
            marker_id: index for index, marker_id in enumerate(board_ids)
        }
        detected_ids = [int(value) for value in np.asarray(ids).reshape(-1)]
        valid_pairs = [
            (index, marker_id, board_id_to_index[marker_id])
            for index, marker_id in enumerate(detected_ids)
            if marker_id in board_id_to_index
        ]
        if not valid_pairs:
            return None

        selected_corners = [
            np.asarray(corners[index], dtype=np.float32)
            for index, _, _ in valid_pairs
        ]
        selected_ids = np.asarray(
            [marker_id for _, marker_id, _ in valid_pairs],
            dtype=np.int32,
        ).reshape(-1, 1)
        board_obj_points = self._charuco_board.getObjPoints()
        object_points = np.concatenate(
            [
                np.asarray(board_obj_points[board_index], dtype=np.float64)
                for _, _, board_index in valid_pairs
            ],
            axis=0,
        ).reshape(-1, 3)
        image_points = np.concatenate(
            [corner.reshape(4, 2) for corner in selected_corners],
            axis=0,
        ).reshape(-1, 2)
        try:
            if hasattr(aruco, "estimatePoseBoard"):
                try:
                    retval, rvec, tvec = aruco.estimatePoseBoard(
                        selected_corners,
                        selected_ids,
                        self._aruco_board,
                        intrinsics_matrix,
                        distortion_coeffs,
                        None,
                        None,
                    )
                except TypeError:
                    rvec = np.zeros((3, 1), dtype=np.float64)
                    tvec = np.zeros((3, 1), dtype=np.float64)
                    retval, rvec, tvec = aruco.estimatePoseBoard(
                        selected_corners,
                        selected_ids,
                        self._aruco_board,
                        intrinsics_matrix,
                        distortion_coeffs,
                        rvec,
                        tvec,
                    )
            else:
                retval, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    intrinsics_matrix,
                    distortion_coeffs,
                )
        except cv2.error:
            return None
        if not bool(retval):
            return None

        rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        rotation, _ = cv2.Rodrigues(rvec)
        rmse = _board_reprojection_rmse(
            cv2,
            object_points,
            image_points,
            rvec,
            tvec,
            intrinsics_matrix,
            distortion_coeffs,
        )
        return TargetDetection(
            T_target_to_camera=make_transform(rotation, tvec),
            rvec=rvec,
            tvec=tvec,
            corner_count=int(image_points.shape[0]),
            marker_ids=[marker_id for _, marker_id, _ in valid_pairs],
            reprojection_rmse_px=rmse,
            target_center_px=_mean_image_point(image_points),
            image_size_px=(int(image_shape[1]), int(image_shape[0])),
            object_points=object_points,
            image_points=image_points,
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
            self._last_charuco_failure = "charuco board unavailable"
            return None
        aruco = self._aruco
        cv2 = self._cv2
        try:
            if hasattr(aruco, "interpolateCornersCharuco"):
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
            elif hasattr(aruco, "CharucoDetector"):
                charuco_detector = aruco.CharucoDetector(self._charuco_board)
                charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(
                    gray,
                    None,
                    None,
                    corners,
                    ids,
                )
                count = 0 if charuco_ids is None else len(charuco_ids)
            else:
                self._last_charuco_failure = "charuco interpolation API unavailable"
                return None
        except cv2.error as exc:
            self._last_charuco_failure = (
                f"charuco interpolation failed ({type(exc).__name__})"
            )
            return None
        try:
            count_int = int(count)
        except (TypeError, ValueError):
            count_int = 0
        self._last_charuco_corner_count = count_int
        self._last_charuco_corners = charuco_corners
        self._last_charuco_ids = charuco_ids
        if (
            charuco_ids is None
            or charuco_corners is None
            or count_int < int(self.config.min_charuco_corners)
        ):
            self._last_charuco_failure = (
                f"charuco corners={count_int}/{int(self.config.min_charuco_corners)}"
            )
            return None
        try:
            if hasattr(aruco, "estimatePoseCharucoBoard"):
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
            else:
                object_points = _charuco_object_points(self._charuco_board, charuco_ids)
                if object_points is None:
                    self._last_charuco_failure = "charuco object points unavailable"
                    return None
                ok, rvec, tvec = cv2.solvePnP(
                    np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
                    np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2),
                    intrinsics_matrix,
                    distortion_coeffs,
                )
        except cv2.error as exc:
            self._last_charuco_failure = (
                f"charuco corners={count_int}/{int(self.config.min_charuco_corners)}, "
                f"pose failed ({type(exc).__name__})"
            )
            return None
        if not ok:
            self._last_charuco_failure = (
                f"charuco corners={count_int}/{int(self.config.min_charuco_corners)}, "
                "pose failed"
            )
            return None
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        rotation, _ = cv2.Rodrigues(rvec)
        object_points = _charuco_object_points(self._charuco_board, charuco_ids)
        image_points = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
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
            corner_count=count_int,
            marker_ids=[int(value) for value in np.asarray(ids).reshape(-1)],
            reprojection_rmse_px=rmse,
            target_center_px=_mean_image_point(image_points),
            image_size_px=(int(gray.shape[1]), int(gray.shape[0])),
            object_points=object_points,
            image_points=image_points,
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
    rs = _require_pyrealsense2()
    device = _device_from_serial(rs, serial)
    return _available_video_profiles(
        rs,
        device,
        profile_type=VideoStreamProfileInfo,
        stream_names=_PROFILE_STREAM_NAMES,
        format_names=_PROFILE_FORMAT_NAMES,
    )


def _stop_pipeline_quietly(pipeline: Any | None) -> None:
    if pipeline is None:
        return
    try:
        pipeline.stop()
    except Exception:
        pass


def _is_startup_connection_error(message: str) -> bool:
    lowered = str(message).lower()
    if "couldn't resolve requests" in lowered or "could not resolve requests" in lowered:
        return False
    return any(
        token in lowered
        for token in (
            "startup warmup received no frames",
            "frame didn't arrive",
            "frame did not arrive",
            "device or resource busy",
            "no device connected",
            "failed to set power state",
            "failed to open",
            "device disconnected",
            "uvc",
            "usb",
            "backend",
            "ioctl",
            "resource busy",
        )
    )


def _select_video_profile(
    rs: Any,
    device: Any,
    *,
    stream: Any,
    preferred_formats: Sequence[Any],
    requested_width: int,
    requested_height: int,
    requested_fps: int,
) -> VideoStreamProfileInfo | None:
    stream_name = _stream_name(rs, stream)
    format_names = {_format_name(rs, item) for item in preferred_formats}
    candidates = [
        item
        for item in available_video_profiles(_safe_device_info(rs, device, rs.camera_info.serial_number))
        if item.stream == stream_name and item.format_name in format_names
    ]
    if not candidates:
        return None

    format_rank = {
        _format_name(rs, value): index for index, value in enumerate(preferred_formats)
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


def _stream_name(rs: Any, stream: Any) -> str:
    return _enum_name(rs.stream, stream, _PROFILE_STREAM_NAMES)


def _format_name(rs: Any, fmt: Any) -> str:
    return _enum_name(rs.format, fmt, _PROFILE_FORMAT_NAMES)


def _serial_from_index(index: int) -> str:
    devices = RealSenseCamera.list_devices()
    if 0 <= int(index) < len(devices):
        return devices[int(index)].serial
    if 0 <= int(index) < len(DEFAULT_CAMERA_SERIALS):
        return DEFAULT_CAMERA_SERIALS[int(index)]
    raise ValueError(f"camera index {index} is not available.")


def _sensor_for_stream(rs: Any, device: Any, stream: Any):
    try:
        for sensor in device.sensors:
            for profile in sensor.get_stream_profiles():
                try:
                    if profile.stream_type() == stream:
                        return sensor
                except Exception:
                    continue
    except Exception:
        return None
    return None


def _set_sensor_option(
    rs: Any,
    sensor: Any,
    option_name: str,
    value: float | int | bool | None,
    report: dict[str, Any],
    *,
    prefix: str,
    snapshot: list[_SensorOptionSnapshot] | None = None,
) -> None:
    key = f"{prefix}.{option_name}"
    option = getattr(rs.option, option_name, None)
    if option is None:
        report[key] = "option_missing"
        return
    try:
        supported = bool(sensor.supports(option))
    except Exception as exc:
        report[key] = f"support_check_failed: {exc}"
        return
    if not supported:
        report[key] = "unsupported"
        return
    if value is not None:
        try:
            previous = float(sensor.get_option(option))
        except Exception as exc:
            previous = None
            report[f"{key}.previous"] = f"read_failed: {exc}"
        else:
            report[f"{key}.previous"] = previous
            if snapshot is not None:
                snapshot.append(
                    _SensorOptionSnapshot(
                        sensor=sensor,
                        option=option,
                        key=key,
                        value=previous,
                    )
                )
        try:
            sensor.set_option(option, float(value))
        except Exception as exc:
            report[key] = f"set_failed: {exc}"
            return
    try:
        report[key] = float(sensor.get_option(option))
    except Exception:
        report[key] = float(value) if value is not None else "read_failed"


def _marker_object_points(marker_length_m: float) -> np.ndarray:
    half = float(marker_length_m) * 0.5
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


def _mean_image_point(image_points: np.ndarray) -> tuple[float, float]:
    center = np.mean(np.asarray(image_points, dtype=np.float64).reshape(-1, 2), axis=0)
    return float(center[0]), float(center[1])


def _marker_reprojection_rmse(
    cv2: Any,
    image_corners: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    marker_length_m: float,
    intrinsics_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
) -> float:
    object_points = _marker_object_points(marker_length_m)
    projected, _ = cv2.projectPoints(
        object_points,
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        intrinsics_matrix,
        distortion_coeffs,
    )
    diff = projected.reshape(-1, 2) - np.asarray(image_corners, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

def _charuco_object_points(board: Any, charuco_ids: np.ndarray) -> np.ndarray | None:
    object_corners = None
    if hasattr(board, "getChessboardCorners"):
        object_corners = board.getChessboardCorners()
    elif hasattr(board, "chessboardCorners"):
        object_corners = board.chessboardCorners
    if object_corners is None:
        return None
    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    return np.asarray(object_corners, dtype=np.float64)[ids]


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
    object_points = _charuco_object_points(board, charuco_ids)
    if object_points is None:
        return None
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


def _board_reprojection_rmse(
    cv2: Any,
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    intrinsics_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        intrinsics_matrix,
        distortion_coeffs,
    )
    diff = projected.reshape(-1, 2) - np.asarray(
        image_points, dtype=np.float64
    ).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
