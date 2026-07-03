from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

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
class VideoStreamProfileInfo:
    stream: str
    format_name: str
    width: int
    height: int
    fps: int

    def describe(self) -> str:
        return f"{self.stream} {self.width}x{self.height}@{self.fps} {self.format_name}"


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

        rs = self._rs
        pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(str(serial))
        width, height = self.config.resolution
        if self.config.enable_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, int(self.config.fps))
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, int(self.config.fps))
        try:
            profile = pipeline.start(config)
            selected_profiles = {
                "requested": _requested_profile_text(
                    width,
                    height,
                    int(self.config.fps),
                    enable_depth=self.config.enable_depth,
                ),
                "fallback_used": False,
            }
        except Exception as exc:
            if not self.config.auto_select_supported_profile:
                raise RuntimeError(self._profile_error_message(exc, serial)) from exc
            fallback_config, selected_profiles = self._fallback_stream_config(serial)
            try:
                profile = pipeline.start(fallback_config)
            except Exception as fallback_exc:
                raise RuntimeError(self._profile_error_message(fallback_exc, serial)) from fallback_exc
        self._pipeline = pipeline
        self._profile = profile
        self._align = rs.align(rs.stream.color) if self.config.enable_depth else None
        self.settings_report = self._apply_device_settings(profile)
        self.settings_report.update(selected_profiles)
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
        return self

    def _fallback_stream_config(self, serial: str | None):
        rs = self._rs
        width, height = self.config.resolution
        fps = int(self.config.fps)
        device = _device_from_serial(rs, serial)
        color_profile = _select_video_profile(
            rs,
            device,
            stream=rs.stream.color,
            preferred_formats=[rs.format.bgr8],
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
        return config, report

    def _profile_error_message(self, exc: object, serial: str | None) -> str:
        width, height = self.config.resolution
        lines = [
            "RealSense could not start the requested stream profile.",
            f"requested camera={serial or 'auto'} color={width}x{height}@{int(self.config.fps)} bgr8 depth={bool(self.config.enable_depth)}",
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
        lines.append("Try --resolution 640 480 --fps 30, or run --list-cameras --show-profiles.")
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
        color_image = np.asanyarray(color_frame.get_data())
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
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._align = None
        self._profile = None

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
            )
            _set_sensor_option(
                rs,
                color_sensor,
                "exposure",
                self.config.color_exposure_us,
                report,
                prefix="color",
            )
            _set_sensor_option(
                rs,
                color_sensor,
                "gain",
                self.config.color_gain,
                report,
                prefix="color",
            )
            _set_sensor_option(
                rs,
                color_sensor,
                "enable_auto_white_balance",
                self.config.color_auto_white_balance,
                report,
                prefix="color",
            )
            _set_sensor_option(
                rs,
                color_sensor,
                "white_balance",
                self.config.color_white_balance,
                report,
                prefix="color",
            )
            _set_sensor_option(
                rs,
                color_sensor,
                "focus",
                self.config.color_focus,
                report,
                prefix="color",
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
            )
            _set_sensor_option(
                rs,
                depth_sensor,
                "laser_power",
                self.config.depth_laser_power,
                report,
                prefix="depth",
            )
        else:
            report["depth.sensor"] = "not_found"
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
            detection = self._detect_charuco(
                gray,
                corners,
                ids,
                intrinsics_matrix,
                distortion_coeffs,
            )
        else:
            detection = self._detect_aruco(
                corners,
                ids,
                intrinsics_matrix,
                distortion_coeffs,
                image_shape=gray.shape[:2],
            )
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
            corner_count=int(count),
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


def _require_pyrealsense2():
    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on calibration env
        raise ImportError(
            "pyrealsense2 is required for RealSense capture. "
            "Install it in the calibration environment or use --load-samples."
        ) from exc
    return rs


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on calibration env
        raise ImportError(
            "OpenCV is required for target detection. Install opencv-contrib-python."
        ) from exc
    return cv2


def _require_aruco(cv2: Any):
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise ImportError("cv2.aruco is missing; install opencv-contrib-python.")
    return aruco


def available_video_profiles(serial: str | None = None) -> list[VideoStreamProfileInfo]:
    rs = _require_pyrealsense2()
    device = _device_from_serial(rs, serial)
    profiles: list[VideoStreamProfileInfo] = []
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
            info = _video_profile_info(rs, profile)
            if info is not None:
                profiles.append(info)
    profiles.sort(key=lambda item: (item.stream, item.width, item.height, item.fps, item.format_name))
    return profiles


def _device_from_serial(rs: Any, serial: str | None):
    ctx = rs.context()
    devices = list(ctx.devices)
    if not devices:
        raise RuntimeError("no RealSense devices detected.")
    if serial:
        for device in devices:
            if _safe_device_info(rs, device, rs.camera_info.serial_number) == str(serial):
                return device
        available = ", ".join(
            _safe_device_info(rs, device, rs.camera_info.serial_number) for device in devices
        )
        raise ValueError(f"RealSense serial {serial!r} not found; available: {available}")
    return devices[0]


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

    requested_area = int(requested_width) * int(requested_height)

    def score(item: VideoStreamProfileInfo) -> tuple[int, int, int, int, int]:
        exact_resolution = 0 if (item.width, item.height) == (requested_width, requested_height) else 1
        fps_penalty = abs(item.fps - int(requested_fps))
        area_penalty = abs(item.width * item.height - requested_area)
        width_penalty = abs(item.width - int(requested_width))
        height_penalty = abs(item.height - int(requested_height))
        return exact_resolution, fps_penalty, area_penalty, width_penalty, height_penalty

    return min(candidates, key=score)


def _video_profile_info(rs: Any, profile: Any) -> VideoStreamProfileInfo | None:
    try:
        video = profile.as_video_stream_profile()
        stream = _stream_name(rs, profile.stream_type())
        format_name = _format_name(rs, profile.format())
        return VideoStreamProfileInfo(
            stream=stream,
            format_name=format_name,
            width=int(video.width()),
            height=int(video.height()),
            fps=int(profile.fps()),
        )
    except Exception:
        return None


def _stream_name(rs: Any, stream: Any) -> str:
    for name in ("color", "depth"):
        if getattr(rs.stream, name, None) == stream:
            return name
    return str(stream).split(".")[-1]


def _format_name(rs: Any, fmt: Any) -> str:
    for name in ("bgr8", "rgb8", "z16", "y8", "yuyv", "rgba8", "bgra8"):
        if getattr(rs.format, name, None) == fmt:
            return name
    return str(fmt).split(".")[-1]


def _requested_profile_text(width: int, height: int, fps: int, *, enable_depth: bool) -> str:
    streams = [f"color {int(width)}x{int(height)}@{int(fps)} bgr8"]
    if enable_depth:
        streams.append(f"depth {int(width)}x{int(height)}@{int(fps)} z16")
    return ", ".join(streams)


def _serial_from_index(index: int) -> str:
    devices = RealSenseCamera.list_devices()
    if 0 <= int(index) < len(devices):
        return devices[int(index)].serial
    if 0 <= int(index) < len(DEFAULT_CAMERA_SERIALS):
        return DEFAULT_CAMERA_SERIALS[int(index)]
    raise ValueError(f"camera index {index} is not available.")


def _safe_device_info(rs: Any, device: Any, field: Any) -> str:
    try:
        if device.supports(field):
            return str(device.get_info(field))
    except Exception:
        pass
    return ""


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
            sensor.set_option(option, float(value))
        except Exception as exc:
            report[key] = f"set_failed: {exc}"
            return
    try:
        report[key] = float(sensor.get_option(option))
    except Exception:
        report[key] = float(value) if value is not None else "read_failed"


def _aruco_dictionary(aruco: Any, dictionary: str):
    name = str(dictionary)
    if hasattr(aruco, name):
        return aruco.getPredefinedDictionary(getattr(aruco, name))
    try:
        return aruco.getPredefinedDictionary(int(name))
    except Exception as exc:
        raise ValueError(f"unknown ArUco dictionary {dictionary!r}.") from exc


def _detector_parameters(aruco: Any):
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


def _make_aruco_detector(aruco: Any, dictionary: Any, parameters: Any):
    if hasattr(aruco, "ArucoDetector"):
        return aruco.ArucoDetector(dictionary, parameters)
    return None


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


def _put_status(cv2: Any, image: np.ndarray, text: str, *, ok: bool) -> None:
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
