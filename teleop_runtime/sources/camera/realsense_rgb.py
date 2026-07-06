from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ...interfaces import ObservationSource
from ...recording import _to_serializable
from ...registry import register_source
from ...schemas import ObservationChannel
from ...collection.video_frame import NearestFrameBuffer, VideoFrame


def _load_pyrealsense2():
    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RealSenseRGBSource requires pyrealsense2 to be importable."
        ) from exc
    return rs


@dataclass
class RealSenseRGBCameraConfig:
    serial: str
    width: int = 1280
    height: int = 720
    fps: int = 30
    camera_key: str = "front"
    name: str | None = None
    keep_last: int = 2
    color_space: str = "rgb"
    wait_timeout_ms: int = 1000
    stale_after_sec: float = 2.0
    restart_on_stale_timeout: bool = True
    restart_after_timeouts: int = 3
    restart_cooldown_sec: float = 0.5
    auto_exposure_priority: int | None = None
    enable_depth: bool = False
    depth_width: int | None = None
    depth_height: int | None = None
    depth_fps: int = 15
    align_depth_to_color: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.serial = str(self.serial)
        self.camera_key = str(self.camera_key)
        if self.name is None:
            self.name = f"realsense_rgb_{self.camera_key}"
        self.name = str(self.name)
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("RealSense RGB resolution must be positive.")
        if int(self.fps) <= 0:
            raise ValueError("RealSense RGB fps must be positive.")
        if int(self.keep_last) <= 0:
            raise ValueError("keep_last must be positive.")
        if int(self.wait_timeout_ms) <= 0:
            raise ValueError("wait_timeout_ms must be positive.")
        if float(self.stale_after_sec) <= 0.0:
            raise ValueError("stale_after_sec must be positive.")
        if int(self.restart_after_timeouts) <= 0:
            raise ValueError("restart_after_timeouts must be positive.")
        if float(self.restart_cooldown_sec) < 0.0:
            raise ValueError("restart_cooldown_sec must be >= 0.")
        if self.depth_width is None:
            self.depth_width = int(self.width)
        if self.depth_height is None:
            self.depth_height = int(self.height)
        if int(self.depth_width) <= 0 or int(self.depth_height) <= 0:
            raise ValueError("RealSense depth resolution must be positive.")
        if int(self.depth_fps) <= 0:
            raise ValueError("depth_fps must be positive.")
        self.color_space = str(self.color_space).lower()
        if self.color_space not in {"rgb", "bgr"}:
            raise ValueError("color_space must be either 'rgb' or 'bgr'.")

    def to_dict(self) -> dict[str, Any]:
        return _to_serializable(
            {
                "serial": self.serial,
                "width": int(self.width),
                "height": int(self.height),
                "fps": int(self.fps),
                "camera_key": self.camera_key,
                "name": self.name,
                "keep_last": int(self.keep_last),
                "color_space": self.color_space,
                "wait_timeout_ms": int(self.wait_timeout_ms),
                "stale_after_sec": float(self.stale_after_sec),
                "restart_on_stale_timeout": bool(self.restart_on_stale_timeout),
                "restart_after_timeouts": int(self.restart_after_timeouts),
                "restart_cooldown_sec": float(self.restart_cooldown_sec),
                "auto_exposure_priority": self.auto_exposure_priority,
                "enable_depth": bool(self.enable_depth),
                "depth_width": int(self.depth_width),
                "depth_height": int(self.depth_height),
                "depth_fps": int(self.depth_fps),
                "align_depth_to_color": bool(self.align_depth_to_color),
                "metadata": dict(self.metadata),
            }
        )


def load_realsense_camera_config_map(
    path: str | Path,
) -> dict[str, RealSenseRGBCameraConfig]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("RealSense config JSON must be an object keyed by camera name.")
    configs: dict[str, RealSenseRGBCameraConfig] = {}
    for camera_key, raw_config in payload.items():
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"Camera config for '{camera_key}' must be an object.")
        config_payload = dict(raw_config)
        config_payload.setdefault("camera_key", str(camera_key))
        config_payload.setdefault("name", f"realsense_rgb_{camera_key}")
        configs[str(camera_key)] = RealSenseRGBCameraConfig(**config_payload)
    return configs


@register_source("realsense_rgb")
class RealSenseRGBSource(ObservationSource):
    def __init__(
        self,
        config: RealSenseRGBCameraConfig,
    ):
        self.config = config
        self._frame_buffer = NearestFrameBuffer(keep_last=self.config.keep_last)
        self._pipeline = None
        self._profile = None
        self._pipeline_lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.RLock()
        self._latest_frame: VideoFrame | None = None
        self._frame_counter = 0
        self._last_error: str | None = None
        self._last_runtime_error: str | None = None
        self._started_at: float | None = None
        self._last_frame_receive_timestamp: float | None = None
        self._timeout_count = 0
        self._consecutive_timeout_count = 0
        self._restart_count = 0
        self._last_restart_timestamp: float | None = None
        self._device_info: dict[str, Any] = {}
        self._depth_align = None
        self._depth_scale_m_per_unit: float | None = None

    @property
    def name(self) -> str:
        return self.config.name or f"realsense_rgb_{self.config.camera_key}"

    @property
    def camera_key(self) -> str:
        return self.config.camera_key

    def expected_channel_names(self) -> tuple[str, ...]:
        return ()

    def get_latest_channels(self) -> dict[str, ObservationChannel]:
        return {}

    def start(self) -> None:
        if self._running:
            return
        rs = _load_pyrealsense2()
        with self._lock:
            self._last_error = None
            self._last_runtime_error = None
        pipeline, profile = self._start_pipeline(rs)
        with self._pipeline_lock:
            self._pipeline = pipeline
            self._profile = profile
        with self._lock:
            self._device_info = self._device_info_from_profile(profile)
            self._started_at = time.time()
            self._running = True
            self._last_runtime_error = None
            self._last_frame_receive_timestamp = None
            self._consecutive_timeout_count = 0
            self._frame_counter = 0
            self._latest_frame = None
            self._timeout_count = 0
            self._restart_count = 0
            self._last_restart_timestamp = None
        self._frame_buffer.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"{self.name}_capture",
            daemon=True,
        )
        self._thread.start()

    def _start_pipeline(self, rs: Any) -> tuple[Any, Any]:
        pipeline = rs.pipeline()
        stream_config = rs.config()
        stream_config.enable_device(self.config.serial)
        stream_config.enable_stream(
            rs.stream.color,
            int(self.config.width),
            int(self.config.height),
            rs.format.bgr8,
            int(self.config.fps),
        )
        if bool(self.config.enable_depth):
            stream_config.enable_stream(
                rs.stream.depth,
                int(self.config.depth_width),
                int(self.config.depth_height),
                rs.format.z16,
                int(self.config.depth_fps),
            )
        profile = pipeline.start(stream_config)
        self._configure_sensor_options(profile)
        self._configure_depth_state(rs, profile)
        return pipeline, profile

    def _configure_depth_state(self, rs: Any, profile: Any) -> None:
        if not bool(self.config.enable_depth):
            self._depth_align = None
            self._depth_scale_m_per_unit = None
            return
        self._depth_align = rs.align(rs.stream.color) if bool(self.config.align_depth_to_color) else None
        try:
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale_m_per_unit = float(depth_sensor.get_depth_scale())
        except Exception as exc:  # noqa: BLE001
            self._depth_scale_m_per_unit = None
            self._last_error = f"depth_scale_failed:{type(exc).__name__}:{exc}"

    def _configure_sensor_options(self, profile: Any) -> None:
        rs = _load_pyrealsense2()
        option_value = self.config.auto_exposure_priority
        if option_value is None:
            return
        try:
            device = profile.get_device()
            for sensor in device.query_sensors():
                if not sensor.supports(rs.option.auto_exposure_priority):
                    continue
                sensor.set_option(rs.option.auto_exposure_priority, float(option_value))
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"set_option_failed:{type(exc).__name__}:{exc}"

    def _device_info_from_profile(self, profile: Any) -> dict[str, Any]:
        rs = _load_pyrealsense2()
        device = profile.get_device()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        info: dict[str, Any] = {
            "camera_key": self.config.camera_key,
            "serial": self.config.serial,
            "name": self.name,
            "width": int(intr.width),
            "height": int(intr.height),
            "fps": int(self.config.fps),
            "color_space": self.config.color_space,
            "wait_timeout_ms": int(self.config.wait_timeout_ms),
            "stale_after_sec": float(self.config.stale_after_sec),
            "restart_on_stale_timeout": bool(self.config.restart_on_stale_timeout),
            "restart_after_timeouts": int(self.config.restart_after_timeouts),
            "restart_cooldown_sec": float(self.config.restart_cooldown_sec),
            "depth_enabled": bool(self.config.enable_depth),
            "depth_width": int(self.config.depth_width),
            "depth_height": int(self.config.depth_height),
            "depth_fps": int(self.config.depth_fps),
            "align_depth_to_color": bool(self.config.align_depth_to_color),
            "depth_scale_m_per_unit": self._depth_scale_m_per_unit,
            "intrinsics": {
                "width": int(intr.width),
                "height": int(intr.height),
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "ppx": float(intr.ppx),
                "ppy": float(intr.ppy),
                "coeffs": [float(value) for value in intr.coeffs],
            },
            "metadata": dict(self.config.metadata),
        }
        if bool(self.config.enable_depth):
            try:
                depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
                depth_intr = depth_profile.get_intrinsics()
                info["depth_intrinsics"] = {
                    "width": int(depth_intr.width),
                    "height": int(depth_intr.height),
                    "fx": float(depth_intr.fx),
                    "fy": float(depth_intr.fy),
                    "ppx": float(depth_intr.ppx),
                    "ppy": float(depth_intr.ppy),
                    "coeffs": [float(value) for value in depth_intr.coeffs],
                }
            except Exception as exc:  # noqa: BLE001
                info["depth_intrinsics_error"] = f"{type(exc).__name__}: {exc}"
        for key, info_name in {
            "device_name": rs.camera_info.name,
            "product_line": rs.camera_info.product_line,
            "serial_number": rs.camera_info.serial_number,
            "firmware_version": rs.camera_info.firmware_version,
        }.items():
            try:
                info[key] = device.get_info(info_name)
            except Exception:  # noqa: BLE001
                info[key] = None
        return info

    @staticmethod
    def _is_frame_timeout_error(message: str) -> bool:
        lowered = str(message).lower()
        return "frame didn't arrive" in lowered or "frame did not arrive" in lowered

    def _set_transient_or_stale_timeout_error(self, message: str) -> bool:
        now = time.time()
        stale_after_sec = float(self.config.stale_after_sec)
        with self._lock:
            self._timeout_count += 1
            self._consecutive_timeout_count += 1
            self._last_runtime_error = str(message)
            last_frame_at = self._last_frame_receive_timestamp
            started_at = self._started_at
            if last_frame_at is not None:
                is_stale = now - float(last_frame_at) >= stale_after_sec
            elif started_at is not None:
                is_stale = now - float(started_at) >= stale_after_sec
            else:
                is_stale = True
            if is_stale:
                self._last_error = f"frame_timeout_stale: {message}"
            else:
                self._last_error = None
            should_restart = (
                bool(self.config.restart_on_stale_timeout)
                and is_stale
                and self._consecutive_timeout_count >= int(self.config.restart_after_timeouts)
            )
        return should_restart

    def _restart_pipeline_after_stale_timeout(self, message: str) -> None:
        if not self._running:
            return
        rs = _load_pyrealsense2()
        with self._lock:
            self._last_error = f"restarting_after_frame_timeout: {message}"
            self._last_runtime_error = str(message)
        with self._pipeline_lock:
            old_pipeline = self._pipeline
            self._pipeline = None
            self._profile = None
        self._frame_buffer.clear()
        with self._lock:
            self._latest_frame = None
        if old_pipeline is not None:
            try:
                old_pipeline.stop()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._last_runtime_error = f"pipeline_stop_failed:{type(exc).__name__}:{exc}"
        cooldown_sec = float(self.config.restart_cooldown_sec)
        if cooldown_sec > 0.0:
            time.sleep(cooldown_sec)
        if not self._running:
            return
        try:
            pipeline, profile = self._start_pipeline(rs)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._last_error = f"pipeline_restart_failed:{type(exc).__name__}:{exc}"
                self._last_runtime_error = self._last_error
            return
        with self._pipeline_lock:
            if not self._running:
                try:
                    pipeline.stop()
                except Exception:
                    pass
                return
            self._pipeline = pipeline
            self._profile = profile
        with self._lock:
            self._device_info = self._device_info_from_profile(profile)
            self._started_at = time.time()
            self._last_error = None
            self._last_runtime_error = None
            self._last_frame_receive_timestamp = None
            self._consecutive_timeout_count = 0
            self._restart_count += 1
            self._last_restart_timestamp = time.time()

    def _capture_loop(self) -> None:
        rs = _load_pyrealsense2()
        while self._running:
            try:
                with self._pipeline_lock:
                    pipeline = self._pipeline
                if pipeline is None:
                    self._restart_pipeline_after_stale_timeout("pipeline_missing")
                    continue
                frames = pipeline.wait_for_frames(timeout_ms=int(self.config.wait_timeout_ms))
                if bool(self.config.enable_depth) and self._depth_align is not None:
                    frames = self._depth_align.process(frames)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                depth_frame = frames.get_depth_frame() if bool(self.config.enable_depth) else None
                if bool(self.config.enable_depth) and not depth_frame:
                    continue
                received_at = time.time()
                frame_index = int(self._frame_counter)
                self._frame_counter += 1
                image = np.asanyarray(color_frame.get_data()).copy()
                if self.config.color_space == "rgb":
                    image = image[..., ::-1]
                depth_image = None
                depth_metadata: dict[str, Any] = {}
                if depth_frame is not None:
                    depth_image = np.asanyarray(depth_frame.get_data()).copy()
                    depth_timestamp_sec = float(depth_frame.get_timestamp()) / 1000.0
                    try:
                        depth_sensor_timestamp_ms = float(
                            depth_frame.get_frame_metadata(rs.frame_metadata_value.sensor_timestamp)
                        )
                    except Exception:  # noqa: BLE001
                        depth_sensor_timestamp_ms = None
                    try:
                        depth_frame_number = int(depth_frame.get_frame_number())
                    except Exception:  # noqa: BLE001
                        depth_frame_number = None
                    depth_metadata = {
                        "enabled": True,
                        "aligned_to": "color" if bool(self.config.align_depth_to_color) else "none",
                        "width": int(depth_image.shape[1]),
                        "height": int(depth_image.shape[0]),
                        "fps": int(self.config.depth_fps),
                        "format": "z16",
                        "depth_scale_m_per_unit": self._depth_scale_m_per_unit,
                        "camera_timestamp_sec": depth_timestamp_sec,
                        "sensor_timestamp_ms": depth_sensor_timestamp_ms,
                        "frame_number": depth_frame_number,
                    }
                frame_timestamp_sec = float(color_frame.get_timestamp()) / 1000.0
                try:
                    timestamp_domain = str(color_frame.get_frame_timestamp_domain())
                except Exception:  # noqa: BLE001
                    timestamp_domain = "unknown"
                try:
                    sensor_timestamp_ms = float(
                        color_frame.get_frame_metadata(rs.frame_metadata_value.sensor_timestamp)
                    )
                except Exception:  # noqa: BLE001
                    sensor_timestamp_ms = None
                try:
                    color_frame_number = int(color_frame.get_frame_number())
                except Exception:  # noqa: BLE001
                    color_frame_number = None
                frame = VideoFrame(
                    image=np.ascontiguousarray(image),
                    source_timestamp=received_at,
                    receive_timestamp=received_at,
                    frame_index=frame_index,
                    metadata={
                        "camera_key": self.config.camera_key,
                        "serial": self.config.serial,
                        "width": int(self.config.width),
                        "height": int(self.config.height),
                        "fps": int(self.config.fps),
                        "color_space": self.config.color_space,
                        "camera_timestamp_sec": frame_timestamp_sec,
                        "timestamp_domain": timestamp_domain,
                        "sensor_timestamp_ms": sensor_timestamp_ms,
                        "color_frame_number": color_frame_number,
                    },
                    depth_image=None if depth_image is None else np.ascontiguousarray(depth_image),
                    depth_metadata=depth_metadata,
                )
                with self._lock:
                    self._latest_frame = frame
                    self._last_error = None
                    self._last_runtime_error = None
                    self._last_frame_receive_timestamp = received_at
                    self._consecutive_timeout_count = 0
                self._frame_buffer.push(frame)
            except RuntimeError as exc:
                if not self._running:
                    break
                message = str(exc)
                if self._is_frame_timeout_error(message):
                    if self._set_transient_or_stale_timeout_error(message):
                        self._restart_pipeline_after_stale_timeout(message)
                else:
                    with self._lock:
                        self._last_runtime_error = message
                        self._last_error = message
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._last_runtime_error = f"{type(exc).__name__}: {exc}"
                    self._last_error = f"{type(exc).__name__}: {exc}"

    def get_latest_frame(self) -> VideoFrame | None:
        with self._lock:
            if self._latest_frame is None:
                return None
            frame = self._latest_frame
        return VideoFrame(
            image=np.asarray(frame.image).copy(),
            source_timestamp=float(frame.source_timestamp),
            receive_timestamp=frame.receive_timestamp,
            frame_index=frame.frame_index,
            metadata=dict(frame.metadata),
            depth_image=None if frame.depth_image is None else np.asarray(frame.depth_image).copy(),
            depth_metadata=dict(frame.depth_metadata),
        )

    def frame_buffer(self) -> NearestFrameBuffer:
        return self._frame_buffer

    def device_info(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            info = dict(self._device_info)
            info["last_error"] = self._last_error
            info["last_runtime_error"] = self._last_runtime_error
            info["frame_count"] = int(self._frame_counter)
            info["timeout_count"] = int(self._timeout_count)
            info["consecutive_timeout_count"] = int(self._consecutive_timeout_count)
            info["restart_count"] = int(self._restart_count)
            info["last_restart_timestamp"] = self._last_restart_timestamp
            info["last_frame_receive_timestamp"] = self._last_frame_receive_timestamp
            info["last_frame_age_sec"] = (
                None
                if self._last_frame_receive_timestamp is None
                else float(now - float(self._last_frame_receive_timestamp))
            )
            info["stale_after_sec"] = float(self.config.stale_after_sec)
        return info

    def close(self) -> None:
        self._running = False
        with self._pipeline_lock:
            pipeline = self._pipeline
            self._pipeline = None
            self._profile = None
            self._depth_align = None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._last_runtime_error = f"pipeline_stop_failed:{type(exc).__name__}:{exc}"
                    self._last_error = self._last_runtime_error
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
