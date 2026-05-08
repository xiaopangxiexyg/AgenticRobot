from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..common import to_serializable
from ..video_frame import NearestFrameBuffer, VideoFrame


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
    start_retry_attempts: int = 8
    start_retry_delay_sec: float = 0.5
    auto_exposure_priority: int | None = None
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
        if int(self.start_retry_attempts) <= 0:
            raise ValueError("start_retry_attempts must be positive.")
        if float(self.start_retry_delay_sec) < 0.0:
            raise ValueError("start_retry_delay_sec must be >= 0.")
        self.color_space = str(self.color_space).lower()
        if self.color_space not in {"rgb", "bgr"}:
            raise ValueError("color_space must be either 'rgb' or 'bgr'.")

    def to_dict(self) -> dict[str, Any]:
        return dict(
            to_serializable(
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
                    "start_retry_attempts": int(self.start_retry_attempts),
                    "start_retry_delay_sec": float(self.start_retry_delay_sec),
                    "auto_exposure_priority": self.auto_exposure_priority,
                    "metadata": dict(self.metadata),
                }
            )
        )


def load_realsense_camera_config_map(
    path: str | Path,
) -> dict[str, RealSenseRGBCameraConfig]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("RealSense config JSON must be an object keyed by camera key.")
    configs: dict[str, RealSenseRGBCameraConfig] = {}
    for camera_key, raw_config in payload.items():
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"Camera config for '{camera_key}' must be an object.")
        config_payload = dict(raw_config)
        config_payload.setdefault("camera_key", str(camera_key))
        config_payload.setdefault("name", f"realsense_rgb_{camera_key}")
        configs[str(camera_key)] = RealSenseRGBCameraConfig(**config_payload)
    return configs


class RealSenseRGBSource:
    def __init__(self, config: RealSenseRGBCameraConfig):
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

    @property
    def name(self) -> str:
        return self.config.name or f"realsense_rgb_{self.config.camera_key}"

    @property
    def camera_key(self) -> str:
        return self.config.camera_key

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
            self._consecutive_timeout_count = 0
            self._last_frame_receive_timestamp = None
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
        last_error: RuntimeError | None = None
        attempts = max(1, int(self.config.start_retry_attempts))
        for attempt in range(1, attempts + 1):
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
            try:
                profile = pipeline.start(stream_config)
                self._configure_sensor_options(profile)
                return pipeline, profile
            except RuntimeError as exc:
                last_error = exc
                try:
                    pipeline.stop()
                except Exception:
                    pass
                if not self._is_device_busy_error(str(exc)) or attempt >= attempts:
                    raise
                delay = float(self.config.start_retry_delay_sec)
                with self._lock:
                    self._last_runtime_error = str(exc)
                    self._last_error = (
                        f"pipeline_start_busy_retry:{attempt}/{attempts}: {exc}"
                    )
                if delay > 0.0:
                    time.sleep(delay)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _is_device_busy_error(message: str) -> bool:
        lowered = str(message).lower()
        return (
            "device or resource busy" in lowered
            or "errno=16" in lowered
            or "vidioc_s_fmt" in lowered
        )

    def _configure_sensor_options(self, profile: Any) -> None:
        rs = _load_pyrealsense2()
        option_value = self.config.auto_exposure_priority
        if option_value is None:
            return
        try:
            device = profile.get_device()
            for sensor in device.query_sensors():
                if sensor.supports(rs.option.auto_exposure_priority):
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
        actual_serial = str(info.get("serial_number") or "").strip()
        requested_serial = str(self.config.serial or "").strip()
        info["requested_serial"] = requested_serial
        info["serial_match"] = (actual_serial == requested_serial) if requested_serial else None
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
                frames = pipeline.wait_for_frames(
                    timeout_ms=int(self.config.wait_timeout_ms)
                )
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                received_at = time.time()
                frame_index = int(self._frame_counter)
                self._frame_counter += 1
                image = np.asanyarray(color_frame.get_data()).copy()
                if self.config.color_space == "rgb":
                    image = image[..., ::-1]
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
                    },
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
