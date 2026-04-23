from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .common import to_serializable
from .video_frame import NearestFrameBuffer, VideoFrame


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
    wait_timeout_ms: int = 200
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
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.RLock()
        self._latest_frame: VideoFrame | None = None
        self._frame_counter = 0
        self._last_error: str | None = None
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
        profile = pipeline.start(stream_config)
        self._configure_sensor_options(profile)
        self._pipeline = pipeline
        self._profile = profile
        self._device_info = self._device_info_from_profile(profile)
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"{self.name}_capture",
            daemon=True,
        )
        self._thread.start()

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
        return info

    def _capture_loop(self) -> None:
        assert self._pipeline is not None
        rs = _load_pyrealsense2()
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(
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
                self._frame_buffer.push(frame)
            except RuntimeError as exc:
                if not self._running:
                    break
                with self._lock:
                    self._last_error = str(exc)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
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
        with self._lock:
            info = dict(self._device_info)
            info["last_error"] = self._last_error
            info["frame_count"] = int(self._frame_counter)
        return info

    def close(self) -> None:
        self._running = False
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None
                self._profile = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
