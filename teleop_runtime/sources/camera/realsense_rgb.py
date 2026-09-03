from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from hardware.realsense.color import (
    color_format_value as _color_format_value,
    color_image_to_bgr as _color_image_to_bgr,
    normalize_color_format as _normalize_color_format,
)

from robot_motion.runtime.lifecycle import LifecycleFailure, raise_lifecycle_failures

from ...interfaces import ObservationSource
from ...recording import _to_serializable
from ...registry import register_source
from ...schemas import ObservationChannel
from ...video_frame import NearestFrameBuffer, VideoFrame


REALSENSE_STARTUP_WARMUP_LEGACY = "legacy"
REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME = "bounded_first_frame"
REALSENSE_STARTUP_WARMUP_POLICIES = (
    REALSENSE_STARTUP_WARMUP_LEGACY,
    REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
)

# librealsense pipeline start/stop/reset operations share native process state.
# Serialize only those native lifecycle edges across source instances; frame
# capture remains one worker per camera.
_REALSENSE_SDK_LIFECYCLE_LOCK = threading.RLock()

_LIFECYCLE_STOPPED = "stopped"
_LIFECYCLE_STARTING = "starting"
_LIFECYCLE_RUNNING = "running"
_LIFECYCLE_STOPPING = "stopping"
_LIFECYCLE_FAULTED = "faulted"


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
    # Native transport format. Public frames keep the configured RGB/BGR space.
    color_format: str = "yuyv"
    camera_key: str = "front"
    name: str | None = None
    keep_last: int = 2
    color_space: str = "rgb"
    # Keep the runtime wait short enough for freshness reporting, but give a
    # newly negotiated multi-camera pipeline its own longer first-frame
    # budget.  RealSense may take several seconds to produce the first frame
    # when another camera is already streaming.
    wait_timeout_ms: int = 1000
    startup_wait_timeout_ms: int = 5000
    startup_frame_grace_sec: float = 5.0
    stale_after_sec: float = 2.0
    restart_on_stale_timeout: bool = True
    restart_on_runtime_error: bool = True
    restart_after_timeouts: int = 3
    restart_on_black_frames: bool = True
    restart_after_black_frames: int = 5
    black_frame_max_value: int = 1
    # Some reviewed sensors intentionally return a near-zero color stream while
    # still providing valid depth. Keep the historical drop/restart behavior by
    # default and allow such frames only in an explicit deployment profile.
    accept_near_zero_frames: bool = False
    # Default-off compatibility controls. Formal bounded-first-frame profiles
    # enable them explicitly; the legacy profile preserves the historical
    # any-frameset warmup and unbounded runtime retry behavior.
    require_complete_frameset: bool = False
    restart_on_incomplete_frameset: bool = False
    restart_after_incomplete_framesets: int = 5
    # Zero preserves the historical unbounded retry behavior.
    restart_max_attempts_without_complete_frame: int = 0
    restart_cooldown_sec: float = 0.5
    startup_reset: bool = True
    startup_reset_sleep_sec: float = 0.2
    startup_warmup_frames: int = 10
    # ``legacy`` preserves the historical N-frame synchronous warmup and its
    # hardware-reset retry. ``bounded_first_frame`` waits only for the first
    # valid frameset within startup_frame_grace_sec.  Multi-camera callers own
    # any deferred second pass so a failed pipeline is not reopened
    # immediately while the remaining cameras have not yet been initialized;
    # the existing global camera self-check remains responsible for proving
    # multi-frame freshness.
    startup_warmup_policy: str = REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME
    # Default-off, process-local retry. This only rebuilds the SDK pipeline; it
    # never resets or re-enumerates hardware. The wait remains cancellable so a
    # shutdown request cannot be held behind a retry sleep.
    startup_software_retry_attempts: int = 0
    startup_software_retry_sleep_sec: float = 0.5
    startup_hardware_reset: bool = False
    startup_hardware_reset_on_failure: bool = True
    startup_hardware_reset_retry_attempts: int = 1
    startup_hardware_reset_sleep_sec: float = 2.0
    startup_hardware_reset_reconnect_timeout_sec: float = 6.0
    startup_hardware_reset_reconnect_poll_sec: float = 0.25
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
        self.color_format = _normalize_color_format(self.color_format)
        if int(self.keep_last) <= 0:
            raise ValueError("keep_last must be positive.")
        if int(self.wait_timeout_ms) <= 0:
            raise ValueError("wait_timeout_ms must be positive.")
        if int(self.startup_wait_timeout_ms) <= 0:
            raise ValueError("startup_wait_timeout_ms must be positive.")
        if float(self.startup_frame_grace_sec) < 0.0:
            raise ValueError("startup_frame_grace_sec must be >= 0.")
        if float(self.stale_after_sec) <= 0.0:
            raise ValueError("stale_after_sec must be positive.")
        if int(self.restart_after_timeouts) <= 0:
            raise ValueError("restart_after_timeouts must be positive.")
        if int(self.restart_after_black_frames) <= 0:
            raise ValueError("restart_after_black_frames must be positive.")
        if int(self.restart_after_incomplete_framesets) <= 0:
            raise ValueError("restart_after_incomplete_framesets must be positive.")
        if int(self.restart_max_attempts_without_complete_frame) < 0:
            raise ValueError(
                "restart_max_attempts_without_complete_frame must be >= 0."
            )
        if not 0 <= int(self.black_frame_max_value) <= 255:
            raise ValueError("black_frame_max_value must be in [0, 255].")
        if float(self.restart_cooldown_sec) < 0.0:
            raise ValueError("restart_cooldown_sec must be >= 0.")
        if float(self.startup_reset_sleep_sec) < 0.0:
            raise ValueError("startup_reset_sleep_sec must be >= 0.")
        if int(self.startup_warmup_frames) < 0:
            raise ValueError("startup_warmup_frames must be >= 0.")
        self.startup_warmup_policy = str(self.startup_warmup_policy).strip().lower()
        if self.startup_warmup_policy not in REALSENSE_STARTUP_WARMUP_POLICIES:
            raise ValueError(
                "startup_warmup_policy must be one of "
                f"{REALSENSE_STARTUP_WARMUP_POLICIES}; got "
                f"{self.startup_warmup_policy!r}."
            )
        if int(self.startup_software_retry_attempts) < 0:
            raise ValueError("startup_software_retry_attempts must be >= 0.")
        if float(self.startup_software_retry_sleep_sec) < 0.0:
            raise ValueError("startup_software_retry_sleep_sec must be >= 0.")
        if int(self.startup_hardware_reset_retry_attempts) < 0:
            raise ValueError("startup_hardware_reset_retry_attempts must be >= 0.")
        if float(self.startup_hardware_reset_sleep_sec) < 0.0:
            raise ValueError("startup_hardware_reset_sleep_sec must be >= 0.")
        if float(self.startup_hardware_reset_reconnect_timeout_sec) < 0.0:
            raise ValueError("startup_hardware_reset_reconnect_timeout_sec must be >= 0.")
        if float(self.startup_hardware_reset_reconnect_poll_sec) <= 0.0:
            raise ValueError("startup_hardware_reset_reconnect_poll_sec must be positive.")
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
                "color_format": self.color_format,
                "camera_key": self.camera_key,
                "name": self.name,
                "keep_last": int(self.keep_last),
                "color_space": self.color_space,
                "wait_timeout_ms": int(self.wait_timeout_ms),
                "startup_wait_timeout_ms": int(self.startup_wait_timeout_ms),
                "startup_frame_grace_sec": float(self.startup_frame_grace_sec),
                "stale_after_sec": float(self.stale_after_sec),
                "restart_on_stale_timeout": bool(self.restart_on_stale_timeout),
                "restart_on_runtime_error": bool(self.restart_on_runtime_error),
                "restart_after_timeouts": int(self.restart_after_timeouts),
                "restart_on_black_frames": bool(self.restart_on_black_frames),
                "restart_after_black_frames": int(self.restart_after_black_frames),
                "black_frame_max_value": int(self.black_frame_max_value),
                "accept_near_zero_frames": bool(self.accept_near_zero_frames),
                "require_complete_frameset": bool(self.require_complete_frameset),
                "restart_on_incomplete_frameset": bool(
                    self.restart_on_incomplete_frameset
                ),
                "restart_after_incomplete_framesets": int(
                    self.restart_after_incomplete_framesets
                ),
                "restart_max_attempts_without_complete_frame": int(
                    self.restart_max_attempts_without_complete_frame
                ),
                "restart_cooldown_sec": float(self.restart_cooldown_sec),
                "startup_reset": bool(self.startup_reset),
                "startup_reset_sleep_sec": float(self.startup_reset_sleep_sec),
                "startup_warmup_frames": int(self.startup_warmup_frames),
                "startup_warmup_policy": self.startup_warmup_policy,
                "startup_software_retry_attempts": int(
                    self.startup_software_retry_attempts
                ),
                "startup_software_retry_sleep_sec": float(
                    self.startup_software_retry_sleep_sec
                ),
                "startup_hardware_reset": bool(self.startup_hardware_reset),
                "startup_hardware_reset_on_failure": bool(self.startup_hardware_reset_on_failure),
                "startup_hardware_reset_retry_attempts": int(self.startup_hardware_reset_retry_attempts),
                "startup_hardware_reset_sleep_sec": float(self.startup_hardware_reset_sleep_sec),
                "startup_hardware_reset_reconnect_timeout_sec": float(
                    self.startup_hardware_reset_reconnect_timeout_sec
                ),
                "startup_hardware_reset_reconnect_poll_sec": float(
                    self.startup_hardware_reset_reconnect_poll_sec
                ),
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
        self._lifecycle_lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._lifecycle_state = _LIFECYCLE_STOPPED
        self._generation = 0
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()
        self._latest_frame: VideoFrame | None = None
        self._frame_counter = 0
        self._last_error: str | None = None
        self._last_runtime_error: str | None = None
        self._started_at: float | None = None
        self._last_frame_receive_timestamp: float | None = None
        self._last_frameset_receive_timestamp: float | None = None
        self._last_color_receive_timestamp: float | None = None
        self._last_depth_receive_timestamp: float | None = None
        self._incomplete_frameset_count = 0
        self._consecutive_incomplete_frameset_count = 0
        self._missing_color_frame_count = 0
        self._missing_depth_frame_count = 0
        self._timeout_count = 0
        self._consecutive_timeout_count = 0
        self._black_frame_count = 0
        self._consecutive_black_frame_count = 0
        self._last_sampled_max_value: int | None = None
        self._restart_count = 0
        self._restart_attempt_count = 0
        self._restart_failure_count = 0
        self._last_restart_timestamp: float | None = None
        self._restart_attempts_without_complete_frame = 0
        self._device_info: dict[str, Any] = {}
        self._startup_timing: dict[str, float] = {}
        self._depth_align = None
        self._depth_scale_m_per_unit: float | None = None
        self._active_color_format = self.config.color_format

    @property
    def name(self) -> str:
        return self.config.name or f"realsense_rgb_{self.config.camera_key}"

    @property
    def camera_key(self) -> str:
        return self.config.camera_key

    @property
    def startup_timing(self) -> dict[str, float]:
        """Return the most recent native startup stage timings in seconds."""

        with self._lock:
            return dict(self._startup_timing)

    def mark_startup_timeout(self, timeout_sec: float) -> None:
        """Fault a startup worker without waiting for a native SDK call.

        ``pipeline.start`` can enter an uninterruptible kernel USB wait.  The
        caller that owns the batch startup therefore needs a non-blocking
        containment seam: request shutdown, invalidate the startup generation,
        and leave native cleanup to the worker if/when the SDK call returns.
        This method deliberately does not acquire ``_lifecycle_lock`` or call
        into librealsense.
        """

        normalized_timeout_sec = float(timeout_sec)
        if not math.isfinite(normalized_timeout_sec) or normalized_timeout_sec < 0.0:
            raise ValueError("timeout_sec must be finite and >= 0.")
        message = (
            "startup_operation_timeout:"
            f">{normalized_timeout_sec:.3f}s;"
            "native_sdk_worker_not_joined"
        )
        self._mark_faulted(message)

    def _mark_faulted(self, message: str) -> None:
        with self._lock:
            self._running = False
            self._lifecycle_state = _LIFECYCLE_FAULTED
            self._generation += 1
            if self._last_error is None:
                self._last_error = str(message)
            self._last_runtime_error = str(message)
            shutdown_event = self._shutdown_event
        shutdown_event.set()

    def expected_channel_names(self) -> tuple[str, ...]:
        return ()

    def get_latest_channels(self) -> dict[str, ObservationChannel]:
        return {}

    def start(self, *, startup_software_retry_attempts: int | None = None) -> None:
        with self._lifecycle_lock:
            with self._lock:
                if self._running and self._lifecycle_state == _LIFECYCLE_RUNNING:
                    return
                worker = self._thread
                if worker is not None and worker.is_alive():
                    raise RuntimeError(
                        f"{self.name} cannot start while prior worker "
                        f"{worker.name!r} is still alive"
                    )
                if self._lifecycle_state == _LIFECYCLE_FAULTED:
                    raise RuntimeError(
                        f"{self.name} cannot start from faulted state; "
                        "call close() again after the prior worker exits or rebuild the source"
                    )
                self._lifecycle_state = _LIFECYCLE_STARTING
                self._generation += 1
                generation = int(self._generation)
                shutdown_event = threading.Event()
                self._shutdown_event = shutdown_event
                self._last_error = None
                self._last_runtime_error = None
            pipeline = None
            try:
                rs = _load_pyrealsense2()
                with self._pipeline_lock:
                    leftover_pipeline = self._pipeline
                    self._pipeline = None
                    self._profile = None
                    self._depth_align = None
                with _REALSENSE_SDK_LIFECYCLE_LOCK:
                    self._stop_pipeline_quietly(leftover_pipeline)
                    if startup_software_retry_attempts is None:
                        pipeline, profile = self._start_pipeline(rs)
                    else:
                        pipeline, profile = self._start_pipeline(
                            rs,
                            software_retry_attempts=startup_software_retry_attempts,
                        )
                if not self._is_startup_generation_active(
                    generation,
                    shutdown_event,
                ):
                    self._stop_pipeline_quietly(pipeline)
                    pipeline = None
                    raise RuntimeError("startup_cancelled:shutdown_requested")
                device_info = self._device_info_from_profile(profile)
                with self._pipeline_lock:
                    self._pipeline = pipeline
                    self._profile = profile
                with self._lock:
                    self._device_info = device_info
                    self._started_at = time.time()
                    self._running = True
                    self._lifecycle_state = _LIFECYCLE_RUNNING
                    self._last_runtime_error = None
                    self._last_frame_receive_timestamp = None
                    self._last_frameset_receive_timestamp = None
                    self._last_color_receive_timestamp = None
                    self._last_depth_receive_timestamp = None
                    self._incomplete_frameset_count = 0
                    self._consecutive_incomplete_frameset_count = 0
                    self._missing_color_frame_count = 0
                    self._missing_depth_frame_count = 0
                    self._consecutive_timeout_count = 0
                    self._frame_counter = 0
                    self._latest_frame = None
                    self._timeout_count = 0
                    self._black_frame_count = 0
                    self._consecutive_black_frame_count = 0
                    self._last_sampled_max_value = None
                    self._restart_count = 0
                    self._restart_attempt_count = 0
                    self._restart_failure_count = 0
                    self._last_restart_timestamp = None
                    self._restart_attempts_without_complete_frame = 0
                self._frame_buffer.clear()
                worker = threading.Thread(
                    target=self._capture_loop,
                    args=(generation, shutdown_event),
                    name=f"{self.name}_capture",
                    daemon=True,
                )
                self._thread = worker
                worker.start()
            except BaseException:
                shutdown_event.set()
                with self._lock:
                    self._running = False
                    if self._lifecycle_state != _LIFECYCLE_FAULTED:
                        self._lifecycle_state = _LIFECYCLE_STOPPED
                with self._pipeline_lock:
                    installed_pipeline = self._pipeline
                    self._pipeline = None
                    self._profile = None
                    self._depth_align = None
                cleanup_pipeline = (
                    installed_pipeline if installed_pipeline is not None else pipeline
                )
                with _REALSENSE_SDK_LIFECYCLE_LOCK:
                    self._stop_pipeline_quietly(cleanup_pipeline)
                self._thread = None
                raise

    def _is_generation_active(
        self,
        generation: int,
        shutdown_event: threading.Event,
    ) -> bool:
        if shutdown_event.is_set():
            return False
        with self._lock:
            return bool(
                self._running
                and self._lifecycle_state == _LIFECYCLE_RUNNING
                and int(generation) == int(self._generation)
            )

    def _is_startup_generation_active(
        self,
        generation: int,
        shutdown_event: threading.Event,
    ) -> bool:
        if shutdown_event.is_set():
            return False
        with self._lock:
            return bool(
                self._lifecycle_state == _LIFECYCLE_STARTING
                and int(generation) == int(self._generation)
            )

    def _start_pipeline(
        self,
        rs: Any,
        *,
        software_retry_attempts: int | None = None,
    ) -> tuple[Any, Any]:
        self._hardware_reset_device_if_requested(rs, reason="configured_startup_reset")
        if software_retry_attempts is None:
            software_retries_remaining = int(
                self.config.startup_software_retry_attempts
            )
        else:
            software_retries_remaining = int(software_retry_attempts)
            if software_retries_remaining < 0:
                raise ValueError("software_retry_attempts must be >= 0.")
        hardware_reset_retries_remaining = (
            int(self.config.startup_hardware_reset_retry_attempts)
            if bool(self.config.startup_hardware_reset_on_failure)
            else 0
        )
        attempt_index = 0
        while True:
            if self._shutdown_event.is_set():
                raise RuntimeError("startup_cancelled:shutdown_requested")
            attempt_index += 1
            try:
                return self._start_pipeline_once(rs)
            except RuntimeError as exc:
                message = str(exc)
                if (
                    software_retries_remaining > 0
                    and self._is_startup_connection_error(message)
                ):
                    software_retries_remaining -= 1
                    with self._lock:
                        self._last_runtime_error = (
                            "startup_software_retry:"
                            f"attempt={attempt_index}:{exc}"
                        )
                    retry_sleep_sec = float(
                        self.config.startup_software_retry_sleep_sec
                    )
                    if retry_sleep_sec > 0.0 and self._shutdown_event.wait(
                        retry_sleep_sec
                    ):
                        raise RuntimeError(
                            "startup_cancelled:shutdown_requested"
                        ) from exc
                    continue
                if (
                    hardware_reset_retries_remaining <= 0
                    or not self._should_hardware_reset_retry_startup_error(message)
                ):
                    raise
                hardware_reset_retries_remaining -= 1
                with self._lock:
                    self._last_runtime_error = (
                        "startup_connection_error_reset_retry:"
                        f"attempt={attempt_index}:{exc}"
                    )
                self._hardware_reset_device(
                    rs,
                    reason="startup_connection_error",
                    force=True,
                )

    def _start_pipeline_once(self, rs: Any) -> tuple[Any, Any]:
        startup_started = time.perf_counter()
        pipeline = rs.pipeline()
        stream_config = rs.config()
        stream_config.enable_device(self.config.serial)
        color_format = _color_format_value(rs, self.config.color_format)
        stream_config.enable_stream(
            rs.stream.color,
            int(self.config.width),
            int(self.config.height),
            color_format,
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
        if bool(self.config.startup_reset):
            self._stop_pipeline_quietly(pipeline)
            reset_sleep_sec = float(self.config.startup_reset_sleep_sec)
            if reset_sleep_sec > 0.0:
                time.sleep(reset_sleep_sec)
        try:
            pipeline_start_started = time.perf_counter()
            profile = pipeline.start(stream_config)
            pipeline_start_elapsed = time.perf_counter() - pipeline_start_started
        except RuntimeError as exc:
            self._stop_pipeline_quietly(pipeline)
            request = self._requested_stream_profile_text()
            raise RuntimeError(
                f"RealSense pipeline start failed for camera_key={self.config.camera_key!r} "
                f"serial={self.config.serial!r} request={request}: {exc}. "
                "The configured profile was preserved. Verify the device's enumerated native "
                "profiles, USB link, firmware, and exclusive access; do not silently lower "
                "collection resolution to bypass a device/profile failure."
            ) from exc
        self._active_color_format = self.config.color_format
        try:
            configure_started = time.perf_counter()
            self._configure_sensor_options(profile)
            self._configure_depth_state(rs, profile)
            configure_elapsed = time.perf_counter() - configure_started
            warmup_started = time.perf_counter()
            self._warmup_pipeline(pipeline)
            warmup_elapsed = time.perf_counter() - warmup_started
        except RuntimeError:
            self._stop_pipeline_quietly(pipeline)
            raise
        with self._lock:
            self._startup_timing = {
                "pipeline_start_sec": float(pipeline_start_elapsed),
                "sensor_configure_sec": float(configure_elapsed),
                "warmup_sec": float(warmup_elapsed),
                "total_sec": float(time.perf_counter() - startup_started),
                "warmup_frames": float(self.config.startup_warmup_frames),
            }
        return pipeline, profile

    def _requested_stream_profile_text(self) -> str:
        color = (
            f"color={int(self.config.width)}x{int(self.config.height)}"
            f"@{int(self.config.fps)} {self.config.color_format}"
        )
        if not bool(self.config.enable_depth):
            return color
        depth = (
            f"depth={int(self.config.depth_width)}x{int(self.config.depth_height)}"
            f"@{int(self.config.depth_fps)} z16"
        )
        return f"{color}, {depth}, align_depth_to_color={bool(self.config.align_depth_to_color)}"

    @staticmethod
    def _stop_pipeline_quietly(pipeline: Any | None) -> None:
        if pipeline is None:
            return
        try:
            pipeline.stop()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _stop_pipeline_bounded(
        pipeline: Any | None,
        *,
        timeout_sec: float,
    ) -> BaseException | None:
        """Attempt native stop without allowing teardown to block forever.

        The stop worker is daemonized because Python cannot interrupt a native
        SDK call that is already blocked in the kernel.  A timeout is reported
        as a lifecycle failure and the source is left faulted; a later explicit
        rebuild can then deal with the still-running native worker.
        """

        if pipeline is None:
            return None
        normalized_timeout_sec = float(timeout_sec)
        if not math.isfinite(normalized_timeout_sec) or normalized_timeout_sec < 0.0:
            raise ValueError("timeout_sec must be finite and >= 0.")
        errors: list[BaseException] = []

        def stop_native() -> None:
            try:
                with _REALSENSE_SDK_LIFECYCLE_LOCK:
                    pipeline.stop()
            except BaseException as exc:  # noqa: BLE001 - preserve teardown evidence.
                errors.append(exc)

        worker = threading.Thread(
            target=stop_native,
            name="realsense_pipeline_stop",
            daemon=True,
        )
        try:
            worker.start()
            worker.join(timeout=normalized_timeout_sec)
        except BaseException as exc:  # noqa: BLE001 - report startup teardown failure.
            return exc
        if worker.is_alive():
            return TimeoutError(
                "native RealSense pipeline.stop did not finish within "
                f"{normalized_timeout_sec:.3f}s"
            )
        return errors[0] if errors else None

    def _warmup_pipeline(self, pipeline: Any) -> None:
        warmup_frames = int(self.config.startup_warmup_frames)
        if warmup_frames <= 0:
            return
        if (
            self.config.startup_warmup_policy
            == REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME
        ):
            self._warmup_pipeline_bounded_first_frame(
                pipeline,
                warmup_frames=warmup_frames,
            )
            return
        timeout_ms = int(self.config.startup_wait_timeout_ms)
        success_count = 0
        failure_count = 0
        last_error: str | None = None
        for _ in range(warmup_frames):
            if self._shutdown_event.is_set():
                raise RuntimeError("startup_cancelled:shutdown_requested")
            try:
                frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
                if bool(self.config.require_complete_frameset):
                    color_frame, depth_frame = self._required_stream_frames(frames)
                    if not self._required_streams_complete(color_frame, depth_frame):
                        raise RuntimeError(
                            self._incomplete_frameset_message(
                                color_frame,
                                depth_frame,
                            )
                        )
                success_count += 1
            except Exception as exc:  # noqa: BLE001
                failure_count += 1
                last_error = f"{type(exc).__name__}: {exc}"
                if (
                    success_count <= 0
                    and failure_count
                    >= min(warmup_frames, int(self.config.restart_after_timeouts))
                ):
                    break
        if success_count <= 0 and failure_count > 0:
            raise RuntimeError(
                "startup_warmup_failed:"
                f"no_frames_after={warmup_frames}:startup_timeout_ms={timeout_ms}:"
                f"last_error={last_error}"
            )

    def _warmup_pipeline_bounded_first_frame(
        self,
        pipeline: Any,
        *,
        warmup_frames: int,
    ) -> None:
        """Wait for one native frame under a single bounded startup budget.

        This is intentionally not the final camera admission check. Real
        teleop runs its existing cross-camera, consecutive-new-frame self-check
        after every pipeline has been started. Keeping this wait local and
        bounded avoids multiplying a 5-second SDK timeout by ten warmup frames.
        """

        grace_sec = float(self.config.startup_frame_grace_sec)
        deadline = time.monotonic() + grace_sec
        configured_timeout_ms = int(self.config.startup_wait_timeout_ms)
        # Keep each SDK call short even when startup_wait_timeout_ms equals the
        # entire grace period (the default is 5000/5000). This preserves the
        # same total deadline while allowing late frames and cancellation
        # checks instead of one opaque 4999 ms wait.
        attempt_timeout_ms = min(
            configured_timeout_ms,
            int(self.config.wait_timeout_ms),
        )
        failure_count = 0
        last_error: str | None = None
        # ``warmup_frames`` belongs to the legacy N-frame policy.  It must not
        # truncate this policy's time budget: stations commonly configure only
        # two legacy warmup frames while using a 1 s SDK wait and a 5 s bounded
        # first-frame grace period.  The previous implementation therefore
        # reported a 5 s budget after actually making only two 1 s attempts.
        # Bound the number of calls from the grace/timeout pair so mocked or
        # immediately failing SDK calls cannot spin, while a real blocking SDK
        # receives the complete configured deadline.
        del warmup_frames
        attempt_budget = max(
            1,
            int(
                math.ceil(
                    (grace_sec * 1000.0) / max(1, attempt_timeout_ms)
                )
            ),
        )
        for _ in range(attempt_budget):
            if self._shutdown_event.is_set():
                raise RuntimeError("startup_cancelled:shutdown_requested")
            remaining_sec = deadline - time.monotonic()
            if remaining_sec <= 0.0:
                break
            timeout_ms = max(
                1,
                min(attempt_timeout_ms, int(remaining_sec * 1000.0)),
            )
            try:
                frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
                if not bool(self.config.require_complete_frameset):
                    return
                color_frame, depth_frame = self._required_stream_frames(frames)
                if self._required_streams_complete(color_frame, depth_frame):
                    return
                failure_count += 1
                last_error = self._incomplete_frameset_message(
                    color_frame,
                    depth_frame,
                )
            except Exception as exc:  # noqa: BLE001
                failure_count += 1
                last_error = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(
            "startup_warmup_failed:"
            "policy=bounded_first_frame:"
            f"no_frames_within_sec={grace_sec:.3f}:"
            f"attempts={failure_count}:last_error={last_error}"
        )

    def _required_stream_frames(self, frames: Any) -> tuple[Any | None, Any | None]:
        if bool(self.config.enable_depth) and self._depth_align is not None:
            frames = self._depth_align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = (
            frames.get_depth_frame() if bool(self.config.enable_depth) else None
        )
        return color_frame, depth_frame

    def _required_streams_complete(
        self,
        color_frame: Any | None,
        depth_frame: Any | None,
    ) -> bool:
        return bool(color_frame) and (
            not bool(self.config.enable_depth) or bool(depth_frame)
        )

    def _incomplete_frameset_message(
        self,
        color_frame: Any | None,
        depth_frame: Any | None,
    ) -> str:
        return (
            "incomplete_frameset:"
            f"missing_color={not bool(color_frame)}:"
            f"missing_depth={bool(self.config.enable_depth) and not bool(depth_frame)}"
        )

    def _record_incomplete_frameset(
        self,
        color_frame: Any | None,
        depth_frame: Any | None,
        *,
        received_at: float,
    ) -> bool:
        missing_color = not bool(color_frame)
        missing_depth = bool(self.config.enable_depth) and not bool(depth_frame)
        message = self._incomplete_frameset_message(color_frame, depth_frame)
        with self._lock:
            self._last_frameset_receive_timestamp = float(received_at)
            if not missing_color:
                self._last_color_receive_timestamp = float(received_at)
            if bool(self.config.enable_depth) and not missing_depth:
                self._last_depth_receive_timestamp = float(received_at)
            self._incomplete_frameset_count += 1
            self._consecutive_incomplete_frameset_count += 1
            if missing_color:
                self._missing_color_frame_count += 1
            if missing_depth:
                self._missing_depth_frame_count += 1
            streak = int(self._consecutive_incomplete_frameset_count)
            self._last_runtime_error = message
            should_restart = bool(
                self.config.restart_on_incomplete_frameset
                and streak >= int(self.config.restart_after_incomplete_framesets)
            )
            self._last_error = (
                (
                    f"{message}:streak={streak}/"
                    f"{int(self.config.restart_after_incomplete_framesets)}"
                )
                if should_restart
                else None
            )
        return should_restart

    def _record_complete_frameset(self, *, received_at: float) -> None:
        with self._lock:
            self._last_frameset_receive_timestamp = float(received_at)
            self._last_color_receive_timestamp = float(received_at)
            if bool(self.config.enable_depth):
                self._last_depth_receive_timestamp = float(received_at)
            self._consecutive_incomplete_frameset_count = 0

    def _hardware_reset_device_if_requested(self, rs: Any, *, reason: str) -> None:
        if not bool(self.config.startup_hardware_reset):
            return
        self._hardware_reset_device(rs, reason=reason, force=True)

    def _hardware_reset_device(self, rs: Any, *, reason: str, force: bool = False) -> bool:
        if not force and not bool(self.config.startup_hardware_reset):
            return False
        try:
            target_device = self._find_device_by_serial(rs)
            if target_device is None:
                with self._lock:
                    self._last_runtime_error = (
                        f"{reason}_hardware_reset_device_not_found:{self.config.serial}"
                    )
                return False
            target_device.hardware_reset()
            reset_sleep_sec = float(self.config.startup_hardware_reset_sleep_sec)
            if reset_sleep_sec > 0.0 and self._shutdown_event.wait(reset_sleep_sec):
                return False
            reconnected = self._wait_for_device_reconnect(rs)
            if not reconnected:
                with self._lock:
                    self._last_runtime_error = (
                        f"{reason}_hardware_reset_reconnect_timeout:{self.config.serial}"
                    )
                return False
            return True
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._last_runtime_error = (
                    f"{reason}_hardware_reset_failed:{type(exc).__name__}:{exc}"
                )
            return False

    def _find_device_by_serial(self, rs: Any) -> Any | None:
        context = rs.context()
        for device in context.query_devices():
            try:
                serial = str(device.get_info(rs.camera_info.serial_number))
            except Exception:  # noqa: BLE001
                continue
            if serial == str(self.config.serial):
                return device
        return None

    def _wait_for_device_reconnect(self, rs: Any) -> bool:
        timeout_sec = float(self.config.startup_hardware_reset_reconnect_timeout_sec)
        deadline = time.time() + timeout_sec
        while True:
            if self._find_device_by_serial(rs) is not None:
                return True
            if time.time() >= deadline or self._shutdown_event.is_set():
                return False
            if self._shutdown_event.wait(
                float(self.config.startup_hardware_reset_reconnect_poll_sec)
            ):
                return False

    @staticmethod
    def _is_startup_connection_error(message: str) -> bool:
        lowered = str(message).lower()
        if "couldn't resolve requests" in lowered or "could not resolve requests" in lowered:
            return False
        return any(
            token in lowered
            for token in (
                "startup_warmup_failed",
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

    def _should_hardware_reset_retry_startup_error(self, message: str) -> bool:
        # A missing first frame is deliberately fail-fast in the bounded
        # bounded policy. Hardware reset adds reset/re-enumeration latency and
        # has historically destabilized sibling cameras. Explicit device,
        # USB, UVC, or exclusive-access failures still use the legacy retry.
        if (
            self.config.startup_warmup_policy
            == REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME
            and "startup_warmup_failed" in str(message).lower()
        ):
            return False
        return self._is_startup_connection_error(message)

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
        option_value = self.config.auto_exposure_priority
        if option_value is None:
            return
        rs = _load_pyrealsense2()
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
            "color_format": self.config.color_format,
            "wait_timeout_ms": int(self.config.wait_timeout_ms),
            "startup_wait_timeout_ms": int(self.config.startup_wait_timeout_ms),
            "startup_frame_grace_sec": float(self.config.startup_frame_grace_sec),
            "stale_after_sec": float(self.config.stale_after_sec),
            "restart_on_stale_timeout": bool(self.config.restart_on_stale_timeout),
            "restart_on_runtime_error": bool(self.config.restart_on_runtime_error),
            "restart_after_timeouts": int(self.config.restart_after_timeouts),
            "restart_on_black_frames": bool(self.config.restart_on_black_frames),
            "restart_after_black_frames": int(self.config.restart_after_black_frames),
            "black_frame_max_value": int(self.config.black_frame_max_value),
            "accept_near_zero_frames": bool(self.config.accept_near_zero_frames),
            "require_complete_frameset": bool(self.config.require_complete_frameset),
            "restart_on_incomplete_frameset": bool(
                self.config.restart_on_incomplete_frameset
            ),
            "restart_after_incomplete_framesets": int(
                self.config.restart_after_incomplete_framesets
            ),
            "restart_max_attempts_without_complete_frame": int(
                self.config.restart_max_attempts_without_complete_frame
            ),
            "restart_cooldown_sec": float(self.config.restart_cooldown_sec),
            "startup_reset": bool(self.config.startup_reset),
            "startup_reset_sleep_sec": float(self.config.startup_reset_sleep_sec),
            "startup_warmup_frames": int(self.config.startup_warmup_frames),
            "startup_warmup_policy": self.config.startup_warmup_policy,
            "startup_software_retry_attempts": int(
                self.config.startup_software_retry_attempts
            ),
            "startup_software_retry_sleep_sec": float(
                self.config.startup_software_retry_sleep_sec
            ),
            "startup_hardware_reset": bool(self.config.startup_hardware_reset),
            "startup_hardware_reset_on_failure": bool(self.config.startup_hardware_reset_on_failure),
            "startup_hardware_reset_retry_attempts": int(self.config.startup_hardware_reset_retry_attempts),
            "startup_hardware_reset_sleep_sec": float(self.config.startup_hardware_reset_sleep_sec),
            "startup_hardware_reset_reconnect_timeout_sec": float(
                self.config.startup_hardware_reset_reconnect_timeout_sec
            ),
            "startup_hardware_reset_reconnect_poll_sec": float(
                self.config.startup_hardware_reset_reconnect_poll_sec
            ),
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

    @staticmethod
    def _is_recoverable_runtime_error(message: str) -> bool:
        """Return whether rebuilding the SDK pipeline can recover this error."""

        lowered = str(message).lower()
        return any(
            token in lowered
            for token in (
                "device disconnected",
                "device has been disconnected",
                "no device connected",
                "device or resource busy",
                "failed to resolve request",
                "backend",
                "uvc",
                "usb",
                "pipeline",
                "wait_for_frames cannot be called before start",
                "stop() cannot be called before start",
                "i/o error",
                "io error",
            )
        )

    def _set_transient_or_stale_timeout_error(self, message: str) -> bool:
        now = time.time()
        stale_after_sec = float(self.config.stale_after_sec)
        with self._lock:
            self._timeout_count += 1
            self._consecutive_timeout_count += 1
            self._last_runtime_error = str(message)
            last_frame_at = self._last_frame_receive_timestamp
            started_at = self._started_at
            # A pipeline can be successfully started before its first frame is
            # available.  During this bounded startup grace period, record the
            # timeout for diagnostics but do not mark the camera stale or
            # rebuild the native pipeline.  Repeated stop/start cycles while
            # sibling RealSense devices are negotiating can corrupt the SDK's
            # native heap (the observed ``free(): corrupted unsorted chunks``).
            startup_grace_active = (
                last_frame_at is None
                and started_at is not None
                and now - float(started_at) < float(self.config.startup_frame_grace_sec)
            )
            if startup_grace_active:
                is_stale = False
            elif last_frame_at is not None:
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
                and self._consecutive_timeout_count
                >= int(self.config.restart_after_timeouts)
            )
        return should_restart

    def _restart_pipeline(
        self,
        message: str,
        *,
        reason: str,
        generation: int | None = None,
        shutdown_event: threading.Event | None = None,
    ) -> None:
        if generation is None:
            with self._lock:
                generation = int(self._generation)
        if shutdown_event is None:
            shutdown_event = self._shutdown_event
        if not self._is_generation_active(generation, shutdown_event):
            return
        with self._lock:
            restart_attempt_limit = int(
                self.config.restart_max_attempts_without_complete_frame
            )
            if (
                restart_attempt_limit > 0
                and self._restart_attempts_without_complete_frame
                >= restart_attempt_limit
            ):
                self._running = False
                self._lifecycle_state = _LIFECYCLE_FAULTED
                self._last_error = (
                    "restart_circuit_open:"
                    f"attempts_without_complete_frame="
                    f"{self._restart_attempts_without_complete_frame}:"
                    f"last_reason={reason}:{message}"
                )
                self._last_runtime_error = self._last_error
                shutdown_event.set()
                circuit_open = True
            else:
                self._restart_attempts_without_complete_frame += 1
                circuit_open = False
            self._restart_attempt_count += 1
            if not circuit_open:
                self._last_error = f"restarting_after_{reason}: {message}"
                self._last_runtime_error = str(message)
        if circuit_open:
            with self._pipeline_lock:
                faulted_pipeline = self._pipeline
                self._pipeline = None
                self._profile = None
                self._depth_align = None
            with _REALSENSE_SDK_LIFECYCLE_LOCK:
                self._stop_pipeline_quietly(faulted_pipeline)
            self._frame_buffer.clear()
            with self._lock:
                self._latest_frame = None
            return
        rs = _load_pyrealsense2()
        with _REALSENSE_SDK_LIFECYCLE_LOCK:
            if not self._is_generation_active(generation, shutdown_event):
                return
            with self._pipeline_lock:
                old_pipeline = self._pipeline
                self._pipeline = None
                self._profile = None
                self._depth_align = None
            if old_pipeline is not None:
                try:
                    old_pipeline.stop()
                except Exception as exc:  # noqa: BLE001
                    with self._lock:
                        self._last_runtime_error = (
                            f"pipeline_stop_failed:{type(exc).__name__}:{exc}"
                        )
        self._frame_buffer.clear()
        with self._lock:
            self._latest_frame = None
        cooldown_sec = float(self.config.restart_cooldown_sec)
        if cooldown_sec > 0.0 and shutdown_event.wait(cooldown_sec):
            return
        if not self._is_generation_active(generation, shutdown_event):
            return
        try:
            with _REALSENSE_SDK_LIFECYCLE_LOCK:
                if not self._is_generation_active(generation, shutdown_event):
                    return
                pipeline, profile = self._start_pipeline(rs)
        except Exception as exc:  # noqa: BLE001
            if not self._is_generation_active(generation, shutdown_event):
                return
            with self._lock:
                self._restart_failure_count += 1
                self._last_error = (
                    f"pipeline_restart_failed:{type(exc).__name__}:{exc}"
                )
                self._last_runtime_error = self._last_error
            return
        with self._pipeline_lock:
            install_pipeline = self._is_generation_active(
                generation,
                shutdown_event,
            )
            if install_pipeline:
                self._pipeline = pipeline
                self._profile = profile
        if not install_pipeline:
            with _REALSENSE_SDK_LIFECYCLE_LOCK:
                self._stop_pipeline_quietly(pipeline)
            return
        with self._lock:
            self._device_info = self._device_info_from_profile(profile)
            self._started_at = time.time()
            self._last_error = None
            self._last_runtime_error = None
            self._last_frame_receive_timestamp = None
            self._consecutive_timeout_count = 0
            self._consecutive_incomplete_frameset_count = 0
            self._consecutive_black_frame_count = 0
            self._last_sampled_max_value = None
            self._restart_count += 1
            self._last_restart_timestamp = time.time()

    def _restart_pipeline_after_stale_timeout(self, message: str) -> None:
        # Compatibility seam for callers/tests that exercised the previous
        # timeout-only recovery method directly.
        self._restart_pipeline(message, reason="frame_timeout")

    def _is_near_zero_frame(self, image: np.ndarray) -> tuple[bool, int]:
        """Detect an uninitialised all-zero payload using at most ~1K pixels."""

        array = np.asarray(image)
        if array.size == 0:
            return True, 0
        row_step = max(1, int(array.shape[0]) // 32)
        col_step = max(1, int(array.shape[1]) // 32)
        sample = array[::row_step, ::col_step]
        sampled_max = int(np.max(sample))
        return sampled_max <= int(self.config.black_frame_max_value), sampled_max

    def _handle_black_frame(
        self,
        image: np.ndarray,
        *,
        generation: int | None = None,
        shutdown_event: threading.Event | None = None,
    ) -> bool:
        is_black, sampled_max = self._is_near_zero_frame(image)
        with self._lock:
            self._last_sampled_max_value = sampled_max
            if not is_black:
                self._consecutive_black_frame_count = 0
                return False
            self._black_frame_count += 1
            self._consecutive_black_frame_count += 1
            streak = int(self._consecutive_black_frame_count)
            if bool(self.config.accept_near_zero_frames):
                self._last_error = None
                return False
            self._last_error = (
                "near_zero_color_frame:"
                f"sampled_max={sampled_max}:streak={streak}/"
                f"{int(self.config.restart_after_black_frames)}"
            )
        if (
            bool(self.config.restart_on_black_frames)
            and streak >= int(self.config.restart_after_black_frames)
        ):
            self._restart_pipeline(
                self._last_error or "near_zero_color_frame",
                reason="black_frame",
                generation=generation,
                shutdown_event=shutdown_event,
            )
        return True

    def _capture_loop(
        self,
        generation: int,
        shutdown_event: threading.Event,
    ) -> None:
        rs = _load_pyrealsense2()
        while self._is_generation_active(generation, shutdown_event):
            try:
                with self._pipeline_lock:
                    pipeline = self._pipeline
                if pipeline is None:
                    self._restart_pipeline(
                        "pipeline_missing",
                        reason="pipeline_missing",
                        generation=generation,
                        shutdown_event=shutdown_event,
                    )
                    continue
                frames = pipeline.wait_for_frames(
                    timeout_ms=int(self.config.wait_timeout_ms)
                )
                received_at = time.time()
                color_frame, depth_frame = self._required_stream_frames(frames)
                if not self._required_streams_complete(color_frame, depth_frame):
                    if self._record_incomplete_frameset(
                        color_frame,
                        depth_frame,
                        received_at=received_at,
                    ):
                        self._restart_pipeline(
                            self._last_error or "incomplete_frameset",
                            reason="incomplete_frameset",
                            generation=generation,
                            shutdown_event=shutdown_event,
                        )
                    continue
                self._record_complete_frameset(received_at=received_at)
                frame_index = int(self._frame_counter)
                self._frame_counter += 1
                image = _color_image_to_bgr(
                    np.asanyarray(color_frame.get_data()),
                    self._active_color_format,
                )
                if self._handle_black_frame(
                    image,
                    generation=generation,
                    shutdown_event=shutdown_event,
                ):
                    continue
                if self.config.color_space == "rgb":
                    image = np.ascontiguousarray(image[..., ::-1])
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
                # RealSense numpy views may alias SDK-owned frame memory.  One
                # explicit copy per new camera frame creates the immutable
                # history snapshot; sample-rate lookups never recopy it.
                image_snapshot = np.array(image, copy=True, order="C")
                image_snapshot.setflags(write=False)
                depth_snapshot = None
                if depth_image is not None:
                    depth_snapshot = np.array(depth_image, copy=True, order="C")
                    depth_snapshot.setflags(write=False)
                frame = VideoFrame(
                    image=image_snapshot,
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
                        "color_format": self.config.color_format,
                        "source_clock": "receiver_wall_time",
                        "available_clock": "local_wall_time",
                        "source_generation": generation,
                        "camera_timestamp_sec": frame_timestamp_sec,
                        "timestamp_domain": timestamp_domain,
                        "sensor_timestamp_ms": sensor_timestamp_ms,
                        "color_frame_number": color_frame_number,
                    },
                    depth_image=depth_snapshot,
                    depth_metadata=depth_metadata,
                )
                frame = self._frame_buffer.publish(frame)
                with self._lock:
                    self._latest_frame = frame
                    self._last_error = None
                    self._last_runtime_error = None
                    self._last_frame_receive_timestamp = received_at
                    self._consecutive_timeout_count = 0
                    self._restart_attempts_without_complete_frame = 0
            except RuntimeError as exc:
                if not self._is_generation_active(generation, shutdown_event):
                    break
                message = str(exc)
                if self._is_frame_timeout_error(message):
                    if self._set_transient_or_stale_timeout_error(message):
                        self._restart_pipeline(
                            message,
                            reason="frame_timeout",
                            generation=generation,
                            shutdown_event=shutdown_event,
                        )
                else:
                    with self._lock:
                        self._last_runtime_error = message
                        self._last_error = message
                    if (
                        bool(self.config.restart_on_runtime_error)
                        and self._is_recoverable_runtime_error(message)
                    ):
                        self._restart_pipeline(
                            message,
                            reason="runtime_error",
                            generation=generation,
                            shutdown_event=shutdown_event,
                        )
            except Exception as exc:  # noqa: BLE001
                message = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._last_runtime_error = message
                    self._last_error = message
                if (
                    bool(self.config.restart_on_runtime_error)
                    and self._is_recoverable_runtime_error(message)
                ):
                    self._restart_pipeline(
                        message,
                        reason="runtime_error",
                        generation=generation,
                        shutdown_event=shutdown_event,
                    )
                elif self._is_generation_active(generation, shutdown_event):
                    time.sleep(
                        min(
                            0.1,
                            max(0.01, float(self.config.restart_cooldown_sec)),
                        )
                    )

    def get_latest_frame(self) -> VideoFrame | None:
        with self._lock:
            if self._latest_frame is None:
                return None
            frame = self._latest_frame
        return VideoFrame(
            image=np.asarray(frame.image).copy(),
            source_timestamp=float(frame.source_timestamp),
            receive_timestamp=frame.receive_timestamp,
            available_timestamp=frame.available_timestamp,
            frame_index=frame.frame_index,
            metadata=dict(frame.metadata),
            depth_image=(
                None
                if frame.depth_image is None
                else np.asarray(frame.depth_image).copy()
            ),
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
            info["black_frame_count"] = int(self._black_frame_count)
            info["consecutive_black_frame_count"] = int(
                self._consecutive_black_frame_count
            )
            info["last_sampled_max_value"] = self._last_sampled_max_value
            info["restart_count"] = int(self._restart_count)
            info["restart_attempt_count"] = int(self._restart_attempt_count)
            info["restart_failure_count"] = int(self._restart_failure_count)
            info["last_restart_timestamp"] = self._last_restart_timestamp
            info["restart_attempts_without_complete_frame"] = int(
                self._restart_attempts_without_complete_frame
            )
            info["last_frame_receive_timestamp"] = self._last_frame_receive_timestamp
            info["last_frameset_receive_timestamp"] = (
                self._last_frameset_receive_timestamp
            )
            info["last_color_receive_timestamp"] = self._last_color_receive_timestamp
            info["last_depth_receive_timestamp"] = self._last_depth_receive_timestamp
            info["incomplete_frameset_count"] = int(self._incomplete_frameset_count)
            info["consecutive_incomplete_frameset_count"] = int(
                self._consecutive_incomplete_frameset_count
            )
            info["missing_color_frame_count"] = int(self._missing_color_frame_count)
            info["missing_depth_frame_count"] = int(self._missing_depth_frame_count)
            info["last_frame_age_sec"] = (
                None
                if self._last_frame_receive_timestamp is None
                else float(now - float(self._last_frame_receive_timestamp))
            )
            info["stale_after_sec"] = float(self.config.stale_after_sec)
            info["startup_wait_timeout_ms"] = int(self.config.startup_wait_timeout_ms)
            info["startup_frame_grace_sec"] = float(self.config.startup_frame_grace_sec)
            info["running"] = bool(self._running)
            info["lifecycle_state"] = str(self._lifecycle_state)
            info["generation"] = int(self._generation)
            worker = self._thread
            info["worker_alive"] = bool(worker is not None and worker.is_alive())
        return info

    def close(self) -> None:
        if not self._lifecycle_lock.acquire(blocking=False):
            message = (
                f"{self.name} close could not acquire its lifecycle lock; "
                "a native startup operation may still be blocked"
            )
            self._mark_faulted("close_lifecycle_lock_unavailable:" + message)
            raise_lifecycle_failures(
                f"{self.name}.close",
                (
                    LifecycleFailure(
                        label="lifecycle_lock",
                        error=RuntimeError(message),
                    ),
                ),
            )
            return
        try:
            self._close_locked()
        finally:
            self._lifecycle_lock.release()

    def _close_locked(self) -> None:
        failures: list[LifecycleFailure] = []
        with self._lock:
            self._running = False
            self._lifecycle_state = _LIFECYCLE_STOPPING
            self._generation += 1
            shutdown_event = self._shutdown_event
            shutdown_event.set()
        with self._pipeline_lock:
            pipeline = self._pipeline
            self._pipeline = None
            self._profile = None
            self._depth_align = None
        self._active_color_format = self.config.color_format
        if pipeline is not None:
            try:
                join_timeout_sec = max(
                    1.5,
                    float(self.config.wait_timeout_ms) / 1000.0 + 0.5,
                )
                stop_error = self._stop_pipeline_bounded(
                    pipeline,
                    timeout_sec=join_timeout_sec,
                )
            except BaseException as exc:  # noqa: BLE001 - preserve teardown evidence.
                stop_error = exc
            if stop_error is not None:
                with self._lock:
                    self._last_runtime_error = (
                        f"pipeline_stop_failed:{type(stop_error).__name__}:{stop_error}"
                    )
                    if self._last_error is None:
                        self._last_error = self._last_runtime_error
                failures.append(
                    LifecycleFailure(label="pipeline.stop", error=stop_error)
                )
        worker = self._thread
        worker_alive = False
        if worker is not None:
            try:
                join_timeout_sec = max(
                    1.5,
                    float(self.config.wait_timeout_ms) / 1000.0 + 0.5,
                )
                worker.join(timeout=join_timeout_sec)
            except BaseException as exc:  # noqa: BLE001
                failures.append(
                    LifecycleFailure(label=f"worker[{worker.name}].join", error=exc)
                )
            worker_alive = bool(worker.is_alive())
            if worker_alive:
                failures.append(
                    LifecycleFailure(
                        label=f"worker[{worker.name}].join_timeout",
                        error=RuntimeError(
                            f"{self.name} worker did not stop within "
                            f"{join_timeout_sec:.3f}s"
                        ),
                    )
                )
            else:
                self._thread = None
        with self._lock:
            if worker_alive or failures:
                self._lifecycle_state = _LIFECYCLE_FAULTED
            else:
                self._lifecycle_state = _LIFECYCLE_STOPPED
        raise_lifecycle_failures(f"{self.name}.close", failures)
