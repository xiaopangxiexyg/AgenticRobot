"""RealSense camera construction and lifecycle helpers for real teleoperation."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Protocol

from hardware.realsense.color import normalize_color_format
from robot_motion.runtime import run_best_effort_cleanup

from ..collection.sample_assembler import DEFAULT_CAMERA_FRAME_MAX_AGE_SEC
from ..log_utils import log_event as _log
from ..sources.camera import (
    REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
    REALSENSE_STARTUP_WARMUP_POLICIES,
    RealSenseRGBCameraConfig,
    RealSenseRGBSource,
    load_realsense_camera_config_map,
    start_real_sense_source_with_containment,
)
from .wrist_depth_freshness import (
    DEFAULT_WRIST_DEPTH_SAVE_FPS,
    DEFAULT_WRIST_DEPTH_STREAM_FPS,
    WRIST_DEPTH_CAMERA_KEYS,
)

CAMERA_START_STAGGER_SEC = 0.5
CAMERA_DEFERRED_RETRY_COOLDOWN_SEC = 1.0
# A native librealsense start can block below Python (for example while USB
# xHCI is waiting for a device response). Keep the batch caller bounded while
# preserving the historical synchronous API for non-RealSense sources.
CAMERA_STARTUP_OPERATION_TIMEOUT_SEC = 15.0
CAMERA_STARTUP_SELF_CHECK_DURATION_SEC = 3.0
CAMERA_STARTUP_SELF_CHECK_INTERVAL_SEC = 0.2
CAMERA_STARTUP_SELF_CHECK_CONSECUTIVE_FRESH_SAMPLES = 3
CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC = DEFAULT_CAMERA_FRAME_MAX_AGE_SEC

_CAMERA_STARTUP_SEQUENCE = count()
_CAMERA_STARTUP_SEQUENCE_ATTR = "_real_teleop_startup_sequence"


class CameraStartupProgress(Protocol):
    """Narrow presentation port accepted by camera lifecycle helpers."""

    def begin(
        self,
        key: str,
        *,
        label: str,
        message: str,
        requirement: str = "feature_dependent",
    ) -> None: ...

    def update(self, key: str, *, message: str) -> None: ...

    def succeed(self, key: str, *, message: str = "连接完成") -> None: ...

    def fail(self, key: str, error: str, *, message: str = "连接失败") -> None: ...


@dataclass
class CameraStartupFreshnessGate:
    """Shared consecutive-new-frame admission used after camera startup."""

    camera_keys: tuple[str, ...]
    required_consecutive_samples: int = (
        CAMERA_STARTUP_SELF_CHECK_CONSECUTIVE_FRESH_SAMPLES
    )
    max_frame_age_sec: float = CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC
    previous_frame_counts: dict[str, int | None] = field(init=False)
    fresh_frame_streaks: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.camera_keys = tuple(str(key) for key in self.camera_keys)
        self.required_consecutive_samples = max(
            1,
            int(self.required_consecutive_samples),
        )
        self.max_frame_age_sec = max(0.0, float(self.max_frame_age_sec))
        self.previous_frame_counts = {key: None for key in self.camera_keys}
        self.fresh_frame_streaks = {key: 0 for key in self.camera_keys}

    def update(self, status: Mapping[str, object]) -> bool:
        for key in self.camera_keys:
            item = status.get(key)
            item = item if isinstance(item, Mapping) else {}
            try:
                frame_count = int(item.get("frame_count"))
            except (TypeError, ValueError):
                frame_count = None
            try:
                frame_age_sec = float(item.get("last_frame_age_sec"))
            except (TypeError, ValueError):
                frame_age_sec = None
            previous_count = self.previous_frame_counts[key]
            is_new_fresh_frame = bool(
                frame_count is not None
                and frame_count > 0
                and (previous_count is None or frame_count > previous_count)
                and frame_age_sec is not None
                and 0.0 <= frame_age_sec <= self.max_frame_age_sec
                and str(item.get("status_level") or "") == "ok"
            )
            self.fresh_frame_streaks[key] = (
                self.fresh_frame_streaks[key] + 1
                if is_new_fresh_frame
                else 0
            )
            self.previous_frame_counts[key] = frame_count
        return bool(self.fresh_frame_streaks) and all(
            streak >= self.required_consecutive_samples
            for streak in self.fresh_frame_streaks.values()
        )


def _camera_warmup_policy(source: object) -> str:
    config = getattr(source, "config", None)
    return str(getattr(config, "startup_warmup_policy", "unknown"))


def _is_bounded_first_frame_startup_failure(
    source: object,
    detail: str,
) -> bool:
    return bool(
        _camera_warmup_policy(source) == "bounded_first_frame"
        and "startup_warmup_failed:policy=bounded_first_frame" in str(detail).lower()
    )


def _configured_camera_startup_retry_attempts(source: object) -> int:
    config = getattr(source, "config", None)
    raw_attempts = getattr(config, "startup_software_retry_attempts", 0)
    if isinstance(raw_attempts, bool):
        raise TypeError("camera startup_software_retry_attempts must be an integer")
    try:
        attempts = int(raw_attempts)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "camera startup_software_retry_attempts must be an integer"
        ) from exc
    if attempts < 0:
        raise ValueError("camera startup_software_retry_attempts must be >= 0")
    return attempts


def _deferred_camera_startup_retry_budget(source: object, detail: str) -> int:
    """Return retries owned by the serial batch, never by one source start call."""

    configured = _configured_camera_startup_retry_attempts(source)
    if _is_bounded_first_frame_startup_failure(source, detail):
        # Preserve the existing one deferred recovery attempt for every formal
        # bounded-first-frame source. A larger reviewed per-camera budget (for
        # example the front camera's two attempts) remains authoritative.
        return max(1, configured)
    if configured > 0 and RealSenseRGBSource._is_startup_connection_error(detail):
        return configured
    return 0


def _camera_startup_items(
    camera_sources: Mapping[str, RealSenseRGBSource],
) -> tuple[tuple[str, RealSenseRGBSource], ...]:
    """Return a deterministic hardware-start order independent of camera role.

    RealSense devices on the same station can be sensitive to multi-pipeline
    negotiation order.  Logical keys must continue to describe the physical
    mount, so an optional numeric ``metadata.startup_priority`` controls only
    lifecycle ordering.  Config declaration order remains the stable fallback.
    """

    ranked: list[tuple[float, int, str, RealSenseRGBSource]] = []
    for declaration_index, (camera_key, source) in enumerate(camera_sources.items()):
        config = getattr(source, "config", None)
        metadata = getattr(config, "metadata", {})
        raw_priority = (
            metadata.get("startup_priority", declaration_index)
            if isinstance(metadata, Mapping)
            else declaration_index
        )
        try:
            priority = float(raw_priority)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"camera {camera_key!r} metadata.startup_priority must be numeric; "
                f"got {raw_priority!r}."
            ) from exc
        ranked.append((priority, declaration_index, str(camera_key), source))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return tuple((camera_key, source) for _, _, camera_key, source in ranked)


def apply_real_teleop_camera_startup_defaults(
    camera_configs: Mapping[str, RealSenseRGBCameraConfig],
    *,
    color_format: str = "bgr8",
    warmup_policy: str = REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
) -> None:
    """Apply the canonical formal-collection RealSense startup contract."""

    requested_color_format = normalize_color_format(
        str(color_format or "bgr8").strip().lower()
    )
    requested_warmup_policy = str(
        warmup_policy or REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME
    ).strip().lower()
    if requested_warmup_policy not in REALSENSE_STARTUP_WARMUP_POLICIES:
        raise ValueError(
            "camera startup warmup policy must be one of "
            f"{REALSENSE_STARTUP_WARMUP_POLICIES}; got "
            f"{requested_warmup_policy!r}."
        )
    for camera_key, config in camera_configs.items():
        config.metadata = dict(config.metadata)
        configured_warmup_policy = str(
            config.metadata.get(
                "configured_startup_warmup_policy",
                config.startup_warmup_policy,
            )
        )
        configured_require_complete_frameset = bool(
            config.metadata.get(
                "configured_require_complete_frameset",
                config.require_complete_frameset,
            )
        )
        configured_restart_on_incomplete_frameset = bool(
            config.metadata.get(
                "configured_restart_on_incomplete_frameset",
                config.restart_on_incomplete_frameset,
            )
        )
        configured_restart_attempt_limit = int(
            config.metadata.get(
                "configured_restart_max_attempts_without_complete_frame",
                config.restart_max_attempts_without_complete_frame,
            )
        )
        configured_hardware_reset_on_failure = bool(
            config.metadata.get(
                "configured_startup_hardware_reset_on_failure",
                config.startup_hardware_reset_on_failure,
            )
        )
        config.startup_warmup_policy = requested_warmup_policy
        config.metadata.setdefault(
            "configured_startup_warmup_policy",
            configured_warmup_policy
        )
        config.metadata["real_teleop_startup_warmup_policy"] = (
            requested_warmup_policy
        )
        config.metadata.setdefault(
            "configured_require_complete_frameset",
            configured_require_complete_frameset
        )
        config.metadata.setdefault(
            "configured_restart_on_incomplete_frameset",
            configured_restart_on_incomplete_frameset
        )
        config.metadata.setdefault(
            "configured_restart_max_attempts_without_complete_frame",
            configured_restart_attempt_limit
        )
        config.metadata.setdefault(
            "configured_startup_hardware_reset_on_failure",
            configured_hardware_reset_on_failure
        )
        robust_stream_lifecycle = (
            requested_warmup_policy
            == REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME
        )
        config.require_complete_frameset = robust_stream_lifecycle
        config.restart_on_incomplete_frameset = robust_stream_lifecycle
        config.restart_max_attempts_without_complete_frame = (
            (
                configured_restart_attempt_limit
                if configured_restart_attempt_limit > 0
                else 3
            )
            if robust_stream_lifecycle
            else 0
        )
        # The robust profile owns a clean, bounded reopen first. Hardware reset
        # remains available only through the explicit startup-reset option;
        # legacy keeps the configured connection-error reset fallback.
        config.startup_hardware_reset_on_failure = (
            False
            if robust_stream_lifecycle
            else configured_hardware_reset_on_failure
        )
        config.metadata["real_teleop_robust_stream_lifecycle"] = (
            robust_stream_lifecycle
        )
        configured_color_format = str(config.color_format)
        if configured_color_format != requested_color_format:
            config.metadata["configured_color_format"] = configured_color_format
            config.metadata["real_teleop_color_format_override"] = (
                requested_color_format
            )
            _log(
                "Camera",
                f"{camera_key} color transport overridden for real teleop: "
                f"{configured_color_format} -> {requested_color_format}",
            )
            config.color_format = requested_color_format


def apply_real_teleop_camera_history_defaults(
    camera_configs: Mapping[str, RealSenseRGBCameraConfig],
    *,
    frame_max_age_sec: float,
) -> None:
    """Retain enough causal RGB history for the configured age contract.

    A sample cut is taken before command/IK work.  Several camera frames may
    arrive before the writer asks for the newest frame at or before that cut,
    so the historical two-frame default is insufficient even when every
    camera is healthy and sustaining its configured FPS.
    """

    max_age_sec = float(frame_max_age_sec)
    if not math.isfinite(max_age_sec) or max_age_sec <= 0.0:
        raise ValueError("camera frame max age must be finite and positive")
    for config in camera_configs.values():
        configured_keep_last = int(config.keep_last)
        required_keep_last = int(math.ceil(max_age_sec * float(config.fps))) + 2
        config.keep_last = max(configured_keep_last, required_keep_last)
        config.metadata = dict(config.metadata)
        config.metadata.setdefault("configured_keep_last", configured_keep_last)
        config.metadata.update(
            {
                "real_teleop_causal_history_keep_last": int(config.keep_last),
                "real_teleop_causal_history_max_age_sec": max_age_sec,
            }
        )


def _build_camera_sources(args: argparse.Namespace) -> dict[str, RealSenseRGBSource]:
    if bool(getattr(args, "no_camera", False)):
        return {}
    config_text = str(getattr(args, "camera_config", "") or "").strip()
    if not config_text:
        if not bool(getattr(args, "no_collection", False)):
            raise ValueError(
                "formal collection requires --camera-config; use --no-camera only "
                "when the entire run intentionally has no camera modality"
            )
        return {}
    config_path = Path(config_text).expanduser()
    if not config_path.exists():
        if (
            bool(getattr(args, "vis_cam", False))
            or not bool(getattr(args, "no_collection", False))
        ):
            raise FileNotFoundError(f"Camera config not found: {config_path}")
        _log("Camera", f"optional camera config not found; preview disabled: {config_path}")
        return {}
    camera_configs = load_realsense_camera_config_map(config_path)
    # The historical station configs under ``cameras/configs/...`` request
    # YUYV.  That transport is valid on a single device but is unreliable
    # while several RealSense devices negotiate concurrently on this station;
    # the native SDK test showed BGR8 to be stable for the same resolutions.
    # Apply an explicit real-teleop default so stale machine-local configs do
    # not reintroduce the crash.  Users can still request another transport
    # with --camera-color-format for a validated setup.
    apply_real_teleop_camera_startup_defaults(
        camera_configs,
        color_format=str(getattr(args, "camera_color_format", "bgr8") or "bgr8"),
        warmup_policy=str(
            getattr(
                args,
                "camera_startup_warmup_policy",
                REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
            )
            or REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME
        ),
    )
    apply_real_teleop_camera_history_defaults(
        camera_configs,
        frame_max_age_sec=float(
            getattr(
                args,
                "camera_frame_max_age_sec",
                DEFAULT_CAMERA_FRAME_MAX_AGE_SEC,
            )
        ),
    )
    wrist_depth_requested = bool(getattr(args, "wrist_depth_mask", False)) or bool(
        getattr(args, "wrist_depth_raw", False)
    )
    if wrist_depth_requested:
        available_wrist_keys = sorted(set(camera_configs) & set(WRIST_DEPTH_CAMERA_KEYS))
        missing_wrist_keys = sorted(set(WRIST_DEPTH_CAMERA_KEYS) - set(camera_configs))
        requested_wrist_cam_width = getattr(args, "wrist_cam_width", None)
        requested_wrist_cam_height = getattr(args, "wrist_cam_height", None)
        align_depth_to_color = bool(getattr(args, "wrist_depth_align_to_color", True))
        if missing_wrist_keys:
            _log(
                "Camera",
                "wrist depth requested but some wrist cameras are absent from config: "
                f"missing={missing_wrist_keys}",
            )
        for camera_key, config in camera_configs.items():
            if camera_key in WRIST_DEPTH_CAMERA_KEYS:
                config.enable_depth = True
                config.depth_width = int(
                    requested_wrist_cam_width
                    if requested_wrist_cam_width is not None
                    else config.width
                )
                config.depth_height = int(
                    requested_wrist_cam_height
                    if requested_wrist_cam_height is not None
                    else config.height
                )
                config.depth_fps = int(getattr(args, "wrist_depth_stream_fps", DEFAULT_WRIST_DEPTH_STREAM_FPS))
                config.align_depth_to_color = align_depth_to_color
                if requested_wrist_cam_width is not None and requested_wrist_cam_height is not None:
                    old_rgb_resolution = (int(config.width), int(config.height))
                    new_rgb_resolution = (int(config.depth_width), int(config.depth_height))
                    if old_rgb_resolution != new_rgb_resolution:
                        config.metadata = dict(config.metadata)
                        config.metadata.update(
                            {
                                "wrist_cam_original_rgb_width": old_rgb_resolution[0],
                                "wrist_cam_original_rgb_height": old_rgb_resolution[1],
                                "wrist_cam_matched_rgb_depth_resolution": True,
                            }
                        )
                        config.width, config.height = new_rgb_resolution
                        _log(
                            "Camera",
                            f"{camera_key} wrist camera stream resolution changed for RGB+depth: "
                            f"{old_rgb_resolution[0]}x{old_rgb_resolution[1]} -> "
                            f"{new_rgb_resolution[0]}x{new_rgb_resolution[1]}. "
                            "D405 color/depth profiles cannot always resolve mixed resolutions.",
                        )
            else:
                config.enable_depth = False
        wrist_cam_resolution_text = (
            f"{int(requested_wrist_cam_width)}x{int(requested_wrist_cam_height)}"
            if requested_wrist_cam_width is not None and requested_wrist_cam_height is not None
            else "match_rgb"
        )
        _log(
            "Camera",
            "wrist depth enabled for D405 wrist cameras only: "
            f"keys={available_wrist_keys} save_fps={DEFAULT_WRIST_DEPTH_SAVE_FPS} "
            f"depth_stream_fps={int(getattr(args, 'wrist_depth_stream_fps', DEFAULT_WRIST_DEPTH_STREAM_FPS))} "
            f"wrist_cam_stream_resolution={wrist_cam_resolution_text} "
            f"align_depth_to_color={align_depth_to_color} "
            f"mask={bool(getattr(args, 'wrist_depth_mask', False))} "
            f"raw={bool(getattr(args, 'wrist_depth_raw', False))}",
        )
    return {
        camera_key: RealSenseRGBSource(config)
        for camera_key, config in camera_configs.items()
    }


def _select_preview_camera_sources(
    camera_sources: dict[str, RealSenseRGBSource],
    requested_key: str,
) -> dict[str, RealSenseRGBSource]:
    if not camera_sources:
        return {}
    key = str(requested_key or "").strip()
    if not key or key.lower() in {"all", "*"}:
        return dict(camera_sources)
    if key:
        if key not in camera_sources:
            available = ", ".join(sorted(camera_sources))
            raise ValueError(
                f"--vis-cam-key {key!r} not found. Available camera keys: {available}"
            )
        return {key: camera_sources[key]}


def _reset_camera_startup_sequences(
    camera_sources: Mapping[str, object],
) -> None:
    for source in camera_sources.values():
        try:
            setattr(source, _CAMERA_STARTUP_SEQUENCE_ATTR, None)
        except (AttributeError, TypeError):
            # A read-only test double or proxy does not affect startup itself.
            continue


def _record_camera_start_attempt(source: object) -> None:
    try:
        setattr(source, _CAMERA_STARTUP_SEQUENCE_ATTR, next(_CAMERA_STARTUP_SEQUENCE))
    except (AttributeError, TypeError):
        # Ordering is an observability aid; failure to annotate a source must
        # never change whether the source is admitted.
        return


def _start_camera_source_with_containment(
    source: object,
    *,
    timeout_sec: float = CAMERA_STARTUP_OPERATION_TIMEOUT_SEC,
    software_retry_attempts: int | None = 0,
) -> None:
    """Start one RealSense source with a bounded Python-side wait.

    librealsense does not expose a cancellable timeout for every native start
    edge. Running only the RealSense start call in a daemon worker allows the
    batch owner to fail closed instead of being held behind a kernel USB wait.
    Non-RealSense sources retain their historical direct call semantics.
    """

    if not isinstance(source, RealSenseRGBSource):
        _record_camera_start_attempt(source)
        source.start()
        return
    _record_camera_start_attempt(source)
    start_real_sense_source_with_containment(
        source,
        timeout_sec=timeout_sec,
        software_retry_attempts=software_retry_attempts,
    )


def _start_camera_sources(
    camera_sources: Mapping[str, RealSenseRGBSource],
    *,
    stagger_sec: float = CAMERA_START_STAGGER_SEC,
    startup_progress: CameraStartupProgress | None = None,
    startup_timeout_sec: float = CAMERA_STARTUP_OPERATION_TIMEOUT_SEC,
) -> None:
    _reset_camera_startup_sequences(camera_sources)
    batch_started = time.perf_counter()
    source_elapsed_sec: dict[str, float] = {}
    stagger_elapsed_sec = 0.0
    startup_items = _camera_startup_items(camera_sources)
    _log("Camera", f"serial startup order={[key for key, _ in startup_items]}")
    for index, (camera_key, source) in enumerate(startup_items):
        progress_key = f"camera.{camera_key}"
        if startup_progress is not None:
            startup_progress.begin(
                progress_key,
                label=f"相机 {camera_key}",
                message=(
                    "等待错峰启动"
                    if index > 0 and float(stagger_sec) > 0.0
                    else "正在启动视频流"
                ),
                requirement="required_for_collection",
            )
        if index > 0 and float(stagger_sec) > 0.0:
            _log(
                "Camera",
                f"waiting {float(stagger_sec):.2f}s before starting camera "
                f"{camera_key}",
            )
            stagger_started = time.perf_counter()
            time.sleep(float(stagger_sec))
            stagger_elapsed_sec += time.perf_counter() - stagger_started
            if startup_progress is not None:
                startup_progress.update(progress_key, message="正在启动视频流")
        _log("Camera", f"starting camera {camera_key}")
        start_started = time.perf_counter()
        try:
            _start_camera_source_with_containment(
                source,
                timeout_sec=startup_timeout_sec,
                software_retry_attempts=None,
            )
        except Exception as exc:
            source_elapsed_sec[str(camera_key)] = time.perf_counter() - start_started
            if startup_progress is not None:
                startup_progress.fail(
                    progress_key,
                    f"{type(exc).__name__}: {exc}",
                )
            _log_camera_startup_batch_timing(
                batch_started=batch_started,
                source_elapsed_sec=source_elapsed_sec,
                stagger_elapsed_sec=stagger_elapsed_sec,
                failed_camera=str(camera_key),
            )
            raise
        else:
            source_elapsed_sec[str(camera_key)] = time.perf_counter() - start_started
            _log(
                "Camera",
                f"camera {camera_key} video stream ready; "
                f"source_start_elapsed={source_elapsed_sec[str(camera_key)]:.3f}s "
                f"warmup_policy={_camera_warmup_policy(source)} "
                f"stages={getattr(source, 'startup_timing', {})}",
            )
            if startup_progress is not None:
                startup_progress.succeed(progress_key, message="视频流已启动")
    _log_camera_startup_batch_timing(
        batch_started=batch_started,
        source_elapsed_sec=source_elapsed_sec,
        stagger_elapsed_sec=stagger_elapsed_sec,
    )


def _start_camera_sources_best_effort(
    camera_sources: Mapping[str, RealSenseRGBSource],
    *,
    stagger_sec: float = CAMERA_START_STAGGER_SEC,
    deferred_retry_cooldown_sec: float = CAMERA_DEFERRED_RETRY_COOLDOWN_SEC,
    startup_progress: CameraStartupProgress | None = None,
    startup_timeout_sec: float = CAMERA_STARTUP_OPERATION_TIMEOUT_SEC,
) -> dict[str, str]:
    """Attempt every camera, then run configured retries in serial rounds.

    The first pass remains deliberately serial.  A failed RealSense pipeline
    is retried only after the other configured cameras have completed their
    first startup attempt and a short cooldown has elapsed. Bounded-first-frame
    failures retain at least one deferred attempt; a larger configured
    ``startup_software_retry_attempts`` budget is consumed here instead of
    reopening one device immediately while siblings are still negotiating.
    This mirrors the successful manual reconnect timing without parallel
    pipeline negotiation or a USB hardware reset. A native startup timeout is
    terminal for the batch: the blocked source is faulted, remaining sources
    are not opened, and no immediate ``close``/retry is attempted on that
    source.
    """
    _reset_camera_startup_sequences(camera_sources)
    errors: dict[str, str] = {}
    first_pass_errors: dict[str, str] = {}
    batch_started = time.perf_counter()
    source_elapsed_sec: dict[str, float] = {}
    retry_elapsed_sec: dict[str, float] = {}
    stagger_elapsed_sec = 0.0
    startup_items = _camera_startup_items(camera_sources)
    _log("Camera", f"serial startup order={[key for key, _ in startup_items]}")
    startup_aborted_after_timeout: str | None = None
    for index, (camera_key, source) in enumerate(startup_items):
        progress_key = f"camera.{camera_key}"
        if startup_aborted_after_timeout is not None:
            detail = (
                "startup_aborted_after_timeout:"
                f"camera={startup_aborted_after_timeout}"
            )
            errors[str(camera_key)] = detail
            first_pass_errors[str(camera_key)] = detail
            if startup_progress is not None:
                startup_progress.begin(
                    progress_key,
                    label=f"相机 {camera_key}",
                    message="因前一相机启动超时而停止尝试",
                    requirement="required_for_collection",
                )
                startup_progress.fail(progress_key, detail)
            _log("Camera", f"WARNING: camera {camera_key} not attempted: {detail}")
            continue
        if startup_progress is not None:
            startup_progress.begin(
                progress_key,
                label=f"相机 {camera_key}",
                message=(
                    "等待错峰启动"
                    if index > 0 and float(stagger_sec) > 0.0
                    else "正在启动视频流"
                ),
                requirement="required_for_collection",
            )
        if index > 0 and float(stagger_sec) > 0.0:
            _log(
                "Camera",
                f"waiting {float(stagger_sec):.2f}s before starting camera "
                f"{camera_key}",
            )
            stagger_started = time.perf_counter()
            time.sleep(float(stagger_sec))
            stagger_elapsed_sec += time.perf_counter() - stagger_started
            if startup_progress is not None:
                startup_progress.update(progress_key, message="正在启动视频流")
        _log("Camera", f"starting camera {camera_key}")
        start_started = time.perf_counter()
        try:
            _start_camera_source_with_containment(
                source,
                timeout_sec=startup_timeout_sec,
            )
        except Exception as exc:  # noqa: BLE001 - optional-device admission records failure.
            source_elapsed_sec[str(camera_key)] = time.perf_counter() - start_started
            detail = f"{type(exc).__name__}: {exc}"
            errors[str(camera_key)] = detail
            first_pass_errors[str(camera_key)] = detail
            _log("Camera", f"WARNING: camera {camera_key} unavailable: {detail}")
            if isinstance(exc, TimeoutError):
                startup_aborted_after_timeout = str(camera_key)
                _log(
                    "Camera",
                    "native startup timeout is terminal for this serial batch; "
                    "remaining cameras will not be opened or retried",
                )
            if startup_progress is not None:
                if _is_bounded_first_frame_startup_failure(source, detail):
                    startup_progress.update(
                        progress_key,
                        message="首轮未收到视频帧；等待批次结束后串行重试",
                    )
                else:
                    startup_progress.fail(progress_key, detail)
        else:
            source_elapsed_sec[str(camera_key)] = time.perf_counter() - start_started
            _log(
                "Camera",
                f"camera {camera_key} video stream ready; "
                f"source_start_elapsed={source_elapsed_sec[str(camera_key)]:.3f}s "
                f"warmup_policy={_camera_warmup_policy(source)} "
                f"stages={getattr(source, 'startup_timing', {})}",
            )
            if startup_progress is not None:
                startup_progress.succeed(progress_key, message="视频流已启动")

    # A timed-out native start may still own librealsense's process-wide
    # lifecycle lock inside its daemon worker. Do not let an earlier
    # first-frame failure queue a second start against that potentially wedged
    # SDK state; the batch remains fail-closed and its caller can rebuild the
    # process after the failed attempt is recorded.
    remaining_retry_attempts = (
        {}
        if startup_aborted_after_timeout is not None
        else {
            str(camera_key): _deferred_camera_startup_retry_budget(
                source,
                first_pass_errors[str(camera_key)],
            )
            for camera_key, source in startup_items
            if str(camera_key) in errors
        }
    )
    remaining_retry_attempts = {
        camera_key: attempts
        for camera_key, attempts in remaining_retry_attempts.items()
        if attempts > 0
    }
    retry_cooldown_elapsed_sec = 0.0
    recovered_keys: list[str] = []
    retry_round = 0
    while remaining_retry_attempts:
        retry_round += 1
        deferred_retry_keys = [
            str(camera_key)
            for camera_key, _source in startup_items
            if remaining_retry_attempts.get(str(camera_key), 0) > 0
        ]
        cooldown_sec = max(0.0, float(deferred_retry_cooldown_sec))
        _log(
            "Camera",
            "deferred serial startup retry scheduled: "
            f"round={retry_round} cameras={deferred_retry_keys} "
            f"remaining={remaining_retry_attempts} cooldown={cooldown_sec:.2f}s",
        )
        if cooldown_sec > 0.0:
            cooldown_started = time.perf_counter()
            time.sleep(cooldown_sec)
            retry_cooldown_elapsed_sec += time.perf_counter() - cooldown_started
        retry_aborted_after_timeout: str | None = None
        for camera_key in deferred_retry_keys:
            source = camera_sources[camera_key]
            progress_key = f"camera.{camera_key}"
            if retry_aborted_after_timeout is not None:
                errors[camera_key] = (
                    "deferred_startup_retry_aborted_after_timeout:"
                    f"camera={retry_aborted_after_timeout}"
                )
                if startup_progress is not None:
                    startup_progress.fail(progress_key, errors[camera_key])
                _log(
                    "Camera",
                    f"WARNING: camera {camera_key} deferred retry not attempted: "
                    f"{errors[camera_key]}",
                )
                remaining_retry_attempts.pop(camera_key, None)
                continue
            remaining_retry_attempts[camera_key] -= 1
            if startup_progress is not None:
                startup_progress.update(
                    progress_key,
                    message="正在执行延迟串行重试",
                )
            _log("Camera", f"deferred retry starting camera {camera_key}")
            retry_started = time.perf_counter()
            try:
                # A failed start already tears its pipeline down.  ``close``
                # also makes this robust to a source implementation that
                # retained partial state before raising.
                source.close()
                _start_camera_source_with_containment(
                    source,
                    timeout_sec=startup_timeout_sec,
                )
            except Exception as exc:  # noqa: BLE001 - preserve GUI/device admission.
                retry_elapsed_sec[camera_key] = retry_elapsed_sec.get(
                    camera_key, 0.0
                ) + (time.perf_counter() - retry_started)
                retry_detail = f"{type(exc).__name__}: {exc}"
                errors[camera_key] = (
                    "deferred_startup_retry_failed:"
                    f"round={retry_round}:first={first_pass_errors[camera_key]};"
                    f"retry={retry_detail}"
                )
                if isinstance(exc, TimeoutError):
                    retry_aborted_after_timeout = camera_key
                    errors[camera_key] = (
                        "deferred_startup_retry_timeout:"
                        f"round={retry_round}:first={first_pass_errors[camera_key]};"
                        f"retry={retry_detail}"
                    )
                    remaining_retry_attempts.pop(camera_key, None)
                elif (
                    not RealSenseRGBSource._is_startup_connection_error(retry_detail)
                    or remaining_retry_attempts[camera_key] <= 0
                ):
                    remaining_retry_attempts.pop(camera_key, None)
                _log(
                    "Camera",
                    f"WARNING: camera {camera_key} deferred retry unavailable: "
                    f"{retry_detail}",
                )
                if startup_progress is not None:
                    startup_progress.fail(progress_key, errors[camera_key])
            else:
                retry_elapsed_sec[camera_key] = retry_elapsed_sec.get(
                    camera_key, 0.0
                ) + (time.perf_counter() - retry_started)
                errors.pop(camera_key, None)
                remaining_retry_attempts.pop(camera_key, None)
                recovered_keys.append(camera_key)
                _log(
                    "Camera",
                    f"camera {camera_key} recovered on deferred serial retry; "
                    f"retry_elapsed={retry_elapsed_sec[camera_key]:.3f}s "
                    f"stages={getattr(source, 'startup_timing', {})}",
                )
                if startup_progress is not None:
                    startup_progress.succeed(
                        progress_key,
                        message="延迟串行重试成功，视频流已启动",
                    )
        if retry_aborted_after_timeout is not None:
            remaining_retry_attempts.clear()
    _log_camera_startup_batch_timing(
        batch_started=batch_started,
        source_elapsed_sec=source_elapsed_sec,
        stagger_elapsed_sec=stagger_elapsed_sec,
        retry_elapsed_sec=retry_elapsed_sec,
        retry_cooldown_elapsed_sec=retry_cooldown_elapsed_sec,
        recovered_cameras=recovered_keys,
        failed_camera=",".join(sorted(errors)) if errors else None,
    )
    return errors


def _log_camera_startup_batch_timing(
    *,
    batch_started: float,
    source_elapsed_sec: Mapping[str, float],
    stagger_elapsed_sec: float,
    retry_elapsed_sec: Mapping[str, float] | None = None,
    retry_cooldown_elapsed_sec: float = 0.0,
    recovered_cameras: list[str] | None = None,
    failed_camera: str | None = None,
) -> None:
    """Emit one compact receipt for the deliberately serial startup policy."""
    total_sec = max(0.0, time.perf_counter() - float(batch_started))
    source_total_sec = sum(max(0.0, float(value)) for value in source_elapsed_sec.values())
    per_camera = ",".join(
        f"{key}:{float(value):.3f}s" for key, value in source_elapsed_sec.items()
    ) or "none"
    retry_elapsed_sec = retry_elapsed_sec or {}
    retry_per_camera = ",".join(
        f"{key}:{float(value):.3f}s" for key, value in retry_elapsed_sec.items()
    ) or "none"
    recovered_text = ",".join(recovered_cameras or ()) or "none"
    failure_text = f" failed={failed_camera}" if failed_camera else ""
    _log(
        "CameraTiming",
        "serial_start "
        f"total={total_sec:.3f}s source_start_sum={source_total_sec:.3f}s "
        f"stagger_wait={max(0.0, float(stagger_elapsed_sec)):.3f}s "
        f"per_camera={{{per_camera}}} "
        f"deferred_retry_cooldown={max(0.0, float(retry_cooldown_elapsed_sec)):.3f}s "
        f"retry_per_camera={{{retry_per_camera}}} recovered={recovered_text}"
        f"{failure_text}",
    )


def _close_camera_sources(camera_sources: Mapping[str, RealSenseRGBSource]) -> None:
    declaration_items = tuple(camera_sources.items())

    def close_order(
        item: tuple[int, tuple[str, RealSenseRGBSource]],
    ) -> tuple[bool, int, int]:
        declaration_index, (_, source) = item
        sequence = getattr(source, _CAMERA_STARTUP_SEQUENCE_ATTR, None)
        return (
            sequence is not None,
            int(sequence) if sequence is not None else -1,
            declaration_index,
        )

    ordered_items = sorted(
        enumerate(declaration_items),
        key=close_order,
        reverse=True,
    )
    run_best_effort_cleanup(
        "real_teleop.camera_sources.close",
        (
            (f"camera[{camera_key}]", source.close)
            for _, (camera_key, source) in ordered_items
        ),
    )

__all__ = [
    "CAMERA_DEFERRED_RETRY_COOLDOWN_SEC",
    "CAMERA_STARTUP_OPERATION_TIMEOUT_SEC",
    "CAMERA_STARTUP_SELF_CHECK_CONSECUTIVE_FRESH_SAMPLES",
    "CAMERA_STARTUP_SELF_CHECK_DURATION_SEC",
    "CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC",
    "CAMERA_STARTUP_SELF_CHECK_INTERVAL_SEC",
    "CAMERA_START_STAGGER_SEC",
    "CameraStartupFreshnessGate",
    "apply_real_teleop_camera_history_defaults",
    "_build_camera_sources",
    "_camera_startup_items",
    "_close_camera_sources",
    "_select_preview_camera_sources",
    "_start_camera_sources",
    "_start_camera_sources_best_effort",
    "apply_real_teleop_camera_startup_defaults",
]
