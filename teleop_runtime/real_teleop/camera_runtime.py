"""RealSense camera construction and lifecycle helpers for real teleoperation."""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from pathlib import Path

from hardware.realsense.color import normalize_color_format
from robot_motion.runtime import run_best_effort_cleanup

from ..gui.startup_progress import GuiStartupProgress
from ..log_utils import log_event as _log
from ..sources.camera import RealSenseRGBSource, load_realsense_camera_config_map
from .wrist_depth_freshness import (
    DEFAULT_WRIST_DEPTH_SAVE_FPS,
    DEFAULT_WRIST_DEPTH_STREAM_FPS,
    WRIST_DEPTH_CAMERA_KEYS,
)


CAMERA_START_STAGGER_SEC = 0.5


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
    requested_color_format = normalize_color_format(
        str(getattr(args, "camera_color_format", "bgr8") or "bgr8").strip().lower()
    )
    for camera_key, config in camera_configs.items():
        configured_color_format = str(config.color_format)
        if configured_color_format != requested_color_format:
            config.metadata = dict(config.metadata)
            config.metadata["configured_color_format"] = configured_color_format
            config.metadata["real_teleop_color_format_override"] = requested_color_format
            _log(
                "Camera",
                f"{camera_key} color transport overridden for real teleop: "
                f"{configured_color_format} -> {requested_color_format}",
            )
            config.color_format = requested_color_format
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


def _start_camera_sources(
    camera_sources: Mapping[str, RealSenseRGBSource],
    *,
    stagger_sec: float = CAMERA_START_STAGGER_SEC,
    startup_progress: GuiStartupProgress | None = None,
) -> None:
    for index, (camera_key, source) in enumerate(camera_sources.items()):
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
            _log("Camera", f"waiting {float(stagger_sec):.2f}s before starting camera {camera_key}")
            time.sleep(float(stagger_sec))
            if startup_progress is not None:
                startup_progress.update(progress_key, message="正在启动视频流")
        _log("Camera", f"starting camera {camera_key}")
        start_started = time.perf_counter()
        try:
            source.start()
        except Exception as exc:
            if startup_progress is not None:
                startup_progress.fail(
                    progress_key,
                    f"{type(exc).__name__}: {exc}",
                )
            raise
        else:
            _log(
                "Camera",
                f"camera {camera_key} video stream ready; "
                f"source_start_elapsed={(time.perf_counter() - start_started):.3f}s "
                f"stages={getattr(source, 'startup_timing', {})}",
            )
            if startup_progress is not None:
                startup_progress.succeed(progress_key, message="视频流已启动")


def _start_camera_sources_best_effort(
    camera_sources: Mapping[str, RealSenseRGBSource],
    *,
    stagger_sec: float = CAMERA_START_STAGGER_SEC,
    startup_progress: GuiStartupProgress | None = None,
) -> dict[str, str]:
    """Attempt every configured camera without making cameras a GUI gate."""
    errors: dict[str, str] = {}
    for index, (camera_key, source) in enumerate(camera_sources.items()):
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
            _log("Camera", f"waiting {float(stagger_sec):.2f}s before starting camera {camera_key}")
            time.sleep(float(stagger_sec))
            if startup_progress is not None:
                startup_progress.update(progress_key, message="正在启动视频流")
        _log("Camera", f"starting camera {camera_key}")
        start_started = time.perf_counter()
        try:
            source.start()
        except Exception as exc:  # noqa: BLE001 - optional-device admission records failure.
            detail = f"{type(exc).__name__}: {exc}"
            errors[str(camera_key)] = detail
            _log("Camera", f"WARNING: camera {camera_key} unavailable: {detail}")
            if startup_progress is not None:
                startup_progress.fail(progress_key, detail)
        else:
            _log(
                "Camera",
                f"camera {camera_key} video stream ready; "
                f"source_start_elapsed={(time.perf_counter() - start_started):.3f}s "
                f"stages={getattr(source, 'startup_timing', {})}",
            )
            if startup_progress is not None:
                startup_progress.succeed(progress_key, message="视频流已启动")
    return errors


def _close_camera_sources(camera_sources: Mapping[str, RealSenseRGBSource]) -> None:
    run_best_effort_cleanup(
        "real_teleop.camera_sources.close",
        (
            (f"camera[{camera_key}]", source.close)
            for camera_key, source in reversed(tuple(camera_sources.items()))
        ),
    )

__all__ = [
    "CAMERA_START_STAGGER_SEC",
    "_build_camera_sources",
    "_close_camera_sources",
    "_select_preview_camera_sources",
    "_start_camera_sources",
    "_start_camera_sources_best_effort",
]
