"""Camera source construction and startup lifecycle for policy deployment.

Policy-artifact camera input binding is implemented in
:mod:`camera_input_contract` and re-exported here as the deployment facade.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from pathlib import Path

from ..real_teleop.camera_runtime import (
    CAMERA_START_STAGGER_SEC as REAL_TELEOP_CAMERA_START_STAGGER_SEC,
)
from ..real_teleop.camera_runtime import (
    CAMERA_STARTUP_OPERATION_TIMEOUT_SEC as REAL_TELEOP_CAMERA_STARTUP_OPERATION_TIMEOUT_SEC,
)
from ..real_teleop.camera_runtime import (
    CAMERA_STARTUP_SELF_CHECK_CONSECUTIVE_FRESH_SAMPLES,
    CAMERA_STARTUP_SELF_CHECK_DURATION_SEC,
    CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC,
    CAMERA_STARTUP_SELF_CHECK_INTERVAL_SEC,
    CameraStartupFreshnessGate,
    apply_real_teleop_camera_startup_defaults,
)
from ..real_teleop.camera_runtime import (
    _close_camera_sources as _close_real_teleop_camera_sources,
)
from ..real_teleop.camera_runtime import (
    _start_camera_sources_best_effort as _start_real_teleop_camera_sources_best_effort,
)
from ..real_teleop.status_payloads import _build_camera_status
from ..sources.camera import (
    REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
    RealSenseRGBCameraConfig,
    RealSenseRGBSource,
    load_realsense_camera_config_map,
)
from .camera_input_contract import resolve_policy_camera_input_keys

DEFAULT_DEPLOYMENT_CAMERA_FRAME_MAX_AGE_SEC = 0.2
DEFAULT_DEPLOYMENT_CAMERA_KEEP_LAST = 4

# This is an operator hint, not an automatic downgrade or a recovery action.
_CAMERA_STREAM_RECOVERY_HINT = (
    "Camera recovery guidance: a started stream may still deliver no frames. "
    "A bounded clean restart can recover it, but cannot guarantee both streams. "
    "RGB-D tasks require fresh color and depth together; a color-only frame is "
    "not sufficient. If the failure persists, inspect the camera's USB "
    "connection and reconnect it manually. Only tasks explicitly configured "
    "with enable_depth=false may continue as RGB-only."
)


def build_camera_sources(
    args: argparse.Namespace,
    *,
    selected_camera_keys: tuple[str, ...] | None = None,
) -> dict[str, RealSenseRGBSource]:
    if args.camera_mode == "none":
        return {}
    if selected_camera_keys == ():
        raise ValueError(
            "A policy artifact without camera inputs requires --camera-mode none."
        )
    if args.camera_serial:
        if args.camera_mode != "single":
            raise ValueError(
                "--camera-serial can only be used with --camera-mode single."
            )
        config = RealSenseRGBCameraConfig(
            serial=str(args.camera_serial),
            width=int(args.camera_width),
            height=int(args.camera_height),
            fps=int(args.camera_fps),
            color_format=str(getattr(args, "camera_color_format", "bgr8")),
            camera_key=str(args.camera_key),
            keep_last=int(args.camera_keep_last),
            color_space=str(args.camera_color_space),
            wait_timeout_ms=int(args.camera_wait_timeout_ms),
            stale_after_sec=float(args.camera_stale_after_sec),
            auto_exposure_priority=args.camera_auto_exposure_priority,
            startup_hardware_reset=bool(
                getattr(args, "camera_startup_hardware_reset", False)
            ),
            startup_warmup_policy=str(
                getattr(
                    args,
                    "camera_startup_warmup_policy",
                    REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
                )
            ),
            metadata={"source": "teleop_runtime_deploy_cli"},
        )
        if (
            selected_camera_keys is not None
            and config.camera_key not in selected_camera_keys
        ):
            raise ValueError(
                f"--camera-key {config.camera_key!r} is not selected by policy artifact "
                f"{list(selected_camera_keys)}."
            )
        apply_real_teleop_camera_startup_defaults(
            {config.camera_key: config},
            color_format=str(getattr(args, "camera_color_format", "bgr8")),
            warmup_policy=str(
                getattr(
                    args,
                    "camera_startup_warmup_policy",
                    REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
                )
            ),
        )
        return {config.camera_key: RealSenseRGBSource(config)}
    camera_configs = load_realsense_camera_config_map(Path(args.camera_config))
    if selected_camera_keys is not None:
        missing_keys = tuple(
            key for key in selected_camera_keys if key not in camera_configs
        )
        if missing_keys:
            raise ValueError(
                "--camera-config does not provide policy artifact camera inputs: "
                f"{list(missing_keys)}."
            )
        selected = set(selected_camera_keys)
        camera_configs = {
            key: config for key, config in camera_configs.items() if key in selected
        }
    real_camera_configs = {
        key: config
        for key, config in camera_configs.items()
        if isinstance(config, RealSenseRGBCameraConfig)
    }
    apply_real_teleop_camera_startup_defaults(
        real_camera_configs,
        color_format=str(getattr(args, "camera_color_format", "bgr8")),
        warmup_policy=str(
            getattr(
                args,
                "camera_startup_warmup_policy",
                REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
            )
        ),
    )
    # Config files historically omitted keep_last, which left deployment with
    # the source default of two frames even though --camera-keep-last already
    # defaulted to four for the single-camera CLI path.  Apply the deployment
    # setting to config-backed sources as a minimum while preserving a larger
    # value explicitly requested by a config file.
    requested_keep_last = int(
        getattr(args, "camera_keep_last", DEFAULT_DEPLOYMENT_CAMERA_KEEP_LAST)
    )
    for config in camera_configs.values():
        if isinstance(config, RealSenseRGBCameraConfig):
            config.keep_last = max(int(config.keep_last), requested_keep_last)
    if bool(getattr(args, "camera_startup_hardware_reset", False)):
        for config in camera_configs.values():
            config.startup_hardware_reset = True
    if args.camera_mode == "single" and len(camera_configs) != 1:
        raise ValueError(
            f"--camera-mode single expects exactly one camera config, got {len(camera_configs)}."
        )
    if args.camera_mode == "multi" and not camera_configs:
        raise ValueError("--camera-mode multi requires at least one camera config.")
    return {key: RealSenseRGBSource(config) for key, config in camera_configs.items()}


def start_camera_sources(
    camera_sources: Mapping[str, RealSenseRGBSource],
    *,
    stagger_sec: float = REAL_TELEOP_CAMERA_START_STAGGER_SEC,
    startup_timeout_sec: float = REAL_TELEOP_CAMERA_STARTUP_OPERATION_TIMEOUT_SEC,
) -> None:
    """Use the formal real-teleop collection camera startup lifecycle."""

    errors = _start_real_teleop_camera_sources_best_effort(
        camera_sources,
        stagger_sec=float(stagger_sec),
        startup_timeout_sec=float(startup_timeout_sec),
    )
    if errors:
        detail = "; ".join(
            f"{camera_key}={errors[camera_key]}" for camera_key in sorted(errors)
        )
        raise RuntimeError(
            "RealSense startup failed after the formal real-teleop deferred "
            f"retry lifecycle. {_CAMERA_STREAM_RECOVERY_HINT} {detail}"
        )


def close_camera_sources(
    camera_sources: Mapping[str, RealSenseRGBSource],
) -> None:
    """Use the formal real-teleop reverse-order camera shutdown lifecycle."""

    _close_real_teleop_camera_sources(camera_sources)


def run_camera_startup_self_check(
    camera_sources: Mapping[str, RealSenseRGBSource],
    *,
    duration_sec: float = CAMERA_STARTUP_SELF_CHECK_DURATION_SEC,
    interval_sec: float = CAMERA_STARTUP_SELF_CHECK_INTERVAL_SEC,
) -> None:
    """Run the formal collection consecutive-fresh-frame admission headlessly."""

    if not camera_sources:
        return
    started_at = time.time()
    duration_sec = max(0.0, float(duration_sec))
    interval_sec = max(0.01, float(interval_sec))
    freshness_gate = CameraStartupFreshnessGate(
        tuple(str(camera_key) for camera_key in camera_sources)
    )
    final_status: Mapping[str, object] = {}
    while True:
        now = time.time()
        final_status = _build_camera_status(camera_sources, now=now)
        if freshness_gate.update(final_status):
            print(
                "[CameraSelfCheck] formal real-teleop admission passed: "
                f"cameras={sorted(camera_sources)} "
                f"consecutive={CAMERA_STARTUP_SELF_CHECK_CONSECUTIVE_FRESH_SAMPLES} "
                f"max_age_sec={CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC:.3f}",
                flush=True,
            )
            return
        elapsed_sec = now - started_at
        if elapsed_sec >= duration_sec:
            status_detail = "; ".join(
                f"{camera_key}={final_status.get(camera_key)}"
                for camera_key in sorted(camera_sources)
            )
            raise RuntimeError(
                "RealSense formal real-teleop startup self-check failed: "
                f"{_CAMERA_STREAM_RECOVERY_HINT} "
                f"elapsed={elapsed_sec:.3f}s "
                f"streaks={freshness_gate.fresh_frame_streaks}; {status_detail}"
            )
        sleep_sec = min(interval_sec, max(0.0, duration_sec - elapsed_sec))
        if sleep_sec > 0.0:
            time.sleep(sleep_sec)


def select_preview_camera_sources(
    camera_sources: dict[str, RealSenseRGBSource], requested_key: str
) -> dict[str, RealSenseRGBSource]:
    if not camera_sources:
        return {}
    key = str(requested_key or "").strip()
    if key:
        if key not in camera_sources:
            available = ", ".join(sorted(camera_sources))
            raise ValueError(
                f"--vis-cam-key {key!r} not found. Available camera keys: {available}"
            )
        return {key: camera_sources[key]}
    selected_key = "front" if "front" in camera_sources else min(camera_sources)
    return {selected_key: camera_sources[selected_key]}


__all__ = [
    "DEFAULT_DEPLOYMENT_CAMERA_FRAME_MAX_AGE_SEC",
    "DEFAULT_DEPLOYMENT_CAMERA_KEEP_LAST",
    "REAL_TELEOP_CAMERA_STARTUP_OPERATION_TIMEOUT_SEC",
    "REAL_TELEOP_CAMERA_START_STAGGER_SEC",
    "build_camera_sources",
    "close_camera_sources",
    "resolve_policy_camera_input_keys",
    "run_camera_startup_self_check",
    "select_preview_camera_sources",
    "start_camera_sources",
]
