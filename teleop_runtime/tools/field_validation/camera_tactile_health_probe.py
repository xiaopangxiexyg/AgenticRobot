"""No-motion RealSense/DMTAC health probe for an explicitly authorized host.

This script never creates an arm connection or a gripper transport.  It only
opens configured camera streams and tactile sensors, then closes them in a
``finally`` block.  ``--exercise-recovery`` stops one camera pipeline and one
owned tactile child process to verify the runtime supervisors; it does not
reset USB devices or capture a new tactile base frame.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("could not locate the DexTeleop repository root")


PROJECT_ROOT = _repository_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teleop_runtime.backends.grippers import (
    DMTacGripperBackend,
    DMTacGripperBackendConfig,
)
from teleop_runtime.configs.dmtac_hardware import load_dmtac_hardware_assignment
from teleop_runtime.real_teleop.camera_runtime import (
    CAMERA_STARTUP_OPERATION_TIMEOUT_SEC,
    _close_camera_sources,
    _start_camera_sources_best_effort,
)
from teleop_runtime.sources.camera.realsense_rgb import (
    RealSenseRGBSource,
    _load_pyrealsense2,
    load_realsense_camera_config_map,
)


def _tactile_ids_by_side() -> dict[str, tuple[str | int, ...]]:
    assignment = load_dmtac_hardware_assignment(
        "latest",
        validate_current_usb=False,
    )
    from hardware.DMTAC_gripper.portable_dmtac_gripper import usb_assignment

    return {
        side: tuple(usb_assignment.tactile_ids_for_side(assignment, side))
        for side in ("left", "right")
    }


def _fresh_camera(source: RealSenseRGBSource, *, max_age_sec: float = 0.5) -> bool:
    info = source.device_info()
    age = info.get("last_frame_age_sec")
    return bool(age is not None and float(age) <= max_age_sec and not info.get("last_error"))


def _fresh_tactile(backend: DMTacGripperBackend, *, max_age_sec: float = 0.5) -> bool:
    state = backend.get_state(include_tactile_vector=False)
    timestamp = state.get("tactile_timestamp")
    return bool(
        timestamp is not None
        and time.time() - float(timestamp) <= max_age_sec
        and state.get("tactile_dict")
        and state.get("tactile_worker_alive")
    )


def _fresh_solved_tactile(
    backend: DMTacGripperBackend,
    *,
    max_age_sec: float = 0.5,
) -> bool:
    if not _fresh_tactile(backend, max_age_sec=max_age_sec):
        return False
    state = backend.get_state(
        include_tactile_vector=False,
        copy_tactile_arrays=False,
    )
    tactile_dict = state.get("tactile_dict") or {}
    field_by_key = state.get("tactile_field_by_key") or {}
    solved_keys = [
        str(key)
        for key, field in field_by_key.items()
        if str(field) == "distributed_force" and key in tactile_dict
    ]
    return bool(
        solved_keys
        and all(
            np.asarray(tactile_dict[key]).ndim == 3
            and np.asarray(tactile_dict[key]).shape[-1] >= 3
            for key in solved_keys
        )
    )


def _wait_until(predicate: Any, *, timeout_sec: float) -> bool:
    deadline = time.monotonic() + float(timeout_sec)
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


def _camera_profiles() -> dict[str, Any]:
    rs = _load_pyrealsense2()
    result: dict[str, Any] = {}
    for device in rs.context().query_devices():
        serial = str(device.get_info(rs.camera_info.serial_number))
        name = str(device.get_info(rs.camera_info.name))
        color_profiles: set[tuple[int, int, int, str]] = set()
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                try:
                    if profile.stream_type() != rs.stream.color:
                        continue
                    video = profile.as_video_stream_profile()
                    color_profiles.add(
                        (
                            int(video.width()),
                            int(video.height()),
                            int(profile.fps()),
                            str(profile.format()),
                        )
                    )
                except Exception:  # noqa: BLE001,S112 - skip malformed native profiles.
                    continue
        result[serial] = {
            "name": name,
            "color_profiles": [
                {
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "format": color_format,
                }
                for width, height, fps, color_format in sorted(color_profiles)
            ],
        }
    return result


def _camera_profiles_worker(connection: Any) -> None:
    try:
        connection.send(("ok", _camera_profiles()))
    except BaseException as exc:  # noqa: BLE001 - report child failure to parent.
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        except BaseException:  # noqa: BLE001
            return
    finally:
        connection.close()


def _camera_profiles_bounded(*, timeout_sec: float) -> dict[str, Any]:
    """Enumerate profiles outside the probe process's native SDK state.

    ``context.query_devices`` has no Python timeout and can block in the USB
    driver. A short-lived spawned worker keeps that wait from holding the
    health probe's main process or its later camera lifecycle locks.
    """

    normalized_timeout_sec = float(timeout_sec)
    if not math.isfinite(normalized_timeout_sec) or normalized_timeout_sec < 0.0:
        raise ValueError("timeout_sec must be finite and >= 0.")
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_camera_profiles_worker,
        args=(child_connection,),
        name="realsense_camera_profiles",
        daemon=True,
    )
    process_started = False
    try:
        process.start()
        process_started = True
    except BaseException:
        parent_connection.close()
        raise
    finally:
        child_connection.close()
    try:
        process.join(timeout=normalized_timeout_sec)
        if process.is_alive():
            raise TimeoutError(
                "camera profile enumeration exceeded "
                f"{normalized_timeout_sec:.3f}s; child native SDK termination "
                "was requested"
            )
        if not parent_connection.poll():
            raise RuntimeError(
                "camera profile enumeration worker exited without a result "
                f"(exitcode={process.exitcode})"
            )
        status, payload = parent_connection.recv()
        if status == "error":
            raise RuntimeError(str(payload))
        if status != "ok" or not isinstance(payload, dict):
            raise RuntimeError("camera profile enumeration returned an invalid result")
        return payload
    finally:
        parent_connection.close()
        if process_started:
            try:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=min(1.0, max(0.1, normalized_timeout_sec)))
                    if process.is_alive() and hasattr(process, "kill"):
                        process.kill()
                        process.join(timeout=0.1)
                if not process.is_alive():
                    process.close()
            except BaseException:  # noqa: BLE001
                # The parent is already on a bounded failure path; do not let
                # best-effort child cleanup hide the original result.
                process_started = False


def _run_probe() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument(
        "--camera-operation-timeout-sec",
        type=float,
        default=CAMERA_STARTUP_OPERATION_TIMEOUT_SEC,
        help=(
            "Bound each native camera/profile operation; a timeout fails closed "
            "without an immediate retry or hardware reset."
        ),
    )
    parser.add_argument(
        "--soak-sec",
        type=float,
        default=0.0,
        help="Keep healthy workers running for this long and report observed producer rates.",
    )
    parser.add_argument("--exercise-recovery", action="store_true")
    parser.add_argument(
        "--exercise-gui-solving",
        action="store_true",
        help="Start with raw tactile, then exercise the GUI calibrated-solving worker restart.",
    )
    parser.add_argument("--camera-format", choices=("configured", "bgr8", "yuyv"), default="configured")
    parser.add_argument("--camera-key", choices=("all", "front", "left_wrist", "right_wrist"), default="all")
    parser.add_argument("--camera-width", type=int)
    parser.add_argument("--camera-height", type=int)
    parser.add_argument("--camera-fps", type=int)
    parser.add_argument(
        "--allow-camera-hardware-reset",
        action="store_true",
        help="Allow the production startup retry to reset only the selected RealSense device.",
    )
    parser.add_argument("--skip-camera", action="store_true")
    parser.add_argument("--skip-tactile", action="store_true")
    parser.add_argument("--list-camera-profiles", action="store_true")
    args = parser.parse_args()

    if args.list_camera_profiles:
        print(
            json.dumps(
                _camera_profiles_bounded(
                    timeout_sec=float(args.camera_operation_timeout_sec)
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    camera_configs = load_realsense_camera_config_map(
        Path("teleop_runtime/configs/devices/realsense.json")
    )
    if args.camera_format != "configured":
        for config in camera_configs.values():
            config.color_format = str(args.camera_format)
    for config in camera_configs.values():
        if args.camera_width is not None:
            config.width = int(args.camera_width)
        if args.camera_height is not None:
            config.height = int(args.camera_height)
        if args.camera_fps is not None:
            config.fps = int(args.camera_fps)
    if args.camera_key != "all":
        camera_configs = {str(args.camera_key): camera_configs[str(args.camera_key)]}
    cameras = (
        {}
        if args.skip_camera
        else {key: RealSenseRGBSource(config) for key, config in camera_configs.items()}
    )
    tactile_ids = {} if args.skip_tactile else _tactile_ids_by_side()
    tactile = {
        side: DMTacGripperBackend(
            DMTacGripperBackendConfig(
                side=side,
                connect_hardware=False,
                connect_tactile=True,
                enable_telemetry=True,
                tactile_ids=ids,
                tactile_fields=(
                    ("raw",)
                    if args.exercise_gui_solving
                    else ("distributed_force",)
                ),
                tactile_compute_profile="dist-only",
                tactile_calibration_policy=(
                    "auto" if args.exercise_gui_solving else "required"
                ),
                tactile_read_hz=15.0,
                tactile_max_fps=30,
                tactile_process_read=True,
                tactile_reset_on_start=False,
                tactile_warmup_frames=0,
            )
        )
        for side, ids in tactile_ids.items()
        if ids
    }
    for config in camera_configs.values():
        config.startup_reset = False
        config.startup_hardware_reset = False
        config.startup_hardware_reset_on_failure = bool(
            args.allow_camera_hardware_reset
        )
        config.startup_hardware_reset_on_warmup_failure = bool(
            args.allow_camera_hardware_reset
        )
        config.startup_hardware_reset_retry_attempts = (
            1 if args.allow_camera_hardware_reset else 0
        )
        config.startup_warmup_frames = 10

    summary: dict[str, Any] = {"camera": {}, "tactile": {}, "recovery": {}}
    try:
        camera_startup_errors = _start_camera_sources_best_effort(
            cameras,
            startup_timeout_sec=float(args.camera_operation_timeout_sec),
        )
        if camera_startup_errors:
            detail = "; ".join(
                f"{camera_key}={camera_startup_errors[camera_key]}"
                for camera_key in sorted(camera_startup_errors)
            )
            raise RuntimeError(
                "RealSense camera startup failed after the guarded serial "
                f"lifecycle: {detail}"
            )
        for backend in tactile.values():
            backend.start()

        ready = _wait_until(
            lambda: all(_fresh_camera(source) for source in cameras.values())
            and all(_fresh_tactile(backend) for backend in tactile.values()),
            timeout_sec=max(2.0, float(args.duration)),
        )
        if args.exercise_gui_solving and ready and tactile:
            changed_by_side = {
                side: bool(backend.enable_tactile_solving())
                for side, backend in tactile.items()
            }
            solved_ready = _wait_until(
                lambda: all(
                    _fresh_solved_tactile(backend)
                    for backend in tactile.values()
                ),
                timeout_sec=max(10.0, float(args.duration)),
            )
            summary["solving"] = {
                "ok": solved_ready,
                "changed_by_side": changed_by_side,
                "fields_by_side": {
                    side: list(backend.config.tactile_fields)
                    for side, backend in tactile.items()
                },
                "calibration_policy_by_side": {
                    side: str(backend.config.tactile_calibration_policy)
                    for side, backend in tactile.items()
                },
            }
            ready = bool(ready and solved_ready)
        if args.exercise_recovery and ready and cameras:
            camera_key = sorted(cameras)[0]
            camera = cameras[camera_key]
            with camera._pipeline_lock:
                pipeline = camera._pipeline
            if pipeline is not None:
                pipeline.stop()
            camera_recovered = _wait_until(
                lambda: int(camera.device_info().get("restart_count") or 0) >= 1
                and _fresh_camera(camera),
                timeout_sec=8.0,
            )
            summary["recovery"]["camera"] = {
                "key": camera_key,
                "ok": camera_recovered,
                "info": camera.device_info(),
            }

        if args.exercise_recovery and ready and tactile:
            side = sorted(tactile)[0]
            backend = tactile[side]
            process = backend._tactile_worker_process
            if process is not None and process.is_alive():
                process.terminate()
            tactile_recovered = _wait_until(
                lambda: int(
                    backend.get_state(include_tactile_arrays=False).get(
                        "tactile_worker_restart_count"
                    )
                    or 0
                )
                >= 1
                and _fresh_tactile(backend),
                timeout_sec=10.0,
            )
            summary["recovery"]["tactile"] = {
                "side": side,
                "ok": tactile_recovered,
                "state": backend.get_state(include_tactile_arrays=False),
            }

        soak_sec = max(0.0, float(args.soak_sec))
        if ready and soak_sec > 0.0:
            camera_counts_before = {
                key: int(source.device_info().get("frame_count") or 0)
                for key, source in cameras.items()
            }
            tactile_versions_before = {
                side: int(
                    backend.get_state(include_tactile_arrays=False).get(
                        "tactile_version"
                    )
                    or 0
                )
                for side, backend in tactile.items()
            }
            soak_started = time.monotonic()
            time.sleep(soak_sec)
            soak_elapsed = max(1e-9, time.monotonic() - soak_started)
            summary["soak"] = {
                "elapsed_sec": soak_elapsed,
                "camera_hz": {
                    key: (
                        int(source.device_info().get("frame_count") or 0)
                        - camera_counts_before[key]
                    )
                    / soak_elapsed
                    for key, source in cameras.items()
                },
                "tactile_hz": {
                    side: (
                        int(
                            backend.get_state(include_tactile_arrays=False).get(
                                "tactile_version"
                            )
                            or 0
                        )
                        - tactile_versions_before[side]
                    )
                    / soak_elapsed
                    for side, backend in tactile.items()
                },
            }

        for key, source in cameras.items():
            camera_summary = source.device_info()
            frame = source.frame_buffer().get_latest()
            if frame is not None:
                image = np.asarray(frame.image)
                sample = image[
                    :: max(1, image.shape[0] // 32),
                    :: max(1, image.shape[1] // 32),
                ]
                camera_summary["sampled_pixel_min"] = int(np.min(sample))
                camera_summary["sampled_pixel_max"] = int(np.max(sample))
                camera_summary["sampled_pixel_mean"] = float(np.mean(sample))
            summary["camera"][key] = camera_summary
        for side, backend in tactile.items():
            summary["tactile"][side] = backend.get_state(include_tactile_arrays=False)
        summary["ready"] = ready
        summary["ok"] = bool(
            ready
            and all(bool(item.get("ok")) for item in summary["recovery"].values())
            and bool(summary.get("solving", {}).get("ok", True))
        )
        print(json.dumps(summary, ensure_ascii=False, default=str, sort_keys=True))
        return 0 if summary["ok"] else 2
    finally:
        for backend in tactile.values():
            backend.close()
        _close_camera_sources(cameras)


def main() -> int:
    """Run the probe while converting SIGTERM into a finally-safe interrupt."""

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def raise_interrupt(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, raise_interrupt)
    try:
        return _run_probe()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
