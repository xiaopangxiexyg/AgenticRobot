#!/usr/bin/env python3
"""Capture and persist unloaded DM-Tac base frames from real sensors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

import _bootstrap  # noqa: F401


RAW_SHAPE = (480, 640)


def _parse_dev_id(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _validated_raw_array(value: Any, sensor_key: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != RAW_SHAPE or array.dtype != np.uint8:
        raise RuntimeError(
            f"Invalid raw frame for {sensor_key!r}: expected uint8 {RAW_SHAPE}, "
            f"got {array.dtype} {array.shape}"
        )
    return np.ascontiguousarray(array)


def _raw_array(frame: Any, sensor_key: Any) -> np.ndarray:
    value = None if frame is None else getattr(frame, "raw", None)
    return _validated_raw_array(getattr(value, "img", value), sensor_key)


def _confirm_unloaded(*, assume_yes: bool) -> None:
    warning = (
        "All selected DM-Tac surfaces must be completely unloaded and stationary. "
        "This operation replaces their persistent fixed base-frame calibration."
    )
    print(f"WARNING: {warning}", file=sys.stderr)
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise RuntimeError("Interactive confirmation is unavailable; inspect the setup and pass --yes")
    answer = input("Type CALIBRATE to continue: ").strip()
    if answer != "CALIBRATE":
        raise RuntimeError("Calibration cancelled")


def _check_stability(
    sensor: str,
    metrics: Mapping[str, float],
    *,
    max_mean: float,
    max_p99: float,
) -> None:
    failures = []
    if max_mean > 0 and metrics["mean_abs_deviation"] > max_mean:
        failures.append(f"mean {metrics['mean_abs_deviation']:.3f} > {max_mean:.3f}")
    if max_p99 > 0 and metrics["p99_abs_deviation"] > max_p99:
        failures.append(f"p99 {metrics['p99_abs_deviation']:.3f} > {max_p99:.3f}")
    if failures:
        raise RuntimeError(f"Sensor {sensor} is not stable enough: " + ", ".join(failures))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture stable unloaded raw frames from real DM-Tac sensors and save fixed calibration."
    )
    parser.add_argument("--dev-ids", nargs="*", help="Sensor serials; omit to auto-discover all DM-Tac sensors.")
    parser.add_argument("--frames", type=int, default=30, help="Raw frames used for the pixelwise median.")
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=500)
    parser.add_argument("--ready-timeout-s", type=float, default=10.0)
    parser.add_argument("--backend", choices=("cpu", "cuda", "Flux", "flux"), default="cpu")
    parser.add_argument("--mode", choices=("standard", "high"), default="standard")
    parser.add_argument("--max-fps", type=int, default=30)
    parser.add_argument("--sequential", action="store_true", help="Read sensors sequentially instead of in parallel.")
    parser.add_argument("--calibration-dir", type=Path, default=None)
    parser.add_argument("--max-mean-deviation", type=float, default=3.0)
    parser.add_argument("--max-p99-deviation", type=float, default=12.0)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing calibration for a serial.")
    parser.add_argument("--yes", action="store_true", help="Confirm unloaded sensors non-interactively.")
    parser.add_argument("--show-sdk-log", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be non-negative")
    os.environ["DMTAC_SUPPRESS_SDK_OUTPUT"] = "0" if args.show_sdk_log else "1"

    from portable_dm_tac import DMTacMultiDevice, discover_device_ids
    from portable_dmtac_gripper.device import TactileConfig, tactile_device_config
    from portable_dmtac_gripper.tactile_calibration import (
        build_stable_base_frame,
        default_calibration_dir,
        load_tactile_calibration,
        save_tactile_calibration,
    )

    dev_ids = [_parse_dev_id(value) for value in (args.dev_ids or [])]
    if not dev_ids:
        dev_ids = list(discover_device_ids(prefer_serial=True))
    if not dev_ids:
        raise RuntimeError("No DM-Tac sensors discovered; check USB/video permissions or pass --dev-ids")
    if any(isinstance(value, int) for value in dev_ids):
        raise ValueError("Fixed calibration requires sensor serials, not SDK numeric indices")
    sensor_keys = [str(value) for value in dev_ids]
    calibration_dir = Path(args.calibration_dir or default_calibration_dir()).expanduser().resolve()
    existing = [
        sensor
        for sensor in sensor_keys
        if load_tactile_calibration(calibration_dir, sensor, required=False) is not None
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Calibration already exists for {existing}; inspect it and pass --overwrite to replace"
        )

    print("Selected sensors: " + ", ".join(sensor_keys))
    print(f"Calibration directory: {calibration_dir}")
    _confirm_unloaded(assume_yes=bool(args.yes))

    tactile_cfg = TactileConfig(
        dev_ids=dev_ids,
        backend=str(args.backend),
        mode=str(args.mode),
        max_fps=int(args.max_fps),
        fields=("raw",),
        compute_profile="dist-only",
        parallel_read=not bool(args.sequential),
        calibration_policy="off",
        persist_reset_calibration=False,
    )
    configs = [tactile_device_config(dev_id, tactile_cfg) for dev_id in dev_ids]
    reader = DMTacMultiDevice(configs)
    captured: dict[str, list[np.ndarray]] = {sensor: [] for sensor in sensor_keys}
    try:
        reader.connect()
        ready = reader.wait_until_ready(timeout_s=float(args.ready_timeout_s))
        not_ready = [str(key) for key, value in ready.items() if not value]
        if not_ready:
            raise RuntimeError(f"DM-Tac sensors not ready: {not_ready}")
        total = int(args.warmup_frames) + int(args.frames)
        for index in range(total):
            frames = reader.read_all(
                timeout_ms=int(args.timeout_ms),
                wait=True,
                require_ready=True,
                strict=True,
                parallel=not bool(args.sequential),
            )
            if index < int(args.warmup_frames):
                continue
            for dev_id, sensor in zip(dev_ids, sensor_keys):
                captured[sensor].append(_raw_array(frames.get(dev_id), sensor))
            print(f"Captured {index - int(args.warmup_frames) + 1}/{args.frames}", end="\r", flush=True)
        print()

        bases: dict[str, np.ndarray] = {}
        metrics_by_sensor: dict[str, dict[str, float]] = {}
        for sensor in sensor_keys:
            base, metrics = build_stable_base_frame(captured[sensor])
            _check_stability(
                sensor,
                metrics,
                max_mean=float(args.max_mean_deviation),
                max_p99=float(args.max_p99_deviation),
            )
            bases[sensor] = base
            metrics_by_sensor[sensor] = metrics

        reader.set_base_frames({dev_id: bases[sensor] for dev_id, sensor in zip(dev_ids, sensor_keys)})
        summary: dict[str, Any] = {}
        for sensor in sensor_keys:
            saved = save_tactile_calibration(
                calibration_dir,
                sensor,
                bases[sensor],
                source={
                    "kind": "live_sensor_median_capture",
                    "frames": int(args.frames),
                    "warmup_frames": int(args.warmup_frames),
                    "backend": str(args.backend),
                    "mode": str(args.mode),
                    "stability": metrics_by_sensor[sensor],
                },
            )
            verified = load_tactile_calibration(calibration_dir, sensor, required=True)
            if verified is None or verified.sha256 != saved.sha256:
                raise RuntimeError(f"Calibration verification failed for {sensor}")
            summary[sensor] = {
                "sha256": saved.sha256,
                "metadata_path": str(saved.metadata_path),
                "frame_path": str(saved.frame_path),
                "stability": metrics_by_sensor[sensor],
            }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
