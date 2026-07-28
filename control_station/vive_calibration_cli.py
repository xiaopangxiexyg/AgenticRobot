"""CLI adapter for the Control Station VIVE calibration GUI."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .vive_calibration import ViveCoordinateCalibrationConfig, ViveCoordinateCalibrationController
from .vive_calibration_gui import ViveCoordinateCalibrationGui
from .vive_calibration_profiles import (
    VIVE_COORDINATE_PROFILE_KIND,
    normalize_active_sides,
)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m control_station vive-calibration",
        description=(
            "Control Station VIVE coordinate calibration GUI. "
            "Only VIVE input is opened; no robot/sim target or motion execution is created."
        ),
    )
    parser.add_argument("--sides", choices=("left", "right", "both"), default="both")
    parser.add_argument("--desk-tracker", choices=("auto", "left", "right", "auxiliary"), default="auto")
    parser.add_argument("--vive-host", default="0.0.0.0")
    parser.add_argument("--vive-port", type=int, default=9999)
    parser.add_argument("--vive-clock-sync-port", type=int, default=10001)
    parser.add_argument("--left-tracker-name", default="tracker_1")
    parser.add_argument("--right-tracker-name", default="tracker_2")
    parser.add_argument("--auxiliary-tracker-name", default="")
    parser.add_argument("--sweep-capture-sec", type=_positive_float, default=5.0)
    parser.add_argument("--capture-prep-sec", type=_non_negative_float, default=3.0)
    parser.add_argument("--hold-capture-sec", type=_positive_float, default=4.0)
    parser.add_argument("--capture-extra-wait-sec", type=_non_negative_float, default=2.0)
    parser.add_argument("--min-sweep-samples", type=int, default=30)
    parser.add_argument("--min-hold-samples", type=int, default=30)
    parser.add_argument("--min-sweep-extent-m", type=float, default=0.15)
    parser.add_argument("--min-axis-variance-ratio", type=float, default=0.5)
    parser.add_argument("--min-start-end-distance-m", type=float, default=0.08)
    parser.add_argument("--max-axis-correction-deg", type=float, default=20.0)
    parser.add_argument("--virtual-desk-post-yaw-deg", type=float, default=180.0)
    parser.add_argument("--profile-dir", type=Path, default=None)
    parser.add_argument(
        "--ui-host",
        default="127.0.0.1",
        help="Local-only GUI host (127.0.0.1 or localhost).",
    )
    parser.add_argument("--ui-port", type=int, default=0)
    parser.add_argument("--no-open-browser", action="store_true")
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="Open the result viewer without starting the VIVE receiver.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without opening VIVE or the GUI.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ViveCoordinateCalibrationConfig:
    return ViveCoordinateCalibrationConfig(
        active_sides=normalize_active_sides(args.sides),
        desk_tracker=args.desk_tracker,
        vive_host=args.vive_host,
        vive_port=args.vive_port,
        vive_clock_sync_port=args.vive_clock_sync_port,
        left_tracker_name=args.left_tracker_name,
        right_tracker_name=args.right_tracker_name,
        auxiliary_tracker_name=args.auxiliary_tracker_name,
        sweep_capture_sec=args.sweep_capture_sec,
        capture_prep_sec=args.capture_prep_sec,
        hold_capture_sec=args.hold_capture_sec,
        capture_extra_wait_sec=args.capture_extra_wait_sec,
        min_sweep_samples=args.min_sweep_samples,
        min_hold_samples=args.min_hold_samples,
        min_sweep_extent_m=args.min_sweep_extent_m,
        min_axis_variance_ratio=args.min_axis_variance_ratio,
        min_start_end_distance_m=args.min_start_end_distance_m,
        max_axis_correction_deg=args.max_axis_correction_deg,
        virtual_desk_post_yaw_deg=args.virtual_desk_post_yaw_deg,
        profile_dir=args.profile_dir,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = config_from_args(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "profile_kind": VIVE_COORDINATE_PROFILE_KIND,
                    "active_sides": list(config.active_sides),
                    "desk_tracker": config.desk_tracker,
                    "vive_port": config.vive_port,
                    "clock_sync_port": config.vive_clock_sync_port,
                    "profile_dir": str(config.profile_dir),
                    "device_start_skipped": True,
                    "real_arm_motion": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    controller = ViveCoordinateCalibrationController(config)
    gui = ViveCoordinateCalibrationGui(
        controller,
        host=args.ui_host,
        port=args.ui_port,
        open_browser=not args.no_open_browser,
    )
    try:
        controller.start(start_source=not args.view_only)
        gui.start()
        print(
            json.dumps(
                {
                    "url": gui.url,
                    "profile_dir": controller.snapshot()["profile_dir"],
                    "source_started": not args.view_only,
                    "real_arm_motion": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        return 0
    finally:
        gui.close()
        controller.close()


__all__ = ["build_arg_parser", "config_from_args", "main"]
