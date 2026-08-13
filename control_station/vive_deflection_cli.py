"""CLI for the standalone VIVE deflection viewer."""

from __future__ import annotations

import argparse
import json
import time

from .vive_deflection import ViveDeflectionConfig, ViveDeflectionController
from .vive_deflection_calibration import ViveDeflectionCalibrationLauncher
from .vive_deflection_gui import ViveDeflectionGui


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m control_station vive-deflection",
        description=(
            "Read-only VIVE Home-deflection viewer. It creates no simulator, "
            "policy, HIL composition, authority, recorder, or robot target."
        ),
    )
    parser.add_argument("--source-role", choices=("left_wrist", "right_wrist"), default="left_wrist")
    parser.add_argument("--tracker-name", default="tracker_1")
    parser.add_argument("--legacy-index", type=int, default=0)
    parser.add_argument("--vive-host", default="0.0.0.0")
    parser.add_argument("--vive-port", type=int, default=9999)
    parser.add_argument("--vive-clock-sync-port", type=int, default=10001)
    parser.add_argument("--freshness-sec", type=float, default=0.5)
    coordinate_group = parser.add_mutually_exclusive_group()
    coordinate_group.add_argument(
        "--vive-coordinate-profile",
        default="latest",
        help=(
            "Calibrated VIVE coordinate profile name/path; default: latest "
            "profile compatible with --source-role"
        ),
    )
    coordinate_group.add_argument(
        "--raw-vive-frame",
        action="store_true",
        help="Debug only: display uncalibrated vive_world axes",
    )
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=8790)
    parser.add_argument(
        "--calibration-ui-port",
        type=int,
        default=0,
        help="Loopback calibration-page port; 0 selects a free port",
    )
    parser.add_argument("--no-open-browser", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ViveDeflectionConfig(
        source_role=args.source_role,
        tracker_name=args.tracker_name,
        legacy_index=args.legacy_index,
        vive_host=args.vive_host,
        vive_port=args.vive_port,
        vive_clock_sync_port=args.vive_clock_sync_port,
        freshness_sec=args.freshness_sec,
        coordinate_profile_path=(
            None if args.raw_vive_frame else args.vive_coordinate_profile
        ),
    )
    if args.dry_run:
        print(json.dumps({
            "application": "control_station_vive_deflection_viewer",
            "source_role": config.source_role,
            "vive_port": config.vive_port,
            "vive_coordinate_profile": (
                None
                if config.coordinate_profile_path is None
                else str(config.coordinate_profile_path)
            ),
            "device_start_skipped": True,
            "motion_authorized": False,
            "simulator_created": False,
            "composition_created": False,
            "recalibration_available": True,
        }, ensure_ascii=False, sort_keys=True))
        return 0
    controller = ViveDeflectionController(config)
    calibration_launcher = ViveDeflectionCalibrationLauncher(
        source=controller.source,
        source_role=config.source_role,
        tracker_name=config.tracker_name,
        ui_host=args.ui_host,
        ui_port=args.calibration_ui_port,
        profile_applier=controller.apply_coordinate_profile,
    )
    gui = ViveDeflectionGui(
        controller,
        host=args.ui_host,
        port=args.ui_port,
        open_browser=not args.no_open_browser,
        calibration_launcher=calibration_launcher,
    )
    try:
        controller.start()
        gui.start()
        print(json.dumps({"url": gui.url, "source_role": config.source_role, "motion_authorized": False}, ensure_ascii=False), flush=True)
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        return 0
    finally:
        gui.close()
        calibration_launcher.close()
        controller.close()


__all__ = ["build_arg_parser", "main"]
