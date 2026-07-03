from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from ..sources.camera import RealSenseRGBSource, load_realsense_camera_config_map
from .registry import build_camera_registry_snapshot, load_camera_registry


DEFAULT_CAMERA_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "devices" / "realsense.json"
)
DEFAULT_CAMERA_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "cameras" / "realsense_registry.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture RealSense runtime intrinsics into a camera registry snapshot."
    )
    parser.add_argument("--camera-config", default=str(DEFAULT_CAMERA_CONFIG_PATH))
    parser.add_argument("--registry", default=str(DEFAULT_CAMERA_REGISTRY_PATH))
    parser.add_argument(
        "--camera-key",
        action="append",
        default=[],
        help="Camera key to capture. Repeat to capture a subset; default captures all configured cameras.",
    )
    parser.add_argument("--warmup-sec", type=float, default=1.0)
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path. If omitted, the snapshot is printed to stdout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    camera_configs = load_realsense_camera_config_map(args.camera_config)
    requested = {str(key) for key in args.camera_key}
    if requested:
        missing = sorted(requested - set(camera_configs))
        if missing:
            print(
                f"ERROR: unknown camera keys {missing}; available={sorted(camera_configs)}",
                file=sys.stderr,
            )
            return 2
        camera_configs = {
            key: config for key, config in camera_configs.items() if key in requested
        }

    sources = {
        camera_key: RealSenseRGBSource(config)
        for camera_key, config in camera_configs.items()
    }
    try:
        for source in sources.values():
            source.start()
        warmup_sec = max(0.0, float(args.warmup_sec))
        if warmup_sec > 0.0:
            time.sleep(warmup_sec)
        device_info_by_key = {
            camera_key: source.device_info()
            for camera_key, source in sources.items()
        }
        registry = load_camera_registry(args.registry)
        snapshot = build_camera_registry_snapshot(
            registry,
            device_info_by_key=device_info_by_key,
        )
    finally:
        for source in sources.values():
            source.close()

    text = json.dumps(snapshot, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(output_path)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
