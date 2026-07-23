from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .processor import ForceProcessConfig, process_dataset


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()  # type: ignore[no-any-return]
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess DM-Tac raw uint8 tactile streams into SDK distributed_force "
            "float32 streams."
        )
    )
    parser.add_argument("--dataset-root", required=True, type=Path, help="Source LeRobot-style dataset root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root. Defaults to --dataset-root and writes under tactile_force/.",
    )
    parser.add_argument(
        "--episode",
        action="append",
        type=int,
        default=None,
        help="Episode index to process. Repeatable. Defaults to all discovered raw tactile manifests.",
    )
    parser.add_argument(
        "--side",
        action="append",
        choices=("left", "right"),
        default=None,
        help="Side to process. Repeatable. Defaults to both sides when present.",
    )
    parser.add_argument(
        "--sensor",
        action="append",
        default=None,
        help="Sensor key/serial to process. Repeatable. Defaults to all raw streams.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Parallel process count. 0 selects a conservative automatic value.",
    )
    parser.add_argument("--backend", default="cpu", choices=("cpu", "cuda", "Flux", "flux"))
    parser.add_argument("--mode", default="standard", choices=("standard", "high"))
    parser.add_argument("--cuda-id", type=int, default=0)
    parser.add_argument("--remote-addr", default="192.168.127.10:50051")
    parser.add_argument("--pc-host", default="0.0.0.0")
    parser.add_argument("--pc-port", type=int, default=60000)
    parser.add_argument(
        "--base-frame-index",
        type=int,
        default=None,
        help=(
            "Explicit raw frame index to install as the SDK base frame before processing each "
            "stream. Overrides any fixed sensor calibration for this run."
        ),
    )
    parser.add_argument(
        "--calibration-policy",
        choices=("auto", "required", "off"),
        default="auto",
        help="Load a fixed base frame by sensor serial when available, require one, or disable loading.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="Fixed calibration directory. Defaults to hardware/DMTAC_gripper/configs/tactile_calibration.",
    )
    parser.add_argument(
        "--save-calibration",
        action="store_true",
        help="Persist the explicitly selected --base-frame-index for every sensor serial; requires one --episode.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Debug limit per sensor stream. 0 processes all frames.",
    )
    parser.add_argument(
        "--compression",
        choices=("zstd", "none"),
        default="zstd",
        help="Force stream storage compression. Zstd is lossless and enabled by default.",
    )
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--chunk-frames", type=int, default=16)
    parser.add_argument(
        "--sdk-root",
        type=Path,
        default=None,
        help="Optional dmrobotics SDK root. Defaults to assets/DM_Tac_SDK_Publish_1.2.10 when import fails.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output manifests/streams.",
    )
    parser.add_argument(
        "--no-scan-all",
        action="store_true",
        help="Only use meta/episodes.jsonl tactile_paths instead of also scanning tactile/**/manifest.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover work and write no output.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress messages.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ForceProcessConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root or args.dataset_root,
        episodes=tuple(args.episode or ()),
        sides=tuple(args.side or ()),
        sensors=tuple(args.sensor or ()),
        jobs=int(args.jobs),
        backend=str(args.backend),
        mode=str(args.mode),
        cuda_id=int(args.cuda_id),
        remote_addr=str(args.remote_addr),
        pc_host=str(args.pc_host),
        pc_port=int(args.pc_port),
        base_frame_index=(
            None if args.base_frame_index is None else int(args.base_frame_index)
        ),
        calibration_policy=str(args.calibration_policy),
        calibration_dir=args.calibration_dir,
        save_calibration=bool(args.save_calibration),
        compression=str(args.compression),
        compression_level=int(args.compression_level),
        chunk_frames=int(args.chunk_frames),
        max_frames=int(args.max_frames),
        sdk_root=args.sdk_root,
        overwrite=bool(args.overwrite),
        scan_all_tactile=not bool(args.no_scan_all),
        dry_run=bool(args.dry_run),
        progress=not bool(args.no_progress),
    )
    result = process_dataset(config)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True, default=_json_default))
    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
