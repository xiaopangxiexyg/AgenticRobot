from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dataset import inspect_intrinsics, undistort_dataset
from .intrinsics import CameraIntrinsics
from .video import process_video


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()  # type: ignore[no-any-return]
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _load_intrinsics_json(path: Path, camera_key: str | None) -> CameraIntrinsics:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if camera_key is not None:
        payload = payload[camera_key]["intrinsics"]
    elif "intrinsics" in payload and isinstance(payload["intrinsics"], dict):
        payload = payload["intrinsics"]
    return CameraIntrinsics.from_mapping(payload, camera_key=camera_key or "")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fast OpenCV undistortion for DexTeleop RealSense videos and datasets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Print raw and undistorted intrinsics.")
    inspect_parser.add_argument(
        "--devices-json",
        required=True,
        type=Path,
        help="Path to meta/devices/realsense.json.",
    )
    inspect_parser.add_argument("--camera", action="append", default=None, help="Camera key. Repeatable.")
    inspect_parser.add_argument("--alpha", type=float, default=0.0, help="OpenCV undistort alpha in [0, 1].")

    dataset_parser = subparsers.add_parser("dataset", help="Create an undistorted dataset copy.")
    dataset_parser.add_argument("--dataset-root", required=True, type=Path, help="Source dataset root.")
    dataset_parser.add_argument("--output-root", required=True, type=Path, help="Output dataset root.")
    dataset_parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Camera key to undistort. Repeatable. Default: cameras with non-zero distortion.",
    )
    dataset_parser.add_argument(
        "--episode",
        action="append",
        type=int,
        default=None,
        help="Episode index to process. Repeatable. Default: all episodes.",
    )
    dataset_parser.add_argument("--alpha", type=float, default=0.0, help="OpenCV undistort alpha in [0, 1].")
    dataset_parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc for output videos.")
    dataset_parser.add_argument("--jobs", type=int, default=1, help="Parallel video jobs.")
    dataset_parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy", "skip"),
        default="hardlink",
        help="How to clone unchanged files into the output dataset.",
    )
    dataset_parser.add_argument(
        "--size-policy",
        choices=("strict", "scale"),
        default="strict",
        help="What to do if video size differs from recorded intrinsics.",
    )
    dataset_parser.add_argument("--max-frames", type=int, default=0, help="Debug limit per video. 0 means all.")
    dataset_parser.add_argument("--overwrite", action="store_true", help="Overwrite output files if present.")

    video_parser = subparsers.add_parser("video", help="Undistort one video.")
    video_parser.add_argument("--input", required=True, type=Path, help="Input video path.")
    video_parser.add_argument("--output", required=True, type=Path, help="Output video path.")
    video_parser.add_argument(
        "--intrinsics-json",
        required=True,
        type=Path,
        help="Either a single intrinsics JSON object or meta/devices/realsense.json.",
    )
    video_parser.add_argument(
        "--camera",
        default=None,
        help="Camera key when --intrinsics-json points to meta/devices/realsense.json.",
    )
    video_parser.add_argument("--alpha", type=float, default=0.0, help="OpenCV undistort alpha in [0, 1].")
    video_parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc for output video.")
    video_parser.add_argument(
        "--size-policy",
        choices=("strict", "scale"),
        default="strict",
        help="What to do if video size differs from recorded intrinsics.",
    )
    video_parser.add_argument("--max-frames", type=int, default=0, help="Debug limit. 0 means all.")
    video_parser.add_argument("--overwrite", action="store_true", help="Overwrite output video if present.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "inspect":
        payload = inspect_intrinsics(args.devices_json, cameras=args.camera, alpha=args.alpha)
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
        return 0
    if args.command == "dataset":
        result = undistort_dataset(
            args.dataset_root,
            args.output_root,
            cameras=args.camera,
            episodes=args.episode,
            alpha=float(args.alpha),
            codec=str(args.codec),
            copy_mode=args.copy_mode,
            jobs=int(args.jobs),
            overwrite=bool(args.overwrite),
            size_policy=str(args.size_policy),
            max_frames=int(args.max_frames),
        )
        summary: dict[str, Any] = {
            "output_root": result.output_root,
            "manifest_path": result.manifest_path,
            "cameras": list(result.cameras),
            "episodes": list(result.episodes),
            "processed_video_count": len(result.processed_videos),
            "processed_videos": [
                {
                    "output_path": item.output_path,
                    "frame_count": item.frame_count,
                    "processing_fps": item.processing_fps,
                }
                for item in result.processed_videos
            ],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
        return 0
    if args.command == "video":
        intrinsics = _load_intrinsics_json(args.intrinsics_json, args.camera)
        result = process_video(
            args.input,
            args.output,
            intrinsics,
            alpha=float(args.alpha),
            codec=str(args.codec),
            size_policy=str(args.size_policy),
            max_frames=int(args.max_frames),
            overwrite=bool(args.overwrite),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

