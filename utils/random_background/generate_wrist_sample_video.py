from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.random_background import (
    LocalImageBackgroundProvider,
    PhotometricAugmentConfig,
    RandomBackgroundAugmentConfig,
    RandomBackgroundVideoAugmenter,
)
from utils.random_background.mask_io import BitpackMaskReader


DEFAULT_FINAL_MASK_ROOT = Path(
    "utils/dataset_postprocess/wrist_cam_adjust/outputs/final_keep_mask"
)


def _load_episode(session: Path, episode: int) -> dict[str, Any]:
    episodes_path = session / "meta" / "episodes.jsonl"
    with episodes_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("episode_index", -1)) == int(episode):
                return record
    raise ValueError(f"episode {episode} not found in {episodes_path}")


def _resolve_depth_manifest(
    *,
    session: Path,
    episode: dict[str, Any],
    episode_index: int,
    camera_key: str,
    depth_mask_root: Path,
    allow_raw_depth_mask: bool,
) -> tuple[Path, str]:
    episode_tag = f"episode_{int(episode_index):06d}"
    processed = depth_mask_root / session.name / episode_tag / camera_key / "manifest.json"
    if processed.exists():
        return processed, "postprocessed_depth_mask"
    if allow_raw_depth_mask:
        depth_rel = episode["wrist_depth_mask_paths"][camera_key]
        return session / depth_rel, "raw_depth_mask"
    raise FileNotFoundError(
        "postprocessed depth mask not found; run "
        "'python -m utils.dataset_postprocess.wrist_cam_adjust postprocess-depth-mask' first "
        f"or pass --allow-raw-depth-mask. missing={processed}"
    )


def _resolve_final_mask_manifest(
    *,
    session: Path,
    episode_index: int,
    camera_key: str,
    final_mask_root: Path,
) -> Path:
    episode_tag = f"episode_{int(episode_index):06d}"
    manifest = final_mask_root / session.name / episode_tag / camera_key / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"final keep-mask manifest not found: {manifest}")
    return manifest


def _open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    return capture


def _read_frame(capture: cv2.VideoCapture, path: Path) -> np.ndarray | None:
    ok, frame = capture.read()
    if not ok:
        return None
    if frame is None or frame.ndim != 3:
        raise RuntimeError(f"invalid frame read from {path}")
    return frame


def _mask_from_frame(frame: np.ndarray, *, height: int, width: int) -> np.ndarray:
    mask = frame[..., 0] if frame.ndim == 3 else frame
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def _open_writer(path: Path, fps: float, width: int, height: int, codec: str) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer: {path}")
    return writer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create wrist-camera random-background sample videos from final keep-masks."
    )
    parser.add_argument("--session", required=True, type=Path, help="Dataset root containing meta/episodes.jsonl.")
    parser.add_argument(
        "--table-mask-root",
        default=None,
        type=Path,
        help="Component table-mask root; only valid with --use-component-masks.",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--camera", action="append", default=None, help="Camera key. Repeatable.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-tag", default="", help="Optional suffix in output filenames.")
    parser.add_argument(
        "--depth-mask-root",
        type=Path,
        default=Path("utils/dataset_postprocess/wrist_cam_adjust/outputs/depth_mask_postprocess"),
        help="Root containing postprocess-depth-mask outputs.",
    )
    parser.add_argument(
        "--final-mask-root",
        type=Path,
        default=DEFAULT_FINAL_MASK_ROOT,
        help="Root containing compose-final-mask outputs (default: canonical final_keep_mask root).",
    )
    parser.add_argument(
        "--use-component-masks",
        action="store_true",
        help=(
            "Explicit diagnostic rollback: compose postprocessed depth and table masks "
            "during preview instead of reading final_keep_mask."
        ),
    )
    parser.add_argument(
        "--allow-raw-depth-mask",
        action="store_true",
        help="Fallback to the dataset's raw wrist_depth/mask stream when postprocessed output is absent.",
    )
    parser.add_argument("--max-frames", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--provider", choices=("procedural", "real_image"), default="procedural")
    parser.add_argument("--real-image-root", type=Path, default=None)
    parser.add_argument("--real-image-manifest", type=Path, default=None)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--side-by-side", action="store_true", help="Write original/mask/random comparison videos.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    session = args.session.resolve()
    depth_mask_root = args.depth_mask_root.resolve()
    if args.use_component_masks:
        if args.table_mask_root is None:
            raise ValueError("--table-mask-root is required with --use-component-masks.")
        table_root = args.table_mask_root.resolve()
        final_mask_root = None
    else:
        if args.table_mask_root is not None:
            raise ValueError(
                "--table-mask-root requires --use-component-masks; previews use "
                "final_keep_mask by default."
            )
        table_root = None
        final_mask_root = args.final_mask_root.resolve()
    output_dir = args.output_dir.resolve()
    episode = _load_episode(session, args.episode)
    cameras = args.camera or ["left_wrist", "right_wrist"]

    real_provider = None
    if args.provider == "real_image":
        real_provider = LocalImageBackgroundProvider(
            image_root=args.real_image_root,
            manifest_path=args.real_image_manifest,
        )

    augmenter = RandomBackgroundVideoAugmenter(
        RandomBackgroundAugmentConfig(
            global_seed=int(args.seed),
            provider=args.provider,
            mask_role="keep",
            edge_blur_px=2,
            enable_epoch_variation=False,
            photometric=PhotometricAugmentConfig(enabled=False),
        ),
        real_image_provider=real_provider,
    )

    summary: list[dict[str, Any]] = []
    for camera_key in cameras:
        rgb_rel = episode["video_paths"][camera_key]
        rgb_path = session / rgb_rel
        table_path = None if table_root is None else table_root / rgb_rel
        if final_mask_root is not None:
            mask_path = _resolve_final_mask_manifest(
                session=session,
                episode_index=int(args.episode),
                camera_key=camera_key,
                final_mask_root=final_mask_root,
            )
            mask_source = "final_keep_mask"
        else:
            mask_path, mask_source = _resolve_depth_manifest(
                session=session,
                episode=episode,
                episode_index=int(args.episode),
                camera_key=camera_key,
                depth_mask_root=depth_mask_root,
                allow_raw_depth_mask=bool(args.allow_raw_depth_mask),
            )
        tag = f"_{args.output_tag}" if args.output_tag else ""
        output_path = output_dir / f"episode_{int(args.episode):06d}_{camera_key}{tag}_random_bg.mp4"
        compare_path = output_dir / f"episode_{int(args.episode):06d}_{camera_key}{tag}_compare.mp4"

        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"output exists: {output_path}")
        if args.side_by_side and compare_path.exists() and not args.overwrite:
            raise FileExistsError(f"output exists: {compare_path}")

        rgb_cap = _open_video(rgb_path)
        table_cap = None if table_path is None else _open_video(table_path)
        fps = rgb_cap.get(cv2.CAP_PROP_FPS) or 15.0
        width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = _open_writer(output_path, fps, width, height, args.codec)
        compare_writer = (
            _open_writer(compare_path, fps, width * 3, height, args.codec) if args.side_by_side else None
        )

        frame_count = 0
        keep_fractions: list[float] = []
        try:
            with BitpackMaskReader(mask_path) as mask_reader:
                limit = int(args.max_frames) if int(args.max_frames) > 0 else int(episode.get("frame_count", 0) or 0)
                while limit <= 0 or frame_count < limit:
                    rgb = _read_frame(rgb_cap, rgb_path)
                    if rgb is None:
                        break
                    raw_mask = mask_reader.read(frame_count)
                    if raw_mask.shape[:2] != (height, width):
                        raw_mask = cv2.resize(raw_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    keep_mask = np.where(raw_mask > 0, 255, 0).astype(np.uint8)
                    if table_cap is not None:
                        table_frame = _read_frame(table_cap, table_path)  # type: ignore[arg-type]
                        if table_frame is None:
                            break
                        table_mask = _mask_from_frame(table_frame, height=height, width=width)
                        keep_mask = np.maximum(table_mask, keep_mask)
                    keep_fractions.append(float(np.count_nonzero(keep_mask) / keep_mask.size))
                    augmented = augmenter.apply(
                        rgb,
                        keep_mask,
                        episode_index=int(args.episode),
                        camera_key=camera_key,
                        frame_index=frame_count,
                    )
                    writer.write(augmented)
                    if compare_writer is not None:
                        mask_bgr = cv2.cvtColor(keep_mask, cv2.COLOR_GRAY2BGR)
                        compare_writer.write(np.concatenate([rgb, mask_bgr, augmented], axis=1))
                    frame_count += 1
        finally:
            rgb_cap.release()
            if table_cap is not None:
                table_cap.release()
            writer.release()
            if compare_writer is not None:
                compare_writer.release()

        summary.append(
            {
                "camera": camera_key,
                "frames": frame_count,
                "rgb": str(rgb_path),
                "table_mask": None if table_path is None else str(table_path),
                "mask": str(mask_path),
                "mask_source": mask_source,
                "output": str(output_path),
                "compare_output": str(compare_path) if args.side_by_side else None,
                "provider": args.provider,
                "seed": int(args.seed),
                "mask_policy": (
                    "final_keep_mask direct; no proxy mask"
                    if final_mask_root is not None
                    else "postprocessed_depth_keep_mask OR table_geometry_keep_mask; no proxy mask"
                ),
                "keep_fraction_mean": float(np.mean(keep_fractions)) if keep_fractions else 0.0,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_tag = f"_{args.output_tag}" if args.output_tag else ""
    summary_path = output_dir / f"episode_{int(args.episode):06d}{summary_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "summary_path": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
