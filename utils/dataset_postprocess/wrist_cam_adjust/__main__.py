from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from utils.dataset_postprocess.io import read_jsonl as _read_jsonl
from utils.dataset_postprocess.real_undistort.intrinsics import CameraIntrinsics, load_realsense_intrinsics

from .geometry import load_scene_geometry, tool_transform_from_profile
from .geometry_mask.dataset import adjust_dataset
from .geometry_mask.video import (
    TABLE_MASK_MODE_CHOICES,
    MaskBuildConfig,
    WristMaskBuilder,
    process_video,
)
from .progress import ProgressPrinter
from .depth_mask.postprocess import (
    DepthMaskPostprocessConfig,
    process_mask_stream,
)
from .depth_mask.visualize import (
    DEFAULT_WRIST_CAMERA_KEYS,
    find_session_mask_manifests,
    load_mask_stream,
    write_mask_preview_video,
)
from .final_mask import compose_final_masks


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()  # type: ignore[no-any-return]
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _load_devices_intrinsics(path: Path, camera_key: str) -> CameraIntrinsics:
    payload = json.loads(path.read_text(encoding="utf-8"))
    intrinsics = load_realsense_intrinsics(payload)
    if not intrinsics and isinstance(payload, Mapping) and isinstance(payload.get("cameras"), Mapping):
        intrinsics = load_realsense_intrinsics(payload["cameras"])
    if camera_key not in intrinsics:
        raise KeyError(f"camera {camera_key!r} missing intrinsics in {path}")
    return intrinsics[camera_key]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("video mode with --rows-parquet requires pyarrow.") from exc
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows.sort(key=lambda row: int(row.get("frame_index", row.get("index", 0)) or 0))
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and inspect wrist-camera masks. The dataset/video commands use "
            "URDF table geometry projection masks; depth commands use saved wrist depth range masks."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser(
        "dataset",
        help="Create a dataset copy using geometry projection masks, not saved wrist depth masks.",
    )
    dataset.add_argument("--dataset-root", required=True, type=Path, help="Source dataset root.")
    dataset.add_argument("--output-root", required=True, type=Path, help="Output dataset root.")
    dataset.add_argument("--urdf", required=True, type=Path, help="URDF containing table and wrist camera frames.")
    dataset.add_argument(
        "--devices-json",
        type=Path,
        default=None,
        help="Optional devices JSON with intrinsics. Defaults to dataset meta/devices/realsense.json.",
    )
    dataset.add_argument("--camera", action="append", default=None, help="Camera key. Repeatable.")
    dataset.add_argument("--episode", action="append", type=int, default=None, help="Episode index. Repeatable.")
    dataset.add_argument("--left-camera-key", default="left_wrist")
    dataset.add_argument("--right-camera-key", default="right_wrist")
    dataset.add_argument("--left-body-mask", type=Path, default=None)
    dataset.add_argument("--right-body-mask", type=Path, default=None)
    dataset.add_argument("--left-tool-profile", type=Path, default=None)
    dataset.add_argument("--right-tool-profile", type=Path, default=None)
    dataset.add_argument(
        "--pose-frame",
        choices=("auto", "arm_end", "tool_tcp", "camera"),
        default="auto",
        help="Frame of dataset pose columns. auto reads dataset metadata when possible.",
    )
    dataset.add_argument(
        "--pose-source",
        choices=("auto", "pose", "qpos"),
        default="auto",
        help="Use pose columns, qpos FK, or pose with qpos fallback.",
    )
    dataset.add_argument(
        "--pose-column",
        default=None,
        help="Optional pose column override. Supports {side}, e.g. observation.arm.{side}.actual_tcp_pose.",
    )
    dataset.add_argument(
        "--qpos-column",
        default=None,
        help="Optional qpos column override. Supports {side}, e.g. observation.arm.{side}.actual_qpos.",
    )
    dataset.add_argument(
        "--output-kind",
        choices=("masked", "overlay", "background-mask", "keep-mask"),
        default="masked",
        help="What to write into selected wrist camera video channels using the geometry keep mask.",
    )
    dataset.add_argument("--table-margin-m", type=float, default=0.0)
    dataset.add_argument("--table-height-offset-m", type=float, default=0.0)
    dataset.add_argument(
        "--table-mask-mode",
        choices=TABLE_MASK_MODE_CHOICES,
        default="sampled-raycast",
        help="sampled-raycast is the robust fast default; raycast is exact; polygon is experimental fastest.",
    )
    dataset.add_argument("--table-raycast-stride", type=int, default=2)
    dataset.add_argument("--table-edge-samples", type=int, default=16)
    dataset.add_argument("--table-min-depth-m", type=float, default=0.03)
    dataset.add_argument("--table-dilate-px", type=int, default=8)
    dataset.add_argument("--table-boundary-extend-px", type=int, default=None)
    dataset.add_argument("--body-dilate-px", type=int, default=0)
    dataset.add_argument("--keep-dilate-px", type=int, default=0)
    dataset.add_argument("--mask-blur-px", type=int, default=0)
    dataset.add_argument("--mask-blur-sigma", type=float, default=0.0)
    dataset.add_argument("--codec", default="mp4v")
    dataset.add_argument("--jobs", type=int, default=1)
    dataset.add_argument("--copy-mode", choices=("hardlink", "copy", "skip"), default="hardlink")
    dataset.add_argument("--max-frames", type=int, default=0)
    dataset.add_argument("--debug-timing", action="store_true", help="Record per-phase timing in the JSON output/manifest.")
    dataset.add_argument("--no-progress", action="store_true", help="Disable stderr progress display.")
    dataset.add_argument("--overwrite", action="store_true")

    video = subparsers.add_parser(
        "video",
        help="Process one wrist video with a geometry projection mask from one parquet file.",
    )
    video.add_argument("--input", required=True, type=Path)
    video.add_argument("--output", required=True, type=Path)
    video.add_argument("--rows-parquet", required=True, type=Path)
    video.add_argument("--camera-key", required=True)
    video.add_argument("--side", choices=("left", "right"), required=True)
    video.add_argument("--urdf", required=True, type=Path)
    video.add_argument("--devices-json", required=True, type=Path)
    video.add_argument("--body-mask", type=Path, default=None)
    video.add_argument("--tool-profile", type=Path, default=None)
    video.add_argument("--pose-frame", choices=("arm_end", "tool_tcp", "camera"), default="arm_end")
    video.add_argument("--pose-source", choices=("auto", "pose", "qpos"), default="auto")
    video.add_argument("--pose-column", default=None)
    video.add_argument("--qpos-column", default=None)
    video.add_argument(
        "--output-kind",
        choices=("masked", "overlay", "background-mask", "keep-mask"),
        default="masked",
    )
    video.add_argument("--table-margin-m", type=float, default=0.0)
    video.add_argument("--table-height-offset-m", type=float, default=0.0)
    video.add_argument(
        "--table-mask-mode",
        choices=TABLE_MASK_MODE_CHOICES,
        default="sampled-raycast",
        help="sampled-raycast is the robust fast default; raycast is exact; polygon is experimental fastest.",
    )
    video.add_argument("--table-raycast-stride", type=int, default=2)
    video.add_argument("--table-edge-samples", type=int, default=16)
    video.add_argument("--table-min-depth-m", type=float, default=0.03)
    video.add_argument("--table-dilate-px", type=int, default=8)
    video.add_argument("--table-boundary-extend-px", type=int, default=None)
    video.add_argument("--body-dilate-px", type=int, default=0)
    video.add_argument("--keep-dilate-px", type=int, default=0)
    video.add_argument("--mask-blur-px", type=int, default=0)
    video.add_argument("--mask-blur-sigma", type=float, default=0.0)
    video.add_argument("--codec", default="mp4v")
    video.add_argument("--max-frames", type=int, default=0)
    video.add_argument("--debug-timing", action="store_true", help="Record per-phase timing in the JSON output.")
    video.add_argument("--no-progress", action="store_true", help="Disable stderr progress display.")
    video.add_argument("--overwrite", action="store_true")

    depth_mask = subparsers.add_parser(
        "visualize-depth-mask",
        help="Create a compact preview video for saved wrist depth range 0/1 masks.",
    )
    depth_mask.add_argument(
        "--session",
        type=Path,
        default=None,
        help="Dataset/session root containing meta/episodes.jsonl and wrist_depth/mask streams.",
    )
    depth_mask.add_argument("--episode", type=int, default=None, help="Episode index when reading --session.")
    depth_mask.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Camera key to read from --session. Repeatable. Defaults to left_wrist and right_wrist.",
    )
    depth_mask.add_argument(
        "--mask",
        action="append",
        type=Path,
        default=None,
        help="Direct mask manifest or bit-packed stream. Repeat for side-by-side visualization.",
    )
    depth_mask.add_argument("--left-mask", type=Path, default=None, help="Direct left wrist mask manifest/stream.")
    depth_mask.add_argument("--right-mask", type=Path, default=None, help="Direct right wrist mask manifest/stream.")
    depth_mask.add_argument(
        "--shape",
        default=None,
        help="Required for direct .bin streams without a sibling manifest, formatted as HxW, e.g. 480x640.",
    )
    depth_mask.add_argument(
        "--output",
        type=Path,
        default=Path("utils/dataset_postprocess/wrist_cam_adjust/outputs/wrist_depth_mask_preview.mp4"),
        help="Output preview mp4 path.",
    )
    depth_mask.add_argument("--fps", type=float, default=15.0)
    depth_mask.add_argument("--scale", type=float, default=1.0)
    depth_mask.add_argument("--max-frames", type=int, default=0)
    depth_mask.add_argument("--codec", default="mp4v")
    depth_mask.add_argument("--overwrite", action="store_true")

    post_depth_mask = subparsers.add_parser(
        "postprocess-depth-mask",
        help="Fill holes and smooth saved wrist depth range 0/1 masks.",
    )
    post_depth_mask.add_argument("--session", type=Path, default=None)
    post_depth_mask.add_argument("--episode", type=int, default=None)
    post_depth_mask.add_argument(
        "--all-episodes",
        action="store_true",
        help="Process every episode in --session instead of only --episode/the first episode.",
    )
    post_depth_mask.add_argument("--camera", action="append", default=None)
    post_depth_mask.add_argument("--mask", action="append", type=Path, default=None)
    post_depth_mask.add_argument("--left-mask", type=Path, default=None)
    post_depth_mask.add_argument("--right-mask", type=Path, default=None)
    post_depth_mask.add_argument("--shape", default=None)
    post_depth_mask.add_argument(
        "--output-root",
        type=Path,
        default=Path("utils/dataset_postprocess/wrist_cam_adjust/outputs/depth_mask_postprocess"),
    )
    default_post_cfg = DepthMaskPostprocessConfig()
    post_depth_mask.add_argument("--close-radius-px", type=int, default=default_post_cfg.close_radius_px)
    post_depth_mask.add_argument("--open-radius-px", type=int, default=0)
    post_depth_mask.add_argument("--median-ksize", type=int, default=default_post_cfg.median_ksize)
    post_depth_mask.add_argument("--fill-hole-max-area-px", type=int, default=default_post_cfg.fill_hole_max_area_px)
    post_depth_mask.add_argument("--min-component-area-px", type=int, default=0)
    post_depth_mask.add_argument("--gaussian-blur-px", type=int, default=0)
    post_depth_mask.add_argument("--gaussian-threshold", type=float, default=0.5)
    post_depth_mask.add_argument("--temporal-gap-fill-frames", type=int, default=default_post_cfg.temporal_gap_fill_frames)
    post_depth_mask.add_argument("--temporal-majority-window", type=int, default=default_post_cfg.temporal_majority_window)
    post_depth_mask.add_argument("--temporal-majority-threshold", type=float, default=default_post_cfg.temporal_majority_threshold)
    post_depth_mask.add_argument("--max-frames", type=int, default=0)
    post_depth_mask.add_argument("--jobs", type=int, default=1)
    post_depth_mask.add_argument("--preview-output", type=Path, default=None)
    post_depth_mask.add_argument("--no-preview", action="store_true")
    post_depth_mask.add_argument("--fps", type=float, default=15.0)
    post_depth_mask.add_argument("--scale", type=float, default=1.0)
    post_depth_mask.add_argument("--codec", default="mp4v")
    post_depth_mask.add_argument("--overwrite", action="store_true")

    final_mask = subparsers.add_parser(
        "compose-final-mask",
        help="Write final keep-mask bitpack streams: postprocessed depth OR table geometry, no proxy mask.",
    )
    final_mask.add_argument("--session", required=True, type=Path)
    final_mask.add_argument("--table-mask-root", required=True, type=Path)
    final_mask.add_argument(
        "--depth-mask-root",
        type=Path,
        default=Path("utils/dataset_postprocess/wrist_cam_adjust/outputs/depth_mask_postprocess"),
    )
    final_mask.add_argument(
        "--output-root",
        type=Path,
        default=Path("utils/dataset_postprocess/wrist_cam_adjust/outputs/final_keep_mask"),
    )
    final_mask.add_argument("--episode", action="append", type=int, default=None)
    final_mask.add_argument("--camera", action="append", default=None)
    final_mask.add_argument("--max-frames", type=int, default=0)
    final_mask.add_argument("--jobs", type=int, default=1)
    final_mask.add_argument("--overwrite", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "dataset":
        result = adjust_dataset(
            args.dataset_root,
            args.output_root,
            urdf_path=args.urdf,
            cameras=args.camera,
            episodes=args.episode,
            devices_json=args.devices_json,
            left_camera_key=str(args.left_camera_key),
            right_camera_key=str(args.right_camera_key),
            left_body_mask=args.left_body_mask,
            right_body_mask=args.right_body_mask,
            left_tool_profile=args.left_tool_profile,
            right_tool_profile=args.right_tool_profile,
            pose_frame=str(args.pose_frame),
            pose_source=str(args.pose_source),
            pose_column=args.pose_column,
            qpos_column=args.qpos_column,
            output_kind=str(args.output_kind),
            table_margin_m=float(args.table_margin_m),
            table_height_offset_m=float(args.table_height_offset_m),
            table_mask_mode=str(args.table_mask_mode),
            table_raycast_stride=int(args.table_raycast_stride),
            table_edge_samples=int(args.table_edge_samples),
            table_min_depth_m=float(args.table_min_depth_m),
            table_dilate_px=int(args.table_dilate_px),
            table_boundary_extend_px=args.table_boundary_extend_px,
            body_dilate_px=int(args.body_dilate_px),
            keep_dilate_px=int(args.keep_dilate_px),
            mask_blur_px=int(args.mask_blur_px),
            mask_blur_sigma=float(args.mask_blur_sigma),
            codec=str(args.codec),
            copy_mode=args.copy_mode,
            jobs=int(args.jobs),
            overwrite=bool(args.overwrite),
            max_frames=int(args.max_frames),
            show_progress=not bool(args.no_progress),
            debug_timing=bool(args.debug_timing),
        )
        summary = {
            "output_root": result.output_root,
            "manifest_path": result.manifest_path,
            "cameras": list(result.cameras),
            "episodes": list(result.episodes),
            "processed_video_count": len(result.processed_videos),
            "processed_videos": [item.to_dict() for item in result.processed_videos],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
        return 0
    if args.command == "video":
        scene = load_scene_geometry(args.urdf)
        intrinsics = _load_devices_intrinsics(args.devices_json, str(args.camera_key))
        T_end_tool = tool_transform_from_profile(args.tool_profile) if args.tool_profile else None
        builder = WristMaskBuilder(
            MaskBuildConfig(
                scene=scene,
                side=str(args.side),
                intrinsics=intrinsics,
                body_mask_path=args.body_mask,
                pose_frame=str(args.pose_frame),
                pose_source=str(args.pose_source),
                pose_column=args.pose_column,
                qpos_column=args.qpos_column,
                T_end_tool=T_end_tool,
                table_margin_m=float(args.table_margin_m),
                table_height_offset_m=float(args.table_height_offset_m),
                table_mask_mode=str(args.table_mask_mode),
                table_raycast_stride=int(args.table_raycast_stride),
                table_edge_samples=int(args.table_edge_samples),
                table_min_depth_m=float(args.table_min_depth_m),
                table_dilate_px=int(args.table_dilate_px),
                table_boundary_extend_px=args.table_boundary_extend_px,
                body_dilate_px=int(args.body_dilate_px),
                keep_dilate_px=int(args.keep_dilate_px),
            )
        )
        rows = _read_rows(args.rows_parquet)
        total_frames = int(args.max_frames) if int(args.max_frames) > 0 else len(rows)
        progress = ProgressPrinter(
            label="wrist_cam_adjust video",
            total_items=1,
            total_units=total_frames,
            enabled=not bool(args.no_progress),
        )
        progress_label = str(args.camera_key)
        progress.start_item(progress_label)
        try:
            result = process_video(
                args.input,
                args.output,
                rows=rows,
                mask_builder=builder,
                camera_key=str(args.camera_key),
                output_kind=str(args.output_kind),
                codec=str(args.codec),
                max_frames=int(args.max_frames),
                overwrite=bool(args.overwrite),
                mask_blur_px=int(args.mask_blur_px),
                mask_blur_sigma=float(args.mask_blur_sigma),
                progress=progress.callback(progress_label),
                debug_timing=bool(args.debug_timing),
            )
            progress.complete_item(progress_label, result.frame_count)
        finally:
            progress.finish()
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=_json_default))
        return 0
    if args.command == "visualize-depth-mask":
        direct_masks: list[tuple[str | None, Path]] = []
        if args.left_mask is not None:
            direct_masks.append(("left_wrist", args.left_mask))
        if args.right_mask is not None:
            direct_masks.append(("right_wrist", args.right_mask))
        if args.mask:
            direct_masks.extend((None, path) for path in args.mask)

        if direct_masks:
            root_hint = args.session if args.session is not None else None
            streams = [
                load_mask_stream(path, root_hint=root_hint, label=label, shape=args.shape)
                for label, path in direct_masks
            ]
        else:
            if args.session is None:
                parser.error("visualize-depth-mask requires --session or at least one direct --mask/--left-mask/--right-mask.")
            camera_keys = tuple(args.camera or DEFAULT_WRIST_CAMERA_KEYS)
            manifests = find_session_mask_manifests(
                args.session,
                episode=args.episode,
                camera_keys=camera_keys,
            )
            streams = [
                load_mask_stream(manifest, root_hint=args.session)
                for manifest in manifests
            ]
        result = write_mask_preview_video(
            streams,
            args.output,
            fps=float(args.fps),
            max_frames=int(args.max_frames),
            scale=float(args.scale),
            codec=str(args.codec),
            overwrite=bool(args.overwrite),
        )
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=_json_default))
        return 0
    if args.command == "postprocess-depth-mask":
        direct_masks: list[tuple[str | None, Path]] = []
        if args.left_mask is not None:
            direct_masks.append(("left_wrist", args.left_mask))
        if args.right_mask is not None:
            direct_masks.append(("right_wrist", args.right_mask))
        if args.mask:
            direct_masks.extend((None, path) for path in args.mask)

        if direct_masks:
            if bool(args.all_episodes):
                parser.error("--all-episodes cannot be used with direct --mask/--left-mask/--right-mask inputs.")
            root_hint = args.session if args.session is not None else None
            streams = [
                load_mask_stream(path, root_hint=root_hint, label=label, shape=args.shape)
                for label, path in direct_masks
            ]
            jobs_to_run = [(stream, Path(args.output_root) / str(stream.label), None) for stream in streams]
        else:
            if args.session is None:
                parser.error("postprocess-depth-mask requires --session or at least one direct --mask/--left-mask/--right-mask.")
            camera_keys = tuple(args.camera or DEFAULT_WRIST_CAMERA_KEYS)
            if bool(args.all_episodes):
                episode_records = _read_jsonl(Path(args.session) / "meta" / "episodes.jsonl")
                selected_episode_indices = [int(record["episode_index"]) for record in episode_records]
            else:
                if args.episode is None:
                    episode_records = _read_jsonl(Path(args.session) / "meta" / "episodes.jsonl")
                    if not episode_records:
                        raise ValueError(f"no episodes in {Path(args.session) / 'meta' / 'episodes.jsonl'}")
                    selected_episode_indices = [int(episode_records[0]["episode_index"])]
                else:
                    selected_episode_indices = [int(args.episode)]
            jobs_to_run = []
            for episode_index in selected_episode_indices:
                manifests = find_session_mask_manifests(
                    args.session,
                    episode=episode_index,
                    camera_keys=camera_keys,
                )
                episode_tag = f"episode_{int(episode_index):06d}"
                output_base = Path(args.output_root) / Path(args.session).expanduser().resolve().name / episode_tag
                for manifest in manifests:
                    stream = load_mask_stream(manifest, root_hint=args.session)
                    jobs_to_run.append((stream, output_base / str(stream.label), episode_index))

        config = DepthMaskPostprocessConfig(
            close_radius_px=int(args.close_radius_px),
            open_radius_px=int(args.open_radius_px),
            median_ksize=int(args.median_ksize),
            fill_hole_max_area_px=int(args.fill_hole_max_area_px),
            min_component_area_px=int(args.min_component_area_px),
            gaussian_blur_px=int(args.gaussian_blur_px),
            gaussian_threshold=float(args.gaussian_threshold),
            temporal_gap_fill_frames=int(args.temporal_gap_fill_frames),
            temporal_majority_window=int(args.temporal_majority_window),
            temporal_majority_threshold=float(args.temporal_majority_threshold),
        )
        def run_postprocess(job: tuple[Any, Path, int | None]):
            stream, output_dir, episode_index = job
            result = process_mask_stream(
                stream,
                output_dir,
                config=config,
                max_frames=int(args.max_frames),
                overwrite=bool(args.overwrite),
            )
            return episode_index, result

        processed_pairs = []
        worker_count = max(1, int(args.jobs))
        if worker_count == 1:
            for job in jobs_to_run:
                processed_pairs.append(run_postprocess(job))
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_job = {executor.submit(run_postprocess, job): job for job in jobs_to_run}
                for future in as_completed(future_to_job):
                    processed_pairs.append(future.result())
        processed_pairs.sort(key=lambda item: (int(item[0]) if item[0] is not None else -1, str(item[1].manifest_path)))
        results = [result for _episode_index, result in processed_pairs]

        preview = None
        if not bool(args.no_preview):
            if bool(args.all_episodes):
                preview = {"skipped": "preview is skipped for --all-episodes; run per episode for contact sheets"}
            else:
                processed_streams = [load_mask_stream(result.manifest_path) for result in results]
                preview_base = Path(args.output_root)
                if args.session is not None:
                    processed_episode = processed_pairs[0][0] if processed_pairs else args.episode
                    if processed_episode is not None:
                        episode_tag = f"episode_{int(processed_episode):06d}"
                        preview_base = Path(args.output_root) / Path(args.session).expanduser().resolve().name / episode_tag
                preview_output = args.preview_output or (preview_base / "postprocessed_preview.mp4")
                preview = write_mask_preview_video(
                    processed_streams,
                    preview_output,
                    fps=float(args.fps),
                    max_frames=int(args.max_frames),
                    scale=float(args.scale),
                    codec=str(args.codec),
                    overwrite=True,
                )
        summary = {
            "output_root": str(args.output_root),
            "config": config.to_dict(),
            "processed": [result.to_dict() for result in results],
            "preview": None if preview is None else (preview.to_dict() if hasattr(preview, "to_dict") else preview),
            "jobs": worker_count,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
        return 0
    if args.command == "compose-final-mask":
        summary = compose_final_masks(
            session_root=args.session,
            table_mask_root=args.table_mask_root,
            depth_mask_root=args.depth_mask_root,
            output_root=args.output_root,
            episodes=args.episode,
            cameras=args.camera,
            max_frames=int(args.max_frames),
            jobs=int(args.jobs),
            overwrite=bool(args.overwrite),
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
