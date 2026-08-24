from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from utils.dataset_postprocess.real_undistort.intrinsics import (
    CameraIntrinsics,
    load_realsense_intrinsics,
)

from ..geometry import (
    load_scene_geometry,
    resolve_repo_relative_path,
    tool_transform_from_profile,
)
from ..progress import ProgressPrinter
from .mask import (
    GEOMETRY_BACKGROUND_MASK_NAME,
    GEOMETRY_KEEP_MASK_NAME,
    GEOMETRY_MASK_FAMILY,
    STATIC_BODY_MASK_NAME,
    TABLE_GEOMETRY_MASK_NAME,
)
from .video import MaskBuildConfig, VideoAdjustResult, WristMaskBuilder, process_video

CopyMode = Literal["hardlink", "copy", "skip"]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError(f"expected object in {path}:{line_number}")
        rows.append(value)
    return rows


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _episode_matches(record: Mapping[str, Any], requested: set[int] | None) -> bool:
    if requested is None:
        return True
    try:
        return int(record.get("episode_index")) in requested
    except Exception:
        return False


def _parse_episode_filter(episodes: Sequence[int] | None) -> set[int] | None:
    if not episodes:
        return None
    return {int(value) for value in episodes}


def _copy_or_link(source: Path, target: Path, *, copy_mode: CopyMode, overwrite: bool) -> None:
    if copy_mode == "skip":
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not overwrite:
            return
        target.unlink()
    if copy_mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def _clone_non_video_files(source_root: Path, output_root: Path, *, copy_mode: CopyMode, overwrite: bool) -> None:
    if copy_mode == "skip":
        return
    for source_path in source_root.rglob("*"):
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(source_root)
        if relative_path.parts and relative_path.parts[0] == "videos":
            continue
        effective_copy_mode = "copy" if relative_path.parts and relative_path.parts[0] == "meta" else copy_mode
        _copy_or_link(
            source_path,
            output_root / relative_path,
            copy_mode=effective_copy_mode,  # type: ignore[arg-type]
            overwrite=overwrite,
        )


def _clone_unprocessed_videos(
    source_root: Path,
    output_root: Path,
    *,
    processed_relative_paths: set[str],
    copy_mode: CopyMode,
    overwrite: bool,
) -> None:
    if copy_mode == "skip":
        return
    videos_root = source_root / "videos"
    if not videos_root.exists():
        return
    for source_path in videos_root.rglob("*"):
        if not source_path.is_file():
            continue
        relative_path = str(source_path.relative_to(source_root))
        if relative_path in processed_relative_paths:
            continue
        _copy_or_link(
            source_path,
            output_root / relative_path,
            copy_mode=copy_mode,
            overwrite=overwrite,
        )


def _read_parquet_rows(path: Path, *, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("wrist_cam_adjust dataset mode requires pyarrow.") from exc
    table = pq.read_table(path, columns=list(columns) if columns else None)
    rows = table.to_pylist()
    rows.sort(key=lambda row: int(row.get("frame_index", row.get("index", 0)) or 0))
    return rows


def _side_from_camera(camera_key: str, *, left_camera_key: str, right_camera_key: str) -> str:
    if camera_key == left_camera_key:
        return "left"
    if camera_key == right_camera_key:
        return "right"
    raise ValueError(f"camera {camera_key!r} is not a configured wrist camera")


def _select_cameras(
    *,
    requested: Sequence[str] | None,
    left_camera_key: str,
    right_camera_key: str,
) -> tuple[str, ...]:
    available = (left_camera_key, right_camera_key)
    if not requested:
        return available
    selected = tuple(str(value) for value in requested)
    unknown = [key for key in selected if key not in available]
    if unknown:
        raise ValueError(f"unsupported wrist camera(s): {', '.join(unknown)}")
    return selected


def _intrinsics_by_camera(dataset_root: Path, devices_json: Path | None) -> dict[str, CameraIntrinsics]:
    devices = _read_json(devices_json) if devices_json is not None else {}
    out = load_realsense_intrinsics(devices)
    if not out and isinstance(devices.get("cameras"), Mapping):
        out = load_realsense_intrinsics(devices["cameras"])
    if out:
        return out
    dataset_devices = _read_json(dataset_root / "meta" / "devices" / "realsense.json")
    out = load_realsense_intrinsics(dataset_devices)
    if not out and isinstance(dataset_devices.get("cameras"), Mapping):
        out = load_realsense_intrinsics(dataset_devices["cameras"])
    return out


def _find_metadata_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for item in value.values():
            found = _find_metadata_value(item, key)
            if found is not None:
                return found
    elif isinstance(value, list | tuple):
        for item in value:
            found = _find_metadata_value(item, key)
            if found is not None:
                return found
    return None


def _tool_profile_path(
    *,
    side: str,
    explicit_profile: str | Path | None,
    record: Mapping[str, Any],
    repo_root: Path,
) -> Path | None:
    if explicit_profile is not None:
        return resolve_repo_relative_path(explicit_profile, repo_root=repo_root)
    path = _find_metadata_value(record.get("metadata"), f"{side}_tool_profile")
    resolved = resolve_repo_relative_path(str(path), repo_root=repo_root) if path else None
    if resolved is not None and resolved.exists():
        return resolved
    default_profile = repo_root / "hardware" / "rm65" / "utils" / "configs" / "calibrations" / f"rm65_{side}_DMTAC_M.json"
    if default_profile.exists():
        return default_profile
    return None


def _resolve_pose_frame(
    *,
    pose_frame: str,
    side: str,
    record: Mapping[str, Any],
) -> str:
    if pose_frame != "auto":
        return pose_frame
    metadata = record.get("metadata")
    frame = None
    if isinstance(metadata, Mapping):
        by_side = metadata.get("arm_controlled_frame_by_side")
        if isinstance(by_side, Mapping):
            frame = by_side.get(side)
    return "tool_tcp" if str(frame or "").strip() == "tool_tcp" else "arm_end"


def _base_from_stored_pose_by_side(
    info: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], str]:
    contract = info.get("pose_frame_contract")
    if not isinstance(contract, Mapping):
        return {}, "native_arm_base"
    storage_frame = str(contract.get("pose_storage_frame") or "").strip()
    if storage_frame != "desk_absolute":
        return {}, storage_frame or "native_arm_base"
    source = contract.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("desk_absolute pose_frame_contract has no source mapping")
    transforms = source.get("raw_to_canonical_desk_transform_by_side")
    if not isinstance(transforms, Mapping):
        raise ValueError(
            "desk_absolute pose_frame_contract has no per-side raw-to-desk transforms"
        )
    result: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        value = np.asarray(transforms.get(side), dtype=np.float64)
        if value.shape != (4, 4) or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid raw-to-desk transform for {side}")
        result[side] = np.linalg.inv(value)
    return result, storage_frame


def _expected_video_frames(record: Mapping[str, Any], *, max_frames: int) -> int:
    raw_count = record.get("frame_count")
    try:
        frame_count = int(raw_count)
    except (TypeError, ValueError):
        frame_count = 0
    if int(max_frames) > 0:
        return int(max_frames) if frame_count <= 0 else min(frame_count, int(max_frames))
    return max(0, frame_count)


@dataclass(frozen=True)
class DatasetVideoJob:
    episode_index: int
    camera_key: str
    side: str
    relative_path: str
    source_path: Path
    output_path: Path
    expected_frame_count: int
    parquet_path: Path
    record: Mapping[str, Any]
    intrinsics: CameraIntrinsics
    devices_json_path: Path
    body_mask_path: Path | None
    tool_profile_path: Path | None
    T_end_tool: np.ndarray | None
    T_base_from_stored_pose: np.ndarray | None
    pose_frame: str


@dataclass(frozen=True)
class DatasetAdjustResult:
    source_root: Path
    output_root: Path
    cameras: tuple[str, ...]
    episodes: tuple[int, ...]
    processed_videos: tuple[VideoAdjustResult, ...]
    manifest_path: Path
    output_kind: str
    copy_mode: CopyMode
    manifest: dict[str, Any] = field(default_factory=dict)


def adjust_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    urdf_path: str | Path,
    cameras: Sequence[str] | None = None,
    episodes: Sequence[int] | None = None,
    devices_json: str | Path | None = None,
    devices_json_by_episode: Mapping[int, str | Path] | None = None,
    left_camera_key: str = "left_wrist",
    right_camera_key: str = "right_wrist",
    left_body_mask: str | Path | None = None,
    right_body_mask: str | Path | None = None,
    left_tool_profile: str | Path | None = None,
    right_tool_profile: str | Path | None = None,
    pose_frame: str = "auto",
    pose_source: str = "auto",
    pose_column: str | None = None,
    qpos_column: str | None = None,
    output_kind: str = "masked",
    table_margin_m: float = 0.0,
    table_height_offset_m: float = 0.0,
    table_mask_mode: str = "sampled-raycast",
    table_raycast_stride: int = 2,
    table_edge_samples: int = 16,
    table_min_depth_m: float = 0.03,
    table_dilate_px: int = 8,
    table_boundary_extend_px: int | None = None,
    body_dilate_px: int = 0,
    keep_dilate_px: int = 0,
    mask_blur_px: int = 0,
    mask_blur_sigma: float = 0.0,
    codec: str = "mp4v",
    copy_mode: CopyMode = "hardlink",
    jobs: int = 1,
    overwrite: bool = False,
    max_frames: int = 0,
    show_progress: bool = False,
    debug_timing: bool = False,
) -> DatasetAdjustResult:
    repo_root = Path(__file__).resolve().parents[2]
    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"source dataset root does not exist: {source_root}")
    if source_root == output_root:
        raise ValueError("output root must be different from source dataset root")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"output root is non-empty; use --overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    scene = load_scene_geometry(urdf_path)
    dataset_info = _read_json(source_root / "meta" / "info.json")
    base_from_stored_pose, pose_storage_frame = _base_from_stored_pose_by_side(
        dataset_info
    )
    if devices_json is not None and devices_json_by_episode:
        raise ValueError(
            "devices_json and devices_json_by_episode are mutually exclusive"
        )
    episode_devices = {
        int(episode_index): Path(path).expanduser().resolve()
        for episode_index, path in (devices_json_by_episode or {}).items()
    }
    default_devices_path = (
        (source_root / "meta" / "devices" / "realsense.json").resolve()
        if devices_json is None
        else Path(devices_json).expanduser().resolve()
    )
    intrinsics_cache: dict[Path, dict[str, CameraIntrinsics]] = {}
    selected_cameras = _select_cameras(
        requested=cameras,
        left_camera_key=left_camera_key,
        right_camera_key=right_camera_key,
    )
    requested_episodes = _parse_episode_filter(episodes)
    episode_records = _read_jsonl(source_root / "meta" / "episodes.jsonl")
    selected_records = [record for record in episode_records if _episode_matches(record, requested_episodes)]
    if not selected_records:
        raise ValueError("no episodes selected from meta/episodes.jsonl")

    body_masks = {
        "left": None if left_body_mask is None else Path(left_body_mask).expanduser().resolve(),
        "right": None if right_body_mask is None else Path(right_body_mask).expanduser().resolve(),
    }
    explicit_profiles = {"left": left_tool_profile, "right": right_tool_profile}

    jobs_to_run: list[DatasetVideoJob] = []
    for record in selected_records:
        episode_index = int(record["episode_index"])
        episode_devices_path = episode_devices.get(
            episode_index,
            default_devices_path,
        )
        if episode_devices and episode_index not in episode_devices:
            raise KeyError(
                "devices_json_by_episode has no entry for selected episode "
                f"{episode_index}."
            )
        if episode_devices_path not in intrinsics_cache:
            intrinsics_cache[episode_devices_path] = _intrinsics_by_camera(
                source_root,
                episode_devices_path,
            )
        intrinsics = intrinsics_cache[episode_devices_path]
        missing_intrinsics = [
            key for key in selected_cameras if key not in intrinsics
        ]
        if missing_intrinsics:
            raise KeyError(
                f"missing camera intrinsics for episode {episode_index}: "
                + ", ".join(missing_intrinsics)
            )
        video_paths = record.get("video_paths")
        if not isinstance(video_paths, Mapping):
            continue
        parquet_path = source_root / str(record.get("parquet_path"))
        if not parquet_path.exists():
            raise FileNotFoundError(f"episode {episode_index} parquet missing: {parquet_path}")
        for camera_key in selected_cameras:
            relative_path = video_paths.get(camera_key)
            if not relative_path:
                continue
            side = _side_from_camera(
                str(camera_key),
                left_camera_key=left_camera_key,
                right_camera_key=right_camera_key,
            )
            source_path = source_root / str(relative_path)
            if not source_path.exists():
                raise FileNotFoundError(f"video for {camera_key} episode {episode_index} is missing: {source_path}")
            effective_pose_frame = _resolve_pose_frame(
                pose_frame=pose_frame,
                side=side,
                record=record,
            )
            profile_path = _tool_profile_path(
                side=side,
                explicit_profile=explicit_profiles[side],
                record=record,
                repo_root=repo_root,
            )
            T_end_tool = None
            if effective_pose_frame == "tool_tcp":
                if profile_path is None or not profile_path.exists():
                    raise FileNotFoundError(
                        f"{side} pose frame resolved to tool_tcp but no tool profile was found. "
                        f"Pass --{side}-tool-profile or --pose-frame arm_end if the pose is already link6."
                    )
                T_end_tool = tool_transform_from_profile(profile_path)
            jobs_to_run.append(
                DatasetVideoJob(
                    episode_index=episode_index,
                    camera_key=str(camera_key),
                    side=side,
                    relative_path=str(relative_path),
                    source_path=source_path,
                    output_path=output_root / str(relative_path),
                    expected_frame_count=_expected_video_frames(
                        record,
                        max_frames=max_frames,
                    ),
                    parquet_path=parquet_path,
                    record=record,
                    intrinsics=intrinsics[str(camera_key)],
                    devices_json_path=episode_devices_path,
                    body_mask_path=body_masks[side],
                    tool_profile_path=profile_path,
                    T_end_tool=T_end_tool,
                    T_base_from_stored_pose=base_from_stored_pose.get(side),
                    pose_frame=effective_pose_frame,
                )
            )
    if not jobs_to_run:
        raise ValueError(f"no videos found for selected cameras: {', '.join(selected_cameras)}")
    missing_frame_contract = [
        f"episode={job.episode_index},camera={job.camera_key}"
        for job in jobs_to_run
        if job.expected_frame_count <= 0
    ]
    if missing_frame_contract:
        raise ValueError(
            "selected videos have no positive frame_count contract: "
            + ", ".join(missing_frame_contract[:8])
        )

    expected_total_frames = sum(job.expected_frame_count for job in jobs_to_run)
    progress = ProgressPrinter(
        label="wrist_cam_adjust dataset",
        total_items=len(jobs_to_run),
        total_units=expected_total_frames,
        enabled=bool(show_progress),
    )

    processed_relative_paths = {job.relative_path for job in jobs_to_run}
    _clone_non_video_files(source_root, output_root, copy_mode=copy_mode, overwrite=overwrite)
    _clone_unprocessed_videos(
        source_root,
        output_root,
        processed_relative_paths=processed_relative_paths,
        copy_mode=copy_mode,
        overwrite=overwrite,
    )

    def run_job(job: DatasetVideoJob) -> VideoAdjustResult:
        job_label = f"episode_{job.episode_index:06d}:{job.camera_key}"
        progress.start_item(job_label)
        rows = _read_parquet_rows(job.parquet_path)
        builder = WristMaskBuilder(
            MaskBuildConfig(
                scene=scene,
                side=job.side,
                intrinsics=job.intrinsics,
                body_mask_path=job.body_mask_path,
                pose_frame=job.pose_frame,
                pose_source=pose_source,
                pose_column=pose_column,
                qpos_column=qpos_column,
                T_end_tool=job.T_end_tool,
                T_base_from_stored_pose=job.T_base_from_stored_pose,
                table_margin_m=table_margin_m,
                table_height_offset_m=table_height_offset_m,
                table_mask_mode=table_mask_mode,
                table_raycast_stride=table_raycast_stride,
                table_edge_samples=table_edge_samples,
                table_min_depth_m=table_min_depth_m,
                table_dilate_px=table_dilate_px,
                table_boundary_extend_px=table_boundary_extend_px,
                body_dilate_px=body_dilate_px,
                keep_dilate_px=keep_dilate_px,
            )
        )
        result = process_video(
            job.source_path,
            job.output_path,
            rows=rows,
            mask_builder=builder,
            camera_key=job.camera_key,
            output_kind=output_kind,
            codec=codec,
            max_frames=max_frames,
            expected_frame_count=job.expected_frame_count,
            overwrite=True,
            mask_blur_px=mask_blur_px,
            mask_blur_sigma=mask_blur_sigma,
            progress=progress.callback(job_label),
            debug_timing=debug_timing,
        )
        progress.complete_item(job_label, result.frame_count)
        return result

    results: list[VideoAdjustResult] = []
    worker_count = max(1, int(jobs))
    try:
        if worker_count == 1:
            for job in jobs_to_run:
                results.append(run_job(job))
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_job = {executor.submit(run_job, job): job for job in jobs_to_run}
                for future in as_completed(future_to_job):
                    results.append(future.result())
    finally:
        progress.finish()
    results.sort(key=lambda item: str(item.output_path))

    selected_episode_indices = tuple(
        int(record["episode_index"])
        for record in selected_records
        if record.get("episode_index") is not None
    )
    manifest = {
        "format": "dexteleop_wrist_cam_adjust_dataset_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "urdf_path": str(Path(urdf_path).expanduser().resolve()),
        "cameras": list(selected_cameras),
        "episodes": list(selected_episode_indices),
        "pose_frame": str(pose_frame),
        "pose_source": str(pose_source),
        "pose_storage_frame": pose_storage_frame,
        "stored_pose_to_arm_base": {
            side: transform.tolist()
            for side, transform in sorted(base_from_stored_pose.items())
        },
        "pose_column": pose_column,
        "qpos_column": qpos_column,
        "output_kind": str(output_kind),
        "mask_family": GEOMETRY_MASK_FAMILY,
        "mask_names": {
            "table": TABLE_GEOMETRY_MASK_NAME,
            "body": STATIC_BODY_MASK_NAME,
            "keep": GEOMETRY_KEEP_MASK_NAME,
            "background": GEOMETRY_BACKGROUND_MASK_NAME,
        },
        "mask_semantics": {
            "table": "URDF finite table top projected into the wrist RGB image from camera pose and intrinsics.",
            "body": "Optional static painted wrist/body mask loaded from --left-body-mask/--right-body-mask.",
            "keep": "Geometry keep mask: table geometry projection OR static body mask.",
            "background": "Inverse of the geometry keep mask.",
        },
        "depth_mask_note": (
            "This geometry projection path does not read wrist_depth/mask streams. "
            "Saved depth range masks are handled by visualize-depth-mask and postprocess-depth-mask."
        ),
        "table_margin_m": float(table_margin_m),
        "table_height_offset_m": float(table_height_offset_m),
        "table_mask_mode": str(table_mask_mode),
        "table_raycast_stride": int(table_raycast_stride),
        "table_edge_samples": int(table_edge_samples),
        "table_min_depth_m": float(table_min_depth_m),
        "table_dilate_px": int(table_dilate_px),
        "table_boundary_extend_px": None if table_boundary_extend_px is None else int(table_boundary_extend_px),
        "body_dilate_px": int(body_dilate_px),
        "keep_dilate_px": int(keep_dilate_px),
        "mask_blur_px": int(mask_blur_px),
        "mask_blur_sigma": float(mask_blur_sigma),
        "codec": str(codec),
        "copy_mode": str(copy_mode),
        "max_frames": int(max_frames),
        "debug_timing": bool(debug_timing),
        "processed_videos": [result.to_dict() for result in results],
        "jobs": [
            {
                "episode_index": job.episode_index,
                "camera_key": job.camera_key,
                "side": job.side,
                "relative_path": job.relative_path,
                "expected_frame_count": job.expected_frame_count,
                "pose_frame": job.pose_frame,
                "stored_pose_transform_applied": (
                    job.T_base_from_stored_pose is not None
                ),
                "body_mask_path": None if job.body_mask_path is None else str(job.body_mask_path),
                "devices_json_path": str(job.devices_json_path),
                "tool_profile_path": None if job.tool_profile_path is None else str(job.tool_profile_path),
            }
            for job in jobs_to_run
        ],
    }
    manifest_path = output_root / "meta" / "wrist_cam_adjust.json"
    _write_json(manifest_path, _jsonable(manifest))
    info_path = output_root / "meta" / "info.json"
    info = _read_json(info_path)
    if info:
        info["wrist_cam_adjust"] = _jsonable(manifest)
        _write_json(info_path, info)

    return DatasetAdjustResult(
        source_root=source_root,
        output_root=output_root,
        cameras=tuple(selected_cameras),
        episodes=tuple(selected_episode_indices),
        processed_videos=tuple(results),
        manifest_path=manifest_path,
        output_kind=output_kind,
        copy_mode=copy_mode,
        manifest=manifest,
    )
