from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

from .intrinsics import CameraIntrinsics, build_undistort_maps, load_realsense_intrinsics
from .video import VideoProcessResult, process_video


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
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object in {path}:{line_number}")
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
        index = int(record.get("episode_index"))
    except Exception:
        return False
    return index in requested


def _video_paths_from_records(records: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], str]:
    out: dict[tuple[int, str], str] = {}
    for record in records:
        try:
            episode_index = int(record.get("episode_index"))
        except Exception:
            continue
        video_paths = record.get("video_paths")
        if not isinstance(video_paths, Mapping):
            continue
        for camera_key, relative_path in video_paths.items():
            if relative_path:
                out[(episode_index, str(camera_key))] = str(relative_path)
    return out


def _nonzero_distortion(intrinsics: CameraIntrinsics, *, eps: float = 1e-12) -> bool:
    return any(abs(float(value)) > eps for value in intrinsics.coeffs)


def _select_cameras(
    intrinsics_by_camera: Mapping[str, CameraIntrinsics],
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    if requested:
        missing = [key for key in requested if key not in intrinsics_by_camera]
        if missing:
            raise KeyError(f"requested camera(s) missing from devices intrinsics: {', '.join(missing)}")
        return tuple(str(key) for key in requested)
    distorted = [key for key, intrinsics in intrinsics_by_camera.items() if _nonzero_distortion(intrinsics)]
    return tuple(sorted(distorted or intrinsics_by_camera))


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


@dataclass(frozen=True)
class DatasetVideoJob:
    episode_index: int
    camera_key: str
    relative_path: str
    source_path: Path
    output_path: Path
    intrinsics: CameraIntrinsics


@dataclass(frozen=True)
class DatasetUndistortResult:
    source_root: Path
    output_root: Path
    cameras: tuple[str, ...]
    episodes: tuple[int, ...]
    processed_videos: tuple[VideoProcessResult, ...]
    manifest_path: Path
    alpha: float
    copy_mode: CopyMode
    output_intrinsics_by_camera: dict[str, CameraIntrinsics] = field(default_factory=dict)


def _clone_non_video_files(source_root: Path, output_root: Path, *, copy_mode: CopyMode, overwrite: bool) -> None:
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


def _update_devices_json(
    output_root: Path,
    *,
    source_devices: Mapping[str, Any],
    output_intrinsics_by_camera: Mapping[str, CameraIntrinsics],
    alpha: float,
) -> None:
    devices_path = output_root / "meta" / "devices" / "realsense.json"
    backup_path = output_root / "meta" / "devices" / "realsense_raw_before_undistort.json"
    _write_json(backup_path, source_devices)
    updated = json.loads(json.dumps(_jsonable(source_devices)))
    for camera_key, output_intrinsics in output_intrinsics_by_camera.items():
        device = updated.get(camera_key)
        if not isinstance(device, dict):
            continue
        raw_intrinsics = source_devices.get(camera_key, {}).get("intrinsics") if isinstance(source_devices.get(camera_key), Mapping) else None
        device["intrinsics"] = output_intrinsics.to_dict()
        metadata = device.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            device["metadata"] = metadata
        metadata["undistorted"] = True
        metadata["undistort_alpha"] = float(alpha)
        metadata["raw_intrinsics_path"] = "meta/devices/realsense_raw_before_undistort.json"
        if isinstance(raw_intrinsics, Mapping):
            metadata["raw_intrinsics"] = dict(raw_intrinsics)
    _write_json(devices_path, updated)


def _update_info_json(output_root: Path, manifest: Mapping[str, Any]) -> None:
    info_path = output_root / "meta" / "info.json"
    info = _read_json(info_path)
    if not info:
        return
    info["real_undistort"] = _jsonable(manifest)
    _write_json(info_path, info)


def _parse_episode_filter(episodes: Sequence[int] | None) -> set[int] | None:
    if not episodes:
        return None
    return {int(value) for value in episodes}


def undistort_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    cameras: Sequence[str] | None = None,
    episodes: Sequence[int] | None = None,
    alpha: float = 0.0,
    codec: str = "mp4v",
    copy_mode: CopyMode = "hardlink",
    jobs: int = 1,
    overwrite: bool = False,
    size_policy: str = "strict",
    max_frames: int = 0,
) -> DatasetUndistortResult:
    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"source dataset root does not exist: {source_root}")
    if source_root == output_root:
        raise ValueError("output root must be different from source dataset root")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"output root is non-empty; use --overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_devices = _read_json(source_root / "meta" / "devices" / "realsense.json")
    intrinsics_by_camera = load_realsense_intrinsics(source_devices)
    selected_cameras = _select_cameras(intrinsics_by_camera, cameras)
    requested_episodes = _parse_episode_filter(episodes)
    episode_records = _read_jsonl(source_root / "meta" / "episodes.jsonl")
    selected_records = [record for record in episode_records if _episode_matches(record, requested_episodes)]
    if not selected_records:
        raise ValueError("no episodes selected from meta/episodes.jsonl")
    video_paths = _video_paths_from_records(selected_records)

    jobs_to_run: list[DatasetVideoJob] = []
    for (episode_index, camera_key), relative_path in sorted(video_paths.items()):
        if camera_key not in selected_cameras:
            continue
        source_path = source_root / relative_path
        if not source_path.exists():
            raise FileNotFoundError(f"video for {camera_key} episode {episode_index} is missing: {source_path}")
        jobs_to_run.append(
            DatasetVideoJob(
                episode_index=episode_index,
                camera_key=camera_key,
                relative_path=relative_path,
                source_path=source_path,
                output_path=output_root / relative_path,
                intrinsics=intrinsics_by_camera[camera_key],
            )
        )
    if not jobs_to_run:
        raise ValueError(f"no videos found for selected cameras: {', '.join(selected_cameras)}")

    processed_relative_paths = {job.relative_path for job in jobs_to_run}
    _clone_non_video_files(source_root, output_root, copy_mode=copy_mode, overwrite=overwrite)
    _clone_unprocessed_videos(
        source_root,
        output_root,
        processed_relative_paths=processed_relative_paths,
        copy_mode=copy_mode,
        overwrite=overwrite,
    )

    def run_job(job: DatasetVideoJob) -> VideoProcessResult:
        return process_video(
            job.source_path,
            job.output_path,
            job.intrinsics,
            alpha=alpha,
            codec=codec,
            size_policy=size_policy,  # type: ignore[arg-type]
            max_frames=max_frames,
            overwrite=True,
        )

    results: list[VideoProcessResult] = []
    worker_count = max(1, int(jobs))
    if worker_count == 1:
        for job in jobs_to_run:
            results.append(run_job(job))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_job = {executor.submit(run_job, job): job for job in jobs_to_run}
            for future in as_completed(future_to_job):
                results.append(future.result())
    results.sort(key=lambda item: str(item.output_path))

    output_intrinsics_by_camera: dict[str, CameraIntrinsics] = {}
    for camera_key in selected_cameras:
        output_intrinsics_by_camera[camera_key] = build_undistort_maps(
            intrinsics_by_camera[camera_key],
            alpha=alpha,
        ).output
    _update_devices_json(
        output_root,
        source_devices=source_devices,
        output_intrinsics_by_camera=output_intrinsics_by_camera,
        alpha=alpha,
    )

    selected_episode_indices = tuple(
        int(record["episode_index"])
        for record in selected_records
        if record.get("episode_index") is not None
    )
    manifest = {
        "format": "dexteleop_real_undistort_dataset_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "cameras": list(selected_cameras),
        "episodes": list(selected_episode_indices),
        "alpha": float(alpha),
        "codec": str(codec),
        "copy_mode": str(copy_mode),
        "size_policy": str(size_policy),
        "max_frames": int(max_frames),
        "processed_videos": [
            {
                "input_path": str(result.input_path),
                "output_path": str(result.output_path),
                "frame_count": int(result.frame_count),
                "fps": float(result.fps),
                "width": int(result.width),
                "height": int(result.height),
                "elapsed_sec": float(result.elapsed_sec),
                "processing_fps": float(result.processing_fps),
            }
            for result in results
        ],
        "output_intrinsics": {
            key: intrinsics.to_dict() for key, intrinsics in output_intrinsics_by_camera.items()
        },
    }
    manifest_path = output_root / "meta" / "real_undistort.json"
    _write_json(manifest_path, manifest)
    _update_info_json(output_root, manifest)
    return DatasetUndistortResult(
        source_root=source_root,
        output_root=output_root,
        cameras=tuple(selected_cameras),
        episodes=tuple(selected_episode_indices),
        processed_videos=tuple(results),
        manifest_path=manifest_path,
        alpha=alpha,
        copy_mode=copy_mode,
        output_intrinsics_by_camera=output_intrinsics_by_camera,
    )


def inspect_intrinsics(
    devices_json: str | Path,
    *,
    cameras: Iterable[str] | None = None,
    alpha: float = 0.0,
) -> dict[str, Any]:
    devices = _read_json(Path(devices_json).expanduser().resolve())
    intrinsics_by_camera = load_realsense_intrinsics(devices)
    selected = _select_cameras(intrinsics_by_camera, tuple(cameras or ()))
    return {
        camera_key: {
            "raw": intrinsics_by_camera[camera_key].to_dict(),
            "undistorted": build_undistort_maps(
                intrinsics_by_camera[camera_key],
                alpha=alpha,
            ).output.to_dict(),
        }
        for camera_key in selected
    }
