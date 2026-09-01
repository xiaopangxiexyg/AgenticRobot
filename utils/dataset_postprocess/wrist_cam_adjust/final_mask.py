from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utils.dataset_postprocess.real_undistort.intrinsics import require_cv2

from .depth_mask.core import (
    WRIST_DEPTH_MASK_FORMAT,
    pack_binary_mask,
    unpack_binary_mask,
)
from .depth_mask.visualize import DEFAULT_WRIST_CAMERA_KEYS, load_mask_stream

FINAL_KEEP_MASK_FAMILY = "wrist_final_keep"
FINAL_KEEP_MASK_NAME = "postprocessed_depth_or_table_keep_mask"


@dataclass(frozen=True)
class FinalMaskTask:
    session_root: Path
    table_mask_root: Path
    depth_mask_root: Path
    output_root: Path
    episode_index: int
    camera_key: str
    rgb_relative_path: str
    depth_manifest_path: Path
    table_video_path: Path
    output_dir: Path
    max_frames: int = 0


@dataclass(frozen=True)
class FinalMaskResult:
    episode_index: int
    camera_key: str
    manifest_path: Path
    stream_path: Path
    frame_count: int
    height: int
    width: int
    depth_fraction_mean: float
    table_fraction_mean: float
    output_fraction_mean: float
    rollback_dir: Path | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": int(self.episode_index),
            "camera_key": self.camera_key,
            "manifest_path": str(self.manifest_path),
            "stream_path": str(self.stream_path),
            "frame_count": int(self.frame_count),
            "height": int(self.height),
            "width": int(self.width),
            "depth_fraction_mean": float(self.depth_fraction_mean),
            "table_fraction_mean": float(self.table_fraction_mean),
            "output_fraction_mean": float(self.output_fraction_mean),
            "rollback_dir": None
            if self.rollback_dir is None
            else str(self.rollback_dir),
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = line.strip()
        if not text:
            continue
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object in {path}:{line_number}")
        rows.append(value)
    return rows


def _episode_tag(episode_index: int) -> str:
    return f"episode_{int(episode_index):06d}"


def _dataset_namespace(session_root: Path, value: str | None) -> str:
    namespace = session_root.name if value is None else str(value)
    if (
        not namespace
        or namespace != namespace.strip()
        or namespace in {".", ".."}
        or "/" in namespace
        or "\\" in namespace
        or "\x00" in namespace
        or Path(namespace).name != namespace
    ):
        raise ValueError(
            f"dataset_namespace must be one safe path component: {namespace!r}"
        )
    return namespace


def _resolve_processed_depth_manifest(
    *,
    session_root: Path,
    depth_mask_root: Path,
    episode_index: int,
    camera_key: str,
    dataset_namespace: str | None = None,
) -> Path:
    namespace = _dataset_namespace(session_root, dataset_namespace)
    return (
        depth_mask_root
        / namespace
        / _episode_tag(episode_index)
        / camera_key
        / "manifest.json"
    )


def _mask_from_table_frame(frame: np.ndarray, *, height: int, width: int) -> np.ndarray:
    cv2 = require_cv2()
    mask = frame[..., 0] if frame.ndim == 3 else frame
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def _manifest_payload(
    *,
    task: FinalMaskTask,
    stream_path: Path,
    frame_count: int,
    height: int,
    width: int,
    records: list[Mapping[str, Any]],
    depth_fraction_mean: float,
    table_fraction_mean: float,
    output_fraction_mean: float,
    rollback_dir: Path | None,
) -> dict[str, Any]:
    return {
        "format": WRIST_DEPTH_MASK_FORMAT,
        "camera_key": task.camera_key,
        "relative_dir": str(task.output_dir),
        "mask_family": FINAL_KEEP_MASK_FAMILY,
        "mask_name": FINAL_KEEP_MASK_NAME,
        "stream": {
            "path": stream_path.name,
            "encoding": "np.packbits",
            "dtype": "uint1",
            "shape": [int(height), int(width)],
            "packed_bytes_per_frame": int((int(height) * int(width) + 7) // 8),
            "frame_count": int(frame_count),
        },
        "semantics": {
            "name": FINAL_KEEP_MASK_NAME,
            "value_1": "foreground/keep pixel: postprocessed wrist depth mask OR projected table geometry mask",
            "value_0": "replaceable background pixel",
            "proxy_mask": False,
            "body_mask": False,
        },
        "source": {
            "session_root": str(task.session_root),
            "table_mask_root": str(task.table_mask_root),
            "depth_mask_root": str(task.depth_mask_root),
            "depth_manifest_path": str(task.depth_manifest_path),
            "table_video_path": str(task.table_video_path),
            "rgb_relative_path": task.rgb_relative_path,
            "integrity": {
                "depth_stream_policy": "exact_declared_frames_v1",
                "table_video_policy": "exact_declared_frames_and_tail_decode_v1",
                "source_modified": False,
            },
        },
        "publication": {
            "mode": "atomic_directory_replace_v1",
            "rollback_dir": None if rollback_dir is None else str(rollback_dir),
        },
        "stats": {
            "depth_fraction_mean": float(depth_fraction_mean),
            "table_fraction_mean": float(table_fraction_mean),
            "output_fraction_mean": float(output_fraction_mean),
        },
        "frames": [dict(record) for record in records],
    }


def compose_final_mask(
    task: FinalMaskTask, *, overwrite: bool = False
) -> FinalMaskResult:
    cv2 = require_cv2()
    if not task.depth_manifest_path.exists():
        raise FileNotFoundError(
            f"postprocessed depth mask manifest missing: {task.depth_manifest_path}"
        )
    if not task.table_video_path.exists():
        raise FileNotFoundError(
            f"table keep-mask video missing: {task.table_video_path}"
        )
    task.output_dir.parent.mkdir(parents=True, exist_ok=True)
    if task.output_dir.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass overwrite=True: {task.output_dir}")
    transaction_id = uuid.uuid4().hex
    staging_dir = (
        task.output_dir.parent / f".{task.output_dir.name}.tmp-{transaction_id}"
    )
    staging_dir.mkdir(parents=False, exist_ok=False)
    staging_stream_path = staging_dir / "final_keep.bitpack.bin"
    staging_manifest_path = staging_dir / "manifest.json"
    final_stream_path = task.output_dir / staging_stream_path.name
    final_manifest_path = task.output_dir / staging_manifest_path.name
    rollback_dir = (
        task.output_dir.parent / f".{task.output_dir.name}.rollback-{transaction_id}"
        if task.output_dir.exists()
        else None
    )

    depth_stream = load_mask_stream(task.depth_manifest_path)
    expected_depth_bytes = int(depth_stream.frame_count) * int(
        depth_stream.packed_bytes_per_frame
    )
    actual_depth_bytes = int(depth_stream.stream_path.stat().st_size)
    if actual_depth_bytes != expected_depth_bytes:
        raise ValueError(
            "postprocessed depth stream size differs from its manifest: "
            f"actual={actual_depth_bytes} expected={expected_depth_bytes} "
            f"path={depth_stream.stream_path}"
        )
    cap = cv2.VideoCapture(str(task.table_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open table mask video: {task.table_video_path}")

    frame_limit = int(depth_stream.frame_count)
    if int(task.max_frames) > 0:
        frame_limit = min(frame_limit, int(task.max_frames))
    records: list[dict[str, Any]] = []
    depth_sum = 0.0
    table_sum = 0.0
    output_sum = 0.0
    frame_count = 0
    try:
        with (
            depth_stream.stream_path.open("rb") as handle,
            staging_stream_path.open("wb") as out,
        ):
            for frame_index in range(frame_limit):
                data = handle.read(int(depth_stream.packed_bytes_per_frame))
                if len(data) != int(depth_stream.packed_bytes_per_frame):
                    raise ValueError(
                        "postprocessed depth stream ended before the declared "
                        f"frame limit at frame {frame_index}."
                    )
                ok, table_frame = cap.read()
                if not ok or table_frame is None:
                    raise ValueError(
                        "table keep-mask video ended before the declared frame "
                        f"limit at frame {frame_index}: {task.table_video_path}"
                    )
                depth_mask = (
                    unpack_binary_mask(
                        data, height=depth_stream.height, width=depth_stream.width
                    )
                    > 0
                )
                table_mask = _mask_from_table_frame(
                    table_frame,
                    height=depth_stream.height,
                    width=depth_stream.width,
                )
                final_mask = depth_mask | table_mask
                total = int(final_mask.size)
                depth_fraction = (
                    0.0
                    if total <= 0
                    else float(np.count_nonzero(depth_mask)) / float(total)
                )
                table_fraction = (
                    0.0
                    if total <= 0
                    else float(np.count_nonzero(table_mask)) / float(total)
                )
                output_fraction = (
                    0.0
                    if total <= 0
                    else float(np.count_nonzero(final_mask)) / float(total)
                )
                depth_sum += depth_fraction
                table_sum += table_fraction
                output_sum += output_fraction
                source_record: dict[str, Any] = {}
                if 0 <= frame_index < len(depth_stream.frames):
                    source_record = dict(depth_stream.frames[frame_index])
                records.append(
                    {
                        **source_record,
                        "frame_index": int(frame_index),
                        "depth_foreground_fraction": depth_fraction,
                        "table_keep_fraction": table_fraction,
                        "foreground_fraction": output_fraction,
                    }
                )
                out.write(pack_binary_mask(final_mask))
                frame_count += 1
            out.flush()
            os.fsync(out.fileno())
        if int(task.max_frames) <= 0:
            extra_ok, extra_frame = cap.read()
            if extra_ok and extra_frame is not None:
                raise ValueError(
                    "table keep-mask video contains frames beyond the declared "
                    f"depth contract: {task.table_video_path}"
                )
    finally:
        cap.release()

    if frame_count != frame_limit or frame_count <= 0:
        raise RuntimeError(
            "final mask frame count differs from its declared limit: "
            f"actual={frame_count} expected={frame_limit} camera={task.camera_key} "
            f"episode={task.episode_index}"
        )
    payload = _manifest_payload(
        task=task,
        stream_path=final_stream_path,
        frame_count=frame_count,
        height=depth_stream.height,
        width=depth_stream.width,
        records=records,
        depth_fraction_mean=depth_sum / float(frame_count),
        table_fraction_mean=table_sum / float(frame_count),
        output_fraction_mean=output_sum / float(frame_count),
        rollback_dir=rollback_dir,
    )
    expected_output_bytes = frame_count * int(depth_stream.packed_bytes_per_frame)
    actual_output_bytes = int(staging_stream_path.stat().st_size)
    if actual_output_bytes != expected_output_bytes:
        raise RuntimeError(
            "final keep-mask stream size differs from its frame contract: "
            f"actual={actual_output_bytes} expected={expected_output_bytes}"
        )
    with staging_manifest_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=True))
        handle.flush()
        os.fsync(handle.fileno())
    if rollback_dir is not None:
        os.replace(task.output_dir, rollback_dir)
    try:
        os.replace(staging_dir, task.output_dir)
    except BaseException:
        if (
            rollback_dir is not None
            and rollback_dir.exists()
            and not task.output_dir.exists()
        ):
            os.replace(rollback_dir, task.output_dir)
        raise
    return FinalMaskResult(
        episode_index=task.episode_index,
        camera_key=task.camera_key,
        manifest_path=final_manifest_path,
        stream_path=final_stream_path,
        frame_count=frame_count,
        height=depth_stream.height,
        width=depth_stream.width,
        depth_fraction_mean=depth_sum / float(frame_count),
        table_fraction_mean=table_sum / float(frame_count),
        output_fraction_mean=output_sum / float(frame_count),
        rollback_dir=rollback_dir,
    )


def build_final_mask_tasks(
    *,
    session_root: str | Path,
    table_mask_root: str | Path,
    depth_mask_root: str | Path,
    output_root: str | Path,
    episodes: Sequence[int] | None = None,
    cameras: Sequence[str] | None = None,
    max_frames: int = 0,
    dataset_namespace: str | None = None,
) -> list[FinalMaskTask]:
    session = Path(session_root).expanduser().resolve()
    table_root = Path(table_mask_root).expanduser().resolve()
    depth_root = Path(depth_mask_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    namespace = _dataset_namespace(session, dataset_namespace)
    selected_episodes = None if not episodes else {int(value) for value in episodes}
    selected_cameras = tuple(cameras or DEFAULT_WRIST_CAMERA_KEYS)
    tasks: list[FinalMaskTask] = []
    for record in _read_jsonl(session / "meta" / "episodes.jsonl"):
        episode_index = int(record["episode_index"])
        if selected_episodes is not None and episode_index not in selected_episodes:
            continue
        video_paths = record.get("video_paths")
        if not isinstance(video_paths, Mapping):
            continue
        for camera_key in selected_cameras:
            rgb_relative = video_paths.get(camera_key)
            if not isinstance(rgb_relative, str) or not rgb_relative:
                continue
            output_dir = output / namespace / _episode_tag(episode_index) / camera_key
            tasks.append(
                FinalMaskTask(
                    session_root=session,
                    table_mask_root=table_root,
                    depth_mask_root=depth_root,
                    output_root=output,
                    episode_index=episode_index,
                    camera_key=str(camera_key),
                    rgb_relative_path=rgb_relative,
                    depth_manifest_path=_resolve_processed_depth_manifest(
                        session_root=session,
                        depth_mask_root=depth_root,
                        episode_index=episode_index,
                        camera_key=str(camera_key),
                        dataset_namespace=namespace,
                    ),
                    table_video_path=table_root / rgb_relative,
                    output_dir=output_dir,
                    max_frames=int(max_frames),
                )
            )
    return tasks


def compose_final_masks(
    *,
    session_root: str | Path,
    table_mask_root: str | Path,
    depth_mask_root: str | Path,
    output_root: str | Path,
    episodes: Sequence[int] | None = None,
    cameras: Sequence[str] | None = None,
    max_frames: int = 0,
    jobs: int = 1,
    overwrite: bool = False,
    dataset_namespace: str | None = None,
) -> dict[str, Any]:
    session = Path(session_root).expanduser().resolve()
    namespace = _dataset_namespace(session, dataset_namespace)
    tasks = build_final_mask_tasks(
        session_root=session,
        table_mask_root=table_mask_root,
        depth_mask_root=depth_mask_root,
        output_root=output_root,
        episodes=episodes,
        cameras=cameras,
        max_frames=max_frames,
        dataset_namespace=namespace,
    )
    if not tasks:
        raise ValueError(
            "no final mask tasks were found; check --session, --episode, --camera, "
            "--table-mask-root, and --depth-mask-root."
        )
    worker_count = max(1, int(jobs))
    results: list[FinalMaskResult] = []
    if worker_count == 1:
        for task in tasks:
            results.append(compose_final_mask(task, overwrite=overwrite))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_task = {
                executor.submit(compose_final_mask, task, overwrite=overwrite): task
                for task in tasks
            }
            for future in as_completed(future_to_task):
                results.append(future.result())
    results.sort(key=lambda item: (item.episode_index, item.camera_key))
    summary = {
        "format": "dexteleop_wrist_final_mask_batch_v1",
        "session_root": str(Path(session_root).expanduser().resolve()),
        "table_mask_root": str(Path(table_mask_root).expanduser().resolve()),
        "depth_mask_root": str(Path(depth_mask_root).expanduser().resolve()),
        "output_root": str(Path(output_root).expanduser().resolve()),
        "dataset_namespace": namespace,
        "mask_policy": "postprocessed_depth_keep_mask OR table_geometry_keep_mask; no proxy mask",
        "task_count": len(tasks),
        "processed_count": len(results),
        "jobs": worker_count,
        "results": [result.to_dict() for result in results],
    }
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / namespace / "final_mask_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    summary["summary_path"] = str(summary_path)
    return summary
