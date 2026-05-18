from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET_ROOT = Path("BaselineVLA_deployment/Data/RawData/RealRobot/paper_fold")
DEFAULT_REPORT_PATH = Path(
    "BaselineVLA_deployment/Datasets/post_fix_realrobot_data/outputs/"
    "paper_fold_edge_stuck_qc/paper_fold_edge_trim_report.csv"
)


@dataclass(frozen=True)
class ClipPlan:
    dataset_name: str
    dataset_root: Path
    episode_index: int
    parquet_relpath: str
    original_frame_count: int
    keep_start_frame: int
    keep_end_frame_exclusive: int
    trim_start_frames: int
    trim_end_frames: int
    trim_start_sec: float
    trim_end_sec: float

    @property
    def keep_frame_count(self) -> int:
        return max(0, int(self.keep_end_frame_exclusive) - int(self.keep_start_frame))

    @property
    def was_trimmed(self) -> bool:
        return self.trim_start_frames > 0 or self.trim_end_frames > 0


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required. Use /home/zc/anaconda3/envs/HumanData/bin/python on this workstation."
        ) from exc
    return pa, pq


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _resolve_path(path_text: str, *, cwd: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else cwd / path


def load_clip_plans(report_path: Path, *, cwd: Path) -> dict[Path, dict[int, ClipPlan]]:
    plans: dict[Path, dict[int, ClipPlan]] = {}
    with report_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset_root = _resolve_path(str(row["dataset_root"]), cwd=cwd)
            episode_index = int(row["episode_index"])
            frame_count = int(row["frame_count"])
            keep_start = int(row["keep_start_frame"])
            keep_end = int(row["keep_end_frame_exclusive"])
            if keep_start < 0 or keep_end < keep_start or keep_end > frame_count:
                raise ValueError(
                    f"Invalid keep range for {dataset_root} episode {episode_index}: "
                    f"[{keep_start}, {keep_end}) with frame_count={frame_count}"
                )
            plan = ClipPlan(
                dataset_name=str(row["dataset_name"]),
                dataset_root=dataset_root,
                episode_index=episode_index,
                parquet_relpath=str(row["parquet_relpath"]),
                original_frame_count=frame_count,
                keep_start_frame=keep_start,
                keep_end_frame_exclusive=keep_end,
                trim_start_frames=int(row["trim_start_frames"]),
                trim_end_frames=int(row["trim_end_frames"]),
                trim_start_sec=float(row["trim_start_sec"]),
                trim_end_sec=float(row["trim_end_sec"]),
            )
            plans.setdefault(dataset_root, {})[episode_index] = plan
    if not plans:
        raise ValueError(f"No clip plans loaded from {report_path}")
    return plans


def _copy_meta_static_files(source_root: Path, output_root: Path) -> None:
    source_meta = source_root / "meta"
    output_meta = output_root / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)
    skip = {"info.json", "episodes.jsonl", "episodes_stats.jsonl"}
    for source in source_meta.iterdir():
        if source.name in skip:
            continue
        target = output_meta / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        elif source.is_file():
            shutil.copy2(source, target)


def _episode_relpath(episode_index: int, *, chunk_size: int) -> str:
    chunk_id = int(episode_index) // max(1, int(chunk_size))
    return f"data/chunk-{chunk_id:03d}/episode_{int(episode_index):06d}.parquet"


def _video_relpath(episode_index: int, *, chunk_size: int, camera_key: str = "front") -> str:
    chunk_id = int(episode_index) // max(1, int(chunk_size))
    return f"videos/observation.images.{camera_key}/chunk-{chunk_id:03d}/episode_{int(episode_index):06d}.mp4"


def _read_rows_by_episode(path: Path) -> dict[int, dict[str, Any]]:
    rows = _iter_jsonl(path)
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        out[int(row["episode_index"])] = row
    return out


def _median_dt(timestamps: np.ndarray, nominal_fps: float) -> float:
    if timestamps.shape[0] > 1:
        diffs = np.diff(timestamps.astype(np.float64))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(np.mean(diffs))
    return float(1.0 / nominal_fps) if nominal_fps > 0 else 0.0


def _stats_from_rows(
    rows: dict[str, list[Any]],
    *,
    episode_index: int,
    nominal_fps: float,
) -> dict[str, Any]:
    frame_count = len(next(iter(rows.values()))) if rows else 0
    timestamps = np.asarray(rows.get("timestamp", []), dtype=np.float64)
    if frame_count and timestamps.size == frame_count:
        start_timestamp = float(timestamps[0])
        end_timestamp = float(timestamps[-1])
        duration_sec = float(max(0.0, end_timestamp - start_timestamp))
        average_dt_sec = _median_dt(timestamps, nominal_fps)
    else:
        average_dt_sec = float(1.0 / nominal_fps) if nominal_fps > 0 else 0.0
        start_timestamp = 0.0
        duration_sec = float(max(0, frame_count - 1) * average_dt_sec)
        end_timestamp = float(duration_sec)
    control_hz_values = np.asarray(rows.get("control_hz", []), dtype=np.float64)
    control_hz_values = control_hz_values[np.isfinite(control_hz_values)]
    active_sides = rows.get("active_sides", [[]])
    arm_state_source = rows.get("arm_state_source", [""])
    camera_keys = rows.get("camera_keys", [[]])
    fault_codes = rows.get("fault_codes", [])
    fault_count = 0
    for item in fault_codes:
        if isinstance(item, list):
            fault_count += len(item)
        elif item:
            fault_count += 1
    return {
        "episode_index": int(episode_index),
        "frame_count": int(frame_count),
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "duration_sec": duration_sec,
        "average_dt_sec": float(average_dt_sec),
        "average_hz": float(1.0 / average_dt_sec) if average_dt_sec > 0 else 0.0,
        "mean_control_hz": float(np.mean(control_hz_values)) if control_hz_values.size else 0.0,
        "nominal_fps": float(nominal_fps),
        "active_sides": active_sides[0] if active_sides else [],
        "arm_state_source": arm_state_source[0] if arm_state_source else "",
        "camera_keys": camera_keys[0] if camera_keys else [],
        "fault_count": int(fault_count),
    }


def _replace_column(data: dict[str, list[Any]], name: str, values: list[Any]) -> None:
    if name in data:
        data[name] = values


def _nan_action() -> list[float]:
    return [float("nan")] * 26


def _read_video_frame_count(video_path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return -1
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _copy_or_clip_video(
    *,
    source_video_path: Path,
    output_video_path: Path,
    keep_start_frame: int,
    keep_frame_count: int,
    original_frame_count: int,
) -> None:
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    if keep_start_frame == 0 and keep_frame_count == original_frame_count:
        shutil.copy2(source_video_path, output_video_path)
        return

    import cv2

    cap = cv2.VideoCapture(str(source_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {source_video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid source video size for {source_video_path}: {width}x{height}")
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {output_video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(keep_start_frame))
        written = 0
        while written < keep_frame_count:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        cap.release()
    if written != keep_frame_count:
        raise RuntimeError(
            f"Video clip frame mismatch for {source_video_path}: wrote {written}, "
            f"expected {keep_frame_count}"
        )


def _relative_video_path_for_rows(rows: dict[str, list[Any]], fallback_relpath: str) -> str:
    paths = rows.get("observation.images.front.path") or []
    for item in paths:
        if item:
            return str(item)
    return fallback_relpath


def _write_clipped_episode(
    *,
    source_root: Path,
    output_root: Path,
    plan: ClipPlan,
    chunk_size: int,
    global_index_start: int,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    pa, pq = _require_pyarrow()
    source_parquet = source_root / plan.parquet_relpath
    if not source_parquet.exists():
        source_parquet = source_root / _episode_relpath(plan.episode_index, chunk_size=chunk_size)
    table = pq.read_table(source_parquet)
    if table.num_rows != plan.original_frame_count:
        raise ValueError(
            f"Frame count mismatch for {source_parquet}: report={plan.original_frame_count}, "
            f"parquet={table.num_rows}"
        )
    clipped = table.slice(plan.keep_start_frame, plan.keep_frame_count)
    data = clipped.to_pydict()
    frame_count = plan.keep_frame_count

    _replace_column(data, "index", list(range(global_index_start, global_index_start + frame_count)))
    _replace_column(data, "frame_index", list(range(frame_count)))
    _replace_column(data, "episode_index", [plan.episode_index] * frame_count)
    video_relpath = _video_relpath(plan.episode_index, chunk_size=chunk_size, camera_key="front")
    _replace_column(data, "observation.images.front.path", [video_relpath] * frame_count)
    _replace_column(data, "observation.images.front.video_frame_index", list(range(frame_count)))
    if "observation.last_action" in data and frame_count > 0 and plan.keep_start_frame > 0:
        data["observation.last_action"][0] = _nan_action()

    output_parquet_relpath = _episode_relpath(plan.episode_index, chunk_size=chunk_size)
    output_parquet = output_root / output_parquet_relpath
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_table = pa.Table.from_pydict(data, schema=table.schema)
    pq.write_table(output_table, output_parquet)

    original_data = table.to_pydict()
    source_video_relpath = _relative_video_path_for_rows(
        original_data,
        _video_relpath(plan.episode_index, chunk_size=chunk_size, camera_key="front"),
    )
    source_video = source_root / source_video_relpath
    output_video = output_root / video_relpath
    if source_video.exists():
        _copy_or_clip_video(
            source_video_path=source_video,
            output_video_path=output_video,
            keep_start_frame=plan.keep_start_frame,
            keep_frame_count=frame_count,
            original_frame_count=plan.original_frame_count,
        )

    stats = _stats_from_rows(data, episode_index=plan.episode_index, nominal_fps=30.0)
    return {"parquet_path": output_parquet_relpath, "video_path": video_relpath}, stats, frame_count


def _update_episode_metadata(
    original: dict[str, Any],
    *,
    output_paths: dict[str, Any],
    stats: dict[str, Any],
    plan: ClipPlan,
    output_root: Path,
    source_root: Path,
) -> dict[str, Any]:
    updated = dict(original)
    updated["frame_count"] = int(stats["frame_count"])
    updated["start_timestamp"] = float(stats["start_timestamp"])
    updated["end_timestamp"] = float(stats["end_timestamp"])
    updated["duration_sec"] = float(stats["duration_sec"])
    updated["parquet_path"] = output_paths["parquet_path"]
    updated["video_paths"] = {"front": output_paths["video_path"]}
    metadata = dict(updated.get("metadata") or {})
    metadata["clipped_from_dataset_root"] = str(source_root)
    metadata["clipped_dataset_root"] = str(output_root)
    metadata["clip_rule"] = "right_eef_edge_stuck_xyz_1cm_rot_2deg_min_0.5s"
    metadata["clip_trim_start_frames"] = int(plan.trim_start_frames)
    metadata["clip_trim_end_frames"] = int(plan.trim_end_frames)
    metadata["clip_trim_start_sec"] = float(plan.trim_start_sec)
    metadata["clip_trim_end_sec"] = float(plan.trim_end_sec)
    metadata["clip_keep_start_frame"] = int(plan.keep_start_frame)
    metadata["clip_keep_end_frame_exclusive"] = int(plan.keep_end_frame_exclusive)
    metadata["dataset_dir_name"] = output_root.name
    summary = updated.get("summary")
    if isinstance(summary, dict):
        summary = dict(summary)
        summary["episode_frame_count"] = int(stats["frame_count"])
        summary["submitted_samples_total"] = int(stats["frame_count"])
        updated["summary"] = summary
    updated["metadata"] = metadata
    return updated


def export_dataset(
    *,
    source_root: Path,
    output_root: Path,
    plans: dict[int, ClipPlan],
    overwrite: bool,
    limit_episodes: int,
) -> dict[str, Any]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _copy_meta_static_files(source_root, output_root)

    info = _load_json(source_root / "meta" / "info.json")
    chunk_size = int(info.get("chunk_size") or 1000)
    nominal_fps = float(info.get("nominal_fps") or 30.0)
    original_episodes = _read_rows_by_episode(source_root / "meta" / "episodes.jsonl")
    episode_indices = sorted(plans)
    if limit_episodes > 0:
        episode_indices = episode_indices[: int(limit_episodes)]

    output_episodes: list[dict[str, Any]] = []
    output_stats: list[dict[str, Any]] = []
    global_index = 0
    trimmed_episode_count = 0
    exported_original_frame_count = 0
    exported_removed_frame_count = 0
    for counter, episode_index in enumerate(episode_indices, start=1):
        plan = plans[episode_index]
        output_paths, stats, frame_count = _write_clipped_episode(
            source_root=source_root,
            output_root=output_root,
            plan=plan,
            chunk_size=chunk_size,
            global_index_start=global_index,
        )
        stats["nominal_fps"] = nominal_fps
        global_index += frame_count
        trimmed_episode_count += int(plan.was_trimmed)
        exported_original_frame_count += int(plan.original_frame_count)
        exported_removed_frame_count += int(plan.original_frame_count - plan.keep_frame_count)
        original_episode = original_episodes.get(episode_index, {"episode_index": episode_index})
        output_episodes.append(
            _update_episode_metadata(
                original_episode,
                output_paths=output_paths,
                stats=stats,
                plan=plan,
                output_root=output_root,
                source_root=source_root,
            )
        )
        output_stats.append(stats)
        if counter % 50 == 0 or counter == len(episode_indices):
            print(
                f"[{source_root.name}] exported {counter}/{len(episode_indices)} episodes "
                f"to {output_root.name}"
            )

    clipped_info = dict(info)
    clipped_info["dataset_name"] = f"{info.get('dataset_name') or source_root.name}_clipped"
    clipped_info["episode_count"] = len(output_episodes)
    clipped_info["frame_count"] = int(global_index)
    _write_json(output_root / "meta" / "info.json", clipped_info)
    _write_jsonl(output_root / "meta" / "episodes.jsonl", output_episodes)
    _write_jsonl(output_root / "meta" / "episodes_stats.jsonl", output_stats)
    return {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "episode_count": len(output_episodes),
        "frame_count": int(global_index),
        "trimmed_episode_count": int(trimmed_episode_count),
        "original_frame_count": int(exported_original_frame_count),
        "removed_frame_count": int(exported_removed_frame_count),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export clipped paper_fold datasets from the edge-stuck QC report."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--output-parent", type=Path, default=None)
    parser.add_argument("--suffix", default="_clipped")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-episodes", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else cwd / args.dataset_root
    report_path = args.report_path if args.report_path.is_absolute() else cwd / args.report_path
    output_parent = args.output_parent
    if output_parent is None:
        output_parent = dataset_root.parent / f"{dataset_root.name}_clipped"
    output_parent = output_parent if output_parent.is_absolute() else cwd / output_parent

    plans_by_root = load_clip_plans(report_path, cwd=cwd)
    summaries: list[dict[str, Any]] = []
    for source_root in sorted(plans_by_root):
        if not source_root.exists():
            raise FileNotFoundError(f"Source dataset root not found: {source_root}")
        output_root = output_parent / f"{source_root.name}{args.suffix}"
        summaries.append(
            export_dataset(
                source_root=source_root,
                output_root=output_root,
                plans=plans_by_root[source_root],
                overwrite=bool(args.overwrite),
                limit_episodes=int(args.limit_episodes),
            )
        )

    print("Exported clipped datasets:")
    for item in summaries:
        print(
            f"  {item['output_root']}: episodes={item['episode_count']} "
            f"frames={item['frame_count']} removed_frames={item['removed_frame_count']} "
            f"trimmed_episodes={item['trimmed_episode_count']}"
        )


if __name__ == "__main__":
    main()
