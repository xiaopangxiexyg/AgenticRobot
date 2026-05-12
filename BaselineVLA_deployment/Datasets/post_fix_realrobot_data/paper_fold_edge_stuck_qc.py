from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_DATASET_ROOT = Path("BaselineVLA_deployment/Data/RawData/RealRobot/paper_fold")
DEFAULT_OUTPUT_DIR = Path(
    "BaselineVLA_deployment/Datasets/post_fix_realrobot_data/outputs/paper_fold_edge_stuck_qc"
)

STATE_ARM_POSE_SLICES = {
    "left": slice(6, 13),
    "right": slice(25, 32),
}

AUTO_POSE_COLUMNS = {
    "left": (
        "observation.left_arm.pose",
        "observation.arm.left.current_tcp_pose",
    ),
    "right": (
        "observation.right_arm.pose",
        "observation.arm.right.current_tcp_pose",
    ),
}


@dataclass(frozen=True)
class EpisodeRef:
    dataset_root: Path
    dataset_name: str
    episode_index: int
    parquet_path: Path
    parquet_relpath: str
    video_relpath: str | None = None


@dataclass
class EpisodeData:
    ref: EpisodeRef
    timestamps: np.ndarray
    frame_indices: np.ndarray
    pose: np.ndarray
    pose_column: str
    video_paths: list[str | None]
    video_frame_indices: np.ndarray | None
    nominal_fps: float | None


@dataclass
class TrimResult:
    dataset_name: str
    dataset_root: str
    episode_index: int
    parquet_path: str
    parquet_relpath: str
    pose_column: str
    frame_count: int
    duration_sec: float
    median_dt_sec: float
    average_hz: float
    xyz_range_threshold_m: float
    rotation_range_threshold_deg: float
    min_stuck_sec: float
    min_keep_sec: float
    lead_stuck_raw_frames: int
    lead_stuck_raw_sec: float
    lead_stuck_xyz_range_m: float
    lead_stuck_rotation_range_deg: float
    trail_stuck_raw_frames: int
    trail_stuck_raw_sec: float
    trail_stuck_xyz_range_m: float
    trail_stuck_rotation_range_deg: float
    trim_start_frames: int
    trim_start_sec: float
    trim_end_frames: int
    trim_end_sec: float
    keep_start_frame: int
    keep_end_frame_exclusive: int
    keep_duration_sec: float
    total_xyz_range_m: float
    total_rotation_range_deg: float
    status: str


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to read LeRobot parquet files. "
            "On this workstation, /home/zc/anaconda3/envs/HumanData/bin/python has pyarrow."
        ) from exc
    return pq


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                yield json.loads(text)


def _discover_dataset_roots(root: Path) -> list[Path]:
    root = root.expanduser()
    if (root / "meta" / "info.json").exists():
        return [root]
    roots = sorted(path.parent.parent for path in root.glob("*/meta/info.json"))
    if not roots:
        raise FileNotFoundError(f"No LeRobot dataset roots found under {root}")
    return roots


def _discover_episodes(dataset_root: Path) -> list[EpisodeRef]:
    info = _load_json(dataset_root / "meta" / "info.json")
    dataset_name = str(info.get("dataset_name") or dataset_root.name)
    refs: list[EpisodeRef] = []
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    for item in _iter_jsonl(episodes_path):
        episode_index = int(item.get("episode_index", len(refs)))
        relpath = str(item.get("parquet_path") or "")
        if not relpath:
            chunk_size = int(info.get("chunk_size") or 1000)
            chunk_id = episode_index // max(1, chunk_size)
            relpath = f"data/chunk-{chunk_id:03d}/episode_{episode_index:06d}.parquet"
        video_paths = item.get("video_paths") or {}
        video_relpath = video_paths.get("front") if isinstance(video_paths, dict) else None
        refs.append(
            EpisodeRef(
                dataset_root=dataset_root,
                dataset_name=dataset_name,
                episode_index=episode_index,
                parquet_path=dataset_root / relpath,
                parquet_relpath=relpath,
                video_relpath=video_relpath,
            )
        )
    if refs:
        return refs

    for parquet_path in sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet")):
        match = re.search(r"episode_(\d+)\.parquet$", parquet_path.name)
        episode_index = int(match.group(1)) if match else len(refs)
        refs.append(
            EpisodeRef(
                dataset_root=dataset_root,
                dataset_name=dataset_name,
                episode_index=episode_index,
                parquet_path=parquet_path,
                parquet_relpath=str(parquet_path.relative_to(dataset_root)),
            )
        )
    return refs


def discover_episodes(dataset_root: Path) -> list[EpisodeRef]:
    refs: list[EpisodeRef] = []
    for root in _discover_dataset_roots(dataset_root):
        refs.extend(_discover_episodes(root))
    refs = [ref for ref in refs if ref.parquet_path.exists()]
    if not refs:
        raise FileNotFoundError(f"No episode parquet files found under {dataset_root}")
    return refs


def _resolve_pose_column(
    available_columns: set[str],
    *,
    pose_column: str,
    side: str,
) -> tuple[str, bool]:
    if pose_column != "auto":
        if pose_column not in available_columns:
            raise KeyError(f"Requested pose column {pose_column!r} not found.")
        return pose_column, False
    for candidate in AUTO_POSE_COLUMNS[side]:
        if candidate in available_columns:
            return candidate, False
    if "observation.state" in available_columns:
        return "observation.state", True
    raise KeyError(
        "Could not find a right-arm pose column. Tried "
        f"{AUTO_POSE_COLUMNS[side]} and observation.state."
    )


def _as_pose_array(values: list[Any], *, side: str, from_state: bool) -> np.ndarray:
    rows: list[np.ndarray] = []
    state_slice = STATE_ARM_POSE_SLICES[side]
    for value in values:
        if value is None:
            rows.append(np.full(7, np.nan, dtype=np.float64))
            continue
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if from_state:
            if vector.shape[0] < state_slice.stop:
                rows.append(np.full(7, np.nan, dtype=np.float64))
            else:
                rows.append(vector[state_slice].astype(np.float64, copy=False))
        elif vector.shape[0] >= 7:
            rows.append(vector[:7].astype(np.float64, copy=False))
        else:
            rows.append(np.full(7, np.nan, dtype=np.float64))
    return np.stack(rows, axis=0)


def _positive_median_dt(timestamps: np.ndarray, nominal_fps: float | None) -> float:
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if timestamps.shape[0] > 1:
        diffs = np.diff(timestamps)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(np.median(diffs))
    if nominal_fps and nominal_fps > 0:
        return 1.0 / float(nominal_fps)
    return 1.0 / 30.0


def read_episode(ref: EpisodeRef, *, pose_column: str, side: str) -> EpisodeData:
    pq = _require_pyarrow()
    schema = pq.read_schema(ref.parquet_path)
    available = set(schema.names)
    selected_pose_column, from_state = _resolve_pose_column(
        available,
        pose_column=pose_column,
        side=side,
    )
    columns = [selected_pose_column]
    for candidate in (
        "timestamp",
        "frame_index",
        "observation.images.front.path",
        "observation.images.front.video_frame_index",
    ):
        if candidate in available:
            columns.append(candidate)
    table = pq.read_table(ref.parquet_path, columns=columns)
    data = table.to_pydict()
    row_count = table.num_rows

    info_path = ref.dataset_root / "meta" / "info.json"
    info = _load_json(info_path) if info_path.exists() else {}
    nominal_fps = info.get("nominal_fps")
    nominal_fps = float(nominal_fps) if nominal_fps is not None else None

    timestamps = data.get("timestamp")
    if timestamps is None:
        dt = 1.0 / nominal_fps if nominal_fps and nominal_fps > 0 else 1.0 / 30.0
        timestamps_array = np.arange(row_count, dtype=np.float64) * dt
    else:
        timestamps_array = np.asarray(timestamps, dtype=np.float64)

    frame_indices = data.get("frame_index")
    if frame_indices is None:
        frame_indices_array = np.arange(row_count, dtype=np.int64)
    else:
        frame_indices_array = np.asarray(frame_indices, dtype=np.int64)

    pose = _as_pose_array(data[selected_pose_column], side=side, from_state=from_state)
    video_paths = data.get("observation.images.front.path")
    if video_paths is None:
        video_paths = [ref.video_relpath] * row_count
    video_frame_indices = data.get("observation.images.front.video_frame_index")
    video_frame_indices_array = (
        None if video_frame_indices is None else np.asarray(video_frame_indices, dtype=np.int64)
    )
    effective_pose_column = (
        f"{selected_pose_column}[{STATE_ARM_POSE_SLICES[side].start}:{STATE_ARM_POSE_SLICES[side].stop}]"
        if from_state
        else selected_pose_column
    )
    return EpisodeData(
        ref=ref,
        timestamps=timestamps_array,
        frame_indices=frame_indices_array,
        pose=pose,
        pose_column=effective_pose_column,
        video_paths=list(video_paths),
        video_frame_indices=video_frame_indices_array,
        nominal_fps=nominal_fps,
    )


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    out = np.full_like(quat, np.nan)
    valid = np.isfinite(norm[..., 0]) & (norm[..., 0] > 1e-12)
    if np.any(valid):
        out[valid] = quat[valid] / norm[valid]
    return out


def quat_angle_deg(quat: np.ndarray, reference: np.ndarray) -> np.ndarray:
    quat = _normalize_quat(np.asarray(quat, dtype=np.float64).reshape(-1, 4))
    reference = _normalize_quat(np.asarray(reference, dtype=np.float64).reshape(1, 4))[0]
    dots = np.abs(np.sum(quat * reference.reshape(1, 4), axis=1))
    dots = np.clip(dots, 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dots))


def _range_until_edge(
    pose: np.ndarray,
    *,
    from_start: bool,
    xyz_threshold_m: float,
    rotation_threshold_deg: float,
) -> tuple[int, float, float]:
    if pose.shape[0] == 0:
        return 0, 0.0, 0.0
    indices = range(pose.shape[0]) if from_start else range(pose.shape[0] - 1, -1, -1)
    anchor = pose[0] if from_start else pose[-1]
    count = 0
    max_xyz = 0.0
    max_rot = 0.0
    for index in indices:
        current = pose[index]
        xyz_dist = float(np.linalg.norm(current[:3] - anchor[:3]))
        rot_dist = float(quat_angle_deg(current[3:7].reshape(1, 4), anchor[3:7])[0])
        if not (math.isfinite(xyz_dist) and math.isfinite(rot_dist)):
            break
        if xyz_dist <= xyz_threshold_m and rot_dist <= rotation_threshold_deg:
            count += 1
            max_xyz = max(max_xyz, xyz_dist)
            max_rot = max(max_rot, rot_dist)
        else:
            break
    return count, max_xyz, max_rot


def analyze_episode(
    episode: EpisodeData,
    *,
    xyz_threshold_m: float,
    rotation_threshold_deg: float,
    min_stuck_sec: float,
    min_keep_sec: float,
) -> TrimResult:
    pose = episode.pose
    frame_count = int(pose.shape[0])
    median_dt_sec = _positive_median_dt(episode.timestamps, episode.nominal_fps)
    duration_sec = float(max(0.0, episode.timestamps[-1] - episode.timestamps[0])) if frame_count else 0.0
    average_hz = float(1.0 / median_dt_sec) if median_dt_sec > 0 else 0.0

    if frame_count == 0:
        return TrimResult(
            dataset_name=episode.ref.dataset_name,
            dataset_root=str(episode.ref.dataset_root),
            episode_index=episode.ref.episode_index,
            parquet_path=str(episode.ref.parquet_path),
            parquet_relpath=episode.ref.parquet_relpath,
            pose_column=episode.pose_column,
            frame_count=0,
            duration_sec=0.0,
            median_dt_sec=median_dt_sec,
            average_hz=average_hz,
            xyz_range_threshold_m=xyz_threshold_m,
            rotation_range_threshold_deg=rotation_threshold_deg,
            min_stuck_sec=min_stuck_sec,
            min_keep_sec=min_keep_sec,
            lead_stuck_raw_frames=0,
            lead_stuck_raw_sec=0.0,
            lead_stuck_xyz_range_m=0.0,
            lead_stuck_rotation_range_deg=0.0,
            trail_stuck_raw_frames=0,
            trail_stuck_raw_sec=0.0,
            trail_stuck_xyz_range_m=0.0,
            trail_stuck_rotation_range_deg=0.0,
            trim_start_frames=0,
            trim_start_sec=0.0,
            trim_end_frames=0,
            trim_end_sec=0.0,
            keep_start_frame=0,
            keep_end_frame_exclusive=0,
            keep_duration_sec=0.0,
            total_xyz_range_m=0.0,
            total_rotation_range_deg=0.0,
            status="empty",
        )

    finite_pose = np.all(np.isfinite(pose), axis=1)
    if not np.all(finite_pose):
        status = "nonfinite_pose"
    else:
        status = "ok"

    lead_raw, lead_xyz_range, lead_rot_range = _range_until_edge(
        pose,
        from_start=True,
        xyz_threshold_m=xyz_threshold_m,
        rotation_threshold_deg=rotation_threshold_deg,
    )
    trail_raw, trail_xyz_range, trail_rot_range = _range_until_edge(
        pose,
        from_start=False,
        xyz_threshold_m=xyz_threshold_m,
        rotation_threshold_deg=rotation_threshold_deg,
    )
    lead_raw_sec = float(lead_raw * median_dt_sec)
    trail_raw_sec = float(trail_raw * median_dt_sec)
    trim_start = lead_raw if lead_raw_sec >= min_stuck_sec else 0
    trim_end = trail_raw if trail_raw_sec >= min_stuck_sec else 0

    min_keep_frames = int(math.ceil(max(0.0, min_keep_sec) / median_dt_sec))
    if frame_count - trim_start - trim_end < min_keep_frames:
        status = "min_keep_guard"
        trim_start = 0
        trim_end = 0

    keep_start = int(trim_start)
    keep_end_exclusive = int(frame_count - trim_end)
    keep_duration_sec = float(max(0, keep_end_exclusive - keep_start) * median_dt_sec)
    xyz_from_start = np.linalg.norm(pose[:, :3] - pose[0, :3], axis=1)
    rot_from_start = quat_angle_deg(pose[:, 3:7], pose[0, 3:7])

    return TrimResult(
        dataset_name=episode.ref.dataset_name,
        dataset_root=str(episode.ref.dataset_root),
        episode_index=episode.ref.episode_index,
        parquet_path=str(episode.ref.parquet_path),
        parquet_relpath=episode.ref.parquet_relpath,
        pose_column=episode.pose_column,
        frame_count=frame_count,
        duration_sec=duration_sec,
        median_dt_sec=median_dt_sec,
        average_hz=average_hz,
        xyz_range_threshold_m=xyz_threshold_m,
        rotation_range_threshold_deg=rotation_threshold_deg,
        min_stuck_sec=min_stuck_sec,
        min_keep_sec=min_keep_sec,
        lead_stuck_raw_frames=int(lead_raw),
        lead_stuck_raw_sec=lead_raw_sec,
        lead_stuck_xyz_range_m=float(lead_xyz_range),
        lead_stuck_rotation_range_deg=float(lead_rot_range),
        trail_stuck_raw_frames=int(trail_raw),
        trail_stuck_raw_sec=trail_raw_sec,
        trail_stuck_xyz_range_m=float(trail_xyz_range),
        trail_stuck_rotation_range_deg=float(trail_rot_range),
        trim_start_frames=int(trim_start),
        trim_start_sec=float(trim_start * median_dt_sec),
        trim_end_frames=int(trim_end),
        trim_end_sec=float(trim_end * median_dt_sec),
        keep_start_frame=keep_start,
        keep_end_frame_exclusive=keep_end_exclusive,
        keep_duration_sec=keep_duration_sec,
        total_xyz_range_m=float(np.nanmax(xyz_from_start)),
        total_rotation_range_deg=float(np.nanmax(rot_from_start)),
        status=status,
    )


def summarize_results(results: list[TrimResult]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "episode_count": len(results),
        "trimmed_episode_count": int(
            sum(1 for row in results if row.trim_start_frames > 0 or row.trim_end_frames > 0)
        ),
        "trim_start_episode_count": int(sum(1 for row in results if row.trim_start_frames > 0)),
        "trim_end_episode_count": int(sum(1 for row in results if row.trim_end_frames > 0)),
        "total_duration_sec": float(sum(row.duration_sec for row in results)),
        "total_trim_start_sec": float(sum(row.trim_start_sec for row in results)),
        "total_trim_end_sec": float(sum(row.trim_end_sec for row in results)),
        "status_counts": {},
        "by_dataset": {},
    }
    for row in results:
        summary["status_counts"][row.status] = int(summary["status_counts"].get(row.status, 0) + 1)
        group = summary["by_dataset"].setdefault(
            row.dataset_name,
            {
                "episode_count": 0,
                "trimmed_episode_count": 0,
                "trim_start_episode_count": 0,
                "trim_end_episode_count": 0,
                "total_duration_sec": 0.0,
                "total_trim_start_sec": 0.0,
                "total_trim_end_sec": 0.0,
            },
        )
        group["episode_count"] += 1
        group["trimmed_episode_count"] += int(row.trim_start_frames > 0 or row.trim_end_frames > 0)
        group["trim_start_episode_count"] += int(row.trim_start_frames > 0)
        group["trim_end_episode_count"] += int(row.trim_end_frames > 0)
        group["total_duration_sec"] += float(row.duration_sec)
        group["total_trim_start_sec"] += float(row.trim_start_sec)
        group["total_trim_end_sec"] += float(row.trim_end_sec)
    return summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _relative_seconds(timestamps: np.ndarray) -> np.ndarray:
    if timestamps.size == 0:
        return timestamps
    return timestamps - timestamps[0]


def _load_video_rgb(video_path: Path, frame_index: int, *, height: int = 180, width: int = 240) -> np.ndarray:
    import cv2

    if not video_path.exists():
        return np.zeros((height, width, 3), dtype=np.uint8)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((height, width, 3), dtype=np.uint8)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _frame_video_location(episode: EpisodeData, frame: int) -> tuple[Path | None, int]:
    video_rel = None
    for candidate in (episode.video_paths[frame], episode.ref.video_relpath):
        if candidate:
            video_rel = str(candidate)
            break
    if not video_rel:
        return None, int(frame)
    video_frame_index = int(frame)
    if episode.video_frame_indices is not None and frame < episode.video_frame_indices.shape[0]:
        video_frame_index = int(episode.video_frame_indices[frame])
    return episode.ref.dataset_root / video_rel, video_frame_index


def visualize_episode(
    episode: EpisodeData,
    result: TrimResult,
    *,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = output_path.parent / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = _relative_seconds(episode.timestamps)
    pose = episode.pose
    xyz = pose[:, :3]
    step = np.zeros(pose.shape[0], dtype=np.float64)
    if pose.shape[0] > 1:
        step[1:] = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    frame_candidates = [
        0,
        result.keep_start_frame,
        (result.keep_start_frame + max(result.keep_start_frame, result.keep_end_frame_exclusive - 1)) // 2,
        max(result.keep_start_frame, result.keep_end_frame_exclusive - 1),
        pose.shape[0] - 1,
    ]
    frames = sorted({int(np.clip(frame, 0, pose.shape[0] - 1)) for frame in frame_candidates})

    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    grid = fig.add_gridspec(3, max(1, len(frames)), height_ratios=[1.25, 0.8, 1.1])
    ax_xyz = fig.add_subplot(grid[0, :])
    ax_step = fig.add_subplot(grid[1, :], sharex=ax_xyz)

    labels = ("x", "y", "z")
    for i, label in enumerate(labels):
        ax_xyz.plot(t, xyz[:, i], label=label, linewidth=1.5)
    for axis in (ax_xyz, ax_step):
        if result.trim_start_frames > 0:
            axis.axvspan(t[0], t[min(result.trim_start_frames, len(t) - 1)], color="#d62728", alpha=0.16)
        if result.trim_end_frames > 0:
            end_start = max(0, len(t) - result.trim_end_frames)
            axis.axvspan(t[end_start], t[-1], color="#ff7f0e", alpha=0.18)
        axis.grid(True, alpha=0.25)

    ax_xyz.set_title(
        f"{result.dataset_name} episode_{result.episode_index:06d} | "
        f"trim_start={result.trim_start_sec:.2f}s trim_end={result.trim_end_sec:.2f}s"
    )
    ax_xyz.set_ylabel("right arm xyz (m)")
    ax_xyz.legend(loc="best", ncols=3)
    ax_step.plot(t, step, color="#222222", linewidth=1.1)
    ax_step.axhline(result.xyz_range_threshold_m, color="#d62728", linestyle="--", linewidth=1.0)
    ax_step.set_ylabel("frame delta (m)")
    ax_step.set_xlabel("relative time (s)")

    for col, frame in enumerate(frames):
        ax_img = fig.add_subplot(grid[2, col])
        video_path, video_frame_index = _frame_video_location(episode, frame)
        if video_path is None:
            image = np.zeros((180, 240, 3), dtype=np.uint8)
        else:
            image = _load_video_rgb(video_path, video_frame_index)
        ax_img.imshow(image)
        ax_img.set_title(f"frame {frame}\nt={t[frame]:.2f}s")
        ax_img.axis("off")

    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def select_visualization_samples(
    results: list[TrimResult],
    *,
    sample_count: int,
    sample_seed: int,
    sample_from: str,
) -> list[TrimResult]:
    if sample_count <= 0:
        return []
    if sample_from == "trimmed":
        population = [row for row in results if row.trim_start_frames > 0 or row.trim_end_frames > 0]
    elif sample_from == "leading":
        population = [row for row in results if row.trim_start_frames > 0]
    elif sample_from == "trailing":
        population = [row for row in results if row.trim_end_frames > 0]
    elif sample_from == "all":
        population = list(results)
    else:
        raise ValueError(f"Unsupported sample_from={sample_from!r}")
    if not population and sample_from != "all":
        population = list(results)
    rng = random.Random(int(sample_seed))
    return rng.sample(population, min(int(sample_count), len(population)))


def print_summary(summary: dict[str, Any], *, output_dir: Path) -> None:
    print("Paper-fold stuck-edge QC")
    print(f"  episodes: {summary['episode_count']}")
    print(f"  trimmed episodes: {summary['trimmed_episode_count']}")
    print(
        "  total trim seconds: "
        f"start={summary['total_trim_start_sec']:.2f}, "
        f"end={summary['total_trim_end_sec']:.2f}"
    )
    print("  by dataset:")
    for dataset_name, row in sorted(summary["by_dataset"].items()):
        print(
            f"    {dataset_name}: episodes={row['episode_count']} "
            f"trimmed={row['trimmed_episode_count']} "
            f"start_sec={row['total_trim_start_sec']:.2f} "
            f"end_sec={row['total_trim_end_sec']:.2f}"
        )
    print(f"  output: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect still leading/trailing segments in paper_fold episodes using "
            "the right-arm end-effector pose, and generate random QC visualizations."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--pose-column",
        default="auto",
        help="Pose column to inspect. Use 'auto' for observation.<side>_arm.pose.",
    )
    parser.add_argument("--xyz-range-threshold-m", type=float, default=0.01)
    parser.add_argument("--rotation-range-threshold-deg", type=float, default=2.0)
    parser.add_argument("--min-stuck-sec", type=float, default=0.5)
    parser.add_argument("--min-keep-sec", type=float, default=2.0)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=17)
    parser.add_argument(
        "--sample-from",
        choices=("all", "trimmed", "leading", "trailing"),
        default="all",
    )
    parser.add_argument("--limit-episodes", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    refs = discover_episodes(args.dataset_root)
    if args.limit_episodes > 0:
        refs = refs[: int(args.limit_episodes)]

    results: list[TrimResult] = []
    for ref in refs:
        episode = read_episode(ref, pose_column=args.pose_column, side=args.side)
        results.append(
            analyze_episode(
                episode,
                xyz_threshold_m=float(args.xyz_range_threshold_m),
                rotation_threshold_deg=float(args.rotation_range_threshold_deg),
                min_stuck_sec=float(args.min_stuck_sec),
                min_keep_sec=float(args.min_keep_sec),
            )
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    row_dicts = [asdict(row) for row in results]
    report_jsonl = output_dir / "paper_fold_edge_trim_report.jsonl"
    report_csv = output_dir / "paper_fold_edge_trim_report.csv"
    summary_json = output_dir / "summary.json"
    write_jsonl(report_jsonl, row_dicts)
    write_csv(report_csv, row_dicts)
    summary = summarize_results(results)
    summary["config"] = {
        "dataset_root": str(args.dataset_root),
        "side": args.side,
        "pose_column": args.pose_column,
        "xyz_range_threshold_m": float(args.xyz_range_threshold_m),
        "rotation_range_threshold_deg": float(args.rotation_range_threshold_deg),
        "min_stuck_sec": float(args.min_stuck_sec),
        "min_keep_sec": float(args.min_keep_sec),
        "sample_count": int(args.sample_count),
        "sample_seed": int(args.sample_seed),
        "sample_from": args.sample_from,
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")

    samples = select_visualization_samples(
        results,
        sample_count=int(args.sample_count),
        sample_seed=int(args.sample_seed),
        sample_from=str(args.sample_from),
    )
    samples_output: list[dict[str, Any]] = []
    result_by_key = {(row.dataset_root, row.episode_index): row for row in results}
    ref_by_key = {(str(ref.dataset_root), ref.episode_index): ref for ref in refs}
    for result in samples:
        key = (result.dataset_root, result.episode_index)
        ref = ref_by_key[key]
        episode = read_episode(ref, pose_column=args.pose_column, side=args.side)
        image_name = f"{_safe_stem(result.dataset_name)}_episode_{result.episode_index:06d}.png"
        image_path = output_dir / "visualizations" / image_name
        visualize_episode(episode, result_by_key[key], output_path=image_path)
        item = asdict(result)
        item["visualization_path"] = str(image_path)
        samples_output.append(item)
    (output_dir / "visualization_samples.json").write_text(
        json.dumps(samples_output, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print_summary(summary, output_dir=output_dir)


if __name__ == "__main__":
    main()
