from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SOURCE_ROOT = Path("BaselineVLA_deployment/Data/RawData/RealRobot/paper_fold_clipped")

ARM_POSE_SLICES = {
    "left": slice(0, 7),
    "right": slice(7, 14),
}
ACTION_HAND_SLICES = {
    "left": slice(14, 20),
    "right": slice(20, 26),
}
STATE_ARM_QPOS_SLICES = {
    "left": slice(0, 6),
    "right": slice(19, 25),
}
STATE_ARM_POSE_SLICES = {
    "left": slice(6, 13),
    "right": slice(25, 32),
}
STATE_HAND_SLICES = {
    "left": slice(13, 19),
    "right": slice(32, 38),
}


@dataclass(frozen=True)
class EpisodeStart:
    episode_index: int
    arm_pose: np.ndarray
    arm_qpos: np.ndarray
    hand_qpos: np.ndarray
    action_hand: np.ndarray


@dataclass(frozen=True)
class ReturnPlan:
    source_episode_index: int
    target_episode_index: int
    return_duration_sec: float
    base_duration_sec: float
    append_dt_sec: float
    trans_distance_m: float
    rot_distance_rad: float
    max_trans_speed_mps: float
    max_rot_speed_radps: float


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
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
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


def _copy_meta_static_files(source_root: Path, output_root: Path) -> None:
    source_meta = source_root / "meta"
    output_meta = output_root / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)
    skip = {"info.json", "episodes.jsonl", "episodes_stats.jsonl", "return8_target_mapping.json"}
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


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(-1, 4)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    out = quat.copy()
    valid = norm[:, 0] > 1e-12
    out[valid] /= norm[valid]
    out[~valid] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return out


def _quat_angle_rad(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = _normalize_quat(left)
    right = _normalize_quat(right)
    dots = np.abs(np.sum(left * right, axis=1))
    dots = np.clip(dots, 0.0, 1.0)
    return 2.0 * np.arccos(dots)


def _slerp_quat(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize_quat(np.asarray(start, dtype=np.float64).reshape(1, 4))[0]
    q1 = _normalize_quat(np.asarray(end, dtype=np.float64).reshape(1, 4))[0]
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        out = q0 + float(alpha) * (q1 - q0)
        return _normalize_quat(out.reshape(1, 4))[0]
    theta_0 = math.acos(dot)
    theta = theta_0 * float(alpha)
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return _normalize_quat((s0 * q0 + s1 * q1).reshape(1, 4))[0]


def _interp_vector(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    return np.asarray(start, dtype=np.float64) + float(alpha) * (
        np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    )


def _interp_pose(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    start = np.asarray(start, dtype=np.float64).reshape(7)
    end = np.asarray(end, dtype=np.float64).reshape(7)
    xyz = _interp_vector(start[:3], end[:3], alpha)
    quat = _slerp_quat(start[3:7], end[3:7], alpha)
    return np.concatenate([xyz, quat], dtype=np.float64)


def _as_vector(value: Any, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] < int(length):
        raise ValueError(f"Expected vector length >= {length}, got {arr.shape[0]}")
    return arr[: int(length)].astype(np.float64, copy=True)


def _table_rows(path: Path, columns: list[str] | None = None):
    _, pq = _require_pyarrow()
    return pq.read_table(path, columns=columns)


def _episode_path(dataset_root: Path, episode_index: int, *, chunk_size: int) -> Path:
    return dataset_root / _episode_relpath(episode_index, chunk_size=chunk_size)


def _load_episode_starts(dataset_root: Path, *, side: str, chunk_size: int) -> dict[int, EpisodeStart]:
    episodes = _iter_jsonl(dataset_root / "meta" / "episodes.jsonl")
    starts: dict[int, EpisodeStart] = {}
    columns = [
        f"observation.{side}_arm.pose",
        f"observation.{side}_arm.qpos",
        f"observation.{side}_hand.qpos",
        f"action.{side}_hand.joint_target",
    ]
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        table = _table_rows(dataset_root / str(episode["parquet_path"]), columns=columns)
        row = table.slice(0, 1).to_pylist()[0]
        starts[episode_index] = EpisodeStart(
            episode_index=episode_index,
            arm_pose=_as_vector(row[f"observation.{side}_arm.pose"], 7),
            arm_qpos=_as_vector(row[f"observation.{side}_arm.qpos"], 6),
            hand_qpos=_as_vector(row[f"observation.{side}_hand.qpos"], 6),
            action_hand=_as_vector(row[f"action.{side}_hand.joint_target"], 6),
        )
    return starts


def _make_target_mapping(episode_indices: list[int], *, seed: int, allow_self: bool) -> dict[int, int]:
    indices = list(sorted(int(index) for index in episode_indices))
    if not indices:
        return {}
    if len(indices) == 1:
        return {indices[0]: indices[0]}
    rng = random.Random(int(seed))
    targets = indices.copy()
    if allow_self:
        rng.shuffle(targets)
        return dict(zip(indices, targets))
    for _ in range(1000):
        rng.shuffle(targets)
        if all(source != target for source, target in zip(indices, targets)):
            return dict(zip(indices, targets))
    shifted = indices[1:] + indices[:1]
    return dict(zip(indices, shifted))


def _mean_dt_from_tail(timestamps: list[Any], nominal_fps: float) -> float:
    values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if values.shape[0] >= 2 and np.isfinite(values[-1]) and np.isfinite(values[-2]) and values[-1] > values[-2]:
        return float(values[-1] - values[-2])
    return float(1.0 / nominal_fps) if nominal_fps > 0 else 0.1


def _mean_numeric_sequence(left: list[Any], right: list[Any]) -> list[float]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return ((a + b) * 0.5).astype(np.float64).tolist()


def _tail_average_value(values: list[Any]) -> Any:
    if not values:
        return None
    last = values[-1]
    if len(values) == 1:
        return last
    prev = values[-2]
    if last is None or prev is None:
        return last
    if isinstance(last, bool):
        return bool(last)
    if isinstance(last, int) and isinstance(prev, int):
        return int(round((prev + last) * 0.5))
    if isinstance(last, float) or isinstance(prev, float):
        try:
            return float((float(prev) + float(last)) * 0.5)
        except Exception:
            return last
    if isinstance(last, list) and isinstance(prev, list) and len(last) == len(prev):
        if not last:
            return []
        if all(isinstance(item, (int, float)) or item is None for item in last + prev):
            try:
                return _mean_numeric_sequence(prev, last)
            except Exception:
                return list(last)
        return list(last)
    return last


def _make_base_append_rows(data: dict[str, list[Any]], append_count: int) -> list[dict[str, Any]]:
    base = {name: _tail_average_value(values) for name, values in data.items()}
    return [{name: _copy_jsonable(value) for name, value in base.items()} for _ in range(int(append_count))]


def _copy_jsonable(value: Any) -> Any:
    if isinstance(value, list):
        return [_copy_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _copy_jsonable(item) for key, item in value.items()}
    return value


def _max_pose_speeds(pose: np.ndarray, timestamps: np.ndarray) -> tuple[float, float]:
    pose = np.asarray(pose, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if pose.shape[0] < 2:
        return 0.0, 0.0
    dt = np.diff(timestamps)
    valid = np.isfinite(dt) & (dt > 1e-9)
    if not np.any(valid):
        return 0.0, 0.0
    trans_steps = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1)
    rot_steps = _quat_angle_rad(pose[1:, 3:7], pose[:-1, 3:7])
    trans_speed = np.zeros_like(trans_steps, dtype=np.float64)
    rot_speed = np.zeros_like(rot_steps, dtype=np.float64)
    trans_speed[valid] = trans_steps[valid] / dt[valid]
    rot_speed[valid] = rot_steps[valid] / dt[valid]
    return float(np.nanmax(trans_speed)), float(np.nanmax(rot_speed))


def _build_return_plan(
    *,
    source_episode_index: int,
    target_episode_index: int,
    final_pose: np.ndarray,
    target_pose: np.ndarray,
    pose_history: np.ndarray,
    timestamps: np.ndarray,
    append_dt_sec: float,
    base_duration_sec: float,
    speed_eps: float,
) -> ReturnPlan:
    max_trans_speed, max_rot_speed = _max_pose_speeds(pose_history, timestamps)
    trans_distance = float(np.linalg.norm(np.asarray(target_pose[:3]) - np.asarray(final_pose[:3])))
    rot_distance = float(_quat_angle_rad(np.asarray(target_pose[3:7]).reshape(1, 4), np.asarray(final_pose[3:7]).reshape(1, 4))[0])
    duration = float(base_duration_sec)
    if trans_distance > speed_eps:
        if max_trans_speed <= speed_eps:
            raise ValueError(f"Episode {source_episode_index} has zero translation speed but nonzero target distance.")
        duration = max(duration, trans_distance / max_trans_speed)
    if rot_distance > speed_eps:
        if max_rot_speed <= speed_eps:
            raise ValueError(f"Episode {source_episode_index} has zero rotation speed but nonzero target distance.")
        duration = max(duration, rot_distance / max_rot_speed)
    duration = max(duration, append_dt_sec)
    return ReturnPlan(
        source_episode_index=int(source_episode_index),
        target_episode_index=int(target_episode_index),
        return_duration_sec=float(duration),
        base_duration_sec=float(base_duration_sec),
        append_dt_sec=float(append_dt_sec),
        trans_distance_m=trans_distance,
        rot_distance_rad=rot_distance,
        max_trans_speed_mps=float(max_trans_speed),
        max_rot_speed_radps=float(max_rot_speed),
    )


def _set_vector(row: dict[str, Any], key: str, value: np.ndarray) -> None:
    if key in row:
        row[key] = np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _set_action_slice(row: dict[str, Any], slc: slice, value: np.ndarray) -> None:
    if "action" not in row or row["action"] is None:
        return
    vector = list(row["action"])
    vector[slc] = np.asarray(value, dtype=np.float64).reshape(-1).tolist()
    row["action"] = vector


def _set_state_slice(row: dict[str, Any], slc: slice, value: np.ndarray) -> None:
    if "observation.state" not in row or row["observation.state"] is None:
        return
    vector = list(row["observation.state"])
    vector[slc] = np.asarray(value, dtype=np.float64).reshape(-1).tolist()
    row["observation.state"] = vector


def _update_generated_row(
    row: dict[str, Any],
    *,
    side: str,
    generated_pose: np.ndarray,
    generated_arm_qpos: np.ndarray,
    generated_state_hand: np.ndarray,
    generated_action_hand: np.ndarray,
    previous_action: list[float],
) -> None:
    _set_vector(row, f"observation.{side}_arm.pose", generated_pose)
    _set_vector(row, f"observation.arm.{side}.current_tcp_pose", generated_pose)
    _set_vector(row, f"action.{side}_arm.pose", generated_pose)
    _set_vector(row, f"teleop.action_source.{side}_wrist_pose_base", generated_pose)
    _set_action_slice(row, ARM_POSE_SLICES[side], generated_pose)
    _set_state_slice(row, STATE_ARM_POSE_SLICES[side], generated_pose)

    _set_vector(row, f"observation.{side}_arm.qpos", generated_arm_qpos)
    _set_vector(row, f"observation.arm.{side}.current_qpos", generated_arm_qpos)
    _set_vector(row, f"observation.arm.{side}.last_target_qpos", generated_arm_qpos)
    _set_vector(row, f"action.replay.{side}_arm_joint_target", generated_arm_qpos)
    _set_state_slice(row, STATE_ARM_QPOS_SLICES[side], generated_arm_qpos)

    _set_vector(row, f"observation.{side}_hand.qpos", generated_state_hand)
    _set_vector(row, f"observation.hand.{side}.joint_pos", generated_state_hand)
    _set_state_slice(row, STATE_HAND_SLICES[side], generated_state_hand)

    _set_vector(row, f"action.{side}_hand.joint_target", generated_action_hand)
    _set_vector(row, f"action.replay.{side}_hand_joint_target", generated_action_hand)
    _set_vector(row, f"teleop.action_source.{side}_hand_q6", generated_action_hand)
    _set_action_slice(row, ACTION_HAND_SLICES[side], generated_action_hand)
    if "observation.last_action" in row:
        row["observation.last_action"] = list(previous_action)


def _append_rows_to_columns(data: dict[str, list[Any]], rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    out = {name: list(values) for name, values in data.items()}
    for row in rows:
        for name in out:
            out[name].append(row.get(name))
    return out


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
        diffs = np.diff(timestamps)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        average_dt_sec = float(np.mean(diffs)) if diffs.size else (1.0 / nominal_fps if nominal_fps > 0 else 0.0)
    else:
        average_dt_sec = float(1.0 / nominal_fps) if nominal_fps > 0 else 0.0
        start_timestamp = 0.0
        duration_sec = float(max(0, frame_count - 1) * average_dt_sec)
        end_timestamp = float(duration_sec)
    control_hz_values = np.asarray(rows.get("control_hz", []), dtype=np.float64)
    control_hz_values = control_hz_values[np.isfinite(control_hz_values)]
    return {
        "episode_index": int(episode_index),
        "frame_count": int(frame_count),
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "duration_sec": duration_sec,
        "average_dt_sec": average_dt_sec,
        "average_hz": float(1.0 / average_dt_sec) if average_dt_sec > 0 else 0.0,
        "mean_control_hz": float(np.mean(control_hz_values)) if control_hz_values.size else 0.0,
        "nominal_fps": float(nominal_fps),
        "active_sides": rows.get("active_sides", [[]])[0] if rows.get("active_sides") else [],
        "arm_state_source": rows.get("arm_state_source", [""])[0] if rows.get("arm_state_source") else "",
        "camera_keys": rows.get("camera_keys", [[]])[0] if rows.get("camera_keys") else [],
        "fault_count": int(sum(len(item) if isinstance(item, list) else bool(item) for item in rows.get("fault_codes", []))),
    }


def _rewrite_video_with_tail_repeats(
    *,
    source_video: Path,
    output_video: Path,
    append_count: int,
) -> None:
    import cv2

    output_video.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {source_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer: {output_video}")
    last_frame = None
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            writer.write(frame)
            last_frame = frame
            written += 1
        if last_frame is None:
            raise RuntimeError(f"Source video is empty: {source_video}")
        for _ in range(int(append_count)):
            writer.write(last_frame)
            written += 1
    finally:
        writer.release()
        cap.release()


def _source_video_path(dataset_root: Path, data: dict[str, list[Any]], episode_index: int, *, chunk_size: int) -> Path:
    paths = data.get("observation.images.front.path") or []
    for item in reversed(paths):
        if item:
            return dataset_root / str(item)
    return dataset_root / _video_relpath(episode_index, chunk_size=chunk_size)


def _build_generated_rows(
    *,
    data: dict[str, list[Any]],
    side: str,
    episode_index: int,
    target_start: EpisodeStart,
    append_count: int,
    base_duration_sec: float,
    nominal_fps: float,
    speed_eps: float,
) -> tuple[list[dict[str, Any]], ReturnPlan]:
    pose_history = np.asarray(data[f"observation.{side}_arm.pose"], dtype=np.float64)
    timestamps = np.asarray(data["timestamp"], dtype=np.float64)
    final_pose = pose_history[-1]
    final_arm_qpos = _as_vector(data[f"observation.{side}_arm.qpos"][-1], 6)
    final_state_hand = _as_vector(data[f"observation.{side}_hand.qpos"][-1], 6)
    final_action_hand = _as_vector(data[f"action.{side}_hand.joint_target"][-1], 6)
    append_dt = _mean_dt_from_tail(data.get("timestamp", []), nominal_fps)
    plan = _build_return_plan(
        source_episode_index=episode_index,
        target_episode_index=target_start.episode_index,
        final_pose=final_pose,
        target_pose=target_start.arm_pose,
        pose_history=pose_history,
        timestamps=timestamps,
        append_dt_sec=append_dt,
        base_duration_sec=base_duration_sec,
        speed_eps=speed_eps,
    )
    rows = _make_base_append_rows(data, append_count)
    previous_action = list(data["action"][-1])
    original_frame_count = len(data["timestamp"])
    last_timestamp = float(data["timestamp"][-1])
    last_video_frame = int(data.get("observation.images.front.video_frame_index", [original_frame_count - 1])[-1])
    last_source_frame = int(data.get("observation.images.front.source_frame_index", [original_frame_count - 1])[-1])
    video_path = data.get("observation.images.front.path", [None])[-1]
    for offset, row in enumerate(rows, start=1):
        alpha = min(1.0, float(offset) * append_dt / plan.return_duration_sec)
        generated_pose = _interp_pose(final_pose, target_start.arm_pose, alpha)
        generated_arm_qpos = _interp_vector(final_arm_qpos, target_start.arm_qpos, alpha)
        generated_state_hand = _interp_vector(final_state_hand, target_start.hand_qpos, alpha)
        generated_action_hand = _interp_vector(final_action_hand, target_start.action_hand, alpha)
        _update_generated_row(
            row,
            side=side,
            generated_pose=generated_pose,
            generated_arm_qpos=generated_arm_qpos,
            generated_state_hand=generated_state_hand,
            generated_action_hand=generated_action_hand,
            previous_action=previous_action,
        )
        frame_index = original_frame_count + offset - 1
        if "frame_index" in row:
            row["frame_index"] = frame_index
        if "timestamp" in row:
            row["timestamp"] = last_timestamp + float(offset) * append_dt
        if "observation.images.front.video_frame_index" in row:
            row["observation.images.front.video_frame_index"] = last_video_frame + offset
        if "observation.images.front.source_frame_index" in row:
            row["observation.images.front.source_frame_index"] = last_source_frame + offset
        if "observation.images.front.path" in row and video_path is not None:
            row["observation.images.front.path"] = video_path
        previous_action = list(row["action"])
    return rows, plan


def _update_episode_record(
    original: dict[str, Any],
    *,
    stats: dict[str, Any],
    return_plan: ReturnPlan,
    output_root: Path,
    source_root: Path,
    append_count: int,
) -> dict[str, Any]:
    updated = dict(original)
    updated["frame_count"] = int(stats["frame_count"])
    updated["start_timestamp"] = float(stats["start_timestamp"])
    updated["end_timestamp"] = float(stats["end_timestamp"])
    updated["duration_sec"] = float(stats["duration_sec"])
    metadata = dict(updated.get("metadata") or {})
    metadata["return8_source_dataset_root"] = str(source_root)
    metadata["return8_dataset_root"] = str(output_root)
    metadata["return8_target_episode_index"] = int(return_plan.target_episode_index)
    metadata["return8_append_frame_count"] = int(append_count)
    metadata["return8_base_duration_sec"] = float(return_plan.base_duration_sec)
    metadata["return8_duration_sec"] = float(return_plan.return_duration_sec)
    metadata["return8_trans_distance_m"] = float(return_plan.trans_distance_m)
    metadata["return8_rot_distance_rad"] = float(return_plan.rot_distance_rad)
    metadata["return8_max_trans_speed_mps"] = float(return_plan.max_trans_speed_mps)
    metadata["return8_max_rot_speed_radps"] = float(return_plan.max_rot_speed_radps)
    metadata["dataset_dir_name"] = output_root.name
    updated["metadata"] = metadata
    summary = updated.get("summary")
    if isinstance(summary, dict):
        summary = dict(summary)
        summary["episode_frame_count"] = int(stats["frame_count"])
        summary["submitted_samples_total"] = int(stats["frame_count"])
        updated["summary"] = summary
    return updated


def _export_episode(
    *,
    source_root: Path,
    output_root: Path,
    episode_record: dict[str, Any],
    target_start: EpisodeStart,
    side: str,
    chunk_size: int,
    nominal_fps: float,
    append_count: int,
    base_duration_sec: float,
    global_index_start: int,
    speed_eps: float,
) -> tuple[dict[str, Any], dict[str, Any], ReturnPlan, int]:
    pa, pq = _require_pyarrow()
    episode_index = int(episode_record["episode_index"])
    source_parquet = source_root / str(episode_record["parquet_path"])
    table = pq.read_table(source_parquet)
    data = table.to_pydict()
    generated_rows, return_plan = _build_generated_rows(
        data=data,
        side=side,
        episode_index=episode_index,
        target_start=target_start,
        append_count=append_count,
        base_duration_sec=base_duration_sec,
        nominal_fps=nominal_fps,
        speed_eps=speed_eps,
    )
    full_data = _append_rows_to_columns(data, generated_rows)
    frame_count = len(full_data["timestamp"])
    full_data["index"] = list(range(global_index_start, global_index_start + frame_count))
    full_data["episode_index"] = [episode_index] * frame_count
    full_data["frame_index"] = list(range(frame_count))
    video_relpath = _video_relpath(episode_index, chunk_size=chunk_size)
    if "observation.images.front.path" in full_data:
        full_data["observation.images.front.path"] = [video_relpath] * frame_count
    if "observation.images.front.video_frame_index" in full_data:
        full_data["observation.images.front.video_frame_index"] = list(range(frame_count))

    output_parquet_relpath = _episode_relpath(episode_index, chunk_size=chunk_size)
    output_parquet = output_root / output_parquet_relpath
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(full_data, schema=table.schema), output_parquet)

    source_video = _source_video_path(source_root, data, episode_index, chunk_size=chunk_size)
    output_video = output_root / video_relpath
    if source_video.exists():
        _rewrite_video_with_tail_repeats(source_video=source_video, output_video=output_video, append_count=append_count)

    stats = _stats_from_rows(full_data, episode_index=episode_index, nominal_fps=nominal_fps)
    updated_episode = dict(episode_record)
    updated_episode["parquet_path"] = output_parquet_relpath
    updated_episode["video_paths"] = {"front": video_relpath}
    updated_episode = _update_episode_record(
        updated_episode,
        stats=stats,
        return_plan=return_plan,
        output_root=output_root,
        source_root=source_root,
        append_count=append_count,
    )
    return updated_episode, stats, return_plan, frame_count


def export_dataset(
    *,
    source_root: Path,
    output_root: Path,
    side: str,
    append_count: int,
    base_duration_sec: float,
    seed: int,
    allow_self_target: bool,
    overwrite: bool,
    limit_episodes: int,
    speed_eps: float,
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
    episodes = _iter_jsonl(source_root / "meta" / "episodes.jsonl")
    if limit_episodes > 0:
        episodes = episodes[: int(limit_episodes)]
    episode_indices = [int(episode["episode_index"]) for episode in episodes]
    starts = _load_episode_starts(source_root, side=side, chunk_size=chunk_size)
    mapping = _make_target_mapping(
        episode_indices,
        seed=int(seed),
        allow_self=bool(allow_self_target),
    )

    output_episodes: list[dict[str, Any]] = []
    output_stats: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    global_index = 0
    for counter, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        target_episode_index = int(mapping[episode_index])
        updated_episode, stats, return_plan, frame_count = _export_episode(
            source_root=source_root,
            output_root=output_root,
            episode_record=episode,
            target_start=starts[target_episode_index],
            side=side,
            chunk_size=chunk_size,
            nominal_fps=nominal_fps,
            append_count=append_count,
            base_duration_sec=base_duration_sec,
            global_index_start=global_index,
            speed_eps=speed_eps,
        )
        global_index += frame_count
        output_episodes.append(updated_episode)
        output_stats.append(stats)
        mapping_rows.append({**return_plan.__dict__, "append_count": int(append_count), "side": side})
        if counter % 50 == 0 or counter == len(episodes):
            print(f"[{source_root.name}] exported {counter}/{len(episodes)} episodes to {output_root.name}")

    output_info = dict(info)
    output_info["dataset_name"] = f"{info.get('dataset_name') or source_root.name}_return8"
    output_info["episode_count"] = len(output_episodes)
    output_info["frame_count"] = int(global_index)
    _write_json(output_root / "meta" / "info.json", output_info)
    _write_jsonl(output_root / "meta" / "episodes.jsonl", output_episodes)
    _write_jsonl(output_root / "meta" / "episodes_stats.jsonl", output_stats)
    _write_json(output_root / "meta" / "return8_target_mapping.json", mapping_rows)
    return {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "episode_count": len(output_episodes),
        "frame_count": int(global_index),
        "added_frame_count": int(len(output_episodes) * append_count),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append 8 low-information return-motion frames to each paper_fold_clipped episode. "
            "The generated motion moves from the episode final pose toward a randomly assigned "
            "target episode's start pose within the same subset."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--child-suffix", default="_return8")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--append-count", type=int, default=8)
    parser.add_argument("--base-return-duration-sec", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--allow-self-target", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--speed-eps", type=float, default=1e-9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    source_root = args.source_root if args.source_root.is_absolute() else cwd / args.source_root
    output_root = args.output_root
    if output_root is None:
        output_root = source_root.parent / f"{source_root.name}_return8"
    output_root = output_root if output_root.is_absolute() else cwd / output_root

    dataset_roots = sorted(path.parent.parent for path in source_root.glob("*/meta/info.json"))
    if not dataset_roots:
        raise FileNotFoundError(f"No child datasets found under {source_root}")
    summaries: list[dict[str, Any]] = []
    for dataset_root in dataset_roots:
        child_output = output_root / f"{dataset_root.name}{args.child_suffix}"
        summaries.append(
            export_dataset(
                source_root=dataset_root,
                output_root=child_output,
                side=str(args.side),
                append_count=int(args.append_count),
                base_duration_sec=float(args.base_return_duration_sec),
                seed=int(args.seed) + len(summaries) * 1009,
                allow_self_target=bool(args.allow_self_target),
                overwrite=bool(args.overwrite),
                limit_episodes=int(args.limit_episodes),
                speed_eps=float(args.speed_eps),
            )
        )
    print("Exported return8 datasets:")
    for item in summaries:
        print(
            f"  {item['output_root']}: episodes={item['episode_count']} "
            f"frames={item['frame_count']} added_frames={item['added_frame_count']}"
        )


if __name__ == "__main__":
    main()
