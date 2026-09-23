#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

# Keep the documented direct-script invocation usable outside an installed package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.portable_vis.shared_episode_pose import (  # noqa: I001
    _downsample_indices,
    _episode_id_from_index,
    _episode_index_from_path,
    _episode_matches,
    _float_or_none,
    _int_or_none,
    _quat_xyzw_to_matrix,
    _read_jsonl,
    _rotation_matrix_to_quat_xyzw,
    _vector_or_none,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAMERA_REGISTRY = (
    REPO_ROOT
    / "robot_motion"
    / "configs"
    / "rm65_camera_registry.json"
)
DEFAULT_CAMERA_KEY = "front"
DEFAULT_LAG = 8
SIDES = ("left", "right")
POSE_DIM = 7
ACTION_DIM = 26
STATE_DIM = 38
ACTION_ARM_POSE_SLICES = {
    "left": slice(0, 7),
    "right": slice(7, 14),
}
STATE_ARM_POSE_SLICES = {
    "left": slice(6, 13),
    "right": slice(25, 32),
}


@dataclass(frozen=True)
class EpisodeSource:
    dataset_root: Path
    parquet_path: Path
    episode_index: int | None
    episode_id: str
    record: Mapping[str, Any]


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    coeffs: tuple[float, ...]


@dataclass(frozen=True)
class ProjectionRig:
    camera_key: str
    intrinsics: CameraIntrinsics
    parent_frame: str
    T_camera_to_parent: np.ndarray
    T_parent_to_camera: np.ndarray
    T_D_B: Mapping[str, np.ndarray]
    T_desk_to_parent_base: np.ndarray


@dataclass(frozen=True)
class PoseValue:
    xyz: tuple[float, float, float]
    rotation: tuple[float, ...]

    def matrix(self) -> np.ndarray:
        out = np.eye(4, dtype=float)
        out[:3, :3] = np.asarray(self.rotation, dtype=float).reshape(3, 3)
        out[:3, 3] = np.asarray(self.xyz, dtype=float)
        return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project DexTeleop state/action TCP trajectories into recorded RealSense RGB frames. "
            "The default view uses the static front camera."
        )
    )
    parser.add_argument(
        "dataset_or_parquet",
        type=str,
        help="Dataset root, data directory, or one episode_*.parquet file.",
    )
    parser.add_argument(
        "-e",
        "--episode",
        type=str,
        default="",
        help="Episode index or id, e.g. 10 or episode_000010. Defaults to the first episode.",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default=DEFAULT_CAMERA_KEY,
        help="Recorded RealSense RGB camera key. Defaults to front.",
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=DEFAULT_LAG,
        help="Compare action_t against state_{t+lag}. Defaults to 8 from the lag scan.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Output HTML path. Defaults to <dataset>/visualizations/real_projection_<camera>_<episode>_lagN.html.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Uniformly downsample displayed rows to at most this many. 0 keeps all rows.",
    )
    parser.add_argument(
        "--trail",
        type=int,
        default=24,
        help="Number of displayed history samples to draw as a trajectory trail.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.04,
        help="Current-pose orientation triad length in meters. 0 disables pose axes.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=82,
        help="Embedded JPEG quality for RGB frames, 1-100.",
    )
    parser.add_argument(
        "--camera-registry",
        type=str,
        default=str(DEFAULT_CAMERA_REGISTRY),
        help="RealSense camera registry with static extrinsics.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional HTML title.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically open the generated HTML in a browser.",
    )
    return parser.parse_args()


def _require_pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading episode parquet requires pyarrow. Activate the DexTeleop "
            "environment with `conda activate dexteleop310`, then run "
            "`python utils/portable_vis/real_projection_view/project_real_trajectory.py ...`."
        ) from exc
    return pq


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_dataset_root(path: Path) -> Path:
    current = path.parent if path.is_file() else path
    for candidate in (current, *current.parents):
        if (candidate / "meta" / "episodes.jsonl").exists() or (candidate / "data").exists():
            return candidate
    return current


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True

def _parquet_candidates(root: Path) -> list[tuple[dict[str, Any], Path]]:
    dataset_root = _find_dataset_root(root)
    records = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    candidates: list[tuple[dict[str, Any], Path]] = []
    for record in records:
        rel = record.get("parquet_path")
        if not isinstance(rel, str) or not rel:
            continue
        path = dataset_root / rel
        if path.exists():
            candidates.append((record, path))
    if candidates:
        return candidates

    search_root = root if root.is_dir() else root.parent
    for path in sorted(search_root.rglob("episode_*.parquet")):
        record: dict[str, Any] = {
            "episode_index": _episode_index_from_path(path),
            "episode_id": path.stem,
            "parquet_path": str(path.relative_to(dataset_root)) if _path_is_relative_to(path, dataset_root) else str(path),
        }
        candidates.append((record, path))
    return candidates


def _resolve_episode_source(path_text: str, episode_token: str) -> EpisodeSource:
    input_path = Path(path_text).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    dataset_root = _find_dataset_root(input_path).resolve()

    candidates = _parquet_candidates(input_path)
    if input_path.is_file() and input_path.suffix.lower() == ".parquet":
        matches = [(record, path) for record, path in candidates if path.resolve() == input_path]
        record = matches[0][0] if matches else {"episode_index": _episode_index_from_path(input_path), "episode_id": input_path.stem}
        episode_index = record.get("episode_index")
        try:
            episode_index = int(episode_index) if episode_index is not None else _episode_index_from_path(input_path)
        except (TypeError, ValueError):
            episode_index = _episode_index_from_path(input_path)
        return EpisodeSource(
            dataset_root=dataset_root,
            parquet_path=input_path,
            episode_index=episode_index,
            episode_id=str(record.get("episode_id") or _episode_id_from_index(episode_index, input_path.stem)),
            record=record,
        )

    if not candidates:
        raise FileNotFoundError(f"No episode parquet files found under {input_path}")
    token = str(episode_token).strip()
    if token:
        matches = [(record, path) for record, path in candidates if _episode_matches(record, token, path)]
        if not matches:
            available = ", ".join(str(record.get("episode_index", path.stem)) for record, path in candidates[:20])
            raise FileNotFoundError(f"Episode {token!r} not found. First available episodes: {available}")
        record, parquet_path = matches[0]
    else:
        record, parquet_path = sorted(
            candidates,
            key=lambda item: (
                10**12 if item[0].get("episode_index") is None else int(item[0].get("episode_index")),
                str(item[1]),
            ),
        )[0]

    episode_index = record.get("episode_index")
    try:
        episode_index = int(episode_index) if episode_index is not None else _episode_index_from_path(parquet_path)
    except (TypeError, ValueError):
        episode_index = _episode_index_from_path(parquet_path)
    return EpisodeSource(
        dataset_root=dataset_root,
        parquet_path=parquet_path.resolve(),
        episode_index=episode_index,
        episode_id=str(record.get("episode_id") or _episode_id_from_index(episode_index, parquet_path.stem)),
        record=record,
    )


def _selected_columns(names: Iterable[str], camera_key: str) -> list[str]:
    available = set(names)
    requested = {
        "index",
        "episode_index",
        "frame_index",
        "timestamp",
        "action",
        "observation.state",
        "action.left_arm.pose",
        "action.right_arm.pose",
        "observation.left_arm.pose",
        "observation.right_arm.pose",
        f"observation.images.{camera_key}.source_timestamp",
        f"observation.images.{camera_key}.receive_timestamp",
        f"observation.images.{camera_key}.age_sec",
        f"observation.images.{camera_key}.source_frame_index",
        f"observation.images.{camera_key}.video_frame_index",
        f"observation.images.{camera_key}.path",
    }
    selected = sorted(requested & available)
    missing = {"action", "observation.state"} - set(selected)
    if missing:
        raise ValueError(f"Episode parquet is missing required column(s): {sorted(missing)}")
    return selected


def _read_episode_rows(parquet_path: Path, camera_key: str) -> list[dict[str, Any]]:
    pq = _require_pyarrow_parquet()
    parquet = pq.ParquetFile(parquet_path)
    rows = pq.read_table(parquet_path, columns=_selected_columns(parquet.schema_arrow.names, camera_key)).to_pylist()
    rows.sort(key=lambda row: int(row.get("frame_index", row.get("index", 0)) or 0))
    return rows

def _pose_value(vector: np.ndarray | None) -> PoseValue | None:
    if vector is None:
        return None
    pose = np.asarray(vector, dtype=float).reshape(-1)
    if pose.shape != (POSE_DIM,) or not np.all(np.isfinite(pose)):
        return None
    rotation = _quat_xyzw_to_matrix(pose[3:7])
    if rotation is None:
        return None
    return PoseValue(
        xyz=tuple(float(value) for value in pose[:3].tolist()),
        rotation=tuple(float(value) for value in rotation.reshape(-1).tolist()),
    )


def _action_pose(row: Mapping[str, Any], side: str) -> PoseValue | None:
    split_key = f"action.{side}_arm.pose"
    pose = _vector_or_none(row.get(split_key), length=POSE_DIM)
    if pose is None:
        action = _vector_or_none(row.get("action"), length=ACTION_DIM)
        if action is not None:
            pose = action[ACTION_ARM_POSE_SLICES[side]]
    return _pose_value(pose)


def _state_pose(row: Mapping[str, Any] | None, side: str) -> PoseValue | None:
    if row is None:
        return None
    split_key = f"observation.{side}_arm.pose"
    pose = _vector_or_none(row.get(split_key), length=POSE_DIM)
    if pose is None:
        state = _vector_or_none(row.get("observation.state"), length=STATE_DIM)
        if state is not None:
            pose = state[STATE_ARM_POSE_SLICES[side]]
    return _pose_value(pose)

def _pose_error(a: PoseValue | None, b: PoseValue | None) -> dict[str, float] | None:
    if a is None or b is None:
        return None
    av = np.asarray(a.xyz, dtype=float)
    bv = np.asarray(b.xyz, dtype=float)
    translation_m = float(np.linalg.norm(av - bv))
    qa = _rotation_matrix_to_quat_xyzw(np.asarray(a.rotation, dtype=float).reshape(3, 3))
    qb = _rotation_matrix_to_quat_xyzw(np.asarray(b.rotation, dtype=float).reshape(3, 3))
    rotation_deg = None
    if qa is not None and qb is not None:
        dot = float(np.dot(qa, qb))
        dot = min(1.0, max(-1.0, abs(dot)))
        rotation_deg = float(math.degrees(2.0 * math.acos(dot)))
    return {
        "translation_m": translation_m,
        "translation_mm": translation_m * 1000.0,
        "rotation_deg": rotation_deg,
    }


def _matrix_from_payload(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    return matrix


def _load_intrinsics(dataset_root: Path, camera_key: str, registry: Mapping[str, Any]) -> CameraIntrinsics:
    devices_path = dataset_root / "meta" / "devices" / "realsense.json"
    payload: Mapping[str, Any] = {}
    if devices_path.exists():
        payload = _read_json(devices_path)
    camera_payload = payload.get(camera_key) if isinstance(payload, Mapping) else None
    if not isinstance(camera_payload, Mapping):
        camera_payload = registry.get("cameras", {}).get(camera_key, {})
    intr = camera_payload.get("intrinsics") if isinstance(camera_payload, Mapping) else None
    if not isinstance(intr, Mapping):
        profile = camera_payload.get("profile") if isinstance(camera_payload, Mapping) else {}
        raise ValueError(
            f"Missing intrinsics for camera {camera_key!r}. "
            f"Checked {devices_path}; registry profile is {profile!r}."
        )
    width = int(intr.get("width") or camera_payload.get("width") or camera_payload.get("profile", {}).get("width") or 0)
    height = int(intr.get("height") or camera_payload.get("height") or camera_payload.get("profile", {}).get("height") or 0)
    coeffs = tuple(float(v) for v in intr.get("coeffs", []) or [])
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=float(intr["fx"]),
        fy=float(intr["fy"]),
        ppx=float(intr["ppx"]),
        ppy=float(intr["ppy"]),
        coeffs=coeffs,
    )


def _load_camera_extrinsic(registry_path: Path, camera_key: str) -> tuple[dict[str, Any], str, np.ndarray]:
    registry = _read_json(registry_path)
    camera = registry.get("cameras", {}).get(camera_key)
    if not isinstance(camera, Mapping):
        raise KeyError(f"Camera {camera_key!r} was not found in {registry_path}")
    frames = camera.get("extrinsics", {}).get("frames", {})
    if not isinstance(frames, Mapping) or not frames:
        raise ValueError(f"Camera {camera_key!r} has no static extrinsic frames in {registry_path}")

    chosen_name = None
    chosen_frame = None
    for frame_name, frame in frames.items():
        if isinstance(frame, Mapping) and frame.get("T_camera_to_parent") is not None:
            chosen_name = str(frame_name)
            chosen_frame = frame
            break
    if not isinstance(chosen_frame, Mapping):
        raise ValueError(
            f"Camera {camera_key!r} has no T_camera_to_parent in {registry_path}. "
            "The current projection viewer supports static cameras such as front."
        )
    parent_frame = str(chosen_frame.get("parent_frame") or "")
    if parent_frame not in {"right_arm_base", "right_arm_base_link"}:
        raise ValueError(
            f"Camera {camera_key!r} extrinsic frame {chosen_name!r} uses parent {parent_frame!r}. "
            "This viewer currently expects the static front camera parent to be right_arm_base_link."
        )
    return registry, parent_frame, _matrix_from_payload(chosen_frame["T_camera_to_parent"], name="T_camera_to_parent")


def _calibration_snapshot(source: EpisodeSource) -> Mapping[str, Any] | None:
    metadata = source.record.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    snapshot = metadata.get("calibration_snapshot")
    return snapshot if isinstance(snapshot, Mapping) else None


def _profile_path_candidates(source: EpisodeSource) -> list[Path]:
    metadata = source.record.get("metadata")
    if not isinstance(metadata, Mapping):
        return []
    load_profile = metadata.get("load_profile")
    if not isinstance(load_profile, str) or not load_profile:
        return []
    raw = Path(load_profile).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(REPO_ROOT / raw)
        candidates.append(source.dataset_root / raw)
        candidates.append(source.dataset_root.parent / raw)
    return [path for path in candidates if path.exists()]


def _load_T_D_B(source: EpisodeSource) -> dict[str, np.ndarray]:
    snapshot = _calibration_snapshot(source)
    if isinstance(snapshot, Mapping):
        arms = snapshot.get("arms")
        if isinstance(arms, Mapping):
            out: dict[str, np.ndarray] = {}
            for side in SIDES:
                arm = arms.get(f"{side}_arm") or arms.get(side)
                if isinstance(arm, Mapping) and arm.get("T_D_B") is not None:
                    out[side] = _matrix_from_payload(arm["T_D_B"], name=f"{side}.T_D_B")
            if set(out) == set(SIDES):
                return out

    for path in _profile_path_candidates(source):
        profile = _read_json(path)
        arms = profile.get("arms") if isinstance(profile, Mapping) else None
        if not isinstance(arms, Mapping):
            continue
        out = {}
        for side in SIDES:
            arm = arms.get(f"{side}_arm") or arms.get(side)
            if isinstance(arm, Mapping) and arm.get("T_D_B") is not None:
                out[side] = _matrix_from_payload(arm["T_D_B"], name=f"{side}.T_D_B")
        if set(out) == set(SIDES):
            return out

    raise ValueError(
        "Could not load left/right T_D_B. Expected episode metadata.calibration_snapshot "
        "or a readable metadata.load_profile calibration profile."
    )


def _build_rig(source: EpisodeSource, camera_key: str, registry_path: Path) -> ProjectionRig:
    registry, parent_frame, T_camera_to_parent = _load_camera_extrinsic(registry_path, camera_key)
    intrinsics = _load_intrinsics(source.dataset_root, camera_key, registry)
    T_D_B = _load_T_D_B(source)
    return ProjectionRig(
        camera_key=camera_key,
        intrinsics=intrinsics,
        parent_frame=parent_frame,
        T_camera_to_parent=T_camera_to_parent,
        T_parent_to_camera=np.linalg.inv(T_camera_to_parent),
        T_D_B=T_D_B,
        T_desk_to_parent_base=np.linalg.inv(T_D_B["right"]),
    )


def _pose_to_camera_matrix(rig: ProjectionRig, side: str, pose: PoseValue) -> np.ndarray:
    T_side_pose = pose.matrix()
    T_parent_side = rig.T_desk_to_parent_base @ rig.T_D_B[side]
    return rig.T_parent_to_camera @ T_parent_side @ T_side_pose


def _project_point(rig: ProjectionRig, point_camera: Sequence[float]) -> dict[str, Any]:
    point = np.asarray(point_camera, dtype=float).reshape(3)
    x, y, z = (float(v) for v in point.tolist())
    if not np.all(np.isfinite(point)) or z <= 1e-9:
        return {
            "u": None,
            "v": None,
            "z": z if math.isfinite(z) else None,
            "in_front": False,
            "in_image": False,
        }
    xn = x / z
    yn = y / z
    coeffs = list(rig.intrinsics.coeffs)
    if len(coeffs) >= 4 and any(abs(v) > 1e-12 for v in coeffs):
        k1 = coeffs[0]
        k2 = coeffs[1]
        p1 = coeffs[2]
        p2 = coeffs[3]
        k3 = coeffs[4] if len(coeffs) >= 5 else 0.0
        r2 = xn * xn + yn * yn
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x_distorted = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        y_distorted = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        xn, yn = x_distorted, y_distorted
    u = rig.intrinsics.fx * xn + rig.intrinsics.ppx
    v = rig.intrinsics.fy * yn + rig.intrinsics.ppy
    return {
        "u": float(u),
        "v": float(v),
        "z": z,
        "in_front": True,
        "in_image": 0.0 <= u < rig.intrinsics.width and 0.0 <= v < rig.intrinsics.height,
    }


def _project_pose(rig: ProjectionRig, side: str, pose: PoseValue | None, axis_length: float) -> dict[str, Any] | None:
    if pose is None:
        return None
    T_camera_pose = _pose_to_camera_matrix(rig, side, pose)
    origin = T_camera_pose[:3, 3]
    payload: dict[str, Any] = {
        "xyz": [float(v) for v in pose.xyz],
        "camera_xyz": [float(v) for v in origin.tolist()],
        "origin": _project_point(rig, origin),
        "axes": {},
    }
    if axis_length > 0.0:
        for axis_name, column in (("x", 0), ("y", 1), ("z", 2)):
            endpoint = origin + T_camera_pose[:3, column] * float(axis_length)
            payload["axes"][axis_name] = _project_point(rig, endpoint)
    return payload


def _pixel_distance(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> float | None:
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        return None
    ao = a.get("origin")
    bo = b.get("origin")
    if not isinstance(ao, Mapping) or not isinstance(bo, Mapping):
        return None
    au = _float_or_none(ao.get("u"))
    av = _float_or_none(ao.get("v"))
    bu = _float_or_none(bo.get("u"))
    bv = _float_or_none(bo.get("v"))
    if au is None or av is None or bu is None or bv is None:
        return None
    return float(math.hypot(au - bu, av - bv))


def _resolve_video_path(source: EpisodeSource, camera_key: str, rows: Sequence[Mapping[str, Any]]) -> Path:
    video_paths = source.record.get("video_paths")
    if isinstance(video_paths, Mapping):
        rel = video_paths.get(camera_key)
        if isinstance(rel, str) and rel:
            path = source.dataset_root / rel
            if path.exists():
                return path

    column = f"observation.images.{camera_key}.path"
    for row in rows:
        rel = row.get(column)
        if isinstance(rel, str) and rel:
            path = source.dataset_root / rel
            if path.exists():
                return path
            raw = Path(rel).expanduser()
            if raw.exists():
                return raw
    raise FileNotFoundError(f"Could not find recorded video for camera {camera_key!r} in {source.dataset_root}")


def _video_frame_index(row: Mapping[str, Any], camera_key: str, fallback: int) -> int:
    key = f"observation.images.{camera_key}.video_frame_index"
    value = _int_or_none(row.get(key))
    return fallback if value is None else value


def _encode_video_frames(
    video_path: Path,
    frame_indices: Sequence[int],
    *,
    jpeg_quality: int,
) -> tuple[dict[int, str], tuple[int, int], int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    unique = sorted(set(int(i) for i in frame_indices if int(i) >= 0))
    quality = max(1, min(100, int(jpeg_quality)))
    encoded: dict[int, str] = {}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    for index in unique:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if width <= 0 or height <= 0:
            height, width = frame.shape[:2]
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            continue
        encoded[int(index)] = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")
    cap.release()
    return encoded, (width, height), total


def _safe_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def _frame_projection_score(frame: Mapping[str, Any]) -> float:
    score = 0.0
    sides = frame.get("sides")
    if not isinstance(sides, Mapping):
        return score
    for side_payload in sides.values():
        if not isinstance(side_payload, Mapping):
            continue
        for trace_name in ("state", "action", "lag_state"):
            trace = side_payload.get(trace_name)
            if not isinstance(trace, Mapping):
                continue
            origin = trace.get("origin")
            if not isinstance(origin, Mapping):
                continue
            if origin.get("in_image"):
                score += 2.0
            elif origin.get("in_front"):
                score += 0.25
    return score


def _build_payload(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: EpisodeSource,
    rig: ProjectionRig,
    video_path: Path,
    max_frames: int,
    lag: int,
    trail: int,
    axis_length: float,
    jpeg_quality: int,
    title: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows:
        raise ValueError(f"Episode parquet contains no rows: {source.parquet_path}")
    if lag < 0:
        raise ValueError("--lag must be >= 0")

    indices = _downsample_indices(len(rows), max_frames)
    video_indices = [_video_frame_index(rows[i], rig.camera_key, i) for i in indices]
    images, video_size, video_frame_count = _encode_video_frames(video_path, video_indices, jpeg_quality=jpeg_quality)
    missing_images = sorted(set(video_indices) - set(images))
    if not images:
        raise RuntimeError(f"No requested frames could be decoded from {video_path}")

    frames: list[dict[str, Any]] = []
    projection_counts: dict[str, dict[str, dict[str, int]]] = {
        side: {trace: {"projected": 0, "in_image": 0} for trace in ("state", "action", "lag_state")}
        for side in SIDES
    }
    residuals_mm: dict[str, dict[str, list[float]]] = {
        side: {"state": [], "lag_state": []}
        for side in SIDES
    }
    residuals_px: dict[str, dict[str, list[float]]] = {
        side: {"state": [], "lag_state": []}
        for side in SIDES
    }

    for display_index, row_index in enumerate(indices):
        row = rows[row_index]
        lag_row = rows[row_index + lag] if row_index + lag < len(rows) else None
        video_index = _video_frame_index(row, rig.camera_key, row_index)
        image = images.get(video_index)
        if image is None:
            continue
        frame_payload: dict[str, Any] = {
            "display_index": display_index,
            "row_index": int(row_index),
            "frame_index": _int_or_none(row.get("frame_index", row.get("index"))),
            "timestamp": _float_or_none(row.get("timestamp")),
            "camera_source_timestamp": _float_or_none(row.get(f"observation.images.{rig.camera_key}.source_timestamp")),
            "camera_receive_timestamp": _float_or_none(row.get(f"observation.images.{rig.camera_key}.receive_timestamp")),
            "camera_age_sec": _float_or_none(row.get(f"observation.images.{rig.camera_key}.age_sec")),
            "camera_source_frame_index": _int_or_none(row.get(f"observation.images.{rig.camera_key}.source_frame_index")),
            "video_frame_index": int(video_index),
            "image": image,
            "sides": {},
        }
        for side in SIDES:
            action_pose = _action_pose(row, side)
            state_pose = _state_pose(row, side)
            lag_pose = _state_pose(lag_row, side)
            action_projected = _project_pose(rig, side, action_pose, axis_length)
            state_projected = _project_pose(rig, side, state_pose, axis_length)
            lag_projected = _project_pose(rig, side, lag_pose, axis_length)
            traces = {
                "state": state_projected,
                "action": action_projected,
                "lag_state": lag_projected,
            }
            for trace_name, trace_payload in traces.items():
                if trace_payload is None:
                    continue
                origin = trace_payload.get("origin")
                if isinstance(origin, Mapping) and origin.get("in_front"):
                    projection_counts[side][trace_name]["projected"] += 1
                if isinstance(origin, Mapping) and origin.get("in_image"):
                    projection_counts[side][trace_name]["in_image"] += 1

            same_error = _pose_error(action_pose, state_pose)
            lag_error = _pose_error(action_pose, lag_pose)
            same_px = _pixel_distance(action_projected, state_projected)
            lag_px = _pixel_distance(action_projected, lag_projected)
            if same_error is not None:
                residuals_mm[side]["state"].append(float(same_error["translation_mm"]))
            if lag_error is not None:
                residuals_mm[side]["lag_state"].append(float(lag_error["translation_mm"]))
            if same_px is not None:
                residuals_px[side]["state"].append(same_px)
            if lag_px is not None:
                residuals_px[side]["lag_state"].append(lag_px)
            frame_payload["sides"][side] = {
                "state": state_projected,
                "action": action_projected,
                "lag_state": lag_projected,
                "errors": {
                    "state": same_error,
                    "lag_state": lag_error,
                    "state_px": same_px,
                    "lag_state_px": lag_px,
                },
            }
        frames.append(frame_payload)

    summary = {
        "displayed_frames": len(frames),
        "source_rows": len(rows),
        "missing_video_frames": len(missing_images),
        "projection_counts": projection_counts,
        "residuals_mm": {
            side: {name: _safe_summary(values) for name, values in side_values.items()}
            for side, side_values in residuals_mm.items()
        },
        "residuals_px": {
            side: {name: _safe_summary(values) for name, values in side_values.items()}
            for side, side_values in residuals_px.items()
        },
    }
    initial_index = 0
    if frames:
        initial_index = max(range(len(frames)), key=lambda index: (_frame_projection_score(frames[index]), -index))
    payload_title = title or f"{source.episode_id} {rig.camera_key} RGB projection lag={lag}"
    payload = {
        "schema": "dexteleop_real_projection_view_v1",
        "title": payload_title,
        "dataset_root": str(source.dataset_root),
        "episode_id": source.episode_id,
        "episode_index": source.episode_index,
        "parquet_path": str(source.parquet_path),
        "video_path": str(video_path),
        "camera_key": rig.camera_key,
        "lag": int(lag),
        "trail": int(max(0, trail)),
        "axis_length": float(axis_length),
        "camera": {
            "parent_frame": rig.parent_frame,
            "intrinsics": {
                "width": rig.intrinsics.width,
                "height": rig.intrinsics.height,
                "fx": rig.intrinsics.fx,
                "fy": rig.intrinsics.fy,
                "ppx": rig.intrinsics.ppx,
                "ppy": rig.intrinsics.ppy,
                "coeffs": list(rig.intrinsics.coeffs),
            },
            "T_camera_to_parent": rig.T_camera_to_parent.tolist(),
        },
        "video": {
            "width": video_size[0] or rig.intrinsics.width,
            "height": video_size[1] or rig.intrinsics.height,
            "frame_count": video_frame_count,
        },
        "styles": {
            "left": {"state": "#2563eb", "action": "#0f766e", "lag_state": "#7c3aed"},
            "right": {"state": "#dc2626", "action": "#ea580c", "lag_state": "#0891b2"},
            "axes": {"x": "#ef4444", "y": "#22c55e", "z": "#3b82f6"},
        },
        "summary": summary,
        "initial_index": int(initial_index),
        "frames": frames,
    }
    return payload, summary


def _default_output(source: EpisodeSource, camera_key: str, lag: int) -> Path:
    return source.dataset_root / "visualizations" / f"real_projection_{camera_key}_{source.episode_id}_lag{int(lag)}.html"


def _render_html(payload: Mapping[str, Any]) -> str:
    title = html.escape(str(payload.get("title") or "DexTeleop Real Projection"))
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__TITLE__", title).replace("__PAYLOAD__", payload_json)


def main() -> None:
    args = _parse_args()
    source = _resolve_episode_source(args.dataset_or_parquet, args.episode)
    rig = _build_rig(source, args.camera_key, Path(args.camera_registry).expanduser().resolve())
    rows = _read_episode_rows(source.parquet_path, args.camera_key)
    video_path = _resolve_video_path(source, args.camera_key, rows)
    payload, summary = _build_payload(
        rows,
        source=source,
        rig=rig,
        video_path=video_path,
        max_frames=int(args.max_frames),
        lag=int(args.lag),
        trail=int(args.trail),
        axis_length=float(args.axis_length),
        jpeg_quality=int(args.jpeg_quality),
        title=str(args.title),
    )
    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output(source, args.camera_key, args.lag)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_html(payload), encoding="utf-8")

    print(f"Wrote {output_path}")
    print(f"Episode: {source.episode_id} rows={len(rows)} displayed={summary['displayed_frames']} lag={args.lag}")
    print(f"Video: {video_path}")
    for side in SIDES:
        counts = summary["projection_counts"][side]
        action_count = counts["action"]["in_image"]
        state_count = counts["state"]["in_image"]
        lag_count = counts["lag_state"]["in_image"]
        same_mm = summary["residuals_mm"][side]["state"]["p50"]
        lag_mm = summary["residuals_mm"][side]["lag_state"]["p50"]
        print(
            f"{side}: in-image state/action/lag={state_count}/{action_count}/{lag_count}; "
            f"p50 action-state={same_mm:.1f}mm, action-state+lag={lag_mm:.1f}mm"
            if same_mm is not None and lag_mm is not None
            else f"{side}: in-image state/action/lag={state_count}/{action_count}/{lag_count}"
        )
    if not args.no_open:
        webbrowser.open(output_path.as_uri())


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #171717;
  --muted: #676f7d;
  --line: #d7dce3;
  --soft: #e9edf3;
  --shadow: 0 18px 48px rgba(22, 28, 39, 0.14);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  min-height: 100vh;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
button, input { font: inherit; }
.app {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 340px;
  gap: 14px;
  min-height: 100vh;
  padding: 14px;
}
.stage {
  min-width: 0;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  gap: 10px;
}
.topbar, .controls, .sidepanel {
  background: var(--panel);
  border: 1px solid var(--line);
  box-shadow: var(--shadow);
}
.topbar {
  min-height: 58px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 10px 12px;
}
.title {
  min-width: 0;
}
.title h1 {
  margin: 0;
  font-size: 18px;
  line-height: 1.2;
  font-weight: 720;
}
.title .sub {
  margin-top: 3px;
  color: var(--muted);
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.badges {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 6px;
}
.badge {
  border: 1px solid var(--line);
  background: #f8fafc;
  padding: 4px 7px;
  font-size: 12px;
  color: #293241;
}
.viewer {
  min-width: 0;
  min-height: 0;
  background: #111827;
  border: 1px solid #111827;
  display: grid;
  place-items: center;
  overflow: hidden;
  position: relative;
}
.canvasWrap {
  position: relative;
  width: min(100%, calc((100vh - 150px) * 1.333333));
  aspect-ratio: 4 / 3;
  background: #0b1020;
}
canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  display: block;
}
.controls {
  padding: 10px 12px;
  display: grid;
  grid-template-columns: auto minmax(160px, 1fr) auto auto;
  gap: 10px;
  align-items: center;
}
.iconBtn {
  width: 38px;
  height: 34px;
  border: 1px solid #b8c0cc;
  background: #ffffff;
  color: #111827;
  display: inline-grid;
  place-items: center;
  cursor: pointer;
}
.iconBtn:hover { background: #f1f5f9; }
.slider {
  width: 100%;
  accent-color: #111827;
}
.readout {
  font-variant-numeric: tabular-nums;
  color: #1f2937;
  font-size: 12px;
  white-space: nowrap;
}
.sidepanel {
  min-width: 0;
  padding: 12px;
  overflow: auto;
}
.group {
  border-bottom: 1px solid var(--soft);
  padding-bottom: 12px;
  margin-bottom: 12px;
}
.group:last-child {
  border-bottom: 0;
  margin-bottom: 0;
  padding-bottom: 0;
}
.group h2 {
  margin: 0 0 9px;
  font-size: 13px;
  line-height: 1.2;
  text-transform: uppercase;
  letter-spacing: 0;
  color: #374151;
}
.toggles {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
label.check {
  display: flex;
  align-items: center;
  gap: 7px;
  min-height: 28px;
  color: #263244;
  font-size: 13px;
}
label.check input { width: 16px; height: 16px; }
.legend {
  display: grid;
  gap: 8px;
}
.legendRow {
  display: grid;
  grid-template-columns: 18px minmax(0, 1fr);
  gap: 8px;
  align-items: center;
  color: #263244;
  font-size: 13px;
}
.swatch {
  width: 18px;
  height: 4px;
  border-radius: 99px;
}
.kv {
  display: grid;
  grid-template-columns: minmax(86px, auto) minmax(0, 1fr);
  gap: 5px 10px;
  font-size: 12px;
  color: #263244;
}
.kv .key { color: var(--muted); }
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  overflow-wrap: anywhere;
}
.sideBlock {
  border: 1px solid var(--line);
  padding: 9px;
  margin-top: 8px;
}
.sideHead {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 8px;
  font-size: 13px;
  font-weight: 700;
}
.pill {
  border: 1px solid var(--line);
  padding: 2px 6px;
  color: #334155;
  background: #f8fafc;
  font-size: 11px;
  font-weight: 500;
}
.hint {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}
@media (max-width: 980px) {
  .app { grid-template-columns: 1fr; }
  .sidepanel { max-height: none; }
  .controls { grid-template-columns: auto minmax(120px, 1fr) auto; }
  .controls .readout.extra { display: none; }
}
</style>
</head>
<body>
<main class="app">
  <section class="stage">
    <div class="topbar">
      <div class="title">
        <h1 id="title"></h1>
        <div class="sub mono" id="pathLine"></div>
      </div>
      <div class="badges">
        <span class="badge" id="cameraBadge"></span>
        <span class="badge" id="lagBadge"></span>
        <span class="badge" id="frameBadge"></span>
      </div>
    </div>
    <div class="viewer">
      <div class="canvasWrap" id="canvasWrap">
        <canvas id="imageCanvas"></canvas>
        <canvas id="overlayCanvas"></canvas>
      </div>
    </div>
    <div class="controls">
      <button class="iconBtn" id="playBtn" title="Play or pause" aria-label="Play or pause">Play</button>
      <input class="slider" id="frameSlider" type="range" min="0" max="0" value="0">
      <div class="readout" id="frameReadout"></div>
      <div class="readout extra" id="timeReadout"></div>
    </div>
  </section>
  <aside class="sidepanel">
    <div class="group">
      <h2>Layers</h2>
      <div class="toggles">
        <label class="check"><input type="checkbox" id="showLeft" checked>Left</label>
        <label class="check"><input type="checkbox" id="showRight" checked>Right</label>
        <label class="check"><input type="checkbox" id="showState" checked>State</label>
        <label class="check"><input type="checkbox" id="showAction" checked>Action</label>
        <label class="check"><input type="checkbox" id="showLag" checked>Lag state</label>
        <label class="check"><input type="checkbox" id="showAxes" checked>Axes</label>
        <label class="check"><input type="checkbox" id="showTrail" checked>Trail</label>
        <label class="check"><input type="checkbox" id="showResiduals" checked>Residuals</label>
      </div>
    </div>
    <div class="group">
      <h2>Legend</h2>
      <div class="legend" id="legend"></div>
    </div>
    <div class="group">
      <h2>Current</h2>
      <div id="currentStats"></div>
    </div>
    <div class="group">
      <h2>Summary</h2>
      <div class="kv" id="summaryStats"></div>
    </div>
    <div class="group">
      <h2>Frame Chain</h2>
      <div class="hint">
        side base pose -> desk via T_D_B(side) -> right base via inv(T_D_B(right)) -> front camera via inv(T_camera_to_parent) -> RGB pixel.
      </div>
    </div>
  </aside>
</main>
<script>
const payload = __PAYLOAD__;
const frames = payload.frames || [];
const state = {
  index: 0,
  playing: false,
  image: new Image(),
  pendingSrc: "",
  timer: null
};
const els = {
  title: document.getElementById("title"),
  pathLine: document.getElementById("pathLine"),
  cameraBadge: document.getElementById("cameraBadge"),
  lagBadge: document.getElementById("lagBadge"),
  frameBadge: document.getElementById("frameBadge"),
  imageCanvas: document.getElementById("imageCanvas"),
  overlayCanvas: document.getElementById("overlayCanvas"),
  slider: document.getElementById("frameSlider"),
  playBtn: document.getElementById("playBtn"),
  frameReadout: document.getElementById("frameReadout"),
  timeReadout: document.getElementById("timeReadout"),
  legend: document.getElementById("legend"),
  currentStats: document.getElementById("currentStats"),
  summaryStats: document.getElementById("summaryStats"),
  toggles: {
    left: document.getElementById("showLeft"),
    right: document.getElementById("showRight"),
    state: document.getElementById("showState"),
    action: document.getElementById("showAction"),
    lag: document.getElementById("showLag"),
    axes: document.getElementById("showAxes"),
    trail: document.getElementById("showTrail"),
    residuals: document.getElementById("showResiduals")
  }
};
const traceLabels = {state: "state_t", action: "action_t", lag_state: `state_t+${payload.lag}`};
const sideLabels = {left: "Left", right: "Right"};

function fmtNumber(value, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}
function fmtMs(value) {
  return Number.isFinite(value) ? `${(value * 1000).toFixed(1)} ms` : "n/a";
}
function fmtTime(value) {
  return Number.isFinite(value) ? value.toFixed(6) : "n/a";
}
function visibleTrace(trace) {
  if (trace === "state") return els.toggles.state.checked;
  if (trace === "action") return els.toggles.action.checked;
  if (trace === "lag_state") return els.toggles.lag.checked;
  return true;
}
function visibleSide(side) {
  return side === "left" ? els.toggles.left.checked : els.toggles.right.checked;
}
function pointOf(item) {
  if (!item || !item.origin) return null;
  const u = item.origin.u;
  const v = item.origin.v;
  if (!Number.isFinite(u) || !Number.isFinite(v) || !item.origin.in_front) return null;
  return {u, v, inImage: !!item.origin.in_image};
}
function canvasScale(canvas) {
  const width = payload.video.width || payload.camera.intrinsics.width || 640;
  const height = payload.video.height || payload.camera.intrinsics.height || 480;
  const rect = canvas.getBoundingClientRect();
  return {sx: rect.width / width, sy: rect.height / height};
}
function cx(point, scale) { return point.u * scale.sx; }
function cy(point, scale) { return point.v * scale.sy; }
function setCanvasSize(canvas) {
  const width = payload.video.width || payload.camera.intrinsics.width || 640;
  const height = payload.video.height || payload.camera.intrinsics.height || 480;
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
function setupStaticText() {
  els.title.textContent = payload.title || "DexTeleop Real Projection";
  els.pathLine.textContent = payload.parquet_path || "";
  els.cameraBadge.textContent = `camera ${payload.camera_key}`;
  els.lagBadge.textContent = `lag ${payload.lag}`;
  els.frameBadge.textContent = `${frames.length} frames`;
  els.slider.max = String(Math.max(0, frames.length - 1));
  const styles = payload.styles || {};
  const legendRows = [];
  for (const side of ["left", "right"]) {
    for (const trace of ["state", "action", "lag_state"]) {
      const color = styles[side]?.[trace] || "#111827";
      legendRows.push(`<div class="legendRow"><span class="swatch" style="background:${color}"></span><span>${sideLabels[side]} ${traceLabels[trace]}</span></div>`);
    }
  }
  els.legend.innerHTML = legendRows.join("");
  drawSummary();
}
function drawSummary() {
  const summary = payload.summary || {};
  const parts = [
    ["source rows", summary.source_rows],
    ["displayed", summary.displayed_frames],
    ["video missing", summary.missing_video_frames],
    ["intrinsics", `${fmtNumber(payload.camera.intrinsics.fx, 1)}, ${fmtNumber(payload.camera.intrinsics.fy, 1)}`],
    ["parent", payload.camera.parent_frame],
    ["video", `${payload.video.width}x${payload.video.height}`]
  ];
  for (const side of ["left", "right"]) {
    const r = summary.residuals_mm?.[side] || {};
    parts.push([`${side} p50 same`, `${fmtNumber(r.state?.p50, 1)} mm`]);
    parts.push([`${side} p50 lag`, `${fmtNumber(r.lag_state?.p50, 1)} mm`]);
  }
  els.summaryStats.innerHTML = parts.map(([k, v]) => `<div class="key">${k}</div><div class="mono">${v ?? "n/a"}</div>`).join("");
}
function drawImage() {
  const frame = frames[state.index];
  const ctx = els.imageCanvas.getContext("2d");
  const width = payload.video.width || payload.camera.intrinsics.width || 640;
  const height = payload.video.height || payload.camera.intrinsics.height || 480;
  ctx.clearRect(0, 0, width, height);
  if (!frame) return;
  if (state.image.src !== frame.image) {
    state.pendingSrc = frame.image;
    state.image.onload = () => {
      if (state.pendingSrc === frame.image) {
        ctx.drawImage(state.image, 0, 0, width, height);
        drawOverlay();
      }
    };
    state.image.src = frame.image;
  } else if (state.image.complete) {
    ctx.drawImage(state.image, 0, 0, width, height);
  }
}
function drawCircle(ctx, p, color, radius, scale, hollow = false) {
  ctx.beginPath();
  ctx.arc(cx(p, scale), cy(p, scale), radius, 0, Math.PI * 2);
  if (hollow) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  } else {
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.92)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}
function drawLine(ctx, a, b, color, scale, alpha = 1, dash = []) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(cx(a, scale), cy(a, scale));
  ctx.lineTo(cx(b, scale), cy(b, scale));
  ctx.stroke();
  ctx.restore();
}
function drawAxes(ctx, item, scale) {
  if (!els.toggles.axes.checked || !item || !item.axes) return;
  const origin = pointOf(item);
  if (!origin) return;
  const axisColors = payload.styles.axes || {x: "#ef4444", y: "#22c55e", z: "#3b82f6"};
  for (const axis of ["x", "y", "z"]) {
    const p = pointOf({origin: item.axes[axis]});
    if (!p) continue;
    drawLine(ctx, origin, p, axisColors[axis], scale, 0.9);
  }
}
function drawTrail(ctx, side, trace, color, scale) {
  if (!els.toggles.trail.checked) return;
  const start = Math.max(0, state.index - (payload.trail || 0));
  let last = null;
  for (let i = start; i <= state.index; i++) {
    const item = frames[i]?.sides?.[side]?.[trace];
    const p = pointOf(item);
    if (!p) {
      last = null;
      continue;
    }
    if (last) {
      const alpha = 0.12 + 0.45 * ((i - start) / Math.max(1, state.index - start + 1));
      drawLine(ctx, last, p, color, scale, alpha);
    }
    last = p;
  }
}
function drawResiduals(ctx, side, sidePayload, scale) {
  if (!els.toggles.residuals.checked) return;
  const action = pointOf(sidePayload.action);
  const statePoint = pointOf(sidePayload.state);
  const lagPoint = pointOf(sidePayload.lag_state);
  if (action && statePoint && els.toggles.state.checked) drawLine(ctx, action, statePoint, "rgba(17,24,39,0.75)", scale, 0.8, [5, 4]);
  if (action && lagPoint && els.toggles.lag.checked) drawLine(ctx, action, lagPoint, "rgba(20,83,45,0.78)", scale, 0.8, [2, 4]);
}
function drawOverlay() {
  const canvas = els.overlayCanvas;
  const ctx = canvas.getContext("2d");
  const width = payload.video.width || payload.camera.intrinsics.width || 640;
  const height = payload.video.height || payload.camera.intrinsics.height || 480;
  ctx.clearRect(0, 0, width, height);
  const scale = {sx: 1, sy: 1};
  const frame = frames[state.index];
  if (!frame) return;
  for (const side of ["left", "right"]) {
    if (!visibleSide(side)) continue;
    const sidePayload = frame.sides?.[side];
    if (!sidePayload) continue;
    drawResiduals(ctx, side, sidePayload, scale);
    for (const trace of ["state", "lag_state", "action"]) {
      if (!visibleTrace(trace)) continue;
      const item = sidePayload[trace];
      const p = pointOf(item);
      const color = payload.styles?.[side]?.[trace] || "#111827";
      drawTrail(ctx, side, trace, color, scale);
      drawAxes(ctx, item, scale);
      if (!p) continue;
      const hollow = trace === "lag_state";
      const radius = trace === "action" ? 6 : 5;
      drawCircle(ctx, p, color, radius, scale, hollow);
    }
  }
}
function currentMetric(side, key, suffix) {
  const frame = frames[state.index];
  const value = frame?.sides?.[side]?.errors?.[key];
  if (!Number.isFinite(value)) return "n/a";
  return `${value.toFixed(1)} ${suffix}`;
}
function drawCurrentStats() {
  const frame = frames[state.index];
  if (!frame) {
    els.currentStats.innerHTML = "";
    return;
  }
  const blocks = [];
  for (const side of ["left", "right"]) {
    const errors = frame.sides?.[side]?.errors || {};
    const same = errors.state || {};
    const lag = errors.lag_state || {};
    blocks.push(`<div class="sideBlock">
      <div class="sideHead"><span>${sideLabels[side]}</span><span class="pill">row ${frame.row_index}</span></div>
      <div class="kv">
        <div class="key">same xyz</div><div>${fmtNumber(same.translation_mm, 1)} mm</div>
        <div class="key">lag xyz</div><div>${fmtNumber(lag.translation_mm, 1)} mm</div>
        <div class="key">same rot</div><div>${fmtNumber(same.rotation_deg, 2)} deg</div>
        <div class="key">lag rot</div><div>${fmtNumber(lag.rotation_deg, 2)} deg</div>
        <div class="key">same pixel</div><div>${currentMetric(side, "state_px", "px")}</div>
        <div class="key">lag pixel</div><div>${currentMetric(side, "lag_state_px", "px")}</div>
      </div>
    </div>`);
  }
  els.currentStats.innerHTML = blocks.join("");
}
function updateReadouts() {
  const frame = frames[state.index];
  els.slider.value = String(state.index);
  els.playBtn.textContent = state.playing ? "Pause" : "Play";
  if (!frame) {
    els.frameReadout.textContent = "0 / 0";
    els.timeReadout.textContent = "";
    return;
  }
  els.frameReadout.textContent = `${state.index + 1} / ${frames.length} | row ${frame.row_index} | video ${frame.video_frame_index}`;
  els.timeReadout.textContent = `t ${fmtTime(frame.timestamp)} | cam age ${fmtMs(frame.camera_age_sec)}`;
}
function render() {
  drawImage();
  drawOverlay();
  drawCurrentStats();
  updateReadouts();
}
function setIndex(index) {
  if (!frames.length) return;
  state.index = Math.max(0, Math.min(frames.length - 1, index));
  render();
}
function playStep() {
  if (!state.playing) return;
  if (state.index >= frames.length - 1) {
    state.playing = false;
    updateReadouts();
    return;
  }
  setIndex(state.index + 1);
  state.timer = window.setTimeout(playStep, 90);
}
function togglePlay() {
  state.playing = !state.playing;
  updateReadouts();
  if (state.playing) playStep();
  if (!state.playing && state.timer) window.clearTimeout(state.timer);
}
function setupEvents() {
  els.slider.addEventListener("input", () => {
    state.playing = false;
    setIndex(Number(els.slider.value || 0));
  });
  els.playBtn.addEventListener("click", togglePlay);
  for (const input of Object.values(els.toggles)) {
    input.addEventListener("change", render);
  }
  window.addEventListener("keydown", (event) => {
    if (event.key === " ") {
      event.preventDefault();
      togglePlay();
    } else if (event.key === "ArrowRight") {
      setIndex(state.index + 1);
    } else if (event.key === "ArrowLeft") {
      setIndex(state.index - 1);
    }
  });
  window.addEventListener("resize", render);
}
function initCanvas() {
  setCanvasSize(els.imageCanvas);
  setCanvasSize(els.overlayCanvas);
}
setupStaticText();
setupEvents();
initCanvas();
setIndex(Number.isFinite(payload.initial_index) ? payload.initial_index : 0);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
