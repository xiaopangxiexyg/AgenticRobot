#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import pyarrow.parquet as pq

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from common.video_io import DEFAULT_VIDEO_CANDIDATES, PortableVideoWriter, RobustVideoReader, resolve_video_source
from spatial_tactile.common import BETA_DIM, THETA_DIM
from spatial_tactile.dataset_schema import (
    TACTILE_CLOUD_COLUMNS,
    reshape_sensor_valid_mask,
    reshape_sensor_xyz,
)
from spatial_tactile.mano_runtime import ManoRuntime, resolve_torch_device


FIRST_PERSON_SERIAL = "151322070562"
EEF_DIM = 3
ROW_BATCH_SIZE = 128
DEFAULT_HAND_COLORS = {
    "left": (120, 255, 140),
    "right": (80, 180, 255),
}
PORTABLE_ROOT = Path(__file__).resolve().parent
PORTABLE_CAMERA_INTRINSICS_DIR = PORTABLE_ROOT / "assets" / "camera_intrinsics"
_MANO_RUNTIME_CACHE: Dict[str, ManoRuntime] = {}
RENDER_MODE_ALIASES = {
    "tactile": "tactile",
    "tactile_only": "tactile",
    "pointcloud": "tactile",
    "point_cloud": "tactile",
    "mano": "mano",
    "mano_only": "mano",
    "mesh": "mano",
    "both": "both",
    "overlay": "both",
    "mano+tactile": "both",
    "tactile+mano": "both",
    "mano_and_tactile": "both",
    "tactile_and_mano": "both",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project saved spatial tactile point clouds and/or MANO states back onto the episode video.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python visualization/portable_vis/visualize_lerobot_spatial_tactile.py /path/to/dataset_root --episode 12\n"
            "  python visualization/portable_vis/visualize_lerobot_spatial_tactile.py /path/to/data/episode_000012.parquet\n"
            "  python visualization/portable_vis/visualize_lerobot_spatial_tactile.py /path/to/dataset_root --episode 12 --output_dir /tmp/spatial_tactile_vis\n"
            "  python visualization/portable_vis/visualize_lerobot_spatial_tactile.py /path/to/dataset_root --episode 12 --hand right --point_radius 5\n"
            "  python visualization/portable_vis/visualize_lerobot_spatial_tactile.py /path/to/dataset_root --episode 12 --render-mode both --mano-alpha 0.3\n"
            "  python visualization/portable_vis/visualize_lerobot_spatial_tactile.py /path/to/dataset_root --episode 12 --render-mode mano"
        ),
    )
    parser.add_argument("input_path", type=str, help="Input dataset root, data/ directory, or one episode parquet.")
    parser.add_argument(
        "output_path",
        nargs="?",
        default="",
        help="Optional output video path or output directory. Defaults to dataset_root/visualizations/.",
    )
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        dest="dataset_root",
        type=str,
        default="",
        help="Advanced override for the dataset root containing data/meta/videos.",
    )
    parser.add_argument(
        "-e",
        "--episode",
        "--episode_index",
        dest="episode_indices",
        nargs="+",
        default=None,
        metavar="EPISODE",
        help="Episode index or indices to render when INPUT points to a directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--output-target",
        "--output_target",
        dest="output_target",
        type=str,
        default="",
        help="Generic output target. Can be a video file path or an output directory.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default="",
        help="Explicit output directory. The episode filename will be created inside this directory.",
    )
    parser.add_argument(
        "--output-file",
        "--output_file",
        "--output-video",
        "--output_video",
        "--output-path",
        "--output_path",
        dest="output_file",
        type=str,
        default="",
        help="Explicit output video file path for a single rendered episode.",
    )
    parser.add_argument("--hand", type=str, default="both", choices=["left", "right", "both"], help="Which hand(s) to draw.")
    parser.add_argument("--point-radius", "--point_radius", dest="point_radius", type=int, default=4, help="Point radius in pixels.")
    parser.add_argument(
        "--render-mode",
        "--render_mode",
        "--mode",
        dest="render_mode",
        type=str,
        default="",
        metavar="MODE",
        help=(
            "Render mode: tactile, mano, or both. "
            "Default is tactile. Legacy --with-mano is treated as --render-mode both."
        ),
    )
    parser.add_argument("--with-mano", action="store_true", help="Legacy alias for --render-mode both.")
    parser.add_argument(
        "--mano-alpha",
        "--mano_alpha",
        dest="mano_alpha",
        type=float,
        default=0.30,
        help="Alpha blending factor for the MANO mesh overlay when render mode includes MANO.",
    )
    parser.add_argument(
        "--mano-device",
        "--mano_device",
        dest="mano_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="MANO rendering device when render mode includes MANO.",
    )
    parser.add_argument(
        "--with-mano-axes",
        action="store_true",
        help="Draw wrist axes together with the MANO mesh overlay when render mode includes MANO.",
    )
    parser.add_argument("--max-frames", "--max_frames", dest="max_frames", type=int, default=0, help="Optional frame limit.")
    parser.add_argument(
        "--frame-interval",
        "--frame_interval",
        dest="frame_interval",
        type=int,
        default=1,
        help="Render every Nth frame while keeping the output FPS unchanged.",
    )
    parser.add_argument("--camera_serial", type=str, default="", help="Override first-person camera serial if needed.")
    parser.add_argument(
        "--camera-intrinsics-json",
        "--camera_intrinsics_json",
        dest="camera_intrinsics_json",
        type=str,
        default="",
        help="Optional camera intrinsics JSON path.",
    )
    return parser.parse_args()


def _parse_episode_indices(raw_values: Optional[Sequence[str]]) -> List[int]:
    if not raw_values:
        return [0]
    episode_indices: List[int] = []
    seen: set[int] = set()
    for raw_value in raw_values:
        for token in str(raw_value).split(","):
            stripped = token.strip()
            if not stripped:
                continue
            value = int(stripped)
            if value not in seen:
                seen.add(value)
                episode_indices.append(value)
    return episode_indices or [0]


def _resolve_episode_file(episode_path_arg: str, episode_index: int) -> str:
    candidate = Path(os.path.abspath(episode_path_arg))
    if candidate.is_file():
        if candidate.suffix.lower() != ".parquet":
            raise ValueError(f"Expected a parquet file, got: {candidate}")
        return str(candidate)

    search_dirs: List[Path] = []
    if candidate.is_dir():
        search_dirs.append(candidate)
        data_dir = candidate / "data"
        if data_dir.is_dir():
            search_dirs.append(data_dir)

    requested = f"episode_{int(episode_index):06d}.parquet"
    for search_dir in search_dirs:
        parquet_path = search_dir / requested
        if parquet_path.is_file():
            return str(parquet_path)

    raise FileNotFoundError(f"Could not resolve {requested} from {episode_path_arg}")


def _guess_dataset_root(episode_path: str, dataset_root_arg: str) -> Optional[str]:
    if dataset_root_arg:
        return os.path.abspath(dataset_root_arg)
    episode_parent = Path(os.path.abspath(episode_path)).parent
    if episode_parent.name == "data":
        return str(episode_parent.parent)
    return None


class EpisodeRowStream:
    def __init__(self, episode_path: str, batch_size: int = ROW_BATCH_SIZE) -> None:
        self.episode_path = os.path.abspath(episode_path)
        self.batch_size = max(1, int(batch_size))
        self.parquet_file = pq.ParquetFile(self.episode_path)
        self.row_count = int(self.parquet_file.metadata.num_rows)
        if self.row_count <= 0:
            raise RuntimeError(f"Episode parquet is empty: {self.episode_path}")
        self._batch_iter = self.parquet_file.iter_batches(batch_size=self.batch_size)
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_index = 0

    def _fill_buffer(self) -> bool:
        if self._buffer_index < len(self._buffer):
            return True
        batch = next(self._batch_iter, None)
        if batch is None:
            self._buffer = []
            self._buffer_index = 0
            return False
        self._buffer = batch.to_pylist()
        self._buffer_index = 0
        return bool(self._buffer)

    def peek_first_row(self) -> Dict[str, Any]:
        if not self._fill_buffer():
            raise RuntimeError(f"Episode parquet is empty: {self.episode_path}")
        return dict(self._buffer[self._buffer_index])

    def read_row(self) -> Optional[Dict[str, Any]]:
        if not self._fill_buffer():
            return None
        row = dict(self._buffer[self._buffer_index])
        self._buffer_index += 1
        if self._buffer_index >= len(self._buffer):
            self._buffer = []
            self._buffer_index = 0
        return row


def _load_episode_meta_row(dataset_root: Optional[str], episode_index: int) -> Dict[str, Any]:
    if not dataset_root:
        return {}
    episodes_meta_path = os.path.join(dataset_root, "meta", "episodes.parquet")
    if not os.path.isfile(episodes_meta_path):
        return {}
    parquet_file = pq.ParquetFile(episodes_meta_path)
    for batch in parquet_file.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            if int(row.get("episode_index", -1)) == int(episode_index):
                return row
    return {}


def _resolve_camera_serial(episode_meta: Dict[str, Any], override_serial: str) -> str:
    if override_serial.strip():
        return override_serial.strip()
    meta_serial = str(episode_meta.get("first_person_camera_serial", "")).strip()
    if meta_serial:
        return meta_serial
    return FIRST_PERSON_SERIAL


def _camera_intrinsics_from_dict(intrinsics: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    K = np.array(
        [
            [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    D = np.asarray(intrinsics.get("coeffs", []), dtype=np.float32)
    return K, D


def _load_intrinsics_from_session_meta(meta_path: str, camera_serial: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as file:
        meta = json.load(file)
    for camera in meta.get("cameras", []):
        if str(camera.get("serial", "")).strip() != str(camera_serial):
            continue
        intrinsics = camera["streams"]["color"]["intrinsics"]
        return _camera_intrinsics_from_dict(intrinsics)
    return None


def _load_intrinsics_from_json_payload(payload: Dict[str, Any], camera_serial: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    payload_serial = str(payload.get("camera_serial", "")).strip()
    if payload_serial and payload_serial != str(camera_serial):
        return None

    if "intrinsics" in payload and isinstance(payload["intrinsics"], dict):
        intrinsics = payload["intrinsics"]
        required = {"fx", "fy", "cx", "cy"}
        if required.issubset(intrinsics):
            return _camera_intrinsics_from_dict(intrinsics)

    serial_key = str(camera_serial)
    if serial_key in payload and isinstance(payload[serial_key], dict):
        nested = payload[serial_key]
        if "intrinsics" in nested and isinstance(nested["intrinsics"], dict):
            return _camera_intrinsics_from_dict(nested["intrinsics"])
        required = {"fx", "fy", "cx", "cy"}
        if required.issubset(nested):
            return _camera_intrinsics_from_dict(nested)

    cameras = payload.get("cameras", [])
    if isinstance(cameras, list):
        for camera in cameras:
            if str(camera.get("serial", "")).strip() != serial_key:
                continue
            color = camera.get("streams", {}).get("color", {})
            intrinsics = color.get("intrinsics", {})
            required = {"fx", "fy", "cx", "cy"}
            if required.issubset(intrinsics):
                return _camera_intrinsics_from_dict(intrinsics)

    required = {"fx", "fy", "cx", "cy"}
    if required.issubset(payload):
        return _camera_intrinsics_from_dict(payload)
    return None


def _load_intrinsics_from_json_file(json_path: str, camera_serial: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if not json_path or not os.path.isfile(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Camera intrinsics JSON must contain an object: {json_path}")
    return _load_intrinsics_from_json_payload(payload, camera_serial)


def _candidate_intrinsics_paths(dataset_root: Optional[str], camera_serial: str, override_path: str) -> List[tuple[str, str]]:
    candidates: List[tuple[str, str]] = []
    if override_path:
        candidates.append(("explicit_json", os.path.abspath(os.path.expanduser(override_path))))
    if dataset_root:
        root = Path(dataset_root)
        candidates.append(("dataset_meta_camera_intrinsics_serial", str(root / "meta" / "camera_intrinsics" / f"{camera_serial}.json")))
        candidates.append(("dataset_meta_camera_intrinsics", str(root / "meta" / "camera_intrinsics.json")))
    return candidates


def _load_camera_intrinsics(
    session_dir: str,
    camera_serial: str,
    dataset_root: Optional[str] = None,
    override_path: str = "",
) -> tuple[np.ndarray, np.ndarray, str]:
    search_paths = _candidate_intrinsics_paths(dataset_root, camera_serial, override_path)
    for source_name, candidate_path in search_paths:
        intrinsics = _load_intrinsics_from_json_file(candidate_path, camera_serial)
        if intrinsics is not None:
            return intrinsics[0], intrinsics[1], f"{source_name}:{candidate_path}"

    meta_path = os.path.join(session_dir, "meta.json") if session_dir else ""
    intrinsics = _load_intrinsics_from_session_meta(meta_path, camera_serial) if meta_path else None
    if intrinsics is not None:
        return intrinsics[0], intrinsics[1], f"session_meta:{meta_path}"

    portable_paths = [
        ("portable_asset_camera_intrinsics_serial", str(PORTABLE_CAMERA_INTRINSICS_DIR / f"{camera_serial}.json")),
        ("portable_asset_camera_intrinsics", str(PORTABLE_CAMERA_INTRINSICS_DIR / "default.json")),
    ]
    for source_name, candidate_path in portable_paths:
        intrinsics = _load_intrinsics_from_json_file(candidate_path, camera_serial)
        if intrinsics is not None:
            return intrinsics[0], intrinsics[1], f"{source_name}:{candidate_path}"

    raise FileNotFoundError(
        f"Could not resolve camera intrinsics for serial {camera_serial}. "
        f"Searched session meta {meta_path or '<none>'} and JSON fallbacks."
    )


def _resolve_episode_video_path(
    rows: Sequence[Dict[str, Any]],
    dataset_root: Optional[str],
    session_dir: str,
    camera_serial: str,
) -> str:
    video_relpath = str(rows[0].get("video.top_view", "")).strip()
    if dataset_root and video_relpath:
        candidate = os.path.join(dataset_root, video_relpath)
        if os.path.isfile(candidate):
            return candidate
    if not session_dir:
        raise FileNotFoundError("Could not resolve video.top_view and no session_dir was available.")
    camera_dir = os.path.join(session_dir, f"RealSense_{camera_serial}")
    video_path = resolve_video_source(camera_dir, candidates=DEFAULT_VIDEO_CANDIDATES)
    if not video_path:
        raise FileNotFoundError(f"Could not resolve a video source under {camera_dir}")
    return video_path


def _resolve_render_mode(render_mode_arg: str, with_mano: bool = False) -> str:
    normalized = str(render_mode_arg or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return "both" if with_mano else "tactile"
    resolved = RENDER_MODE_ALIASES.get(normalized)
    if resolved is None:
        valid_modes = ", ".join(sorted(set(RENDER_MODE_ALIASES.values())))
        raise ValueError(f"Unsupported --render-mode value: {render_mode_arg!r}. Expected one of: {valid_modes}")
    if with_mano and resolved == "tactile":
        raise ValueError("Conflicting MANO options: --with-mano cannot be combined with --render-mode tactile.")
    return resolved


def _render_mode_includes_tactile(render_mode: str) -> bool:
    return str(render_mode) in {"tactile", "both"}


def _render_mode_includes_mano(render_mode: str) -> bool:
    return str(render_mode) in {"mano", "both"}


def _default_output_filename(episode_path: str, episode_index: int, render_mode: str = "tactile") -> str:
    path = Path(os.path.abspath(episode_path))
    if render_mode == "both":
        suffix = "_spatial_tactile_mano.mp4"
    elif render_mode == "mano":
        suffix = "_spatial_tactile_mano_only.mp4"
    else:
        suffix = "_spatial_tactile.mp4"
    if path.stem.startswith("episode_"):
        return f"{path.stem}{suffix}"
    return f"episode_{int(episode_index):06d}{suffix}"


def _resolve_output_video_path(
    output_arg: str,
    episode_path: str,
    dataset_root: Optional[str],
    episode_index: int,
    render_mode: str = "tactile",
    force_output_dir: bool = False,
    force_output_file: bool = False,
) -> str:
    default_filename = _default_output_filename(episode_path, episode_index, render_mode=render_mode)
    if not output_arg:
        if dataset_root:
            return os.path.join(dataset_root, "visualizations", default_filename)
        episode_parent = Path(os.path.abspath(episode_path)).parent
        return str(episode_parent / "visualizations" / default_filename)

    candidate = Path(os.path.abspath(os.path.expanduser(output_arg)))
    if force_output_dir:
        return str(candidate / default_filename)
    if force_output_file:
        return str(candidate)
    if (candidate.exists() and candidate.is_dir()) or not candidate.suffix:
        return str(candidate / default_filename)
    return str(candidate)


def _resolve_cli_output_arg(args: argparse.Namespace) -> tuple[str, str]:
    raw_sources = [
        ("positional OUTPUT", str(getattr(args, "output_path", "") or "").strip(), "auto"),
        ("--output", str(getattr(args, "output_target", "") or "").strip(), "auto"),
        ("--output_dir", str(getattr(args, "output_dir", "") or "").strip(), "dir"),
        ("--output_file", str(getattr(args, "output_file", "") or "").strip(), "file"),
    ]
    sources = [
        (label, os.path.abspath(os.path.expanduser(value)), mode)
        for label, value, mode in raw_sources
        if value
    ]
    if not sources:
        return "", "auto"

    unique_paths: Dict[str, List[tuple[str, str]]] = {}
    for label, path, mode in sources:
        unique_paths.setdefault(path, []).append((label, mode))
    if len(unique_paths) > 1:
        details = ", ".join(f"{label}={path}" for label, path, _ in sources)
        raise ValueError(
            "Please provide only one output target via positional OUTPUT, --output, --output_dir, or --output_file. "
            f"Received: {details}"
        )

    selected_path = next(iter(unique_paths))
    selected_sources = unique_paths[selected_path]
    explicit_modes = {mode for _, mode in selected_sources if mode in {"dir", "file"}}
    if len(explicit_modes) > 1:
        raise ValueError("Please do not use --output_dir and --output_file for the same render command.")
    selected_mode = next(iter(explicit_modes)) if explicit_modes else "auto"
    return selected_path, selected_mode


def _compute_output_fps_from_row_timestamps(
    episode_path: str,
    frame_limit: int,
    fallback_fps: float,
) -> float:
    if int(frame_limit) <= 1:
        return float(fallback_fps)
    table = pq.read_table(episode_path, columns=["timestamp_ms"]).slice(0, int(frame_limit))
    timestamps_ms = np.asarray(table.column("timestamp_ms").to_pylist(), dtype=np.float64)
    if timestamps_ms.size <= 1:
        return float(fallback_fps)
    duration_s = float((timestamps_ms[-1] - timestamps_ms[0]) / 1000.0)
    if not np.isfinite(duration_s) or duration_s <= 1e-9:
        return float(fallback_fps)
    derived_fps = float((timestamps_ms.size - 1) / duration_s)
    if not np.isfinite(derived_fps) or derived_fps <= 1e-9:
        return float(fallback_fps)
    return derived_fps


def _normalize_tactile_values(values: Any) -> np.ndarray:
    tactile_values = np.asarray(values, dtype=np.float32).reshape(-1)
    if tactile_values.size <= 0:
        return tactile_values
    max_value = float(np.nanmax(tactile_values))
    if max_value > 1.1:
        tactile_values = tactile_values / 255.0
    tactile_values = np.clip(tactile_values, 0.0, 1.0)
    return tactile_values.astype(np.float32, copy=False)


def _project_points(points_cam: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    points_cam = np.ascontiguousarray(np.asarray(points_cam, dtype=np.float64).reshape(-1, 3))
    intrinsic_matrix = np.ascontiguousarray(np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3))
    distortion = np.ascontiguousarray(np.asarray(dist_coeffs, dtype=np.float64).reshape(-1))

    x = np.full((points_cam.shape[0],), np.nan, dtype=np.float64)
    y = np.full((points_cam.shape[0],), np.nan, dtype=np.float64)
    valid_z = np.isfinite(points_cam[:, 2]) & (np.abs(points_cam[:, 2]) > 1e-9)
    x[valid_z] = points_cam[valid_z, 0] / points_cam[valid_z, 2]
    y[valid_z] = points_cam[valid_z, 1] / points_cam[valid_z, 2]

    k1 = float(distortion[0]) if distortion.size > 0 else 0.0
    k2 = float(distortion[1]) if distortion.size > 1 else 0.0
    p1 = float(distortion[2]) if distortion.size > 2 else 0.0
    p2 = float(distortion[3]) if distortion.size > 3 else 0.0
    k3 = float(distortion[4]) if distortion.size > 4 else 0.0
    k4 = float(distortion[5]) if distortion.size > 5 else 0.0
    k5 = float(distortion[6]) if distortion.size > 6 else 0.0
    k6 = float(distortion[7]) if distortion.size > 7 else 0.0
    s1 = float(distortion[8]) if distortion.size > 8 else 0.0
    s2 = float(distortion[9]) if distortion.size > 9 else 0.0
    s3 = float(distortion[10]) if distortion.size > 10 else 0.0
    s4 = float(distortion[11]) if distortion.size > 11 else 0.0

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial_num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    radial_den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
    radial_den = np.where(np.abs(radial_den) > 1e-12, radial_den, 1.0)
    radial = radial_num / radial_den

    xy = x * y
    x_tangential = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x * x)
    y_tangential = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * xy
    x_prism = s1 * r2 + s2 * r4
    y_prism = s3 * r2 + s4 * r4
    x_distorted = x * radial + x_tangential + x_prism
    y_distorted = y * radial + y_tangential + y_prism
    u = intrinsic_matrix[0, 0] * x_distorted + intrinsic_matrix[0, 1] * y_distorted + intrinsic_matrix[0, 2]
    v = intrinsic_matrix[1, 1] * y_distorted + intrinsic_matrix[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32, copy=False)


def _axis_angle_to_matrix_numpy(axis_angle: np.ndarray) -> np.ndarray:
    rvec = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rvec))
    if theta <= 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rvec / theta
    x, y, z = axis.tolist()
    cross = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    eye = np.eye(3, dtype=np.float64)
    return eye + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


def _safe_int_point(xy: Any) -> Optional[tuple[int, int]]:
    point = np.asarray(xy, dtype=np.float64).reshape(-1)
    if point.size < 2 or not np.all(np.isfinite(point[:2])):
        return None
    return (int(round(float(point[0]))), int(round(float(point[1]))))


def _fill_triangle_numpy(image: np.ndarray, triangle_2d: np.ndarray, color: tuple[int, int, int]) -> None:
    points = np.asarray(triangle_2d, dtype=np.float64).reshape(3, 2)
    if not np.all(np.isfinite(points)):
        return

    height, width = image.shape[:2]
    x_min = max(int(np.floor(np.min(points[:, 0]))), 0)
    x_max = min(int(np.ceil(np.max(points[:, 0]))), width - 1)
    y_min = max(int(np.floor(np.min(points[:, 1]))), 0)
    y_max = min(int(np.ceil(np.max(points[:, 1]))), height - 1)
    if x_min > x_max or y_min > y_max:
        return

    p0, p1, p2 = points

    def _edge(a: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x - a[0]) * (b[1] - a[1]) - (y - a[1]) * (b[0] - a[0])

    area = float(_edge(p0, p1, np.array(p2[0]), np.array(p2[1])))
    if abs(area) <= 1e-9:
        return

    xs = np.arange(x_min, x_max + 1, dtype=np.float64) + 0.5
    ys = np.arange(y_min, y_max + 1, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)

    w0 = _edge(p1, p2, grid_x, grid_y)
    w1 = _edge(p2, p0, grid_x, grid_y)
    w2 = _edge(p0, p1, grid_x, grid_y)
    if area > 0.0:
        mask = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
    else:
        mask = (w0 <= 0.0) & (w1 <= 0.0) & (w2 <= 0.0)
    if not np.any(mask):
        return

    region = image[y_min : y_max + 1, x_min : x_max + 1]
    yy, xx = np.nonzero(mask)
    if yy.size <= 0:
        return
    if region.ndim == 3 and region.shape[2] >= 3:
        region[yy, xx, 0] = int(color[0])
        region[yy, xx, 1] = int(color[1])
        region[yy, xx, 2] = int(color[2])
    else:
        region[yy, xx] = int(color[0])


def _draw_frame_axes_numpy(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    axis_length: float,
    thickness: int,
) -> None:
    rotation_matrix = _axis_angle_to_matrix_numpy(rvec)
    origin_and_axes = np.array(
        [
            [0.0, 0.0, 0.0],
            [float(axis_length), 0.0, 0.0],
            [0.0, float(axis_length), 0.0],
            [0.0, 0.0, float(axis_length)],
        ],
        dtype=np.float64,
    )
    translation = np.asarray(tvec, dtype=np.float64).reshape(1, 3)
    axis_points_cam = origin_and_axes @ rotation_matrix.T + translation
    projected = _project_points(axis_points_cam, camera_matrix, dist_coeffs)
    if not np.all(np.isfinite(projected[0])) or axis_points_cam[0, 2] <= 1e-6:
        return

    origin_xy = _safe_int_point(projected[0])
    if origin_xy is None:
        return
    colors = (
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
    )
    for idx, color in zip((1, 2, 3), colors):
        if not np.all(np.isfinite(projected[idx])) or axis_points_cam[idx, 2] <= 1e-6:
            continue
        endpoint_xy = _safe_int_point(projected[idx])
        if endpoint_xy is None:
            continue
        cv2.line(frame, origin_xy, endpoint_xy, color, int(thickness), lineType=cv2.LINE_AA)


def _render_mesh_overlay(
    frame: np.ndarray,
    mesh_payloads: Sequence[Dict[str, Any]],
    alpha: float,
) -> np.ndarray:
    overlay = frame.copy()
    face_entries: List[tuple[float, np.ndarray, tuple[int, int, int]]] = []
    for payload in mesh_payloads:
        verts_cam = payload["verts_cam"]
        projected = payload["projected"]
        faces = payload["faces"]
        color = payload["color"]
        valid_face_mask = np.all(verts_cam[faces, 2] > 1e-4, axis=1)
        valid_faces = faces[valid_face_mask]
        for face in valid_faces:
            triangle_projected = projected[face]
            if not np.all(np.isfinite(triangle_projected)):
                continue
            triangle_2d = np.ascontiguousarray(np.round(triangle_projected).astype(np.int32, copy=False).reshape(3, 2))
            mean_depth = float(np.mean(verts_cam[face, 2]))
            face_entries.append((mean_depth, triangle_2d, color))

    face_entries.sort(key=lambda item: item[0], reverse=True)
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    for _, triangle_2d, color in face_entries:
        _fill_triangle_numpy(overlay, triangle_2d, color)

    blended = cv2.addWeighted(overlay, alpha_clamped, frame, 1.0 - alpha_clamped, 0.0)
    for payload in mesh_payloads:
        wrist_xy = _safe_int_point(payload["projected_wrist"])
        if wrist_xy is None:
            continue
        cv2.circle(blended, wrist_xy, 4, payload["color"], -1, lineType=cv2.LINE_AA)
        cv2.putText(
            blended,
            payload["label"],
            (wrist_xy[0] + 6, wrist_xy[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            payload["color"],
            2,
            lineType=cv2.LINE_AA,
        )
    return blended


def _draw_spatial_tactile_points(
    frame: np.ndarray,
    points_cam: np.ndarray,
    tactile_values: np.ndarray,
    valid_mask: np.ndarray,
    hand_side: str,
    point_radius: int,
) -> int:
    tactile_values = _normalize_tactile_values(tactile_values)
    projected = _project_points(points_cam, camera_matrix=_draw_spatial_tactile_points.camera_matrix, dist_coeffs=_draw_spatial_tactile_points.dist_coeffs)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    valid &= np.isfinite(projected[:, 0]) & np.isfinite(projected[:, 1])
    valid &= np.isfinite(points_cam[:, 2]) & (points_cam[:, 2] > 1e-6)
    if not np.any(valid):
        return 0

    valid_indices = np.nonzero(valid)[0]
    sorted_indices = valid_indices[np.argsort(points_cam[valid_indices, 2])[::-1]]
    value_colors = cv2.applyColorMap(
        np.clip(np.round(tactile_values * 255.0), 0.0, 255.0).astype(np.uint8).reshape(-1, 1),
        cv2.COLORMAP_TURBO,
    ).reshape(-1, 3)
    outline_color = DEFAULT_HAND_COLORS[hand_side]

    for index in sorted_indices:
        point_xy = projected[index]
        center = (int(round(float(point_xy[0]))), int(round(float(point_xy[1]))))
        fill_color = tuple(int(v) for v in value_colors[index].tolist())
        cv2.circle(frame, center, int(point_radius), fill_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, center, int(max(1, point_radius)), outline_color, 1, lineType=cv2.LINE_AA)
    return int(sorted_indices.size)


_draw_spatial_tactile_points.camera_matrix = np.eye(3, dtype=np.float32)
_draw_spatial_tactile_points.dist_coeffs = np.zeros((5,), dtype=np.float32)


def _to_vector(row: Dict[str, Any], key: str, dim: int) -> np.ndarray:
    values = row.get(key, [])
    vector = np.zeros((dim,), dtype=np.float32)
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size:
        vector[: min(dim, array.size)] = array[:dim]
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _should_render_hand(row: Dict[str, Any], hand_side: str) -> bool:
    if hand_side == "right":
        position_key = "state.eef_position"
        rotation_key = "state.eef_rotation"
        theta_key = "state.right_theta"
    else:
        position_key = "state.left_eef_position"
        rotation_key = "state.left_eef_rotation"
        theta_key = "state.left_theta"
    signal = np.linalg.norm(_to_vector(row, position_key, EEF_DIM))
    signal += np.linalg.norm(_to_vector(row, rotation_key, EEF_DIM))
    signal += np.linalg.norm(_to_vector(row, theta_key, THETA_DIM))
    return float(signal) > 1e-6


def _get_mano_runtime(device: str) -> ManoRuntime:
    if device not in _MANO_RUNTIME_CACHE:
        _MANO_RUNTIME_CACHE[device] = ManoRuntime(device=device)
    return _MANO_RUNTIME_CACHE[device]


def _mano_vertices_in_camera_frame(row: Dict[str, Any], hand_side: str, device: str) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if not _should_render_hand(row, hand_side):
        return None
    if hand_side == "right":
        position = _to_vector(row, "state.eef_position", EEF_DIM)
        rotation = _to_vector(row, "state.eef_rotation", EEF_DIM)
        theta = _to_vector(row, "state.right_theta", THETA_DIM)
        beta = _to_vector(row, "state.right_beta", BETA_DIM)
    else:
        position = _to_vector(row, "state.left_eef_position", EEF_DIM)
        rotation = _to_vector(row, "state.left_eef_rotation", EEF_DIM)
        theta = _to_vector(row, "state.left_theta", THETA_DIM)
        beta = _to_vector(row, "state.left_beta", BETA_DIM)

    runtime = _get_mano_runtime(device)
    output = runtime.compute(hand_side=hand_side, theta=theta, beta=beta, root_rotation=rotation)
    verts_cam = np.ascontiguousarray(output.verts + position.reshape(1, 3), dtype=np.float32)
    return verts_cam, output.faces, position, rotation


def _build_mano_mesh_payloads(
    row: Dict[str, Any],
    requested_hands: Sequence[str],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    device: str,
) -> List[Dict[str, Any]]:
    mesh_payloads: List[Dict[str, Any]] = []
    for hand_side in requested_hands:
        mano_mesh = _mano_vertices_in_camera_frame(row, hand_side, device)
        if mano_mesh is None:
            continue
        verts_cam, faces, position, rotation = mano_mesh
        projected = _project_points(verts_cam, camera_matrix, dist_coeffs)
        mesh_payloads.append(
            {
                "hand_side": hand_side,
                "verts_cam": verts_cam,
                "faces": faces,
                "projected": projected,
                "projected_wrist": projected[0],
                "color": DEFAULT_HAND_COLORS[hand_side],
                "label": f"{hand_side.upper()} MANO",
                "rvec": rotation,
                "tvec": position,
            }
        )
    return mesh_payloads


def render_episode_spatial_tactile_overlay(
    episode_path_arg: str,
    *,
    dataset_root: str = "",
    output_video: str = "",
    output_mode: str = "auto",
    hand: str = "both",
    point_radius: int = 4,
    render_mode: str = "",
    with_mano: bool = False,
    mano_alpha: float = 0.30,
    mano_device: str = "auto",
    with_mano_axes: bool = False,
    max_frames: int = 0,
    frame_interval: int = 1,
    camera_serial: str = "",
    camera_intrinsics_json: str = "",
    episode_index: int = 0,
    verbose: bool = True,
) -> str:
    episode_path = _resolve_episode_file(episode_path_arg, int(episode_index))
    row_stream = EpisodeRowStream(episode_path)
    first_row = row_stream.peek_first_row()
    resolved_episode_index = int(first_row.get("episode_index", 0))
    requested_hands = ["left", "right"] if hand == "both" else [hand]
    resolved_render_mode = _resolve_render_mode(render_mode, with_mano=bool(with_mano))
    render_tactile = _render_mode_includes_tactile(resolved_render_mode)
    render_mano = _render_mode_includes_mano(resolved_render_mode)
    if with_mano_axes and not render_mano:
        raise ValueError("--with-mano-axes requires render mode to include MANO.")

    if render_tactile:
        missing_columns = []
        for hand_side in requested_hands:
            for column_name in (
                TACTILE_CLOUD_COLUMNS[hand_side]["xyz"],
                TACTILE_CLOUD_COLUMNS[hand_side]["valid_mask"],
                TACTILE_CLOUD_COLUMNS[hand_side]["source_tactile"],
            ):
                if column_name not in first_row:
                    missing_columns.append(column_name)
        if missing_columns:
            raise KeyError(
                "The episode parquet does not contain the columns required for spatial tactile rendering. "
                f"Missing: {missing_columns}"
            )

    resolved_dataset_root = _guess_dataset_root(episode_path, dataset_root)
    episode_meta = _load_episode_meta_row(resolved_dataset_root, resolved_episode_index)
    session_dir = str(first_row.get("session_dir", "")).strip() or str(episode_meta.get("session_dir", "")).strip()
    resolved_camera_serial = _resolve_camera_serial(episode_meta, camera_serial)
    camera_matrix, dist_coeffs, intrinsics_source = _load_camera_intrinsics(
        session_dir=session_dir,
        camera_serial=resolved_camera_serial,
        dataset_root=resolved_dataset_root,
        override_path=camera_intrinsics_json,
    )
    _draw_spatial_tactile_points.camera_matrix = camera_matrix
    _draw_spatial_tactile_points.dist_coeffs = dist_coeffs

    video_path = _resolve_episode_video_path([first_row], resolved_dataset_root, session_dir, resolved_camera_serial)
    resolved_output_video = _resolve_output_video_path(
        output_arg=output_video,
        episode_path=episode_path,
        dataset_root=resolved_dataset_root,
        episode_index=resolved_episode_index,
        render_mode=resolved_render_mode,
        force_output_dir=(str(output_mode) == "dir"),
        force_output_file=(str(output_mode) == "file"),
    )
    resolved_mano_device = resolve_torch_device(mano_device) if render_mano else ""

    output_dir = os.path.dirname(resolved_output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    reader = RobustVideoReader(video_path, expected_frame_count=row_stream.row_count)
    frame_limit = int(row_stream.row_count)
    if int(max_frames) > 0:
        frame_limit = min(frame_limit, int(max_frames))
    frame_interval = max(1, int(frame_interval))
    output_fps = _compute_output_fps_from_row_timestamps(
        episode_path=episode_path,
        frame_limit=frame_limit,
        fallback_fps=float(reader.fps if reader.fps > 0 else 30.0),
    )

    writer = PortableVideoWriter(
        resolved_output_video,
        float(output_fps),
        (int(reader.width), int(reader.height)),
    )

    if verbose:
        print(f"episode_path={episode_path}")
        print(f"video_path={video_path}")
        print(f"output_video={resolved_output_video}")
        print(f"camera_serial={resolved_camera_serial}")
        print(f"camera_intrinsics_source={intrinsics_source}")
        print(f"render_mode={resolved_render_mode}")
        if render_mano:
            print(f"mano_device={resolved_mano_device}")
            print(f"mano_alpha={float(mano_alpha):.3f}")
            print(f"with_mano_axes={bool(with_mano_axes)}")

    rendered_frame_count = 0

    try:
        for frame_index in range(frame_limit):
            ret, frame = reader.read()
            if not ret:
                break
            row = row_stream.read_row()
            if row is None:
                break
            if frame_index % frame_interval != 0:
                continue

            output_frame = frame.copy()
            mesh_payloads: List[Dict[str, Any]] = []
            if render_mano:
                mesh_payloads = _build_mano_mesh_payloads(
                    row=row,
                    requested_hands=requested_hands,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    device=resolved_mano_device,
                )
                if mesh_payloads:
                    output_frame = _render_mesh_overlay(output_frame, mesh_payloads, alpha=float(mano_alpha))
                    if with_mano_axes:
                        for payload in mesh_payloads:
                            _draw_frame_axes_numpy(
                                output_frame,
                                camera_matrix,
                                dist_coeffs,
                                payload["rvec"],
                                payload["tvec"],
                                0.05,
                                2,
                            )

            tactile_counts: Dict[str, int] = {}
            if render_tactile:
                for hand_side in requested_hands:
                    xyz = reshape_sensor_xyz(row[TACTILE_CLOUD_COLUMNS[hand_side]["xyz"]])
                    valid_mask = reshape_sensor_valid_mask(row[TACTILE_CLOUD_COLUMNS[hand_side]["valid_mask"]])
                    tactile_values = row[TACTILE_CLOUD_COLUMNS[hand_side]["source_tactile"]]
                    visible_count = _draw_spatial_tactile_points(
                        output_frame,
                        points_cam=xyz,
                        tactile_values=tactile_values,
                        valid_mask=valid_mask,
                        hand_side=hand_side,
                        point_radius=int(point_radius),
                    )
                    tactile_counts[hand_side] = int(visible_count)

            hand_status: List[str] = []
            if render_tactile:
                for hand_side in requested_hands:
                    hand_status.append(f"{hand_side[0].upper()}Tac={tactile_counts.get(hand_side, 0):03d}")
            if render_mano:
                active_mano_hands = {str(payload.get("hand_side", "")) for payload in mesh_payloads}
                for hand_side in requested_hands:
                    hand_status.append(f"{hand_side[0].upper()}Mano={int(hand_side in active_mano_hands)}")

            cv2.putText(
                output_frame,
                f"episode={resolved_episode_index:06d} frame={frame_index}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                output_frame,
                f"mode={resolved_render_mode} {' '.join(hand_status)}".strip(),
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
            writer.write(output_frame)
            rendered_frame_count += 1

            if verbose and ((frame_index + 1) % 200 == 0 or (frame_index + 1) == frame_limit):
                print(
                    f"processed {frame_index + 1}/{frame_limit} source frames, "
                    f"rendered {rendered_frame_count} frames"
                )
    finally:
        reader.release()
        writer.release()

    if verbose:
        print(f"reader_stats={reader.stats()}")
        print(f"rendered_frame_count_actual={rendered_frame_count}")
        print(f"Saved render video to {resolved_output_video}")
    return resolved_output_video


def main() -> None:
    args = _parse_args()
    try:
        resolved_output_arg, resolved_output_mode = _resolve_cli_output_arg(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    episode_indices = _parse_episode_indices(args.episode_indices)
    input_path = os.path.abspath(args.input_path)
    if len(episode_indices) > 1 and resolved_output_arg:
        output_candidate = Path(os.path.abspath(os.path.expanduser(resolved_output_arg)))
        output_is_dir = (
            resolved_output_mode == "dir"
            or ((output_candidate.exists() and output_candidate.is_dir()) or not output_candidate.suffix)
        )
        if not output_is_dir:
            raise SystemExit(
                "When rendering multiple episodes, the output target must be a directory. "
                "Use positional OUTPUT as a directory or pass --output_dir."
            )

    if Path(input_path).is_file():
        episode_indices = episode_indices[:1]

    for episode_index in episode_indices:
        render_episode_spatial_tactile_overlay(
            episode_path_arg=input_path,
            dataset_root=args.dataset_root,
            output_video=resolved_output_arg,
            output_mode=resolved_output_mode,
            hand=args.hand,
            point_radius=int(args.point_radius),
            render_mode=str(args.render_mode),
            with_mano=bool(args.with_mano),
            mano_alpha=float(args.mano_alpha),
            mano_device=str(args.mano_device),
            with_mano_axes=bool(args.with_mano_axes),
            max_frames=int(args.max_frames),
            frame_interval=int(args.frame_interval),
            camera_serial=args.camera_serial,
            camera_intrinsics_json=args.camera_intrinsics_json,
            episode_index=int(episode_index),
            verbose=True,
        )


if __name__ == "__main__":
    main()
