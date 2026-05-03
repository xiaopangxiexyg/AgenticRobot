import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from manotorch.manolayer import ManoLayer
from common.video_io import DEFAULT_VIDEO_CANDIDATES, PortableVideoWriter, RobustVideoReader, resolve_video_source


FIRST_PERSON_SERIAL = "151322070562"
BETA_DIM = 10
THETA_DIM = 45
EEF_DIM = 3
ROW_BATCH_SIZE = 128
PORTABLE_ROOT = Path(__file__).resolve().parent
PORTABLE_CAMERA_INTRINSICS_DIR = PORTABLE_ROOT / "assets" / "camera_intrinsics"
_MANO_LAYER_CACHE: Dict[tuple[str, str], ManoLayer] = {}
_MANO_FACE_CACHE: Dict[tuple[str, str], np.ndarray] = {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the saved LeRobot MANO state back onto the first-person video for one or more episodes.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python visualization/visualize_lerobot_mano.py /path/to/dataset_root\n"
            "  python visualization/visualize_lerobot_mano.py /path/to/dataset_root --episode 12\n"
            "  python visualization/visualize_lerobot_mano.py /path/to/dataset_root --episode 12 15 18\n"
            "  python visualization/visualize_lerobot_mano.py /path/to/dataset_root --episode 12,15,18\n"
            "  python visualization/visualize_lerobot_mano.py /path/to/data/episode_000012.parquet\n"
            "  python visualization/visualize_lerobot_mano.py /path/to/dataset_root --output_dir /tmp/mano_renders\n"
            "  python visualization/visualize_lerobot_mano.py /path/to/dataset_root -o /tmp/episode_000012_mano.mp4"
        ),
    )
    parser.add_argument(
        "input_path",
        type=str,
        metavar="INPUT",
        help="Input path: one episode parquet file, a dataset data/ directory, or a dataset root.",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default="",
        metavar="OUTPUT",
        help=(
            "Optional output video path or output directory.\n"
            "If omitted, save to a visualizations/ directory instead of next to the parquet."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        dest="dataset_root",
        type=str,
        default="",
        metavar="DATASET_ROOT",
        help="Advanced override for the dataset root containing data/meta/videos; usually not needed.",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--output-target",
        "--output_target",
        dest="output_target",
        type=str,
        default="",
        metavar="OUTPUT",
        help="Generic output target. Can be a video file path or an output directory.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default="",
        metavar="OUTPUT_DIR",
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
        metavar="OUTPUT_FILE",
        help="Explicit output video file path for a single rendered episode.",
    )
    parser.add_argument("--hand", type=str, default="both", choices=["left", "right", "both"], help="Which hand(s) to render")
    parser.add_argument("--alpha", type=float, default=0.35, help="Mesh alpha blending factor")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="MANO rendering device")
    parser.add_argument("--max_frames", type=int, default=0, help="Optional maximum number of frames to render")
    parser.add_argument(
        "--frame-interval",
        "--frame_interval",
        dest="frame_interval",
        type=int,
        default=1,
        help="Render every Nth frame while keeping the output FPS unchanged, so videos play faster when N > 1",
    )
    parser.add_argument("--camera_serial", type=str, default="", help="Override first-person camera serial if needed")
    parser.add_argument(
        "--camera-intrinsics-json",
        "--camera_intrinsics_json",
        dest="camera_intrinsics_json",
        type=str,
        default="",
        help=(
            "Optional camera intrinsics JSON path. "
            "If provided, use it instead of reading the original session meta.json."
        ),
    )
    parser.add_argument(
        "-e",
        "--episode",
        "--episode_index",
        dest="episode_indices",
        nargs="+",
        default=None,
        metavar="EPISODE",
        help=(
            "Episode index or indices to render when input_path points to a directory.\n"
            "Supports spaces and commas, for example: --episode 12 15 18 or --episode 12,15,18"
        ),
    )
    return parser.parse_args()


def _resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return requested_device


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

    for search_dir in search_dirs:
        requested = search_dir / f"episode_{int(episode_index):06d}.parquet"
        if requested.is_file():
            return str(requested)

    available_files: List[Path] = []
    for search_dir in search_dirs:
        available_files.extend(sorted(search_dir.glob("episode_*.parquet")))

    if not available_files:
        raise FileNotFoundError(f"Could not resolve any episode parquet from: {episode_path_arg}")

    available_names = ", ".join(path.name for path in available_files[:8])
    if len(available_files) > 8:
        available_names += ", ..."
    raise FileNotFoundError(
        f"Could not resolve episode_{int(episode_index):06d}.parquet from {episode_path_arg}. "
        f"Available examples: {available_names}"
    )


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


def _guess_dataset_root(episode_path: str, dataset_root_arg: str) -> Optional[str]:
    if dataset_root_arg:
        return os.path.abspath(dataset_root_arg)
    episode_parent = Path(os.path.abspath(episode_path)).parent
    if episode_parent.name == "data":
        return str(episode_parent.parent)
    return None


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


def _resolve_camera_serial(rows: Sequence[Dict[str, Any]], episode_meta: Dict[str, Any], override_serial: str) -> str:
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
        candidates.append(
            (
                "dataset_meta_camera_intrinsics_serial",
                str(root / "meta" / "camera_intrinsics" / f"{camera_serial}.json"),
            )
        )
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

    if override_path:
        raise FileNotFoundError(
            f"Could not load camera intrinsics for serial {camera_serial} from explicit JSON: {override_path}"
        )
    if meta_path and os.path.isfile(meta_path):
        raise KeyError(f"Camera serial {camera_serial} not found in {meta_path}")

    searched = [path for _, path in (search_paths + portable_paths) if path]
    search_summary = ", ".join(searched)
    raise FileNotFoundError(
        f"Could not resolve camera intrinsics for serial {camera_serial}. "
        f"Searched session meta {meta_path or '<none>'} and JSON fallbacks: {search_summary}"
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
        raise FileNotFoundError(
            "Could not resolve video.top_view from the dataset, and no session_dir was available for raw-video fallback."
        )
    camera_dir = os.path.join(session_dir, f"RealSense_{camera_serial}")
    video_path = resolve_video_source(camera_dir, candidates=DEFAULT_VIDEO_CANDIDATES)
    if not video_path:
        raise FileNotFoundError(f"Could not resolve a video source under {camera_dir}")
    return video_path


def _default_output_filename(episode_path: str, episode_index: int) -> str:
    path = Path(os.path.abspath(episode_path))
    if path.stem.startswith("episode_"):
        return f"{path.stem}_mano_overlay.mp4"
    return f"episode_{int(episode_index):06d}_mano_overlay.mp4"


def _resolve_output_video_path(
    output_arg: str,
    episode_path: str,
    dataset_root: Optional[str],
    episode_index: int,
    force_output_dir: bool = False,
    force_output_file: bool = False,
) -> str:
    default_filename = _default_output_filename(episode_path, episode_index)
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
) -> tuple[float, Optional[float]]:
    if int(frame_limit) <= 1:
        return float(fallback_fps), None
    table = pq.read_table(episode_path, columns=["timestamp_ms"]).slice(0, int(frame_limit))
    timestamps_ms = np.asarray(table.column("timestamp_ms").to_pylist(), dtype=np.float64)
    if timestamps_ms.size <= 1:
        return float(fallback_fps), None
    duration_s = float((timestamps_ms[-1] - timestamps_ms[0]) / 1000.0)
    if not np.isfinite(duration_s) or duration_s <= 1e-9:
        return float(fallback_fps), None
    derived_fps = float((timestamps_ms.size - 1) / duration_s)
    if not np.isfinite(derived_fps) or derived_fps <= 1e-9:
        return float(fallback_fps), None
    return derived_fps, duration_s


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


def _get_mano_layer(hand_side: str, device: str) -> ManoLayer:
    key = (hand_side, device)
    if key not in _MANO_LAYER_CACHE:
        layer = ManoLayer(
            rot_mode="axisang",
            side=hand_side,
            use_pca=False,
            flat_hand_mean=True,
            center_idx=0,
            device=device,
        )
        layer.eval()
        _MANO_LAYER_CACHE[key] = layer
    return _MANO_LAYER_CACHE[key]


def _get_mano_faces(hand_side: str, device: str) -> np.ndarray:
    key = (hand_side, device)
    if key not in _MANO_FACE_CACHE:
        layer = _get_mano_layer(hand_side, device)
        _MANO_FACE_CACHE[key] = layer.th_faces.detach().cpu().numpy().astype(np.int32, copy=True)
    return _MANO_FACE_CACHE[key]


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

    pose = np.concatenate([rotation, theta], axis=0).astype(np.float32, copy=False)
    layer = _get_mano_layer(hand_side, device)
    with torch.inference_mode():
        output = layer(
            torch.from_numpy(pose[None, :]).to(device=device, dtype=torch.float32),
            torch.from_numpy(beta[None, :]).to(device=device, dtype=torch.float32),
        )
    verts = output.verts[0].detach().cpu().numpy().astype(np.float32, copy=True)
    del output
    faces = _get_mano_faces(hand_side, device)
    verts_cam = np.ascontiguousarray(verts + position.reshape(1, 3), dtype=np.float32)
    return verts_cam, faces, position, rotation


def _project_vertices(
    verts_cam: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    points_cam = np.ascontiguousarray(np.asarray(verts_cam, dtype=np.float64).reshape(-1, 3))
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
    projected = _project_vertices(axis_points_cam, camera_matrix, dist_coeffs)
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
    try:
        if region.ndim == 3 and region.shape[2] >= 3:
            region[yy, xx, 0] = int(color[0])
            region[yy, xx, 1] = int(color[1])
            region[yy, xx, 2] = int(color[2])
        else:
            region[yy, xx] = int(color[0])
    except Exception as exc:
        debug_lines = [
            "Triangle fill failed",
            _array_debug_summary("image", image),
            _array_debug_summary("region", region),
            _array_debug_summary("mask", mask),
            _array_debug_summary("triangle_2d", np.asarray(triangle_2d)),
            f"color={tuple(int(v) for v in color)}",
            f"bbox=({x_min},{y_min})-({x_max},{y_max})",
            f"mask_points={int(yy.size)}",
        ]
        raise RuntimeError("\n".join(debug_lines)) from exc


def _array_debug_summary(name: str, value: Any) -> str:
    if not isinstance(value, np.ndarray):
        return f"{name}: type={type(value)!r}"
    summary = (
        f"{name}: type=np.ndarray dtype={value.dtype} shape={value.shape} "
        f"contiguous={bool(value.flags.c_contiguous)}"
    )
    if value.size <= 0:
        return summary + " size=0"
    finite = np.asarray(value, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size <= 0:
        return summary + " finite=0"
    return summary + f" finite_min={float(finite.min()):.6g} finite_max={float(finite.max()):.6g}"


def _safe_int_point(xy: Any) -> Optional[tuple[int, int]]:
    point = np.asarray(xy, dtype=np.float64).reshape(-1)
    if point.size < 2 or not np.all(np.isfinite(point[:2])):
        return None
    return (int(round(float(point[0]))), int(round(float(point[1]))))


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


def render_episode_mano_overlay(
    episode_path_arg: str,
    *,
    dataset_root: str = "",
    output_video: str = "",
    output_mode: str = "auto",
    hand: str = "both",
    alpha: float = 0.35,
    device: str = "auto",
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
    resolved_dataset_root = _guess_dataset_root(episode_path, dataset_root)
    episode_meta = _load_episode_meta_row(resolved_dataset_root, resolved_episode_index)
    session_dir = str(first_row.get("session_dir", "")).strip() or str(episode_meta.get("session_dir", "")).strip()

    resolved_camera_serial = _resolve_camera_serial([first_row], episode_meta, camera_serial)
    camera_matrix, dist_coeffs, intrinsics_source = _load_camera_intrinsics(
        session_dir=session_dir,
        camera_serial=resolved_camera_serial,
        dataset_root=resolved_dataset_root,
        override_path=camera_intrinsics_json,
    )
    video_path = _resolve_episode_video_path([first_row], resolved_dataset_root, session_dir, resolved_camera_serial)
    resolved_output_video = _resolve_output_video_path(
        output_arg=output_video,
        episode_path=episode_path,
        dataset_root=resolved_dataset_root,
        episode_index=resolved_episode_index,
        force_output_dir=(str(output_mode) == "dir"),
        force_output_file=(str(output_mode) == "file"),
    )
    resolved_device = _resolve_device(device)

    output_dir = os.path.dirname(resolved_output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    reader = RobustVideoReader(video_path, expected_frame_count=row_stream.row_count)
    frame_limit = int(row_stream.row_count)
    if int(max_frames) > 0:
        frame_limit = min(frame_limit, int(max_frames))
    frame_interval = max(1, int(frame_interval))
    expected_rendered_frames = (frame_limit + frame_interval - 1) // frame_interval if frame_limit > 0 else 0
    if verbose:
        print(f"frame_interval={frame_interval}")
        print(f"rendered_frame_count_expected={expected_rendered_frames}")

    source_video_fps = float(reader.fps if reader.fps > 0 else 30.0)
    output_fps, row_timeline_duration_s = _compute_output_fps_from_row_timestamps(
        episode_path=episode_path,
        frame_limit=frame_limit,
        fallback_fps=source_video_fps,
    )
    if verbose:
        print(f"source_video_fps={source_video_fps:.6f}")
        print(f"output_video_fps={output_fps:.6f}")
    if row_timeline_duration_s is not None and verbose:
        render_duration_s = float(frame_limit / max(output_fps, 1e-9))
        source_duration_s = float(frame_limit / max(source_video_fps, 1e-9))
        print(f"row_timeline_duration_s={row_timeline_duration_s:.6f}")
        print(f"source_frame_span_duration_s={source_duration_s:.6f}")
        print(f"render_output_duration_s={render_duration_s:.6f}")
        if abs(output_fps - source_video_fps) > 1e-3:
            print(
                "timing_warning=timestamp-derived fps differs from source video fps; "
                "using timestamp-derived fps for output to reduce cumulative drift"
            )

    writer = PortableVideoWriter(
        resolved_output_video,
        float(output_fps),
        (int(reader.width), int(reader.height)),
    )
    if not writer.isOpened():
        reader.release()
        raise RuntimeError(f"Failed to open output video writer: {resolved_output_video}")

    if verbose:
        print(f"episode_path={episode_path}")
        print(f"session_dir={session_dir}")
        print(f"camera_serial={resolved_camera_serial}")
        print(f"camera_intrinsics_source={intrinsics_source}")
        print(f"video_path={video_path}")
        print(f"output_video={resolved_output_video}")
        print(f"writer_backend={writer.backend}")
        print(f"device={resolved_device}")

    requested_hands = ["left", "right"] if hand == "both" else [hand]
    hand_colors = {
        "left": (120, 255, 140),
        "right": (80, 180, 255),
    }
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

            mesh_payloads: List[Dict[str, Any]] = []
            for hand_side in requested_hands:
                mano_mesh = _mano_vertices_in_camera_frame(row, hand_side, resolved_device)
                if mano_mesh is None:
                    continue
                verts_cam, faces, position, rotation = mano_mesh
                try:
                    projected = _project_vertices(verts_cam, camera_matrix, dist_coeffs)
                except Exception as exc:
                    debug_lines = [
                        f"Projection failed at episode={resolved_episode_index} frame={frame_index} hand={hand_side}",
                        f"numpy_version={np.__version__}",
                        f"opencv_version={cv2.__version__}",
                        _array_debug_summary("verts_cam", verts_cam),
                        _array_debug_summary("faces", faces),
                        _array_debug_summary("position", position),
                        _array_debug_summary("rotation", rotation),
                        _array_debug_summary("camera_matrix", camera_matrix),
                        _array_debug_summary("dist_coeffs", dist_coeffs),
                    ]
                    raise RuntimeError("\n".join(debug_lines)) from exc
                mesh_payloads.append(
                    {
                        "verts_cam": verts_cam,
                        "faces": faces,
                        "projected": projected,
                        "projected_wrist": projected[0],
                        "color": hand_colors[hand_side],
                        "label": f"{hand_side.upper()} MANO",
                        "rvec": rotation,
                        "tvec": position,
                    }
                )

            output_frame = frame
            if mesh_payloads:
                output_frame = _render_mesh_overlay(frame, mesh_payloads, alpha=float(alpha))
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
            writer.write(output_frame)
            rendered_frame_count += 1

            if resolved_device == "cuda" and rendered_frame_count > 0 and rendered_frame_count % 128 == 0:
                torch.cuda.empty_cache()
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
        print(f"Saved MANO overlay video to {resolved_output_video}")
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
        render_episode_mano_overlay(
            episode_path_arg=input_path,
            dataset_root=args.dataset_root,
            output_video=resolved_output_arg,
            output_mode=resolved_output_mode,
            hand=args.hand,
            alpha=float(args.alpha),
            device=args.device,
            max_frames=int(args.max_frames),
            frame_interval=max(1, int(args.frame_interval)),
            camera_serial=args.camera_serial,
            camera_intrinsics_json=args.camera_intrinsics_json,
            episode_index=int(episode_index),
            verbose=True,
        )


if __name__ == "__main__":
    main()
