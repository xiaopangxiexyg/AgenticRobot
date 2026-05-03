from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from Datasets.video_utils import load_video_frames_ffmpeg_uint8
from Models.med_support import normalize_med_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
PORTABLE_VIS_ROOT = PROJECT_ROOT / "TestScripts" / "portable_vis"
PORTABLE_CAMERA_INTRINSICS_DIR = PORTABLE_VIS_ROOT / "assets" / "camera_intrinsics"
DEFAULT_CAMERA_SERIAL = "151322070562"
DEFAULT_VIDEO_CANDIDATES = (
    "color_aligned.mp4",
    "rgb.mp4",
    "color.mp4",
    "rgb.mkv",
)


@dataclass
class ProjectionRenderContext:
    frames: Any
    frame_indices: List[int]
    frame_offsets: List[int]
    camera_matrix: Any
    dist_coeffs: Any
    video_path: str
    fps: float


def resolve_checkpoint_path(checkpoint_ref: str | Path, preferred_kind: str = "auto") -> tuple[Path, str]:
    if preferred_kind not in {"auto", "best", "last"}:
        raise ValueError(f"Unsupported checkpoint kind: {preferred_kind}")

    raw_ref = Path(str(checkpoint_ref)).expanduser()
    if raw_ref.exists():
        resolved = raw_ref.resolve()
        if resolved.is_file():
            return resolved, "file"
        return _resolve_checkpoint_from_dir(resolved, preferred_kind=preferred_kind)

    run_matches = [path for path in OUTPUTS_DIR.rglob(raw_ref.name) if path.is_dir()]
    if len(run_matches) == 1:
        return _resolve_checkpoint_from_dir(run_matches[0].resolve(), preferred_kind=preferred_kind)
    if len(run_matches) > 1:
        match_list = ", ".join(str(path) for path in run_matches[:5])
        raise FileNotFoundError(
            f"Checkpoint reference {checkpoint_ref!r} matches multiple run directories under {OUTPUTS_DIR}: {match_list}"
        )

    file_matches = [path for path in OUTPUTS_DIR.rglob(raw_ref.name) if path.is_file()]
    if len(file_matches) == 1:
        return file_matches[0].resolve(), "searched_file"
    if len(file_matches) > 1:
        match_list = ", ".join(str(path) for path in file_matches[:5])
        raise FileNotFoundError(
            f"Checkpoint reference {checkpoint_ref!r} matches multiple checkpoint files under {OUTPUTS_DIR}: {match_list}"
        )

    raise FileNotFoundError(f"Could not resolve checkpoint from {checkpoint_ref!r}")


def _resolve_checkpoint_from_dir(run_dir: Path, preferred_kind: str) -> tuple[Path, str]:
    order = ["best", "last"] if preferred_kind == "auto" else [preferred_kind, "last" if preferred_kind == "best" else "best"]
    for kind in order:
        candidate = run_dir / f"{kind}.pt"
        if candidate.is_file():
            return candidate.resolve(), kind
    raise FileNotFoundError(f"Could not find best.pt or last.pt under {run_dir}")


def format_split_metrics(split_name: str, split_result: Dict[str, Any]) -> List[str]:
    lines = [f"{split_name}: samples={split_result['samples']} batches={split_result['batches']}"]
    metrics = split_result.get("metrics", {})
    for key in sorted(metrics.keys()):
        metric = metrics[key]
        lines.append(
            f"  {key}: rmse={metric['rmse']:.6f} mae={metric['mae']:.6f} mse={metric['mse']:.6f}"
        )
    return lines


def maybe_render_point_cloud_projection_artifacts(
    *,
    sample_payload: Dict[str, Any],
    output_dir: Path,
    file_stem: str,
    max_points: int,
    point_radius: int,
    frame_height: int,
    camera_serial: str = "",
    camera_intrinsics_json: str = "",
    max_steps: int | None = None,
) -> Dict[str, Any]:
    metadata = dict(sample_payload.get("metadata", {}))
    predictions = sample_payload.get("predictions", {})
    targets = sample_payload.get("targets", {})
    if "med" not in predictions or "med" not in targets:
        return {"paths": [], "notes": ["sample does not contain matching med prediction/target tensors"]}

    med_spec = sample_payload.get("med_spec")
    if med_spec is None:
        return {"paths": [], "notes": ["missing med_spec; cannot infer point-cloud layout"]}

    normalized_med_spec = normalize_med_spec(med_spec)
    if str(normalized_med_spec.get("kind", "vector")).lower() != "point_cloud":
        return {"paths": [], "notes": ["med_spec.kind is not point_cloud; skipping video projection"]}

    try:
        prediction = _reshape_point_cloud_sequence(predictions["med"], normalized_med_spec, metadata)
        target = _reshape_point_cloud_sequence(targets["med"], normalized_med_spec, metadata)
    except Exception as error:
        return {"paths": [], "notes": [f"failed to reshape point-cloud tensors: {error}"]}

    time_steps = min(int(prediction.shape[0]), int(target.shape[0]))
    if max_steps is not None and int(max_steps) > 0:
        time_steps = min(time_steps, int(max_steps))
    if time_steps <= 0:
        return {"paths": [], "notes": ["point-cloud tensors contain no renderable timesteps"]}

    try:
        context = _build_projection_context(
            metadata=metadata,
            time_steps=time_steps,
            camera_serial=camera_serial,
            camera_intrinsics_json=camera_intrinsics_json,
        )
    except Exception as error:
        return {"paths": [], "notes": [f"failed to build video projection context: {error}"]}

    prediction = prediction[:time_steps]
    target = target[:time_steps]
    rendered_frames = _build_projection_frames(
        prediction=prediction,
        target=target,
        metadata=metadata,
        context=context,
        max_points=int(max_points),
        point_radius=int(point_radius),
        frame_height=int(frame_height),
    )
    if not rendered_frames:
        return {"paths": [], "notes": ["no projection frames were rendered"]}

    output_dir.mkdir(parents=True, exist_ok=True)
    projection_image = output_dir / f"{file_stem}_projection.png"
    _save_contact_sheet(rendered_frames, projection_image)
    paths = [str(projection_image)]

    if len(rendered_frames) > 1:
        projection_video = output_dir / f"{file_stem}_projection.mp4"
        _save_video(rendered_frames, projection_video, fps=max(1.0, min(10.0, float(context.fps))))
        paths.append(str(projection_video))

    return {"paths": paths, "notes": []}


def _reshape_point_cloud_sequence(value: Any, med_spec: Dict[str, Any], metadata: Dict[str, Any]) -> torch.Tensor:
    tensor = value.detach().to(dtype=torch.float32).cpu() if torch.is_tensor(value) else torch.tensor(value, dtype=torch.float32)
    shape = [int(dim) for dim in med_spec.get("shape", [])]
    future_steps = int(metadata.get("future_tactile_steps", 0) or 0)

    if tensor.dim() == 3:
        return tensor
    if tensor.dim() == 2:
        if len(shape) == 3 and tensor.numel() == math.prod(shape):
            return tensor.reshape(shape[0], shape[1], shape[2])
        if future_steps > 0 and tensor.shape[0] % future_steps == 0:
            point_count = int(tensor.shape[0] // future_steps)
            return tensor.reshape(future_steps, point_count, int(tensor.shape[-1]))
        return tensor.unsqueeze(0)
    if tensor.dim() == 1:
        if len(shape) == 3 and tensor.numel() == math.prod(shape):
            return tensor.reshape(shape[0], shape[1], shape[2])
        if future_steps > 0 and tensor.numel() % future_steps == 0:
            point_dim = int(tensor.numel() // future_steps)
            return tensor.reshape(future_steps, point_dim, 1)
    raise ValueError(f"Unsupported point-cloud tensor shape: {tuple(tensor.shape)}")


def _build_projection_context(
    *,
    metadata: Dict[str, Any],
    time_steps: int,
    camera_serial: str,
    camera_intrinsics_json: str,
) -> ProjectionRenderContext:
    episode_parquet_path = _resolve_episode_parquet_path(metadata)
    video_key = str(metadata.get("video_key", "video.top_view"))
    frame_offsets = _resolve_future_offsets(metadata, time_steps)
    current_frame_index = int(metadata.get("frame_index", 0))
    row_indices = [current_frame_index + offset for offset in frame_offsets]
    future_rows = _read_episode_rows(
        episode_parquet_path=episode_parquet_path,
        row_indices=row_indices,
        columns=[video_key, f"{video_key}_frame_index", "session_dir"],
    )

    dataset_root = str(metadata.get("dataset_root", "") or "")
    episode_index = int(metadata.get("episode_index", 0))
    episode_meta = _load_episode_meta_row(dataset_root=dataset_root, episode_index=episode_index)
    session_dir = str(metadata.get("session_dir", "") or future_rows[0].get("session_dir", "") or episode_meta.get("session_dir", "")).strip()
    resolved_camera_serial = str(camera_serial).strip() or str(episode_meta.get("first_person_camera_serial", "")).strip() or DEFAULT_CAMERA_SERIAL
    video_path = _resolve_video_path(
        metadata=metadata,
        future_rows=future_rows,
        dataset_root=dataset_root,
        session_dir=session_dir,
        camera_serial=resolved_camera_serial,
        video_key=video_key,
    )
    width, height, fps = _probe_video(video_path)
    frame_indices = [int(row[f"{video_key}_frame_index"]) for row in future_rows]
    frames = load_video_frames_ffmpeg_uint8(
        video_path=video_path,
        frame_indices=frame_indices,
        resize_height=int(height),
        resize_width=int(width),
        color_order="bgr",
    )
    camera_matrix, dist_coeffs, _ = _load_camera_intrinsics(
        session_dir=session_dir,
        camera_serial=resolved_camera_serial,
        dataset_root=dataset_root,
        override_path=camera_intrinsics_json,
    )
    return ProjectionRenderContext(
        frames=frames,
        frame_indices=frame_indices,
        frame_offsets=frame_offsets,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        video_path=video_path,
        fps=float(fps),
    )


def _resolve_future_offsets(metadata: Dict[str, Any], time_steps: int) -> List[int]:
    start = int(metadata.get("med_future_start", 1))
    end = int(metadata.get("med_future_end", start + max(0, int(time_steps) - 1)))
    stride = max(1, int(metadata.get("med_stride", 1)))
    offsets = list(range(start, end + 1, stride))
    if not offsets:
        offsets = list(range(1, int(time_steps) + 1))
    return offsets[: int(time_steps)]


def _resolve_episode_parquet_path(metadata: Dict[str, Any]) -> Path:
    explicit_path = Path(str(metadata.get("episode_parquet_path", "") or "")).expanduser()
    if explicit_path.is_file():
        return explicit_path.resolve()

    dataset_root = Path(str(metadata.get("dataset_root", "") or "")).expanduser()
    episode_index = int(metadata.get("episode_index", 0))
    if dataset_root.exists():
        default_path = dataset_root / "data" / f"episode_{episode_index:06d}.parquet"
        if default_path.is_file():
            return default_path.resolve()

        episode_meta = _load_episode_meta_row(str(dataset_root), episode_index)
        parquet_relpath = str(episode_meta.get("parquet_path", "")).strip()
        if parquet_relpath:
            candidate = dataset_root / parquet_relpath
            if candidate.is_file():
                return candidate.resolve()

    raise FileNotFoundError("Could not resolve episode parquet path from sample metadata.")


def _read_episode_rows(episode_parquet_path: Path, row_indices: Sequence[int], columns: Sequence[str]) -> List[Dict[str, Any]]:
    pq = _import_pyarrow_parquet()
    if pq is None:
        raise RuntimeError("pyarrow is not installed; cannot read episode parquet for projection rendering.")
    if not row_indices:
        return []
    start = min(int(index) for index in row_indices)
    stop = max(int(index) for index in row_indices)
    table = pq.read_table(str(episode_parquet_path), columns=list(columns)).slice(start, stop - start + 1)
    contiguous_rows = table.to_pylist()
    lookup = {start + offset: dict(row) for offset, row in enumerate(contiguous_rows)}
    missing = [index for index in row_indices if int(index) not in lookup]
    if missing:
        raise IndexError(f"Episode parquet {episode_parquet_path} is missing requested row indices: {missing}")
    return [lookup[int(index)] for index in row_indices]


def _load_episode_meta_row(dataset_root: str, episode_index: int) -> Dict[str, Any]:
    if not dataset_root:
        return {}
    pq = _import_pyarrow_parquet()
    if pq is None:
        return {}
    episodes_meta_path = Path(dataset_root).expanduser() / "meta" / "episodes.parquet"
    if not episodes_meta_path.is_file():
        return {}
    parquet_file = pq.ParquetFile(str(episodes_meta_path))
    for batch in parquet_file.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            if int(row.get("episode_index", -1)) == int(episode_index):
                return dict(row)
    return {}


def _resolve_video_path(
    *,
    metadata: Dict[str, Any],
    future_rows: Sequence[Dict[str, Any]],
    dataset_root: str,
    session_dir: str,
    camera_serial: str,
    video_key: str,
) -> str:
    explicit_video = Path(str(metadata.get("video_path", "") or "")).expanduser()
    if explicit_video.is_file():
        return str(explicit_video.resolve())

    if dataset_root and future_rows:
        relpath = str(future_rows[0].get(video_key, "")).strip()
        if relpath:
            candidate = Path(dataset_root).expanduser() / relpath
            if candidate.is_file():
                return str(candidate.resolve())

    if session_dir:
        camera_dir = Path(session_dir).expanduser() / f"RealSense_{camera_serial}"
        for candidate_name in DEFAULT_VIDEO_CANDIDATES:
            candidate = camera_dir / candidate_name
            if candidate.is_file():
                return str(candidate.resolve())

    raise FileNotFoundError("Could not resolve source video for point-cloud projection rendering.")


def _probe_video(video_path: str) -> tuple[int, int, float]:
    cv2 = _import_cv2()
    if cv2 is None:
        raise RuntimeError("opencv-python is not installed; cannot probe source video.")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions for {video_path}: {width}x{height}")
    if fps <= 1e-6:
        fps = 30.0
    return width, height, fps


def _camera_intrinsics_from_dict(intrinsics: Dict[str, Any]) -> tuple[Any, Any]:
    np = _import_numpy()
    return (
        np.array(
            [
                [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        np.asarray(intrinsics.get("coeffs", []), dtype=np.float32),
    )


def _load_intrinsics_from_session_meta(meta_path: str, camera_serial: str) -> tuple[Any, Any] | None:
    if not meta_path or not Path(meta_path).is_file():
        return None
    with open(meta_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    for camera in payload.get("cameras", []):
        if str(camera.get("serial", "")).strip() != str(camera_serial):
            continue
        intrinsics = camera.get("streams", {}).get("color", {}).get("intrinsics", {})
        required = {"fx", "fy", "cx", "cy"}
        if required.issubset(intrinsics):
            return _camera_intrinsics_from_dict(intrinsics)
    return None


def _load_intrinsics_from_json_payload(payload: Dict[str, Any], camera_serial: str) -> tuple[Any, Any] | None:
    payload_serial = str(payload.get("camera_serial", "")).strip()
    if payload_serial and payload_serial != str(camera_serial):
        return None

    intrinsics = payload.get("intrinsics")
    if isinstance(intrinsics, dict) and {"fx", "fy", "cx", "cy"}.issubset(intrinsics):
        return _camera_intrinsics_from_dict(intrinsics)

    serial_key = str(camera_serial)
    nested = payload.get(serial_key)
    if isinstance(nested, dict):
        nested_intrinsics = nested.get("intrinsics", nested)
        if isinstance(nested_intrinsics, dict) and {"fx", "fy", "cx", "cy"}.issubset(nested_intrinsics):
            return _camera_intrinsics_from_dict(nested_intrinsics)

    for camera in payload.get("cameras", []):
        if str(camera.get("serial", "")).strip() != serial_key:
            continue
        nested_intrinsics = camera.get("streams", {}).get("color", {}).get("intrinsics", {})
        if {"fx", "fy", "cx", "cy"}.issubset(nested_intrinsics):
            return _camera_intrinsics_from_dict(nested_intrinsics)

    if {"fx", "fy", "cx", "cy"}.issubset(payload):
        return _camera_intrinsics_from_dict(payload)
    return None


def _load_intrinsics_from_json_file(json_path: str, camera_serial: str) -> tuple[Any, Any] | None:
    if not json_path or not Path(json_path).is_file():
        return None
    with open(json_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Camera intrinsics JSON must contain an object: {json_path}")
    return _load_intrinsics_from_json_payload(payload, camera_serial)


def _candidate_intrinsics_paths(dataset_root: str, camera_serial: str, override_path: str) -> List[str]:
    candidates: List[str] = []
    if override_path:
        candidates.append(str(Path(override_path).expanduser()))
    if dataset_root:
        root = Path(dataset_root).expanduser()
        candidates.append(str(root / "meta" / "camera_intrinsics" / f"{camera_serial}.json"))
        candidates.append(str(root / "meta" / "camera_intrinsics.json"))
    candidates.append(str(PORTABLE_CAMERA_INTRINSICS_DIR / f"{camera_serial}.json"))
    candidates.append(str(PORTABLE_CAMERA_INTRINSICS_DIR / "default.json"))
    return candidates


def _load_camera_intrinsics(
    *,
    session_dir: str,
    camera_serial: str,
    dataset_root: str,
    override_path: str,
) -> tuple[Any, Any, str]:
    for candidate_path in _candidate_intrinsics_paths(dataset_root, camera_serial, override_path):
        intrinsics = _load_intrinsics_from_json_file(candidate_path, camera_serial)
        if intrinsics is not None:
            return intrinsics[0], intrinsics[1], candidate_path

    session_meta_path = str(Path(session_dir).expanduser() / "meta.json") if session_dir else ""
    intrinsics = _load_intrinsics_from_session_meta(session_meta_path, camera_serial)
    if intrinsics is not None:
        return intrinsics[0], intrinsics[1], session_meta_path

    raise FileNotFoundError(f"Could not resolve camera intrinsics for camera serial {camera_serial}")


def _build_projection_frames(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    metadata: Dict[str, Any],
    context: ProjectionRenderContext,
    max_points: int,
    point_radius: int,
    frame_height: int,
) -> List[Any]:
    cv2 = _import_cv2()
    np = _import_numpy()
    rendered: List[Any] = []
    for step_index in range(min(len(context.frame_offsets), int(prediction.shape[0]), int(target.shape[0]))):
        base_frame = np.asarray(context.frames[step_index]).copy()
        target_frame = base_frame.copy()
        prediction_frame = base_frame.copy()
        target_visible = _draw_projected_points(
            frame=target_frame,
            points=target[step_index],
            metadata=metadata,
            camera_matrix=context.camera_matrix,
            dist_coeffs=context.dist_coeffs,
            point_radius=point_radius,
            max_points=max_points,
            outline_color=(120, 255, 140),
        )
        prediction_visible = _draw_projected_points(
            frame=prediction_frame,
            points=prediction[step_index],
            metadata=metadata,
            camera_matrix=context.camera_matrix,
            dist_coeffs=context.dist_coeffs,
            point_radius=point_radius,
            max_points=max_points,
            outline_color=(80, 180, 255),
        )
        offset = int(context.frame_offsets[step_index])
        source_frame_index = int(context.frame_indices[step_index])
        original_panel = _caption_frame(
            base_frame,
            title="Original Frame",
            subtitle=f"future=+{offset} video={source_frame_index}",
        )
        target_panel = _caption_frame(
            target_frame,
            title="Target Projection",
            subtitle=f"visible={target_visible}",
        )
        prediction_panel = _caption_frame(
            prediction_frame,
            title="Predicted Projection",
            subtitle=f"visible={prediction_visible}",
        )
        combined = np.concatenate([original_panel, target_panel, prediction_panel], axis=1)
        rendered.append(_resize_to_height(combined, frame_height))
    return rendered


def _caption_frame(frame: Any, title: str, subtitle: str = "") -> Any:
    cv2 = _import_cv2()
    np = _import_numpy()
    image = np.asarray(frame)
    caption_height = 58
    canvas = np.full((image.shape[0] + caption_height, image.shape[1], 3), 18, dtype=np.uint8)
    canvas[caption_height:] = image
    cv2.putText(canvas, title, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, lineType=cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (14, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 215, 220), 1, lineType=cv2.LINE_AA)
    return canvas


def _resize_to_height(frame: Any, target_height: int) -> Any:
    cv2 = _import_cv2()
    image = frame
    if int(target_height) <= 0 or int(image.shape[0]) <= int(target_height):
        return image
    scale = float(target_height) / float(image.shape[0])
    target_width = max(1, int(round(float(image.shape[1]) * scale)))
    return cv2.resize(image, (target_width, int(target_height)), interpolation=cv2.INTER_AREA)


def _point_values(points: Any) -> Any:
    if int(points.shape[-1]) <= 3:
        return torch.zeros(points.shape[0], dtype=torch.float32)
    extra = points[:, 3:]
    if int(extra.shape[-1]) == 1:
        return extra[:, 0]
    return extra.mean(dim=-1)


def _infer_valid_mask(points: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    valid = torch.isfinite(points[:, 0]) & torch.isfinite(points[:, 1]) & torch.isfinite(points[:, 2])
    valid &= points[:, 2] > 1e-6
    selected_fields = metadata.get("med_selected_fields", {}) or {}
    if bool(selected_fields.get("include_valid_feature", False)) and int(points.shape[-1]) >= 4:
        valid &= points[:, -1] > 0.5
    return valid


def _draw_projected_points(
    *,
    frame: Any,
    points: torch.Tensor,
    metadata: Dict[str, Any],
    camera_matrix: Any,
    dist_coeffs: Any,
    point_radius: int,
    max_points: int,
    outline_color: tuple[int, int, int],
) -> int:
    cv2 = _import_cv2()
    np = _import_numpy()
    clipped = points.detach().to(dtype=torch.float32).cpu()
    if int(max_points) > 0:
        clipped = clipped[: int(max_points)]
    valid = _infer_valid_mask(clipped, metadata)
    if not bool(valid.any()):
        return 0

    points_xyz = clipped[:, :3].numpy()
    projected = _project_points(points_xyz, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    valid &= torch.from_numpy(np.isfinite(projected[:, 0]) & np.isfinite(projected[:, 1]))
    if not bool(valid.any()):
        return 0

    visible_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
    sorted_indices = visible_indices[torch.argsort(clipped[visible_indices, 2], descending=True)]
    value_colors = cv2.applyColorMap(
        np.clip(np.round(_point_values(clipped).numpy() * 255.0), 0.0, 255.0).astype(np.uint8).reshape(-1, 1),
        cv2.COLORMAP_TURBO,
    ).reshape(-1, 3)
    for tensor_index in sorted_indices.tolist():
        xy = projected[int(tensor_index)]
        center = (int(round(float(xy[0]))), int(round(float(xy[1]))))
        fill_color = tuple(int(value) for value in value_colors[int(tensor_index)].tolist())
        cv2.circle(frame, center, int(point_radius), fill_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, center, int(max(1, point_radius)), outline_color, 1, lineType=cv2.LINE_AA)
    return int(sorted_indices.numel())


def _project_points(points_cam: Any, camera_matrix: Any, dist_coeffs: Any) -> Any:
    np = _import_numpy()
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


def _save_contact_sheet(frames: Sequence[Any], output_path: Path) -> None:
    cv2 = _import_cv2()
    np = _import_numpy()
    if not frames:
        raise ValueError("No frames available for contact sheet rendering.")
    frame_height = int(frames[0].shape[0])
    frame_width = int(frames[0].shape[1])
    columns = 1 if len(frames) <= 2 else 2
    rows = int(math.ceil(len(frames) / float(columns)))
    gap = 14
    canvas = np.full(
        (
            rows * frame_height + max(0, rows - 1) * gap,
            columns * frame_width + max(0, columns - 1) * gap,
            3,
        ),
        20,
        dtype=np.uint8,
    )
    for index, frame in enumerate(frames):
        row = index // columns
        column = index % columns
        y0 = row * (frame_height + gap)
        x0 = column * (frame_width + gap)
        canvas[y0 : y0 + frame_height, x0 : x0 + frame_width] = frame
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to write contact sheet image: {output_path}")


def _save_video(frames: Sequence[Any], output_path: Path, fps: float) -> None:
    cv2 = _import_cv2()
    if not frames:
        raise ValueError("No frames available for video rendering.")
    frame_height = int(frames[0].shape[0])
    frame_width = int(frames[0].shape[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1.0, fps)),
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video writer: {output_path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _import_numpy():
    import numpy as np

    return np


def _import_cv2():
    try:
        import cv2
    except Exception:
        return None
    return cv2


def _import_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except Exception:
        return None
    return pq
