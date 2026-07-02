from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .dataset import LoadedReplayEpisode


def _require_cv2() -> Any:
    try:
        import cv2  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Replay media rendering requires opencv-python.") from exc
    return cv2


def _resolve_dataset_path(root: Path, path_text: Any) -> Path | None:
    if path_text is None or not str(path_text):
        return None
    path = Path(str(path_text)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _finite_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if array.size <= 0 or not np.all(np.isfinite(array)):
        return None
    return array


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _encode_jpeg(bgr: np.ndarray, *, quality: int = 86) -> bytes:
    cv2 = _require_cv2()
    image = np.asarray(bgr, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("OpenCV failed to encode replay JPEG.")
    return bytes(encoded)


def placeholder_jpeg(title: str, lines: list[str] | None = None, *, width: int = 640, height: int = 360) -> bytes:
    cv2 = _require_cv2()
    image = np.full((max(64, int(height)), max(64, int(width)), 3), 238, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, image.shape[0] - 1), (120, 128, 136), 1)
    cv2.putText(
        image,
        str(title)[:64],
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (36, 42, 48),
        2,
        cv2.LINE_AA,
    )
    y = 82
    for line in lines or []:
        cv2.putText(
            image,
            str(line)[:80],
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (78, 86, 94),
            1,
            cv2.LINE_AA,
        )
        y += 28
        if y > image.shape[0] - 20:
            break
    return _encode_jpeg(image, quality=82)


def _resize_to_fit(image: np.ndarray, *, max_width: int, max_height: int) -> np.ndarray:
    cv2 = _require_cv2()
    height, width = image.shape[:2]
    scale = min(float(max_width) / max(1, width), float(max_height) / max(1, height), 1.0)
    if scale >= 0.999:
        return image
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _hstack_equal_height(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("images must not be empty")
    cv2 = _require_cv2()
    max_h = max(int(image.shape[0]) for image in images)
    resized: list[np.ndarray] = []
    for image in images:
        if image.shape[0] == max_h:
            resized.append(image)
            continue
        new_w = max(1, int(round(float(image.shape[1]) * max_h / max(1, image.shape[0]))))
        resized.append(cv2.resize(image, (new_w, max_h), interpolation=cv2.INTER_AREA))
    return np.concatenate(resized, axis=1)


def _video_frame_index(row: Mapping[str, Any], camera_key: str, fallback: int) -> int | None:
    value = row.get(f"observation.images.{camera_key}.video_frame_index")
    if value is None:
        return int(fallback)
    index = _int_or_none(value)
    if index is None or index < 0:
        return None
    return index


class CameraVideoTrack:
    def __init__(self, *, dataset_root: Path, camera_key: str, path_text: str):
        self.dataset_root = Path(dataset_root)
        self.camera_key = str(camera_key)
        self.path = _resolve_dataset_path(self.dataset_root, path_text)
        self._cv2: Any | None = None
        self._capture: Any | None = None
        self._last_frame_index: int | None = None
        self._last_jpeg: bytes | None = None

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._last_frame_index = None
        self._last_jpeg = None

    def _capture_or_none(self) -> Any | None:
        if self.path is None or not self.path.exists():
            return None
        if self._capture is not None:
            return self._capture
        cv2 = _require_cv2()
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            return None
        self._cv2 = cv2
        self._capture = capture
        return capture

    def frame_jpeg(self, row: Mapping[str, Any], *, replay_index: int) -> bytes:
        frame_index = _video_frame_index(row, self.camera_key, replay_index)
        if frame_index is None:
            return placeholder_jpeg(self.camera_key, ["invalid video frame index"])
        if self._last_frame_index == int(frame_index) and self._last_jpeg is not None:
            return self._last_jpeg
        capture = self._capture_or_none()
        if capture is None:
            path_text = "missing video" if self.path is None else str(self.path)
            return placeholder_jpeg(self.camera_key, [path_text])
        assert self._cv2 is not None
        if self._last_frame_index is None or int(frame_index) != int(self._last_frame_index) + 1:
            capture.set(self._cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            return placeholder_jpeg(self.camera_key, [f"frame {frame_index} unavailable"])
        frame = _resize_to_fit(frame, max_width=1280, max_height=720)
        self._last_frame_index = int(frame_index)
        self._last_jpeg = _encode_jpeg(frame, quality=86)
        return self._last_jpeg


@dataclass
class TactileFrame:
    side: str
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    vector: np.ndarray | None = None
    tactile_frame_index: int | None = None
    sample_frame_index: int | None = None
    source: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return bool(self.arrays) or (self.vector is not None and self.vector.size > 0)


class ExternalTactileStore:
    def __init__(self, *, dataset_root: Path, manifest_path: Path):
        self.dataset_root = Path(dataset_root)
        self.manifest_path = Path(manifest_path)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Tactile manifest is not a JSON object: {self.manifest_path}")
        self.manifest = dict(payload)
        self.side = str(self.manifest.get("side") or "")
        self.frame_count = max(0, int(self.manifest.get("frame_count") or 0))
        self.sensor_keys = tuple(str(key) for key in self.manifest.get("sensor_keys", ()) or ())
        self.frames = [
            dict(frame)
            for frame in self.manifest.get("frames", []) or []
            if isinstance(frame, Mapping)
        ]
        self._streams: dict[str, np.memmap] = {}

    def close(self) -> None:
        self._streams.clear()

    def _stream(self, sensor_key: str) -> np.ndarray | None:
        if sensor_key in self._streams:
            return self._streams[sensor_key]
        streams = self.manifest.get("streams")
        if not isinstance(streams, Mapping):
            return None
        spec = streams.get(sensor_key)
        if not isinstance(spec, Mapping):
            return None
        path = _resolve_dataset_path(self.dataset_root, spec.get("path"))
        if path is None or not path.exists():
            return None
        shape = tuple(int(dim) for dim in spec.get("shape", ()) or ())
        frame_count = int(spec.get("frame_count") or self.frame_count)
        dtype = np.dtype(str(spec.get("dtype") or "float32"))
        if frame_count <= 0 or not shape:
            return None
        array = np.memmap(path, dtype=dtype, mode="r", shape=(frame_count, *shape))
        self._streams[sensor_key] = array
        return array

    def _manifest_frame_record(self, tactile_frame_index: int) -> dict[str, Any]:
        for record in self.frames:
            if _int_or_none(record.get("tactile_frame_index")) == int(tactile_frame_index):
                return dict(record)
        if 0 <= int(tactile_frame_index) < len(self.frames):
            return dict(self.frames[int(tactile_frame_index)])
        return {}

    def frame_index_at_or_before_sample(self, sample_frame_index: int) -> int | None:
        best: int | None = None
        for record in self.frames:
            sample_index = _int_or_none(record.get("sample_frame_index"))
            tactile_index = _int_or_none(record.get("tactile_frame_index"))
            if sample_index is None or tactile_index is None:
                continue
            if sample_index <= int(sample_frame_index):
                best = int(tactile_index)
            else:
                break
        return best

    def frame(self, *, tactile_frame_index: int) -> TactileFrame:
        arrays: dict[str, np.ndarray] = {}
        for sensor_key in self.sensor_keys:
            stream = self._stream(sensor_key)
            if stream is None:
                continue
            if 0 <= int(tactile_frame_index) < int(stream.shape[0]):
                arrays[str(sensor_key)] = np.asarray(stream[int(tactile_frame_index)], dtype=np.float32).copy()
        record = self._manifest_frame_record(int(tactile_frame_index))
        return TactileFrame(
            side=self.side,
            arrays=arrays,
            tactile_frame_index=int(tactile_frame_index),
            sample_frame_index=_int_or_none(record.get("sample_frame_index")),
            source="external",
            metadata=record,
        )


class TactileEpisodeReader:
    def __init__(self, episode: LoadedReplayEpisode, side: str):
        self.episode = episode
        self.side = str(side)
        self._stores: dict[Path, ExternalTactileStore] = {}

    def close(self) -> None:
        for store in self._stores.values():
            store.close()
        self._stores.clear()

    def _store_for_path(self, path: Path) -> ExternalTactileStore:
        resolved = path.resolve()
        store = self._stores.get(resolved)
        if store is None:
            store = ExternalTactileStore(dataset_root=self.episode.dataset_root, manifest_path=resolved)
            self._stores[resolved] = store
        return store

    def _manifest_path_for_row(self, row: Mapping[str, Any]) -> Path | None:
        prefix = f"observation.hand.{self.side}.tactile"
        path = _resolve_dataset_path(self.episode.dataset_root, row.get(f"{prefix}_path"))
        if path is not None:
            return path
        summary_path = self.episode.summary.tactile_paths.get(self.side)
        return _resolve_dataset_path(self.episode.dataset_root, summary_path)

    def _inline_frame(self, row: Mapping[str, Any], *, replay_index: int) -> TactileFrame:
        prefix = f"observation.hand.{self.side}.tactile"
        vector = _finite_vector(row.get(prefix))
        if vector is None:
            return TactileFrame(side=self.side, source="none")
        arrays: dict[str, np.ndarray] = {}
        sensor_keys = [str(key) for key in row.get(f"{prefix}_sensor_keys", []) or []]
        shapes = _safe_json_dict(row.get(f"{prefix}_shapes_json"))
        offset = 0
        for sensor_key in sensor_keys:
            shape_value = shapes.get(sensor_key)
            if not shape_value:
                continue
            shape = tuple(int(dim) for dim in shape_value)
            size = int(np.prod(shape))
            if size <= 0 or offset + size > vector.size:
                arrays = {}
                break
            arrays[sensor_key] = vector[offset : offset + size].reshape(shape).copy()
            offset += size
        return TactileFrame(
            side=self.side,
            arrays=arrays,
            vector=vector.copy(),
            tactile_frame_index=_int_or_none(row.get(f"{prefix}_frame_index")),
            sample_frame_index=_int_or_none(row.get("frame_index")) or int(replay_index),
            source="parquet_inline",
        )

    def frame_for(self, row: Mapping[str, Any], *, replay_index: int) -> TactileFrame:
        prefix = f"observation.hand.{self.side}.tactile"
        storage_format = str(row.get(f"{prefix}_storage_format") or "").strip()
        manifest_path = self._manifest_path_for_row(row)
        tactile_frame_index = _int_or_none(row.get(f"{prefix}_frame_index"))
        if manifest_path is not None and manifest_path.exists() and tactile_frame_index is not None:
            frame = self._store_for_path(manifest_path).frame(tactile_frame_index=tactile_frame_index)
            if frame.available:
                frame.side = self.side
                frame.source = storage_format or frame.source
                return frame
        if manifest_path is not None and manifest_path.exists() and tactile_frame_index is None:
            store = self._store_for_path(manifest_path)
            sample_frame_index = _int_or_none(row.get("frame_index"))
            if sample_frame_index is not None:
                tactile_frame_index = store.frame_index_at_or_before_sample(sample_frame_index)
            if tactile_frame_index is None:
                tactile_frame_index = int(replay_index)
            frame = store.frame(tactile_frame_index=tactile_frame_index)
            if frame.available:
                frame.side = self.side
                frame.source = storage_format or frame.source
                return frame
        inline = self._inline_frame(row, replay_index=replay_index)
        if inline.available:
            return inline
        return TactileFrame(side=self.side, source=storage_format or "none")


def _robust_ref(values: np.ndarray, fixed_ref: float | None = None) -> float:
    if fixed_ref is not None and fixed_ref > 0.0:
        return float(fixed_ref)
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size <= 0:
        return 1.0
    ref = float(np.percentile(np.abs(finite), 98))
    if ref <= 1e-8:
        ref = float(np.max(np.abs(finite)))
    return max(ref, 1e-6)


def _render_force_array(array: np.ndarray, *, title: str) -> np.ndarray:
    cv2 = _require_cv2()
    force = np.asarray(array, dtype=np.float32)
    rows = 20
    cols = 30
    cell = 14
    if force.ndim == 3 and force.shape[2] >= 3:
        source = np.nan_to_num(force[..., :3], nan=0.0, posinf=0.0, neginf=0.0)
        grid = cv2.resize(source, (cols, rows), interpolation=cv2.INTER_AREA)
        pressure = np.maximum(grid[..., 1], 0.0)
        plane = grid[..., (0, 2)]
    else:
        scalar = np.asarray(force, dtype=np.float32)
        while scalar.ndim > 2:
            scalar = scalar[..., 0]
        if scalar.ndim == 1:
            scalar = scalar.reshape(1, -1)
        grid_scalar = cv2.resize(np.nan_to_num(scalar, nan=0.0), (cols, rows), interpolation=cv2.INTER_AREA)
        pressure = np.abs(grid_scalar)
        plane = np.zeros((rows, cols, 2), dtype=np.float32)
    pressure_ref = _robust_ref(pressure)
    pressure_u8 = np.clip(pressure / pressure_ref, 0.0, 1.0)
    heat = cv2.applyColorMap((pressure_u8 * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    image = cv2.resize(heat, (cols * cell, rows * cell), interpolation=cv2.INTER_NEAREST)
    grid_color = (48, 48, 48)
    for col in range(cols + 1):
        x = col * cell
        cv2.line(image, (x, 0), (x, rows * cell), grid_color, 1)
    for row in range(rows + 1):
        y = row * cell
        cv2.line(image, (0, y), (cols * cell, y), grid_color, 1)

    plane_norm = np.linalg.norm(plane, axis=2)
    vector_ref = _robust_ref(plane_norm)
    for row in range(0, rows, 2):
        for col in range(0, cols, 2):
            norm = float(plane_norm[row, col])
            if norm / vector_ref < 0.02:
                continue
            direction = plane[row, col] / max(norm, 1e-8)
            length = min(float(cell) * 0.42, float(cell) * 0.42 * norm / vector_ref)
            cx = int(round((col + 0.5) * cell))
            cy = int(round((row + 0.5) * cell))
            ex = int(round(cx + float(direction[0]) * length))
            ey = int(round(cy - float(direction[1]) * length))
            cv2.arrowedLine(image, (cx, cy), (ex, ey), (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.35)

    header_h = 34
    footer_h = 24
    canvas = np.zeros((header_h + image.shape[0] + footer_h, image.shape[1], 3), dtype=np.uint8)
    canvas[header_h : header_h + image.shape[0], :, :] = image
    cv2.putText(canvas, title[:48], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (236, 238, 240), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"ref={pressure_ref:.3g}  shape={tuple(int(dim) for dim in force.shape)}",
        (8, canvas.shape[0] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (206, 212, 218),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _render_vector(vector: np.ndarray, *, title: str) -> np.ndarray:
    cv2 = _require_cv2()
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    width = 640
    height = 260
    image = np.full((height, width, 3), 246, dtype=np.uint8)
    cv2.putText(image, title[:48], (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (32, 40, 48), 1, cv2.LINE_AA)
    if values.size:
        ref = _robust_ref(values)
        bar_w = max(1, int((width - 40) / max(1, values.size)))
        mid_y = height - 46
        cv2.line(image, (18, mid_y), (width - 18, mid_y), (148, 154, 160), 1, cv2.LINE_AA)
        for index, value in enumerate(values[:200]):
            x0 = 20 + index * bar_w
            x1 = min(width - 20, x0 + max(1, bar_w - 1))
            length = int(round(min(1.0, abs(float(value)) / ref) * 150.0))
            color = (30, 132, 118) if value >= 0 else (170, 82, 64)
            y0 = mid_y - length if value >= 0 else mid_y
            y1 = mid_y if value >= 0 else mid_y + length
            cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)
        cv2.putText(
            image,
            f"n={values.size}  ref={ref:.3g}",
            (16, height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (78, 86, 94),
            1,
            cv2.LINE_AA,
        )
    return image


def tactile_frame_jpeg(frame: TactileFrame) -> bytes:
    if not frame.available:
        return placeholder_jpeg(f"{frame.side} tactile", ["no tactile frame"])
    panels: list[np.ndarray] = []
    for sensor_key, array in sorted(frame.arrays.items()):
        panels.append(_render_force_array(array, title=f"{frame.side}:{sensor_key}"))
    if not panels and frame.vector is not None:
        panels.append(_render_vector(frame.vector, title=f"{frame.side}:vector"))
    if not panels:
        return placeholder_jpeg(f"{frame.side} tactile", ["empty tactile frame"])
    canvas = panels[0] if len(panels) == 1 else _hstack_equal_height(panels)
    canvas = _resize_to_fit(canvas, max_width=1600, max_height=720)
    return _encode_jpeg(canvas, quality=88)


def build_camera_tracks(episode: LoadedReplayEpisode) -> dict[str, CameraVideoTrack]:
    paths: dict[str, str] = dict(episode.summary.video_paths)
    for row in episode.rows:
        for key, value in row.items():
            prefix = "observation.images."
            suffix = ".path"
            if key.startswith(prefix) and key.endswith(suffix) and value is not None:
                camera_key = key[len(prefix) : -len(suffix)]
                if camera_key:
                    paths.setdefault(camera_key, str(value))
    return {
        key: CameraVideoTrack(dataset_root=episode.dataset_root, camera_key=key, path_text=path_text)
        for key, path_text in sorted(paths.items())
    }


def build_tactile_readers(episode: LoadedReplayEpisode) -> dict[str, TactileEpisodeReader]:
    return {side: TactileEpisodeReader(episode, side) for side in episode.summary.tactile_sides}
