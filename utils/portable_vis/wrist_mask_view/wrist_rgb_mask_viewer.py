#!/usr/bin/env python3
"""Browser viewer for recorded wrist RGB videos and 0/1 depth masks."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse
import webbrowser


def _ensure_repo_root_on_path() -> None:
    """Allow running this portable script directly by file path."""

    for parent in Path(__file__).resolve().parents:
        if (parent / "dataset_postprocess").is_dir() and (parent / "teleop_runtime").is_dir():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return


_ensure_repo_root_on_path()

try:
    import cv2
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("OpenCV is required to run this script. Install python-opencv/cv2 first.") from exc

import numpy as np

from dataset_postprocess.wrist_cam_adjust.depth import WRIST_DEPTH_MASK_FORMAT, unpack_binary_mask
from dataset_postprocess.wrist_cam_adjust.visualize_depth_mask import MaskStream, load_mask_stream


DEFAULT_CAMERA_KEYS = ("left_wrist", "right_wrist")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8776
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_NOMINAL_FPS = 15.0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a recorded episode's left/right wrist RGB videos and "
            "bit-packed 0/1 wrist depth masks in a local browser GUI."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="?",
        help=(
            "Dataset root, episode parquet/video/mask path, wrist_depth episode directory, "
            "or mask manifest. If omitted, the GUI opens with an empty path field."
        ),
    )
    parser.add_argument("--episode", type=int, help="Episode index when PATH is a dataset root.")
    parser.add_argument(
        "--camera-keys",
        default=",".join(DEFAULT_CAMERA_KEYS),
        help="Comma-separated wrist camera keys to display.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host used by the web UI.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port used by the web UI.")
    parser.add_argument("--no-open", action="store_true", help="Print the URL without opening a browser.")
    parser.add_argument("--start-frame", type=int, default=0, help="Initial frame index.")
    parser.add_argument(
        "--mode",
        choices=("mask", "overlay"),
        default="mask",
        help="Initial mask display mode.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Render one frame to a PNG and exit instead of starting the GUI.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index used with --snapshot.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=DEFAULT_FRAME_WIDTH,
        help="Composite image width used by the GUI and --snapshot.",
    )
    return parser.parse_args(argv)


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object in {path}:{line_number}")
        rows.append(value)
    return rows


def _episode_id_from_index(index: int) -> str:
    return f"episode_{int(index):06d}"


def _chunk_id_from_episode_index(index: int, *, chunk_size: int) -> str:
    return f"chunk-{int(index) // int(chunk_size):03d}"


def _parse_episode_index_from_path(path: str | Path) -> int | None:
    match = re.search(r"episode_(\d+)", str(path))
    return int(match.group(1)) if match else None


def _parse_camera_keys(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    else:
        items = [str(part).strip() for part in value]
    camera_keys = tuple(item for item in items if item)
    if not camera_keys:
        raise ValueError("at least one camera key is required.")
    return camera_keys


def _dataset_root_from_path(path: Path) -> Path | None:
    base = path if path.is_dir() else path.parent
    for candidate in (base, *base.parents):
        if (candidate / "meta" / "episodes.jsonl").exists():
            return candidate.resolve()

    parts = path.parts
    for marker in ("data", "videos", "wrist_depth", "tactile", "meta"):
        if marker not in parts:
            continue
        marker_index = parts.index(marker)
        if marker_index <= 0:
            continue
        return Path(*parts[:marker_index]).resolve()
    return None


def _read_dataset_info(dataset_root: Path | None) -> dict[str, Any]:
    if dataset_root is None:
        return {}
    info_path = dataset_root / "meta" / "info.json"
    return _read_json(info_path) if info_path.exists() else {}


def _chunk_size_from_info(info: Mapping[str, Any]) -> int:
    try:
        value = int(info.get("chunk_size") or 1000)
    except (TypeError, ValueError):
        value = 1000
    return max(1, value)


def _nominal_fps_from_info_record(info: Mapping[str, Any], record: Mapping[str, Any]) -> float:
    for value in (
        info.get("nominal_fps"),
        record.get("nominal_fps"),
        (record.get("summary") or {}).get("nominal_fps") if isinstance(record.get("summary"), Mapping) else None,
        (record.get("summary") or {}).get("wrist_depth_fps") if isinstance(record.get("summary"), Mapping) else None,
    ):
        try:
            fps = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fps) and fps > 0.0:
            return fps
    return DEFAULT_NOMINAL_FPS


def _dataset_relative_path(root: Path, value: str | Path) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        if raw.exists():
            return raw.resolve()
        parts = raw.parts
        for marker in ("videos", "wrist_depth", "data", "tactile", "meta"):
            if marker not in parts:
                continue
            candidate = root / Path(*parts[parts.index(marker) :])
            if candidate.exists():
                return candidate.resolve()
        return raw
    return (root / raw).resolve()


def _path_payload(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _conventional_episode_record(
    dataset_root: Path | None,
    episode_index: int,
    *,
    camera_keys: Sequence[str],
    chunk_size: int = 1000,
) -> dict[str, Any]:
    episode_id = _episode_id_from_index(episode_index)
    chunk_id = _chunk_id_from_episode_index(episode_index, chunk_size=chunk_size)
    video_paths = {
        camera_key: str(Path("videos") / f"observation.images.{camera_key}" / chunk_id / f"{episode_id}.mp4")
        for camera_key in camera_keys
    }
    mask_paths = {
        camera_key: str(
            Path("wrist_depth")
            / "mask"
            / f"observation.images.{camera_key}"
            / chunk_id
            / episode_id
            / "manifest.json"
        )
        for camera_key in camera_keys
    }
    return {
        "episode_index": int(episode_index),
        "episode_id": episode_id,
        "frame_count": 0,
        "parquet_path": str(Path("data") / chunk_id / f"{episode_id}.parquet"),
        "video_paths": video_paths,
        "wrist_depth_mask_paths": mask_paths,
        "metadata": {},
        "summary": {"dataset_root": str(dataset_root) if dataset_root is not None else None},
    }


def _available_episode_indices(records: Sequence[Mapping[str, Any]]) -> list[int]:
    indices: list[int] = []
    for record in records:
        try:
            indices.append(int(record.get("episode_index")))
        except (TypeError, ValueError):
            continue
    return sorted(indices)


def _episode_record_from_dataset(
    dataset_root: Path,
    *,
    episode_index: int | None,
    camera_keys: Sequence[str],
) -> dict[str, Any]:
    info = _read_dataset_info(dataset_root)
    chunk_size = _chunk_size_from_info(info)
    records = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    if episode_index is None:
        if records:
            return dict(records[0])
        episode_index = 0

    for record in records:
        try:
            if int(record.get("episode_index")) == int(episode_index):
                return dict(record)
        except (TypeError, ValueError):
            continue

    fallback = _conventional_episode_record(
        dataset_root,
        int(episode_index),
        camera_keys=camera_keys,
        chunk_size=chunk_size,
    )
    candidates = []
    for mapping_key in ("video_paths", "wrist_depth_mask_paths"):
        values = fallback.get(mapping_key)
        if isinstance(values, Mapping):
            candidates.extend(_dataset_relative_path(dataset_root, str(value)) for value in values.values())
    if any(path.exists() for path in candidates):
        return fallback

    available = _available_episode_indices(records)
    preview = ", ".join(str(index) for index in available[:20])
    if len(available) > 20:
        preview += ", ..."
    raise ValueError(f"episode {episode_index} not found in {dataset_root}/meta/episodes.jsonl. Available: {preview}")


def _mask_manifest_camera(path: Path) -> str | None:
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".json":
        return None
    try:
        payload = _read_json(path)
    except Exception:  # noqa: BLE001
        return None
    if payload.get("format") != WRIST_DEPTH_MASK_FORMAT:
        return None
    camera_key = payload.get("camera_key")
    return str(camera_key) if camera_key else None


def _direct_mask_manifest_overrides(path: Path, camera_keys: Sequence[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    if path.is_file():
        camera = _mask_manifest_camera(path)
        if camera:
            overrides[camera] = path.resolve()
        return overrides

    candidates: list[Path] = []
    if path.is_dir():
        candidates.extend([path / "manifest.json", path / "mask_manifest.json"])
        for camera_key in camera_keys:
            candidates.extend(
                [
                    path / camera_key / "manifest.json",
                    path / camera_key / "mask_manifest.json",
                    path / f"observation.images.{camera_key}" / "manifest.json",
                    path / f"observation.images.{camera_key}" / "mask_manifest.json",
                ]
            )

    for candidate in candidates:
        camera = _mask_manifest_camera(candidate)
        if camera and camera in camera_keys:
            overrides[camera] = candidate.resolve()
    return overrides


def _dataset_root_from_mask_source(manifest_path: Path) -> Path | None:
    try:
        payload = _read_json(manifest_path)
    except Exception:  # noqa: BLE001
        return None
    source = payload.get("source")
    if not isinstance(source, Mapping):
        return None
    for key in ("manifest_path", "stream_path"):
        value = source.get(key)
        if not isinstance(value, str) or not value:
            continue
        root = _dataset_root_from_path(Path(value).expanduser())
        if root is not None:
            return root
    return None


def _resolve_episode_record(
    source: str | Path,
    *,
    episode_index: int | None,
    camera_keys: Sequence[str],
) -> tuple[Path | None, dict[str, Any]]:
    path = Path(source).expanduser().resolve()
    inferred_episode = _parse_episode_index_from_path(path)
    requested_episode = int(episode_index) if episode_index is not None else inferred_episode
    direct_masks = _direct_mask_manifest_overrides(path, camera_keys)

    dataset_root = _dataset_root_from_path(path)
    if dataset_root is None and direct_masks:
        for manifest in direct_masks.values():
            dataset_root = _dataset_root_from_mask_source(manifest)
            if dataset_root is not None:
                break

    if dataset_root is not None and (dataset_root / "meta" / "episodes.jsonl").exists():
        record = _episode_record_from_dataset(
            dataset_root,
            episode_index=requested_episode,
            camera_keys=camera_keys,
        )
    elif dataset_root is not None and requested_episode is not None:
        info = _read_dataset_info(dataset_root)
        record = _conventional_episode_record(
            dataset_root,
            int(requested_episode),
            camera_keys=camera_keys,
            chunk_size=_chunk_size_from_info(info),
        )
    elif direct_masks and requested_episode is not None:
        record = _conventional_episode_record(
            dataset_root,
            int(requested_episode),
            camera_keys=camera_keys,
        )
    elif direct_masks:
        first_count = 0
        try:
            first_stream = load_mask_stream(next(iter(direct_masks.values())))
            first_count = int(first_stream.frame_count)
        except Exception:  # noqa: BLE001
            pass
        record = {
            "episode_index": int(requested_episode or 0),
            "episode_id": _episode_id_from_index(int(requested_episode or 0)),
            "frame_count": first_count,
            "video_paths": {},
            "wrist_depth_mask_paths": {},
            "metadata": {},
            "summary": {},
        }
    else:
        raise FileNotFoundError(
            f"Could not resolve a DexTeleop dataset episode from path: {path}. "
            "Pass a dataset root with --episode, a parquet/video/mask path, or a mask manifest."
        )

    if direct_masks:
        mask_paths = dict(record.get("wrist_depth_mask_paths") or {})
        for camera_key, manifest in direct_masks.items():
            mask_paths[camera_key] = str(manifest)
        record["wrist_depth_mask_paths"] = mask_paths
    return dataset_root, record


def _video_path_for_camera(
    dataset_root: Path | None,
    record: Mapping[str, Any],
    camera_key: str,
    *,
    chunk_size: int,
) -> Path | None:
    if dataset_root is None:
        return None
    raw_paths = record.get("video_paths")
    if isinstance(raw_paths, Mapping):
        value = raw_paths.get(camera_key)
        if isinstance(value, str) and value:
            candidate = _dataset_relative_path(dataset_root, value)
            if candidate.exists():
                return candidate

    try:
        episode_index = int(record.get("episode_index"))
    except (TypeError, ValueError):
        return None
    fallback = _conventional_episode_record(
        dataset_root,
        episode_index,
        camera_keys=(camera_key,),
        chunk_size=chunk_size,
    )
    value = (fallback.get("video_paths") or {}).get(camera_key)
    candidate = _dataset_relative_path(dataset_root, str(value))
    return candidate if candidate.exists() else None


def _mask_manifest_for_camera(
    dataset_root: Path | None,
    record: Mapping[str, Any],
    camera_key: str,
    *,
    chunk_size: int,
) -> Path | None:
    raw_paths = record.get("wrist_depth_mask_paths")
    if isinstance(raw_paths, Mapping):
        value = raw_paths.get(camera_key)
        if isinstance(value, str) and value:
            candidate = Path(value).expanduser()
            if dataset_root is not None:
                candidate = _dataset_relative_path(dataset_root, candidate)
            else:
                candidate = candidate.resolve()
            if candidate.exists():
                return candidate

    if dataset_root is None:
        return None
    try:
        episode_index = int(record.get("episode_index"))
    except (TypeError, ValueError):
        return None
    fallback = _conventional_episode_record(
        dataset_root,
        episode_index,
        camera_keys=(camera_key,),
        chunk_size=chunk_size,
    )
    value = (fallback.get("wrist_depth_mask_paths") or {}).get(camera_key)
    candidate = _dataset_relative_path(dataset_root, str(value))
    return candidate if candidate.exists() else None


def _read_video_metadata(path: Path | None) -> tuple[int, float, int, int]:
    if path is None or not path.exists():
        return 0, DEFAULT_NOMINAL_FPS, 0, 0
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0, DEFAULT_NOMINAL_FPS, 0, 0
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if fps <= 0.0 or not np.isfinite(fps):
        fps = DEFAULT_NOMINAL_FPS
    return max(0, frame_count), fps, max(0, width), max(0, height)


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        return cv2.cvtColor(array.astype(np.uint8, copy=False), cv2.COLOR_GRAY2BGR)
    if array.ndim == 3 and array.shape[2] == 4:
        return cv2.cvtColor(array.astype(np.uint8, copy=False), cv2.COLOR_BGRA2BGR)
    if array.ndim == 3 and array.shape[2] == 3:
        return array.astype(np.uint8, copy=False)
    raise ValueError(f"unsupported image shape: {array.shape!r}")


def _message_frame(width: int, height: int, title: str, detail: str = "") -> np.ndarray:
    w = max(320, int(width or 640))
    h = max(240, int(height or 480))
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:, :] = (28, 31, 37)
    cv2.putText(image, title, (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (232, 237, 244), 2, cv2.LINE_AA)
    if detail:
        y = 78
        for line in _wrap_text(detail, max_chars=70)[:7]:
            cv2.putText(image, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (176, 187, 201), 1, cv2.LINE_AA)
            y += 24
    return image


def _wrap_text(text: str, *, max_chars: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines or [""]


@dataclass
class WristCameraSource:
    camera_key: str
    video_path: Path | None
    mask_stream: MaskStream | None
    video_frame_count: int = 0
    video_fps: float = DEFAULT_NOMINAL_FPS
    video_width: int = 0
    video_height: int = 0
    capture: cv2.VideoCapture | None = field(default=None, init=False, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def mask_frame_count(self) -> int:
        return 0 if self.mask_stream is None else int(self.mask_stream.frame_count)

    @property
    def frame_count(self) -> int:
        return max(int(self.video_frame_count), int(self.mask_frame_count))

    def close(self) -> None:
        with self.lock:
            if self.capture is not None:
                self.capture.release()
                self.capture = None

    def read_rgb(self, frame_index: int) -> np.ndarray:
        if self.video_path is None or not self.video_path.exists():
            return _message_frame(640, 480, f"{self.camera_key} RGB unavailable", "video file was not found")
        if self.video_frame_count <= 0:
            return _message_frame(640, 480, f"{self.camera_key} RGB unavailable", str(self.video_path))
        index = int(frame_index)
        if index < 0 or index >= int(self.video_frame_count):
            return _message_frame(self.video_width, self.video_height, f"{self.camera_key} RGB ended", str(self.video_path))

        with self.lock:
            if self.capture is None or not self.capture.isOpened():
                self.capture = cv2.VideoCapture(str(self.video_path))
            if self.capture is None or not self.capture.isOpened():
                return _message_frame(640, 480, f"{self.camera_key} RGB unavailable", str(self.video_path))
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self.capture.read()
        if not ok or frame is None:
            return _message_frame(self.video_width, self.video_height, f"{self.camera_key} RGB read failed", str(self.video_path))
        return _ensure_bgr(frame)

    def read_mask(self, frame_index: int) -> np.ndarray:
        stream = self.mask_stream
        if stream is None:
            return _message_frame(640, 480, f"{self.camera_key} mask unavailable", "mask manifest was not found")
        if stream.frame_count <= 0:
            return _message_frame(stream.width, stream.height, f"{self.camera_key} mask unavailable", str(stream.stream_path))
        index = int(frame_index)
        if index < 0 or index >= int(stream.frame_count):
            return _message_frame(stream.width, stream.height, f"{self.camera_key} mask ended", str(stream.stream_path))
        with stream.stream_path.open("rb") as handle:
            handle.seek(index * int(stream.packed_bytes_per_frame))
            data = handle.read(int(stream.packed_bytes_per_frame))
        if len(data) != int(stream.packed_bytes_per_frame):
            return _message_frame(stream.width, stream.height, f"{self.camera_key} mask read failed", str(stream.stream_path))
        mask = unpack_binary_mask(data, height=stream.height, width=stream.width)
        return (mask.astype(np.uint8, copy=False) * 255)

    def foreground_fraction(self, frame_index: int, mask_image: np.ndarray | None = None) -> float | None:
        stream = self.mask_stream
        if stream is None:
            return None
        if 0 <= int(frame_index) < len(stream.frames):
            for key in ("foreground_fraction", "spatial_foreground_fraction", "input_foreground_fraction"):
                raw = stream.frames[int(frame_index)].get(key)
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    continue
        if mask_image is None or mask_image.size <= 0:
            return None
        return float(np.count_nonzero(mask_image)) / float(mask_image.size)


@dataclass
class EpisodeView:
    dataset_root: Path | None
    episode_index: int
    episode_id: str
    record: dict[str, Any]
    cameras: list[WristCameraSource]
    nominal_fps: float
    source_path: Path | None = None

    @property
    def frame_count(self) -> int:
        counts = [int(camera.frame_count) for camera in self.cameras]
        try:
            counts.append(int(self.record.get("frame_count") or 0))
        except (TypeError, ValueError):
            pass
        return max([1, *counts])

    @property
    def duration_s(self) -> float:
        if self.nominal_fps <= 0.0:
            return 0.0
        return max(0.0, (int(self.frame_count) - 1) / float(self.nominal_fps))

    def close(self) -> None:
        for camera in self.cameras:
            camera.close()

    def to_payload(self) -> dict[str, Any]:
        return {
            "loaded": True,
            "datasetRoot": None if self.dataset_root is None else str(self.dataset_root),
            "sourcePath": None if self.source_path is None else str(self.source_path),
            "episodeIndex": int(self.episode_index),
            "episodeId": self.episode_id,
            "frameCount": int(self.frame_count),
            "nominalFps": float(self.nominal_fps),
            "duration": float(self.duration_s),
            "cameras": [
                {
                    "cameraKey": camera.camera_key,
                    "videoPath": _path_payload(camera.video_path),
                    "maskManifest": _path_payload(None if camera.mask_stream is None else camera.mask_stream.manifest_path),
                    "maskStream": _path_payload(None if camera.mask_stream is None else camera.mask_stream.stream_path),
                    "videoFrameCount": int(camera.video_frame_count),
                    "maskFrameCount": int(camera.mask_frame_count),
                    "videoSize": [int(camera.video_width), int(camera.video_height)],
                    "maskSize": (
                        []
                        if camera.mask_stream is None
                        else [int(camera.mask_stream.width), int(camera.mask_stream.height)]
                    ),
                }
                for camera in self.cameras
            ],
        }


def load_episode_view(
    source: str | Path,
    *,
    episode_index: int | None = None,
    camera_keys: Sequence[str] = DEFAULT_CAMERA_KEYS,
) -> EpisodeView:
    keys = _parse_camera_keys(camera_keys)
    source_path = Path(source).expanduser().resolve()
    dataset_root, record = _resolve_episode_record(source_path, episode_index=episode_index, camera_keys=keys)
    info = _read_dataset_info(dataset_root)
    chunk_size = _chunk_size_from_info(info)

    try:
        resolved_episode_index = int(record.get("episode_index"))
    except (TypeError, ValueError):
        inferred = _parse_episode_index_from_path(source_path)
        resolved_episode_index = int(inferred or 0)
    episode_id = str(record.get("episode_id") or _episode_id_from_index(resolved_episode_index))
    nominal_fps = _nominal_fps_from_info_record(info, record)

    cameras: list[WristCameraSource] = []
    for camera_key in keys:
        video_path = _video_path_for_camera(dataset_root, record, camera_key, chunk_size=chunk_size)
        mask_manifest = _mask_manifest_for_camera(dataset_root, record, camera_key, chunk_size=chunk_size)
        mask_stream = load_mask_stream(mask_manifest, root_hint=dataset_root, label=camera_key) if mask_manifest else None
        video_count, video_fps, video_width, video_height = _read_video_metadata(video_path)
        cameras.append(
            WristCameraSource(
                camera_key=str(camera_key),
                video_path=video_path,
                mask_stream=mask_stream,
                video_frame_count=video_count,
                video_fps=video_fps,
                video_width=video_width,
                video_height=video_height,
            )
        )

    return EpisodeView(
        dataset_root=dataset_root,
        episode_index=resolved_episode_index,
        episode_id=episode_id,
        record=record,
        cameras=cameras,
        nominal_fps=nominal_fps,
        source_path=source_path,
    )


def _fit_image(image: np.ndarray, width: int, height: int, *, interpolation: int) -> np.ndarray:
    frame = _ensure_bgr(image)
    target_w = max(1, int(width))
    target_h = max(1, int(height))
    src_h, src_w = frame.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / src_w, target_h / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:, :] = (7, 9, 12)
    x0 = (target_w - resized_w) // 2
    y0 = (target_h - resized_h) // 2
    canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return canvas


def _overlay_mask(rgb: np.ndarray, mask_image: np.ndarray) -> np.ndarray:
    rgb_bgr = _ensure_bgr(rgb)
    mask = np.asarray(mask_image)
    if mask.ndim == 3:
        mask = cv2.cvtColor(_ensure_bgr(mask), cv2.COLOR_BGR2GRAY)
    if mask.shape[:2] != rgb_bgr.shape[:2]:
        mask = cv2.resize(mask, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    active = mask > 0
    overlay = rgb_bgr.copy()
    color = np.zeros_like(overlay)
    color[:, :] = (40, 220, 130)
    overlay[active] = cv2.addWeighted(overlay[active], 0.55, color[active], 0.45, 0.0)
    return overlay


def _panel(image: np.ndarray, *, label: str, width: int, height: int, mask_like: bool = False) -> np.ndarray:
    header_h = 34
    body = _fit_image(
        image,
        width,
        height,
        interpolation=cv2.INTER_NEAREST if mask_like else cv2.INTER_AREA,
    )
    canvas = np.zeros((height + header_h, width, 3), dtype=np.uint8)
    canvas[:header_h, :] = (33, 38, 47)
    canvas[header_h:, :] = body
    cv2.putText(
        canvas,
        label[:120],
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (232, 237, 244),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (0, 0), (width - 1, height + header_h - 1), (62, 70, 84), 1)
    return canvas


def compose_episode_frame(
    view: EpisodeView,
    *,
    frame_index: int,
    mode: str = "mask",
    width: int = DEFAULT_FRAME_WIDTH,
) -> np.ndarray:
    camera_count = max(1, len(view.cameras))
    cell_w = max(300, int(width) // camera_count)
    cell_h = max(220, int(round(cell_w * 0.75)))
    frame_index = max(0, min(int(frame_index), int(view.frame_count) - 1))

    rgb_panels: list[np.ndarray] = []
    mask_panels: list[np.ndarray] = []
    for camera in view.cameras:
        rgb = camera.read_rgb(frame_index)
        mask = camera.read_mask(frame_index)
        fg = camera.foreground_fraction(frame_index, mask)
        rgb_label = f"{camera.camera_key} RGB  frame {frame_index + 1}/{view.frame_count}"
        fg_text = "n/a" if fg is None else f"{fg:.4f}"
        mask_label = f"{camera.camera_key} {'overlay' if mode == 'overlay' else 'mask 0/1'}  fg={fg_text}"
        mask_display = _overlay_mask(rgb, mask) if mode == "overlay" else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask.ndim == 2 else mask
        rgb_panels.append(_panel(rgb, label=rgb_label, width=cell_w, height=cell_h))
        mask_panels.append(
            _panel(
                mask_display,
                label=mask_label,
                width=cell_w,
                height=cell_h,
                mask_like=(mode != "overlay"),
            )
        )

    top = np.hstack(rgb_panels)
    bottom = np.hstack(mask_panels)
    footer_h = 34
    canvas = np.vstack([top, bottom, np.zeros((footer_h, top.shape[1], 3), dtype=np.uint8)])
    canvas[-footer_h:, :] = (18, 21, 27)
    dataset_label = "direct masks" if view.dataset_root is None else str(view.dataset_root)
    footer = f"{view.episode_id}  frame={frame_index}/{view.frame_count - 1}  fps={view.nominal_fps:.2f}  {dataset_label}"
    cv2.putText(canvas, footer[:180], (10, canvas.shape[0] - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (174, 184, 198), 1, cv2.LINE_AA)
    return canvas


def _jpeg_frame_for_view(
    view: EpisodeView,
    *,
    frame_index: int,
    mode: str = "mask",
    width: int = DEFAULT_FRAME_WIDTH,
) -> bytes:
    canvas = compose_episode_frame(view, frame_index=frame_index, mode=mode, width=width)
    ok, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok:
        raise RuntimeError("failed to encode JPEG frame.")
    return encoded.tobytes()


class EpisodeRegistry:
    def __init__(self, *, camera_keys: Sequence[str]) -> None:
        self.camera_keys = _parse_camera_keys(camera_keys)
        self._lock = threading.RLock()
        self._view: EpisodeView | None = None
        self._error: str = ""

    def load(self, source: str | Path, *, episode_index: int | None) -> dict[str, Any]:
        view = load_episode_view(source, episode_index=episode_index, camera_keys=self.camera_keys)
        with self._lock:
            old = self._view
            self._view = view
            self._error = ""
        if old is not None:
            old.close()
        return view.to_payload()

    def set_error(self, error: str) -> None:
        with self._lock:
            self._error = str(error)

    def state_payload(self) -> dict[str, Any]:
        with self._lock:
            if self._view is None:
                return {"loaded": False, "error": self._error}
            payload = self._view.to_payload()
            if self._error:
                payload["error"] = self._error
            return payload

    def render_jpeg(self, *, frame_index: int, mode: str, width: int) -> bytes:
        with self._lock:
            view = self._view
        if view is None:
            placeholder = _message_frame(width, max(360, int(width * 0.56)), "No episode loaded", "Enter a dataset or episode path.")
            ok, encoded = cv2.imencode(".jpg", placeholder, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            if not ok:
                raise RuntimeError("failed to encode placeholder.")
            return encoded.tobytes()
        return _jpeg_frame_for_view(view, frame_index=frame_index, mode=mode, width=width)

    def close(self) -> None:
        with self._lock:
            view = self._view
            self._view = None
        if view is not None:
            view.close()


WEB_APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Wrist RGB Mask Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111316;
      --panel: #1a1d22;
      --panel-2: #232832;
      --line: #3c4350;
      --text: #eef2f5;
      --muted: #aeb7c2;
      --accent: #6bd0a5;
      --accent-2: #7db7ff;
      --danger: #ff817a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, input, select { font: inherit; }
    button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #252a33;
      color: var(--text);
      min-height: 34px;
      min-width: 36px;
      padding: 7px 10px;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent-2); }
    button.primary { background: var(--accent); border-color: var(--accent); color: #06150f; font-weight: 700; }
    input, select {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0c0f13;
      color: var(--text);
      padding: 8px 9px;
      min-height: 34px;
    }
    label { color: var(--muted); font-size: 12px; }
    .app {
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
      height: 100vh;
      min-height: 100vh;
    }
    .side {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      overflow: auto;
    }
    .brand { display: flex; justify-content: space-between; gap: 10px; align-items: center; }
    h1 { margin: 0; font-size: 18px; line-height: 1.2; }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      color: var(--muted);
      font-size: 11px;
      white-space: nowrap;
    }
    .stack { display: grid; gap: 8px; }
    .row { display: flex; gap: 8px; align-items: center; }
    .row > * { min-width: 0; }
    .grow { flex: 1; }
    .status {
      min-height: 38px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .details {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #0e1116;
      font-size: 12px;
    }
    .details div {
      display: grid;
      grid-template-columns: 88px minmax(0, 1fr);
      gap: 8px;
      padding: 7px 8px;
      border-bottom: 1px solid #252a33;
    }
    .details div:last-child { border-bottom: 0; }
    .details span:first-child { color: var(--muted); }
    .details span:last-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .main {
      min-width: 0;
      min-height: 0;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .toolbar {
      position: sticky;
      top: 0;
      z-index: 3;
      display: grid;
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: rgba(17, 19, 22, 0.96);
      flex: 0 0 auto;
    }
    .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .controls input[type="range"] {
      flex: 1 1 420px;
      min-width: 180px;
      padding: 0;
      accent-color: var(--accent);
    }
    .frame-label {
      color: var(--muted);
      min-width: 148px;
      font-variant-numeric: tabular-nums;
      font-size: 13px;
    }
    .viewer {
      flex: 1 1 auto;
      min-height: 0;
      padding: 12px;
      display: grid;
      place-items: center;
      overflow: auto;
    }
    #frameImage {
      max-width: 100%;
      max-height: 100%;
      width: auto;
      height: auto;
      object-fit: contain;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #050607;
      display: block;
    }
    #frameImage.actual-size {
      max-width: none;
      max-height: none;
    }
    @media (max-width: 920px) {
      .app { grid-template-columns: 1fr; height: auto; }
      .side { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <script id="initial-data" type="application/json">__INITIAL_DATA__</script>
  <div class="app">
    <aside class="side">
      <div class="brand">
        <h1>Wrist RGB Mask</h1>
        <span class="pill" id="loadPill">empty</span>
      </div>
      <div class="stack">
        <label for="pathInput">Dataset or episode path</label>
        <input id="pathInput" placeholder="/path/to/dataset">
      </div>
      <div class="row">
        <div class="stack grow">
          <label for="episodeInput">Episode</label>
          <input id="episodeInput" type="number" min="0" placeholder="auto">
        </div>
        <button class="primary" id="loadBtn" title="Load">Load</button>
      </div>
      <div class="status" id="status">Ready</div>
      <div class="details" id="details"></div>
    </aside>
    <main class="main">
      <div class="toolbar">
        <div class="controls">
          <button class="primary" id="playBtn" title="Play or pause">Play</button>
          <button id="prevBtn" title="Previous frame">&lt;</button>
          <button id="nextBtn" title="Next frame">&gt;</button>
          <button id="backBtn" title="Back 30 frames">-30</button>
          <button id="forwardBtn" title="Forward 30 frames">+30</button>
          <div class="frame-label" id="frameLabel">0 / 0</div>
          <input id="timeline" type="range" min="0" max="0" value="0">
        </div>
        <div class="controls">
          <label>Speed <select id="speedSelect">
            <option value="0.25">0.25x</option>
            <option value="0.5">0.5x</option>
            <option value="1" selected>1x</option>
            <option value="2">2x</option>
            <option value="4">4x</option>
          </select></label>
          <label>Mask <select id="modeSelect">
            <option value="mask">0/1</option>
            <option value="overlay">Overlay</option>
          </select></label>
          <label>Width <input id="widthInput" type="number" min="640" max="3200" step="80"></label>
          <label><input id="fitBox" type="checkbox" checked> Fit</label>
          <button id="restartBtn" title="Restart">Reset</button>
          <button id="snapshotBtn" title="Open current JPEG">JPEG</button>
        </div>
      </div>
      <section class="viewer">
        <img id="frameImage" alt="wrist RGB and mask frame">
      </section>
    </main>
  </div>
  <script>
    const initial = JSON.parse(document.getElementById("initial-data").textContent);
    const state = {
      loaded: false,
      frame: Number(initial.startFrame || 0),
      frameCount: 1,
      fps: 15,
      playing: false,
      speed: 1,
      mode: initial.mode || "mask",
      width: Number(initial.frameWidth || 1280),
      fit: true,
      anchorFrame: 0,
      anchorClock: 0,
      cache: 0,
      view: null
    };
    const els = {
      pathInput: document.getElementById("pathInput"),
      episodeInput: document.getElementById("episodeInput"),
      loadBtn: document.getElementById("loadBtn"),
      status: document.getElementById("status"),
      details: document.getElementById("details"),
      loadPill: document.getElementById("loadPill"),
      playBtn: document.getElementById("playBtn"),
      prevBtn: document.getElementById("prevBtn"),
      nextBtn: document.getElementById("nextBtn"),
      backBtn: document.getElementById("backBtn"),
      forwardBtn: document.getElementById("forwardBtn"),
      frameLabel: document.getElementById("frameLabel"),
      timeline: document.getElementById("timeline"),
      speedSelect: document.getElementById("speedSelect"),
      modeSelect: document.getElementById("modeSelect"),
      widthInput: document.getElementById("widthInput"),
      fitBox: document.getElementById("fitBox"),
      restartBtn: document.getElementById("restartBtn"),
      snapshotBtn: document.getElementById("snapshotBtn"),
      frameImage: document.getElementById("frameImage")
    };
    function setStatus(text) { els.status.textContent = text || ""; }
    function clampFrame(value) {
      const max = Math.max(0, state.frameCount - 1);
      return Math.max(0, Math.min(max, Math.round(Number(value) || 0)));
    }
    function renderDetails(payload) {
      if (!payload || !payload.loaded) {
        els.details.innerHTML = "";
        els.loadPill.textContent = "empty";
        return;
      }
      els.loadPill.textContent = payload.episodeId || "loaded";
      const rows = [
        ["episode", `${payload.episodeId} (${payload.episodeIndex})`],
        ["frames", String(payload.frameCount || 0)],
        ["fps", Number(payload.nominalFps || 0).toFixed(2)],
        ["dataset", payload.datasetRoot || "direct"],
        ["source", payload.sourcePath || ""]
      ];
      for (const camera of payload.cameras || []) {
        rows.push([camera.cameraKey, `rgb ${camera.videoFrameCount || 0}, mask ${camera.maskFrameCount || 0}`]);
      }
      els.details.innerHTML = rows.map(([key, value]) => `<div><span>${escapeHtml(key)}</span><span title="${escapeHtml(value)}">${escapeHtml(value)}</span></div>`).join("");
    }
    function applyPayload(payload) {
      state.view = payload;
      state.loaded = Boolean(payload && payload.loaded);
      if (state.loaded) {
        state.frameCount = Math.max(1, Number(payload.frameCount || 1));
        state.fps = Math.max(1, Number(payload.nominalFps || 15));
        state.frame = clampFrame(state.frame);
        setStatus("Loaded");
      } else {
        state.frameCount = 1;
        state.frame = 0;
        if (payload && payload.error) setStatus(payload.error);
      }
      renderDetails(payload);
      updateChrome();
      renderFrame(true);
    }
    async function loadState() {
      const response = await fetch("/api/state");
      applyPayload(await response.json());
    }
    async function loadPath() {
      const path = els.pathInput.value.trim();
      if (!path) {
        setStatus("Enter a path first");
        return;
      }
      const episodeText = els.episodeInput.value.trim();
      setStatus("Loading...");
      const response = await fetch("/api/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path, episode: episodeText === "" ? null : Number(episodeText) })
      });
      const payload = await response.json();
      if (!response.ok) {
        setStatus(payload.error || "Load failed");
        return;
      }
      state.frame = 0;
      applyPayload(payload);
    }
    function frameUrl() {
      state.cache += 1;
      return `/frame.jpg?i=${state.frame}&mode=${encodeURIComponent(state.mode)}&w=${state.width}&ck=${state.cache}`;
    }
    function renderFrame(force) {
      if (!force && !state.loaded) return;
      els.frameImage.src = frameUrl();
    }
    function applyFitMode() {
      els.frameImage.classList.toggle("actual-size", !state.fit);
    }
    function updateChrome() {
      els.playBtn.textContent = state.playing ? "Pause" : "Play";
      els.timeline.max = String(Math.max(0, state.frameCount - 1));
      els.timeline.value = String(clampFrame(state.frame));
      els.frameLabel.textContent = `${state.frame} / ${Math.max(0, state.frameCount - 1)}`;
      els.modeSelect.value = state.mode;
      els.widthInput.value = String(state.width);
      els.fitBox.checked = Boolean(state.fit);
      applyFitMode();
    }
    function seek(frame) {
      state.frame = clampFrame(frame);
      state.anchorFrame = state.frame;
      state.anchorClock = performance.now();
      updateChrome();
      renderFrame(true);
    }
    function play() {
      if (!state.loaded) return;
      state.playing = true;
      state.anchorFrame = state.frame;
      state.anchorClock = performance.now();
      updateChrome();
    }
    function pause() {
      state.playing = false;
      state.frame = clampFrame(state.frame);
      updateChrome();
    }
    function togglePlay() { state.playing ? pause() : play(); }
    function tick() {
      if (state.playing) {
        const elapsed = (performance.now() - state.anchorClock) / 1000;
        const next = clampFrame(state.anchorFrame + elapsed * state.fps * state.speed);
        if (next !== state.frame) {
          state.frame = next;
          updateChrome();
          renderFrame(true);
        }
        if (state.frame >= state.frameCount - 1) pause();
      }
      requestAnimationFrame(tick);
    }
    function escapeHtml(value) {
      return String(value || "").replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      })[ch]);
    }
    els.loadBtn.addEventListener("click", () => loadPath().catch((err) => setStatus(String(err))));
    els.pathInput.addEventListener("keydown", (event) => { if (event.key === "Enter") loadPath().catch((err) => setStatus(String(err))); });
    els.episodeInput.addEventListener("keydown", (event) => { if (event.key === "Enter") loadPath().catch((err) => setStatus(String(err))); });
    els.playBtn.addEventListener("click", togglePlay);
    els.prevBtn.addEventListener("click", () => seek(state.frame - 1));
    els.nextBtn.addEventListener("click", () => seek(state.frame + 1));
    els.backBtn.addEventListener("click", () => seek(state.frame - 30));
    els.forwardBtn.addEventListener("click", () => seek(state.frame + 30));
    els.restartBtn.addEventListener("click", () => seek(0));
    els.timeline.addEventListener("input", () => seek(Number(els.timeline.value || 0)));
    els.speedSelect.addEventListener("change", () => { state.speed = Number(els.speedSelect.value || 1); state.anchorFrame = state.frame; state.anchorClock = performance.now(); });
    els.modeSelect.addEventListener("change", () => { state.mode = els.modeSelect.value; renderFrame(true); });
    els.widthInput.addEventListener("change", () => { state.width = Math.max(640, Number(els.widthInput.value || 1280)); renderFrame(true); updateChrome(); });
    els.fitBox.addEventListener("change", () => { state.fit = els.fitBox.checked; updateChrome(); });
    els.snapshotBtn.addEventListener("click", () => window.open(`/frame.jpg?i=${state.frame}&mode=${encodeURIComponent(state.mode)}&w=${state.width}`, "_blank"));
    document.addEventListener("keydown", (event) => {
      const tag = event.target && event.target.tagName ? event.target.tagName.toLowerCase() : "";
      if (["input", "select"].includes(tag)) return;
      if (event.key === " ") { event.preventDefault(); togglePlay(); }
      if (event.key === "ArrowLeft") seek(state.frame - 1);
      if (event.key === "ArrowRight") seek(state.frame + 1);
      if (event.key === "Home") seek(0);
      if (event.key === "End") seek(state.frameCount - 1);
    });
    els.pathInput.value = initial.path || "";
    if (initial.episode !== null && initial.episode !== undefined) els.episodeInput.value = String(initial.episode);
    state.mode = initial.mode || "mask";
    state.width = Number(initial.frameWidth || 1280);
    updateChrome();
    loadState().catch((err) => setStatus(String(err)));
    tick();
  </script>
</body>
</html>
"""


def _web_index_html(initial_data: Mapping[str, Any]) -> bytes:
    payload = json.dumps(dict(initial_data), ensure_ascii=True).replace("</", "<\\/")
    return WEB_APP_HTML.replace("__INITIAL_DATA__", payload).encode("utf-8")


def _make_handler(
    registry: EpisodeRegistry,
    initial_data: Mapping[str, Any],
    *,
    default_width: int,
) -> type[BaseHTTPRequestHandler]:
    class WristMaskViewHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format: str, *args: object) -> None:
            return

        def _send_bytes(self, status: int, body: bytes, media_type: str) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", media_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _send_json(self, status: int, payload: object) -> None:
            self._send_bytes(status, _json_bytes(payload), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_bytes(HTTPStatus.OK, _web_index_html(initial_data), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/state":
                self._send_json(HTTPStatus.OK, registry.state_payload())
                return
            if parsed.path == "/frame.jpg":
                self._serve_frame(parsed.query)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/load":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(body.decode("utf-8") or "{}")
                source = str(data.get("path") or "").strip()
                if not source:
                    raise ValueError("path is required.")
                raw_episode = data.get("episode")
                episode = None if raw_episode in (None, "") else int(raw_episode)
                payload = registry.load(source, episode_index=episode)
            except Exception as exc:  # noqa: BLE001
                registry.set_error(str(exc))
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, payload)

        def _serve_frame(self, query: str) -> None:
            params = parse_qs(query)
            try:
                frame_index = int(float(params.get("i", ["0"])[0]))
                mode = str(params.get("mode", ["mask"])[0])
                if mode not in {"mask", "overlay"}:
                    mode = "mask"
                width = int(float(params.get("w", [str(default_width)])[0]))
                width = max(640, min(3200, width))
                body = registry.render_jpeg(frame_index=frame_index, mode=mode, width=width)
            except Exception as exc:  # noqa: BLE001
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return
            self._send_bytes(HTTPStatus.OK, body, "image/jpeg")

    return WristMaskViewHandler


def _bind_web_server(
    host: str,
    port: int,
    handler_class: type[BaseHTTPRequestHandler],
) -> ThreadingHTTPServer:
    ports = [port] if port == 0 else list(range(port, port + 20))
    last_error: OSError | None = None
    for candidate in ports:
        try:
            return ThreadingHTTPServer((host, candidate), handler_class)
        except OSError as exc:
            last_error = exc
    raise RuntimeError(f"could not bind web UI on {host}:{port}: {last_error}")


def run_web_gui(
    *,
    source: str | Path | None,
    episode_index: int | None,
    camera_keys: Sequence[str],
    host: str,
    port: int,
    no_open: bool,
    start_frame: int,
    mode: str,
    frame_width: int,
) -> int:
    registry = EpisodeRegistry(camera_keys=camera_keys)
    if source:
        try:
            registry.load(source, episode_index=episode_index)
        except Exception as exc:  # noqa: BLE001
            registry.set_error(str(exc))

    initial_data = {
        "path": "" if source is None else str(source),
        "episode": episode_index,
        "startFrame": int(start_frame),
        "mode": mode,
        "frameWidth": int(frame_width),
    }
    handler = _make_handler(registry, initial_data, default_width=frame_width)
    server = _bind_web_server(host, port, handler)
    actual_host, actual_port = server.server_address
    url = f"http://{actual_host}:{actual_port}/"
    print(f"Wrist RGB/mask viewer: {url}")
    if not no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        registry.close()
        server.server_close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    camera_keys = _parse_camera_keys(args.camera_keys)
    frame_width = max(640, int(args.frame_width))
    if args.snapshot is not None:
        if not args.path:
            raise SystemExit("--snapshot requires PATH.")
        view = load_episode_view(args.path, episode_index=args.episode, camera_keys=camera_keys)
        try:
            frame = compose_episode_frame(view, frame_index=int(args.frame), mode=str(args.mode), width=frame_width)
            output = args.snapshot.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), frame):
                raise RuntimeError(f"failed to write snapshot: {output}")
            print(output)
        finally:
            view.close()
        return 0

    return run_web_gui(
        source=args.path,
        episode_index=args.episode,
        camera_keys=camera_keys,
        host=str(args.host),
        port=int(args.port),
        no_open=bool(args.no_open),
        start_frame=int(args.start_frame),
        mode=str(args.mode),
        frame_width=frame_width,
    )


if __name__ == "__main__":
    raise SystemExit(main())
