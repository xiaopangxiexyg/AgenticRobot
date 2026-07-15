from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .core import WRIST_DEPTH_MASK_FORMAT, unpack_binary_mask


DEFAULT_WRIST_CAMERA_KEYS = ("left_wrist", "right_wrist")


@dataclass(frozen=True)
class MaskStream:
    label: str
    stream_path: Path
    height: int
    width: int
    frame_count: int
    packed_bytes_per_frame: int
    manifest_path: Path | None = None
    frames: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class PreviewResult:
    output_path: Path
    frame_count: int
    fps: float
    labels: tuple[str, ...]
    size: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "frame_count": int(self.frame_count),
            "fps": float(self.fps),
            "labels": list(self.labels),
            "size": [int(self.size[0]), int(self.size[1])],
        }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _parse_shape(shape: str | Sequence[int] | None) -> tuple[int, int] | None:
    if shape is None:
        return None
    if isinstance(shape, str):
        text = shape.lower().replace(",", "x").strip()
        parts = [part for part in text.split("x") if part]
        if len(parts) != 2:
            raise ValueError(f"shape must be HxW, got {shape!r}")
        height, width = int(parts[0]), int(parts[1])
    else:
        if len(shape) != 2:
            raise ValueError(f"shape must have two values, got {shape!r}")
        height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"shape must be positive, got {(height, width)!r}")
    return height, width


def _resolve_stream_path(manifest_path: Path, stream_path: str, *, root_hint: Path | None) -> Path:
    candidate = Path(stream_path).expanduser()
    if candidate.is_absolute():
        return candidate
    candidates = [manifest_path.parent / candidate]
    if root_hint is not None:
        candidates.append(root_hint / candidate)
    candidates.append(candidate)
    for path in candidates:
        if path.exists():
            return path.resolve()
    return (root_hint / candidate).resolve() if root_hint is not None else (manifest_path.parent / candidate).resolve()


def _infer_root_from_manifest_relative_dir(manifest_path: Path, payload: Mapping[str, Any]) -> Path | None:
    relative_dir = payload.get("relative_dir")
    if not isinstance(relative_dir, str) or not relative_dir:
        return None
    rel_path = Path(relative_dir)
    if rel_path.is_absolute():
        return None
    rel_parts = rel_path.parts
    parent_parts = manifest_path.parent.parts
    if len(parent_parts) < len(rel_parts) or parent_parts[-len(rel_parts):] != rel_parts:
        return None
    root_parts = parent_parts[: -len(rel_parts)]
    if not root_parts:
        return Path(".").resolve()
    return Path(*root_parts)


def _guess_manifest_for_stream(stream_path: Path) -> Path | None:
    candidates = (
        stream_path.with_name("manifest.json"),
        stream_path.with_name("mask_manifest.json"),
        stream_path.parent / "manifest.json",
        stream_path.parent / "mask_manifest.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_mask_stream(
    path: str | Path,
    *,
    root_hint: str | Path | None = None,
    label: str | None = None,
    shape: str | Sequence[int] | None = None,
    frame_count: int | None = None,
) -> MaskStream:
    """Load a bit-packed wrist depth mask stream from a manifest or raw .bin file."""

    source_path = Path(path).expanduser().resolve()
    root = None if root_hint is None else Path(root_hint).expanduser().resolve()
    shape_hw = _parse_shape(shape)

    manifest_path: Path | None = source_path if source_path.suffix.lower() == ".json" else _guess_manifest_for_stream(source_path)
    if manifest_path is not None and manifest_path.exists():
        payload = _read_json(manifest_path)
        if payload.get("format") != WRIST_DEPTH_MASK_FORMAT:
            raise ValueError(f"{manifest_path} is not a wrist depth mask manifest.")
        if root is None:
            root = _infer_root_from_manifest_relative_dir(manifest_path, payload)
        stream = payload.get("stream")
        if not isinstance(stream, Mapping):
            raise ValueError(f"manifest missing stream object: {manifest_path}")
        stream_shape = stream.get("shape")
        if not isinstance(stream_shape, Sequence) or isinstance(stream_shape, str) or len(stream_shape) != 2:
            raise ValueError(f"manifest stream.shape must be [height, width]: {manifest_path}")
        height, width = int(stream_shape[0]), int(stream_shape[1])
        packed_bytes = int(stream.get("packed_bytes_per_frame") or ((height * width + 7) // 8))
        count = int(stream.get("frame_count") or 0)
        stream_value = stream.get("path")
        if not isinstance(stream_value, str) or not stream_value:
            raise ValueError(f"manifest stream.path missing: {manifest_path}")
        stream_path = _resolve_stream_path(manifest_path, stream_value, root_hint=root)
        if not stream_path.exists() and source_path.suffix.lower() != ".json" and source_path.exists():
            stream_path = source_path
        frames = payload.get("frames")
        return MaskStream(
            label=str(label or payload.get("camera_key") or source_path.stem),
            stream_path=stream_path,
            height=height,
            width=width,
            frame_count=count,
            packed_bytes_per_frame=packed_bytes,
            manifest_path=manifest_path,
            frames=tuple(frames) if isinstance(frames, list) else (),
        )

    if shape_hw is None:
        raise ValueError(
            f"{source_path} is a raw mask stream without a readable sibling manifest; pass --shape HxW."
        )
    height, width = shape_hw
    packed_bytes = int((height * width + 7) // 8)
    size = source_path.stat().st_size
    if size % packed_bytes != 0:
        raise ValueError(
            f"{source_path} size={size} is not divisible by packed frame size={packed_bytes}."
        )
    inferred_count = int(size // packed_bytes)
    return MaskStream(
        label=str(label or source_path.stem),
        stream_path=source_path,
        height=height,
        width=width,
        frame_count=int(frame_count or inferred_count),
        packed_bytes_per_frame=packed_bytes,
        manifest_path=None,
    )


def find_session_mask_manifests(
    session_root: str | Path,
    *,
    episode: int | None = None,
    camera_keys: Sequence[str] = DEFAULT_WRIST_CAMERA_KEYS,
) -> list[Path]:
    root = Path(session_root).expanduser().resolve()
    episodes_path = root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"session does not contain meta/episodes.jsonl: {root}")
    records = _read_jsonl(episodes_path)
    if not records:
        raise ValueError(f"no episode records in {episodes_path}")

    selected: Mapping[str, Any] | None = None
    if episode is None:
        selected = records[0]
    else:
        for record in records:
            try:
                if int(record.get("episode_index")) == int(episode):
                    selected = record
                    break
            except (TypeError, ValueError):
                continue
    if selected is None:
        raise ValueError(f"episode {episode} not found in {episodes_path}")

    paths = selected.get("wrist_depth_mask_paths")
    if not isinstance(paths, Mapping) or not paths:
        raise ValueError(f"episode {selected.get('episode_index')} has no wrist_depth_mask_paths")
    manifests: list[Path] = []
    missing: list[str] = []
    for camera_key in camera_keys:
        value = paths.get(camera_key)
        if not isinstance(value, str) or not value:
            missing.append(str(camera_key))
            continue
        manifest = Path(value)
        manifests.append((manifest if manifest.is_absolute() else root / manifest).resolve())
    if missing:
        raise ValueError(f"episode {selected.get('episode_index')} missing mask paths for: {missing}")
    return manifests


def iter_mask_frames(stream: MaskStream, *, max_frames: int = 0) -> Iterable[np.ndarray]:
    count = int(stream.frame_count)
    if int(max_frames) > 0:
        count = min(count, int(max_frames))
    with stream.stream_path.open("rb") as handle:
        for _index in range(count):
            data = handle.read(stream.packed_bytes_per_frame)
            if len(data) < stream.packed_bytes_per_frame:
                break
            yield unpack_binary_mask(data, height=stream.height, width=stream.width)


def _foreground_fraction(stream: MaskStream, frame_index: int, mask: np.ndarray) -> float:
    if 0 <= frame_index < len(stream.frames):
        raw = stream.frames[frame_index].get("foreground_fraction")
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    return 0.0 if mask.size <= 0 else float(np.count_nonzero(mask)) / float(mask.size)


def _mask_panel(
    mask: np.ndarray,
    *,
    label: str,
    frame_index: int,
    frame_count: int,
    foreground_fraction: float,
    scale: float,
) -> np.ndarray:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("depth mask visualization requires opencv-python/cv2.") from exc

    image = (np.asarray(mask, dtype=np.uint8) * 255)
    panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if float(scale) != 1.0:
        interpolation = cv2.INTER_NEAREST if float(scale) > 1.0 else cv2.INTER_AREA
        panel = cv2.resize(panel, None, fx=float(scale), fy=float(scale), interpolation=interpolation)
    title_h = 34
    canvas = np.zeros((panel.shape[0] + title_h, panel.shape[1], 3), dtype=np.uint8)
    canvas[title_h:, :, :] = panel
    text = f"{label}  {frame_index + 1}/{frame_count}  fg={foreground_fraction:.4f}"
    cv2.putText(canvas, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def compose_preview_frame(
    masks: Sequence[np.ndarray],
    *,
    streams: Sequence[MaskStream],
    frame_index: int,
    scale: float = 1.0,
) -> np.ndarray:
    panels = [
        _mask_panel(
            mask,
            label=streams[index].label,
            frame_index=frame_index,
            frame_count=streams[index].frame_count,
            foreground_fraction=_foreground_fraction(streams[index], frame_index, mask),
            scale=float(scale),
        )
        for index, mask in enumerate(masks)
    ]
    max_height = max(panel.shape[0] for panel in panels)
    padded: list[np.ndarray] = []
    for panel in panels:
        if panel.shape[0] == max_height:
            padded.append(panel)
            continue
        pad = np.zeros((max_height - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
        padded.append(np.vstack([panel, pad]))
    return np.hstack(padded)


def write_mask_preview_video(
    streams: Sequence[MaskStream],
    output_path: str | Path,
    *,
    fps: float = 15.0,
    max_frames: int = 0,
    scale: float = 1.0,
    codec: str = "mp4v",
    overwrite: bool = False,
) -> PreviewResult:
    if not streams:
        raise ValueError("at least one mask stream is required.")
    if float(fps) <= 0.0:
        raise ValueError("fps must be positive.")
    if float(scale) <= 0.0:
        raise ValueError("scale must be positive.")

    out = Path(output_path).expanduser().resolve()
    if out.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame_count = min(int(stream.frame_count) for stream in streams)
    if int(max_frames) > 0:
        frame_count = min(frame_count, int(max_frames))
    if frame_count <= 0:
        raise ValueError("mask streams do not contain any frames.")

    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("depth mask visualization requires opencv-python/cv2.") from exc

    iterators = [iter_mask_frames(stream, max_frames=frame_count) for stream in streams]
    writer = None
    written = 0
    try:
        for frame_index in range(frame_count):
            masks = [next(iterator) for iterator in iterators]
            frame = compose_preview_frame(masks, streams=streams, frame_index=frame_index, scale=float(scale))
            if writer is None:
                size = (int(frame.shape[1]), int(frame.shape[0]))
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*str(codec)), float(fps), size)
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open VideoWriter for {out}")
            writer.write(frame)
            written += 1
    finally:
        if writer is not None:
            writer.release()
    return PreviewResult(
        output_path=out,
        frame_count=written,
        fps=float(fps),
        labels=tuple(stream.label for stream in streams),
        size=size if writer is not None else (0, 0),
    )
