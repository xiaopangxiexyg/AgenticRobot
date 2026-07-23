#!/usr/bin/env python3
"""Visualize the four tactile parts of a recorded DM-Tac episode."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import sys
import threading
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlparse


def _ensure_repo_root_on_path() -> None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "dataset_contracts" / "tactile.py").is_file():
            text = str(parent)
            if text not in sys.path:
                sys.path.insert(0, text)
            return


_ensure_repo_root_on_path()

import cv2
import numpy as np

from dataset_contracts.tactile import ChunkedTactileStream, tactile_stream_compression


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8777
DEFAULT_WIDTH = 1280
SIDES = ("left", "right")
TACTILE_SLOTS = ("inner", "outer")
# Viewed from the audience/operator side: the two left pads stay on the left,
# the two right pads stay on the right, and the inner pads meet at the center.
PHYSICAL_TACTILE_ORDER = (
    ("left", "outer"),
    ("left", "inner"),
    ("right", "inner"),
    ("right", "outer"),
)
CORE_VISUAL_STYLES = (
    "classic",
    "neon",
    "neon_focus",
    "neon_contour",
    "components",
    "contour",
    "vectors",
    "surface",
    "surface_left",
    "surface_front",
    "split",
)
CANDIDATE_VISUAL_STYLES = (
    "neon_amber",
    "neon_mint",
    "neon_viridis_coral",
    "neon_cividis_cyan",
    "neon_ocean_amber",
    "neon_focus_viridis",
    "neon_contour_viridis",
    "surface_viridis",
    "surface_viridis_left",
    "surface_viridis_front",
    "surface_turbo",
    "surface_turbo_left",
    "surface_turbo_front",
    "surface_cividis",
    "surface_cividis_left",
    "surface_cividis_front",
    "surface_ocean",
    "surface_ocean_left",
    "surface_ocean_front",
    "split_viridis",
    "surface_turbo_hero",
    "surface_turbo_hero_left",
    "surface_turbo_high",
    "surface_turbo_top",
    "surface_turbo_profile",
    "surface_turbo_hero_cyan",
    "surface_turbo_hero_amber",
    "surface_turbo_top_cyan",
)
VISUAL_STYLES = (*CORE_VISUAL_STYLES, *CANDIDATE_VISUAL_STYLES)
COP_NOISE_FLOOR_RATIO = 0.04
COP_PEAK_GATE_RATIO = 0.10
COP_MASS_GATE_RATIO = 0.20


class MissingDerivedForceError(ValueError):
    """Raised when force visualization was requested before offline solving."""

    def __init__(self, dataset_root: Path, episode_index: int) -> None:
        command = (
            "python utils/portable_vis/DMTAC_vis/dmtac_viewer.py "
            f'"{dataset_root}" --episode {episode_index} --tactile-source force '
            "--derive-force --force-backend cpu "
            "--force-sdk-root assets/DM_Tac_SDK_Publish_1.2.10"
        )
        super().__init__(
            f"Episode {episode_index} has raw tactile data but no derived "
            "distributed_force streams. Generate them once with:\n\n"
            f"  {command}\n\n"
            "To inspect the camera-like sensor images without solving force, use "
            "--tactile-source raw."
        )
        self.dataset_root = dataset_root
        self.episode_index = episode_index


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _episode_index_from_path(path: Path) -> int | None:
    match = re.search(r"episode_(\d+)", str(path))
    return int(match.group(1)) if match else None


def _dataset_root_from_path(path: Path) -> Path | None:
    base = path if path.is_dir() else path.parent
    for candidate in (base, *base.parents):
        if (candidate / "meta" / "episodes.jsonl").is_file():
            return candidate.resolve()
    parts = path.resolve().parts
    for marker in ("data", "tactile", "meta"):
        if marker in parts and parts.index(marker) > 0:
            return Path(*parts[: parts.index(marker)]).resolve()
    return None


def _resolve_dataset_path(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _chunk_size(root: Path) -> int:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return 1000
    try:
        return max(1, int(_read_json(info_path).get("chunk_size") or 1000))
    except (TypeError, ValueError):
        return 1000


def _episode_record(root: Path, episode_index: int) -> dict[str, Any]:
    for record in _read_jsonl(root / "meta" / "episodes.jsonl"):
        if int(record.get("episode_index", -1)) == int(episode_index):
            return record
    chunk = int(episode_index) // _chunk_size(root)
    return {
        "episode_index": int(episode_index),
        "episode_id": f"episode_{int(episode_index):06d}",
        "parquet_path": f"data/chunk-{chunk:03d}/episode_{int(episode_index):06d}.parquet",
        "tactile_paths": {
            side: (
                f"tactile/observation.hand.{side}.tactile/chunk-{chunk:03d}/"
                f"episode_{int(episode_index):06d}/manifest.json"
            )
            for side in SIDES
        },
    }


class _ArrayStream:
    def __init__(self, values: Sequence[np.ndarray]):
        self.values = [np.asarray(value) for value in values]

    def frame(self, index: int) -> np.ndarray | None:
        return self.values[index] if 0 <= index < len(self.values) else None


class _RawStream:
    def __init__(self, *, root: Path, spec: Mapping[str, Any]):
        self.shape = tuple(int(value) for value in spec.get("shape", ()) or ())
        self.frame_count = int(spec.get("frame_count") or 0)
        self.dtype = np.dtype(str(spec.get("dtype") or "float32"))
        path = _resolve_dataset_path(root, spec.get("path"))
        if not path.is_file():
            raise FileNotFoundError(f"Tactile stream does not exist: {path}")
        self.array = np.memmap(
            path, dtype=self.dtype, mode="r", shape=(self.frame_count, *self.shape)
        )

    def frame(self, index: int) -> np.ndarray | None:
        return self.array[index] if 0 <= index < self.frame_count else None


@dataclass
class TactileChannel:
    side: str
    sensor_key: str
    field_name: str
    dtype: str
    shape: tuple[int, ...]
    stream: Any
    sample_indices: tuple[int, ...]
    tactile_indices: tuple[int, ...]
    tactile_slot: str = "unknown"
    slot_source: str = "unresolved"
    change_series: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    @property
    def label(self) -> str:
        # OpenCV's built-in font is ASCII-only; keep snapshots readable without
        # requiring an additional font package.
        slot = self.tactile_slot.upper() if self.tactile_slot in TACTILE_SLOTS else "SENSOR"
        return f"{self.side.upper()} {slot} / {self.sensor_key}"

    @property
    def orientation_label(self) -> str:
        if self.tactile_slot == "outer":
            return "portrait / outer CW + vertical mirror"
        if self.tactile_slot == "inner":
            return "portrait / inner CCW"
        return "native orientation"

    def tactile_index_for_sample(self, sample_index: int) -> int | None:
        if not self.sample_indices:
            return sample_index
        position = bisect.bisect_right(self.sample_indices, int(sample_index)) - 1
        return self.tactile_indices[position] if position >= 0 else None

    def frame(self, sample_index: int) -> np.ndarray | None:
        index = self.tactile_index_for_sample(sample_index)
        return None if index is None else self.stream.frame(index)


@dataclass
class TactileEpisode:
    dataset_root: Path
    episode_index: int
    schema: str
    frame_count: int
    channels: list[TactileChannel]
    source_paths: tuple[Path, ...]
    nominal_fps: float = 15.0
    tactile_source: str = "unknown"
    force_scales: dict[str, tuple[float, float]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            "episode_index": self.episode_index,
            "schema": self.schema,
            "frame_count": self.frame_count,
            "fps": self.nominal_fps,
            "tactile_source": self.tactile_source,
            "source_paths": [str(path) for path in self.source_paths],
            "force_scales": {
                mode: {"normal_fz": values[0], "tangent_xy": values[1]}
                for mode, values in self.force_scales.items()
            },
            "channels": [
                {
                    "side": channel.side,
                    "sensor_key": channel.sensor_key,
                    "tactile_slot": channel.tactile_slot,
                    "slot_source": channel.slot_source,
                    "orientation": channel.orientation_label,
                    "field": channel.field_name,
                    "dtype": channel.dtype,
                    "shape": list(channel.shape),
                }
                for channel in self.channels
            ],
        }


@dataclass(frozen=True)
class ForceMetrics:
    """Presentation metrics computed from the same 20 x 30 force grid as the view."""

    mean_xyz: tuple[float, float, float]
    peak_force: float
    normal_mean: float
    normal_peak: float
    shear_mean: float
    shear_peak: float
    active_percent: float
    center_of_pressure: tuple[float, float] | None
    contact_strength: float


@dataclass(frozen=True)
class VideoExport:
    path: Path
    style: str
    mode: str
    frame_count: int
    fps: float
    size: tuple[int, int]


@dataclass(frozen=True)
class StyleProfile:
    family: str
    palette: str = "magma"
    arrow_theme: str = "cyan"
    surface_view: str = "right"


@dataclass(frozen=True)
class VisualizationProfile:
    """Locked end-user presentation settings for reproducible exports."""

    name: str
    version: str
    tactile_source: str
    style: str
    mode: str
    width: int
    temporal_smoothing: float


STANDARD_VISUALIZATION_PROFILE = VisualizationProfile(
    name="standard",
    version="parallel_jaw_top_v1",
    tactile_source="force",
    style="surface_turbo_top",
    mode="current",
    width=2560,
    temporal_smoothing=0.18,
)
VISUALIZATION_PROFILES: dict[str, VisualizationProfile] = {
    STANDARD_VISUALIZATION_PROFILE.name: STANDARD_VISUALIZATION_PROFILE,
}


_STYLE_PROFILES: dict[str, StyleProfile] = {
    "classic": StyleProfile("classic", "turbo", "legacy"),
    "neon": StyleProfile("neon", "magma", "cyan"),
    "neon_focus": StyleProfile("neon_focus", "magma", "cyan"),
    "neon_contour": StyleProfile("neon_contour", "magma", "cyan"),
    "components": StyleProfile("components", "magma", "cyan"),
    "contour": StyleProfile("contour", "inferno", "cyan"),
    "vectors": StyleProfile("vectors", "magma", "cyan"),
    "surface": StyleProfile("surface", "magma", "cyan", "right"),
    "surface_left": StyleProfile("surface", "magma", "cyan", "left"),
    "surface_front": StyleProfile("surface", "magma", "cyan", "front"),
    "split": StyleProfile("split", "magma", "cyan", "right"),
    "neon_amber": StyleProfile("neon", "magma", "amber"),
    "neon_mint": StyleProfile("neon", "magma", "mint"),
    "neon_viridis_coral": StyleProfile("neon", "viridis", "coral"),
    "neon_cividis_cyan": StyleProfile("neon", "cividis", "cyan"),
    "neon_ocean_amber": StyleProfile("neon", "ocean", "amber"),
    "neon_focus_viridis": StyleProfile("neon_focus", "viridis", "amber"),
    "neon_contour_viridis": StyleProfile("neon_contour", "viridis", "coral"),
    "surface_viridis": StyleProfile("surface", "viridis", "amber", "right"),
    "surface_viridis_left": StyleProfile("surface", "viridis", "amber", "left"),
    "surface_viridis_front": StyleProfile("surface", "viridis", "amber", "front"),
    "surface_turbo": StyleProfile("surface", "turbo", "mint", "right"),
    "surface_turbo_left": StyleProfile("surface", "turbo", "mint", "left"),
    "surface_turbo_front": StyleProfile("surface", "turbo", "mint", "front"),
    "surface_cividis": StyleProfile("surface", "cividis", "cyan", "right"),
    "surface_cividis_left": StyleProfile("surface", "cividis", "cyan", "left"),
    "surface_cividis_front": StyleProfile("surface", "cividis", "cyan", "front"),
    "surface_ocean": StyleProfile("surface", "ocean", "coral", "right"),
    "surface_ocean_left": StyleProfile("surface", "ocean", "coral", "left"),
    "surface_ocean_front": StyleProfile("surface", "ocean", "coral", "front"),
    "split_viridis": StyleProfile("split", "viridis", "amber", "right"),
    "surface_turbo_hero": StyleProfile("surface", "turbo", "mint", "hero"),
    "surface_turbo_hero_left": StyleProfile("surface", "turbo", "mint", "hero_left"),
    "surface_turbo_high": StyleProfile("surface", "turbo", "mint", "high"),
    "surface_turbo_top": StyleProfile("surface", "turbo", "mint", "top"),
    "surface_turbo_profile": StyleProfile("surface", "turbo", "mint", "profile"),
    "surface_turbo_hero_cyan": StyleProfile("surface", "turbo", "cyan", "hero"),
    "surface_turbo_hero_amber": StyleProfile("surface", "turbo", "amber", "hero"),
    "surface_turbo_top_cyan": StyleProfile("surface", "turbo", "cyan", "top"),
}

# A deliberately curated review set: planar current/delta treatments, focused
# contact, explicit components, multiple palettes/views, 3D delta, and two
# combined views. It is not a Cartesian product, so each item adds information.
SHOWCASE_30: tuple[tuple[str, str], ...] = (
    ("neon", "current"),
    ("neon", "delta"),
    ("neon_amber", "current"),
    ("neon_amber", "delta"),
    ("neon_mint", "current"),
    ("neon_viridis_coral", "current"),
    ("neon_viridis_coral", "delta"),
    ("neon_cividis_cyan", "current"),
    ("neon_ocean_amber", "current"),
    ("neon_focus", "delta"),
    ("neon_focus_viridis", "delta"),
    ("neon_contour", "current"),
    ("neon_contour_viridis", "current"),
    ("components", "current"),
    ("surface", "current"),
    ("surface_left", "current"),
    ("surface_front", "current"),
    ("surface_viridis", "current"),
    ("surface_viridis_left", "current"),
    ("surface_viridis_front", "current"),
    ("surface_turbo", "current"),
    ("surface_turbo_left", "current"),
    ("surface_turbo_front", "current"),
    ("surface_cividis", "current"),
    ("surface_ocean", "current"),
    ("surface", "delta"),
    ("surface_viridis", "delta"),
    ("surface_turbo", "delta"),
    ("split", "current"),
    ("split_viridis", "current"),
)

SHOWCASE_3D_10: tuple[tuple[str, str, float], ...] = (
    ("surface_turbo_hero", "current", 0.18),
    ("surface_turbo_hero_left", "current", 0.18),
    ("surface_turbo_high", "current", 0.18),
    ("surface_turbo_top", "current", 0.18),
    ("surface_turbo_front", "current", 0.18),
    ("surface_turbo_profile", "current", 0.18),
    ("surface_turbo", "current", 0.18),
    ("surface_turbo_hero_cyan", "current", 0.18),
    ("surface_turbo_hero_amber", "current", 0.18),
    ("surface_turbo_top_cyan", "current", 0.26),
)


def _slot_hint_from_sensor_key(sensor_key: str) -> str | None:
    """Recognize descriptive keys without guessing from opaque serial numbers."""

    tokens = tuple(token for token in re.split(r"[^a-z0-9]+", sensor_key.lower()) if token)
    if any(token in {"inner", "in"} for token in tokens):
        return "inner"
    if any(token in {"outer", "out"} for token in tokens):
        return "outer"
    return None


def _configured_tactile_slots() -> dict[tuple[str, str], str]:
    """Read the canonical assignment as a fallback for opaque recorded ids."""

    assignment_path = (
        Path(__file__).resolve().parents[3]
        / "teleop_runtime/configs/hardware/dmtac/latest.json"
    )
    if not assignment_path.is_file():
        return {}
    try:
        assignments = _read_json(assignment_path).get("assignments")
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(assignments, Mapping):
        return {}
    result: dict[tuple[str, str], str] = {}
    for side in SIDES:
        side_config = assignments.get(side)
        tactile = side_config.get("tactile") if isinstance(side_config, Mapping) else None
        if not isinstance(tactile, Mapping):
            continue
        for slot in TACTILE_SLOTS:
            item = tactile.get(slot)
            if not isinstance(item, Mapping):
                continue
            sensor_key = item.get("dev_id")
            if sensor_key:
                result[(side, str(sensor_key))] = slot
    return result


def _assign_physical_slots(channels: Sequence[TactileChannel]) -> None:
    configured = _configured_tactile_slots()
    for channel in channels:
        hint = _slot_hint_from_sensor_key(channel.sensor_key)
        if hint is not None:
            channel.tactile_slot = hint
            channel.slot_source = "sensor_key"
            continue
        configured_slot = configured.get((channel.side, channel.sensor_key))
        if configured_slot is not None:
            channel.tactile_slot = configured_slot
            channel.slot_source = "dmtac_assignment"


def _physical_channel_key(channel: TactileChannel) -> tuple[int, int]:
    try:
        return PHYSICAL_TACTILE_ORDER.index((channel.side, channel.tactile_slot)), 0
    except ValueError:
        side_index = SIDES.index(channel.side) if channel.side in SIDES else len(SIDES)
        return len(PHYSICAL_TACTILE_ORDER) + side_index, 1


def _physical_channel_slots(episode: TactileEpisode) -> list[TactileChannel | None]:
    slots: list[TactileChannel | None] = [None] * len(PHYSICAL_TACTILE_ORDER)
    remaining: list[TactileChannel] = []
    for channel in episode.channels:
        try:
            index = PHYSICAL_TACTILE_ORDER.index((channel.side, channel.tactile_slot))
        except ValueError:
            remaining.append(channel)
            continue
        if slots[index] is None:
            slots[index] = channel
        else:
            remaining.append(channel)
    for channel in remaining:
        candidates = [
            index
            for index, (side, _) in enumerate(PHYSICAL_TACTILE_ORDER)
            if slots[index] is None and side == channel.side
        ]
        if not candidates:
            candidates = [index for index, value in enumerate(slots) if value is None]
        if candidates:
            slots[candidates[0]] = channel
    return slots


def _frame_mapping(manifest: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    pairs: list[tuple[int, int]] = []
    for fallback, record in enumerate(manifest.get("frames", ()) or ()):
        if not isinstance(record, Mapping):
            continue
        sample = int(record.get("sample_frame_index", fallback))
        tactile = int(record.get("tactile_frame_index", fallback))
        pairs.append((sample, tactile))
    pairs.sort()
    return tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs)


def _channels_from_manifest(root: Path, path: Path) -> tuple[list[TactileChannel], int]:
    manifest = _read_json(path)
    if str(manifest.get("format")) != "binary_stream_v1":
        raise ValueError(f"Unsupported tactile manifest format: {path}")
    streams = manifest.get("streams")
    if not isinstance(streams, Mapping):
        raise ValueError(f"Tactile manifest has no streams mapping: {path}")
    side = str(manifest.get("side") or "unknown")
    sample_indices, tactile_indices = _frame_mapping(manifest)
    frames = manifest.get("frames") or []
    first_record = frames[0] if frames and isinstance(frames[0], Mapping) else {}
    fields = first_record.get("field_by_key") if isinstance(first_record, Mapping) else {}
    keys = [str(key) for key in manifest.get("sensor_keys", ()) or streams.keys()]
    channels: list[TactileChannel] = []
    for key in keys:
        spec = streams.get(key)
        if not isinstance(spec, Mapping):
            continue
        compression = tactile_stream_compression(spec)
        stream: Any
        if compression == "none":
            stream = _RawStream(root=root, spec=spec)
        else:
            stream = ChunkedTactileStream(dataset_root=root, spec=spec)
        channels.append(
            TactileChannel(
                side=side,
                sensor_key=key,
                field_name=str(fields.get(key, "unknown")) if isinstance(fields, Mapping) else "unknown",
                dtype=str(spec.get("dtype") or "float32"),
                shape=tuple(int(value) for value in spec.get("shape", ()) or ()),
                stream=stream,
                sample_indices=sample_indices,
                tactile_indices=tactile_indices,
            )
        )
    frame_count = int(manifest.get("frame_count") or 0)
    if sample_indices:
        frame_count = max(frame_count, sample_indices[-1] + 1)
    return channels, frame_count


def _safe_json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _inline_channels(parquet_path: Path) -> tuple[list[TactileChannel], int]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Inline tactile parquet requires pyarrow. External binary_stream_v1 data does not."
        ) from exc
    rows = pq.read_table(parquet_path).to_pylist()
    channels: list[TactileChannel] = []
    for side in SIDES:
        prefix = f"observation.hand.{side}.tactile"
        keys: list[str] = []
        shapes: dict[str, Any] = {}
        field_name = "inline_numeric"
        first_vector_size = 0
        raw_shape: tuple[int, ...] = ()
        for row in rows:
            keys = [str(value) for value in row.get(f"{prefix}_sensor_keys", ()) or ()]
            if not keys:
                keys = [str(value) for value in row.get(f"{prefix}_regions", ()) or ()]
            shapes = _safe_json_mapping(row.get(f"{prefix}_shapes_json"))
            vector_value = row.get(prefix)
            if vector_value is not None:
                first_vector_size = np.asarray(vector_value).size
                raw_shape = tuple(
                    int(value)
                    for value in row.get(f"{prefix}_raw_shape_per_sensor", ()) or ()
                )
                fields = [str(value) for value in row.get(f"{prefix}_fields", ()) or ()]
                if fields:
                    field_name = ",".join(fields)
                break
        if first_vector_size <= 0:
            continue
        if not keys:
            sensor_size = math.prod(raw_shape) if raw_shape else 0
            sensor_count = first_vector_size // sensor_size if sensor_size and first_vector_size % sensor_size == 0 else 1
            keys = [f"{side}_sensor_{index}" for index in range(sensor_count)]
        if not shapes:
            if raw_shape and math.prod(raw_shape) * len(keys) == first_vector_size:
                shapes = {key: raw_shape for key in keys}
            elif first_vector_size % len(keys) == 0:
                shapes = {key: (first_vector_size // len(keys),) for key in keys}
        values_by_key: dict[str, list[np.ndarray]] = {key: [] for key in keys}
        for row in rows:
            vector = np.asarray(row.get(prefix) if row.get(prefix) is not None else [], dtype=np.float32).reshape(-1)
            offset = 0
            for key in keys:
                shape = tuple(int(value) for value in shapes.get(key, ()) or ())
                size = math.prod(shape) if shape else 0
                if size <= 0 or offset + size > vector.size:
                    values_by_key[key].append(np.full(shape or (1,), np.nan, dtype=np.float32))
                else:
                    values_by_key[key].append(vector[offset : offset + size].reshape(shape))
                offset += size
        for key in keys:
            values = values_by_key[key]
            shape = tuple(values[0].shape) if values else ()
            channels.append(
                TactileChannel(
                    side=side,
                    sensor_key=key,
                    field_name=field_name,
                    dtype="float32",
                    shape=shape,
                    stream=_ArrayStream(values),
                    sample_indices=tuple(range(len(values))),
                    tactile_indices=tuple(range(len(values))),
                )
            )
    return channels, len(rows)


def _schema_from_root(root: Path, record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    for source in (metadata, _read_json(root / "meta" / "info.json") if (root / "meta" / "info.json").is_file() else {}):
        value = source.get("dexteleop_dataset_schema") or source.get("schema")
        if value:
            return str(value)
    return "unknown/legacy"


def _derived_force_manifest_paths(root: Path, episode_index: int) -> list[Path]:
    episode_id = f"episode_{int(episode_index):06d}"
    paths = list(
        (root / "tactile_force").glob(
            f"observation.hand.*.distributed_force/chunk-*/{episode_id}/manifest.json"
        )
    )

    def order(path: Path) -> tuple[int, str]:
        text = path.as_posix()
        side_order = next(
            (index for index, side in enumerate(SIDES) if f"hand.{side}." in text),
            len(SIDES),
        )
        return side_order, text

    return sorted((path.resolve() for path in paths if path.is_file()), key=order)


def _recorded_tactile_manifest_paths(
    root: Path,
    record: Mapping[str, Any],
) -> list[Path]:
    tactile_paths = record.get("tactile_paths")
    paths: list[Path] = []
    if isinstance(tactile_paths, Mapping):
        paths.extend(
            _resolve_dataset_path(root, value) for value in tactile_paths.values() if value
        )
    return list(dict.fromkeys(path for path in paths if path.is_file()))


def load_tactile_episode(
    path: str | Path,
    *,
    episode: int | None = None,
    tactile_source: str = "auto",
) -> TactileEpisode:
    """Resolve a dataset, parquet, manifest, or tactile episode directory."""

    input_path = Path(path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    root = _dataset_root_from_path(input_path)
    if root is None:
        raise ValueError(f"Could not locate the dataset root for {input_path}")
    source_choice = str(tactile_source).strip().lower()
    if source_choice not in {"auto", "raw", "force"}:
        raise ValueError("tactile_source must be auto, raw, or force")
    episode_index = episode if episode is not None else _episode_index_from_path(input_path)
    if episode_index is None:
        records = _read_jsonl(root / "meta" / "episodes.jsonl")
        if len(records) == 1:
            episode_index = int(records[0].get("episode_index", 0))
        else:
            available = [int(row.get("episode_index", -1)) for row in records]
            raise ValueError(f"--episode is required; available episodes: {available}")
    record = _episode_record(root, episode_index)
    manifest_paths: list[Path] = []
    selected_source = source_choice
    is_direct_manifest = input_path.name == "manifest.json" and any(
        part in {"tactile", "tactile_force"} for part in input_path.parts
    )
    if is_direct_manifest and source_choice == "auto":
        manifest_paths = [input_path]
        selected_source = "force" if "tactile_force" in input_path.parts else "raw"
    elif source_choice in {"auto", "force"}:
        manifest_paths = _derived_force_manifest_paths(root, episode_index)
        if manifest_paths:
            selected_source = "force"
    if not manifest_paths and source_choice in {"auto", "raw"}:
        manifest_paths = _recorded_tactile_manifest_paths(root, record)
        selected_source = "raw"
    if not manifest_paths and source_choice == "force":
        raise MissingDerivedForceError(root, episode_index)
    manifest_paths = list(dict.fromkeys(path for path in manifest_paths if path.is_file()))
    channels: list[TactileChannel] = []
    detected_frames = 0
    for manifest_path in manifest_paths:
        loaded, count = _channels_from_manifest(root, manifest_path)
        channels.extend(loaded)
        detected_frames = max(detected_frames, count)
    parquet_path = _resolve_dataset_path(root, record.get("parquet_path"))
    if not channels and selected_source != "force" and parquet_path.is_file():
        channels, detected_frames = _inline_channels(parquet_path)
        selected_source = "inline"
    if not channels:
        raise ValueError(f"No tactile streams found for episode {episode_index} in {root}")
    if len(channels) > 4:
        channels = channels[:4]
    _assign_physical_slots(channels)
    channels.sort(key=_physical_channel_key)
    frame_count = max(int(record.get("frame_count") or 0), detected_frames, 1)
    summary = record.get("summary") if isinstance(record.get("summary"), Mapping) else {}
    duration = float(record.get("duration_sec") or 0.0)
    fps = float(summary.get("nominal_fps") or (frame_count / duration if duration > 0 else 15.0))
    result = TactileEpisode(
        dataset_root=root,
        episode_index=int(episode_index),
        schema=_schema_from_root(root, record),
        frame_count=frame_count,
        channels=channels,
        source_paths=tuple(manifest_paths or [parquet_path]),
        nominal_fps=max(0.1, fps),
        tactile_source=selected_source,
    )
    _compute_change_series(result)
    _compute_force_trajectory_scales(result)
    return result


def _scalar_plane(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    while value.ndim > 3:
        value = value[..., 0]
    if value.ndim == 3:
        if value.shape[-1] == 1:
            value = value[..., 0]
        elif value.shape[-1] in (2, 3, 4):
            value = np.linalg.norm(value[..., :3], axis=-1)
        else:
            value = np.mean(value, axis=-1)
    if value.ndim == 1:
        side = max(1, int(math.ceil(math.sqrt(value.size))))
        padded = np.full(side * side, np.nan, dtype=np.float32)
        padded[: value.size] = value
        value = padded.reshape(side, side)
    if value.ndim == 0:
        value = value.reshape(1, 1)
    return value


def _change_score(current: np.ndarray | None, previous: np.ndarray | None) -> float:
    if current is None or previous is None or current.shape != previous.shape:
        return 0.0
    delta = np.asarray(current, dtype=np.float32) - np.asarray(previous, dtype=np.float32)
    finite = np.abs(delta[np.isfinite(delta)])
    return float(np.mean(finite)) if finite.size else 0.0


def _compute_change_series(episode: TactileEpisode) -> None:
    for channel in episode.channels:
        scores = np.zeros(episode.frame_count, dtype=np.float32)
        previous: np.ndarray | None = None
        for index in range(episode.frame_count):
            current = channel.frame(index)
            scores[index] = _change_score(current, previous)
            previous = None if current is None else np.asarray(current)
        channel.change_series = scores


def _trajectory_peak(chunks: Sequence[np.ndarray], *, default: float = 1.0) -> float:
    if not chunks:
        return default
    values = np.concatenate([np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in chunks])
    finite = np.abs(values[np.isfinite(values)])
    finite = finite[finite > 1e-12]
    if not finite.size:
        return default
    return float(np.max(finite))


def _compute_force_trajectory_scales(episode: TactileEpisode) -> None:
    """Compute fixed episode-wide force scales shared by all four tactile parts."""

    normal_by_mode: dict[str, list[np.ndarray]] = {
        "current": [],
        "delta": [],
        "baseline": [],
    }
    tangent_by_mode: dict[str, list[np.ndarray]] = {
        "current": [],
        "delta": [],
        "baseline": [],
    }
    for channel in episode.channels:
        first = channel.frame(0)
        if first is None or not _is_distributed_force(channel, np.asarray(first)):
            continue
        baseline = np.asarray(first, dtype=np.float32)
        previous: np.ndarray | None = None
        for index in range(episode.frame_count):
            current_value = channel.frame(index)
            if current_value is None:
                continue
            current = np.asarray(current_value, dtype=np.float32)
            variants = {
                "current": current,
                "delta": np.zeros_like(current) if previous is None else current - previous,
                "baseline": current - baseline,
            }
            for mode, value in variants.items():
                grid = cv2.resize(value[..., :3], (30, 20), interpolation=cv2.INTER_AREA)
                normal_by_mode[mode].append(-grid[..., 2])
                tangent_by_mode[mode].append(np.linalg.norm(grid[..., :2], axis=2))
            previous = current
    scales: dict[str, tuple[float, float]] = {}
    for mode in ("current", "delta", "baseline"):
        if not normal_by_mode[mode]:
            continue
        # One physical force reference is shared by Fz pressure and Fx/Fy
        # arrows. Separate component-wise normalization would exaggerate the
        # smaller tangential components and defeat trajectory normalization.
        force_ref = max(
            _trajectory_peak(normal_by_mode[mode]),
            _trajectory_peak(tangent_by_mode[mode]),
        )
        scales[mode] = (force_ref, force_ref)
    episode.force_scales = scales


def _robust_scale(values: np.ndarray, *, default: float = 1.0) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if not finite.size:
        return default
    scale = float(np.percentile(finite, 99.0))
    return scale if scale > 1e-8 else max(float(np.max(finite)), default)


def _is_distributed_force(channel: TactileChannel, value: np.ndarray) -> bool:
    field = channel.field_name.strip().lower().replace("-", "_")
    array = np.asarray(value)
    return (
        field in {"distributed_force", "dist_force", "distributedforce"}
        or (
            field not in {"raw", "image", "infer"}
            and array.dtype.kind == "f"
            and array.ndim == 3
            and array.shape[-1] >= 3
        )
    )


def _force_grid(force: np.ndarray) -> np.ndarray:
    source = np.nan_to_num(
        np.asarray(force, dtype=np.float32)[..., :3],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    # Keep the fixed 600-cell numerical grid while preserving whether a pad is
    # landscape (20 x 30) or has already been mounted portrait (30 x 20).
    size = (20, 30) if source.shape[0] > source.shape[1] else (30, 20)
    return cv2.resize(source, size, interpolation=cv2.INTER_AREA)


def _orient_tactile_plane(value: np.ndarray, tactile_slot: str) -> np.ndarray:
    """Rotate the recorded landscape pad into its mounted portrait direction."""

    array = np.asarray(value)
    if array.ndim < 2 or tactile_slot not in TACTILE_SLOTS:
        return array
    # Opposite rotations make both mounted pads portrait.  The outer pad then
    # receives the additional top/bottom mirror required by the physical rig.
    rotated = np.rot90(array, k=1 if tactile_slot == "inner" else 3, axes=(0, 1))
    if tactile_slot == "outer":
        rotated = np.flip(rotated, axis=0)
    return rotated


def _orient_force_grid(grid: np.ndarray, tactile_slot: str) -> np.ndarray:
    """Rotate both force samples and Fx/Fy coordinates into display space."""

    if tactile_slot not in TACTILE_SLOTS:
        return np.asarray(grid)
    oriented = np.array(
        _orient_tactile_plane(grid, tactile_slot),
        dtype=np.float32,
        copy=True,
    )
    old_fx = oriented[..., 0].copy()
    old_fy = oriented[..., 1].copy()
    if tactile_slot == "inner":  # 90 degrees counter-clockwise
        oriented[..., 0] = -old_fy
        oriented[..., 1] = old_fx
    else:  # 90 degrees clockwise, then vertically mirrored
        oriented[..., 0] = old_fy
        oriented[..., 1] = old_fx
    return oriented


def _surface_outward_direction(side: str, tactile_slot: str) -> np.ndarray:
    """Return the projected direction in which the cavity opens."""

    directions = {
        ("left", "outer"): (-1.0, 0.0),
        ("left", "inner"): (1.0, 0.0),
        ("right", "inner"): (-1.0, 0.0),
        ("right", "outer"): (1.0, 0.0),
    }
    # Keep the historical downward projection for unresolved/non-physical
    # synthetic channels; all resolved four-pad recordings use the mapping.
    return np.asarray(directions.get((side, tactile_slot), (0.0, 1.0)), dtype=np.float32)


def _surface_compression_direction(side: str, tactile_slot: str) -> np.ndarray:
    """Return inward -Fz displacement, opposite the cavity opening."""

    outward = _surface_outward_direction(side, tactile_slot)
    if (side, tactile_slot) not in PHYSICAL_TACTILE_ORDER:
        return outward
    return -outward


def _force_metrics(
    grid: np.ndarray,
    *,
    normal_ref: float,
    signed_normal: bool = False,
) -> ForceMetrics:
    normal = -grid[..., 2]
    positive = np.maximum(normal, 0.0)
    contact_pressure = np.abs(normal) if signed_normal else positive
    shear = np.linalg.norm(grid[..., :2], axis=2)
    magnitude = np.linalg.norm(grid, axis=2)
    noise_floor = normal_ref * COP_NOISE_FLOOR_RATIO
    weights = np.maximum(contact_pressure - noise_floor, 0.0)
    weight_sum = float(np.sum(weights))
    active_count = int(np.count_nonzero(weights))
    contact_strength = float(np.max(contact_pressure)) / normal_ref
    center: tuple[float, float] | None = None
    contact_visible = (
        contact_strength >= COP_PEAK_GATE_RATIO
        and weight_sum / normal_ref >= COP_MASS_GATE_RATIO
        and active_count >= 3
    )
    if contact_visible:
        yy, xx = np.indices(weights.shape, dtype=np.float32)
        center = (
            float(np.sum(xx * weights) / weight_sum / max(1, weights.shape[1] - 1)),
            float(np.sum(yy * weights) / weight_sum / max(1, weights.shape[0] - 1)),
        )
    mean_xyz = np.mean(grid, axis=(0, 1))
    return ForceMetrics(
        mean_xyz=tuple(float(value) for value in mean_xyz[:3]),
        peak_force=float(np.max(magnitude)),
        normal_mean=float(np.mean(normal)),
        normal_peak=float(np.max(normal)),
        shear_mean=float(np.mean(shear)),
        shear_peak=float(np.max(shear)),
        active_percent=float(np.mean(weights > 0.0) * 100.0),
        center_of_pressure=center,
        contact_strength=contact_strength,
    )


def _normal_u8(normal: np.ndarray, *, reference: float, signed: bool) -> np.ndarray:
    if signed:
        return np.clip((normal / reference + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return np.clip(np.maximum(normal, 0.0) / reference * 255.0, 0, 255).astype(np.uint8)


def _palette(
    values: np.ndarray,
    *,
    style: str,
    signed: bool,
    palette: str | None = None,
) -> np.ndarray:
    if signed:
        return cv2.applyColorMap(values, cv2.COLORMAP_COOL)
    palette_name = palette or {
        "classic": "turbo",
        "neon": "magma",
        "neon_focus": "magma",
        "neon_contour": "magma",
        "components": "magma",
        "contour": "inferno",
        "split": "magma",
    }.get(style, "turbo")
    maps = {
        "magma": cv2.COLORMAP_MAGMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "turbo": cv2.COLORMAP_TURBO,
        "cividis": cv2.COLORMAP_CIVIDIS,
        "ocean": cv2.COLORMAP_OCEAN,
        "inferno": cv2.COLORMAP_INFERNO,
    }
    return cv2.applyColorMap(values, maps.get(palette_name, cv2.COLORMAP_TURBO))


def _arrow_colors(theme: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return a restrained shadow/main BGR pair for arrows."""

    themes = {
        "cyan": ((24, 22, 18), (220, 205, 72)),
        "amber": ((18, 20, 28), (72, 180, 242)),
        "mint": ((22, 22, 18), (168, 224, 104)),
        "coral": ((24, 18, 28), (128, 112, 238)),
        "legacy": ((10, 10, 10), (245, 245, 245)),
    }
    return themes.get(theme, themes["cyan"])


def _draw_pressure_legend(
    image: np.ndarray,
    *,
    reference: float,
    signed: bool,
    style: str,
    palette: str | None = None,
) -> None:
    height, width = image.shape[:2]
    bar_h = max(70, min(150, height - 38))
    x1, x2 = width - 24, width - 13
    y1, y2 = (height - bar_h) // 2, (height + bar_h) // 2
    ramp = np.linspace(255, 0, bar_h, dtype=np.uint8)[:, None]
    color = _palette(ramp, style=style, signed=signed, palette=palette)
    image[y1:y2, x1:x2] = cv2.resize(color, (x2 - x1, y2 - y1))
    cv2.rectangle(image, (x1 - 1, y1 - 1), (x2, y2), (238, 242, 246), 1)
    high = f"+{reference:.2g}" if signed else f"{reference:.2g}"
    low = f"-{reference:.2g}" if signed else "0"
    cv2.putText(image, high, (max(2, x1 - 52), y1 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (244, 246, 249), 1, cv2.LINE_AA)
    cv2.putText(image, low, (max(2, x1 - 52), y2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (244, 246, 249), 1, cv2.LINE_AA)


def _draw_force_arrows(
    image: np.ndarray,
    grid: np.ndarray,
    *,
    vector_ref: float,
    stride: int = 3,
    dark_outline: bool = True,
    visibility_mask: np.ndarray | None = None,
    arrow_theme: str = "cyan",
) -> None:
    rows, cols = grid.shape[:2]
    height, width = image.shape[:2]
    cell_w, cell_h = float(width) / cols, float(height) / rows
    max_arrow = 0.85 * min(cell_w, cell_h) * max(1, stride)
    shadow_color, main_color = _arrow_colors(arrow_theme)
    layer = np.zeros_like(image)
    for row in range(0, rows, stride):
        for col in range(0, cols, stride):
            if visibility_mask is not None and not bool(visibility_mask[row, col]):
                continue
            fx, fy = (float(component) for component in grid[row, col, :2])
            norm = math.hypot(fx, fy)
            if norm / vector_ref < 0.04:
                continue
            length = min(
                max_arrow,
                max(max_arrow * 0.24, max_arrow * norm / vector_ref),
            )
            start = (int(round((col + 0.5) * cell_w)), int(round((row + 0.5) * cell_h)))
            end = (
                int(round(start[0] + fx / max(norm, 1e-8) * length)),
                int(round(start[1] - fy / max(norm, 1e-8) * length)),
            )
            if dark_outline:
                cv2.arrowedLine(image, start, end, shadow_color, 3, cv2.LINE_AA, tipLength=0.32)
            cv2.arrowedLine(layer, start, end, main_color, 2, cv2.LINE_AA, tipLength=0.32)
    cv2.addWeighted(image, 1.0, layer, 0.82, 0.0, dst=image)


def _draw_soft_cop_marker(image: np.ndarray, point: tuple[int, int]) -> None:
    """Draw a small, subdued CoP ring that does not dominate pressure color."""

    layer = np.zeros_like(image)
    cv2.circle(layer, point, 5, (205, 190, 154), 1, cv2.LINE_AA)
    cv2.circle(layer, point, 1, (172, 158, 132), -1, cv2.LINE_AA)
    cv2.addWeighted(image, 1.0, layer, 0.62, 0.0, dst=image)


def _draw_center_of_pressure(image: np.ndarray, center: tuple[float, float] | None) -> None:
    if center is None:
        return
    height, width = image.shape[:2]
    point = (int(round(center[0] * (width - 1))), int(round(center[1] * (height - 1))))
    _draw_soft_cop_marker(image, point)


def _surface_force_image(
    grid: np.ndarray,
    *,
    normal_ref: float,
    vector_ref: float | None = None,
    signed_normal: bool,
    output_width: int,
    output_height: int,
    side: str = "unknown",
    tactile_slot: str = "unknown",
    view: str = "right",
    center_of_pressure: tuple[float, float] | None = None,
    palette: str = "magma",
    arrow_theme: str = "cyan",
) -> np.ndarray:
    """Draw an outward-opening cavity with inward -Fz and in-plane Fx/Fy."""

    image = np.full((output_height, output_width, 3), (9, 12, 18), dtype=np.uint8)
    # Interpolation only increases visual mesh density. Metrics and trajectory
    # normalization remain based on the unchanged 20 x 30 force grid.
    dense_grid = cv2.resize(
        grid,
        (grid.shape[1] * 2, grid.shape[0] * 2),
        interpolation=cv2.INTER_CUBIC,
    )
    normal = -dense_grid[..., 2]
    if not signed_normal:
        normal = np.maximum(normal, 0.0)
    normalized = np.clip(normal / normal_ref, -1.0 if signed_normal else 0.0, 1.0)
    colors = _palette(
        _normal_u8(normal, reference=normal_ref, signed=signed_normal),
        style="neon",
        signed=signed_normal,
        palette=palette,
    )
    rows, cols = normalized.shape
    configurations = {
        # Portrait geometry: the projected long (row) axis remains visibly
        # longer than the transverse (column) axis in every 3D camera.
        "right": (0.08, 0.64, 0.55, 0.16, -0.50, 0.26),
        "left": (0.25, 0.64, 0.55, -0.16, -0.50, 0.26),
        # Improved front: enough depth to read contact position, unlike the old
        # near-profile strip, while remaining symmetric around the center.
        "front": (0.10, 0.70, 0.70, 0.00, -0.58, 0.24),
        "hero": (0.06, 0.66, 0.55, 0.17, -0.54, 0.27),
        "hero_left": (0.25, 0.66, 0.55, -0.17, -0.54, 0.27),
        "high": (0.08, 0.71, 0.58, 0.12, -0.62, 0.22),
        "top": (0.12, 0.75, 0.68, 0.00, -0.66, 0.16),
        "profile": (0.07, 0.58, 0.68, 0.05, -0.42, 0.34),
    }
    origin_x, base_y, col_x, row_x, row_y, depth = configurations.get(
        view,
        configurations["right"],
    )
    if signed_normal:
        base_y = max(base_y, 0.52)
    origin = np.asarray((output_width * origin_x, output_height * base_y), dtype=np.float32)
    # side/slot never alter these two in-plane bases. Paired inner/outer pads
    # therefore remain strictly parallel; only origin translation and the
    # opposite inward compression direction differ between the two pads.
    col_basis = np.asarray((output_width * col_x, 0.0), dtype=np.float32)
    row_basis = np.asarray((output_width * row_x, output_height * row_y), dtype=np.float32)
    outward_normal = _surface_outward_direction(side, tactile_slot)
    compression_direction = _surface_compression_direction(side, tactile_slot)
    physical_outward = abs(float(outward_normal[0])) > 0.5
    dent_depth = (
        output_width * 0.16
        if physical_outward
        else output_height * depth
    )
    if physical_outward:
        base_x = (
            float(origin[0]),
            float(origin[0] + col_basis[0]),
            float(origin[0] + row_basis[0]),
            float(origin[0] + col_basis[0] + row_basis[0]),
        )
        if compression_direction[0] < 0:
            origin[0] += max(0.0, 10.0 + dent_depth - min(base_x))
        else:
            origin[0] += min(0.0, output_width - 34.0 - dent_depth - max(base_x))

    def project(row: int, col: int) -> tuple[int, int]:
        point = (
            origin
            + col / max(1, cols - 1) * col_basis
            + row / max(1, rows - 1) * row_basis
            + normalized[row, col] * dent_depth * compression_direction
        )
        return int(round(float(point[0]))), int(round(float(point[1])))

    # Filled 2x2 interpolated cells carry pressure color; dense neutral grid
    # lines then make subtle curvature legible without a noisy pixel surface.
    fill_layer = image.copy()
    for row in range(rows - 3, -1, -2):
        for col in range(0, cols - 2, 2):
            polygon = np.asarray(
                (
                    project(row, col),
                    project(row, col + 2),
                    project(row + 2, col + 2),
                    project(row + 2, col),
                ),
                dtype=np.int32,
            )
            color = np.mean(
                colors[row : row + 3 : 2, col : col + 3 : 2],
                axis=(0, 1),
            )
            cv2.fillConvexPoly(fill_layer, polygon, tuple(int(value) for value in color), cv2.LINE_AA)
    cv2.addWeighted(image, 0.24, fill_layer, 0.76, 0.0, dst=image)

    mesh_layer = np.zeros_like(image)
    mesh_color = (116, 122, 142)
    for row in range(rows):
        points = np.asarray([project(row, col) for col in range(cols)], dtype=np.int32)
        cv2.polylines(mesh_layer, [points], False, mesh_color, 1, cv2.LINE_AA)
    for col in range(0, cols, 2):
        points = np.asarray([project(row, col) for row in range(rows)], dtype=np.int32)
        cv2.polylines(mesh_layer, [points], False, mesh_color, 1, cv2.LINE_AA)
    cv2.addWeighted(image, 1.0, mesh_layer, 0.52, 0.0, dst=image)

    # Fx/Fy are not height. They are projected onto the local sensor-plane
    # basis at surface vertices, preserving their tangential meaning.
    tangent_ref = max(float(vector_ref or normal_ref), 1e-12)
    unit_col = col_basis / max(float(np.linalg.norm(col_basis)), 1e-12)
    # The established top view maps +Fy upward, i.e. opposite increasing row.
    unit_fy = -row_basis / max(float(np.linalg.norm(row_basis)), 1e-12)
    arrow_limit = min(output_width, output_height) * 0.065
    shadow_color, main_color = _arrow_colors(arrow_theme)
    arrow_layer = np.zeros_like(image)
    for row in range(4, rows, 8):
        for col in range(4, cols, 8):
            fx, fy = (float(value) for value in dense_grid[row, col, :2])
            tangent = math.hypot(fx, fy)
            local_signal = max(tangent / tangent_ref, abs(float(normal[row, col])) / normal_ref)
            if tangent / tangent_ref < 0.06 or local_signal < 0.06:
                continue
            displacement = (fx * unit_col + fy * unit_fy) / tangent_ref * arrow_limit
            length = float(np.linalg.norm(displacement))
            if 1e-12 < length < arrow_limit * 0.20:
                displacement *= arrow_limit * 0.20 / length
                length = arrow_limit * 0.20
            if length > arrow_limit:
                displacement *= arrow_limit / length
            start = project(row, col)
            end = (
                int(round(start[0] + float(displacement[0]))),
                int(round(start[1] + float(displacement[1]))),
            )
            cv2.arrowedLine(image, start, end, shadow_color, 3, cv2.LINE_AA, tipLength=0.32)
            cv2.arrowedLine(arrow_layer, start, end, main_color, 2, cv2.LINE_AA, tipLength=0.32)
    cv2.addWeighted(image, 1.0, arrow_layer, 0.86, 0.0, dst=image)

    if center_of_pressure is not None:
        cop_col = min(cols - 1, max(0, int(round(center_of_pressure[0] * (cols - 1)))))
        cop_row = min(rows - 1, max(0, int(round(center_of_pressure[1] * (rows - 1)))))
        cop = project(cop_row, cop_col)
        _draw_soft_cop_marker(image, cop)

    # A small projected axis key makes the distinction between height and
    # in-plane vector components explicit for presentation audiences.
    key_origin = np.asarray((34.0, float(output_height - 28)), dtype=np.float32)
    key_scale = min(output_width, output_height) * 0.075
    for direction, color, label in (
        (unit_col, (255, 190, 80), "Fx"),
        (unit_fy, (100, 235, 140), "Fy"),
        (compression_direction, (235, 110, 255), "-Fz"),
    ):
        end = key_origin + direction * key_scale
        start_px = tuple(int(round(value)) for value in key_origin)
        end_px = tuple(int(round(value)) for value in end)
        cv2.arrowedLine(image, start_px, end_px, color, 2, cv2.LINE_AA, tipLength=0.28)
        cv2.putText(image, label, (end_px[0] + 2, end_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.31, color, 1, cv2.LINE_AA)
    cv2.putText(
        image,
        f"3D {view}: cavity OUT | -Fz IN | Fx/Fy plane",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (220, 229, 238),
        1,
        cv2.LINE_AA,
    )
    return image


def _paired_surface_state(
    channel: TactileChannel | None,
    frame_index: int,
    *,
    mode: str,
    temporal_smoothing: float,
    normal_ref: float,
) -> tuple[np.ndarray | None, ForceMetrics | None]:
    if channel is None:
        return None, None
    value, shown, has_reference = _shown_value(channel, frame_index, mode=mode)
    if value is None or shown is None or not _is_distributed_force(channel, value):
        return None, None
    visual = _temporally_smoothed_value(
        channel,
        frame_index,
        mode=mode,
        current=shown,
        amount=temporal_smoothing,
    )
    visual_grid = _orient_force_grid(_force_grid(visual), channel.tactile_slot)
    metrics_grid = _orient_force_grid(_force_grid(shown), channel.tactile_slot)
    return visual_grid, _force_metrics(
        metrics_grid,
        normal_ref=normal_ref,
        signed_normal=has_reference,
    )


def _draw_parallel_pad_backing(
    image: np.ndarray,
    *,
    origin: np.ndarray,
    col_basis: np.ndarray,
    row_basis: np.ndarray,
    body_offset: np.ndarray,
) -> None:
    """Draw a thin jaw body behind a tactile face to expose its 3D side."""

    face = np.asarray(
        (origin, origin + col_basis, origin + col_basis + row_basis, origin + row_basis),
        dtype=np.int32,
    )
    back = np.rint(face.astype(np.float32) + body_offset).astype(np.int32)
    cv2.fillConvexPoly(image, back, (19, 24, 31), cv2.LINE_AA)
    for index in range(4):
        next_index = (index + 1) % 4
        side = np.asarray(
            (face[index], face[next_index], back[next_index], back[index]),
            dtype=np.int32,
        )
        cv2.fillConvexPoly(image, side, (31, 39, 48), cv2.LINE_AA)
    cv2.polylines(image, [back], True, (69, 81, 91), 1, cv2.LINE_AA)


def _draw_dashed_segment(
    image: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_length: float = 8.0,
    gap_length: float = 8.0,
) -> None:
    vector = end.astype(np.float32) - start.astype(np.float32)
    length = float(np.linalg.norm(vector))
    if length <= 1e-6:
        return
    direction = vector / length
    cursor = 0.0
    while cursor < length:
        dash_end = min(length, cursor + dash_length)
        p1 = np.rint(start + direction * cursor).astype(int)
        p2 = np.rint(start + direction * dash_end).astype(int)
        cv2.line(image, tuple(p1), tuple(p2), color, thickness, cv2.LINE_AA)
        cursor += dash_length + gap_length


def _draw_parallel_pad_surface(
    image: np.ndarray,
    grid: np.ndarray | None,
    *,
    origin: np.ndarray,
    col_basis: np.ndarray,
    row_basis: np.ndarray,
    compression_direction: np.ndarray,
    normal_ref: float,
    vector_ref: float,
    signed_normal: bool,
    palette: str,
    arrow_theme: str,
    center_of_pressure: tuple[float, float] | None,
    depth_pixels: float,
) -> None:
    if grid is None:
        dense_grid = np.zeros((60, 40, 3), dtype=np.float32)
    else:
        dense_grid = cv2.resize(
            grid,
            (grid.shape[1] * 2, grid.shape[0] * 2),
            interpolation=cv2.INTER_CUBIC,
        )
    normal = -dense_grid[..., 2]
    if not signed_normal:
        normal = np.maximum(normal, 0.0)
    normalized = np.clip(
        normal / max(normal_ref, 1e-12),
        -1.0 if signed_normal else 0.0,
        1.0,
    )
    colors = _palette(
        _normal_u8(normal, reference=normal_ref, signed=signed_normal),
        style="neon",
        signed=signed_normal,
        palette=palette,
    )
    rows, cols = normalized.shape

    def project(row: int, col: int) -> tuple[int, int]:
        point = (
            origin
            + col / max(1, cols - 1) * col_basis
            + row / max(1, rows - 1) * row_basis
            + normalized[row, col] * depth_pixels * compression_direction
        )
        return int(round(float(point[0]))), int(round(float(point[1])))

    fill_layer = image.copy()
    for row in range(rows - 3, -1, -2):
        for col in range(0, cols - 2, 2):
            polygon = np.asarray(
                (
                    project(row, col),
                    project(row, col + 2),
                    project(row + 2, col + 2),
                    project(row + 2, col),
                ),
                dtype=np.int32,
            )
            color = np.mean(
                colors[row : row + 3 : 2, col : col + 3 : 2],
                axis=(0, 1),
            )
            cv2.fillConvexPoly(
                fill_layer,
                polygon,
                tuple(int(value) for value in color),
                cv2.LINE_AA,
            )
    cv2.addWeighted(image, 0.22, fill_layer, 0.78, 0.0, dst=image)

    mesh_layer = np.zeros_like(image)
    mesh_color = (112, 119, 139)
    for row in range(rows):
        points = np.asarray([project(row, col) for col in range(cols)], dtype=np.int32)
        cv2.polylines(mesh_layer, [points], False, mesh_color, 1, cv2.LINE_AA)
    for col in range(0, cols, 2):
        points = np.asarray([project(row, col) for row in range(rows)], dtype=np.int32)
        cv2.polylines(mesh_layer, [points], False, mesh_color, 1, cv2.LINE_AA)
    cv2.addWeighted(image, 1.0, mesh_layer, 0.50, 0.0, dst=image)

    if grid is not None:
        tangent_ref = max(vector_ref, 1e-12)
        unit_col = col_basis / max(float(np.linalg.norm(col_basis)), 1e-12)
        unit_fy = -row_basis / max(float(np.linalg.norm(row_basis)), 1e-12)
        arrow_limit = min(image.shape[:2]) * 0.045
        shadow_color, main_color = _arrow_colors(arrow_theme)
        arrow_layer = np.zeros_like(image)
        for row in range(4, rows, 8):
            for col in range(4, cols, 8):
                fx, fy = (float(value) for value in dense_grid[row, col, :2])
                tangent = math.hypot(fx, fy)
                if tangent / tangent_ref < 0.06:
                    continue
                displacement = (fx * unit_col + fy * unit_fy) / tangent_ref * arrow_limit
                length = float(np.linalg.norm(displacement))
                if length > arrow_limit:
                    displacement *= arrow_limit / length
                start = project(row, col)
                end = (
                    int(round(start[0] + float(displacement[0]))),
                    int(round(start[1] + float(displacement[1]))),
                )
                cv2.arrowedLine(image, start, end, shadow_color, 3, cv2.LINE_AA, tipLength=0.30)
                cv2.arrowedLine(arrow_layer, start, end, main_color, 2, cv2.LINE_AA, tipLength=0.30)
        cv2.addWeighted(image, 1.0, arrow_layer, 0.84, 0.0, dst=image)

    if center_of_pressure is not None:
        cop_col = min(cols - 1, max(0, int(round(center_of_pressure[0] * (cols - 1)))))
        cop_row = min(rows - 1, max(0, int(round(center_of_pressure[1] * (rows - 1)))))
        _draw_soft_cop_marker(image, project(cop_row, cop_col))


def _paired_surface_scene(
    episode: TactileEpisode,
    frame_index: int,
    *,
    side: str,
    mode: str,
    style: str,
    width: int,
    height: int,
    temporal_smoothing: float,
    show_text: bool,
) -> np.ndarray:
    profile = _STYLE_PROFILES[style]
    channels = {
        channel.tactile_slot: channel
        for channel in episode.channels
        if channel.side == side and channel.tactile_slot in TACTILE_SLOTS
    }
    slot_order = ("outer", "inner") if side == "left" else ("inner", "outer")
    normal_ref, vector_ref = episode.force_scales.get(mode, (1.0, 1.0))
    states = {
        slot: _paired_surface_state(
            channels.get(slot),
            frame_index,
            mode=mode,
            temporal_smoothing=temporal_smoothing,
            normal_ref=normal_ref,
        )
        for slot in slot_order
    }
    image = np.full((height, width, 3), (9, 11, 16), dtype=np.uint8)
    view = profile.surface_view
    configurations = {
        "right": (0.93, 0.315, 0.070, -0.82, 0.075),
        "left": (0.93, 0.315, -0.070, -0.82, 0.075),
        "front": (0.93, 0.330, 0.000, -0.84, 0.070),
        "hero": (0.94, 0.315, 0.072, -0.84, 0.078),
        "hero_left": (0.94, 0.315, -0.072, -0.84, 0.078),
        "high": (0.94, 0.325, 0.050, -0.86, 0.070),
        "top": (0.95, 0.335, 0.000, -0.87, 0.060),
        "profile": (0.91, 0.340, 0.035, -0.72, 0.085),
    }
    base_y, col_scale, row_x_scale, row_y_scale, depth_scale = configurations.get(
        view,
        configurations["right"],
    )
    # Keep the two faces centered at the quarter points.  Computing the origin
    # from the complete projected footprint gives small, balanced outer margins
    # for both oblique directions instead of leaving broad black side frames.
    footprint_x = col_scale + abs(row_x_scale)
    first_x = 0.24 - footprint_x * 0.5 - min(0.0, row_x_scale)
    second_x = 0.76 - footprint_x * 0.5 - min(0.0, row_x_scale)
    col_basis = np.asarray((width * col_scale, 0.0), dtype=np.float32)
    row_basis = np.asarray((width * row_x_scale, height * row_y_scale), dtype=np.float32)
    origins = (
        np.asarray((width * first_x, height * base_y), dtype=np.float32),
        np.asarray((width * second_x, height * base_y), dtype=np.float32),
    )
    # Both faces occupy parallel geometric planes, but the second face is
    # parameterized from its opposite edge.  This mirrors corresponding taxels
    # without changing their values and makes the two contact maps read as an
    # opposed jaw pair rather than two independent screens.
    pad_origins = (origins[0], origins[1] + col_basis)
    pad_col_bases = (col_basis, -col_basis)

    body_offsets = (
        np.asarray((-width * 0.018, height * 0.012), dtype=np.float32),
        np.asarray((width * 0.018, height * 0.012), dtype=np.float32),
    )
    for pad_origin, pad_col_basis, body_offset in zip(
        pad_origins,
        pad_col_bases,
        body_offsets,
        strict=True,
    ):
        _draw_parallel_pad_backing(
            image,
            origin=pad_origin,
            col_basis=pad_col_basis,
            row_basis=row_basis,
            body_offset=body_offset,
        )

    # Three very quiet alignment rails join equal heights at the two inward
    # edges.  They provide a direct correspondence cue without becoming a
    # foreground annotation or suggesting that the faces are coplanar.
    for fraction in (0.20, 0.50, 0.80):
        left_point = origins[0] + col_basis + row_basis * fraction
        right_point = origins[1] + row_basis * fraction
        _draw_dashed_segment(
            image,
            left_point,
            right_point,
            color=(44, 53, 62),
            dash_length=max(5.0, width * 0.008),
            gap_length=max(5.0, width * 0.007),
        )
    # A restrained lower palm/yoke makes these read as two fingers of one
    # parallel gripper.  A top connector made them look like suspended panels.
    bottom_centers = tuple(
        pad_origin + pad_col_basis * 0.5
        for pad_origin, pad_col_basis in zip(pad_origins, pad_col_bases, strict=True)
    )
    rail_y = min(
        height - 4,
        int(round(max(float(point[1]) for point in bottom_centers) + height * 0.026)),
    )
    yoke_color = (42, 51, 61)
    rail_start = (int(round(float(bottom_centers[0][0]))), rail_y)
    rail_end = (int(round(float(bottom_centers[1][0]))), rail_y)
    cv2.line(image, rail_start, rail_end, (24, 30, 38), max(7, round(height * 0.014)), cv2.LINE_AA)
    cv2.line(
        image,
        rail_start,
        rail_end,
        yoke_color,
        2,
        cv2.LINE_AA,
    )
    for point in bottom_centers:
        cv2.line(
            image,
            (int(round(float(point[0]))), int(round(float(point[1])))),
            (int(round(float(point[0]))), rail_y),
            yoke_color,
            max(2, round(width * 0.004)),
            cv2.LINE_AA,
        )
    depth_pixels = width * depth_scale
    for index, slot in enumerate(slot_order):
        grid, metrics = states[slot]
        _draw_parallel_pad_surface(
            image,
            grid,
            origin=pad_origins[index],
            col_basis=pad_col_bases[index],
            row_basis=row_basis,
            # Pressure displacement points away from the central gripping gap;
            # this is the requested inverse of the v10 candidate.
            compression_direction=np.asarray((-1.0, 0.0) if index == 0 else (1.0, 0.0), dtype=np.float32),
            normal_ref=normal_ref,
            vector_ref=vector_ref,
            signed_normal=mode != "current",
            palette=profile.palette,
            arrow_theme=profile.arrow_theme,
            center_of_pressure=None if metrics is None else metrics.center_of_pressure,
            depth_pixels=depth_pixels,
        )
        if show_text:
            label_x = int(
                round(
                    pad_origins[index][0]
                    + (pad_col_bases[index][0] + row_basis[0]) * 0.5
                )
            )
            font_scale = max(0.42, width / 1600.0)
            label = slot.upper()
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            cv2.putText(
                image,
                label,
                (label_x - text_size[0] // 2, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (205, 216, 224),
                1,
                cv2.LINE_AA,
            )
            if metrics is not None:
                metric = f"Pmax {metrics.normal_peak:.2g}   Smax {metrics.shear_peak:.2g}"
                metric_size = cv2.getTextSize(metric, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.64, 1)[0]
                cv2.putText(
                    image,
                    metric,
                    (label_x - metric_size[0] // 2, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.64,
                    (139, 158, 172),
                    1,
                    cv2.LINE_AA,
                )
    return image


def _render_paired_surface_dashboard(
    episode: TactileEpisode,
    frame_index: int,
    *,
    mode: str,
    width: int,
    style: str,
    temporal_smoothing: float,
    show_text: bool,
) -> np.ndarray:
    """Clean 16:9 presentation view with two parallel-pad gripper scenes."""

    height = max(360, round(width * 9.0 / 16.0))
    header_h = max(40, round(height * 0.050)) if show_text else 0
    footer_h = max(26, round(height * 0.035)) if show_text else max(14, round(height * 0.018))
    gap = max(4, round(width * 0.004))
    scene_h = height - header_h - footer_h
    scene_w = (width - gap) // 2
    canvas = np.full((height, width, 3), (7, 9, 13), dtype=np.uint8)
    left = _paired_surface_scene(
        episode,
        frame_index,
        side="left",
        mode=mode,
        style=style,
        width=scene_w,
        height=scene_h,
        temporal_smoothing=temporal_smoothing,
        show_text=show_text,
    )
    right_w = width - scene_w - gap
    right = _paired_surface_scene(
        episode,
        frame_index,
        side="right",
        mode=mode,
        style=style,
        width=right_w,
        height=scene_h,
        temporal_smoothing=temporal_smoothing,
        show_text=show_text,
    )
    canvas[header_h : header_h + scene_h, :scene_w] = left
    canvas[header_h : header_h + scene_h, scene_w + gap :] = right
    cv2.line(canvas, (scene_w + gap // 2, header_h), (scene_w + gap // 2, height - footer_h), (49, 60, 70), 1)
    title_scale = max(0.62, width / 2100.0)
    if show_text:
        for label, center_x, color in (
            ("LEFT", scene_w // 2, (118, 205, 242)),
            ("RIGHT", scene_w + gap + right_w // 2, (133, 220, 154)),
        ):
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, title_scale, 1)[0]
            cv2.putText(canvas, label, (center_x - label_size[0] // 2, header_h - 14), cv2.FONT_HERSHEY_SIMPLEX, title_scale, color, 1, cv2.LINE_AA)
    normal_ref = episode.force_scales.get(mode, (1.0, 1.0))[0]
    footer = canvas[height - footer_h :]
    bar_x1, bar_x2 = round(width * 0.36), round(width * 0.64)
    bar_y1, bar_y2 = max(4, footer_h // 3), max(6, footer_h // 3 + max(5, footer_h // 7))
    ramp = np.linspace(0, 255, max(2, bar_x2 - bar_x1), dtype=np.uint8)[None, :]
    color = _palette(ramp, style="neon", signed=mode != "current", palette=_STYLE_PROFILES[style].palette)
    footer[bar_y1:bar_y2, bar_x1:bar_x2] = cv2.resize(color, (bar_x2 - bar_x1, bar_y2 - bar_y1))
    if show_text:
        time_s = frame_index / max(episode.nominal_fps, 1e-6)
        time_text = f"{time_s:05.2f}s"
        cv2.putText(footer, time_text, (24, max(16, footer_h * 2 // 3)), cv2.FONT_HERSHEY_SIMPLEX, title_scale * 0.62, (156, 169, 179), 1, cv2.LINE_AA)
        small = max(0.34, width / 4000.0)
        cv2.putText(footer, "0" if mode == "current" else f"-{normal_ref:.2g}", (bar_x1 - 50, bar_y2), cv2.FONT_HERSHEY_SIMPLEX, small, (158, 172, 182), 1, cv2.LINE_AA)
        cv2.putText(footer, f"-Fz {normal_ref:.2g}", (bar_x2 + 12, bar_y2), cv2.FONT_HERSHEY_SIMPLEX, small, (190, 201, 210), 1, cv2.LINE_AA)
    return canvas


def _vector_force_image(
    grid: np.ndarray,
    *,
    normal_ref: float,
    vector_ref: float,
    signed_normal: bool,
    output_width: int,
    output_height: int,
    palette: str = "magma",
    arrow_theme: str = "cyan",
) -> np.ndarray:
    image = np.full((output_height, output_width, 3), (7, 11, 17), dtype=np.uint8)
    normal = -grid[..., 2]
    normalized = _normal_u8(normal, reference=normal_ref, signed=signed_normal)
    colors = _palette(normalized, style="neon", signed=signed_normal, palette=palette)
    rows, cols = normal.shape
    cell_w, cell_h = output_width / cols, output_height / rows
    for row in range(0, rows, 2):
        for col in range(0, cols, 2):
            strength = abs(float(normal[row, col])) / normal_ref
            if strength < 0.02:
                continue
            point = (int((col + 0.5) * cell_w), int((row + 0.5) * cell_h))
            radius = max(1, int(round(min(cell_w, cell_h) * min(strength, 1.0) * 0.55)))
            cv2.circle(image, point, radius, tuple(int(v) for v in colors[row, col]), -1, cv2.LINE_AA)
    _draw_force_arrows(
        image,
        grid,
        vector_ref=vector_ref,
        stride=2,
        dark_outline=True,
        arrow_theme=arrow_theme,
    )
    return image


def _component_force_image(
    grid: np.ndarray,
    *,
    force_ref: float,
    signed_normal: bool,
    output_width: int,
    output_height: int,
    center_of_pressure: tuple[float, float] | None,
    palette: str = "magma",
) -> np.ndarray:
    """Show Fx, Fy, and -Fz separately with one shared physical scale."""

    pane_widths = [output_width // 3, output_width // 3]
    pane_widths.append(output_width - sum(pane_widths))
    panes: list[np.ndarray] = []
    components = (
        (grid[..., 0], "Fx", True),
        (grid[..., 1], "Fy", True),
        (-grid[..., 2], "-Fz", signed_normal),
    )
    for pane_width, (component, label, signed) in zip(pane_widths, components):
        normalized = _normal_u8(component, reference=force_ref, signed=signed)
        pane = cv2.resize(
            _palette(normalized, style="neon", signed=signed, palette=palette),
            (pane_width, output_height),
            interpolation=cv2.INTER_CUBIC,
        )
        cv2.rectangle(pane, (6, 6), (52, 30), (8, 10, 15), -1)
        cv2.putText(pane, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (245, 247, 250), 1, cv2.LINE_AA)
        panes.append(pane)
    image = np.hstack(panes)
    boundary = 0
    for pane_width in pane_widths[:-1]:
        boundary += pane_width
        cv2.line(image, (boundary, 0), (boundary, output_height - 1), (235, 240, 245), 1)
    if center_of_pressure is not None:
        fz_start = pane_widths[0] + pane_widths[1]
        point = (
            fz_start + int(round(center_of_pressure[0] * (pane_widths[2] - 1))),
            int(round(center_of_pressure[1] * (output_height - 1))),
        )
        _draw_soft_cop_marker(image, point)
    return image


def _neon_focus_image(
    grid: np.ndarray,
    *,
    normal_ref: float,
    vector_ref: float,
    signed_normal: bool,
    output_width: int,
    output_height: int,
    palette: str = "magma",
    arrow_theme: str = "cyan",
) -> np.ndarray:
    """Darken sub-threshold pressure while preserving actual displayed values."""

    normal = -grid[..., 2]
    normalized = _normal_u8(normal, reference=normal_ref, signed=signed_normal)
    heat = _palette(normalized, style="neon", signed=signed_normal, palette=palette)
    signal = np.abs(normal) if signed_normal else np.maximum(normal, 0.0)
    mask = signal >= normal_ref * COP_NOISE_FLOOR_RATIO
    dimmed = np.asarray(heat, dtype=np.float32) * np.where(mask[..., None], 1.0, 0.12)
    image = cv2.resize(
        np.clip(dimmed, 0, 255).astype(np.uint8),
        (output_width, output_height),
        interpolation=cv2.INTER_CUBIC,
    )
    outline = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scale_x, scale_y = output_width / mask.shape[1], output_height / mask.shape[0]
    scaled_contours = [
        np.round(contour.astype(np.float32) * (scale_x, scale_y)).astype(np.int32)
        for contour in contours
    ]
    if scaled_contours:
        cv2.drawContours(image, scaled_contours, -1, (235, 245, 255), 1, cv2.LINE_AA)
    arrow_mask = mask | (np.linalg.norm(grid[..., :2], axis=2) >= vector_ref * 0.06)
    _draw_force_arrows(
        image,
        grid,
        vector_ref=vector_ref,
        stride=3,
        visibility_mask=arrow_mask,
        arrow_theme=arrow_theme,
    )
    return image


def _contour_overlay(image: np.ndarray, normalized: np.ndarray) -> None:
    quantized = (normalized // 32).astype(np.int16)
    edges = np.zeros_like(normalized, dtype=np.uint8)
    edges[1:, :] |= quantized[1:, :] != quantized[:-1, :]
    edges[:, 1:] |= quantized[:, 1:] != quantized[:, :-1]
    edges = cv2.resize(edges, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    image[edges > 0] = (245, 247, 250)


def _distributed_force_visual(
    force: np.ndarray,
    *,
    signed_normal: bool,
    style: str,
    metrics_force: np.ndarray | None = None,
    normal_ref: float | None = None,
    vector_ref: float | None = None,
    side: str = "unknown",
    tactile_slot: str = "unknown",
    output_width: int = 600,
    output_height: int = 400,
) -> tuple[np.ndarray, float, ForceMetrics]:
    profile = _STYLE_PROFILES.get(style)
    if profile is None:
        raise ValueError(f"Unknown visual style: {style}")
    family = profile.family
    grid = _orient_force_grid(_force_grid(force), tactile_slot)
    metrics_grid = (
        grid
        if metrics_force is None
        else _orient_force_grid(_force_grid(metrics_force), tactile_slot)
    )
    normal = -grid[..., 2]
    xy_norm = np.linalg.norm(grid[..., :2], axis=2)
    normal_ref = _robust_scale(normal) if normal_ref is None else max(float(normal_ref), 1e-12)
    vector_ref = _robust_scale(xy_norm) if vector_ref is None else max(float(vector_ref), 1e-12)
    metrics = _force_metrics(
        metrics_grid,
        normal_ref=normal_ref,
        signed_normal=signed_normal,
    )
    cop_drawn = False

    if family == "surface":
        image = _surface_force_image(
            grid,
            normal_ref=normal_ref,
            vector_ref=vector_ref,
            signed_normal=signed_normal,
            output_width=output_width,
            output_height=output_height,
            side=side,
            tactile_slot=tactile_slot,
            view=profile.surface_view,
            center_of_pressure=metrics.center_of_pressure,
            palette=profile.palette,
            arrow_theme=profile.arrow_theme,
        )
        cop_drawn = True
    elif family == "components":
        image = _component_force_image(
            grid,
            force_ref=normal_ref,
            signed_normal=signed_normal,
            output_width=output_width,
            output_height=output_height,
            center_of_pressure=metrics.center_of_pressure,
            palette=profile.palette,
        )
        cop_drawn = True
    elif family == "neon_focus":
        image = _neon_focus_image(
            grid,
            normal_ref=normal_ref,
            vector_ref=vector_ref,
            signed_normal=signed_normal,
            output_width=output_width,
            output_height=output_height,
            palette=profile.palette,
            arrow_theme=profile.arrow_theme,
        )
    elif family == "vectors":
        image = _vector_force_image(
            grid,
            normal_ref=normal_ref,
            vector_ref=vector_ref,
            signed_normal=signed_normal,
            output_width=output_width,
            output_height=output_height,
            palette=profile.palette,
            arrow_theme=profile.arrow_theme,
        )
    elif family == "split":
        split = max(1, output_width // 2)
        normalized = _normal_u8(normal, reference=normal_ref, signed=signed_normal)
        left = cv2.resize(
            _palette(
                normalized,
                style="neon",
                signed=signed_normal,
                palette=profile.palette,
            ),
            (split, output_height),
            interpolation=cv2.INTER_CUBIC,
        )
        glow = cv2.GaussianBlur(left, (0, 0), sigmaX=5.0)
        left = cv2.addWeighted(left, 0.84, glow, 0.32, 0.0)
        _draw_force_arrows(
            left,
            grid,
            vector_ref=vector_ref,
            arrow_theme=profile.arrow_theme,
        )
        _draw_center_of_pressure(left, metrics.center_of_pressure)
        right = _surface_force_image(
            grid,
            normal_ref=normal_ref,
            vector_ref=vector_ref,
            signed_normal=signed_normal,
            output_width=output_width - split,
            output_height=output_height,
            side=side,
            tactile_slot=tactile_slot,
            view=profile.surface_view,
            center_of_pressure=metrics.center_of_pressure,
            palette=profile.palette,
            arrow_theme=profile.arrow_theme,
        )
        image = np.hstack((left, right))
        cv2.line(image, (split, 0), (split, output_height - 1), (230, 235, 240), 1)
        cop_drawn = True
    else:
        normalized = _normal_u8(normal, reference=normal_ref, signed=signed_normal)
        heat = _palette(
            normalized,
            style=family,
            signed=signed_normal,
            palette=profile.palette,
        )
        interpolation = cv2.INTER_CUBIC if family in {"neon", "neon_contour"} else cv2.INTER_NEAREST
        image = cv2.resize(heat, (output_width, output_height), interpolation=interpolation)
        if family in {"neon", "neon_contour"}:
            glow = cv2.GaussianBlur(image, (0, 0), sigmaX=5.0)
            image = cv2.addWeighted(image, 0.84, glow, 0.32, 0.0)
        if family in {"contour", "neon_contour"}:
            _contour_overlay(image, normalized)
        _draw_force_arrows(
            image,
            grid,
            vector_ref=vector_ref,
            arrow_theme=profile.arrow_theme,
        )

    if not cop_drawn:
        _draw_center_of_pressure(image, metrics.center_of_pressure)
    _draw_pressure_legend(
        image,
        reference=normal_ref,
        signed=signed_normal,
        style="neon" if family in {"surface", "vectors"} else family,
        palette=profile.palette,
    )
    return image, normal_ref, metrics


def _distributed_force_image(
    force: np.ndarray,
    *,
    signed_normal: bool,
    normal_ref: float | None = None,
    vector_ref: float | None = None,
    output_width: int = 600,
    output_height: int = 400,
) -> tuple[np.ndarray, float, str]:
    """Compatibility wrapper for the original heatmap + arrows renderer."""

    image, reference, metrics = _distributed_force_visual(
        force,
        signed_normal=signed_normal,
        style="classic",
        normal_ref=normal_ref,
        vector_ref=vector_ref,
        output_width=output_width,
        output_height=output_height,
    )
    stats = (
        f"mean Fx={metrics.mean_xyz[0]:.3g} Fy={metrics.mean_xyz[1]:.3g} "
        f"Fz={metrics.mean_xyz[2]:.3g} peak={metrics.peak_force:.3g} | traj ref={reference:.3g}"
    )
    return image, reference, stats


def _shown_value(
    channel: TactileChannel,
    frame_index: int,
    *,
    mode: str,
) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    current = channel.frame(frame_index)
    if current is None:
        return None, None, False
    value = np.asarray(current)
    reference: np.ndarray | None = None
    if mode == "delta":
        reference = channel.frame(max(0, frame_index - 1))
    elif mode == "baseline":
        reference = channel.frame(0)
    has_reference = reference is not None and reference.shape == value.shape
    shown = np.asarray(value)
    if has_reference:
        shown = np.asarray(value, dtype=np.float32) - np.asarray(reference, dtype=np.float32)
    return value, shown, has_reference


def _temporally_smoothed_value(
    channel: TactileChannel,
    frame_index: int,
    *,
    mode: str,
    current: np.ndarray,
    amount: float,
) -> np.ndarray:
    """Blend neighboring display fields without changing stored or reported values."""

    mix = min(max(float(amount), 0.0), 0.48)
    if mix <= 0.0:
        return current
    neighbors: list[np.ndarray] = []
    for index in (max(0, frame_index - 1), min(frame_index + 1, channel.change_series.size - 1)):
        _, shown, _ = _shown_value(channel, index, mode=mode)
        if shown is not None and shown.shape == current.shape:
            neighbors.append(np.asarray(shown, dtype=np.float32))
    if not neighbors:
        return current
    neighbor_mean = np.mean(np.stack(neighbors), axis=0)
    return np.asarray(current, dtype=np.float32) * (1.0 - mix) + neighbor_mean * mix


def _scalar_visual(
    value: np.ndarray,
    *,
    signed: bool,
    style: str,
    side: str = "unknown",
    tactile_slot: str = "unknown",
    output_width: int,
    output_height: int,
) -> tuple[np.ndarray, float]:
    profile = _STYLE_PROFILES.get(style)
    if profile is None:
        raise ValueError(f"Unknown visual style: {style}")
    family = profile.family
    plane = _orient_tactile_plane(_scalar_plane(value), tactile_slot)
    scale = _robust_scale(plane)
    normalized = _normal_u8(plane, reference=scale, signed=signed)
    if family == "surface":
        fake_force = np.zeros((*plane.shape, 3), dtype=np.float32)
        fake_force[..., 2] = -plane
        image = _surface_force_image(
            _force_grid(fake_force),
            normal_ref=scale,
            signed_normal=signed,
            output_width=output_width,
            output_height=output_height,
            side=side,
            tactile_slot=tactile_slot,
            view=profile.surface_view,
            palette=profile.palette,
            arrow_theme=profile.arrow_theme,
        )
    elif family == "split":
        split = max(1, output_width // 2)
        left = cv2.resize(
            _palette(normalized, style="neon", signed=signed, palette=profile.palette),
            (split, output_height),
            interpolation=cv2.INTER_CUBIC,
        )
        fake_force = np.zeros((*plane.shape, 3), dtype=np.float32)
        fake_force[..., 2] = -plane
        right = _surface_force_image(
            _force_grid(fake_force),
            normal_ref=scale,
            signed_normal=signed,
            output_width=output_width - split,
            output_height=output_height,
            side=side,
            tactile_slot=tactile_slot,
            view=profile.surface_view,
            palette=profile.palette,
            arrow_theme=profile.arrow_theme,
        )
        image = np.hstack((left, right))
    else:
        palette_style = "neon" if family in {"neon", "neon_focus", "vectors", "components"} else family
        image = cv2.resize(
            _palette(normalized, style=palette_style, signed=signed, palette=profile.palette),
            (output_width, output_height),
            interpolation=cv2.INTER_CUBIC if family in {"neon", "neon_focus", "neon_contour"} else cv2.INTER_NEAREST,
        )
        if family in {"contour", "neon_contour"}:
            _contour_overlay(image, normalized)
    return image, scale


def _panel_image(
    channel: TactileChannel,
    frame_index: int,
    *,
    mode: str,
    width: int,
    height: int,
    style: str,
    temporal_smoothing: float = 0.0,
    force_scale: tuple[float, float] | None = None,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    header = 68
    footer = 82
    value, shown, has_reference = _shown_value(channel, frame_index, mode=mode)
    if value is None or shown is None:
        cv2.putText(canvas, f"{channel.label} - unavailable", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 190, 200), 1, cv2.LINE_AA)
        return canvas
    visual_shown = _temporally_smoothed_value(
        channel,
        frame_index,
        mode=mode,
        current=shown,
        amount=temporal_smoothing,
    )
    force_metrics: ForceMetrics | None = None
    if _is_distributed_force(channel, value):
        normal_ref, vector_ref = force_scale if force_scale is not None else (None, None)
        visual, scale, force_metrics = _distributed_force_visual(
            visual_shown,
            signed_normal=has_reference,
            style=style,
            metrics_force=shown,
            normal_ref=normal_ref,
            vector_ref=vector_ref,
            side=channel.side,
            tactile_slot=channel.tactile_slot,
            output_width=max(1, width),
            output_height=max(1, height - header - footer),
        )
    elif (
        not has_reference
        and style == "classic"
        and value.dtype == np.uint8
        and value.ndim in (2, 3)
        and channel.field_name.lower() in {"raw", "image", "unknown"}
    ):
        portrait_value = _orient_tactile_plane(value, channel.tactile_slot)
        if portrait_value.ndim == 2:
            visual = cv2.cvtColor(portrait_value, cv2.COLOR_GRAY2BGR)
        elif portrait_value.shape[-1] == 3:
            visual = cv2.cvtColor(portrait_value, cv2.COLOR_RGB2BGR)
        else:
            visual = cv2.cvtColor(portrait_value[..., 0], cv2.COLOR_GRAY2BGR)
        scale = 255.0
    else:
        visual, scale = _scalar_visual(
            visual_shown,
            signed=has_reference,
            style=style,
            side=channel.side,
            tactile_slot=channel.tactile_slot,
            output_width=max(1, width),
            output_height=max(1, height - header - footer),
        )
    available_h = max(1, height - header - footer)
    ratio = min(width / visual.shape[1], available_h / visual.shape[0])
    resized = cv2.resize(visual, (max(1, round(visual.shape[1] * ratio)), max(1, round(visual.shape[0] * ratio))), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_NEAREST)
    x = (width - resized.shape[1]) // 2
    y = header + (available_h - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    score = channel.change_series[frame_index] if frame_index < channel.change_series.size else 0.0
    cv2.putText(canvas, channel.label[:52], (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (238, 242, 246), 1, cv2.LINE_AA)
    field_text = (
        f"{channel.field_name} | {channel.orientation_label} | "
        f"{channel.dtype} {channel.shape} | {style}"
    )
    if _is_distributed_force(channel, value):
        field_text += " | -Fz pressure, Fx/Fy arrows"
    cv2.putText(canvas, field_text[:100], (12, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 185, 198), 1, cv2.LINE_AA)
    if force_metrics is not None:
        fx, fy, fz = force_metrics.mean_xyz
        force_lines = (
            f"Fx {fx:+.3g}   Fy {fy:+.3g}",
            f"Fz {fz:+.3g}   |F|max {force_metrics.peak_force:.3g}",
            f"P avg/max {force_metrics.normal_mean:+.2g} / {force_metrics.normal_peak:+.2g}",
        )
        cop = force_metrics.center_of_pressure
        cop_text = "hidden" if cop is None else f"({cop[0]:.2f}, {cop[1]:.2f})"
        force_lines += (
            f"Smax {force_metrics.shear_peak:.2g}   A {force_metrics.active_percent:.0f}%   CoP {cop_text}",
        )
    else:
        finite = np.asarray(shown, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            line_1 = (
                f"min {float(np.min(finite)):+.4g}   max {float(np.max(finite)):+.4g}   "
                f"mean {float(np.mean(finite)):+.4g}   rms {float(np.sqrt(np.mean(finite**2))):.4g}"
            )
        else:
            line_1 = "no finite numeric samples"
        line_2 = f"frame change mean(abs) {score:.4g}   display ref {scale:.4g}"
    if force_metrics is not None:
        for line_index, text in enumerate(force_lines):
            color = (225, 232, 239) if line_index < 2 else (166, 188, 205)
            cv2.putText(canvas, text, (12, height - 61 + line_index * 17), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, line_1[:120], (12, height - 31), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (225, 232, 239), 1, cv2.LINE_AA)
        cv2.putText(canvas, line_2[:145], (12, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (166, 188, 205), 1, cv2.LINE_AA)
    return canvas


def _timeline(episode: TactileEpisode, frame_index: int, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 12, dtype=np.uint8)
    colors = ((76, 190, 255), (106, 220, 120), (255, 155, 78), (210, 105, 255))
    left, right, top, bottom = 54, width - 18, 18, height - 28
    maximum = max((_robust_scale(channel.change_series, default=1.0) for channel in episode.channels), default=1.0)
    for channel_index, channel in enumerate(episode.channels):
        values = channel.change_series
        if values.size < 2:
            continue
        points = []
        for x in range(left, right):
            index = min(values.size - 1, round((x - left) * (values.size - 1) / max(1, right - left - 1)))
            y = bottom - round(min(float(values[index]) / maximum, 1.0) * (bottom - top))
            points.append((x, y))
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, colors[channel_index], 1, cv2.LINE_AA)
        cv2.putText(canvas, channel.label[:18], (left + channel_index * max(150, (right-left)//4), height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, colors[channel_index], 1, cv2.LINE_AA)
    cursor_x = left + round(frame_index * (right - left - 1) / max(1, episode.frame_count - 1))
    cv2.line(canvas, (cursor_x, top), (cursor_x, bottom), (245, 245, 245), 1)
    cv2.putText(canvas, "per-frame mean absolute change", (8, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 190, 200), 1, cv2.LINE_AA)
    return canvas


def render_episode_frame(
    episode: TactileEpisode,
    frame_index: int,
    *,
    mode: str = "current",
    width: int = DEFAULT_WIDTH,
    style: str = "classic",
    temporal_smoothing: float = 0.0,
    show_text: bool = True,
) -> np.ndarray:
    """Render four portrait pads in their physical left-to-right arrangement."""

    if mode not in {"current", "delta", "baseline"}:
        raise ValueError("mode must be current, delta, or baseline")
    if style not in VISUAL_STYLES:
        raise ValueError(f"style must be one of {', '.join(VISUAL_STYLES)}")
    index = min(max(0, int(frame_index)), episode.frame_count - 1)
    width = max(640, int(width))
    if _STYLE_PROFILES[style].family == "surface":
        return _render_paired_surface_dashboard(
            episode,
            index,
            mode=mode,
            width=width,
            style=style,
            temporal_smoothing=temporal_smoothing,
            show_text=show_text,
        )
    if not show_text:
        raise ValueError("show_text=False currently requires a 3D surface style")
    gap = 8
    header_h = 78
    timeline_h = 145
    usable_width = width - gap * 3
    panel_widths = [usable_width // 4] * 3
    panel_widths.append(usable_width - sum(panel_widths))
    representative_width = max(panel_widths)
    panel_h = max(480, round(representative_width * 4.0 / 3.0) + 150)
    canvas = np.full((header_h + panel_h + timeline_h, width, 3), 8, dtype=np.uint8)
    title = (
        f"DM-Tac episode {episode.episode_index} | {episode.tactile_source} | "
        f"frame {index}/{episode.frame_count - 1} | {mode} | {style}"
    )
    if temporal_smoothing > 0:
        title += f" | visual smooth {temporal_smoothing:.0%}"
    cv2.putText(canvas, title, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (242, 245, 248), 1, cv2.LINE_AA)
    cv2.putText(canvas, episode.schema, (16, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (155, 175, 190), 1, cv2.LINE_AA)
    midpoint = width // 2
    cv2.putText(canvas, "LEFT GRIPPER   OUTER  |  INNER", (16, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (115, 204, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, "RIGHT GRIPPER   INNER  |  OUTER", (midpoint + 16, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (136, 220, 154), 1, cv2.LINE_AA)
    cv2.line(canvas, (midpoint, 53), (midpoint, header_h + panel_h - 1), (56, 73, 82), 2, cv2.LINE_AA)
    channel_slots = _physical_channel_slots(episode)
    x = 0
    for slot, panel_w in enumerate(panel_widths):
        y = header_h
        channel = channel_slots[slot]
        if channel is not None:
            panel = _panel_image(
                channel,
                index,
                mode=mode,
                width=panel_w,
                height=panel_h,
                style=style,
                temporal_smoothing=temporal_smoothing,
                force_scale=episode.force_scales.get(mode),
            )
        else:
            panel = np.full((panel_h, panel_w, 3), 18, dtype=np.uint8)
            side, tactile_slot = PHYSICAL_TACTILE_ORDER[slot]
            cv2.putText(panel, f"{side.upper()} {tactile_slot.upper()}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 145, 156), 1, cv2.LINE_AA)
            cv2.putText(panel, "No tactile channel", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (105, 118, 128), 1, cv2.LINE_AA)
        canvas[y : y + panel_h, x : x + panel_w] = panel
        border_color = (66, 90, 104) if slot < 2 else (67, 99, 76)
        cv2.rectangle(canvas, (x, y), (x + panel_w - 1, y + panel_h - 1), border_color, 1)
        x += panel_w + gap
    canvas[-timeline_h:] = _timeline(episode, index, width, timeline_h)
    return canvas


def _selection(value: str | Sequence[str], *, allowed: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    else:
        requested = [str(item).strip().lower() for item in value if str(item).strip()]
    if requested == ["all"]:
        return tuple(allowed)
    invalid = [item for item in requested if item not in allowed]
    if invalid or not requested:
        choices = ", ".join((*allowed, "all"))
        raise ValueError(f"{name} must contain comma-separated values from: {choices}")
    return tuple(dict.fromkeys(requested))


def _export_path(base: Path, *, style: str, mode: str, multiple: bool) -> Path:
    suffix = base.suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
        base = base.with_suffix(".mp4")
    if not multiple:
        return base
    return base.with_name(f"{base.stem}--{style}--{mode}{base.suffix}")


def export_episode_videos(
    episode: TactileEpisode,
    output: str | Path,
    *,
    styles: str | Sequence[str] = ("classic",),
    modes: str | Sequence[str] = ("current",),
    width: int = DEFAULT_WIDTH,
    fps: float | None = None,
    codec: str | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    frame_step: int = 1,
    temporal_smoothing: float = 0.0,
    show_text: bool = True,
) -> list[VideoExport]:
    """Export one video per style/mode combination using the normal frame renderer."""

    selected_styles = _selection(styles, allowed=VISUAL_STYLES, name="styles")
    selected_modes = _selection(
        modes,
        allowed=("current", "delta", "baseline"),
        name="modes",
    )
    start = min(max(0, int(start_frame)), episode.frame_count - 1)
    end = episode.frame_count - 1 if end_frame is None else min(
        max(start, int(end_frame)), episode.frame_count - 1
    )
    step = max(1, int(frame_step))
    indices = tuple(range(start, end + 1, step))
    # When frames are subsampled, reduce the default output FPS so elapsed
    # episode time remains faithful. An explicit FPS intentionally overrides it.
    output_fps = episode.nominal_fps / step if fps is None else float(fps)
    if not math.isfinite(output_fps) or output_fps <= 0:
        raise ValueError("fps must be a finite value greater than zero")
    combinations = [(style, mode) for style in selected_styles for mode in selected_modes]
    base = Path(output).expanduser().resolve()
    exports: list[VideoExport] = []
    for style, mode in combinations:
        path = _export_path(base, style=style, mode=mode, multiple=len(combinations) > 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        first = render_episode_frame(
            episode,
            indices[0],
            mode=mode,
            style=style,
            width=width,
            temporal_smoothing=temporal_smoothing,
            show_text=show_text,
        )
        height, actual_width = first.shape[:2]
        # Most codecs require even frame dimensions. Padding does not resample data.
        video_width = actual_width + actual_width % 2
        video_height = height + height % 2
        fourcc_text = codec or ("MJPG" if path.suffix.lower() == ".avi" else "mp4v")
        if len(fourcc_text) != 4:
            raise ValueError("codec must be a four-character OpenCV FourCC")
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*fourcc_text),
            output_fps,
            (video_width, video_height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(
                f"Could not open video writer for {path} with codec {fourcc_text!r}"
            )
        try:
            for position, frame_index in enumerate(indices):
                frame = first if position == 0 else render_episode_frame(
                    episode,
                    frame_index,
                    mode=mode,
                    style=style,
                    width=width,
                    temporal_smoothing=temporal_smoothing,
                    show_text=show_text,
                )
                if frame.shape[1] != video_width or frame.shape[0] != video_height:
                    frame = cv2.copyMakeBorder(
                        frame,
                        0,
                        video_height - frame.shape[0],
                        0,
                        video_width - frame.shape[1],
                        cv2.BORDER_CONSTANT,
                        value=(8, 8, 8),
                    )
                writer.write(frame)
        finally:
            writer.release()
        exports.append(
            VideoExport(
                path=path,
                style=style,
                mode=mode,
                frame_count=len(indices),
                fps=output_fps,
                size=(video_width, video_height),
            )
        )
    return exports


def export_showcase_30(
    episode: TactileEpisode,
    output: str | Path,
    *,
    width: int = DEFAULT_WIDTH,
    fps: float | None = None,
    codec: str | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    frame_step: int = 1,
    temporal_smoothing: float = 0.0,
) -> list[VideoExport]:
    """Export the curated 30-item presentation review set."""

    base = Path(output).expanduser().resolve()
    if base.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
        base = base.with_suffix(".mp4")
    exports: list[VideoExport] = []
    for index, (style, mode) in enumerate(SHOWCASE_30, 1):
        path = base.with_name(
            f"{base.stem}--v{index:02d}--{style}--{mode}{base.suffix}"
        )
        exports.extend(
            export_episode_videos(
                episode,
                path,
                styles=style,
                modes=mode,
                width=width,
                fps=fps,
                codec=codec,
                start_frame=start_frame,
                end_frame=end_frame,
                frame_step=frame_step,
                temporal_smoothing=temporal_smoothing,
            )
        )
    return exports


def export_showcase_3d_10(
    episode: TactileEpisode,
    output: str | Path,
    *,
    width: int = DEFAULT_WIDTH,
    fps: float | None = None,
    codec: str | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    frame_step: int = 1,
) -> list[VideoExport]:
    """Export ten refined turbo 3D candidates with subtle visual smoothing."""

    base = Path(output).expanduser().resolve()
    if base.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
        base = base.with_suffix(".mp4")
    exports: list[VideoExport] = []
    for index, (style, mode, smoothing) in enumerate(SHOWCASE_3D_10, 1):
        path = base.with_name(
            f"{base.stem}--v{index:02d}--{style}--{mode}--smooth{smoothing:.0%}{base.suffix}"
        )
        exports.extend(
            export_episode_videos(
                episode,
                path,
                styles=style,
                modes=mode,
                width=width,
                fps=fps,
                codec=codec,
                start_frame=start_frame,
                end_frame=end_frame,
                frame_step=frame_step,
                temporal_smoothing=smoothing,
            )
        )
    return exports


def _encode_jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("Could not encode viewer frame")
    return encoded.tobytes()


_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>DM-Tac Viewer</title>
<style>html,body{margin:0;background:#090c10;color:#e8edf2;font:14px system-ui}main{max-width:1500px;margin:auto;padding:12px}.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px}button,select,input{background:#18212b;color:#e8edf2;border:1px solid #394857;border-radius:5px;padding:6px}input[type=range]{flex:1;min-width:260px}img{display:block;width:100%;height:auto;background:#111923}.muted{color:#91a3b3}kbd{background:#222d38;padding:2px 5px;border-radius:3px}</style></head>
<body><main><div class="bar"><button id="play">Play</button><select id="mode"><option value="current">Current</option><option value="delta">Previous-frame difference</option><option value="baseline">First-frame difference</option></select><select id="style"><option value="surface_turbo_top">Standard parallel jaws (top)</option><option value="neon">Neon heat + arrows</option><option value="neon_focus">Neon contact focus</option><option value="neon_contour">Neon contours</option><option value="components">Fx / Fy / -Fz components</option><option value="vectors">Vector field</option><option value="surface">3D right oblique</option><option value="surface_left">3D left oblique</option><option value="surface_front">3D front</option><option value="split">Neon + 3D split</option><option value="classic">Classic heat + arrows</option><option value="contour">Legacy contours</option></select><input id="slider" type="range" min="0" max="__MAX__" value="__START__"><span id="counter"></span></div><img id="view" alt="four tactile panels"><p class="muted">Keys: <kbd>space</kbd> play/pause, <kbd>←</kbd>/<kbd>→</kbd> step, <kbd>home</kbd>/<kbd>end</kbd>. __SUMMARY__</p></main>
<script>const N=__COUNT__,fps=__FPS__,initialMode="__INITIAL_MODE__",initialStyle="__INITIAL_STYLE__";let frame=__START__,playing=false,timer=null;const s=document.querySelector('#slider'),v=document.querySelector('#view'),m=document.querySelector('#mode'),t=document.querySelector('#style'),c=document.querySelector('#counter'),b=document.querySelector('#play');m.value=initialMode;t.value=initialStyle;function draw(){s.value=frame;c.textContent=`${frame} / ${N-1}`;v.src=`/frame.jpg?index=${frame}&mode=${m.value}&style=${t.value}&_=${Date.now()}`}function stop(){playing=false;clearTimeout(timer);timer=null;b.textContent='Play'}function queue(){if(!playing)return;clearTimeout(timer);timer=setTimeout(()=>{frame=(frame+1)%N;draw()},Math.max(20,1000/fps))}function play(){if(playing){stop();return}playing=true;b.textContent='Pause';queue()}v.onload=queue;v.onerror=queue;b.onclick=play;s.oninput=()=>{stop();frame=+s.value;draw()};m.onchange=draw;t.onchange=draw;document.onkeydown=e=>{if(e.key===' '){e.preventDefault();play()}else if(e.key==='ArrowLeft'){stop();frame=Math.max(0,frame-1);draw()}else if(e.key==='ArrowRight'){stop();frame=Math.min(N-1,frame+1);draw()}else if(e.key==='Home'){stop();frame=0;draw()}else if(e.key==='End'){stop();frame=N-1;draw()}};draw();</script></body></html>"""


def _serve(
    episode: TactileEpisode,
    *,
    host: str,
    port: int,
    start: int,
    width: int,
    mode: str,
    style: str,
    temporal_smoothing: float,
    show_text: bool,
    open_browser: bool,
) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path in {"/", "/index.html"}:
                    summary = (
                        f"{episode.schema}; source={episode.tactile_source}; "
                        f"{len(episode.channels)} channels; {episode.frame_count} frames"
                    )
                    body = (_HTML.replace("__MAX__", str(episode.frame_count - 1)).replace("__START__", str(start)).replace("__COUNT__", str(episode.frame_count)).replace("__FPS__", repr(episode.nominal_fps)).replace("__INITIAL_MODE__", mode).replace("__INITIAL_STYLE__", style).replace("__SUMMARY__", summary)).encode("utf-8")
                    self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "text/html; charset=utf-8")
                elif parsed.path == "/frame.jpg":
                    query = parse_qs(parsed.query)
                    index = int(query.get("index", [start])[0])
                    requested_mode = query.get("mode", [mode])[0]
                    requested_style = query.get("style", [style])[0]
                    body = _encode_jpeg(render_episode_frame(episode, index, mode=requested_mode, style=requested_style, width=width, temporal_smoothing=temporal_smoothing, show_text=show_text))
                    self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "image/jpeg")
                elif parsed.path == "/api/summary":
                    body = json.dumps(episode.summary(), ensure_ascii=False).encode("utf-8")
                    self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "application/json; charset=utf-8")
                else:
                    self.send_error(HTTPStatus.NOT_FOUND); return
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # Browsers routinely cancel an in-flight image request when a
                # slider or playback advances.  The response is no longer
                # writable, so this is not a server error and must not trigger
                # a second attempt to send an HTTP error response.
                return
            except (ValueError, OSError) as exc:
                try:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                except (BrokenPipeError, ConnectionResetError):
                    return

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"DM-Tac viewer: {url}")
    if open_browser:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _apply_visualization_profile(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    if args.profile is None:
        return args
    profile = VISUALIZATION_PROFILES[args.profile]
    if args.export_showcase_30 or args.export_showcase_3d_10:
        parser.error("--profile cannot be combined with showcase export presets")
    # A named profile is intentionally locked: command-line presentation
    # defaults or overrides cannot silently produce a different-looking video.
    args.tactile_source = profile.tactile_source
    args.style = profile.style
    args.mode = profile.mode
    args.width = profile.width
    args.temporal_smoothing = profile.temporal_smoothing
    args.export_styles = profile.style
    args.export_modes = profile.mode
    return args


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize four tactile parts from a DexTeleop dataset episode.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", help="Dataset root, episode parquet, tactile manifest, or tactile episode directory")
    parser.add_argument(
        "--profile",
        choices=tuple(VISUALIZATION_PROFILES),
        default=None,
        help="Locked presentation profile; standard reproduces the approved parallel-jaw top video",
    )
    parser.add_argument("--episode", type=int, help="Episode index (required for a multi-episode dataset root)")
    parser.add_argument(
        "--tactile-source",
        choices=("auto", "raw", "force"),
        default="auto",
        help="auto prefers an existing tactile_force stream, then falls back to recorded raw/inline data",
    )
    parser.add_argument(
        "--derive-force",
        action="store_true",
        help="Offline-solve raw images into tactile_force before opening the viewer",
    )
    parser.add_argument("--force-backend", choices=("cpu", "cuda", "Flux", "flux"), default="cpu")
    parser.add_argument("--force-mode", choices=("standard", "high"), default="standard")
    parser.add_argument("--force-jobs", type=int, default=0)
    parser.add_argument("--force-sdk-root", type=Path, default=None)
    parser.add_argument(
        "--force-base-frame-index",
        type=int,
        default=None,
        help="Explicit episode frame override; otherwise fixed per-sensor calibration is auto-loaded when present",
    )
    parser.add_argument(
        "--force-calibration-policy",
        choices=("auto", "required", "off"),
        default="auto",
        help="Load fixed per-sensor calibration when available, require it, or disable it",
    )
    parser.add_argument("--force-calibration-dir", type=Path, default=None)
    parser.add_argument(
        "--force-save-calibration",
        action="store_true",
        help="Persist --force-base-frame-index as the fixed calibration for each sensor",
    )
    parser.add_argument("--force-compression", choices=("zstd", "none"), default="zstd")
    parser.add_argument("--force-compression-level", type=int, default=3)
    parser.add_argument("--force-chunk-frames", type=int, default=16)
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--snapshot", type=Path, help="Write one PNG/JPEG and exit")
    parser.add_argument("--frame", type=int, default=0, help="Snapshot frame")
    parser.add_argument("--mode", choices=("current", "delta", "baseline"), default="current")
    parser.add_argument(
        "--style",
        choices=VISUAL_STYLES,
        default="classic",
        help="Presentation style used by snapshots (browser playback can switch styles live)",
    )
    parser.add_argument(
        "--export-video",
        type=Path,
        help="Export video and exit; multiple style/mode selections add suffixes to this path",
    )
    parser.add_argument(
        "--export-showcase-30",
        action="store_true",
        help="With --export-video, write the curated 30-item presentation review set",
    )
    parser.add_argument(
        "--export-showcase-3d-10",
        action="store_true",
        help="With --export-video, write ten refined smoothed turbo 3D candidates",
    )
    parser.add_argument(
        "--export-styles",
        default=None,
        help="Comma-separated styles or 'all'; defaults to --style",
    )
    parser.add_argument(
        "--export-modes",
        default=None,
        help="Comma-separated current,delta,baseline or 'all'; defaults to --mode",
    )
    parser.add_argument("--export-fps", type=float, default=None, help="Video FPS; dataset nominal FPS by default")
    parser.add_argument("--video-codec", default=None, help="Four-character OpenCV FourCC (mp4v for MP4, MJPG for AVI)")
    parser.add_argument("--end-frame", type=int, default=None, help="Inclusive final frame for video export")
    parser.add_argument("--frame-step", type=int, default=1, help="Export every Nth frame")
    parser.add_argument(
        "--temporal-smoothing",
        type=float,
        default=0.0,
        help="Display-only symmetric neighbor blend in [0, 0.48]; 0.18 is a subtle video setting",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Hide all text in 3D surface output while retaining the color ramp",
    )
    parser.add_argument("--inspect", action="store_true", help="Print resolved stream metadata as JSON and exit")
    return _apply_visualization_profile(parser.parse_args(argv), parser)


def _resolve_episode_context(path: str | Path, episode: int | None) -> tuple[Path, int]:
    input_path = Path(path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    root = _dataset_root_from_path(input_path)
    if root is None:
        raise ValueError(f"Could not locate the dataset root for {input_path}")
    episode_index = episode if episode is not None else _episode_index_from_path(input_path)
    if episode_index is None:
        records = _read_jsonl(root / "meta" / "episodes.jsonl")
        if len(records) == 1:
            episode_index = int(records[0].get("episode_index", 0))
        else:
            available = [int(row.get("episode_index", -1)) for row in records]
            raise ValueError(f"--episode is required; available episodes: {available}")
    return root, int(episode_index)


def _derive_force_for_viewer(args: argparse.Namespace) -> tuple[Path, int]:
    if args.tactile_source == "raw":
        raise ValueError("--derive-force cannot be combined with --tactile-source raw")
    root, episode_index = _resolve_episode_context(args.path, args.episode)
    from utils.dataset_postprocess.DM_force_calculate import ForceProcessConfig, process_dataset

    result = process_dataset(
        ForceProcessConfig(
            dataset_root=root,
            output_root=root,
            episodes=(episode_index,),
            jobs=int(args.force_jobs),
            backend=str(args.force_backend),
            mode=str(args.force_mode),
            base_frame_index=args.force_base_frame_index,
            calibration_policy=str(args.force_calibration_policy),
            calibration_dir=args.force_calibration_dir,
            save_calibration=bool(args.force_save_calibration),
            compression=str(args.force_compression),
            compression_level=int(args.force_compression_level),
            chunk_frames=int(args.force_chunk_frames),
            sdk_root=args.force_sdk_root,
            overwrite=bool(args.force_overwrite),
            progress=True,
        )
    )
    if result.failed:
        errors = [
            item.error
            for item in result.stream_results
            if item.status == "failed" and item.error
        ]
        detail = "; ".join(errors[:3])
        raise RuntimeError(
            f"DM-Tac force derivation failed for {result.failed}/{result.discovered} streams"
            + (f": {detail}" if detail else "")
        )
    force_paths = _derived_force_manifest_paths(root, episode_index)
    if not force_paths:
        raise RuntimeError(
            "Force derivation produced no tactile_force manifest. "
            "Confirm the episode contains uint8 [480, 640] raw DM-Tac streams."
        )
    print(
        "DM-Tac force derivation: "
        f"processed={result.processed} skipped={result.skipped} "
        f"manifests={len(force_paths)} base_frame={args.force_base_frame_index}"
    )
    return root, episode_index


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.profile is not None:
        profile = VISUALIZATION_PROFILES[args.profile]
        print(
            "DM-Tac visualization profile: "
            f"{profile.name}/{profile.version} source={profile.tactile_source} "
            f"style={profile.style} mode={profile.mode} "
            f"width={profile.width} smoothing={profile.temporal_smoothing:.2f}"
        )
    input_path: str | Path = args.path
    episode_index = args.episode
    tactile_source = args.tactile_source
    if args.derive_force:
        input_path, episode_index = _derive_force_for_viewer(args)
        tactile_source = "force"
    try:
        episode = load_tactile_episode(
            input_path,
            episode=episode_index,
            tactile_source=tactile_source,
        )
    except MissingDerivedForceError as exc:
        print(f"DM-Tac viewer: {exc}", file=sys.stderr)
        return 2
    if args.inspect:
        print(json.dumps(episode.summary(), ensure_ascii=False, indent=2))
        return 0
    if args.snapshot:
        output = args.snapshot.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(output),
            render_episode_frame(
                episode,
                args.frame,
                mode=args.mode,
                style=args.style,
                width=args.width,
                temporal_smoothing=args.temporal_smoothing,
                show_text=not args.no_text,
            ),
        ):
            raise RuntimeError(f"Could not write snapshot: {output}")
        print(f"Wrote {output}")
        return 0
    if args.export_video:
        if args.export_showcase_30 and args.export_showcase_3d_10:
            raise ValueError("Choose only one showcase export preset")
        if args.export_showcase_3d_10:
            exporter = export_showcase_3d_10
        elif args.export_showcase_30:
            exporter = export_showcase_30
        else:
            exporter = export_episode_videos
        export_kwargs: dict[str, Any] = {
            "width": args.width,
            "fps": args.export_fps,
            "codec": args.video_codec,
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "frame_step": args.frame_step,
        }
        if args.export_showcase_3d_10:
            pass
        elif args.export_showcase_30:
            export_kwargs["temporal_smoothing"] = args.temporal_smoothing
        else:
            export_kwargs["temporal_smoothing"] = args.temporal_smoothing
            export_kwargs["show_text"] = not args.no_text
            export_kwargs.update(
                styles=args.export_styles or args.style,
                modes=args.export_modes or args.mode,
            )
        exports = exporter(episode, args.export_video, **export_kwargs)
        for item in exports:
            print(
                f"Wrote {item.path} | style={item.style} mode={item.mode} "
                f"frames={item.frame_count} fps={item.fps:.3g} size={item.size[0]}x{item.size[1]}"
            )
        return 0
    start = min(max(0, args.start_frame), episode.frame_count - 1)
    _serve(
        episode,
        host=args.host,
        port=args.port,
        start=start,
        width=args.width,
        mode=args.mode,
        style=args.style,
        temporal_smoothing=args.temporal_smoothing,
        show_text=not args.no_text,
        open_browser=not args.no_open,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
