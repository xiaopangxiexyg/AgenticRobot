from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


Row = Mapping[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _require_parquet() -> Any:
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Replay dataset loading requires pyarrow in the active Python environment."
        ) from exc
    return pq


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return float(number)


def _positive_float_or_none(value: Any) -> float | None:
    number = _float_or_none(value)
    if number is None or number <= 0.0:
        return None
    return float(number)


def _episode_id_from_index(index: int) -> str:
    return f"episode_{int(index):06d}"


def _chunk_id_from_episode_index(index: int, *, chunk_size: int) -> str:
    return f"chunk-{int(index) // max(1, int(chunk_size)):03d}"


def _parse_episode_index(token: Any) -> int | None:
    value = str(token).strip()
    if value.startswith("episode_"):
        value = value.removeprefix("episode_")
    try:
        return int(value)
    except ValueError:
        return None


def _episode_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("episode_"):
        return stem
    index = _parse_episode_index(stem)
    if index is not None:
        return _episode_id_from_index(index)
    return stem


def _episode_index_from_id(episode_id: str) -> int | None:
    if episode_id.startswith("episode_"):
        return _parse_episode_index(episode_id)
    return None


def _resolve_root_path(root: Path, value: Any) -> Path | None:
    if value is None or not str(value):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _relative_text(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _mapping_strings(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items() if item is not None}


def _row_frame_index(row: Row, fallback: int) -> int:
    for key in ("frame_index", "index"):
        value = _int_or_none(row.get(key))
        if value is not None:
            return value
    return int(fallback)


def _row_timestamp(row: Row) -> float | None:
    for key in ("timestamp", "observation.timestamp"):
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _camera_keys_from_rows(rows: Sequence[Row]) -> tuple[str, ...]:
    prefix = "observation.images."
    keys: set[str] = set()
    for row in rows:
        for column in row:
            if not column.startswith(prefix):
                continue
            suffix = column[len(prefix) :]
            if "." not in suffix:
                continue
            camera_key, _field = suffix.split(".", 1)
            if camera_key:
                keys.add(camera_key)
    return tuple(sorted(keys))


def _tactile_sides_from_rows(rows: Sequence[Row]) -> tuple[str, ...]:
    prefix = "observation.hand."
    marker = ".tactile"
    sides: set[str] = set()
    for row in rows:
        for column in row:
            if not column.startswith(prefix) or marker not in column:
                continue
            side = column[len(prefix) : column.find(marker)]
            if side:
                sides.add(side)
    return tuple(sorted(sides))


def _timestamps_to_frame_times(rows: Sequence[Row], fallback_hz: float) -> tuple[float, ...]:
    timestamps = [_row_timestamp(row) for row in rows]
    if timestamps and timestamps[0] is not None:
        first = float(timestamps[0])
        frame_times: list[float] = []
        for index, timestamp in enumerate(timestamps):
            if timestamp is None:
                frame_times.append(float(index) / float(fallback_hz))
            else:
                frame_times.append(max(0.0, float(timestamp) - first))
        return tuple(frame_times)
    return tuple(float(index) / float(fallback_hz) for index in range(len(rows)))


def _infer_hz_from_rows(rows: Sequence[Row]) -> dict[str, Any] | None:
    timestamps = np.asarray(
        [float(value) for value in (_row_timestamp(row) for row in rows) if value is not None],
        dtype=float,
    )
    if timestamps.shape[0] >= 2:
        dts = np.diff(timestamps)
        dts = dts[np.isfinite(dts) & (dts > 1e-9)]
        if dts.size:
            return {
                "hz": float(1.0 / float(np.median(dts))),
                "source": "parquet.timestamp",
                "interval_count": int(dts.size),
            }
    control_hz = np.asarray(
        [
            float(value)
            for value in (_positive_float_or_none(row.get("control_hz")) for row in rows)
            if value is not None
        ],
        dtype=float,
    )
    control_hz = control_hz[np.isfinite(control_hz) & (control_hz > 0.0)]
    if control_hz.size:
        return {
            "hz": float(np.median(control_hz)),
            "source": "parquet.control_hz",
            "sample_count": int(control_hz.size),
        }
    return None


@dataclass
class EpisodeSummary:
    episode_id: str
    parquet_path: Path
    dataset_root: Path
    episode_index: int | None = None
    frame_count: int | None = None
    duration_sec: float | None = None
    video_paths: dict[str, str] = field(default_factory=dict)
    tactile_paths: dict[str, str] = field(default_factory=dict)
    camera_keys: tuple[str, ...] = ()
    tactile_sides: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    record: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"

    @property
    def exists(self) -> bool:
        return self.parquet_path.exists()

    @property
    def parquet_relative_path(self) -> str:
        return _relative_text(self.dataset_root, self.parquet_path)

    def matches(self, token: Any) -> bool:
        text = str(token).strip()
        if not text:
            return False
        if text == self.episode_id:
            return True
        parsed = _parse_episode_index(text)
        if parsed is not None and self.episode_index is not None:
            return int(parsed) == int(self.episode_index)
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = self.dataset_root / candidate
        try:
            return candidate.resolve() == self.parquet_path.resolve()
        except OSError:
            return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "episode_id": self.episode_id,
            "frame_count": self.frame_count,
            "duration_sec": self.duration_sec,
            "parquet_path": self.parquet_relative_path,
            "parquet_abs_path": str(self.parquet_path),
            "exists": self.exists,
            "video_paths": dict(self.video_paths),
            "tactile_paths": dict(self.tactile_paths),
            "camera_keys": list(self.camera_keys),
            "tactile_sides": list(self.tactile_sides),
            "metadata": dict(self.metadata),
            "stats": dict(self.stats),
            "source": self.source,
        }


@dataclass
class LoadedReplayEpisode:
    dataset_root: Path
    summary: EpisodeSummary
    rows: list[dict[str, Any]]
    columns: tuple[str, ...]
    nominal_hz: float
    timing_source: str
    frame_times_sec: tuple[float, ...]

    @property
    def frame_count(self) -> int:
        return len(self.rows)

    @property
    def duration_sec(self) -> float:
        if self.summary.duration_sec is not None:
            return float(self.summary.duration_sec)
        if self.frame_times_sec:
            return float(self.frame_times_sec[-1])
        return 0.0

    def row_at(self, frame_index: int) -> dict[str, Any]:
        if not self.rows:
            raise IndexError("loaded replay episode has no rows")
        index = max(0, min(int(frame_index), len(self.rows) - 1))
        return self.rows[index]

    def to_dict(self) -> dict[str, Any]:
        payload = self.summary.to_dict()
        payload.update(
            {
                "frame_count": self.frame_count,
                "duration_sec": self.duration_sec,
                "nominal_hz": self.nominal_hz,
                "timing_source": self.timing_source,
                "columns": list(self.columns),
            }
        )
        return payload


class ReplayDatasetIndex:
    def __init__(self, dataset_root: str | Path):
        self.root = Path(dataset_root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Replay dataset root does not exist: {self.root}")
        self.info = _read_json(self.root / "meta" / "info.json")
        self.episode_records = _read_jsonl(self.root / "meta" / "episodes.jsonl")
        self.episode_stats = {
            int(index): row
            for row in _read_jsonl(self.root / "meta" / "episodes_stats.jsonl")
            if (index := _int_or_none(row.get("episode_index"))) is not None
        }
        self._episodes = self._build_episode_summaries()

    def _build_episode_summaries(self) -> list[EpisodeSummary]:
        summaries: list[EpisodeSummary] = []
        seen_paths: set[Path] = set()
        for record in self.episode_records:
            episode_index = _int_or_none(record.get("episode_index"))
            episode_id = str(record.get("episode_id") or "")
            if not episode_id and episode_index is not None:
                episode_id = _episode_id_from_index(episode_index)
            parquet_path = _resolve_root_path(self.root, record.get("parquet_path"))
            if parquet_path is None and episode_index is not None:
                parquet_path = self._fallback_parquet_path(episode_index)
            if parquet_path is None:
                continue
            if not episode_id:
                episode_id = _episode_id_from_path(parquet_path)
            if episode_index is None:
                episode_index = _episode_index_from_id(episode_id)
            seen_paths.add(parquet_path)
            video_paths = _mapping_strings(record.get("video_paths"))
            tactile_paths = _mapping_strings(record.get("tactile_paths"))
            metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
            stats = self.episode_stats.get(int(episode_index), {}) if episode_index is not None else {}
            summaries.append(
                EpisodeSummary(
                    episode_id=episode_id,
                    episode_index=episode_index,
                    dataset_root=self.root,
                    parquet_path=parquet_path,
                    frame_count=_int_or_none(record.get("frame_count")) or _int_or_none(stats.get("frame_count")),
                    duration_sec=_float_or_none(record.get("duration_sec"))
                    or _float_or_none(stats.get("duration_sec")),
                    video_paths=video_paths,
                    tactile_paths=tactile_paths,
                    camera_keys=tuple(sorted(video_paths)),
                    tactile_sides=tuple(sorted(tactile_paths)),
                    metadata=dict(metadata),
                    stats=dict(stats),
                    record=dict(record),
                    source="meta/episodes.jsonl",
                )
            )
        for path in sorted(self.root.glob("data/chunk-*/episode_*.parquet")):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            episode_id = _episode_id_from_path(resolved)
            episode_index = _episode_index_from_id(episode_id)
            stats = self.episode_stats.get(int(episode_index), {}) if episode_index is not None else {}
            summaries.append(
                EpisodeSummary(
                    episode_id=episode_id,
                    episode_index=episode_index,
                    dataset_root=self.root,
                    parquet_path=resolved,
                    frame_count=_int_or_none(stats.get("frame_count")),
                    duration_sec=_float_or_none(stats.get("duration_sec")),
                    stats=dict(stats),
                    source="data/chunk-* scan",
                )
            )
        summaries.sort(
            key=lambda item: (
                item.episode_index if item.episode_index is not None else 10**12,
                item.episode_id,
            )
        )
        return summaries

    def _fallback_parquet_path(self, episode_index: int) -> Path:
        chunk_size = _int_or_none(self.info.get("chunk_size")) or 1000
        return (
            self.root
            / "data"
            / _chunk_id_from_episode_index(int(episode_index), chunk_size=chunk_size)
            / f"{_episode_id_from_index(int(episode_index))}.parquet"
        ).resolve()

    def available_episodes(self) -> list[EpisodeSummary]:
        return list(self._episodes)

    def _available_episode_hint(self) -> str:
        if not self._episodes:
            return " No replay episodes are indexed."
        indices = sorted(
            {
                int(summary.episode_index)
                for summary in self._episodes
                if summary.episode_index is not None
            }
        )
        if indices:
            if len(indices) <= 12:
                text = ", ".join(str(index) for index in indices)
            elif all((right - left) == 1 for left, right in zip(indices, indices[1:])):
                text = f"{indices[0]}..{indices[-1]} ({len(indices)} episodes)"
            else:
                head = ", ".join(str(index) for index in indices[:8])
                tail = ", ".join(str(index) for index in indices[-3:])
                text = f"{head}, ..., {tail} ({len(indices)} episodes)"
            return f" Available episode indices: {text}."
        ids = [summary.episode_id for summary in self._episodes[:12]]
        suffix = "" if len(self._episodes) <= 12 else f", ... ({len(self._episodes)} episodes)"
        return f" Available episode ids: {', '.join(ids)}{suffix}."

    def resolve_episode(self, token: Any) -> EpisodeSummary:
        text = str(token).strip()
        if not text:
            if self._episodes:
                return self._episodes[0]
            raise FileNotFoundError(f"No replay episodes found under {self.root}")

        explicit = Path(text).expanduser()
        candidates = [explicit]
        if not explicit.is_absolute():
            candidates.append(self.root / explicit)
        for candidate in candidates:
            if candidate.exists():
                resolved = candidate.resolve()
                for summary in self._episodes:
                    try:
                        if summary.parquet_path.resolve() == resolved:
                            return summary
                    except OSError:
                        continue
                episode_id = _episode_id_from_path(resolved)
                return EpisodeSummary(
                    episode_id=episode_id,
                    episode_index=_episode_index_from_id(episode_id),
                    dataset_root=self.root,
                    parquet_path=resolved,
                    source="explicit_path",
                )

        for summary in self._episodes:
            if summary.matches(text):
                return summary

        episode_index = _parse_episode_index(text)
        if episode_index is not None:
            fallback = self._fallback_parquet_path(episode_index)
            if fallback.exists():
                return EpisodeSummary(
                    episode_id=_episode_id_from_index(episode_index),
                    episode_index=episode_index,
                    dataset_root=self.root,
                    parquet_path=fallback,
                    source="fallback_path",
                )
        raise FileNotFoundError(
            f"Could not resolve replay episode {text!r} under {self.root}.{self._available_episode_hint()}"
        )

    def load_episode(self, token: Any) -> LoadedReplayEpisode:
        summary = self.resolve_episode(token)
        if not summary.parquet_path.exists():
            raise FileNotFoundError(f"Replay episode parquet does not exist: {summary.parquet_path}")
        rows, columns = self._read_parquet_rows(summary.parquet_path)
        if not rows:
            raise ValueError(f"Replay episode contains no rows: {summary.parquet_path}")

        row_camera_keys = _camera_keys_from_rows(rows)
        row_tactile_sides = _tactile_sides_from_rows(rows)
        summary.camera_keys = tuple(sorted(set(summary.camera_keys).union(row_camera_keys)))
        summary.tactile_sides = tuple(sorted(set(summary.tactile_sides).union(row_tactile_sides)))
        summary.frame_count = len(rows)
        if summary.duration_sec is None and len(rows) >= 2:
            first = _row_timestamp(rows[0])
            last = _row_timestamp(rows[-1])
            if first is not None and last is not None:
                summary.duration_sec = max(0.0, float(last) - float(first))

        hz, source = self._infer_nominal_hz(summary, rows)
        frame_times = _timestamps_to_frame_times(rows, hz)
        return LoadedReplayEpisode(
            dataset_root=self.root,
            summary=summary,
            rows=rows,
            columns=columns,
            nominal_hz=hz,
            timing_source=source,
            frame_times_sec=frame_times,
        )

    def _read_parquet_rows(self, path: Path) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
        pq = _require_parquet()
        schema = pq.read_schema(path)
        columns = tuple(str(name) for name in schema.names)
        rows = pq.read_table(path).to_pylist()
        indexed = list(enumerate(rows))
        indexed.sort(key=lambda item: (_row_frame_index(item[1], item[0]), item[0]))
        return [dict(row) for _index, row in indexed], columns

    def _infer_nominal_hz(
        self,
        summary: EpisodeSummary,
        rows: Sequence[Row],
    ) -> tuple[float, str]:
        row_estimate = _infer_hz_from_rows(rows)
        if row_estimate is not None:
            return float(row_estimate["hz"]), str(row_estimate["source"])
        for key in ("average_hz", "nominal_fps", "mean_control_hz"):
            value = _positive_float_or_none(summary.stats.get(key))
            if value is not None:
                return value, f"episodes_stats.{key}"
        if summary.duration_sec is not None and len(rows) >= 2:
            hz = float((len(rows) - 1) / max(1e-9, float(summary.duration_sec)))
            if hz > 0.0 and np.isfinite(hz):
                return hz, "episodes.duration_sec"
        for mapping, prefix in ((summary.metadata, "episodes.metadata"), (self.info, "info")):
            for key in ("hz", "nominal_fps"):
                value = _positive_float_or_none(mapping.get(key))
                if value is not None:
                    return value, f"{prefix}.{key}"
        return 30.0, "default"
