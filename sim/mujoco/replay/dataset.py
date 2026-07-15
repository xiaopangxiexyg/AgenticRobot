from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


Row = Mapping[str, Any]
EpisodeRecord = Mapping[str, Any]


@dataclass(frozen=True)
class ResolvedEpisode:
    dataset_root: Path
    parquet_path: Path
    episode_index: int | None
    episode_id: str
    record: EpisodeRecord | None = None


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {jsonl_path}:{line_number}: {exc}") from exc
        if isinstance(value, dict):
            records.append(value)
    return records


def _episode_index_from_token(token: str) -> int:
    value = str(token).strip()
    if value.startswith("episode_"):
        value = value.removeprefix("episode_")
    return int(value)


def _episode_id_from_index(index: int) -> str:
    return f"episode_{int(index):06d}"


def resolve_episode(dataset_root: str | Path, episode: str | int) -> ResolvedEpisode:
    root = Path(dataset_root).expanduser().resolve()
    token = str(episode)
    explicit = Path(token).expanduser()
    if explicit.exists():
        parquet_path = explicit.resolve()
        episode_id = parquet_path.stem
        index = None
        if episode_id.startswith("episode_"):
            try:
                index = _episode_index_from_token(episode_id)
            except ValueError:
                index = None
        return ResolvedEpisode(
            dataset_root=root,
            parquet_path=parquet_path,
            episode_index=index,
            episode_id=episode_id,
            record=None,
        )
    if not explicit.is_absolute():
        candidate = (root / explicit).resolve()
        if candidate.exists():
            return resolve_episode(root, str(candidate))

    try:
        episode_index = _episode_index_from_token(token)
    except ValueError as exc:
        raise FileNotFoundError(
            f"Episode {episode!r} is not an existing parquet path or episode index/id."
        ) from exc

    episode_id = _episode_id_from_index(episode_index)
    records = read_jsonl(root / "meta" / "episodes.jsonl")
    for record in reversed(records):
        if int(record.get("episode_index", -1)) == episode_index or str(record.get("episode_id", "")) == episode_id:
            relative_path = record.get("parquet_path")
            if not relative_path:
                raise FileNotFoundError(f"Episode record {episode_id} has no parquet_path.")
            return ResolvedEpisode(
                dataset_root=root,
                parquet_path=(root / str(relative_path)).resolve(),
                episode_index=episode_index,
                episode_id=episode_id,
                record=record,
            )

    candidates = sorted(root.glob(f"data/chunk-*/{episode_id}.parquet"))
    if candidates:
        return ResolvedEpisode(
            dataset_root=root,
            parquet_path=candidates[-1].resolve(),
            episode_index=episode_index,
            episode_id=episode_id,
            record=None,
        )
    raise FileNotFoundError(f"Could not find {episode_id} under {root}.")


def read_parquet_rows(path: str | Path, *, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Dataset replay requires pyarrow in the active environment.") from exc

    parquet_path = Path(path).expanduser().resolve()
    table = pq.read_table(parquet_path, columns=list(columns) if columns is not None else None)
    rows = table.to_pylist()
    rows.sort(key=lambda row: int(row.get("frame_index", row.get("index", 0)) or 0))
    return rows


def parquet_column_names(path: str | Path) -> tuple[str, ...]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Dataset replay requires pyarrow in the active environment.") from exc

    parquet_path = Path(path).expanduser().resolve()
    return tuple(str(name) for name in pq.read_schema(parquet_path).names)


def row_timestamp(row: Row) -> float | None:
    value = row.get("timestamp")
    if value is None:
        return None
    try:
        timestamp = float(value)
    except Exception:
        return None
    if not np.isfinite(timestamp):
        return None
    return timestamp


def fallback_hz(episode: ResolvedEpisode) -> float:
    record = episode.record
    if record is not None:
        metadata = record.get("metadata")
        if isinstance(metadata, Mapping):
            for key in ("hz", "nominal_fps", "fps"):
                value = metadata.get(key)
                try:
                    hz = float(value)
                except Exception:
                    continue
                if np.isfinite(hz) and hz > 0.0:
                    return hz
        frame_count = record.get("frame_count")
        duration_sec = record.get("duration_sec")
        try:
            frames = float(frame_count)
            duration = float(duration_sec)
        except Exception:
            pass
        else:
            if np.isfinite(frames) and np.isfinite(duration) and duration > 0.0:
                return max(1.0, frames / duration)
    info_path = Path(episode.dataset_root) / "meta" / "info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            info = {}
        if isinstance(info, Mapping):
            for key in ("nominal_fps", "hz", "fps"):
                value = info.get(key)
                try:
                    hz = float(value)
                except Exception:
                    continue
                if np.isfinite(hz) and hz > 0.0:
                    return hz
    return 30.0
