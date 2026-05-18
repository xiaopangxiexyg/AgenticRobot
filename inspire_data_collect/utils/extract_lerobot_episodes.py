from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from inspire_data_collect.common import to_serializable
    from inspire_data_collect.utils.writer import chunk_id_from_episode_index, episode_id_from_index
else:
    from ..common import to_serializable
    from .writer import chunk_id_from_episode_index, episode_id_from_index


MANIFEST_FILENAMES = {"info.json", "episodes.jsonl", "episodes_stats.jsonl"}
MERGE_COMPAT_INFO_KEYS = (
    "format",
    "nominal_fps",
    "chunk_size",
    "write_parquet",
    "write_video",
    "video_codec",
    "features",
    "reserved_spaces",
    "vector_layout",
)


@dataclass(frozen=True)
class EpisodeSelection:
    selected_indices: tuple[int, ...]
    selected_records: tuple[dict[str, Any], ...]
    remaining_records: tuple[dict[str, Any], ...]
    stats_by_episode: dict[int, dict[str, Any]]


@dataclass(frozen=True)
class SplitPlan:
    dataset_root: Path
    output_root: Path
    selected_indices: tuple[int, ...]
    selected_count: int
    remaining_count: int
    selected_frames: int
    remaining_frames: int


@dataclass(frozen=True)
class DatasetManifest:
    root: Path
    info: dict[str, Any]
    episode_records: tuple[dict[str, Any], ...]
    stats_by_episode: dict[int, dict[str, Any]]


@dataclass(frozen=True)
class MergePlan:
    dataset_root: Path
    merge_root: Path
    output_root: Path
    first_count: int
    second_count: int
    total_count: int
    first_frames: int
    second_frames: int
    total_frames: int


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_serializable(dict(payload)), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}.")
        rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_serializable(dict(row)), ensure_ascii=True) + "\n")


def parse_episode_indices(text: str) -> tuple[int, ...]:
    """Parse comma-separated episode indices and inclusive ranges.

    Examples:
      0
      0,2,5
      0-3,8
    """

    indices: set[int] = set()
    for raw_part in str(text or "").split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start < 0 or end < 0:
                raise ValueError("Episode indices must be non-negative.")
            if end < start:
                raise ValueError(f"Invalid episode range {part!r}: end is smaller than start.")
            indices.update(range(start, end + 1))
        else:
            index = int(part)
            if index < 0:
                raise ValueError("Episode indices must be non-negative.")
            indices.add(index)
    if not indices:
        raise ValueError("At least one episode index is required.")
    return tuple(sorted(indices))


def _episode_index(row: Mapping[str, Any]) -> int:
    try:
        return int(row["episode_index"])
    except KeyError as exc:
        raise ValueError("Episode manifest row is missing episode_index.") from exc


def load_dataset_manifest(dataset_root: Path) -> DatasetManifest:
    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    stats_path = dataset_root / "meta" / "episodes_stats.jsonl"

    source_info = _read_json(info_path)
    episode_rows = _read_jsonl(episodes_path)
    if not episode_rows:
        raise ValueError(f"No episodes found in {episodes_path}.")

    seen_episodes: set[int] = set()
    for row in episode_rows:
        index = _episode_index(row)
        if index in seen_episodes:
            raise ValueError(f"Duplicate episode_index {index} in {episodes_path}.")
        seen_episodes.add(index)

    stats_by_episode: dict[int, dict[str, Any]] = {}
    for row in _read_jsonl(stats_path):
        if "episode_index" not in row:
            continue
        index = _episode_index(row)
        if index in stats_by_episode:
            raise ValueError(f"Duplicate episode_index {index} in {stats_path}.")
        stats_by_episode[index] = row

    return DatasetManifest(
        root=dataset_root,
        info=source_info,
        episode_records=tuple(episode_rows),
        stats_by_episode=stats_by_episode,
    )


def load_selection(dataset_root: Path, selected_indices: Sequence[int]) -> EpisodeSelection:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    episode_rows = _read_jsonl(episodes_path)
    if not episode_rows:
        raise ValueError(f"No episodes found in {episodes_path}.")

    requested = set(int(index) for index in selected_indices)
    by_index: dict[int, dict[str, Any]] = {}
    for row in episode_rows:
        index = _episode_index(row)
        if index in by_index:
            raise ValueError(f"Duplicate episode_index {index} in {episodes_path}.")
        by_index[index] = row

    missing = sorted(requested - set(by_index))
    if missing:
        raise ValueError(f"Requested episode(s) not found: {missing}")

    stats_by_episode = {
        _episode_index(row): row
        for row in _read_jsonl(stats_path)
        if "episode_index" in row
    }
    selected_records = tuple(by_index[index] for index in sorted(requested))
    remaining_records = tuple(row for row in episode_rows if _episode_index(row) not in requested)
    return EpisodeSelection(
        selected_indices=tuple(sorted(requested)),
        selected_records=selected_records,
        remaining_records=remaining_records,
        stats_by_episode=stats_by_episode,
    )


def make_plan(dataset_root: Path, output_root: Path, selected_indices: Sequence[int]) -> SplitPlan:
    selection = load_selection(dataset_root, selected_indices)
    selected_frames = sum(int(row.get("frame_count", 0) or 0) for row in selection.selected_records)
    remaining_frames = sum(int(row.get("frame_count", 0) or 0) for row in selection.remaining_records)
    return SplitPlan(
        dataset_root=dataset_root,
        output_root=output_root,
        selected_indices=tuple(selected_indices),
        selected_count=len(selection.selected_records),
        remaining_count=len(selection.remaining_records),
        selected_frames=int(selected_frames),
        remaining_frames=int(remaining_frames),
    )


def _ordered_episode_records(records: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(sorted(records, key=_episode_index))


def _episode_frame_total(records: Sequence[Mapping[str, Any]]) -> int:
    return int(sum(int(row.get("frame_count", 0) or 0) for row in records))


def _dataset_video_keys(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    keys: set[str] = set()
    for record in records:
        video_paths = record.get("video_paths") or {}
        if not isinstance(video_paths, Mapping):
            continue
        keys.update(str(key) for key in video_paths)
    return tuple(sorted(keys))


def _dataset_video_prefixes(records: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    prefixes: dict[str, set[str]] = {}
    for record in records:
        video_paths = record.get("video_paths") or {}
        if not isinstance(video_paths, Mapping):
            continue
        for key, relative_path in video_paths.items():
            if not relative_path:
                continue
            path = Path(str(relative_path))
            prefixes.setdefault(str(key), set()).add(str(path.parent.parent))
    return {key: tuple(sorted(value)) for key, value in sorted(prefixes.items())}


def _first_parquet_schema(manifest: DatasetManifest) -> Any:
    _, pq = _require_pyarrow()
    for record in _ordered_episode_records(manifest.episode_records):
        old_episode_index = _episode_index(record)
        parquet_path = record.get("parquet_path")
        if not parquet_path:
            raise ValueError(f"Episode {old_episode_index} in {manifest.root} has no parquet_path.")
        source_path = manifest.root / str(parquet_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Missing parquet for episode {old_episode_index}: {source_path}")
        return pq.read_schema(source_path)
    raise ValueError(f"No episodes found in {manifest.root}.")


def _validate_merge_compatible(first: DatasetManifest, second: DatasetManifest) -> None:
    missing = object()
    for key in MERGE_COMPAT_INFO_KEYS:
        first_value = first.info.get(key, missing)
        second_value = second.info.get(key, missing)
        if first_value != second_value:
            raise ValueError(f"Datasets are not compatible: meta/info.json key {key!r} differs.")

    first_video_keys = _dataset_video_keys(first.episode_records)
    second_video_keys = _dataset_video_keys(second.episode_records)
    if first_video_keys != second_video_keys:
        raise ValueError(
            "Datasets are not compatible: video camera keys differ "
            f"({list(first_video_keys)} != {list(second_video_keys)})."
        )
    first_video_prefixes = _dataset_video_prefixes(first.episode_records)
    second_video_prefixes = _dataset_video_prefixes(second.episode_records)
    if first_video_prefixes != second_video_prefixes:
        raise ValueError(
            "Datasets are not compatible: video path prefixes differ "
            f"({first_video_prefixes} != {second_video_prefixes})."
        )

    first_schema = _first_parquet_schema(first)
    second_schema = _first_parquet_schema(second)
    if not first_schema.equals(second_schema, check_metadata=False):
        raise ValueError("Datasets are not compatible: parquet schemas differ.")


def make_merge_plan(dataset_root: Path, merge_root: Path, output_root: Path) -> MergePlan:
    first = load_dataset_manifest(dataset_root)
    second = load_dataset_manifest(merge_root)
    _validate_merge_compatible(first, second)
    first_frames = _episode_frame_total(first.episode_records)
    second_frames = _episode_frame_total(second.episode_records)
    return MergePlan(
        dataset_root=dataset_root,
        merge_root=merge_root,
        output_root=output_root,
        first_count=len(first.episode_records),
        second_count=len(second.episode_records),
        total_count=len(first.episode_records) + len(second.episode_records),
        first_frames=int(first_frames),
        second_frames=int(second_frames),
        total_frames=int(first_frames + second_frames),
    )


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Episode extraction/merge needs pyarrow to rewrite parquet metadata. "
            "Install pyarrow in the active Python environment."
        ) from exc
    return pa, pq


def _copy_static_meta(source_root: Path, target_root: Path) -> None:
    source_meta = source_root / "meta"
    target_meta = target_root / "meta"
    target_meta.mkdir(parents=True, exist_ok=True)
    if not source_meta.exists():
        return
    for source_path in source_meta.iterdir():
        if source_path.name in MANIFEST_FILENAMES:
            continue
        target_path = target_meta / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, target_path)
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)


def _chunk_size(info: Mapping[str, Any]) -> int:
    try:
        value = int(info.get("chunk_size", 1000) or 1000)
    except (TypeError, ValueError):
        value = 1000
    if value <= 0:
        value = 1000
    return value


def _episode_parquet_relative_path(episode_index: int, *, chunk_size: int) -> str:
    return str(
        Path("data")
        / chunk_id_from_episode_index(int(episode_index), chunk_size=chunk_size)
        / f"{episode_id_from_index(int(episode_index))}.parquet"
    )


def _episode_video_relative_path(
    camera_key: str,
    episode_index: int,
    *,
    chunk_size: int,
    old_relative_path: str,
) -> str:
    old_path = Path(old_relative_path)
    parent_prefix = old_path.parent.parent
    return str(
        parent_prefix
        / chunk_id_from_episode_index(int(episode_index), chunk_size=chunk_size)
        / f"{episode_id_from_index(int(episode_index))}{old_path.suffix or '.mp4'}"
    )


def _updated_video_paths(
    record: Mapping[str, Any],
    new_episode_index: int,
    *,
    chunk_size: int,
) -> dict[str, str]:
    video_paths = record.get("video_paths") or {}
    if not isinstance(video_paths, Mapping):
        return {}
    return {
        str(camera_key): _episode_video_relative_path(
            str(camera_key),
            int(new_episode_index),
            chunk_size=chunk_size,
            old_relative_path=str(relative_path),
        )
        for camera_key, relative_path in video_paths.items()
        if relative_path
    }


def _replace_video_path_values(
    row: dict[str, Any],
    old_video_paths: Mapping[str, str],
    new_video_paths: Mapping[str, str],
) -> None:
    replacements = {
        str(old_path): str(new_video_paths[camera_key])
        for camera_key, old_path in old_video_paths.items()
        if camera_key in new_video_paths and old_path
    }
    if not replacements:
        return
    for key, value in list(row.items()):
        if str(key).endswith(".path") and isinstance(value, str) and value in replacements:
            row[key] = replacements[value]


def _rewrite_episode_parquet(
    *,
    source_path: Path,
    target_path: Path,
    new_episode_index: int,
    sample_index_start: int,
    old_video_paths: Mapping[str, str],
    new_video_paths: Mapping[str, str],
    compression: str,
) -> int:
    pa, pq = _require_pyarrow()
    table = pq.read_table(source_path)
    rows = table.to_pylist()
    for frame_index, row in enumerate(rows):
        if "index" in row:
            row["index"] = int(sample_index_start + frame_index)
        if "episode_index" in row:
            row["episode_index"] = int(new_episode_index)
        if "frame_index" in row:
            row["frame_index"] = int(frame_index)
        _replace_video_path_values(row, old_video_paths, new_video_paths)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    rewritten = pa.Table.from_pylist(rows, schema=table.schema)
    pq.write_table(rewritten, target_path, compression=str(compression or "zstd"))
    return len(rows)


def _copy_episode_videos(
    *,
    source_root: Path,
    target_root: Path,
    old_video_paths: Mapping[str, str],
    new_video_paths: Mapping[str, str],
    allow_missing_videos: bool,
) -> None:
    for camera_key, old_relative_path in old_video_paths.items():
        new_relative_path = new_video_paths.get(camera_key)
        if not new_relative_path:
            continue
        source_path = source_root / str(old_relative_path)
        target_path = target_root / str(new_relative_path)
        if not source_path.exists():
            if allow_missing_videos:
                continue
            raise FileNotFoundError(f"Missing video for camera {camera_key!r}: {source_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def _updated_episode_record(
    record: Mapping[str, Any],
    *,
    new_episode_index: int,
    new_parquet_path: str,
    new_video_paths: Mapping[str, str],
    frame_count: int,
) -> dict[str, Any]:
    updated = dict(record)
    updated["episode_index"] = int(new_episode_index)
    updated["episode_id"] = episode_id_from_index(int(new_episode_index))
    updated["frame_count"] = int(frame_count)
    updated["parquet_path"] = str(new_parquet_path)
    updated["video_paths"] = dict(new_video_paths)
    return updated


def _updated_stats_record(
    record: Mapping[str, Any] | None,
    *,
    episode_record: Mapping[str, Any],
    new_episode_index: int,
    frame_count: int,
) -> dict[str, Any]:
    if record is None:
        updated: dict[str, Any] = {}
        for key in (
            "start_timestamp",
            "end_timestamp",
            "duration_sec",
        ):
            if key in episode_record:
                updated[key] = episode_record[key]
    else:
        updated = dict(record)
    updated["episode_index"] = int(new_episode_index)
    updated["frame_count"] = int(frame_count)
    return updated


def _updated_info(
    source_info: Mapping[str, Any],
    *,
    episode_records: Sequence[Mapping[str, Any]],
    dataset_name: str | None,
    task_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    info = dict(source_info)
    info["episode_count"] = int(len(episode_records))
    info["frame_count"] = int(sum(int(row.get("frame_count", 0) or 0) for row in episode_records))
    if dataset_name is not None:
        info["dataset_name"] = str(dataset_name)
    if task_rows is not None:
        task_names: list[str] = []
        for row in task_rows:
            task = str(row.get("task", "")).strip()
            if task and task not in task_names:
                task_names.append(task)
        if task_names:
            info["tasks"] = task_names
    return info


def _task_dedupe_key(row: Mapping[str, Any]) -> str:
    for key in ("task_id", "task", "description"):
        value = str(row.get(key, "")).strip()
        if value:
            return f"{key}:{value}"
    return json.dumps(to_serializable(dict(row)), ensure_ascii=True, sort_keys=True)


def _merged_task_rows(source_roots: Sequence[Path]) -> tuple[dict[str, Any], ...]:
    source_rows: list[dict[str, Any]] = []
    for source_root in source_roots:
        source_rows.extend(_read_jsonl(source_root / "meta" / "tasks.jsonl"))
    if not source_rows:
        return ()

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    keep_task_index = any("task_index" in row for row in source_rows)
    for source_row in source_rows:
        key = _task_dedupe_key(source_row)
        if key in seen:
            continue
        seen.add(key)
        row = dict(source_row)
        if keep_task_index or "task_index" in row:
            row["task_index"] = int(len(rows))
        rows.append(row)
    return tuple(rows)


def build_subset_dataset(
    *,
    source_root: Path,
    target_root: Path,
    source_info: Mapping[str, Any],
    episode_records: Sequence[Mapping[str, Any]],
    stats_by_episode: Mapping[int, Mapping[str, Any]],
    dataset_name: str | None,
    parquet_compression: str,
    allow_missing_videos: bool,
) -> None:
    target_root.mkdir(parents=True, exist_ok=False)
    (target_root / "data").mkdir(parents=True, exist_ok=True)
    (target_root / "videos").mkdir(parents=True, exist_ok=True)
    _copy_static_meta(source_root, target_root)

    chunk_size = _chunk_size(source_info)
    updated_episode_records: list[dict[str, Any]] = []
    updated_stats_records: list[dict[str, Any]] = []
    sample_index_start = 0

    for new_episode_index, source_record in enumerate(episode_records):
        old_episode_index = _episode_index(source_record)
        old_parquet = source_record.get("parquet_path")
        if not old_parquet:
            raise ValueError(f"Episode {old_episode_index} has no parquet_path.")
        source_parquet_path = source_root / str(old_parquet)
        if not source_parquet_path.exists():
            raise FileNotFoundError(f"Missing parquet for episode {old_episode_index}: {source_parquet_path}")

        new_parquet_path = _episode_parquet_relative_path(
            int(new_episode_index),
            chunk_size=chunk_size,
        )
        old_video_paths = {
            str(key): str(value)
            for key, value in dict(source_record.get("video_paths") or {}).items()
            if value
        }
        new_video_paths = _updated_video_paths(
            source_record,
            int(new_episode_index),
            chunk_size=chunk_size,
        )
        frame_count = _rewrite_episode_parquet(
            source_path=source_parquet_path,
            target_path=target_root / new_parquet_path,
            new_episode_index=int(new_episode_index),
            sample_index_start=int(sample_index_start),
            old_video_paths=old_video_paths,
            new_video_paths=new_video_paths,
            compression=parquet_compression,
        )
        _copy_episode_videos(
            source_root=source_root,
            target_root=target_root,
            old_video_paths=old_video_paths,
            new_video_paths=new_video_paths,
            allow_missing_videos=allow_missing_videos,
        )

        updated_record = _updated_episode_record(
            source_record,
            new_episode_index=int(new_episode_index),
            new_parquet_path=new_parquet_path,
            new_video_paths=new_video_paths,
            frame_count=int(frame_count),
        )
        updated_episode_records.append(updated_record)
        updated_stats_records.append(
            _updated_stats_record(
                stats_by_episode.get(old_episode_index),
                episode_record=updated_record,
                new_episode_index=int(new_episode_index),
                frame_count=int(frame_count),
            )
        )
        sample_index_start += int(frame_count)

    _write_jsonl(target_root / "meta" / "episodes.jsonl", updated_episode_records)
    _write_jsonl(target_root / "meta" / "episodes_stats.jsonl", updated_stats_records)
    _write_json(
        target_root / "meta" / "info.json",
        _updated_info(
            source_info,
            episode_records=updated_episode_records,
            dataset_name=dataset_name,
        ),
    )


def build_merged_dataset(
    *,
    source_manifests: Sequence[DatasetManifest],
    target_root: Path,
    dataset_name: str | None,
    parquet_compression: str,
    allow_missing_videos: bool,
) -> None:
    if len(source_manifests) < 2:
        raise ValueError("At least two source datasets are required for merge.")

    first_manifest = source_manifests[0]
    source_info = first_manifest.info
    target_root.mkdir(parents=True, exist_ok=False)
    (target_root / "data").mkdir(parents=True, exist_ok=True)
    (target_root / "videos").mkdir(parents=True, exist_ok=True)
    _copy_static_meta(first_manifest.root, target_root)

    task_rows = _merged_task_rows([manifest.root for manifest in source_manifests])
    if task_rows:
        _write_jsonl(target_root / "meta" / "tasks.jsonl", task_rows)

    chunk_size = _chunk_size(source_info)
    updated_episode_records: list[dict[str, Any]] = []
    updated_stats_records: list[dict[str, Any]] = []
    sample_index_start = 0

    for manifest in source_manifests:
        for source_record in _ordered_episode_records(manifest.episode_records):
            new_episode_index = len(updated_episode_records)
            old_episode_index = _episode_index(source_record)
            old_parquet = source_record.get("parquet_path")
            if not old_parquet:
                raise ValueError(f"Episode {old_episode_index} has no parquet_path.")
            source_parquet_path = manifest.root / str(old_parquet)
            if not source_parquet_path.exists():
                raise FileNotFoundError(
                    f"Missing parquet for episode {old_episode_index}: {source_parquet_path}"
                )

            new_parquet_path = _episode_parquet_relative_path(
                int(new_episode_index),
                chunk_size=chunk_size,
            )
            old_video_paths = {
                str(key): str(value)
                for key, value in dict(source_record.get("video_paths") or {}).items()
                if value
            }
            new_video_paths = _updated_video_paths(
                source_record,
                int(new_episode_index),
                chunk_size=chunk_size,
            )
            frame_count = _rewrite_episode_parquet(
                source_path=source_parquet_path,
                target_path=target_root / new_parquet_path,
                new_episode_index=int(new_episode_index),
                sample_index_start=int(sample_index_start),
                old_video_paths=old_video_paths,
                new_video_paths=new_video_paths,
                compression=parquet_compression,
            )
            _copy_episode_videos(
                source_root=manifest.root,
                target_root=target_root,
                old_video_paths=old_video_paths,
                new_video_paths=new_video_paths,
                allow_missing_videos=allow_missing_videos,
            )

            updated_record = _updated_episode_record(
                source_record,
                new_episode_index=int(new_episode_index),
                new_parquet_path=new_parquet_path,
                new_video_paths=new_video_paths,
                frame_count=int(frame_count),
            )
            updated_episode_records.append(updated_record)
            updated_stats_records.append(
                _updated_stats_record(
                    manifest.stats_by_episode.get(old_episode_index),
                    episode_record=updated_record,
                    new_episode_index=int(new_episode_index),
                    frame_count=int(frame_count),
                )
            )
            sample_index_start += int(frame_count)

    _write_jsonl(target_root / "meta" / "episodes.jsonl", updated_episode_records)
    _write_jsonl(target_root / "meta" / "episodes_stats.jsonl", updated_stats_records)
    _write_json(
        target_root / "meta" / "info.json",
        _updated_info(
            source_info,
            episode_records=updated_episode_records,
            dataset_name=dataset_name,
            task_rows=task_rows,
        ),
    )


def _ensure_safe_paths(dataset_root: Path, output_root: Path) -> None:
    source = dataset_root.resolve()
    output = output_root.resolve()
    if source == output:
        raise ValueError("--output-root must be different from --dataset-root.")
    if source in output.parents:
        raise ValueError("--output-root must not be inside --dataset-root.")
    if output in source.parents:
        raise ValueError("--dataset-root must not be inside --output-root.")


def _ensure_merge_safe_paths(dataset_root: Path, merge_root: Path, output_root: Path) -> None:
    first = dataset_root.resolve()
    second = merge_root.resolve()
    output = output_root.resolve()
    if first == second:
        raise ValueError("--merge-root must be different from --dataset-root.")
    for option_name, source in (("--dataset-root", first), ("--merge-root", second)):
        if source == output:
            raise ValueError(f"--output-root must be different from {option_name}.")
        if source in output.parents:
            raise ValueError(f"--output-root must not be inside {option_name}.")
        if output in source.parents:
            raise ValueError(f"{option_name} must not be inside --output-root.")


def execute_split(
    *,
    dataset_root: Path,
    output_root: Path,
    selected_indices: Sequence[int],
    output_dataset_name: str | None,
    remaining_dataset_name: str | None,
    parquet_compression: str,
    overwrite_output: bool,
    keep_backup: bool,
    allow_missing_videos: bool,
) -> tuple[Path | None, SplitPlan]:
    dataset_root = dataset_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    _ensure_safe_paths(dataset_root, output_root)

    source_info = _read_json(dataset_root / "meta" / "info.json")
    selection = load_selection(dataset_root, selected_indices)
    plan = make_plan(dataset_root, output_root, selected_indices)

    if output_root.exists():
        if not overwrite_output:
            raise FileExistsError(f"Output dataset already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    run_id = f"{int(time.time())}-{os.getpid()}"
    output_tmp = output_root.parent / f".{output_root.name}.tmp-{run_id}"
    remaining_tmp = dataset_root.parent / f".{dataset_root.name}.remaining-tmp-{run_id}"
    backup_root = dataset_root.parent / f".{dataset_root.name}.before_extract-{run_id}"

    for path in (output_tmp, remaining_tmp, backup_root):
        if path.exists():
            raise FileExistsError(f"Temporary path already exists: {path}")

    try:
        build_subset_dataset(
            source_root=dataset_root,
            target_root=output_tmp,
            source_info=source_info,
            episode_records=selection.selected_records,
            stats_by_episode=selection.stats_by_episode,
            dataset_name=output_dataset_name if output_dataset_name is not None else output_root.name,
            parquet_compression=parquet_compression,
            allow_missing_videos=allow_missing_videos,
        )
        build_subset_dataset(
            source_root=dataset_root,
            target_root=remaining_tmp,
            source_info=source_info,
            episode_records=selection.remaining_records,
            stats_by_episode=selection.stats_by_episode,
            dataset_name=remaining_dataset_name,
            parquet_compression=parquet_compression,
            allow_missing_videos=allow_missing_videos,
        )

        output_tmp.rename(output_root)
        dataset_root.rename(backup_root)
        remaining_tmp.rename(dataset_root)
        if not keep_backup:
            shutil.rmtree(backup_root)
            backup_root = None  # type: ignore[assignment]
        return backup_root, plan
    except Exception:
        for path in (output_tmp, remaining_tmp):
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        raise


def execute_merge(
    *,
    dataset_root: Path,
    merge_root: Path,
    output_root: Path,
    output_dataset_name: str | None,
    parquet_compression: str,
    overwrite_output: bool,
    allow_missing_videos: bool,
) -> MergePlan:
    dataset_root = dataset_root.expanduser().resolve()
    merge_root = merge_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    _ensure_merge_safe_paths(dataset_root, merge_root, output_root)

    first = load_dataset_manifest(dataset_root)
    second = load_dataset_manifest(merge_root)
    _validate_merge_compatible(first, second)
    plan = make_merge_plan(dataset_root, merge_root, output_root)

    if output_root.exists():
        if not overwrite_output:
            raise FileExistsError(f"Output dataset already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    run_id = f"{int(time.time())}-{os.getpid()}"
    output_tmp = output_root.parent / f".{output_root.name}.merge-tmp-{run_id}"
    if output_tmp.exists():
        raise FileExistsError(f"Temporary path already exists: {output_tmp}")

    try:
        build_merged_dataset(
            source_manifests=(first, second),
            target_root=output_tmp,
            dataset_name=output_dataset_name if output_dataset_name is not None else output_root.name,
            parquet_compression=parquet_compression,
            allow_missing_videos=allow_missing_videos,
        )
        output_tmp.rename(output_root)
        return plan
    except Exception:
        if output_tmp.exists():
            shutil.rmtree(output_tmp, ignore_errors=True)
        raise


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Source LeRobot dataset root. In merge mode this is the first dataset.",
    )
    parser.add_argument(
        "--merge-root",
        type=Path,
        default=None,
        help="Second LeRobot dataset root to merge after --dataset-root.",
    )
    parser.add_argument("--output-root", type=Path, required=True, help="Root for the extracted or merged dataset.")
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Episode indices to extract, e.g. 0,2,5-7. Ranges are inclusive.",
    )
    parser.add_argument("--output-dataset-name", type=str, default=None)
    parser.add_argument(
        "--remaining-dataset-name",
        type=str,
        default=None,
        help="Dataset name written back to the remaining source dataset. Default preserves source info.json.",
    )
    parser.add_argument("--parquet-compression", type=str, default="zstd")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument(
        "--allow-missing-videos",
        action="store_true",
        help="Do not fail if a video listed in meta/episodes.jsonl is missing.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Delete the hidden backup of the original source dataset after a successful split.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually perform the split or merge. Without this flag the command only prints the plan.",
    )


def _print_plan(plan: SplitPlan) -> None:
    print("[ExtractLeRobot] plan")
    print(f"  source: {plan.dataset_root}")
    print(f"  extracted output: {plan.output_root}")
    print(f"  episodes: {list(plan.selected_indices)}")
    print(f"  extracted: {plan.selected_count} episode(s), {plan.selected_frames} frame(s)")
    print(f"  remaining source: {plan.remaining_count} episode(s), {plan.remaining_frames} frame(s)")


def _print_merge_plan(plan: MergePlan) -> None:
    print("[ExtractLeRobot] merge plan")
    print(f"  first source: {plan.dataset_root}")
    print(f"  second source: {plan.merge_root}")
    print(f"  merged output: {plan.output_root}")
    print(f"  first source: {plan.first_count} episode(s), {plan.first_frames} frame(s)")
    print(f"  second source: {plan.second_count} episode(s), {plan.second_frames} frame(s)")
    print(f"  merged: {plan.total_count} episode(s), {plan.total_frames} frame(s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract selected episodes from a LeRobot-style dataset, or merge two "
            "compatible LeRobot-style datasets into a new dataset."
        )
    )
    add_args(parser)
    args = parser.parse_args(argv)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if args.merge_root is not None:
        if args.episodes:
            parser.error("--episodes cannot be used with --merge-root.")
        if args.remaining_dataset_name is not None:
            parser.error("--remaining-dataset-name only applies to extract mode.")
        if bool(args.no_backup):
            parser.error("--no-backup only applies to extract mode.")
        merge_root = Path(args.merge_root).expanduser().resolve()
        _ensure_merge_safe_paths(dataset_root, merge_root, output_root)
        merge_plan = make_merge_plan(dataset_root, merge_root, output_root)
        _print_merge_plan(merge_plan)
        if not bool(args.yes):
            print("  dry-run: add --yes to perform the merge.")
            return 0

        execute_merge(
            dataset_root=dataset_root,
            merge_root=merge_root,
            output_root=output_root,
            output_dataset_name=args.output_dataset_name,
            parquet_compression=str(args.parquet_compression),
            overwrite_output=bool(args.overwrite_output),
            allow_missing_videos=bool(args.allow_missing_videos),
        )
        print("[ExtractLeRobot] merge done")
        print(f"  merged dataset: {output_root}")
        return 0

    if not args.episodes:
        parser.error("--episodes is required unless --merge-root is set.")

    selected_indices = parse_episode_indices(args.episodes)
    _ensure_safe_paths(dataset_root, output_root)
    plan = make_plan(dataset_root, output_root, selected_indices)
    _print_plan(plan)
    if not bool(args.yes):
        print("  dry-run: add --yes to perform the split.")
        return 0

    backup_root, _ = execute_split(
        dataset_root=dataset_root,
        output_root=output_root,
        selected_indices=selected_indices,
        output_dataset_name=args.output_dataset_name,
        remaining_dataset_name=args.remaining_dataset_name,
        parquet_compression=str(args.parquet_compression),
        overwrite_output=bool(args.overwrite_output),
        keep_backup=not bool(args.no_backup),
        allow_missing_videos=bool(args.allow_missing_videos),
    )
    print("[ExtractLeRobot] done")
    print(f"  extracted dataset: {output_root}")
    print(f"  remaining dataset: {dataset_root}")
    if backup_root is not None:
        print(f"  original backup: {backup_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
