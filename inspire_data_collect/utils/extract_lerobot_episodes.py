from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..common import to_serializable
from .writer import chunk_id_from_episode_index, episode_id_from_index


MANIFEST_FILENAMES = {"info.json", "episodes.jsonl", "episodes_stats.jsonl"}


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


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Episode extraction needs pyarrow to rewrite parquet metadata. "
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
) -> dict[str, Any]:
    info = dict(source_info)
    info["episode_count"] = int(len(episode_records))
    info["frame_count"] = int(sum(int(row.get("frame_count", 0) or 0) for row in episode_records))
    if dataset_name is not None:
        info["dataset_name"] = str(dataset_name)
    return info


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


def _ensure_safe_paths(dataset_root: Path, output_root: Path) -> None:
    source = dataset_root.resolve()
    output = output_root.resolve()
    if source == output:
        raise ValueError("--output-root must be different from --dataset-root.")
    if source in output.parents:
        raise ValueError("--output-root must not be inside --dataset-root.")
    if output in source.parents:
        raise ValueError("--dataset-root must not be inside --output-root.")


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


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True, help="Source LeRobot dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root for the extracted dataset.")
    parser.add_argument(
        "--episodes",
        type=str,
        required=True,
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
        help="Actually perform the split. Without this flag the command only prints the plan.",
    )


def _print_plan(plan: SplitPlan) -> None:
    print("[ExtractLeRobot] plan")
    print(f"  source: {plan.dataset_root}")
    print(f"  extracted output: {plan.output_root}")
    print(f"  episodes: {list(plan.selected_indices)}")
    print(f"  extracted: {plan.selected_count} episode(s), {plan.selected_frames} frame(s)")
    print(f"  remaining source: {plan.remaining_count} episode(s), {plan.remaining_frames} frame(s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract selected episodes from a LeRobot-style dataset into a new dataset, "
            "then rewrite the source dataset to contain only the remaining episodes."
        )
    )
    add_args(parser)
    args = parser.parse_args(argv)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
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
