from __future__ import annotations

import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .io import (
    episode_artifact_paths,
    episode_records,
    episode_stats_records,
    read_json,
    write_json,
    write_jsonl,
)
from .selection import parse_episode_selection


@dataclass(frozen=True)
class EpisodeSplitPlan:
    dataset_root: str
    clean_output_root: str | None
    extract_output_root: str | None
    requested_indices: tuple[int, ...]
    extracted_indices: tuple[int, ...]
    missing_indices: tuple[int, ...]
    clean_indices: tuple[int, ...]
    clean_removed_artifacts: tuple[str, ...]
    extract_removed_artifacts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _relative_artifact_path(dataset_root: Path, artifact_path: str) -> Path | None:
    path = Path(artifact_path)
    if not path.is_absolute():
        return path
    try:
        return path.relative_to(dataset_root)
    except ValueError:
        return None


def _remove_artifact(output_root: Path, relative_path: Path) -> None:
    target = output_root / relative_path
    remove_target = target.parent if target.name == "manifest.json" else target
    if remove_target.is_dir():
        shutil.rmtree(remove_target)
    elif remove_target.exists():
        remove_target.unlink()


def _records_by_index(records: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(record["episode_index"]): record for record in records}


def plan_episode_split(
    dataset_root: str | Path,
    episode_indices: str | Sequence[int],
    *,
    clean_output_root: str | Path | None = None,
    extract_output_root: str | Path | None = None,
) -> EpisodeSplitPlan:
    root = Path(dataset_root)
    requested = parse_episode_selection(episode_indices)
    records = episode_records(root)
    by_index = _records_by_index(records)
    extracted = tuple(index for index in requested if index in by_index)
    missing = tuple(index for index in requested if index not in by_index)
    extracted_set = set(extracted)
    clean = tuple(sorted(index for index in by_index if index not in extracted_set))
    clean_removed: list[str] = []
    extract_removed: list[str] = []
    for index, record in by_index.items():
        if index in extracted_set:
            clean_removed.extend(episode_artifact_paths(record))
        else:
            extract_removed.extend(episode_artifact_paths(record))
    return EpisodeSplitPlan(
        dataset_root=str(root),
        clean_output_root=None if clean_output_root is None else str(Path(clean_output_root)),
        extract_output_root=None if extract_output_root is None else str(Path(extract_output_root)),
        requested_indices=requested,
        extracted_indices=extracted,
        missing_indices=missing,
        clean_indices=clean,
        clean_removed_artifacts=tuple(clean_removed),
        extract_removed_artifacts=tuple(extract_removed),
    )


def _filter_meta(
    output_root: Path,
    *,
    keep_indices: set[int],
    history_tool: str,
    history_payload: Mapping[str, Any],
) -> None:
    episodes = [
        record
        for record in episode_records(output_root)
        if int(record.get("episode_index", -1)) in keep_indices
    ]
    stats = [
        record
        for record in episode_stats_records(output_root)
        if int(record.get("episode_index", -1)) in keep_indices
    ]
    write_jsonl(output_root / "meta" / "episodes.jsonl", episodes)
    write_jsonl(output_root / "meta" / "episodes_stats.jsonl", stats)
    info_path = output_root / "meta" / "info.json"
    info = read_json(info_path)
    if info:
        frame_count = sum(int(record.get("frame_count", 0) or 0) for record in episodes)
        for key in ("total_episodes", "num_episodes", "episode_count"):
            if key in info:
                info[key] = len(episodes)
        for key in ("total_frames", "num_frames", "frame_count"):
            if key in info:
                info[key] = frame_count
        for key in ("dataset_name", "dataset_dir_name"):
            if key in info:
                info[key] = output_root.name
        history = info.get("postprocess_history")
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "tool": history_tool,
                "updated_at": time.time(),
                **dict(history_payload),
            }
        )
        info["postprocess_history"] = history
        write_json(info_path, info)


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output_root already exists: {path}")
        shutil.rmtree(path)


def _copy_dataset_scaffold(source_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=False)
    meta_source = source_root / "meta"
    if meta_source.exists():
        shutil.copytree(meta_source, output_root / "meta")
    for child in source_root.iterdir():
        if child.name == "meta":
            continue
        if child.is_file():
            shutil.copy2(child, output_root / child.name)


def _copy_artifact(source_root: Path, output_root: Path, artifact_path: str) -> Path | None:
    relative = _relative_artifact_path(source_root, artifact_path)
    if relative is None:
        return None
    source = source_root / relative
    destination = output_root / relative
    if source.name == "manifest.json":
        source = source.parent
        destination = destination.parent
        relative = relative.parent
    if not source.exists():
        return relative
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return relative


def _copy_episode_artifacts(
    source_root: Path,
    output_root: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    copied: set[Path] = set()
    for record in records:
        for artifact in episode_artifact_paths(record):
            relative = _relative_artifact_path(source_root, artifact)
            if relative is None:
                continue
            if relative.name == "manifest.json":
                relative = relative.parent
            if relative in copied:
                continue
            copied.add(relative)
            _copy_artifact(source_root, output_root, artifact)


def _copy_selected_dataset(
    source_root: Path,
    output_root: Path,
    *,
    keep_indices: set[int],
    history_tool: str,
    history_payload: Mapping[str, Any],
) -> None:
    _copy_dataset_scaffold(source_root, output_root)
    selected_records = [
        record
        for record in episode_records(source_root)
        if int(record.get("episode_index", -1)) in keep_indices
    ]
    _copy_episode_artifacts(source_root, output_root, selected_records)
    _filter_meta(
        output_root,
        keep_indices=keep_indices,
        history_tool=history_tool,
        history_payload=history_payload,
    )


def split_episodes(
    dataset_root: str | Path,
    *,
    clean_output_root: str | Path,
    extract_output_root: str | Path,
    episode_indices: str | Sequence[int],
    overwrite: bool = False,
) -> EpisodeSplitPlan:
    root = Path(dataset_root).resolve()
    clean_output = Path(clean_output_root).resolve()
    extract_output = Path(extract_output_root).resolve()
    if root in {clean_output, extract_output}:
        raise ValueError("split_episodes writes to new output roots; in-place splitting is disabled.")
    if clean_output == extract_output:
        raise ValueError("clean_output_root and extract_output_root must be different.")
    plan = plan_episode_split(
        root,
        episode_indices,
        clean_output_root=clean_output,
        extract_output_root=extract_output,
    )
    _prepare_output(clean_output, overwrite=bool(overwrite))
    _prepare_output(extract_output, overwrite=bool(overwrite))
    extracted_set = set(plan.extracted_indices)
    clean_set = set(plan.clean_indices)
    _copy_selected_dataset(
        root,
        clean_output,
        keep_indices=clean_set,
        history_tool="dataset_postprocess.split_episodes.clean",
        history_payload={
            "removed_episode_indices": list(plan.extracted_indices),
            "missing_requested_indices": list(plan.missing_indices),
            "source_dataset_root": str(root),
        },
    )
    _copy_selected_dataset(
        root,
        extract_output,
        keep_indices=extracted_set,
        history_tool="dataset_postprocess.split_episodes.extract",
        history_payload={
            "extracted_episode_indices": list(plan.extracted_indices),
            "missing_requested_indices": list(plan.missing_indices),
            "source_dataset_root": str(root),
        },
    )
    return plan
