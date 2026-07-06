from __future__ import annotations

import shutil
import time
from collections.abc import Sequence
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
class EpisodeDeletionPlan:
    dataset_root: str
    output_root: str | None
    requested_indices: tuple[int, ...]
    deleted_indices: tuple[int, ...]
    missing_indices: tuple[int, ...]
    kept_indices: tuple[int, ...]
    deleted_artifacts: tuple[str, ...]

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


def plan_episode_deletion(
    dataset_root: str | Path,
    episode_indices: str | Sequence[int],
    *,
    output_root: str | Path | None = None,
) -> EpisodeDeletionPlan:
    root = Path(dataset_root)
    requested = parse_episode_selection(episode_indices)
    records = episode_records(root)
    by_index = {int(record["episode_index"]): record for record in records}
    deleted = tuple(index for index in requested if index in by_index)
    missing = tuple(index for index in requested if index not in by_index)
    kept = tuple(sorted(index for index in by_index if index not in set(deleted)))
    artifact_paths: list[str] = []
    for index in deleted:
        artifact_paths.extend(episode_artifact_paths(by_index[index]))
    return EpisodeDeletionPlan(
        dataset_root=str(root),
        output_root=None if output_root is None else str(Path(output_root)),
        requested_indices=requested,
        deleted_indices=deleted,
        missing_indices=missing,
        kept_indices=kept,
        deleted_artifacts=tuple(artifact_paths),
    )


def _remove_artifact(output_root: Path, relative_path: Path) -> None:
    target = output_root / relative_path
    remove_target = target.parent if target.name == "manifest.json" else target
    if remove_target.is_dir():
        shutil.rmtree(remove_target)
    elif remove_target.exists():
        remove_target.unlink()


def _write_filtered_meta(output_root: Path, deleted_indices: set[int]) -> None:
    episodes = [
        record
        for record in episode_records(output_root)
        if int(record.get("episode_index", -1)) not in deleted_indices
    ]
    stats = [
        record
        for record in episode_stats_records(output_root)
        if int(record.get("episode_index", -1)) not in deleted_indices
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
                "tool": "dataset_postprocess.delete_episodes",
                "deleted_indices": sorted(deleted_indices),
                "updated_at": time.time(),
            }
        )
        info["postprocess_history"] = history
        write_json(info_path, info)


def delete_episodes(
    dataset_root: str | Path,
    output_root: str | Path,
    episode_indices: str | Sequence[int],
    *,
    overwrite: bool = False,
) -> EpisodeDeletionPlan:
    root = Path(dataset_root).resolve()
    output = Path(output_root).resolve()
    if root == output:
        raise ValueError("delete_episodes writes to a new output_root; in-place deletion is disabled.")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output_root already exists: {output}")
        shutil.rmtree(output)
    plan = plan_episode_deletion(root, episode_indices, output_root=output)
    shutil.copytree(root, output)
    deleted_set = set(plan.deleted_indices)
    for artifact in plan.deleted_artifacts:
        relative = _relative_artifact_path(root, artifact)
        if relative is None:
            continue
        _remove_artifact(output, relative)
    _write_filtered_meta(output, deleted_set)
    return plan
