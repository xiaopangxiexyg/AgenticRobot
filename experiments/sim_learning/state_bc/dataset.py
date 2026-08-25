"""Read-only FormalCollection state/action loading for state BC experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import (
    STATE_BC_ACTION_DIM,
    STATE_BC_ACCEPTED_LAYOUT_HASHES,
    STATE_BC_STATE_DIM,
    StateBCSelector,
    content_hash,
)


@dataclass(frozen=True)
class FormalCollectionEpisode:
    source_dir: str
    episode_id: str
    episode_index: int
    frame_count: int
    parquet_path: str
    collection_integrity_valid: bool
    summary: Mapping[str, Any] = field(default_factory=dict)

    @property
    def resolved_parquet_path(self) -> Path:
        source = Path(self.source_dir).expanduser()
        parquet = Path(self.parquet_path)
        return parquet if parquet.is_absolute() else source / parquet

    @property
    def qualified_episode_id(self) -> str:
        """Stable row-group identity when one logical day has split folders."""

        return f"{self.source_dir}::{self.episode_id}"

    def exclusion_reason(self, *, min_episode_frames: int) -> str | None:
        if not self.collection_integrity_valid:
            return "collection_integrity_invalid"
        if self.frame_count < int(min_episode_frames):
            return "episode_too_short"
        if not self.resolved_parquet_path.is_file():
            return "parquet_missing"
        return None


@dataclass(frozen=True)
class StateBCSelection:
    selector: StateBCSelector
    selected_episodes: tuple[FormalCollectionEpisode, ...]
    excluded_episodes: Mapping[str, str]
    layout_hash_by_source: Mapping[str, str]
    info_by_source: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.selected_episodes:
            raise ValueError("selection contains no usable episodes.")
        selected_ids = [episode.qualified_episode_id for episode in self.selected_episodes]
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("selected episode ids must be unique across source directories.")
        object.__setattr__(self, "selected_episodes", tuple(self.selected_episodes))
        object.__setattr__(self, "excluded_episodes", dict(self.excluded_episodes))
        object.__setattr__(self, "layout_hash_by_source", dict(self.layout_hash_by_source))
        object.__setattr__(self, "info_by_source", dict(self.info_by_source))

    @property
    def semantic_content_hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector": self.selector.to_dict(),
            "selected_episodes": [
                {
                    "source_dir": episode.source_dir,
                    "episode_id": episode.episode_id,
                    "episode_index": episode.episode_index,
                    "frame_count": episode.frame_count,
                    "parquet_path": episode.parquet_path,
                    "collection_integrity_valid": episode.collection_integrity_valid,
                }
                for episode in self.selected_episodes
            ],
            "excluded_episodes": dict(self.excluded_episodes),
            "layout_hash_by_source": dict(self.layout_hash_by_source),
        }


@dataclass(frozen=True)
class StateBCRows:
    states: np.ndarray
    actions: np.ndarray
    episode_ids: tuple[str, ...]
    selection: StateBCSelection

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.float32)
        ids = tuple(str(value) for value in self.episode_ids)
        if states.ndim != 2 or states.shape[1] != STATE_BC_STATE_DIM:
            raise ValueError(f"states must have shape [N, {STATE_BC_STATE_DIM}], got {states.shape}.")
        if actions.shape != (states.shape[0], STATE_BC_ACTION_DIM):
            raise ValueError(
                f"actions must have shape [{states.shape[0]}, {STATE_BC_ACTION_DIM}], got {actions.shape}."
            )
        if len(ids) != states.shape[0] or any(not value for value in ids):
            raise ValueError("episode_ids must align with finite rows.")
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
            raise ValueError("state BC rows must be finite.")
        object.__setattr__(self, "states", states.copy())
        object.__setattr__(self, "actions", actions.copy())
        object.__setattr__(self, "episode_ids", ids)

    @property
    def frame_count(self) -> int:
        return int(self.states.shape[0])

    @property
    def episode_count(self) -> int:
        return len(set(self.episode_ids))


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object at {path}.")
    return value


def _episode_records(source_dir: Path) -> tuple[FormalCollectionEpisode, ...]:
    path = source_dir / "meta" / "episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"FormalCollection episode manifest is missing: {path}")
    records: list[FormalCollectionEpisode] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            raise ValueError(f"episode manifest line {line_number} is not an object: {path}")
        summary = raw.get("summary")
        summary_map = dict(summary) if isinstance(summary, Mapping) else {}
        integrity = summary_map.get("collection_integrity_valid")
        if integrity is None:
            # Older manifests used a top-level flag.  The fallback is explicit
            # and still recorded in the selection metadata.
            integrity = raw.get("collection_integrity_valid", False)
        episode_id = str(raw.get("episode_id") or f"episode-{raw.get('episode_index', line_number)}")
        records.append(
            FormalCollectionEpisode(
                source_dir=str(source_dir),
                episode_id=episode_id,
                episode_index=int(raw.get("episode_index", len(records))),
                frame_count=int(raw.get("frame_count", 0)),
                parquet_path=str(raw.get("parquet_path") or ""),
                collection_integrity_valid=bool(integrity),
                summary=summary_map,
            )
        )
    return tuple(records)


def select_formal_collection(selector: StateBCSelector) -> StateBCSelection:
    """Select valid episodes and retain every exclusion reason in the manifest."""

    if not isinstance(selector, StateBCSelector):
        raise TypeError("selector must be a StateBCSelector.")
    selected: list[FormalCollectionEpisode] = []
    excluded: dict[str, str] = {}
    layout_hash_by_source: dict[str, str] = {}
    info_by_source: dict[str, Mapping[str, Any]] = {}
    for raw_source in selector.source_dirs:
        source = Path(raw_source).expanduser().resolve()
        info_path = source / "meta" / "info.json"
        info = _read_json(info_path) if info_path.is_file() else {}
        info_by_source[str(source)] = info
        for key, expected in (
            ("pose_storage_frame", "desk_absolute"),
            ("arm_pose_action_mode", "absolute"),
        ):
            declared = info.get(key)
            if declared is not None and str(declared).strip().lower() != expected:
                raise ValueError(
                    f"{source} declares {key}={declared!r}; expected {expected!r}."
                )
        layout_hash = str(info.get("policy_io_layout_hash") or "")
        if layout_hash:
            layout_hash_by_source[str(source)] = layout_hash
        if layout_hash and layout_hash not in STATE_BC_ACCEPTED_LAYOUT_HASHES:
            raise ValueError(
                f"{source} binds unsupported layout {layout_hash!r}; "
                f"accepted={sorted(STATE_BC_ACCEPTED_LAYOUT_HASHES)!r}."
            )
        for episode in _episode_records(source):
            key = f"{source}:{episode.episode_id}"
            reason = episode.exclusion_reason(min_episode_frames=selector.min_episode_frames)
            if selector.require_collection_integrity_valid and not episode.collection_integrity_valid:
                reason = "collection_integrity_invalid"
            if reason is None:
                selected.append(episode)
            else:
                excluded[key] = reason
    return StateBCSelection(
        selector=selector,
        selected_episodes=tuple(selected),
        excluded_episodes=excluded,
        layout_hash_by_source=layout_hash_by_source,
        info_by_source=info_by_source,
    )


def _coerce_row_vector(value: Any, *, name: str, dim: int) -> np.ndarray:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} string is not JSON.") from exc
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (dim,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be one finite {dim}D row, got {array.shape}.")
    return array.astype(np.float32, copy=True)


def _read_episode_rows(episode: FormalCollectionEpisode) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "state BC parquet loading requires pandas and a parquet engine "
            "(pyarrow or fastparquet)."
        ) from exc
    parquet_path = episode.resolved_parquet_path
    if not parquet_path.is_file():
        raise FileNotFoundError(f"selected episode parquet is missing: {parquet_path}")
    try:
        frame = pd.read_parquet(parquet_path, columns=["observation.state", "action"])
    except Exception as exc:  # noqa: BLE001 - preserve the backend error context.
        raise RuntimeError(f"could not read state/action columns from {parquet_path}: {exc}") from exc
    if "observation.state" not in frame or "action" not in frame:
        raise ValueError(f"{parquet_path} does not contain observation.state and action columns.")
    states = np.stack(
        [_coerce_row_vector(value, name="observation.state", dim=STATE_BC_STATE_DIM) for value in frame["observation.state"]],
        axis=0,
    )
    actions = np.stack(
        [_coerce_row_vector(value, name="action", dim=STATE_BC_ACTION_DIM) for value in frame["action"]],
        axis=0,
    )
    return states, actions


def load_state_action_rows(selection: StateBCSelection) -> StateBCRows:
    """Load only low-dimensional state/action columns; media are never opened."""

    if not isinstance(selection, StateBCSelection):
        raise TypeError("selection must be a StateBCSelection.")
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_ids: list[str] = []
    for episode in selection.selected_episodes:
        state_rows, action_rows = _read_episode_rows(episode)
        if state_rows.shape[0] != action_rows.shape[0] or state_rows.shape[0] == 0:
            raise ValueError(f"selected episode has unpaired or empty rows: {episode.episode_id}")
        states.append(state_rows)
        actions.append(action_rows)
        episode_ids.extend([episode.qualified_episode_id] * int(state_rows.shape[0]))
    return StateBCRows(
        states=np.concatenate(states, axis=0),
        actions=np.concatenate(actions, axis=0),
        episode_ids=tuple(episode_ids),
        selection=selection,
    )


def split_rows_by_episode(
    rows: StateBCRows,
    *,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[StateBCRows, StateBCRows]:
    """Split whole episodes, never individual frames, into train and test."""

    if not isinstance(rows, StateBCRows):
        raise TypeError("rows must be StateBCRows.")
    fraction = float(test_fraction)
    if not np.isfinite(fraction) or fraction < 0.0 or fraction >= 1.0:
        raise ValueError("test_fraction must be in [0, 1).")
    episode_ids = np.asarray(sorted(set(rows.episode_ids)), dtype=object)
    if episode_ids.size < 2 and fraction > 0.0:
        raise ValueError("at least two episodes are required for a non-empty test split.")
    rng = np.random.default_rng(int(seed))
    shuffled = episode_ids.copy()
    rng.shuffle(shuffled)
    test_count = int(round(float(shuffled.size) * fraction))
    if fraction > 0.0:
        test_count = max(1, min(int(shuffled.size) - 1, test_count))
    test_ids = set(str(value) for value in shuffled[:test_count])
    train_mask = np.asarray([episode_id not in test_ids for episode_id in rows.episode_ids], dtype=bool)
    test_mask = ~train_mask

    def subset(mask: np.ndarray) -> StateBCRows:
        selected_ids = tuple(episode_id for episode_id, keep in zip(rows.episode_ids, mask) if keep)
        if not np.any(mask):
            return StateBCRows(
                states=np.empty((0, STATE_BC_STATE_DIM), dtype=np.float32),
                actions=np.empty((0, STATE_BC_ACTION_DIM), dtype=np.float32),
                episode_ids=(),
                selection=rows.selection,
            )
        return StateBCRows(
            states=rows.states[mask],
            actions=rows.actions[mask],
            episode_ids=selected_ids,
            selection=rows.selection,
        )

    train = subset(train_mask)
    test = subset(test_mask)
    if train.frame_count == 0:
        raise ValueError("episode split produced an empty train set.")
    return train, test


__all__ = [
    "FormalCollectionEpisode",
    "StateBCRows",
    "StateBCSelection",
    "load_state_action_rows",
    "select_formal_collection",
    "split_rows_by_episode",
]
