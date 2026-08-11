"""Manifest-driven, model-neutral arm-hold training windows.

This layer reads immutable split membership and source Episodes.  It owns no
cache format, augmentation, sampler weighting, model adapter, or deployment
behavior.  Parquet Episode arrays and front video frames are loaded lazily per
reader instance so each DataLoader worker can own its own bounded resources.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.arm_hold_history import (
    ARM_HOLD_CANONICAL_COLLECTED_CONTEXT_PROVENANCE_V1,
    ARM_HOLD_HISTORY_CONTEXT_SCHEMA_V1,
    ARM_HOLD_HISTORY_SEQUENCE_CONTEXT_SCHEMA_V1,
    ArmHoldHistoryContext,
    ArmHoldHistorySequenceContext,
)
from dataset_contracts.v23 import resolve_v23_dataset_contract

from .arm_hold import ArmHoldTrainingView, project_arm_hold_training_view
from .arm_hold_profile import ArmHoldDatasetProfile, resolve_arm_hold_source_roots
from .arm_hold_split_manifest import ArmHoldSplitManifest
from .episode_curation import source_dataset_base_fingerprint
from .prepared_training.sources import _select_annotation
from .progress_annotation_jobs import list_progress_annotation_work_items
from .video_source import VideoReaderCache


ARM_HOLD_INPUT_MODES = frozenset({"front_state", "front_only", "state_only"})
ARM_HOLD_WINDOW_SPLITS = frozenset({"train", "validation"})
ARM_HOLD_INVALID_LABEL_POLICY_V1 = "drop_window_if_any_side_invalid_v1"
ARM_HOLD_HISTORY_CONTEXT_PROVENANCE_V1 = (
    ARM_HOLD_CANONICAL_COLLECTED_CONTEXT_PROVENANCE_V1
)


class ArmHoldWindowError(ValueError):
    """Raised when frozen membership cannot produce unambiguous windows."""


@dataclass(frozen=True)
class ArmHoldHistoryTimeline:
    """Canonical preceding-label context for every row of one Episode."""

    previous_mode: np.ndarray
    available_mask: np.ndarray
    dwell_steps: np.ndarray
    provenance: str = ARM_HOLD_HISTORY_CONTEXT_PROVENANCE_V1

    def __post_init__(self) -> None:
        previous = np.asarray(self.previous_mode, dtype=np.float32)
        available = np.asarray(self.available_mask)
        dwell = np.asarray(self.dwell_steps)
        if (
            previous.ndim != 2
            or previous.shape[-1] != 2
            or available.shape != previous.shape
            or available.dtype != np.bool_
            or dwell.shape != previous.shape
            or not np.issubdtype(dwell.dtype, np.integer)
            or np.any(dwell < 0)
            or np.any((dwell > 0) != available)
            or np.any(~np.isin(previous[available], (0.0, 1.0)))
            or np.any(previous[~available] != 0.0)
            or self.provenance != ARM_HOLD_HISTORY_CONTEXT_PROVENANCE_V1
        ):
            raise ArmHoldWindowError("History timeline arrays are invalid.")
        frozen_previous = np.ascontiguousarray(previous).copy()
        frozen_available = np.ascontiguousarray(available).copy()
        frozen_dwell = np.ascontiguousarray(dwell, dtype=np.int64).copy()
        frozen_previous.setflags(write=False)
        frozen_available.setflags(write=False)
        frozen_dwell.setflags(write=False)
        object.__setattr__(self, "previous_mode", frozen_previous)
        object.__setattr__(self, "available_mask", frozen_available)
        object.__setattr__(self, "dwell_steps", frozen_dwell)

    def __len__(self) -> int:
        return int(self.previous_mode.shape[0])

    def context_at(self, index: int) -> ArmHoldHistoryContext:
        resolved = _integer_or_none(index, nonnegative=True)
        if resolved is None or resolved >= len(self):
            raise ArmHoldWindowError("History timeline index is out of range.")
        return ArmHoldHistoryContext(
            previous_mode=self.previous_mode[resolved],
            available_mask=self.available_mask[resolved],
            dwell_steps=self.dwell_steps[resolved],
            provenance=self.provenance,
        )


def derive_arm_hold_history_timeline(
    *,
    targets: np.ndarray,
    valid_mask: np.ndarray,
    loss_mask: np.ndarray,
    frame_indices: Sequence[int],
    continuity_segments: Sequence[int | None],
    running_mask: np.ndarray,
) -> ArmHoldHistoryTimeline:
    """Derive all preceding contexts in one causal linear Episode pass."""

    resolved_targets = np.asarray(targets, dtype=np.float32)
    resolved_valid = np.asarray(valid_mask)
    resolved_loss = np.asarray(loss_mask)
    resolved_frame_values = tuple(
        _integer_or_none(value, nonnegative=True) for value in frame_indices
    )
    resolved_frames = np.asarray(
        [0 if value is None else value for value in resolved_frame_values],
        dtype=np.int64,
    )
    resolved_running = np.asarray(running_mask)
    count = len(resolved_targets)
    if (
        resolved_targets.shape != (count, 2)
        or resolved_valid.shape != (count, 2)
        or resolved_loss.shape != (count, 2)
        or resolved_valid.dtype != np.bool_
        or resolved_loss.dtype != np.bool_
        or resolved_frames.shape != (count,)
        or resolved_running.shape != (count,)
        or resolved_running.dtype != np.bool_
        or len(continuity_segments) != count
        or any(value is None for value in resolved_frame_values)
    ):
        raise ArmHoldWindowError("History timeline source arrays are invalid.")
    continuity = tuple(
        None if value is None else _integer_or_none(value, nonnegative=True)
        for value in continuity_segments
    )
    if any(
        value is not None and resolved is None
        for value, resolved in zip(continuity_segments, continuity)
    ):
        raise ArmHoldWindowError("History continuity segments are invalid.")

    def adjacent(current: int, previous: int) -> bool:
        return (
            previous >= 0
            and resolved_frames[previous] + 1 == resolved_frames[current]
            and continuity[current] is not None
            and continuity[current] == continuity[previous]
            and resolved_running[current]
            and resolved_running[previous]
        )

    previous_mode = np.zeros((count, 2), dtype=np.float32)
    available = np.zeros((count, 2), dtype=bool)
    dwell_steps = np.zeros((count, 2), dtype=np.int64)
    for current in range(1, count):
        previous = current - 1
        if not adjacent(current, previous):
            continue
        for side_index in range(2):
            if (
                not resolved_valid[previous, side_index]
                or not resolved_loss[previous, side_index]
            ):
                continue
            mode = resolved_targets[previous, side_index]
            if mode not in (0.0, 1.0):
                raise ArmHoldWindowError("Previous canonical mode must be binary.")
            dwell = 1
            earlier = previous - 1
            if (
                adjacent(previous, earlier)
                and resolved_valid[earlier, side_index]
                and resolved_loss[earlier, side_index]
                and resolved_targets[earlier, side_index] == mode
            ):
                dwell = int(dwell_steps[previous, side_index]) + 1
                if dwell <= 1:
                    raise ArmHoldWindowError(
                        "History timeline lost an available preceding dwell."
                    )
            previous_mode[current, side_index] = mode
            available[current, side_index] = True
            dwell_steps[current, side_index] = dwell
    return ArmHoldHistoryTimeline(
        previous_mode=previous_mode,
        available_mask=available,
        dwell_steps=dwell_steps,
    )


def derive_arm_hold_history_context(
    *,
    targets: np.ndarray,
    valid_mask: np.ndarray,
    loss_mask: np.ndarray,
    frame_indices: Sequence[int],
    continuity_segments: Sequence[int | None],
    running_mask: np.ndarray,
    start_index: int,
) -> ArmHoldHistoryContext:
    """Derive previous canonical mode/dwell without crossing a source boundary."""

    resolved_targets = np.asarray(targets, dtype=np.float32)
    resolved_valid = np.asarray(valid_mask)
    resolved_loss = np.asarray(loss_mask)
    resolved_frame_values = tuple(
        _integer_or_none(value, nonnegative=True) for value in frame_indices
    )
    resolved_frames = np.asarray(
        [0 if value is None else value for value in resolved_frame_values],
        dtype=np.int64,
    )
    resolved_running = np.asarray(running_mask)
    count = len(resolved_targets)
    start = _integer_or_none(start_index, nonnegative=True)
    if (
        resolved_targets.shape != (count, 2)
        or resolved_valid.shape != (count, 2)
        or resolved_loss.shape != (count, 2)
        or resolved_valid.dtype != np.bool_
        or resolved_loss.dtype != np.bool_
        or resolved_frames.shape != (count,)
        or resolved_running.shape != (count,)
        or resolved_running.dtype != np.bool_
        or len(continuity_segments) != count
        or any(value is None for value in resolved_frame_values)
        or start is None
        or start >= count
    ):
        raise ArmHoldWindowError("History context source arrays are invalid.")
    continuity = tuple(
        None if value is None else _integer_or_none(value, nonnegative=True)
        for value in continuity_segments
    )
    if any(
        value is not None and resolved is None
        for value, resolved in zip(continuity_segments, continuity)
    ):
        raise ArmHoldWindowError("History continuity segments are invalid.")
    previous_mode = np.zeros(2, dtype=np.float32)
    available = np.zeros(2, dtype=bool)
    dwell_steps = np.zeros(2, dtype=np.int64)

    def adjacent(current: int, previous: int) -> bool:
        return (
            previous >= 0
            and resolved_frames[previous] + 1 == resolved_frames[current]
            and continuity[current] is not None
            and continuity[current] == continuity[previous]
            and resolved_running[current]
            and resolved_running[previous]
        )

    previous = start - 1
    for side_index in range(2):
        if (
            not adjacent(start, previous)
            or not resolved_valid[previous, side_index]
            or not resolved_loss[previous, side_index]
        ):
            continue
        mode = resolved_targets[previous, side_index]
        if mode not in (0.0, 1.0):
            raise ArmHoldWindowError("Previous canonical mode must be binary.")
        dwell = 1
        cursor = previous
        while cursor > 0:
            candidate = cursor - 1
            if (
                not adjacent(cursor, candidate)
                or not resolved_valid[candidate, side_index]
                or not resolved_loss[candidate, side_index]
                or resolved_targets[candidate, side_index] != mode
            ):
                break
            dwell += 1
            cursor = candidate
        previous_mode[side_index] = mode
        available[side_index] = True
        dwell_steps[side_index] = dwell
    return ArmHoldHistoryContext(
        previous_mode=previous_mode,
        available_mask=available,
        dwell_steps=dwell_steps,
        provenance=ARM_HOLD_HISTORY_CONTEXT_PROVENANCE_V1,
    )


def derive_arm_hold_history_sequence_context(
    *,
    targets: np.ndarray,
    valid_mask: np.ndarray,
    loss_mask: np.ndarray,
    frame_indices: Sequence[int],
    continuity_segments: Sequence[int | None],
    running_mask: np.ndarray,
    start_index: int,
) -> ArmHoldHistorySequenceContext:
    """Derive the two preceding causal label/dwell snapshots."""

    latest = derive_arm_hold_history_context(
        targets=targets,
        valid_mask=valid_mask,
        loss_mask=loss_mask,
        frame_indices=frame_indices,
        continuity_segments=continuity_segments,
        running_mask=running_mask,
        start_index=start_index,
    )
    previous = np.zeros((2, 2), dtype=np.float32)
    available = np.zeros((2, 2), dtype=bool)
    dwell = np.zeros((2, 2), dtype=np.int64)
    previous[1] = latest.previous_mode
    available[1] = latest.available_mask
    dwell[1] = latest.dwell_steps
    resolved_start = _integer_or_none(start_index, nonnegative=True)
    if resolved_start is not None and resolved_start > 0:
        older = derive_arm_hold_history_context(
            targets=targets,
            valid_mask=valid_mask,
            loss_mask=loss_mask,
            frame_indices=frame_indices,
            continuity_segments=continuity_segments,
            running_mask=running_mask,
            start_index=resolved_start - 1,
        )
        older_available = older.available_mask & latest.available_mask
        previous[0] = np.where(older_available, older.previous_mode, 0.0)
        available[0] = older_available
        dwell[0] = np.where(older_available, older.dwell_steps, 0)
    return ArmHoldHistorySequenceContext(
        previous_mode=previous,
        available_mask=available,
        dwell_steps=dwell,
        provenance=ARM_HOLD_HISTORY_CONTEXT_PROVENANCE_V1,
    )


def _freeze_counts(value: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType(
        {
            str(key): int(count)
            for key, count in sorted(value.items())
            if int(count) > 0
        }
    )


def _freeze_exact_counts(value: Mapping[str, int]) -> Mapping[str, int]:
    counts = {str(key): int(count) for key, count in sorted(value.items())}
    if any(count < 0 for count in counts.values()):
        raise ArmHoldWindowError("Counts must be nonnegative.")
    return MappingProxyType(counts)


@dataclass(frozen=True)
class ArmHoldWindowStartSelection:
    start_indices: tuple[int, ...]
    candidate_count: int
    rejection_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        starts = tuple(int(value) for value in self.start_indices)
        if (
            self.candidate_count < 0
            or len(starts) != len(set(starts))
            or any(value < 0 for value in starts)
        ):
            raise ArmHoldWindowError("Window start selection is invalid.")
        rejected = sum(int(value) for value in self.rejection_counts.values())
        if len(starts) + rejected != self.candidate_count:
            raise ArmHoldWindowError(
                "Window admission counts do not cover every candidate exactly."
            )
        object.__setattr__(self, "start_indices", starts)
        object.__setattr__(
            self, "rejection_counts", _freeze_counts(self.rejection_counts)
        )


@dataclass(frozen=True)
class ArmHoldWindowTransitionCoverage:
    event_counts: Mapping[str, int]
    covered_event_counts: Mapping[str, int]
    window_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        events = _freeze_counts(self.event_counts)
        covered = _freeze_counts(self.covered_event_counts)
        windows = _freeze_counts(self.window_counts)
        if any(covered.get(key, 0) > count for key, count in events.items()):
            raise ArmHoldWindowError(
                "Covered transition events cannot exceed source events."
            )
        if any(key not in events for key in covered) or any(
            key not in events for key in windows
        ):
            raise ArmHoldWindowError(
                "Transition coverage contains a direction absent from source events."
            )
        object.__setattr__(self, "event_counts", events)
        object.__setattr__(self, "covered_event_counts", covered)
        object.__setattr__(self, "window_counts", windows)


def summarize_arm_hold_window_transition_coverage(
    *,
    targets: np.ndarray,
    transition_mask: np.ndarray,
    start_indices: Sequence[int],
    horizon: int,
) -> ArmHoldWindowTransitionCoverage:
    """Distinguish source events, covered events, and overlapping windows."""

    resolved_targets = np.asarray(targets, dtype=np.float32)
    resolved_transition = np.asarray(transition_mask)
    if (
        resolved_targets.ndim != 2
        or resolved_targets.shape[1:] != (2,)
        or resolved_transition.shape != resolved_targets.shape
        or resolved_transition.dtype != np.bool_
    ):
        raise ArmHoldWindowError("Transition coverage arrays are invalid.")
    resolved_horizon = int(horizon)
    starts = tuple(int(value) for value in start_indices)
    if (
        resolved_horizon <= 0
        or len(starts) != len(set(starts))
        or any(
            value < 0 or value + resolved_horizon > len(resolved_targets)
            for value in starts
        )
    ):
        raise ArmHoldWindowError("Transition coverage window starts are invalid.")
    if not np.all(np.isin(resolved_targets[resolved_transition], (0.0, 1.0))):
        raise ArmHoldWindowError("Transition targets must be binary.")

    covered_rows = np.zeros(len(resolved_targets), dtype=bool)
    for start in starts:
        covered_rows[start : start + resolved_horizon] = True
    events: Counter[str] = Counter()
    covered_events: Counter[str] = Counter()
    windows: Counter[str] = Counter()
    for side_index, side in enumerate(("left", "right")):
        for target_value, direction in ((1.0, "teleop->fixed_hold"), (0.0, "fixed_hold->teleop")):
            key = f"{side}:{direction}"
            event_rows = (
                resolved_transition[:, side_index]
                & (resolved_targets[:, side_index] == target_value)
            )
            events[key] = int(np.count_nonzero(event_rows))
            covered_events[key] = int(np.count_nonzero(event_rows & covered_rows))
            windows[key] = sum(
                bool(np.any(event_rows[start : start + resolved_horizon]))
                for start in starts
            )
    return ArmHoldWindowTransitionCoverage(
        event_counts=events,
        covered_event_counts=covered_events,
        window_counts=windows,
    )


def summarize_arm_hold_window_target_counts(
    *,
    targets: np.ndarray,
    start_indices: Sequence[int],
    horizon: int,
) -> Mapping[str, int]:
    """Count window-weighted target elements for train-fitted class weights."""

    resolved_targets = np.asarray(targets, dtype=np.float32)
    resolved_horizon = int(horizon)
    starts = tuple(int(value) for value in start_indices)
    if resolved_targets.ndim != 2 or resolved_targets.shape[1:] != (2,):
        raise ArmHoldWindowError("Window target count array is invalid.")
    if (
        resolved_horizon <= 0
        or len(starts) != len(set(starts))
        or any(
            value < 0 or value + resolved_horizon > len(resolved_targets)
            for value in starts
        )
    ):
        raise ArmHoldWindowError("Window target count starts are invalid.")
    counts = {
        "left:teleop": 0,
        "left:fixed_hold": 0,
        "right:teleop": 0,
        "right:fixed_hold": 0,
    }
    for start in starts:
        window = resolved_targets[start : start + resolved_horizon]
        if not np.all(np.isin(window, (0.0, 1.0))):
            raise ArmHoldWindowError("Admitted window targets must be binary.")
        for side_index, side in enumerate(("left", "right")):
            positive = int(np.count_nonzero(window[:, side_index] == 1.0))
            counts[f"{side}:fixed_hold"] += positive
            counts[f"{side}:teleop"] += resolved_horizon - positive
    return _freeze_exact_counts(counts)


def select_arm_hold_window_starts(
    *,
    frame_indices: Sequence[int] | np.ndarray,
    session_states: Sequence[object],
    continuity_segments: Sequence[object],
    label_valid_mask: np.ndarray,
    state_valid_mask: Sequence[bool] | np.ndarray,
    front_valid_mask: Sequence[bool] | np.ndarray,
    horizon: int,
    running_session_state: str,
) -> ArmHoldWindowStartSelection:
    """Select full windows with exclusive, deterministic rejection reasons."""

    count = len(session_states)
    frames = np.asarray(frame_indices, dtype=np.int64)
    continuity = np.asarray(continuity_segments, dtype=object)
    labels = np.asarray(label_valid_mask, dtype=bool)
    states = np.asarray(state_valid_mask, dtype=bool)
    front = np.asarray(front_valid_mask, dtype=bool)
    if (
        frames.shape != (count,)
        or continuity.shape != (count,)
        or labels.shape != (count, 2)
        or states.shape != (count,)
        or front.shape != (count,)
    ):
        raise ArmHoldWindowError("Window eligibility arrays have incompatible shapes.")
    resolved_horizon = int(horizon)
    if resolved_horizon <= 0:
        raise ArmHoldWindowError("horizon must be positive.")
    candidate_count = max(0, count - resolved_horizon + 1)
    starts: list[int] = []
    rejected: Counter[str] = Counter()
    for start in range(candidate_count):
        stop = start + resolved_horizon
        expected_frames = np.arange(
            int(frames[start]), int(frames[start]) + resolved_horizon
        )
        if not np.array_equal(frames[start:stop], expected_frames):
            rejected["FRAME_GAP"] += 1
        elif any(str(value) != running_session_state for value in session_states[start:stop]):
            rejected["NON_RUNNING"] += 1
        elif (
            continuity[start] is None
            or any(value is None or value != continuity[start] for value in continuity[start:stop])
        ):
            rejected["CONTINUITY_BREAK"] += 1
        elif not bool(np.all(labels[start:stop])):
            rejected["LABEL_INVALID"] += 1
        elif not bool(states[start]):
            rejected["STATE_INVALID"] += 1
        elif not bool(front[start]):
            rejected["FRONT_FRAME_INVALID"] += 1
        else:
            starts.append(start)
    return ArmHoldWindowStartSelection(
        start_indices=tuple(starts),
        candidate_count=candidate_count,
        rejection_counts=rejected,
    )


@dataclass(frozen=True)
class ArmHoldWindowEpisodeBinding:
    source_dataset_id: str
    source_episode_index: int
    episode_uuid: str
    episode_capture_identity_hash: str
    annotation_set_id: str
    frame_count: int
    state_dim: int
    parquet_path: Path
    front_video_path: Path
    dataset_info: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.source_episode_index < 0 or self.frame_count <= 0 or self.state_dim <= 0:
            raise ArmHoldWindowError("Episode binding counts are invalid.")
        for name in (
            "source_dataset_id",
            "episode_uuid",
            "episode_capture_identity_hash",
            "annotation_set_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ArmHoldWindowError(f"{name} must be non-empty.")
        parquet = Path(self.parquet_path).expanduser().resolve()
        video = Path(self.front_video_path).expanduser().resolve()
        if not parquet.is_file() or not video.is_file():
            raise ArmHoldWindowError("Episode Parquet/front video is missing.")
        object.__setattr__(self, "parquet_path", parquet)
        object.__setattr__(self, "front_video_path", video)
        object.__setattr__(self, "dataset_info", MappingProxyType(dict(self.dataset_info)))


@dataclass(frozen=True)
class ArmHoldWindowLocation:
    source_dataset_id: str
    source_episode_index: int
    episode_capture_identity_hash: str
    start_index: int
    front_video_frame_index: int
    continuity_segment_index: int
    contains_transition: bool = False

    def __post_init__(self) -> None:
        if (
            self.source_episode_index < 0
            or self.start_index < 0
            or self.front_video_frame_index < 0
        ):
            raise ArmHoldWindowError("Window location indices are invalid.")
        if not isinstance(self.contains_transition, (bool, np.bool_)):
            raise ArmHoldWindowError("contains_transition must be bool.")
        object.__setattr__(self, "contains_transition", bool(self.contains_transition))


@dataclass(frozen=True)
class ArmHoldWindowInventory:
    manifest_hash: str
    profile: ArmHoldDatasetProfile
    split: str
    episodes: tuple[ArmHoldWindowEpisodeBinding, ...]
    locations: tuple[ArmHoldWindowLocation, ...]
    candidate_count: int
    rejection_counts: Mapping[str, int]
    window_counts_by_source: Mapping[str, int]
    transition_event_counts: Mapping[str, int] = field(default_factory=dict)
    covered_transition_event_counts: Mapping[str, int] = field(default_factory=dict)
    transition_window_counts: Mapping[str, int] = field(default_factory=dict)
    target_element_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in ARM_HOLD_WINDOW_SPLITS:
            raise ArmHoldWindowError(f"Unsupported split {self.split!r}.")
        if self.manifest_hash == "" or self.candidate_count < len(self.locations):
            raise ArmHoldWindowError("Window inventory identity/counts are invalid.")
        episodes = tuple(
            sorted(
                self.episodes,
                key=lambda item: (
                    item.source_dataset_id,
                    item.source_episode_index,
                    item.episode_capture_identity_hash,
                ),
            )
        )
        identities = tuple(item.episode_capture_identity_hash for item in episodes)
        if len(identities) != len(set(identities)):
            raise ArmHoldWindowError("Window inventory has duplicate Episode bindings.")
        known = set(identities)
        locations = tuple(
            sorted(
                self.locations,
                key=lambda item: (
                    item.source_dataset_id,
                    item.source_episode_index,
                    item.start_index,
                    item.episode_capture_identity_hash,
                ),
            )
        )
        if any(item.episode_capture_identity_hash not in known for item in locations):
            raise ArmHoldWindowError("Window location has no Episode binding.")
        rejected = _freeze_counts(self.rejection_counts)
        if len(locations) + sum(rejected.values()) != self.candidate_count:
            raise ArmHoldWindowError("Inventory counts do not cover all candidates.")
        source_counts = _freeze_counts(self.window_counts_by_source)
        if sum(source_counts.values()) != len(locations):
            raise ArmHoldWindowError("Per-source window counts are inconsistent.")
        transition_coverage = ArmHoldWindowTransitionCoverage(
            event_counts=self.transition_event_counts,
            covered_event_counts=self.covered_transition_event_counts,
            window_counts=self.transition_window_counts,
        )
        target_counts = _freeze_exact_counts(self.target_element_counts)
        if target_counts:
            expected_keys = {
                "left:teleop",
                "left:fixed_hold",
                "right:teleop",
                "right:fixed_hold",
            }
            if set(target_counts) != expected_keys:
                raise ArmHoldWindowError("Target element counts are incomplete.")
            expected_elements = len(locations) * int(self.profile.sample_view["horizon"])
            for side in ("left", "right"):
                if (
                    target_counts[f"{side}:teleop"]
                    + target_counts[f"{side}:fixed_hold"]
                    != expected_elements
                ):
                    raise ArmHoldWindowError(
                        "Target element counts do not cover every window target."
                    )
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "locations", locations)
        object.__setattr__(self, "rejection_counts", rejected)
        object.__setattr__(self, "window_counts_by_source", source_counts)
        object.__setattr__(
            self, "transition_event_counts", transition_coverage.event_counts
        )
        object.__setattr__(
            self,
            "covered_transition_event_counts",
            transition_coverage.covered_event_counts,
        )
        object.__setattr__(
            self, "transition_window_counts", transition_coverage.window_counts
        )
        object.__setattr__(self, "target_element_counts", target_counts)

    def summary_payload(self, *, debug: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "manifest_hash": self.manifest_hash,
            "profile_hash": self.profile.semantic_content_hash,
            "split": self.split,
            "episode_count": len(self.episodes),
            "candidate_count": self.candidate_count,
            "window_count": len(self.locations),
            "rejection_counts": dict(self.rejection_counts),
            "window_counts_by_source": dict(self.window_counts_by_source),
            "transition_event_counts": dict(self.transition_event_counts),
            "covered_transition_event_counts": dict(
                self.covered_transition_event_counts
            ),
            "transition_window_counts": dict(self.transition_window_counts),
            "transition_window_union_count": sum(
                item.contains_transition for item in self.locations
            ),
            "target_element_counts": dict(self.target_element_counts),
        }
        if debug:
            payload["locations"] = [
                {
                    "source_dataset_id": item.source_dataset_id,
                    "source_episode_index": item.source_episode_index,
                    "episode_capture_identity_hash": item.episode_capture_identity_hash,
                    "start_index": item.start_index,
                    "front_video_frame_index": item.front_video_frame_index,
                    "continuity_segment_index": item.continuity_segment_index,
                    "contains_transition": item.contains_transition,
                }
                for item in self.locations
            ]
        return payload


def _state_valid_mask(values: Sequence[object], *, state_dim: int) -> np.ndarray:
    result = np.zeros(len(values), dtype=bool)
    for index, value in enumerate(values):
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError):
            continue
        result[index] = array.shape == (state_dim,) and bool(np.all(np.isfinite(array)))
    return result


def _integer_or_none(value: object, *, nonnegative: bool) -> int | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        return None
    result = int(value)
    if nonnegative and result < 0:
        return None
    return result


def _front_valid_mask(values: Sequence[object]) -> np.ndarray:
    result = np.zeros(len(values), dtype=bool)
    for index, value in enumerate(values):
        if _integer_or_none(value, nonnegative=True) is not None:
            result[index] = True
    return result


def _parquet_columns(path: Path, names: Sequence[str]) -> dict[str, list[Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ArmHoldWindowError("pyarrow is required for arm-hold windows.") from exc
    try:
        parquet = pq.ParquetFile(path)
        missing = sorted(set(names) - set(parquet.schema_arrow.names))
        if missing:
            raise ArmHoldWindowError(f"Episode Parquet is missing columns {missing!r}.")
        table = pq.read_table(path, columns=list(names))
    except ArmHoldWindowError:
        raise
    except Exception as exc:
        raise ArmHoldWindowError(f"Cannot read Episode Parquet {path}: {exc}") from exc
    return {name: table[name].to_pylist() for name in names}


def build_arm_hold_window_inventory(
    manifest: ArmHoldSplitManifest,
    workspace_root: str | Path,
    *,
    split: str,
) -> ArmHoldWindowInventory:
    """Validate frozen membership and enumerate legal windows without decoding RGB."""

    resolved_split = str(split).strip().lower()
    if resolved_split not in ARM_HOLD_WINDOW_SPLITS:
        raise ArmHoldWindowError(f"Unsupported split {split!r}.")
    profile = manifest.profile
    if (
        profile.sample_view["invalid_label_policy"]
        != ARM_HOLD_INVALID_LABEL_POLICY_V1
    ):
        raise ArmHoldWindowError("Unsupported invalid-label window policy.")
    selected_identities = set(
        manifest.cohort.split.train_episode_identities
        if resolved_split == "train"
        else manifest.cohort.split.validation_episode_identities
    )
    records = {
        item.episode_capture_identity_hash: item
        for item in manifest.cohort.included_episodes
        if item.episode_capture_identity_hash in selected_identities
    }
    if set(records) != selected_identities:
        raise ArmHoldWindowError("Split identities are not covered by manifest Episodes.")
    roots = resolve_arm_hold_source_roots(profile, workspace_root)
    receipts = {
        item.source_dataset_id: item for item in manifest.cohort.source_receipts
    }
    required_columns = tuple(
        dict.fromkeys(str(value) for value in profile.episode_admission["required_columns"])
    )
    camera_key = str(profile.sample_view["camera_key"])
    state_key = str(profile.sample_view["state_key"])
    front_index_key = f"observation.images.{camera_key}.video_frame_index"
    horizon = int(profile.sample_view["horizon"])
    running_state = str(profile.sample_view["running_session_state"])
    episodes: list[ArmHoldWindowEpisodeBinding] = []
    locations: list[ArmHoldWindowLocation] = []
    rejection_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    transition_events: Counter[str] = Counter()
    covered_transition_events: Counter[str] = Counter()
    transition_windows: Counter[str] = Counter()
    target_elements: Counter[str] = Counter(
        {
            "left:teleop": 0,
            "left:fixed_hold": 0,
            "right:teleop": 0,
            "right:fixed_hold": 0,
        }
    )
    candidate_count = 0
    accounted: set[str] = set()

    for source in profile.sources:
        source_id = source.source_dataset_id
        source_records = tuple(
            sorted(
                (item for item in records.values() if item.source_dataset_id == source_id),
                key=lambda item: (
                    item.source_episode_index,
                    item.episode_capture_identity_hash,
                ),
            )
        )
        if not source_records:
            continue
        root = roots[source_id]
        if source_dataset_base_fingerprint(root) != receipts[source_id].source_dataset_fingerprint:
            raise ArmHoldWindowError(f"Source dataset fingerprint drift: {source_id}.")
        try:
            info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
            resolved_contract = resolve_v23_dataset_contract(info)
        except Exception as exc:
            raise ArmHoldWindowError(f"Cannot resolve source contract {source_id}: {exc}") from exc
        state_dim = int(resolved_contract.policy_io_layout.state_dim)
        items = list_progress_annotation_work_items(root)
        by_identity = {item.episode_identity_hash: item for item in items}
        if len(by_identity) != len(items):
            raise ArmHoldWindowError(f"Source {source_id} has duplicate Episode identities.")

        for record in source_records:
            item = by_identity.get(record.episode_capture_identity_hash)
            if item is None:
                raise ArmHoldWindowError(
                    f"Frozen Episode is absent from source {source_id}: "
                    f"{record.episode_capture_identity_hash}."
                )
            if (
                item.episode_index != record.source_episode_index
                or item.episode_uuid != record.episode_uuid
                or item.frame_count != record.frame_count
            ):
                raise ArmHoldWindowError("Frozen Episode binding differs from source facts.")
            try:
                annotation = _select_annotation(item, record.annotation_set_id)
            except ValueError as exc:
                raise ArmHoldWindowError(str(exc)) from exc
            if (
                annotation is None
                or annotation.binding.annotation_set_id != record.annotation_set_id
            ):
                raise ArmHoldWindowError("Frozen annotation binding is unavailable.")
            video_path = item.video_paths.get(camera_key)
            if video_path is None or not Path(video_path).is_file():
                raise ArmHoldWindowError(
                    f"Episode {record.episode_capture_identity_hash} has no {camera_key} video."
                )
            columns = _parquet_columns(item.parquet_path, required_columns)
            if len(columns["frame_index"]) != item.frame_count:
                raise ArmHoldWindowError("Episode frame count differs from Parquet rows.")
            view = project_arm_hold_training_view(columns, metadata=info, strict=True)
            normalized_frames = tuple(
                _integer_or_none(value, nonnegative=True)
                for value in columns["frame_index"]
            )
            if any(value is None for value in normalized_frames):
                raise ArmHoldWindowError("Episode frame_index contains invalid values.")
            frame_indices = np.asarray(normalized_frames, dtype=np.int64)
            states_valid = _state_valid_mask(columns[state_key], state_dim=state_dim)
            front_valid = _front_valid_mask(columns[front_index_key])
            continuity_values = tuple(
                _integer_or_none(value, nonnegative=True)
                for value in columns["collection.continuity_segment_index"]
            )
            selection = select_arm_hold_window_starts(
                frame_indices=frame_indices,
                session_states=columns["session_state"],
                continuity_segments=continuity_values,
                label_valid_mask=view.valid_mask,
                state_valid_mask=states_valid,
                front_valid_mask=front_valid,
                horizon=horizon,
                running_session_state=running_state,
            )
            transition_coverage = summarize_arm_hold_window_transition_coverage(
                targets=view.targets,
                transition_mask=view.transition_mask,
                start_indices=selection.start_indices,
                horizon=horizon,
            )
            episode_target_counts = summarize_arm_hold_window_target_counts(
                targets=view.targets,
                start_indices=selection.start_indices,
                horizon=horizon,
            )
            candidate_count += selection.candidate_count
            rejection_counts.update(selection.rejection_counts)
            transition_events.update(transition_coverage.event_counts)
            covered_transition_events.update(
                transition_coverage.covered_event_counts
            )
            transition_windows.update(transition_coverage.window_counts)
            target_elements.update(episode_target_counts)
            binding = ArmHoldWindowEpisodeBinding(
                source_dataset_id=source_id,
                source_episode_index=item.episode_index,
                episode_uuid=item.episode_uuid,
                episode_capture_identity_hash=item.episode_identity_hash,
                annotation_set_id=record.annotation_set_id,
                frame_count=item.frame_count,
                state_dim=state_dim,
                parquet_path=item.parquet_path,
                front_video_path=Path(video_path),
                dataset_info=info,
            )
            episodes.append(binding)
            for start in selection.start_indices:
                front_frame = _integer_or_none(
                    columns[front_index_key][start], nonnegative=True
                )
                continuity = continuity_values[start]
                assert front_frame is not None and continuity is not None
                locations.append(
                    ArmHoldWindowLocation(
                        source_dataset_id=source_id,
                        source_episode_index=item.episode_index,
                        episode_capture_identity_hash=item.episode_identity_hash,
                        start_index=start,
                        front_video_frame_index=front_frame,
                        continuity_segment_index=continuity,
                        contains_transition=bool(
                            np.any(view.transition_mask[start : start + horizon])
                        ),
                    )
                )
                source_counts[source_id] += 1
            accounted.add(record.episode_capture_identity_hash)
    if accounted != selected_identities:
        raise ArmHoldWindowError("Window inventory did not account for the exact split.")
    return ArmHoldWindowInventory(
        manifest_hash=manifest.semantic_content_hash,
        profile=profile,
        split=resolved_split,
        episodes=tuple(episodes),
        locations=tuple(locations),
        candidate_count=candidate_count,
        rejection_counts=rejection_counts,
        window_counts_by_source=source_counts,
        transition_event_counts=transition_events,
        covered_transition_event_counts=covered_transition_events,
        transition_window_counts=transition_windows,
        target_element_counts=target_elements,
    )


@dataclass(frozen=True)
class ArmHoldWindowSample:
    front_rgb: np.ndarray | None
    observation_state: np.ndarray | None
    arm_hold_target: np.ndarray
    arm_hold_valid_mask: np.ndarray
    arm_hold_loss_mask: np.ndarray
    arm_hold_transition_mask: np.ndarray
    arm_hold_history_context: ArmHoldHistoryContext | None = None
    arm_hold_history_sequence_context: ArmHoldHistorySequenceContext | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        target = np.asarray(self.arm_hold_target, dtype=np.float32)
        valid = np.asarray(self.arm_hold_valid_mask, dtype=bool)
        loss = np.asarray(self.arm_hold_loss_mask, dtype=bool)
        transition = np.asarray(self.arm_hold_transition_mask, dtype=bool)
        if (
            target.ndim != 2
            or target.shape[1:] != (2,)
            or valid.shape != target.shape
            or loss.shape != target.shape
            or transition.shape != target.shape
        ):
            raise ArmHoldWindowError("Arm-hold sample target arrays are invalid.")
        if not np.all(valid) or not np.all(loss):
            raise ArmHoldWindowError("Admitted windows must have fully valid labels.")
        state = None
        if self.observation_state is not None:
            state = np.asarray(self.observation_state, dtype=np.float32)
            if state.ndim != 1 or not np.all(np.isfinite(state)):
                raise ArmHoldWindowError("Sample observation_state is invalid.")
            state = state.copy()
        image = None
        if self.front_rgb is not None:
            image = np.asarray(self.front_rgb)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ArmHoldWindowError("Sample front_rgb must be HWC uint8 RGB.")
            image = image.copy()
        object.__setattr__(self, "front_rgb", image)
        object.__setattr__(self, "observation_state", state)
        object.__setattr__(self, "arm_hold_target", target.copy())
        object.__setattr__(self, "arm_hold_valid_mask", valid.copy())
        object.__setattr__(self, "arm_hold_loss_mask", loss.copy())
        object.__setattr__(self, "arm_hold_transition_mask", transition.copy())
        if self.arm_hold_history_context is not None and not isinstance(
            self.arm_hold_history_context, ArmHoldHistoryContext
        ):
            raise ArmHoldWindowError(
                "arm_hold_history_context must use the typed context contract."
            )
        if self.arm_hold_history_sequence_context is not None and not isinstance(
            self.arm_hold_history_sequence_context, ArmHoldHistorySequenceContext
        ):
            raise ArmHoldWindowError(
                "arm_hold_history_sequence_context must use the typed contract."
            )
        if (
            self.arm_hold_history_context is not None
            and self.arm_hold_history_sequence_context is not None
        ):
            raise ArmHoldWindowError(
                "A sample cannot carry one-step and two-step history together."
            )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_training_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "arm_hold_target": self.arm_hold_target.copy(),
            "arm_hold_valid_mask": self.arm_hold_valid_mask.copy(),
            "arm_hold_loss_mask": self.arm_hold_loss_mask.copy(),
            "arm_hold_transition_mask": self.arm_hold_transition_mask.copy(),
            "metadata": dict(self.metadata),
        }
        if self.front_rgb is not None:
            payload["front_rgb"] = self.front_rgb.copy()
        if self.observation_state is not None:
            payload["observation.state"] = self.observation_state.copy()
        if self.arm_hold_history_context is not None:
            payload["arm_hold_history_context"] = (
                self.arm_hold_history_context.to_training_dict()
            )
        if self.arm_hold_history_sequence_context is not None:
            payload["arm_hold_history_sequence_context"] = (
                self.arm_hold_history_sequence_context.to_training_dict()
            )
        return payload


@dataclass(frozen=True)
class _LoadedEpisode:
    states: tuple[np.ndarray | None, ...]
    frame_indices: tuple[int, ...]
    running_mask: np.ndarray
    front_frame_indices: tuple[int | None, ...]
    continuity_segments: tuple[int | None, ...]
    view: ArmHoldTrainingView


class ArmHoldWindowReader:
    """Lazy sample reader; construct one instance per DataLoader worker."""

    def __init__(
        self,
        inventory: ArmHoldWindowInventory,
        *,
        input_mode: str = "front_state",
        include_history_context: bool = False,
        history_context_steps: int = 1,
        max_episode_cache: int = 8,
        max_video_cache: int = 4,
    ) -> None:
        mode = str(input_mode).strip().lower()
        if mode not in ARM_HOLD_INPUT_MODES:
            raise ArmHoldWindowError(f"Unsupported input_mode {input_mode!r}.")
        self.inventory = inventory
        self.input_mode = mode
        self.include_history_context = bool(include_history_context)
        resolved_history_steps = int(history_context_steps)
        if resolved_history_steps not in (1, 2):
            raise ArmHoldWindowError("history_context_steps must be 1 or 2.")
        if not self.include_history_context and resolved_history_steps != 1:
            raise ArmHoldWindowError(
                "Two-step history requires include_history_context=True."
            )
        self.history_context_steps = resolved_history_steps
        self._episode_bindings = {
            item.episode_capture_identity_hash: item for item in inventory.episodes
        }
        self._episode_cache_size = max(1, int(max_episode_cache))
        self._episode_cache: OrderedDict[str, _LoadedEpisode] = OrderedDict()
        self._videos = VideoReaderCache(max_size=max_video_cache)

    def close(self) -> None:
        self._episode_cache.clear()
        self._videos.close()

    def __enter__(self) -> "ArmHoldWindowReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.inventory.locations)

    def _load_episode(self, identity: str) -> _LoadedEpisode:
        cached = self._episode_cache.get(identity)
        if cached is not None:
            self._episode_cache.move_to_end(identity)
            return cached
        binding = self._episode_bindings[identity]
        profile = self.inventory.profile
        names = tuple(
            dict.fromkeys(str(value) for value in profile.episode_admission["required_columns"])
        )
        columns = _parquet_columns(binding.parquet_path, names)
        if len(columns["frame_index"]) != binding.frame_count:
            raise ArmHoldWindowError("Episode changed after window inventory creation.")
        view = project_arm_hold_training_view(
            columns, metadata=binding.dataset_info, strict=True
        )
        states: list[np.ndarray | None] = []
        for value in columns[str(profile.sample_view["state_key"])]:
            try:
                state = np.asarray(value, dtype=np.float32)
            except (TypeError, ValueError):
                state = np.empty((0,), dtype=np.float32)
            states.append(
                state.copy()
                if state.shape == (binding.state_dim,) and np.all(np.isfinite(state))
                else None
            )
        camera = str(profile.sample_view["camera_key"])
        video_key = f"observation.images.{camera}.video_frame_index"
        video_indices = tuple(
            _integer_or_none(value, nonnegative=True)
            for value in columns[video_key]
        )
        continuity = tuple(
            _integer_or_none(value, nonnegative=True)
            for value in columns["collection.continuity_segment_index"]
        )
        resolved_frame_indices = tuple(
            _integer_or_none(item, nonnegative=True)
            for item in columns["frame_index"]
        )
        if any(value is None for value in resolved_frame_indices):
            raise ArmHoldWindowError("Episode frame_index became invalid.")
        frame_indices = tuple(
            value for value in resolved_frame_indices if value is not None
        )
        running_state = str(profile.sample_view["running_session_state"])
        running_mask = np.asarray(
            [str(value) == running_state for value in columns["session_state"]],
            dtype=bool,
        )
        loaded = _LoadedEpisode(
            states=tuple(states),
            frame_indices=frame_indices,
            running_mask=running_mask,
            front_frame_indices=video_indices,
            continuity_segments=continuity,
            view=view,
        )
        self._episode_cache[identity] = loaded
        if len(self._episode_cache) > self._episode_cache_size:
            self._episode_cache.popitem(last=False)
        return loaded

    def __getitem__(self, index: int) -> ArmHoldWindowSample:
        location = self.inventory.locations[int(index)]
        binding = self._episode_bindings[location.episode_capture_identity_hash]
        loaded = self._load_episode(location.episode_capture_identity_hash)
        horizon = int(self.inventory.profile.sample_view["horizon"])
        start = location.start_index
        stop = start + horizon
        valid = loaded.view.valid_mask[start:stop]
        loss = loaded.view.loss_mask[start:stop]
        if valid.shape != (horizon, 2) or not np.all(valid) or not np.all(loss):
            raise ArmHoldWindowError("Frozen window labels are no longer fully valid.")
        if loaded.front_frame_indices[start] != location.front_video_frame_index:
            raise ArmHoldWindowError("Front frame locator changed after inventory creation.")
        if loaded.continuity_segments[start] != location.continuity_segment_index:
            raise ArmHoldWindowError("Continuity binding changed after inventory creation.")
        contains_transition = bool(np.any(loaded.view.transition_mask[start:stop]))
        if contains_transition != location.contains_transition:
            raise ArmHoldWindowError(
                "Transition membership changed after inventory creation."
            )
        state = None
        if self.input_mode in {"front_state", "state_only"}:
            state = loaded.states[start]
            if state is None:
                raise ArmHoldWindowError("Admitted window state became invalid.")
        front_rgb = None
        if self.input_mode in {"front_state", "front_only"}:
            bgr = self._videos.read(
                binding.front_video_path, location.front_video_frame_index
            )
            front_rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        history_context = None
        history_sequence_context = None
        if self.include_history_context:
            history_args = {
                "targets": loaded.view.targets,
                "valid_mask": loaded.view.valid_mask,
                "loss_mask": loaded.view.loss_mask,
                "frame_indices": loaded.frame_indices,
                "continuity_segments": loaded.continuity_segments,
                "running_mask": loaded.running_mask,
                "start_index": start,
            }
            if self.history_context_steps == 1:
                history_context = derive_arm_hold_history_context(**history_args)
            else:
                history_sequence_context = (
                    derive_arm_hold_history_sequence_context(**history_args)
                )
        return ArmHoldWindowSample(
            front_rgb=front_rgb,
            observation_state=state,
            arm_hold_target=loaded.view.targets[start:stop],
            arm_hold_valid_mask=valid,
            arm_hold_loss_mask=loss,
            arm_hold_transition_mask=loaded.view.transition_mask[start:stop],
            arm_hold_history_context=history_context,
            arm_hold_history_sequence_context=history_sequence_context,
            metadata={
                "manifest_hash": self.inventory.manifest_hash,
                "profile_hash": self.inventory.profile.semantic_content_hash,
                "split": self.inventory.split,
                "source_dataset_id": location.source_dataset_id,
                "source_episode_index": location.source_episode_index,
                "episode_capture_identity_hash": location.episode_capture_identity_hash,
                "start_index": start,
                "horizon": horizon,
                "front_video_frame_index": location.front_video_frame_index,
                "continuity_segment_index": location.continuity_segment_index,
                "input_mode": self.input_mode,
            },
        )


__all__ = [
    "ARM_HOLD_HISTORY_CONTEXT_PROVENANCE_V1",
    "ARM_HOLD_HISTORY_CONTEXT_SCHEMA_V1",
    "ARM_HOLD_HISTORY_SEQUENCE_CONTEXT_SCHEMA_V1",
    "ARM_HOLD_INPUT_MODES",
    "ARM_HOLD_INVALID_LABEL_POLICY_V1",
    "ARM_HOLD_WINDOW_SPLITS",
    "ArmHoldHistoryContext",
    "ArmHoldHistorySequenceContext",
    "ArmHoldHistoryTimeline",
    "ArmHoldWindowEpisodeBinding",
    "ArmHoldWindowError",
    "ArmHoldWindowInventory",
    "ArmHoldWindowLocation",
    "ArmHoldWindowReader",
    "ArmHoldWindowSample",
    "ArmHoldWindowStartSelection",
    "ArmHoldWindowTransitionCoverage",
    "build_arm_hold_window_inventory",
    "derive_arm_hold_history_context",
    "derive_arm_hold_history_sequence_context",
    "derive_arm_hold_history_timeline",
    "select_arm_hold_window_starts",
    "summarize_arm_hold_window_transition_coverage",
    "summarize_arm_hold_window_target_counts",
]
