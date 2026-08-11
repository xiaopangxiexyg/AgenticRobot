"""Discrete validation for optional recoverable-failure annotations."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.progress_recoverable_failures import (
    RecoverableFailureAnnotationV1,
    RecoverableFailureContractError,
)
from dataset_contracts.progress_stage_annotations import (
    StageOccurrenceAnnotationV1,
)

from .sample_axis import EpisodeSampleAxis


RECOVERABLE_FAILURE_SAMPLE_AXIS_HASH_DOMAIN_V1 = (
    "dexteleop.recoverable_failure.sample_axis.semantic.v1"
)
_TIME_TOLERANCE = 1e-7


class RecoverableFailureResolutionError(ValueError):
    """Raised when an observation cannot be bound to one success stage."""


@dataclass(frozen=True)
class RecoverableFailureStageWindow:
    """The half-open sample window owned by one successful occurrence."""

    stage_occurrence_annotation_id: str
    coarse_attempt_id: str
    progress_step_id: str
    stage_id: str
    sample_index_begin: int
    sample_index_end_exclusive: int


def recoverable_failure_sample_axis_hash(axis: EpisodeSampleAxis) -> str:
    if not isinstance(axis, EpisodeSampleAxis):
        raise TypeError("axis must be EpisodeSampleAxis.")
    return canonical_content_hash(
        {
            "clock_domain_id": axis.clock_domain_id,
            "cut_times": axis.cut_times,
        },
        domain=RECOVERABLE_FAILURE_SAMPLE_AXIS_HASH_DOMAIN_V1,
    )


def derive_recoverable_failure_stage_windows(
    *,
    occurrences: Sequence[StageOccurrenceAnnotationV1],
    sample_axis: EpisodeSampleAxis,
) -> tuple[RecoverableFailureStageWindow, ...]:
    """Return sample windows for existing successful stage occurrences only."""

    windows: list[RecoverableFailureStageWindow] = []
    seen_ids: set[str] = set()
    for occurrence in occurrences:
        if occurrence.clock_domain_id != sample_axis.clock_domain_id:
            raise RecoverableFailureResolutionError(
                "Stage occurrence and sample axis use different clocks."
            )
        if occurrence.annotation_id in seen_ids:
            raise RecoverableFailureResolutionError(
                "Stage occurrence annotation IDs must be unique."
            )
        seen_ids.add(occurrence.annotation_id)
        if occurrence.outcome != "SUCCESS":
            continue
        interval = sample_axis.half_open_time_interval(
            start_at=occurrence.start_at,
            end_at=occurrence.end_at,
        )
        if interval.sample_count < 2:
            continue
        windows.append(
            RecoverableFailureStageWindow(
                stage_occurrence_annotation_id=occurrence.annotation_id,
                coarse_attempt_id=occurrence.coarse_attempt_id,
                progress_step_id=occurrence.progress_step_id,
                stage_id=occurrence.stage_id,
                sample_index_begin=interval.sample_index_begin,
                sample_index_end_exclusive=(
                    interval.sample_index_end_exclusive
                ),
            )
        )
    return tuple(windows)


def _nearest_sample_index(axis: EpisodeSampleAxis, value: object, *, name: str) -> int:
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise RecoverableFailureResolutionError(
            f"{name} must be a finite timestamp."
        ) from exc
    if not math.isfinite(timestamp) or not axis.cut_times:
        raise RecoverableFailureResolutionError(
            f"{name} must be represented by a non-empty sample axis."
        )
    if (
        timestamp < axis.cut_times[0] - _TIME_TOLERANCE
        or timestamp > axis.cut_times[-1] + _TIME_TOLERANCE
    ):
        raise RecoverableFailureResolutionError(
            f"{name} lies outside the Episode sample axis."
        )
    right = bisect_left(axis.cut_times, timestamp)
    if right <= 0:
        return 0
    if right >= axis.sample_count:
        return axis.sample_count - 1
    left = right - 1
    if timestamp - axis.cut_times[left] <= axis.cut_times[right] - timestamp:
        return left
    return right


def _raw_sample_index(
    raw: Mapping[str, Any],
    *,
    index_name: str,
    time_name: str,
    sample_axis: EpisodeSampleAxis,
) -> int:
    if index_name in raw:
        value = raw[index_name]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value >= sample_axis.sample_count
        ):
            raise RecoverableFailureResolutionError(
                f"{index_name} must reference an existing sample."
            )
        return value
    if time_name not in raw:
        raise RecoverableFailureResolutionError(
            f"Every recoverable failure requires {time_name}."
        )
    return _nearest_sample_index(
        sample_axis,
        raw[time_name],
        name=time_name,
    )


def _matching_window(
    windows: Sequence[RecoverableFailureStageWindow],
    *,
    onset_index: int,
    recovered_index: int,
) -> RecoverableFailureStageWindow:
    matches = tuple(
        window
        for window in windows
        if window.sample_index_begin <= onset_index
        and recovered_index < window.sample_index_end_exclusive
    )
    if not matches:
        raise RecoverableFailureResolutionError(
            "Recoverable-failure onset and recovery must both lie inside one "
            "successful stage occurrence. Adjust the white stage boundary or "
            "choose both points again."
        )
    if len(matches) != 1:
        raise RecoverableFailureResolutionError(
            "Recoverable-failure stage ownership is ambiguous."
        )
    return matches[0]


def _check_declared_reference(
    raw: Mapping[str, Any],
    *,
    name: str,
    expected: str,
) -> None:
    declared = raw.get(name)
    if declared not in (None, "") and str(declared).strip() != expected:
        raise RecoverableFailureResolutionError(
            "Recoverable-failure stage reference changed after its points "
            "were selected; remove and add it again."
        )


def normalize_recoverable_failure_drafts(
    *,
    occurrences: Sequence[StageOccurrenceAnnotationV1],
    sample_axis: EpisodeSampleAxis,
    existing: Sequence[Mapping[str, Any]] | None = None,
    allow_adjacent_legacy: bool = False,
) -> list[dict[str, Any]]:
    """Snap draft points to samples and bind each interval to one success stage."""

    windows = derive_recoverable_failure_stage_windows(
        occurrences=occurrences,
        sample_axis=sample_axis,
    )
    normalized: list[dict[str, Any]] = []
    annotation_ids: set[str] = set()
    for ordinal, raw in enumerate(tuple(existing or ()), start=1):
        if not isinstance(raw, Mapping):
            raise RecoverableFailureResolutionError(
                "Every recoverable_failures draft item must be an object."
            )
        annotation_id = str(
            raw.get("annotation_id", f"recoverable-failure:draft:{ordinal}")
        ).strip()
        if not annotation_id or annotation_id in annotation_ids:
            raise RecoverableFailureResolutionError(
                "Recoverable-failure annotation IDs must be non-empty and unique."
            )
        annotation_ids.add(annotation_id)
        onset_index = _raw_sample_index(
            raw,
            index_name="onset_sample_index",
            time_name="onset_at",
            sample_axis=sample_axis,
        )
        recovered_index = _raw_sample_index(
            raw,
            index_name="recovered_sample_index",
            time_name="recovered_at",
            sample_axis=sample_axis,
        )
        if recovered_index <= onset_index:
            raise RecoverableFailureResolutionError(
                "Recoverable-failure recovery sample must follow its onset sample."
            )
        onset_at = sample_axis.cut_times[onset_index]
        recovered_at = sample_axis.cut_times[recovered_index]
        if recovered_at <= onset_at + _TIME_TOLERANCE:
            raise RecoverableFailureResolutionError(
                "Recoverable-failure points must have distinct sample timestamps."
            )
        window = _matching_window(
            windows,
            onset_index=onset_index,
            recovered_index=recovered_index,
        )
        for name, expected in (
            (
                "stage_occurrence_annotation_id",
                window.stage_occurrence_annotation_id,
            ),
            ("coarse_attempt_id", window.coarse_attempt_id),
            ("progress_step_id", window.progress_step_id),
            ("stage_id", window.stage_id),
        ):
            _check_declared_reference(raw, name=name, expected=expected)
        note = raw.get("note")
        if note is not None and not isinstance(note, str):
            raise RecoverableFailureResolutionError(
                "Recoverable-failure note must be a string or null."
            )
        normalized.append(
            {
                "annotation_id": annotation_id,
                "stage_occurrence_annotation_id": (
                    window.stage_occurrence_annotation_id
                ),
                "coarse_attempt_id": window.coarse_attempt_id,
                "progress_step_id": window.progress_step_id,
                "stage_id": window.stage_id,
                "onset_sample_index": onset_index,
                "recovered_sample_index": recovered_index,
                "onset_at": onset_at,
                "recovered_at": recovered_at,
                "clock_domain_id": sample_axis.clock_domain_id,
                "note": None if note is None else note.strip() or None,
            }
        )
    normalized.sort(
        key=lambda item: (
            item["onset_sample_index"],
            item["recovered_sample_index"],
            item["annotation_id"],
        )
    )
    by_occurrence: dict[str, list[dict[str, Any]]] = {}
    for item in normalized:
        by_occurrence.setdefault(
            item["stage_occurrence_annotation_id"],
            [],
        ).append(item)
    for values in by_occurrence.values():
        for previous, current in zip(values, values[1:]):
            if (
                current["onset_sample_index"]
                < previous["recovered_sample_index"]
            ):
                raise RecoverableFailureResolutionError(
                    "Recoverable-failure intervals cannot overlap in one stage."
                )
            if (
                not allow_adjacent_legacy
                and current["onset_sample_index"]
                == previous["recovered_sample_index"]
            ):
                raise RecoverableFailureResolutionError(
                    "Recoverable-failure intervals must not share one sample "
                    "as both recovery and the next onset."
                )
    return normalized


def build_recoverable_failure_annotations(
    *,
    occurrences: Sequence[StageOccurrenceAnnotationV1],
    sample_axis: EpisodeSampleAxis,
    drafts: Sequence[Mapping[str, Any]],
) -> tuple[RecoverableFailureAnnotationV1, ...]:
    normalized = normalize_recoverable_failure_drafts(
        occurrences=occurrences,
        sample_axis=sample_axis,
        existing=drafts,
    )
    try:
        return tuple(
            RecoverableFailureAnnotationV1(
                annotation_id=value["annotation_id"],
                stage_occurrence_annotation_id=value[
                    "stage_occurrence_annotation_id"
                ],
                coarse_attempt_id=value["coarse_attempt_id"],
                progress_step_id=value["progress_step_id"],
                stage_id=value["stage_id"],
                onset_sample_index=value["onset_sample_index"],
                recovered_sample_index=value["recovered_sample_index"],
                onset_at=value["onset_at"],
                recovered_at=value["recovered_at"],
                clock_domain_id=value["clock_domain_id"],
                note=value["note"],
            )
            for value in normalized
        )
    except (TypeError, ValueError, RecoverableFailureContractError) as exc:
        raise RecoverableFailureResolutionError(
            f"Invalid recoverable-failure draft: {exc}"
        ) from exc


__all__ = [
    "RECOVERABLE_FAILURE_SAMPLE_AXIS_HASH_DOMAIN_V1",
    "RecoverableFailureResolutionError",
    "RecoverableFailureStageWindow",
    "build_recoverable_failure_annotations",
    "derive_recoverable_failure_stage_windows",
    "normalize_recoverable_failure_drafts",
    "recoverable_failure_sample_axis_hash",
]
