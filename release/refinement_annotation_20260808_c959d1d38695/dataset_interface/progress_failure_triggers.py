"""Validation and sample-axis projection for failure-trigger annotations."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.progress_failure_triggers import (
    FailureTriggerAnnotationV1,
    FailureTriggerContractError,
)
from dataset_contracts.progress_stage_annotations import (
    StageOccurrenceAnnotationV1,
)

from .progress_timeline import (
    ResolvedProgressTimeline,
)
from .sample_axis import EpisodeSampleAxis


FAILURE_TRIGGER_SAMPLE_AXIS_HASH_DOMAIN_V1 = (
    "dexteleop.failure_trigger.sample_axis.semantic.v1"
)
_TIME_TOLERANCE = 1e-7


class FailureTriggerResolutionError(ValueError):
    """Raised when a trigger cannot be represented on the Episode sample axis."""


@dataclass(frozen=True)
class FailureTriggerWindow:
    """The inclusive discrete window allowed for one captured Failure."""

    source_failure_fact_id: str
    coarse_attempt_id: str
    progress_step_id: str
    stage_id: str
    lower_bound_sample_index: int
    upper_bound_sample_index: int
    lower_bound_at: float
    upper_bound_at: float


@dataclass(frozen=True)
class FailureTriggerAnalysisItem:
    """Read-only analysis view; it does not change training exclusions."""

    source_failure_fact_id: str
    coarse_attempt_id: str
    progress_step_id: str
    stage_id: str
    trigger_sample_index: int
    trigger_at: float
    failure_at: float
    normal_prefix_start_at: float
    normal_prefix_end_at: float
    failure_suffix_start_at: float
    failure_suffix_end_at: float


def failure_trigger_sample_axis_hash(axis: EpisodeSampleAxis) -> str:
    """Hash the exact discrete timestamp axis used by trigger indices."""

    if not isinstance(axis, EpisodeSampleAxis):
        raise TypeError("axis must be EpisodeSampleAxis.")
    return canonical_content_hash(
        {
            "clock_domain_id": axis.clock_domain_id,
            "cut_times": axis.cut_times,
        },
        domain=FAILURE_TRIGGER_SAMPLE_AXIS_HASH_DOMAIN_V1,
    )


def _close(left: float, right: float) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=_TIME_TOLERANCE,
    )


def _exact_sample_index(
    axis: EpisodeSampleAxis,
    at: float,
    *,
    name: str,
) -> int:
    value = float(at)
    index = bisect_left(axis.cut_times, value)
    if (
        index >= axis.sample_count
        or not _close(axis.cut_times[index], value)
    ):
        raise FailureTriggerResolutionError(
            f"{name} {value!r} is not represented by a sample timestamp."
        )
    # Repeated timestamps are one discrete coordinate; use the last sample so
    # a closed upper bound retains the complete duplicate timestamp run.
    return bisect_right(axis.cut_times, value) - 1


def _first_sample_at_or_after(
    axis: EpisodeSampleAxis,
    at: float,
    *,
    name: str,
) -> int:
    value = float(at)
    index = axis.first_at_or_after(value)
    if index >= axis.sample_count:
        raise FailureTriggerResolutionError(
            f"{name} has no sample at or after {value!r}."
        )
    return index


def _last_sample_at_or_before(
    axis: EpisodeSampleAxis,
    at: float,
    *,
    name: str,
) -> int:
    value = float(at)
    index = bisect_right(axis.cut_times, value) - 1
    if index < 0:
        raise FailureTriggerResolutionError(
            f"{name} has no sample at or before {value!r}."
        )
    if index + 1 < axis.sample_count and _close(
        axis.cut_times[index + 1],
        value,
    ):
        return bisect_right(
            axis.cut_times,
            axis.cut_times[index + 1],
        ) - 1
    return index


def derive_failure_trigger_windows(
    *,
    timeline: ResolvedProgressTimeline,
    occurrences: Sequence[StageOccurrenceAnnotationV1],
    sample_axis: EpisodeSampleAxis,
) -> tuple[FailureTriggerWindow, ...]:
    """Derive one ``[failed-stage-start, recorded-failure]`` window per fact."""

    if timeline.clock_domain_id != sample_axis.clock_domain_id:
        raise FailureTriggerResolutionError(
            "Progress timeline and sample axis use different clocks."
        )
    attempt_by_id = {
        attempt.attempt_id: attempt for attempt in timeline.attempts
    }
    windows: list[FailureTriggerWindow] = []
    by_failure_fact_id: dict[str, StageOccurrenceAnnotationV1] = {}
    for occurrence in occurrences:
        if occurrence.outcome != "FAILURE":
            continue
        source_failure_fact_id = occurrence.source_failure_fact_id
        if source_failure_fact_id is None:
            # Post-capture tail Failure is intentionally outside this first
            # capture-fact-only trigger version.
            continue
        if source_failure_fact_id in by_failure_fact_id:
            raise FailureTriggerResolutionError(
                "A captured Failure fact has more than one Failure occurrence."
            )
        by_failure_fact_id[source_failure_fact_id] = occurrence
        attempt = attempt_by_id.get(occurrence.coarse_attempt_id)
        if attempt is None:
            raise FailureTriggerResolutionError(
                "Failure occurrence references an unknown coarse Attempt."
            )
        if attempt.failure_fact_id != source_failure_fact_id:
            raise FailureTriggerResolutionError(
                "Failure occurrence is not bound to its Attempt Failure fact."
            )
        if attempt.failure_reported_at is None or attempt.end_at is None:
            raise FailureTriggerResolutionError(
                "Captured Failure Attempt has no immutable failure timestamp."
            )
        if not _close(occurrence.end_at, attempt.failure_reported_at):
            raise FailureTriggerResolutionError(
                "Failure occurrence must end at the original Failure time."
            )
        lower_index = _first_sample_at_or_after(
            sample_axis,
            occurrence.start_at,
            name="failed stage start",
        )
        upper_index = _last_sample_at_or_before(
            sample_axis,
            attempt.failure_reported_at,
            name="recorded Failure time",
        )
        if lower_index > upper_index:
            raise FailureTriggerResolutionError(
                "The failed stage starts after its recorded Failure sample."
            )
        windows.append(
            FailureTriggerWindow(
                source_failure_fact_id=source_failure_fact_id,
                coarse_attempt_id=occurrence.coarse_attempt_id,
                progress_step_id=occurrence.progress_step_id,
                stage_id=occurrence.stage_id,
                lower_bound_sample_index=lower_index,
                upper_bound_sample_index=upper_index,
                lower_bound_at=sample_axis.cut_times[lower_index],
                upper_bound_at=sample_axis.cut_times[upper_index],
            )
        )
    return tuple(
        sorted(
            windows,
            key=lambda item: (
                item.upper_bound_at,
                item.source_failure_fact_id,
            ),
        )
    )


def _raw_trigger_index(
    raw: Mapping[str, Any],
    *,
    axis: EpisodeSampleAxis,
) -> int | None:
    if "trigger_sample_index" in raw:
        value = raw["trigger_sample_index"]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise FailureTriggerResolutionError(
                "trigger_sample_index must be a non-negative integer."
            )
        return value
    if "trigger_at" not in raw or raw["trigger_at"] is None:
        return None
    try:
        return _exact_sample_index(
            axis,
            float(raw["trigger_at"]),
            name="trigger_at",
        )
    except (TypeError, ValueError) as exc:
        raise FailureTriggerResolutionError(
            "trigger_at must be a sample timestamp."
        ) from exc


def normalize_failure_trigger_drafts(
    *,
    timeline: ResolvedProgressTimeline,
    occurrences: Sequence[StageOccurrenceAnnotationV1],
    sample_axis: EpisodeSampleAxis,
    existing: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Fill legacy drafts with defaults and keep every trigger discrete."""

    windows = derive_failure_trigger_windows(
        timeline=timeline,
        occurrences=occurrences,
        sample_axis=sample_axis,
    )
    raw_by_failure_id: dict[str, Mapping[str, Any]] = {}
    for raw in tuple(existing or ()):
        if not isinstance(raw, Mapping):
            raise FailureTriggerResolutionError(
                "Every failure_trigger draft item must be an object."
            )
        source_failure_fact_id = str(
            raw.get("source_failure_fact_id", "")
        ).strip()
        if not source_failure_fact_id:
            raise FailureTriggerResolutionError(
                "Every failure_trigger must reference a Failure fact."
            )
        if source_failure_fact_id in raw_by_failure_id:
            raise FailureTriggerResolutionError(
                "A Failure fact has more than one failure_trigger draft item."
            )
        raw_by_failure_id[source_failure_fact_id] = raw

    normalized: list[dict[str, Any]] = []
    for window in windows:
        raw = raw_by_failure_id.get(window.source_failure_fact_id, {})
        requested_index = _raw_trigger_index(raw, axis=sample_axis)
        if requested_index is None:
            trigger_index = window.upper_bound_sample_index
            source = "CAPTURE_DEFAULT"
            confidence = 1.0
        else:
            if not (
                window.lower_bound_sample_index
                <= requested_index
                <= window.upper_bound_sample_index
            ):
                raise FailureTriggerResolutionError(
                    "failure_trigger lies outside the failed-stage window."
                )
            trigger_index = requested_index
            source = str(raw.get("source", "POST_CAPTURE_REFINED")).strip()
            confidence = raw.get("confidence", 1.0)
        normalized.append(
            {
                "annotation_id": str(
                    raw.get(
                        "annotation_id",
                        f"failure-trigger:{window.source_failure_fact_id}",
                    )
                ).strip()
                or f"failure-trigger:{window.source_failure_fact_id}",
                "source_failure_fact_id": window.source_failure_fact_id,
                "coarse_attempt_id": window.coarse_attempt_id,
                "progress_step_id": window.progress_step_id,
                "stage_id": window.stage_id,
                "lower_bound_sample_index": (
                    window.lower_bound_sample_index
                ),
                "upper_bound_sample_index": (
                    window.upper_bound_sample_index
                ),
                "trigger_sample_index": trigger_index,
                "lower_bound_at": window.lower_bound_at,
                "upper_bound_at": window.upper_bound_at,
                "trigger_at": sample_axis.cut_times[trigger_index],
                "clock_domain_id": sample_axis.clock_domain_id,
                "confidence": confidence,
                "source": source,
            }
        )
    # A privileged abnormal edit may intentionally remove the stage Failure
    # occurrence while retaining the immutable coarse Failure fact.  Until a
    # failed fine stage exists again there is no legal lower bound, so the
    # detached trigger is omitted rather than inventing one.
    return normalized


def build_failure_trigger_annotations(
    *,
    timeline: ResolvedProgressTimeline,
    occurrences: Sequence[StageOccurrenceAnnotationV1],
    sample_axis: EpisodeSampleAxis,
    drafts: Sequence[Mapping[str, Any]],
) -> tuple[FailureTriggerAnnotationV1, ...]:
    """Compile normalized draft values into checked immutable observations."""

    normalized = normalize_failure_trigger_drafts(
        timeline=timeline,
        occurrences=occurrences,
        sample_axis=sample_axis,
        existing=drafts,
    )
    try:
        result = tuple(
            FailureTriggerAnnotationV1(
                annotation_id=value["annotation_id"],
                source_failure_fact_id=value["source_failure_fact_id"],
                coarse_attempt_id=value["coarse_attempt_id"],
                progress_step_id=value["progress_step_id"],
                stage_id=value["stage_id"],
                lower_bound_sample_index=value[
                    "lower_bound_sample_index"
                ],
                upper_bound_sample_index=value[
                    "upper_bound_sample_index"
                ],
                trigger_sample_index=value["trigger_sample_index"],
                lower_bound_at=value["lower_bound_at"],
                upper_bound_at=value["upper_bound_at"],
                trigger_at=value["trigger_at"],
                clock_domain_id=value["clock_domain_id"],
                confidence=value["confidence"],
                source=value["source"],
            )
            for value in normalized
        )
    except (TypeError, ValueError, FailureTriggerContractError) as exc:
        raise FailureTriggerResolutionError(
            f"Invalid failure_trigger draft: {exc}"
        ) from exc
    for annotation in result:
        if (
            annotation.clock_domain_id != sample_axis.clock_domain_id
            or not _close(
                annotation.trigger_at,
                sample_axis.cut_times[annotation.trigger_sample_index],
            )
            or not _close(
                annotation.lower_bound_at,
                sample_axis.cut_times[annotation.lower_bound_sample_index],
            )
            or not _close(
                annotation.upper_bound_at,
                sample_axis.cut_times[annotation.upper_bound_sample_index],
            )
        ):
            raise FailureTriggerResolutionError(
                "Failure-trigger times do not match the bound sample axis."
            )
    return result


def derive_failure_trigger_analysis_view(
    annotations: Sequence[FailureTriggerAnnotationV1],
) -> tuple[FailureTriggerAnalysisItem, ...]:
    """Expose normal-before/after-trigger intervals for downstream analysis."""

    return tuple(
        FailureTriggerAnalysisItem(
            source_failure_fact_id=item.source_failure_fact_id,
            coarse_attempt_id=item.coarse_attempt_id,
            progress_step_id=item.progress_step_id,
            stage_id=item.stage_id,
            trigger_sample_index=item.trigger_sample_index,
            trigger_at=item.trigger_at,
            failure_at=item.upper_bound_at,
            normal_prefix_start_at=item.lower_bound_at,
            normal_prefix_end_at=item.trigger_at,
            failure_suffix_start_at=item.trigger_at,
            failure_suffix_end_at=item.upper_bound_at,
        )
        for item in annotations
    )


__all__ = [
    "FAILURE_TRIGGER_SAMPLE_AXIS_HASH_DOMAIN_V1",
    "FailureTriggerAnalysisItem",
    "FailureTriggerResolutionError",
    "FailureTriggerWindow",
    "build_failure_trigger_annotations",
    "derive_failure_trigger_analysis_view",
    "derive_failure_trigger_windows",
    "failure_trigger_sample_axis_hash",
    "normalize_failure_trigger_drafts",
]
