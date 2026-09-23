"""Pure stage-fragment projection for local abnormal-interval draft edits."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from dataset_contracts.progress_stage_annotations import AbnormalIntervalAnnotationV1


class AbnormalProjectionError(ValueError):
    """Raised when a stage fragment cannot be reconstructed without guessing."""


def coalesce_orphaned_fragments(
    occurrences: Sequence[Mapping[str, Any]],
    abnormal_intervals: Sequence[AbnormalIntervalAnnotationV1],
) -> list[dict[str, Any]]:
    projected = sorted(
        (dict(value) for value in occurrences),
        key=lambda value: (
            float(value["start_at"]),
            float(value["end_at"]),
            str(value.get("annotation_id", "")),
        ),
    )
    coalesced: list[dict[str, Any]] = []
    index = 0
    while index < len(projected):
        occurrence = projected[index]
        # A truncated prefix may start at an earlier retained abnormal cut;
        # only its end can justify keeping the truncation.
        touches_abnormal = any(
            math.isclose(
                float(occurrence["end_at"]),
                abnormal.start_at,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for abnormal in abnormal_intervals
        )
        if (
            str(occurrence.get("outcome", "")).upper() == "TRUNCATED_BY_ABNORMAL"
            and not touches_abnormal
        ):
            following = projected[index + 1] if index + 1 < len(projected) else None
            if not (
                following
                and following.get("coarse_attempt_id")
                == occurrence.get("coarse_attempt_id")
                and following.get("progress_step_id")
                == occurrence.get("progress_step_id")
                and following.get("stage_id") == occurrence.get("stage_id")
            ):
                raise AbnormalProjectionError(
                    "A TRUNCATED_BY_ABNORMAL stage no longer touches an "
                    "abnormal interval and has no same-stage suffix to "
                    "restore without guessing."
                )
            restored = dict(following)
            restored["start_at"] = occurrence["start_at"]
            projected[index + 1] = restored
            index += 1
            continue
        coalesced.append(occurrence)
        index += 1
    return coalesced


def renumber_occurrences(
    occurrences: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = [dict(value) for value in occurrences]
    seen_ids: set[str] = set()
    occurrence_index_by_attempt: dict[str, int] = {}
    for occurrence in result:
        annotation_id = str(occurrence.get("annotation_id", ""))
        if annotation_id in seen_ids:
            raise AbnormalProjectionError(
                "Abnormal-boundary projection produced a duplicate stage "
                f"annotation identity {annotation_id!r}."
            )
        seen_ids.add(annotation_id)
        attempt_id = str(occurrence.get("coarse_attempt_id", ""))
        next_index = occurrence_index_by_attempt.get(attempt_id, 0) + 1
        occurrence_index_by_attempt[attempt_id] = next_index
        occurrence["occurrence_index_in_attempt"] = next_index
    return result


def project_occurrences(
    occurrences: Iterable[Mapping[str, Any]],
    abnormal_intervals: Sequence[AbnormalIntervalAnnotationV1],
) -> list[dict[str, Any]]:
    """Cut stages at abnormal bounds without rewriting capture facts.

    A cut prefix becomes TRUNCATED_BY_ABNORMAL; its surviving suffix retains
    the original outcome and Failure binding. A fully covered success is never
    deleted implicitly. A fully covered Failure may be omitted only when its
    immutable Failure cut is itself covered. Removed cuts rejoin only an
    immediate suffix of the same stage and Attempt, without guessing.
    """

    projected: list[dict[str, Any]] = []
    for raw_occurrence in occurrences:
        occurrence = dict(raw_occurrence)
        try:
            original_start = float(occurrence["start_at"])
            original_end = float(occurrence["end_at"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AbnormalProjectionError("Stage occurrence has invalid bounds.") from exc
        if (
            not math.isfinite(original_start)
            or not math.isfinite(original_end)
            or original_end <= original_start
        ):
            raise AbnormalProjectionError("Stage occurrence has invalid bounds.")

        segments = [(original_start, original_end)]
        for abnormal in abnormal_intervals:
            next_segments: list[tuple[float, float]] = []
            for segment_start, segment_end in segments:
                if not (
                    segment_start < abnormal.end_at
                    and abnormal.start_at < segment_end
                ):
                    next_segments.append((segment_start, segment_end))
                    continue
                if segment_start < abnormal.start_at:
                    next_segments.append((segment_start, abnormal.start_at))
                if abnormal.end_at < segment_end:
                    next_segments.append((abnormal.end_at, segment_end))
            segments = next_segments

        if not segments:
            outcome = str(occurrence.get("outcome", "")).upper()
            failure_cut_is_abnormal = bool(
                outcome == "FAILURE"
                and occurrence.get("source_failure_fact_id")
                and any(
                    abnormal.start_at <= original_end <= abnormal.end_at
                    for abnormal in abnormal_intervals
                )
            )
            if failure_cut_is_abnormal:
                continue
            raise AbnormalProjectionError(
                "An abnormal interval fully covers a successful stage "
                f"occurrence {occurrence.get('stage_id', '<unknown>')!r}. "
                "Select that stage block and use the red delete button; "
                "the application will not delete it implicitly."
            )

        original_id = str(occurrence.get("annotation_id", "")).strip()
        for fragment_index, (segment_start, segment_end) in enumerate(segments, start=1):
            fragment = dict(occurrence)
            keeps_original_end = math.isclose(
                segment_end,
                original_end,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            fragment["start_at"] = segment_start
            fragment["end_at"] = segment_end
            if keeps_original_end:
                fragment["annotation_id"] = original_id
            else:
                fragment["annotation_id"] = (
                    f"{original_id}:abnormal-prefix:{fragment_index}"
                )
                fragment["outcome"] = "TRUNCATED_BY_ABNORMAL"
                fragment["source_failure_fact_id"] = None
            projected.append(fragment)

    projected = coalesce_orphaned_fragments(projected, abnormal_intervals)
    return renumber_occurrences(projected)
