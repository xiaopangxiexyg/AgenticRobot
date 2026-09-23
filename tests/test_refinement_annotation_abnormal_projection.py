from __future__ import annotations

from types import SimpleNamespace

import pytest

from control_station.refinement_annotation.abnormal_projection import (
    AbnormalProjectionError,
    coalesce_orphaned_fragments,
    project_occurrences,
    renumber_occurrences,
)
from control_station.refinement_annotation.application import (
    RefinementAnnotationApplication,
    RefinementAnnotationApplicationError,
)
from dataset_contracts.progress_stage_annotations import AbnormalIntervalAnnotationV1

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _occurrence(**changes: object) -> dict[str, object]:
    result: dict[str, object] = {
        "annotation_id": "stage-1",
        "coarse_attempt_id": "attempt-1",
        "progress_step_id": "step-1",
        "stage_id": "stage-1",
        "occurrence_index_in_attempt": 7,
        "start_at": 0.0,
        "end_at": 10.0,
        "outcome": "SUCCESS",
        "source_failure_fact_id": None,
    }
    result.update(changes)
    return result


def _abnormal(annotation_id: str, start_at: float, end_at: float):
    return AbnormalIntervalAnnotationV1(
        annotation_id=annotation_id,
        start_at=start_at,
        end_at=end_at,
        clock_domain_id="episode-clock",
        reason_code="BAD_CAPTURE",
    )


def test_projection_splits_without_mutating_draft_or_failure_binding() -> None:
    source = _occurrence()
    projected = project_occurrences([source], [_abnormal("cut", 4.0, 6.0)])

    assert source["occurrence_index_in_attempt"] == 7
    assert source["start_at"] == 0.0 and source["end_at"] == 10.0
    assert [value["annotation_id"] for value in projected] == [
        "stage-1:abnormal-prefix:1",
        "stage-1",
    ]
    assert [value["occurrence_index_in_attempt"] for value in projected] == [1, 2]
    assert [value["outcome"] for value in projected] == [
        "TRUNCATED_BY_ABNORMAL",
        "SUCCESS",
    ]
    assert [(value["start_at"], value["end_at"]) for value in projected] == [
        (0.0, 4.0),
        (6.0, 10.0),
    ]
    assert projected[0]["source_failure_fact_id"] is None

    failure = _occurrence(outcome="FAILURE", source_failure_fact_id="failure-1")
    failure_fragments = project_occurrences([failure], [_abnormal("cut", 4.0, 6.0)])
    assert [value["outcome"] for value in failure_fragments] == [
        "TRUNCATED_BY_ABNORMAL",
        "FAILURE",
    ]
    assert [value["source_failure_fact_id"] for value in failure_fragments] == [
        None,
        "failure-1",
    ]
    assert failure["source_failure_fact_id"] == "failure-1"


def test_projection_only_rejoins_a_removed_cut_and_never_guesses_suffix() -> None:
    first = _abnormal("first", 2.0, 3.0)
    second = _abnormal("second", 6.0, 7.0)
    fragments = project_occurrences([_occurrence()], [first, second])

    restored = renumber_occurrences(coalesce_orphaned_fragments(fragments, [second]))
    assert [(value["start_at"], value["end_at"]) for value in restored] == [
        (0.0, 6.0),
        (7.0, 10.0),
    ]
    assert [value["outcome"] for value in restored] == [
        "TRUNCATED_BY_ABNORMAL",
        "SUCCESS",
    ]
    assert [value["occurrence_index_in_attempt"] for value in restored] == [1, 2]

    with pytest.raises(AbnormalProjectionError, match="no same-stage suffix"):
        coalesce_orphaned_fragments(fragments[:1], ())
    with pytest.raises(AbnormalProjectionError, match="duplicate stage annotation identity"):
        renumber_occurrences([_occurrence(), _occurrence()])


def test_projection_keeps_full_cover_fail_closed_and_application_error_type() -> None:
    covering = _abnormal("whole-stage", 0.0, 10.0)
    with pytest.raises(AbnormalProjectionError, match="fully covers a successful stage"):
        project_occurrences([_occurrence()], [covering])

    failure = _occurrence(outcome="FAILURE", source_failure_fact_id="failure-1")
    assert project_occurrences([failure], [covering]) == []
    assert failure["source_failure_fact_id"] == "failure-1"

    application = object.__new__(RefinementAnnotationApplication)
    item = SimpleNamespace(
        progress=SimpleNamespace(timeline=SimpleNamespace(clock_domain_id="episode-clock"))
    )
    with pytest.raises(
        RefinementAnnotationApplicationError,
        match="fully covers a successful stage",
    ) as covered:
        application._project_occurrences_around_abnormal_intervals(
            item,
            {"occurrences": [_occurrence()], "abnormal_intervals": [covering.to_payload()]},
        )
    assert covered.value.__cause__ is None

    with pytest.raises(AbnormalProjectionError, match="invalid bounds"):
        project_occurrences([_occurrence(start_at="not-a-timestamp")], ())
    with pytest.raises(RefinementAnnotationApplicationError, match="invalid bounds") as bad_bounds:
        application._project_occurrences_around_abnormal_intervals(
            item,
            {"occurrences": [_occurrence(start_at="not-a-timestamp")]},
        )
    assert isinstance(bad_bounds.value.__cause__, ValueError)
