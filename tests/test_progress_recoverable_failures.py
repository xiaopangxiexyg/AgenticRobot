from __future__ import annotations

from dataclasses import replace

import pytest

from collection_runtime.progress.recoverable_failure_store import (
    commit_recoverable_failure_annotation_set,
    load_recoverable_failure_annotation_set,
)
from dataset_contracts.progress_recoverable_failures import (
    RecoverableFailureAnnotationSetV1,
)
from dataset_contracts.progress_stage_annotations import (
    StageOccurrenceAnnotationV1,
)
from dataset_interface.progress_recoverable_failure_artifacts import (
    RecoverableFailureAnnotationReadError,
    load_optional_recoverable_failure_annotation_set,
    load_recoverable_failure_annotation_set as load_readonly_set,
)
from dataset_interface.progress_recoverable_failures import (
    RecoverableFailureResolutionError,
    build_recoverable_failure_annotations,
    normalize_recoverable_failure_drafts,
    recoverable_failure_sample_axis_hash,
)
from dataset_interface.sample_axis import EpisodeSampleAxis


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _axis() -> EpisodeSampleAxis:
    return EpisodeSampleAxis(
        clock_domain_id="episode_clock",
        cut_times=tuple(float(value) for value in range(11)),
    )


def _occurrence(
    *,
    annotation_id: str = "occurrence:a1",
    stage_id: str = "stage:a1",
    start_at: float = 0.0,
    end_at: float = 5.0,
    outcome: str = "SUCCESS",
) -> StageOccurrenceAnnotationV1:
    return StageOccurrenceAnnotationV1(
        annotation_id=annotation_id,
        coarse_attempt_id="attempt:1",
        progress_step_id="step:a",
        stage_id=stage_id,
        occurrence_index_in_attempt=1,
        start_at=start_at,
        end_at=end_at,
        clock_domain_id="episode_clock",
        outcome=outcome,
    )


def test_recoverable_failure_is_discrete_and_bound_to_one_success_stage() -> None:
    normalized = normalize_recoverable_failure_drafts(
        occurrences=(_occurrence(),),
        sample_axis=_axis(),
        existing=[
            {
                "annotation_id": "recoverable:1",
                "onset_at": 1.2,
                "recovered_at": 3.2,
                "note": "brief correction",
            }
        ],
    )

    assert normalized[0]["onset_sample_index"] == 1
    assert normalized[0]["recovered_sample_index"] == 3
    assert normalized[0]["stage_occurrence_annotation_id"] == "occurrence:a1"
    assert normalized[0]["stage_id"] == "stage:a1"
    annotation = build_recoverable_failure_annotations(
        occurrences=(_occurrence(),),
        sample_axis=_axis(),
        drafts=normalized,
    )[0]
    assert annotation.onset_at == 1.0
    assert annotation.recovered_at == 3.0
    assert annotation.note == "brief correction"


@pytest.mark.parametrize("timestamp", [-0.01, 10.01])
def test_recoverable_failure_rejects_time_outside_sample_axis(
    timestamp: float,
) -> None:
    with pytest.raises(
        RecoverableFailureResolutionError,
        match="outside the Episode sample axis",
    ):
        normalize_recoverable_failure_drafts(
            occurrences=(_occurrence(end_at=10.0),),
            sample_axis=_axis(),
            existing=[
                {
                    "onset_at": timestamp,
                    "recovered_at": 2.0 if timestamp < 0 else 10.0,
                }
            ],
        )


@pytest.mark.parametrize(
    "second_onset, message",
    [
        (2.0, "cannot overlap"),
        (3.0, "must not share one sample"),
    ],
)
def test_recoverable_failures_require_a_distinct_normal_sample_between_events(
    second_onset: float,
    message: str,
) -> None:
    with pytest.raises(RecoverableFailureResolutionError, match=message):
        normalize_recoverable_failure_drafts(
            occurrences=(_occurrence(end_at=10.0),),
            sample_axis=_axis(),
            existing=[
                {
                    "annotation_id": "recoverable:first",
                    "onset_at": 1.0,
                    "recovered_at": 3.0,
                },
                {
                    "annotation_id": "recoverable:second",
                    "onset_at": second_onset,
                    "recovered_at": 5.0,
                },
            ],
        )


def test_legacy_adjacent_recoverable_failures_remain_readable() -> None:
    normalized = normalize_recoverable_failure_drafts(
        occurrences=(_occurrence(end_at=10.0),),
        sample_axis=_axis(),
        existing=[
            {
                "annotation_id": "recoverable:first",
                "onset_at": 1.0,
                "recovered_at": 3.0,
            },
            {
                "annotation_id": "recoverable:second",
                "onset_at": 3.0,
                "recovered_at": 5.0,
            },
        ],
        allow_adjacent_legacy=True,
    )

    assert [value["annotation_id"] for value in normalized] == [
        "recoverable:first",
        "recoverable:second",
    ]


def test_recoverable_failure_cannot_cross_a_stage_boundary() -> None:
    occurrences = (
        _occurrence(end_at=5.0),
        _occurrence(
            annotation_id="occurrence:a2",
            stage_id="stage:a2",
            start_at=5.0,
            end_at=10.0,
        ),
    )
    with pytest.raises(
        RecoverableFailureResolutionError,
        match="both lie inside one successful stage",
    ):
        normalize_recoverable_failure_drafts(
            occurrences=occurrences,
            sample_axis=_axis(),
            existing=[{"onset_at": 4.0, "recovered_at": 6.0}],
        )


def test_recoverable_failure_on_shared_boundary_binds_to_right_stage() -> None:
    occurrences = (
        _occurrence(end_at=5.0),
        _occurrence(
            annotation_id="occurrence:a2",
            stage_id="stage:a2",
            start_at=5.0,
            end_at=10.0,
        ),
    )

    normalized = normalize_recoverable_failure_drafts(
        occurrences=occurrences,
        sample_axis=_axis(),
        existing=[{"onset_at": 5.0, "recovered_at": 7.0}],
    )

    assert normalized[0]["stage_occurrence_annotation_id"] == "occurrence:a2"
    assert normalized[0]["stage_id"] == "stage:a2"


def test_recoverable_failure_does_not_attach_to_failure_outcome() -> None:
    failure = replace(
        _occurrence(outcome="FAILURE"),
        source_failure_fact_id="fact:failure",
    )
    with pytest.raises(
        RecoverableFailureResolutionError,
        match="both lie inside one successful stage",
    ):
        normalize_recoverable_failure_drafts(
            occurrences=(failure,),
            sample_axis=_axis(),
            existing=[{"onset_at": 1.0, "recovered_at": 3.0}],
        )


def test_recoverable_failure_sidecar_is_optional_bound_and_immutable(
    tmp_path,
) -> None:
    annotation = build_recoverable_failure_annotations(
        occurrences=(_occurrence(),),
        sample_axis=_axis(),
        drafts=[
            {
                "annotation_id": "recoverable:1",
                "onset_at": 1.0,
                "recovered_at": 3.0,
            }
        ],
    )[0]
    value = RecoverableFailureAnnotationSetV1(
        annotation_set_id="stage-set:1",
        stage_annotation_set_id="stage-set:1",
        episode_capture_identity_hash="1" * 64,
        source_progress_manifest_hash="2" * 64,
        task_stage_profile_hash="3" * 64,
        stage_annotation_bundle_hash="4" * 64,
        sample_axis_hash=recoverable_failure_sample_axis_hash(_axis()),
        bound_clock_domain_id="episode_clock",
        annotator_id="fixture",
        created_at=10.0,
        created_clock_domain_id="wall_clock",
        annotations=(annotation,),
    )
    annotation_dir = tmp_path / "stage-set:1"

    report = commit_recoverable_failure_annotation_set(
        annotation_dir=annotation_dir,
        annotation_set=value,
    )
    assert not report.idempotent
    assert load_recoverable_failure_annotation_set(annotation_dir) == value
    with pytest.raises(
        RecoverableFailureAnnotationReadError,
        match="stage annotation set",
    ):
        load_readonly_set(annotation_dir)
    assert commit_recoverable_failure_annotation_set(
        annotation_dir=annotation_dir,
        annotation_set=value,
    ).idempotent
    with pytest.raises(
        RecoverableFailureAnnotationReadError,
        match="sealed stage annotation set",
    ):
        load_optional_recoverable_failure_annotation_set(
            tmp_path / "missing-stage-set"
        )
