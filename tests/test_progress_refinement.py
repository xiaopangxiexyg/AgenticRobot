from __future__ import annotations

from dataclasses import replace

import pytest

from dataset_contracts.progress_profile import (
    ProgressStepDefinitionV1,
    TaskProgressProfileV1,
)
from dataset_contracts.progress_refinement import (
    ProgressRefinementContractError,
    RefinementBoundaryDefinitionV1,
    TaskRefinementProfileV1,
    validate_task_refinement_profile,
)
from dataset_interface.progress_refinement import (
    ProgressRefinementResolutionError,
    RefinementBoundaryObservation,
    SelectedRefinementAnnotationSet,
    derive_refinement_failure_training_view,
)
from dataset_interface.progress_timeline import (
    ResolvedProgressAttempt,
    ResolvedProgressTimeline,
)
from dataset_interface.sample_axis import EpisodeSampleAxis, SampleIndexInterval


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _progress_profile() -> TaskProgressProfileV1:
    return TaskProgressProfileV1(
        profile_id="refinement-progress",
        profile_version="1",
        task_id="refinement",
        steps=(
            ProgressStepDefinitionV1("a", "MANIPULATE", "A", 1),
        ),
        initial_step_id="a",
        terminal_step_ids=("a",),
    )


def _refinement_profile() -> TaskRefinementProfileV1:
    progress = _progress_profile()
    return TaskRefinementProfileV1(
        profile_id="refinement-detail",
        profile_version="1",
        task_progress_profile_hash=progress.semantic_content_hash,
        boundaries=(
            RefinementBoundaryDefinitionV1(
                progress_step_id="a",
                refinement_boundary_id="enter_a2",
                ordinal=1,
                semantics="ENTER_STAGE",
            ),
            RefinementBoundaryDefinitionV1(
                progress_step_id="a",
                refinement_boundary_id="enter_a3",
                ordinal=2,
                semantics="ENTER_STAGE",
                parent_refinement_boundary_id="enter_a2",
                eligible_as_exclusion_anchor=True,
                certifies_valid_prefix=True,
                required_predecessor_boundary_ids=("enter_a2",),
            ),
        ),
    )


def _timeline() -> ResolvedProgressTimeline:
    return ResolvedProgressTimeline(
        episode_capture_identity_hash="1" * 64,
        task_progress_profile_hash=_progress_profile().semantic_content_hash,
        progress_episode_binding_hash="2" * 64,
        clock_domain_id="episode_clock",
        fact_bundle_hash="3" * 64,
        fact_count=3,
        sealed=True,
        phase="TERMINATED",
        attempts=(
            ResolvedProgressAttempt(
                attempt_id="attempt:a",
                progress_step_id="a",
                attempt_ordinal_in_capture=1,
                start_fact_id="start:a",
                start_at=0.0,
                clock_domain_id="episode_clock",
                end_fact_id="failure:a",
                end_at=5.0,
                outcome="FAILURE",
                failure_fact_id="failure:a",
                failure_reported_at=5.0,
                failure_timing_semantics="OPERATOR_REPORT_TIME",
            ),
        ),
        transitions=(),
        capture_pause_intervals=(),
        undone_advance_fact_ids=(),
        invalidated_attempt_ids=(),
        current_progress_step_id="a",
        active_attempt_id=None,
        terminal_fact_id="termination",
        termination_cause="SAFETY_STOP",
        full_task_outcome="FAILURE",
        source_progress_manifest_hash="4" * 64,
    )


def _axis() -> EpisodeSampleAxis:
    return EpisodeSampleAxis(
        clock_domain_id="episode_clock",
        cut_times=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    )


def _observation(
    annotation_id: str,
    boundary_id: str,
    at: float,
    *,
    confidence: float = 0.9,
) -> RefinementBoundaryObservation:
    return RefinementBoundaryObservation(
        annotation_id=annotation_id,
        coarse_attempt_id="attempt:a",
        refinement_boundary_id=boundary_id,
        effective_at=at,
        clock_domain_id="episode_clock",
        confidence=confidence,
    )


def _annotation_set(
    *observations: RefinementBoundaryObservation,
    sealed: bool = True,
    source_progress_manifest_hash: str = "4" * 64,
) -> SelectedRefinementAnnotationSet:
    return SelectedRefinementAnnotationSet(
        annotation_set_id="annotation-set:1",
        source_annotation_manifest_hash="5" * 64,
        source_progress_manifest_hash=source_progress_manifest_hash,
        task_refinement_profile_hash=(
            _refinement_profile().semantic_content_hash
        ),
        sealed=sealed,
        observations=tuple(observations),
    )


def test_refinement_profile_roundtrip_and_normalizes_dependencies() -> None:
    profile = _refinement_profile()
    reversed_profile = replace(
        profile,
        boundaries=tuple(reversed(profile.boundaries)),
    )

    assert reversed_profile.semantic_content_hash == profile.semantic_content_hash
    assert (
        TaskRefinementProfileV1.from_payload(profile.to_payload())
        == profile
    )
    validate_task_refinement_profile(
        progress_profile=_progress_profile(),
        refinement_profile=profile,
    )


def test_refinement_profile_rejects_unsafe_or_unknown_boundaries() -> None:
    progress = _progress_profile()
    with pytest.raises(
        ProgressRefinementContractError,
        match="must certify",
    ):
        RefinementBoundaryDefinitionV1(
            progress_step_id="a",
            refinement_boundary_id="unsafe",
            ordinal=1,
            semantics="ENTER_STAGE",
            eligible_as_exclusion_anchor=True,
        )
    with pytest.raises(
        ProgressRefinementContractError,
        match="must be boolean",
    ):
        RefinementBoundaryDefinitionV1(
            progress_step_id="a",
            refinement_boundary_id="bad-bool",
            ordinal=1,
            semantics="ENTER_STAGE",
            eligible_as_exclusion_anchor="false",  # type: ignore[arg-type]
        )

    unknown_step = TaskRefinementProfileV1(
        profile_id="unknown-step",
        profile_version="1",
        task_progress_profile_hash=progress.semantic_content_hash,
        boundaries=(
            RefinementBoundaryDefinitionV1(
                progress_step_id="missing",
                refinement_boundary_id="boundary",
                ordinal=1,
                semantics="ENTER_STAGE",
            ),
        ),
    )
    with pytest.raises(
        ProgressRefinementContractError,
        match="unknown steps",
    ):
        validate_task_refinement_profile(
            progress_profile=progress,
            refinement_profile=unknown_step,
        )


def test_validated_anchor_shrinks_only_the_failed_suffix() -> None:
    view = derive_refinement_failure_training_view(
        timeline=_timeline(),
        sample_axis=_axis(),
        progress_profile=_progress_profile(),
        refinement_profile=_refinement_profile(),
        annotation_set=_annotation_set(
            _observation("annotation:a2", "enter_a2", 2.0),
            _observation("annotation:a3", "enter_a3", 3.0),
        ),
        minimum_confidence=0.8,
    )

    assert len(view.exclusions) == 1
    exclusion = view.exclusions[0]
    assert exclusion.exclusion_start_basis == "VALIDATED_SALVAGE_ANCHOR"
    assert exclusion.salvage_status == "VALIDATED"
    assert exclusion.exclusion_start_at == 3.0
    assert exclusion.failure_cut_at == 5.0
    assert exclusion.sample_interval == SampleIndexInterval(3, 6)
    assert exclusion.salvage_anchor is not None
    assert exclusion.salvage_anchor.supporting_boundary_ids == (
        "enter_a2",
        "enter_a3",
    )
    assert view.source_annotation_manifest_hash == "5" * 64


def test_latest_fully_validated_anchor_is_selected() -> None:
    base = _refinement_profile()
    a2, a3 = base.boundaries
    profile = replace(
        base,
        boundaries=(
            replace(
                a2,
                eligible_as_exclusion_anchor=True,
                certifies_valid_prefix=True,
            ),
            a3,
        ),
    )
    annotation_set = SelectedRefinementAnnotationSet(
        annotation_set_id="annotation-set:latest",
        source_annotation_manifest_hash="5" * 64,
        source_progress_manifest_hash="4" * 64,
        task_refinement_profile_hash=profile.semantic_content_hash,
        sealed=True,
        observations=(
            _observation("annotation:a2", "enter_a2", 2.0),
            _observation("annotation:a3", "enter_a3", 3.0),
        ),
    )
    view = derive_refinement_failure_training_view(
        timeline=_timeline(),
        sample_axis=_axis(),
        progress_profile=_progress_profile(),
        refinement_profile=profile,
        annotation_set=annotation_set,
        minimum_confidence=0.8,
    )

    anchor = view.exclusions[0].salvage_anchor
    assert anchor is not None
    assert anchor.refinement_boundary_id == "enter_a3"
    assert view.exclusions[0].sample_interval == SampleIndexInterval(3, 6)


@pytest.mark.parametrize(
    ("observations", "minimum_confidence", "expected_status"),
    (
        ((), 0.8, "UNANNOTATED"),
        (
            (_observation("annotation:a3", "enter_a3", 3.0),),
            0.8,
            "INCOMPLETE",
        ),
        (
            (
                _observation(
                    "annotation:a2",
                    "enter_a2",
                    2.0,
                    confidence=0.7,
                ),
                _observation("annotation:a3", "enter_a3", 3.0),
            ),
            0.8,
            "BELOW_CONFIDENCE",
        ),
        (
            (
                _observation("annotation:a2", "enter_a2", 2.0),
                _observation("annotation:a3:1", "enter_a3", 3.0),
                _observation("annotation:a3:2", "enter_a3", 3.1),
            ),
            0.8,
            "CONFLICTED",
        ),
        (
            (
                _observation("annotation:a2", "enter_a2", 4.0),
                _observation("annotation:a3", "enter_a3", 3.0),
            ),
            0.8,
            "INVALID",
        ),
        (
            (
                _observation("annotation:a2", "enter_a2", 2.0),
                _observation("annotation:a3", "enter_a3", 5.0),
            ),
            0.8,
            "NO_VALID_ANCHOR",
        ),
    ),
)
def test_unvalidated_refinement_never_releases_coarse_failure_data(
    observations: tuple[RefinementBoundaryObservation, ...],
    minimum_confidence: float,
    expected_status: str,
) -> None:
    view = derive_refinement_failure_training_view(
        timeline=_timeline(),
        sample_axis=_axis(),
        progress_profile=_progress_profile(),
        refinement_profile=_refinement_profile(),
        annotation_set=_annotation_set(*observations),
        minimum_confidence=minimum_confidence,
    )

    exclusion = view.exclusions[0]
    assert exclusion.salvage_status == expected_status
    assert exclusion.exclusion_start_basis == "ATTEMPT_START"
    assert exclusion.exclusion_start_at == 0.0
    assert exclusion.sample_interval == SampleIndexInterval(0, 6)
    assert exclusion.salvage_anchor is None


def test_refinement_requires_sealed_set_and_exact_manifest_binding() -> None:
    kwargs = {
        "timeline": _timeline(),
        "sample_axis": _axis(),
        "progress_profile": _progress_profile(),
        "refinement_profile": _refinement_profile(),
    }
    with pytest.raises(
        ProgressRefinementResolutionError,
        match="sealed annotation set",
    ):
        derive_refinement_failure_training_view(
            **kwargs,
            annotation_set=_annotation_set(sealed=False),
        )
    with pytest.raises(
        ProgressRefinementResolutionError,
        match="not bound",
    ):
        derive_refinement_failure_training_view(
            **kwargs,
            annotation_set=_annotation_set(
                source_progress_manifest_hash="9" * 64,
            ),
        )


def test_refinement_rejects_unknown_attempt_or_foreign_clock() -> None:
    unknown = replace(
        _observation("annotation:a2", "enter_a2", 2.0),
        coarse_attempt_id="attempt:missing",
    )
    with pytest.raises(
        ProgressRefinementResolutionError,
        match="unknown Attempt",
    ):
        derive_refinement_failure_training_view(
            timeline=_timeline(),
            sample_axis=_axis(),
            progress_profile=_progress_profile(),
            refinement_profile=_refinement_profile(),
            annotation_set=_annotation_set(unknown),
        )

    foreign_clock = replace(
        _observation("annotation:a2", "enter_a2", 2.0),
        clock_domain_id="foreign_clock",
    )
    with pytest.raises(
        ProgressRefinementResolutionError,
        match="Episode clock",
    ):
        derive_refinement_failure_training_view(
            timeline=_timeline(),
            sample_axis=_axis(),
            progress_profile=_progress_profile(),
            refinement_profile=_refinement_profile(),
            annotation_set=_annotation_set(foreign_clock),
        )
