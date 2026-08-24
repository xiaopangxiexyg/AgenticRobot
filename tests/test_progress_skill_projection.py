from __future__ import annotations

from dataclasses import replace

import pytest

from collection_runtime.progress.profiles import load_managed_task_profile
from dataset_contracts import (
    ProgressSkillProjectionV1,
    SkillProjectionReceiptV1,
    SkillTrainingContractV1,
)
from dataset_interface import (
    EpisodeTrainingSemanticIndex,
    ProgressSkillProjectionError,
    SampleIndexInterval,
    SemanticSpan,
    load_builtin_writing_skill_projection_catalog,
    project_episode_skill_view,
    validate_progress_skill_projection,
)


_HASH_A = "a" * 64
_HASH_B = "b" * 64


def _writing_bundle(stage_count: int):
    word = {3: "three", 4: "four", 5: "five"}[stage_count]
    return load_managed_task_profile(
        f"dmtac_rm65_bimanual_position_writing_{word}_stage_progress"
    )


def _catalog_binding(skill_id: str, projection_id: str):
    catalog = load_builtin_writing_skill_projection_catalog()
    skill = catalog.skill_by_id[skill_id]
    projection = next(
        item for item in catalog.projections if item.projection_id == projection_id
    )
    return skill, projection


def test_writing_catalog_validates_all_current_managed_profiles() -> None:
    catalog = load_builtin_writing_skill_projection_catalog()
    bundles = tuple(_writing_bundle(count) for count in (3, 4, 5))
    progress_by_hash = {
        item.progress_profile.semantic_content_hash: item.progress_profile
        for item in bundles
    }
    stage_by_hash = {
        item.stage_profile.semantic_content_hash: item.stage_profile
        for item in bundles
        if item.stage_profile is not None
    }
    skill_by_hash = {item.semantic_content_hash: item for item in catalog.skills}
    current = tuple(
        projection
        for projection in catalog.projections
        if projection.source_progress_profile_hash in progress_by_hash
    )
    assert len(current) == 12
    for projection in current:
        validate_progress_skill_projection(
            progress_profile=progress_by_hash[projection.source_progress_profile_hash],
            stage_profile=stage_by_hash[projection.source_stage_profile_hash],
            skill_contract=skill_by_hash[projection.skill_training_contract_hash],
            projection=projection,
        )


def test_writing_acquire_context_keeps_distinct_terminal_skills() -> None:
    catalog = load_builtin_writing_skill_projection_catalog()
    ready = catalog.skill_by_id["writing.acquire_brush_to_writing_ready"]
    presented = catalog.skill_by_id["writing.acquire_brush_and_present_for_cap_removal"]
    assert ready.semantic_content_hash != presented.semantic_content_hash
    assert {
        item.projection_id for item in catalog.projections_for_skill(ready.skill_id)
    } == {"writing3.acquire_to_writing_ready"}
    assert {
        item.projection_id for item in catalog.projections_for_skill(presented.skill_id)
    } == {
        "writing4.acquire_and_present",
        "writing5.acquire_and_present",
        "writing4_legacy.acquire_and_present",
        "writing5_legacy.acquire_and_present",
    }


def test_projection_fails_closed_when_required_predicate_is_unproven() -> None:
    bundle = _writing_bundle(3)
    skill, projection = _catalog_binding(
        "writing.write_with_brush",
        "writing3.write",
    )
    incomplete = replace(
        projection,
        asserted_entry_predicate_ids=("brush.held_by_right_hand",),
    )
    with pytest.raises(ProgressSkillProjectionError, match="does not prove"):
        validate_progress_skill_projection(
            progress_profile=bundle.progress_profile,
            stage_profile=bundle.stage_profile,
            skill_contract=skill,
            projection=incomplete,
        )


def _semantic_index() -> EpisodeTrainingSemanticIndex:
    return EpisodeTrainingSemanticIndex(
        episode_capture_identity_hash=_HASH_A,
        source_progress_manifest_hash=_HASH_B,
        clock_domain_id="episode_clock",
        sample_count=22,
        spans=(
            SemanticSpan(
                semantic_kind="STAGE_OCCURRENCE",
                sample_interval=SampleIndexInterval(0, 5),
                progress_step_id="right_acquire_brush",
                attempt_id="attempt-a",
                stage_id="right_acquire_brush.approach",
                outcome="SUCCESS",
                source_id="occurrence-approach-a",
            ),
            SemanticSpan(
                semantic_kind="STAGE_OCCURRENCE",
                sample_interval=SampleIndexInterval(5, 10),
                progress_step_id="right_acquire_brush",
                attempt_id="attempt-a",
                stage_id="right_acquire_brush.grasp",
                outcome="FAILURE",
                source_id="occurrence-grasp-a",
                source_failure_fact_id="failure-a",
            ),
            SemanticSpan(
                semantic_kind="FAILURE_REGION",
                sample_interval=SampleIndexInterval(7, 10),
                progress_step_id="right_acquire_brush",
                attempt_id="attempt-a",
                stage_id="right_acquire_brush.grasp",
                outcome="FAILURE",
                source_id="failure-trigger-a",
                source_failure_fact_id="failure-a",
            ),
            SemanticSpan(
                semantic_kind="STAGE_OCCURRENCE",
                sample_interval=SampleIndexInterval(12, 17),
                progress_step_id="right_acquire_brush",
                attempt_id="attempt-b",
                stage_id="right_acquire_brush.grasp",
                outcome="SUCCESS",
                source_id="occurrence-grasp-b",
            ),
            SemanticSpan(
                semantic_kind="STAGE_OCCURRENCE",
                sample_interval=SampleIndexInterval(17, 22),
                progress_step_id="right_acquire_brush",
                attempt_id="attempt-b",
                stage_id="right_acquire_brush.lift_ready",
                outcome="SUCCESS",
                source_id="occurrence-lift-b",
            ),
        ),
        anchors=(),
        nominal_intervals=(
            SampleIndexInterval(0, 7),
            SampleIndexInterval(12, 22),
        ),
        failure_intervals=(SampleIndexInterval(7, 10),),
    )


def test_projection_preserves_failure_pause_and_attempt_breaks() -> None:
    bundle = _writing_bundle(3)
    skill, projection = _catalog_binding(
        "writing.acquire_brush_grasp_prefix",
        "writing3.acquire_grasp_prefix",
    )
    result = project_episode_skill_view(
        progress_profile=bundle.progress_profile,
        stage_profile=bundle.stage_profile,
        skill_contract=skill,
        projection=projection,
        semantic_index=_semantic_index(),
    )
    intervals = result.receipt.intervals
    assert [
        (
            item.sample_begin,
            item.sample_end,
            item.training_domain,
            item.source_attempt_id,
        )
        for item in intervals
    ] == [
        (0, 7, "NOMINAL", "attempt-a"),
        (7, 10, "FAILURE", "attempt-a"),
        (12, 17, "NOMINAL", "attempt-b"),
    ]
    # Domain changes, the omitted Hold interval, and the omitted suffix stage
    # are all explicit cuts on the source sample axis.
    assert result.receipt.continuity_break_indices == (7, 10, 12, 17)
    assert intervals[1].source_failure_fact_id == "failure-a"


def test_projection_never_remerges_an_explicit_collection_continuity_break() -> None:
    bundle = _writing_bundle(3)
    skill, projection = _catalog_binding(
        "writing.acquire_brush_grasp_prefix",
        "writing3.acquire_grasp_prefix",
    )
    source = _semantic_index()
    split = replace(
        source,
        nominal_intervals=(
            SampleIndexInterval(0, 3),
            SampleIndexInterval(3, 7),
            SampleIndexInterval(12, 22),
        ),
        continuity_break_indices=(3,),
    )
    result = project_episode_skill_view(
        progress_profile=bundle.progress_profile,
        stage_profile=bundle.stage_profile,
        skill_contract=skill,
        projection=projection,
        semantic_index=split,
    )

    assert [
        (item.sample_begin, item.sample_end, item.training_domain)
        for item in result.receipt.intervals
    ] == [
        (0, 3, "NOMINAL"),
        (3, 7, "NOMINAL"),
        (7, 10, "FAILURE"),
        (12, 17, "NOMINAL"),
    ]
    assert result.receipt.continuity_break_indices == (3, 7, 10, 12, 17)


def test_projection_receipt_round_trip_is_deterministic() -> None:
    bundle = _writing_bundle(3)
    skill, projection = _catalog_binding(
        "writing.acquire_brush_to_writing_ready",
        "writing3.acquire_to_writing_ready",
    )
    first = project_episode_skill_view(
        progress_profile=bundle.progress_profile,
        stage_profile=bundle.stage_profile,
        skill_contract=skill,
        projection=projection,
        semantic_index=_semantic_index(),
    ).receipt
    second = SkillProjectionReceiptV1.from_payload(first.to_payload())
    assert second == first
    assert second.semantic_content_hash == first.semantic_content_hash
    assert second.intervals[-1].sample_end == 22


def test_skill_and_projection_contract_hashes_round_trip() -> None:
    skill, projection = _catalog_binding(
        "writing.return_brush",
        "writing3.return",
    )
    assert SkillTrainingContractV1.from_payload(skill.to_payload()) == skill
    assert ProgressSkillProjectionV1.from_payload(projection.to_payload()) == projection
