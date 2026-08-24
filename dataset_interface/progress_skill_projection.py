"""Validate and resolve non-destructive Progress-to-Skill projections.

The functions in this module only build derived interval views.  They never
rewrite Progress facts, annotations, sample payloads, Parquet, or video.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.progress_profile import TaskProgressProfileV1
from dataset_contracts.progress_skill_projection import (
    ProgressSkillProjectionV1,
    SkillProjectionIntervalV1,
    SkillProjectionReceiptV1,
    SkillTrainingContractV1,
)
from dataset_contracts.progress_stages import TaskStageProfileV1

from .sample_axis import SampleIndexInterval
from .training_semantics import (
    SEMANTIC_SPAN_STAGE,
    TRAINING_DOMAIN_FAILURE,
    TRAINING_DOMAIN_NOMINAL,
    EpisodeTrainingSemanticIndex,
    SemanticSpan,
)


_RECEIPT_ID_DOMAIN = "dataset_interface.skill_projection.receipt_id.v1"


class ProgressSkillProjectionError(ValueError):
    """Raised when a projection cannot be proven from its source artifacts."""


@dataclass(frozen=True)
class ProjectedEpisodeSkillView:
    """One target Skill and its immutable source provenance receipt."""

    skill_contract: SkillTrainingContractV1
    projection: ProgressSkillProjectionV1
    receipt: SkillProjectionReceiptV1


@dataclass(frozen=True)
class SkillProjectionCatalog:
    """Compiled authoring catalog; canonical identity stays in its contracts."""

    skills: tuple[SkillTrainingContractV1, ...]
    projections: tuple[ProgressSkillProjectionV1, ...]

    def __post_init__(self) -> None:
        skill_ids = tuple(item.skill_id for item in self.skills)
        projection_ids = tuple(item.projection_id for item in self.projections)
        if len(skill_ids) != len(set(skill_ids)):
            raise ProgressSkillProjectionError(
                "Skill projection catalog contains duplicate skill_id values."
            )
        if len(projection_ids) != len(set(projection_ids)):
            raise ProgressSkillProjectionError(
                "Skill projection catalog contains duplicate projection_id values."
            )
        known_hashes = {item.semantic_content_hash for item in self.skills}
        unknown = {
            item.skill_training_contract_hash
            for item in self.projections
            if item.skill_training_contract_hash not in known_hashes
        }
        if unknown:
            raise ProgressSkillProjectionError(
                "Projection catalog references an unknown Skill hash."
            )

    @property
    def skill_by_id(self) -> dict[str, SkillTrainingContractV1]:
        return {item.skill_id: item for item in self.skills}

    def projections_for_skill(
        self,
        skill_id: str,
    ) -> tuple[ProgressSkillProjectionV1, ...]:
        skill = self.skill_by_id.get(str(skill_id).strip())
        if skill is None:
            return ()
        return tuple(
            item
            for item in self.projections
            if item.skill_training_contract_hash == skill.semantic_content_hash
        )


def load_skill_projection_catalog(
    path: str | Path,
) -> SkillProjectionCatalog:
    """Compile a small editable catalog into strict canonical contracts."""

    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProgressSkillProjectionError(
            f"Could not read Skill projection catalog {source}: {exc}"
        ) from exc
    if payload.get("schema") != "dexteleop.skill_projection_catalog.authoring.v1":
        raise ProgressSkillProjectionError(
            "Unsupported Skill projection catalog authoring schema."
        )
    skills = tuple(
        SkillTrainingContractV1(
            skill_id=item.get("skill_id", ""),
            skill_version=item.get("skill_version", ""),
            semantic_type=item.get("semantic_type", ""),
            controlled_component_ids=tuple(item.get("controlled_component_ids", ())),
            entry_predicate_ids=tuple(item.get("entry_predicate_ids", ())),
            success_predicate_ids=tuple(item.get("success_predicate_ids", ())),
            conditionable_context_ids=tuple(item.get("conditionable_context_ids", ())),
        )
        for item in payload.get("skills", ())
    )
    by_id = {item.skill_id: item for item in skills}
    projections: list[ProgressSkillProjectionV1] = []
    for item in payload.get("projections", ()):
        target = by_id.get(str(item.get("skill_id", "")).strip())
        if target is None:
            raise ProgressSkillProjectionError(
                "Projection authoring entry references an unknown skill_id."
            )
        projections.append(
            ProgressSkillProjectionV1(
                projection_id=item.get("projection_id", ""),
                projection_version=item.get("projection_version", ""),
                source_progress_profile_hash=item.get(
                    "source_progress_profile_hash", ""
                ),
                source_stage_profile_hash=item.get("source_stage_profile_hash", ""),
                source_progress_step_id=item.get("source_progress_step_id", ""),
                source_stage_ids=tuple(item.get("source_stage_ids", ())),
                projection_kind=item.get("projection_kind", ""),
                skill_training_contract_hash=target.semantic_content_hash,
                projected_component_ids=tuple(item.get("projected_component_ids", ())),
                asserted_entry_predicate_ids=tuple(
                    item.get("asserted_entry_predicate_ids", ())
                ),
                asserted_success_predicate_ids=tuple(
                    item.get("asserted_success_predicate_ids", ())
                ),
                conditioned_context_ids=tuple(item.get("conditioned_context_ids", ())),
            )
        )
    return SkillProjectionCatalog(
        skills=skills,
        projections=tuple(projections),
    )


def load_builtin_writing_skill_projection_catalog() -> SkillProjectionCatalog:
    """Load the reviewed 3/4/5-step writing cross-profile mappings."""

    return load_skill_projection_catalog(
        Path(__file__).with_name("skill_projection_assets") / "writing_v1.json"
    )


def load_builtin_writing_stage1_skill_projection_catalog() -> SkillProjectionCatalog:
    """Load the dual-arm first-subtask mappings shared by 4/5-stage writing."""

    return load_skill_projection_catalog(
        Path(__file__).with_name("skill_projection_assets") / "writing_stage1_v2.json"
    )


def load_builtin_writing_four_stage_subtasks_skill_projection_catalog() -> (
    SkillProjectionCatalog
):
    """Load reviewed stage2/3/4 mappings for four-step writing data."""

    return load_skill_projection_catalog(
        Path(__file__).with_name("skill_projection_assets")
        / "writing_four_stage_subtasks_v2.json"
    )


def validate_progress_skill_projection(
    *,
    progress_profile: TaskProgressProfileV1,
    stage_profile: TaskStageProfileV1,
    skill_contract: SkillTrainingContractV1,
    projection: ProgressSkillProjectionV1,
) -> None:
    """Fail closed unless the explicit projection is complete and unambiguous."""

    if not isinstance(progress_profile, TaskProgressProfileV1):
        raise TypeError("progress_profile must be TaskProgressProfileV1.")
    if not isinstance(stage_profile, TaskStageProfileV1):
        raise TypeError("stage_profile must be TaskStageProfileV1.")
    if not isinstance(skill_contract, SkillTrainingContractV1):
        raise TypeError("skill_contract must be SkillTrainingContractV1.")
    if not isinstance(projection, ProgressSkillProjectionV1):
        raise TypeError("projection must be ProgressSkillProjectionV1.")
    if (
        projection.source_progress_profile_hash
        != progress_profile.semantic_content_hash
        or stage_profile.task_progress_profile_hash
        != progress_profile.semantic_content_hash
    ):
        raise ProgressSkillProjectionError(
            "Projection, stage profile, and Progress profile hashes differ."
        )
    if projection.source_stage_profile_hash != stage_profile.semantic_content_hash:
        raise ProgressSkillProjectionError(
            "Projection references a different stage profile."
        )
    if projection.skill_training_contract_hash != skill_contract.semantic_content_hash:
        raise ProgressSkillProjectionError(
            "Projection references a different Skill contract."
        )
    step = progress_profile.step_by_id.get(projection.source_progress_step_id)
    if step is None:
        raise ProgressSkillProjectionError(
            "Projection references an unknown Progress Step."
        )
    available = stage_profile.stages_for_step(step.progress_step_id)
    available_ids = tuple(item.stage_id for item in available)
    selected_ids = projection.source_stage_ids
    if any(stage_id not in available_ids for stage_id in selected_ids):
        raise ProgressSkillProjectionError(
            "Projection references an unknown or foreign execution stage."
        )
    selected_positions = tuple(
        available_ids.index(stage_id) for stage_id in selected_ids
    )
    if selected_positions != tuple(
        range(selected_positions[0], selected_positions[0] + len(selected_ids))
    ):
        raise ProgressSkillProjectionError(
            "Projected execution stages must be one ordered contiguous subset."
        )
    full_step = selected_ids == available_ids
    if projection.projection_kind in {"EXACT_STEP", "CONDITIONED_STEP"}:
        if not full_step:
            raise ProgressSkillProjectionError(
                f"{projection.projection_kind} must cover every source stage."
            )
    elif full_step:
        raise ProgressSkillProjectionError(
            "STAGE_SUBSET must be a proper subset; use EXACT_STEP or "
            "CONDITIONED_STEP for a complete Progress Step."
        )
    if projection.projected_component_ids != skill_contract.controlled_component_ids:
        raise ProgressSkillProjectionError(
            "Projected components must exactly match the Skill output contract."
        )
    missing_entry = set(skill_contract.entry_predicate_ids) - set(
        projection.asserted_entry_predicate_ids
    )
    missing_success = set(skill_contract.success_predicate_ids) - set(
        projection.asserted_success_predicate_ids
    )
    if missing_entry or missing_success:
        raise ProgressSkillProjectionError(
            "Projection does not prove every required Skill predicate: "
            f"missing_entry={sorted(missing_entry)} "
            f"missing_success={sorted(missing_success)}."
        )
    unknown_context = set(projection.conditioned_context_ids) - set(
        skill_contract.conditionable_context_ids
    )
    if unknown_context:
        raise ProgressSkillProjectionError(
            "Projection conditions on context absent from the Skill contract: "
            f"{sorted(unknown_context)}."
        )
    if (
        projection.projection_kind == "CONDITIONED_STEP"
        and not projection.conditioned_context_ids
    ):
        raise ProgressSkillProjectionError(
            "CONDITIONED_STEP must bind at least one context identifier."
        )


def _intersections(
    source: SampleIndexInterval,
    domains: Iterable[SampleIndexInterval],
) -> tuple[SampleIndexInterval, ...]:
    result: list[SampleIndexInterval] = []
    for domain in domains:
        begin = max(source.sample_index_begin, domain.sample_index_begin)
        end = min(
            source.sample_index_end_exclusive,
            domain.sample_index_end_exclusive,
        )
        if begin < end:
            result.append(SampleIndexInterval(begin, end))
    return tuple(result)


def _source_failure_id(
    *,
    semantic_index: EpisodeTrainingSemanticIndex,
    attempt_id: str,
    interval: SampleIndexInterval,
) -> str | None:
    candidates = {
        span.source_failure_fact_id
        for span in semantic_index.spans
        if (
            span.semantic_kind == "FAILURE_REGION"
            and span.attempt_id == attempt_id
            and span.source_failure_fact_id is not None
            and span.sample_interval.sample_index_begin
            < interval.sample_index_end_exclusive
            and interval.sample_index_begin
            < span.sample_interval.sample_index_end_exclusive
        )
    }
    if len(candidates) > 1:
        raise ProgressSkillProjectionError(
            "One projected interval overlaps multiple Failure facts."
        )
    return next(iter(candidates), None)


def _stage_spans(
    *,
    semantic_index: EpisodeTrainingSemanticIndex,
    projection: ProgressSkillProjectionV1,
) -> tuple[SemanticSpan, ...]:
    selected = set(projection.source_stage_ids)
    result = tuple(
        span
        for span in semantic_index.spans
        if (
            span.semantic_kind == SEMANTIC_SPAN_STAGE
            and span.progress_step_id == projection.source_progress_step_id
            and span.stage_id in selected
            and span.attempt_id is not None
            and span.source_id is not None
            and not span.sample_interval.empty
        )
    )
    return tuple(
        sorted(
            result,
            key=lambda span: (
                span.sample_interval.sample_index_begin,
                span.sample_interval.sample_index_end_exclusive,
                projection.source_stage_ids.index(str(span.stage_id)),
                str(span.source_id),
            ),
        )
    )


def _raw_intervals(
    *,
    semantic_index: EpisodeTrainingSemanticIndex,
    projection: ProgressSkillProjectionV1,
) -> list[SkillProjectionIntervalV1]:
    result: list[SkillProjectionIntervalV1] = []
    for span in _stage_spans(
        semantic_index=semantic_index,
        projection=projection,
    ):
        assert span.attempt_id is not None
        assert span.stage_id is not None
        assert span.source_id is not None
        for domain, allowed in (
            (TRAINING_DOMAIN_NOMINAL, semantic_index.nominal_intervals),
            (TRAINING_DOMAIN_FAILURE, semantic_index.failure_intervals),
        ):
            for interval in _intersections(span.sample_interval, allowed):
                result.append(
                    SkillProjectionIntervalV1(
                        sample_begin=interval.sample_index_begin,
                        sample_end=interval.sample_index_end_exclusive,
                        training_domain=domain,
                        source_attempt_id=span.attempt_id,
                        source_stage_ids=(span.stage_id,),
                        source_occurrence_ids=(span.source_id,),
                        source_failure_fact_id=(
                            None
                            if domain == TRAINING_DOMAIN_NOMINAL
                            else _source_failure_id(
                                semantic_index=semantic_index,
                                attempt_id=span.attempt_id,
                                interval=interval,
                            )
                        ),
                    )
                )
    return sorted(
        result,
        key=lambda item: (
            item.sample_begin,
            item.sample_end,
            item.training_domain,
            item.source_attempt_id,
        ),
    )


def _merge_adjacent(
    intervals: Iterable[SkillProjectionIntervalV1],
    *,
    blocked_break_indices: Iterable[int] = (),
) -> tuple[SkillProjectionIntervalV1, ...]:
    blocked_breaks = frozenset(blocked_break_indices)
    merged: list[SkillProjectionIntervalV1] = []
    for item in intervals:
        if not merged:
            merged.append(item)
            continue
        previous = merged[-1]
        if (
            previous.sample_end == item.sample_begin
            and item.sample_begin not in blocked_breaks
            and previous.training_domain == item.training_domain
            and previous.source_attempt_id == item.source_attempt_id
            and previous.source_failure_fact_id == item.source_failure_fact_id
        ):
            stage_ids = previous.source_stage_ids + tuple(
                stage_id
                for stage_id in item.source_stage_ids
                if stage_id not in previous.source_stage_ids
            )
            occurrence_ids = previous.source_occurrence_ids + tuple(
                occurrence_id
                for occurrence_id in item.source_occurrence_ids
                if occurrence_id not in previous.source_occurrence_ids
            )
            merged[-1] = SkillProjectionIntervalV1(
                sample_begin=previous.sample_begin,
                sample_end=item.sample_end,
                training_domain=item.training_domain,
                source_attempt_id=item.source_attempt_id,
                source_stage_ids=stage_ids,
                source_occurrence_ids=occurrence_ids,
                source_failure_fact_id=item.source_failure_fact_id,
            )
        else:
            merged.append(item)
    return tuple(merged)


def project_episode_skill_view(
    *,
    progress_profile: TaskProgressProfileV1,
    stage_profile: TaskStageProfileV1,
    skill_contract: SkillTrainingContractV1,
    projection: ProgressSkillProjectionV1,
    semantic_index: EpisodeTrainingSemanticIndex,
    source_stage_annotation_manifest_hash: str | None = None,
) -> ProjectedEpisodeSkillView:
    """Project one sealed Episode while preserving every semantic break.

    Missing stage annotations yield an empty receipt rather than guessing a
    coarse interval.  That makes cross-profile eligibility fail closed while
    keeping the source Episode fully usable through its original interfaces.
    """

    validate_progress_skill_projection(
        progress_profile=progress_profile,
        stage_profile=stage_profile,
        skill_contract=skill_contract,
        projection=projection,
    )
    intervals = _merge_adjacent(
        _raw_intervals(
            semantic_index=semantic_index,
            projection=projection,
        ),
        blocked_break_indices=semantic_index.continuity_break_indices,
    )
    break_values: set[int] = set(semantic_index.continuity_break_indices)
    for index, item in enumerate(intervals):
        previous = None if index == 0 else intervals[index - 1]
        following = None if index + 1 == len(intervals) else intervals[index + 1]
        continues_from_previous = bool(
            previous is not None
            and previous.sample_end == item.sample_begin
            and previous.training_domain == item.training_domain
            and previous.source_attempt_id == item.source_attempt_id
        )
        continues_to_following = bool(
            following is not None
            and item.sample_end == following.sample_begin
            and item.training_domain == following.training_domain
            and item.source_attempt_id == following.source_attempt_id
        )
        if item.sample_begin > 0 and not continues_from_previous:
            break_values.add(item.sample_begin)
        if item.sample_end < semantic_index.sample_count and not continues_to_following:
            break_values.add(item.sample_end)
    breaks = tuple(sorted(break_values))
    receipt_basis = {
        "episode_capture_identity_hash": (semantic_index.episode_capture_identity_hash),
        "source_progress_manifest_hash": (semantic_index.source_progress_manifest_hash),
        "source_stage_annotation_manifest_hash": (
            source_stage_annotation_manifest_hash
        ),
        "skill_training_contract_hash": skill_contract.semantic_content_hash,
        "progress_skill_projection_hash": projection.semantic_content_hash,
        "intervals": tuple(item.to_payload() for item in intervals),
        "continuity_break_indices": breaks,
    }
    receipt_digest = canonical_content_hash(
        receipt_basis,
        domain=_RECEIPT_ID_DOMAIN,
    )
    receipt = SkillProjectionReceiptV1(
        receipt_id=f"skill_projection_{receipt_digest[:24]}",
        episode_capture_identity_hash=(semantic_index.episode_capture_identity_hash),
        source_progress_manifest_hash=(semantic_index.source_progress_manifest_hash),
        source_stage_annotation_manifest_hash=(source_stage_annotation_manifest_hash),
        skill_training_contract_hash=skill_contract.semantic_content_hash,
        progress_skill_projection_hash=projection.semantic_content_hash,
        intervals=intervals,
        continuity_break_indices=breaks,
    )
    return ProjectedEpisodeSkillView(
        skill_contract=skill_contract,
        projection=projection,
        receipt=receipt,
    )


__all__ = [
    "ProgressSkillProjectionError",
    "ProjectedEpisodeSkillView",
    "SkillProjectionCatalog",
    "load_builtin_writing_four_stage_subtasks_skill_projection_catalog",
    "load_builtin_writing_stage1_skill_projection_catalog",
    "load_builtin_writing_skill_projection_catalog",
    "load_skill_projection_catalog",
    "project_episode_skill_view",
    "validate_progress_skill_projection",
]
