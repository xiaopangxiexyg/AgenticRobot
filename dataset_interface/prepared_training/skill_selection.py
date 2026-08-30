"""Fail-closed Episode admission for one prepared Skill training view."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from dataset_contracts.canonical_encoding import canonical_content_hash

from ..progress_annotation_jobs import list_progress_annotation_work_items
from ..progress_skill_projection import SkillProjectionCatalog
from ..progress_stage_annotation_artifacts import (
    LoadedStageAnnotationSet,
    STAGE_ANNOTATION_RELATIVE_ROOT,
    load_stage_annotation_set,
)
from .models import PreparedSourceDataset, PreparedTrainingViewError


PREPARED_SKILL_ELIGIBILITY_REPORT_SCHEMA_V1 = (
    "dexteleop.prepared_skill_eligibility_report.provisional.v1"
)
_REPORT_HASH_DOMAIN = (
    "dexteleop.prepared_skill_eligibility_report.semantic.provisional.v1"
)


@dataclass(frozen=True)
class PreparedSkillEligibilityExclusion:
    source_dataset_id: str
    source_episode_index: int
    episode_capture_identity_hash: str
    reason: str

    def to_payload(self) -> dict[str, object]:
        return {
            "source_dataset_id": self.source_dataset_id,
            "source_episode_index": self.source_episode_index,
            "episode_capture_identity_hash": (
                self.episode_capture_identity_hash
            ),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PreparedSkillEligibilityReport:
    target_skill_id: str
    included_episode_identities: tuple[str, ...]
    selected_annotation_sets: tuple[tuple[str, str], ...]
    exclusions: tuple[PreparedSkillEligibilityExclusion, ...]
    require_no_abnormal_intervals: bool = True
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=PREPARED_SKILL_ELIGIBILITY_REPORT_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        payload = self.to_payload(include_hash=False)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain=_REPORT_HASH_DOMAIN),
        )

    @property
    def annotation_selection(self) -> dict[str, str]:
        return dict(self.selected_annotation_sets)

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "target_skill_id": self.target_skill_id,
            "included_episode_identities": list(
                self.included_episode_identities
            ),
            "selected_annotation_sets": [
                {
                    "episode_capture_identity_hash": identity,
                    "annotation_set_id": annotation_set_id,
                }
                for identity, annotation_set_id in self.selected_annotation_sets
            ],
            "exclusions": [item.to_payload() for item in self.exclusions],
            "require_no_abnormal_intervals": (
                self.require_no_abnormal_intervals
            ),
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


def _annotation_candidates(
    dataset_root: Path,
    episode_uuid: str,
) -> tuple[LoadedStageAnnotationSet, ...]:
    root = dataset_root / STAGE_ANNOTATION_RELATIVE_ROOT / episode_uuid
    if not root.is_dir():
        return ()
    return tuple(
        load_stage_annotation_set(path)
        for path in sorted(child for child in root.iterdir() if child.is_dir())
        if (path / "manifest.cjson").is_file()
    )


def _select_annotation(
    *,
    dataset_root: Path,
    episode_uuid: str,
    episode_identity_hash: str,
    selector: str | None,
) -> LoadedStageAnnotationSet | None:
    candidates = _annotation_candidates(dataset_root, episode_uuid)
    if selector is None:
        if len(candidates) > 1:
            raise PreparedTrainingViewError(
                "Episode has multiple annotation sets; eligibility selection "
                f"must bind one explicitly: {episode_identity_hash}."
            )
        return None if not candidates else candidates[0]
    matches = tuple(
        candidate
        for candidate in candidates
        if selector
        in {
            candidate.binding.annotation_set_id,
            candidate.manifest.semantic_content_hash,
        }
    )
    if len(matches) != 1:
        raise PreparedTrainingViewError(
            "Eligibility annotation selector must match exactly one sealed "
            f"set for Episode {episode_identity_hash}."
        )
    return matches[0]


def _successful_complete_projection(
    *,
    item: object,
    annotation: LoadedStageAnnotationSet,
    progress_step_id: str,
    stage_ids: tuple[str, ...],
) -> bool:
    progress = getattr(item, "progress")
    target_attempt_ids = {
        attempt.attempt_id
        for attempt in progress.timeline.attempts
        if attempt.progress_step_id == progress_step_id
    }
    for attempt_id in target_attempt_ids:
        occurrences = tuple(
            sorted(
                (
                    occurrence
                    for occurrence in annotation.bundle.occurrences
                    if (
                        occurrence.coarse_attempt_id == attempt_id
                        and occurrence.progress_step_id == progress_step_id
                        and occurrence.stage_id in stage_ids
                    )
                ),
                key=lambda occurrence: (
                    occurrence.occurrence_index_in_attempt,
                    occurrence.start_at,
                    occurrence.annotation_id,
                ),
            )
        )
        if (
            tuple(occurrence.stage_id for occurrence in occurrences)
            == stage_ids
            and all(occurrence.outcome == "SUCCESS" for occurrence in occurrences)
        ):
            return True
    return False


def select_strict_successful_skill_episodes(
    sources: Sequence[PreparedSourceDataset],
    *,
    skill_projection_catalog: SkillProjectionCatalog,
    target_skill_id: str,
    annotation_selection: Mapping[str, str] | None = None,
    require_no_abnormal_intervals: bool = True,
) -> PreparedSkillEligibilityReport:
    """Select Episodes where one complete projected Skill Attempt succeeded.

    A later task stop does not invalidate an already successful target Skill.
    Whole-Episode abnormal/review annotations always exclude the Episode.  The
    default strict policy also rejects any local abnormal interval, even when
    it lies outside the selected Skill.
    """

    skill_id = str(target_skill_id).strip()
    if skill_id not in skill_projection_catalog.skill_by_id:
        raise PreparedTrainingViewError(f"Unknown target Skill {skill_id!r}.")
    selectors = {
        str(identity).strip(): str(selector).strip()
        for identity, selector in (annotation_selection or {}).items()
    }
    if any(not identity or not selector for identity, selector in selectors.items()):
        raise PreparedTrainingViewError(
            "annotation_selection keys and values must be non-empty."
        )

    included: list[str] = []
    resolved_annotations: list[tuple[str, str]] = []
    exclusions: list[PreparedSkillEligibilityExclusion] = []
    seen: set[str] = set()
    used_selector_keys: set[str] = set()
    for source in sorted(sources, key=lambda value: value.source_dataset_id):
        for item in list_progress_annotation_work_items(source.dataset_root):
            identity = item.episode_identity_hash
            if identity in seen:
                raise PreparedTrainingViewError(
                    "Skill eligibility sources contain duplicate Episode "
                    f"identity {identity}."
                )
            seen.add(identity)
            selector_key = (
                identity
                if identity in selectors
                else item.episode_uuid
                if item.episode_uuid in selectors
                else None
            )
            if selector_key is not None:
                used_selector_keys.add(selector_key)
            annotation = _select_annotation(
                dataset_root=source.dataset_root,
                episode_uuid=item.episode_uuid,
                episode_identity_hash=identity,
                selector=(
                    None if selector_key is None else selectors[selector_key]
                ),
            )
            reason: str | None = None
            if annotation is None:
                reason = "NO_SEALED_STAGE_ANNOTATION"
            else:
                matches = tuple(
                    projection
                    for projection in skill_projection_catalog.projections_for_skill(
                        skill_id
                    )
                    if (
                        projection.source_progress_profile_hash
                        == item.progress.profile.semantic_content_hash
                        and item.progress.stage_profile is not None
                        and projection.source_stage_profile_hash
                        == item.progress.stage_profile.semantic_content_hash
                    )
                )
                if len(matches) != 1:
                    raise PreparedTrainingViewError(
                        "Every eligibility source Episode must resolve exactly "
                        f"one Skill projection: episode={identity} "
                        f"matches={len(matches)}."
                    )
                projection = matches[0]
                bundle = annotation.bundle
                if bundle.episode_abnormal is not None:
                    reason = "WHOLE_EPISODE_ABNORMAL"
                elif bundle.episode_review_request is not None:
                    reason = "WHOLE_EPISODE_REVIEW_PENDING"
                elif require_no_abnormal_intervals and bundle.abnormal_intervals:
                    reason = "LOCAL_ABNORMAL_INTERVAL"
                elif not _successful_complete_projection(
                    item=item,
                    annotation=annotation,
                    progress_step_id=projection.source_progress_step_id,
                    stage_ids=projection.source_stage_ids,
                ):
                    reason = "TARGET_SKILL_NOT_SUCCESSFUL"

            if reason is None:
                assert annotation is not None
                included.append(identity)
                resolved_annotations.append(
                    (identity, annotation.binding.annotation_set_id)
                )
            else:
                exclusions.append(
                    PreparedSkillEligibilityExclusion(
                        source_dataset_id=source.source_dataset_id,
                        source_episode_index=item.episode_index,
                        episode_capture_identity_hash=identity,
                        reason=reason,
                    )
                )

    unused = sorted(set(selectors) - used_selector_keys)
    if unused:
        raise PreparedTrainingViewError(
            "annotation_selection references Episodes absent from the sources: "
            f"{unused}."
        )
    if not included:
        raise PreparedTrainingViewError(
            "No Episode satisfies the strict successful Skill policy."
        )
    return PreparedSkillEligibilityReport(
        target_skill_id=skill_id,
        included_episode_identities=tuple(sorted(included)),
        selected_annotation_sets=tuple(sorted(resolved_annotations)),
        exclusions=tuple(
            sorted(
                exclusions,
                key=lambda value: (
                    value.source_dataset_id,
                    value.source_episode_index,
                    value.episode_capture_identity_hash,
                ),
            )
        ),
        require_no_abnormal_intervals=bool(require_no_abnormal_intervals),
    )


__all__ = [
    "PREPARED_SKILL_ELIGIBILITY_REPORT_SCHEMA_V1",
    "PreparedSkillEligibilityExclusion",
    "PreparedSkillEligibilityReport",
    "select_strict_successful_skill_episodes",
]
