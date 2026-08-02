"""Selected refinement read views and validated salvage-anchor resolution."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias

from dataset_contracts.progress_annotations import (
    RefinementBoundaryAnnotationV1,
)
from dataset_contracts.progress_profile import TaskProgressProfileV1
from dataset_contracts.progress_refinement import (
    RefinementBoundaryDefinitionV1,
    TaskRefinementProfileV1,
    validate_task_refinement_profile,
)

from .progress_exclusions import derive_coarse_failure_training_view
from .progress_timeline import ResolvedProgressAttempt, ResolvedProgressTimeline
from .sample_axis import EpisodeSampleAxis, SampleIndexInterval


class ProgressRefinementResolutionError(ValueError):
    """Raised when selected refinement evidence is not safely resolvable."""


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProgressRefinementResolutionError(
            f"{name} must be a non-empty string."
        )
    return value.strip()


def _sha256(value: object, *, name: str) -> str:
    result = _required(value, name=name).lower()
    if len(result) != 64 or any(
        char not in "0123456789abcdef" for char in result
    ):
        raise ProgressRefinementResolutionError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return result


def _finite(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProgressRefinementResolutionError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ProgressRefinementResolutionError(f"{name} must be finite.")
    return result


def _confidence(value: object) -> float:
    result = _finite(value, name="confidence")
    if result < 0.0 or result > 1.0:
        raise ProgressRefinementResolutionError(
            "confidence must lie in [0, 1]."
        )
    return result


# Compatibility/read-view name. The persisted contract is the single field
# truth; dataset_interface does not maintain a second annotation schema.
RefinementBoundaryObservation: TypeAlias = RefinementBoundaryAnnotationV1


@dataclass(frozen=True)
class SelectedRefinementAnnotationSet:
    """One already-selected annotation-set read view."""

    annotation_set_id: str
    source_annotation_manifest_hash: str
    source_progress_manifest_hash: str
    task_refinement_profile_hash: str
    sealed: bool
    observations: tuple[RefinementBoundaryObservation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "annotation_set_id",
            _required(self.annotation_set_id, name="annotation_set_id"),
        )
        for name in (
            "source_annotation_manifest_hash",
            "source_progress_manifest_hash",
            "task_refinement_profile_hash",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        if not isinstance(self.sealed, bool):
            raise ProgressRefinementResolutionError("sealed must be boolean.")
        observations = tuple(self.observations)
        if any(
            not isinstance(item, RefinementBoundaryAnnotationV1)
            for item in observations
        ):
            raise ProgressRefinementResolutionError(
                "observations must contain refinement-boundary observations."
            )
        annotation_ids = tuple(item.annotation_id for item in observations)
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ProgressRefinementResolutionError(
                "annotation_id values must be unique within a selected set."
            )
        object.__setattr__(self, "observations", observations)


@dataclass(frozen=True)
class ValidatedSalvageAnchor:
    refinement_boundary_id: str
    annotation_id: str
    effective_at: float
    confidence: float
    supporting_boundary_ids: tuple[str, ...]
    supporting_annotation_ids: tuple[str, ...]


@dataclass(frozen=True)
class RefinementFailureTrainingExclusion:
    source_failure_fact_id: str
    attempt_id: str
    progress_step_id: str
    coarse_attempt_start_at: float
    exclusion_start_at: float
    failure_cut_at: float
    sample_interval: SampleIndexInterval
    exclusion_start_basis: str
    salvage_status: str
    salvage_anchor: ValidatedSalvageAnchor | None


@dataclass(frozen=True)
class RefinementFailureTrainingView:
    episode_capture_identity_hash: str
    source_progress_manifest_hash: str
    source_annotation_manifest_hash: str
    task_refinement_profile_hash: str
    annotation_set_id: str
    clock_domain_id: str
    sample_count: int
    minimum_confidence: float
    exclusions: tuple[RefinementFailureTrainingExclusion, ...]


def _attempt_observations(
    *,
    timeline: ResolvedProgressTimeline,
    refinement_profile: TaskRefinementProfileV1,
    annotation_set: SelectedRefinementAnnotationSet,
) -> dict[str, list[RefinementBoundaryObservation]]:
    attempt_by_id = {item.attempt_id: item for item in timeline.attempts}
    if len(attempt_by_id) != len(timeline.attempts):
        raise ProgressRefinementResolutionError(
            "Resolved timeline contains duplicate Attempt identities."
        )
    boundary_by_id = refinement_profile.boundary_by_id
    grouped: dict[str, list[RefinementBoundaryObservation]] = {}
    for observation in annotation_set.observations:
        attempt = attempt_by_id.get(observation.coarse_attempt_id)
        if attempt is None:
            raise ProgressRefinementResolutionError(
                f"Annotation references unknown Attempt "
                f"{observation.coarse_attempt_id!r}."
            )
        boundary = boundary_by_id.get(observation.refinement_boundary_id)
        if boundary is None:
            raise ProgressRefinementResolutionError(
                f"Annotation references unknown refinement boundary "
                f"{observation.refinement_boundary_id!r}."
            )
        if boundary.progress_step_id != attempt.progress_step_id:
            raise ProgressRefinementResolutionError(
                "Annotation boundary does not belong to its Attempt step."
            )
        if observation.clock_domain_id != timeline.clock_domain_id:
            raise ProgressRefinementResolutionError(
                "Refinement annotation must use the Episode clock."
            )
        grouped.setdefault(attempt.attempt_id, []).append(observation)
    return grouped


def _invalid_observation_order(
    *,
    attempt: ResolvedProgressAttempt,
    definitions: tuple[RefinementBoundaryDefinitionV1, ...],
    observations_by_boundary: dict[
        str,
        tuple[RefinementBoundaryObservation, ...],
    ],
) -> bool:
    if attempt.end_at is None:
        return True
    definition_by_id = {
        item.refinement_boundary_id: item for item in definitions
    }
    unique_observations: list[
        tuple[RefinementBoundaryDefinitionV1, RefinementBoundaryObservation]
    ] = []
    for boundary_id, observations in observations_by_boundary.items():
        if len(observations) != 1:
            continue
        observation = observations[0]
        if (
            observation.effective_at < attempt.start_at
            or observation.effective_at > attempt.end_at
        ):
            return True
        unique_observations.append(
            (definition_by_id[boundary_id], observation)
        )
    unique_observations.sort(key=lambda item: item[0].ordinal)
    return any(
        later[1].effective_at < earlier[1].effective_at
        for earlier, later in zip(
            unique_observations,
            unique_observations[1:],
        )
    )


def _validated_anchor(
    *,
    attempt: ResolvedProgressAttempt,
    definitions: tuple[RefinementBoundaryDefinitionV1, ...],
    observations: tuple[RefinementBoundaryObservation, ...],
    minimum_confidence: float,
) -> tuple[str, ValidatedSalvageAnchor | None]:
    if not observations:
        return "UNANNOTATED", None
    observations_by_boundary: dict[
        str,
        tuple[RefinementBoundaryObservation, ...],
    ] = {}
    for definition in definitions:
        matches = tuple(
            item
            for item in observations
            if item.refinement_boundary_id
            == definition.refinement_boundary_id
        )
        if matches:
            observations_by_boundary[definition.refinement_boundary_id] = (
                matches
            )
    if any(len(items) > 1 for items in observations_by_boundary.values()):
        return "CONFLICTED", None
    if _invalid_observation_order(
        attempt=attempt,
        definitions=definitions,
        observations_by_boundary=observations_by_boundary,
    ):
        return "INVALID", None
    if attempt.end_at is None:
        return "INVALID", None

    definition_by_id = {
        item.refinement_boundary_id: item for item in definitions
    }
    eligible_definitions = tuple(
        item
        for item in definitions
        if item.eligible_as_exclusion_anchor
        and item.certifies_valid_prefix
    )
    if not eligible_definitions:
        return "NO_ELIGIBLE_BOUNDARY", None

    valid: list[
        tuple[RefinementBoundaryDefinitionV1, ValidatedSalvageAnchor]
    ] = []
    missing_evidence = False
    below_confidence = False
    for definition in eligible_definitions:
        candidate_items = observations_by_boundary.get(
            definition.refinement_boundary_id,
            (),
        )
        if not candidate_items:
            missing_evidence = True
            continue
        candidate = candidate_items[0]
        if candidate.effective_at >= attempt.end_at:
            continue
        required_definitions = tuple(
            definition_by_id[boundary_id]
            for boundary_id in definition.required_predecessor_boundary_ids
        )
        required_observations: list[RefinementBoundaryObservation] = []
        incomplete = False
        for predecessor in required_definitions:
            predecessor_items = observations_by_boundary.get(
                predecessor.refinement_boundary_id,
                (),
            )
            if len(predecessor_items) != 1:
                incomplete = True
                break
            required_observations.append(predecessor_items[0])
        if incomplete:
            missing_evidence = True
            continue
        supporting = tuple(required_observations) + (candidate,)
        if any(item.effective_at > candidate.effective_at for item in supporting):
            return "INVALID", None
        if any(item.confidence < minimum_confidence for item in supporting):
            below_confidence = True
            continue
        valid.append(
            (
                definition,
                ValidatedSalvageAnchor(
                    refinement_boundary_id=(
                        definition.refinement_boundary_id
                    ),
                    annotation_id=candidate.annotation_id,
                    effective_at=candidate.effective_at,
                    confidence=candidate.confidence,
                    supporting_boundary_ids=tuple(
                        item.refinement_boundary_id
                        for item in required_definitions
                    )
                    + (definition.refinement_boundary_id,),
                    supporting_annotation_ids=tuple(
                        item.annotation_id for item in supporting
                    ),
                ),
            )
        )
    if valid:
        return "VALIDATED", max(
            valid,
            key=lambda item: (
                item[1].effective_at,
                item[0].ordinal,
                item[1].annotation_id,
            ),
        )[1]
    if missing_evidence:
        return "INCOMPLETE", None
    if below_confidence:
        return "BELOW_CONFIDENCE", None
    return "NO_VALID_ANCHOR", None


def derive_refinement_failure_training_view(
    *,
    timeline: ResolvedProgressTimeline,
    sample_axis: EpisodeSampleAxis,
    progress_profile: TaskProgressProfileV1,
    refinement_profile: TaskRefinementProfileV1,
    annotation_set: SelectedRefinementAnnotationSet,
    minimum_confidence: float = 0.0,
) -> RefinementFailureTrainingView:
    """Shrink coarse Failure exclusions only with fully validated evidence."""

    threshold = _confidence(minimum_confidence)
    validate_task_refinement_profile(
        progress_profile=progress_profile,
        refinement_profile=refinement_profile,
    )
    if (
        timeline.task_progress_profile_hash
        != progress_profile.semantic_content_hash
    ):
        raise ProgressRefinementResolutionError(
            "Progress timeline/profile hash mismatch."
        )
    if (
        annotation_set.task_refinement_profile_hash
        != refinement_profile.semantic_content_hash
    ):
        raise ProgressRefinementResolutionError(
            "Annotation/refinement profile hash mismatch."
        )
    if not annotation_set.sealed:
        raise ProgressRefinementResolutionError(
            "Refinement resolution requires a sealed annotation set."
        )
    if (
        timeline.source_progress_manifest_hash is None
        or annotation_set.source_progress_manifest_hash
        != timeline.source_progress_manifest_hash
    ):
        raise ProgressRefinementResolutionError(
            "Annotation set is not bound to the resolved progress manifest."
        )

    coarse = derive_coarse_failure_training_view(
        timeline=timeline,
        sample_axis=sample_axis,
    )
    grouped = _attempt_observations(
        timeline=timeline,
        refinement_profile=refinement_profile,
        annotation_set=annotation_set,
    )
    attempt_by_id = {item.attempt_id: item for item in timeline.attempts}
    exclusions: list[RefinementFailureTrainingExclusion] = []
    for item in coarse.exclusions:
        attempt = attempt_by_id[item.attempt_id]
        definitions = refinement_profile.boundaries_for_step(
            item.progress_step_id
        )
        status, anchor = _validated_anchor(
            attempt=attempt,
            definitions=definitions,
            observations=tuple(grouped.get(item.attempt_id, ())),
            minimum_confidence=threshold,
        )
        start_at = (
            anchor.effective_at
            if anchor is not None
            else item.attempt_start_at
        )
        interval = (
            sample_axis.closed_time_interval(
                start_at=start_at,
                end_at=item.failure_cut_at,
            )
            if anchor is not None
            else item.sample_interval
        )
        exclusions.append(
            RefinementFailureTrainingExclusion(
                source_failure_fact_id=item.source_failure_fact_id,
                attempt_id=item.attempt_id,
                progress_step_id=item.progress_step_id,
                coarse_attempt_start_at=item.attempt_start_at,
                exclusion_start_at=start_at,
                failure_cut_at=item.failure_cut_at,
                sample_interval=interval,
                exclusion_start_basis=(
                    "VALIDATED_SALVAGE_ANCHOR"
                    if anchor is not None
                    else "ATTEMPT_START"
                ),
                salvage_status=status,
                salvage_anchor=anchor,
            )
        )
    return RefinementFailureTrainingView(
        episode_capture_identity_hash=timeline.episode_capture_identity_hash,
        source_progress_manifest_hash=(
            annotation_set.source_progress_manifest_hash
        ),
        source_annotation_manifest_hash=(
            annotation_set.source_annotation_manifest_hash
        ),
        task_refinement_profile_hash=(
            refinement_profile.semantic_content_hash
        ),
        annotation_set_id=annotation_set.annotation_set_id,
        clock_domain_id=timeline.clock_domain_id,
        sample_count=sample_axis.sample_count,
        minimum_confidence=threshold,
        exclusions=tuple(exclusions),
    )


__all__ = [
    "ProgressRefinementResolutionError",
    "RefinementBoundaryObservation",
    "RefinementFailureTrainingExclusion",
    "RefinementFailureTrainingView",
    "SelectedRefinementAnnotationSet",
    "ValidatedSalvageAnchor",
    "derive_refinement_failure_training_view",
]
