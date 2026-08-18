"""Additive contracts for supplementing coarse stage annotations.

Legacy stage annotation sets are intentionally left immutable.  A detail set
only annotates one coarse progress step with a newer fine-stage profile and
binds to the exact legacy annotation manifest it supplements.  Readers can
therefore fall back to the legacy interpretation when no valid detail set is
present without silently changing the meaning of existing data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from .canonical_encoding import canonical_content_hash
from .progress_stage_annotations import (
    StageAnnotationBundleV1,
    StageAnnotationManifestV1,
    StageAnnotationSetBindingV1,
    StageOccurrenceAnnotationV1,
)
from .progress_stages import TaskStageProfileV1


DETAIL_STAGE_ANNOTATION_BUNDLE_SCHEMA_V1 = (
    "dexteleop.detail_stage_annotation_bundle.v1"
)
DETAIL_STAGE_ANNOTATION_SET_BINDING_SCHEMA_V1 = (
    "dexteleop.detail_stage_annotation_set_binding.v1"
)
DETAIL_STAGE_ANNOTATION_MANIFEST_SCHEMA_V1 = (
    "dexteleop.detail_stage_annotation_manifest.v1"
)
DETAIL_STAGE_ANNOTATION_BINDING_FILE_DOMAIN_V1 = (
    "dexteleop.detail_stage_annotation_binding.file.v1"
)
DETAIL_STAGE_ANNOTATION_BUNDLE_FILE_DOMAIN_V1 = (
    "dexteleop.detail_stage_annotation_bundle.file.v1"
)
DETAIL_STAGE_ANNOTATION_MANIFEST_FILE_DOMAIN_V1 = (
    "dexteleop.detail_stage_annotation_manifest.file.v1"
)

_BUNDLE_HASH_DOMAIN = "dexteleop.detail_stage_annotation_bundle.semantic.v1"
_BINDING_HASH_DOMAIN = (
    "dexteleop.detail_stage_annotation_set_binding.semantic.v1"
)
_MANIFEST_HASH_DOMAIN = (
    "dexteleop.detail_stage_annotation_manifest.semantic.v1"
)


class DetailStageAnnotationContractError(ValueError):
    """Raised when a detail overlay is ambiguous or not bound safely."""


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DetailStageAnnotationContractError(
            f"{name} must be a non-empty string."
        )
    return value.strip()


def _sha256(value: object, *, name: str) -> str:
    result = _required(value, name=name).lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise DetailStageAnnotationContractError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return result


def _finite(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DetailStageAnnotationContractError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise DetailStageAnnotationContractError(f"{name} must be finite.")
    return result


def _verify_declared_hash(
    payload: Mapping[str, Any],
    *,
    actual: str,
    name: str,
) -> None:
    declared = payload.get("semantic_content_hash")
    if declared is not None and str(declared) != actual:
        raise DetailStageAnnotationContractError(f"{name} hash mismatch.")


@dataclass(frozen=True)
class DetailStageAnnotationBundleV1:
    """Fine-stage occurrences for one target coarse progress step.

    Episode-level abnormal labels, boundary overrides, and capture-end
    interpretation remain owned by the legacy bundle.  Keeping this bundle
    narrow prevents a detail edit from accidentally replacing those labels.
    """

    occurrences: tuple[StageOccurrenceAnnotationV1, ...] = ()
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=DETAIL_STAGE_ANNOTATION_BUNDLE_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        occurrences = tuple(self.occurrences)
        if any(
            not isinstance(item, StageOccurrenceAnnotationV1)
            for item in occurrences
        ):
            raise DetailStageAnnotationContractError(
                "occurrences must contain StageOccurrenceAnnotationV1 values."
            )
        annotation_ids = tuple(item.annotation_id for item in occurrences)
        if len(annotation_ids) != len(set(annotation_ids)):
            raise DetailStageAnnotationContractError(
                "Detail occurrence annotation IDs must be unique."
            )
        object.__setattr__(
            self,
            "occurrences",
            tuple(
                sorted(
                    occurrences,
                    key=lambda item: (
                        item.coarse_attempt_id,
                        item.start_at,
                        item.end_at,
                        item.annotation_id,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_BUNDLE_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "occurrences": tuple(
                item.to_payload() for item in self.occurrences
            ),
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "DetailStageAnnotationBundleV1":
        if payload.get("schema") != DETAIL_STAGE_ANNOTATION_BUNDLE_SCHEMA_V1:
            raise DetailStageAnnotationContractError(
                "Unsupported detail stage annotation bundle schema."
            )
        raw_occurrences = payload.get("occurrences", ())
        if not isinstance(raw_occurrences, (tuple, list)) or any(
            not isinstance(item, Mapping) for item in raw_occurrences
        ):
            raise DetailStageAnnotationContractError(
                "Detail occurrences must be a sequence of objects."
            )
        result = cls(
            occurrences=tuple(
                StageOccurrenceAnnotationV1.from_payload(item)
                for item in raw_occurrences
            )
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="detail stage annotation bundle",
        )
        return result


@dataclass(frozen=True)
class DetailStageAnnotationSetBindingV1:
    """Bind one immutable detail overlay to one immutable coarse set."""

    annotation_set_id: str
    episode_capture_identity_hash: str
    source_progress_manifest_hash: str
    base_annotation_set_id: str
    base_annotation_set_manifest_hash: str
    base_task_stage_profile_hash: str
    detail_task_stage_profile_hash: str
    target_progress_step_id: str
    bound_clock_domain_id: str
    annotator_id: str
    created_at: float
    created_clock_domain_id: str
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=DETAIL_STAGE_ANNOTATION_SET_BINDING_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "annotation_set_id",
            "base_annotation_set_id",
            "target_progress_step_id",
            "bound_clock_domain_id",
            "annotator_id",
            "created_clock_domain_id",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        for name in (
            "episode_capture_identity_hash",
            "source_progress_manifest_hash",
            "base_annotation_set_manifest_hash",
            "base_task_stage_profile_hash",
            "detail_task_stage_profile_hash",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "created_at",
            _finite(self.created_at, name="created_at"),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_BINDING_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "annotation_set_id": self.annotation_set_id,
            "episode_capture_identity_hash": (
                self.episode_capture_identity_hash
            ),
            "source_progress_manifest_hash": (
                self.source_progress_manifest_hash
            ),
            "base_annotation_set_id": self.base_annotation_set_id,
            "base_annotation_set_manifest_hash": (
                self.base_annotation_set_manifest_hash
            ),
            "base_task_stage_profile_hash": (
                self.base_task_stage_profile_hash
            ),
            "detail_task_stage_profile_hash": (
                self.detail_task_stage_profile_hash
            ),
            "target_progress_step_id": self.target_progress_step_id,
            "bound_clock_domain_id": self.bound_clock_domain_id,
            "annotator_id": self.annotator_id,
            "created_at": self.created_at,
            "created_clock_domain_id": self.created_clock_domain_id,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "DetailStageAnnotationSetBindingV1":
        if (
            payload.get("schema")
            != DETAIL_STAGE_ANNOTATION_SET_BINDING_SCHEMA_V1
        ):
            raise DetailStageAnnotationContractError(
                "Unsupported detail stage annotation binding schema."
            )
        result = cls(
            annotation_set_id=payload.get("annotation_set_id", ""),
            episode_capture_identity_hash=payload.get(
                "episode_capture_identity_hash",
                "",
            ),
            source_progress_manifest_hash=payload.get(
                "source_progress_manifest_hash",
                "",
            ),
            base_annotation_set_id=payload.get(
                "base_annotation_set_id",
                "",
            ),
            base_annotation_set_manifest_hash=payload.get(
                "base_annotation_set_manifest_hash",
                "",
            ),
            base_task_stage_profile_hash=payload.get(
                "base_task_stage_profile_hash",
                "",
            ),
            detail_task_stage_profile_hash=payload.get(
                "detail_task_stage_profile_hash",
                "",
            ),
            target_progress_step_id=payload.get(
                "target_progress_step_id",
                "",
            ),
            bound_clock_domain_id=payload.get("bound_clock_domain_id", ""),
            annotator_id=payload.get("annotator_id", ""),
            created_at=payload.get("created_at", float("nan")),
            created_clock_domain_id=payload.get(
                "created_clock_domain_id",
                "",
            ),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="detail stage annotation binding",
        )
        return result


@dataclass(frozen=True)
class DetailStageAnnotationManifestV1:
    """Visibility marker for one complete detail overlay."""

    annotation_set_binding_hash: str
    annotation_bundle_hash: str
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=DETAIL_STAGE_ANNOTATION_MANIFEST_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in ("annotation_set_binding_hash", "annotation_bundle_hash"):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_MANIFEST_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "annotation_set_binding_hash": self.annotation_set_binding_hash,
            "annotation_bundle_hash": self.annotation_bundle_hash,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "DetailStageAnnotationManifestV1":
        if payload.get("schema") != DETAIL_STAGE_ANNOTATION_MANIFEST_SCHEMA_V1:
            raise DetailStageAnnotationContractError(
                "Unsupported detail stage annotation manifest schema."
            )
        result = cls(
            annotation_set_binding_hash=payload.get(
                "annotation_set_binding_hash",
                "",
            ),
            annotation_bundle_hash=payload.get("annotation_bundle_hash", ""),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="detail stage annotation manifest",
        )
        return result


def _bundle_annotation_ids(bundle: StageAnnotationBundleV1) -> set[str]:
    values: tuple[object, ...] = (
        bundle.occurrences
        + bundle.abnormal_intervals
        + bundle.progress_boundary_overrides
        + (() if bundle.capture_end_step is None else (bundle.capture_end_step,))
        + (() if bundle.episode_abnormal is None else (bundle.episode_abnormal,))
        + (
            ()
            if bundle.episode_review_request is None
            else (bundle.episode_review_request,)
        )
    )
    return {item.annotation_id for item in values}


def validate_detail_stage_profile_compatibility(
    *,
    base_stage_profile: TaskStageProfileV1,
    detail_stage_profile: TaskStageProfileV1,
    target_progress_step_id: str,
) -> None:
    """Require a detail profile to differ only inside one coarse step."""

    if not isinstance(base_stage_profile, TaskStageProfileV1):
        raise TypeError("base_stage_profile must be TaskStageProfileV1.")
    if not isinstance(detail_stage_profile, TaskStageProfileV1):
        raise TypeError("detail_stage_profile must be TaskStageProfileV1.")
    target = _required(
        target_progress_step_id,
        name="target_progress_step_id",
    )
    if (
        base_stage_profile.task_progress_profile_hash
        != detail_stage_profile.task_progress_profile_hash
    ):
        raise DetailStageAnnotationContractError(
            "Base and detail stage profiles use different progress profiles."
        )
    base_step_ids = {
        item.progress_step_id for item in base_stage_profile.stages
    }
    detail_step_ids = {
        item.progress_step_id for item in detail_stage_profile.stages
    }
    if base_step_ids != detail_step_ids:
        raise DetailStageAnnotationContractError(
            "Detail stage profile must cover the same coarse progress steps."
        )
    if target not in base_step_ids:
        raise DetailStageAnnotationContractError(
            f"Target progress step {target!r} is absent from the base profile."
        )
    if target not in detail_step_ids:
        raise DetailStageAnnotationContractError(
            f"Target progress step {target!r} is absent from the detail profile."
        )
    if (
        base_stage_profile.stages_for_step(target)
        == detail_stage_profile.stages_for_step(target)
    ):
        raise DetailStageAnnotationContractError(
            "Detail target stages must differ from the base target stages."
        )
    for step_id in sorted(base_step_ids - {target}):
        if (
            base_stage_profile.stages_for_step(step_id)
            != detail_stage_profile.stages_for_step(step_id)
        ):
            raise DetailStageAnnotationContractError(
                "Detail profile changed a non-target coarse progress step: "
                f"{step_id!r}."
            )


def validate_detail_stage_annotation_set(
    *,
    source_progress_manifest: Any,
    base_binding: StageAnnotationSetBindingV1,
    base_bundle: StageAnnotationBundleV1,
    base_manifest: StageAnnotationManifestV1,
    base_stage_profile: TaskStageProfileV1,
    detail_stage_profile: TaskStageProfileV1,
    binding: DetailStageAnnotationSetBindingV1,
    bundle: DetailStageAnnotationBundleV1,
) -> None:
    """Validate a detail overlay against its immutable coarse sources."""

    if not isinstance(base_binding, StageAnnotationSetBindingV1):
        raise TypeError("base_binding must be StageAnnotationSetBindingV1.")
    if not isinstance(base_bundle, StageAnnotationBundleV1):
        raise TypeError("base_bundle must be StageAnnotationBundleV1.")
    if not isinstance(base_manifest, StageAnnotationManifestV1):
        raise TypeError("base_manifest must be StageAnnotationManifestV1.")
    if not isinstance(binding, DetailStageAnnotationSetBindingV1):
        raise TypeError(
            "binding must be DetailStageAnnotationSetBindingV1."
        )
    if not isinstance(bundle, DetailStageAnnotationBundleV1):
        raise TypeError("bundle must be DetailStageAnnotationBundleV1.")
    validate_detail_stage_profile_compatibility(
        base_stage_profile=base_stage_profile,
        detail_stage_profile=detail_stage_profile,
        target_progress_step_id=binding.target_progress_step_id,
    )
    if (
        base_manifest.annotation_set_binding_hash
        != base_binding.semantic_content_hash
        or base_manifest.annotation_bundle_hash
        != base_bundle.semantic_content_hash
    ):
        raise DetailStageAnnotationContractError(
            "Base annotation manifest does not bind its loaded content."
        )
    if (
        source_progress_manifest.episode_capture_identity_hash
        != binding.episode_capture_identity_hash
    ):
        raise DetailStageAnnotationContractError(
            "Detail binding and progress manifest Episode identity differ."
        )
    if (
        source_progress_manifest.semantic_content_hash
        != binding.source_progress_manifest_hash
    ):
        raise DetailStageAnnotationContractError(
            "Detail binding does not reference the exact progress manifest."
        )
    if (
        source_progress_manifest.task_progress_profile_hash
        != base_stage_profile.task_progress_profile_hash
    ):
        raise DetailStageAnnotationContractError(
            "Progress manifest and base stage profile use different coarse "
            "profiles."
        )
    if (
        base_binding.episode_capture_identity_hash
        != source_progress_manifest.episode_capture_identity_hash
        or base_binding.source_progress_manifest_hash
        != source_progress_manifest.semantic_content_hash
        or base_binding.task_stage_profile_hash
        != base_stage_profile.semantic_content_hash
    ):
        raise DetailStageAnnotationContractError(
            "The loaded base annotation set is not bound to the supplied "
            "progress manifest and base stage profile."
        )
    if binding.base_annotation_set_id != base_binding.annotation_set_id:
        raise DetailStageAnnotationContractError(
            "Detail binding references a different base annotation set."
        )
    if binding.annotation_set_id == base_binding.annotation_set_id:
        raise DetailStageAnnotationContractError(
            "Detail and base annotation set IDs must be different."
        )
    if (
        binding.base_annotation_set_manifest_hash
        != base_manifest.semantic_content_hash
    ):
        raise DetailStageAnnotationContractError(
            "Detail binding does not reference the exact base annotation "
            "manifest."
        )
    if binding.base_task_stage_profile_hash != base_stage_profile.semantic_content_hash:
        raise DetailStageAnnotationContractError(
            "Detail binding and base stage profile differ."
        )
    if binding.detail_task_stage_profile_hash != detail_stage_profile.semantic_content_hash:
        raise DetailStageAnnotationContractError(
            "Detail binding and detail stage profile differ."
        )
    if binding.bound_clock_domain_id != base_binding.bound_clock_domain_id:
        raise DetailStageAnnotationContractError(
            "Detail and base annotation sets use different Episode clocks."
        )
    if any(
        item.clock_domain_id != binding.bound_clock_domain_id
        for item in bundle.occurrences
    ):
        raise DetailStageAnnotationContractError(
            "Every detail occurrence must use the bound Episode clock."
        )

    target = binding.target_progress_step_id
    target_stage_ids = {
        item.stage_id
        for item in detail_stage_profile.stages_for_step(target)
    }
    base_target_occurrences = tuple(
        item
        for item in base_bundle.occurrences
        if item.progress_step_id == target
    )
    detail_attempt_ids = {
        item.coarse_attempt_id for item in bundle.occurrences
    }
    base_attempt_ids = {
        item.coarse_attempt_id for item in base_target_occurrences
    }
    if detail_attempt_ids != base_attempt_ids:
        raise DetailStageAnnotationContractError(
            "Detail occurrences must cover exactly the coarse target Attempts."
        )
    base_failure_ids = {
        item.source_failure_fact_id
        for item in base_target_occurrences
        if item.source_failure_fact_id is not None
    }
    detail_failure_ids = {
        item.source_failure_fact_id
        for item in bundle.occurrences
        if item.source_failure_fact_id is not None
    }
    if detail_failure_ids != base_failure_ids:
        raise DetailStageAnnotationContractError(
            "Detail occurrences must preserve the target Failure facts."
        )
    if any(item.stage_id not in target_stage_ids for item in bundle.occurrences):
        raise DetailStageAnnotationContractError(
            "Detail occurrences must reference stages in the detail target."
        )
    if any(item.progress_step_id != target for item in bundle.occurrences):
        raise DetailStageAnnotationContractError(
            "Detail occurrences must belong to the target progress step."
        )
    if _bundle_annotation_ids(base_bundle) & {
        item.annotation_id for item in bundle.occurrences
    }:
        raise DetailStageAnnotationContractError(
            "Detail occurrence IDs must not overwrite base annotation IDs."
        )

    base_by_attempt: dict[str, list[StageOccurrenceAnnotationV1]] = {}
    for item in base_target_occurrences:
        base_by_attempt.setdefault(item.coarse_attempt_id, []).append(item)

    by_attempt: dict[str, list[StageOccurrenceAnnotationV1]] = {}
    detail_failure_by_attempt: dict[str, set[str]] = {}
    detail_failure_occurrences_by_fact: dict[
        str, list[StageOccurrenceAnnotationV1]
    ] = {}
    for item in bundle.occurrences:
        by_attempt.setdefault(item.coarse_attempt_id, []).append(item)
        if item.outcome == "FAILURE":
            if item.source_failure_fact_id is None:
                raise DetailStageAnnotationContractError(
                    "A detail FAILURE must preserve its source Failure fact."
                )
            detail_failure_by_attempt.setdefault(
                item.coarse_attempt_id,
                set(),
            ).add(item.source_failure_fact_id)
            detail_failure_occurrences_by_fact.setdefault(
                item.source_failure_fact_id,
                [],
            ).append(item)
        if not any(
            base.start_at <= item.start_at
            and item.end_at <= base.end_at
            for base in base_target_occurrences
            if base.coarse_attempt_id == item.coarse_attempt_id
        ):
            raise DetailStageAnnotationContractError(
                "A detail occurrence must remain inside its coarse target "
                "occurrence."
            )

    base_failure_by_attempt = {
        attempt_id: {
            value.source_failure_fact_id
            for value in values
            if value.source_failure_fact_id is not None
        }
        for attempt_id, values in base_by_attempt.items()
    }
    if {
        attempt_id: failure_ids
        for attempt_id, failure_ids in base_failure_by_attempt.items()
        if failure_ids
    } != detail_failure_by_attempt:
        raise DetailStageAnnotationContractError(
            "Detail Failure facts must remain attached to their coarse "
            "Attempts."
        )
    if any(
        len(values) != 1
        for values in detail_failure_occurrences_by_fact.values()
    ):
        raise DetailStageAnnotationContractError(
            "Each source Failure fact must identify exactly one detail "
            "Failure occurrence."
        )

    target_stages = tuple(detail_stage_profile.stages_for_step(target))
    stage_position = {
        stage.stage_id: position
        for position, stage in enumerate(target_stages)
    }
    expected_positions = set(range(len(target_stages)))
    for attempt_id, values in by_attempt.items():
        ordered = sorted(values, key=lambda item: (item.start_at, item.end_at))
        previous_end = None
        previous_position = -1
        for item in ordered:
            if previous_end is not None and item.start_at < previous_end:
                raise DetailStageAnnotationContractError(
                    "Detail occurrences cannot overlap within one Attempt."
                )
            position = stage_position[item.stage_id]
            if position < previous_position:
                raise DetailStageAnnotationContractError(
                    "Detail stages must remain in profile order within one "
                    f"Attempt {attempt_id!r}."
                )
            previous_end = item.end_at
            previous_position = position

        observed_positions = {stage_position[item.stage_id] for item in ordered}
        highest_position = max(observed_positions)
        if observed_positions != set(range(highest_position + 1)):
            raise DetailStageAnnotationContractError(
                "Detail stages must form a contiguous profile prefix within "
                f"Attempt {attempt_id!r}."
            )

        base_values = base_by_attempt[attempt_id]
        base_has_failure = bool(base_failure_by_attempt[attempt_id])
        ordered_base_values = sorted(
            base_values,
            key=lambda item: (item.start_at, item.end_at),
        )
        # An abnormal interval can split one successful coarse occurrence into
        # a truncated prefix and a normal suffix.  The suffix remains the
        # terminal source outcome; only a terminal truncation requires the
        # detail overlay to end in TRUNCATED_BY_ABNORMAL.
        base_has_truncation = (
            not base_has_failure
            and ordered_base_values[-1].outcome
            == "TRUNCATED_BY_ABNORMAL"
        )
        failure_values = [
            item for item in ordered if item.outcome == "FAILURE"
        ]
        if base_has_failure:
            if len(failure_values) != 1 or failure_values[0] is not ordered[-1]:
                raise DetailStageAnnotationContractError(
                    "A detail Failure Attempt must end at exactly one "
                    "detail FAILURE occurrence."
                )
            failure_position = stage_position[failure_values[0].stage_id]
            if observed_positions != set(range(failure_position + 1)):
                raise DetailStageAnnotationContractError(
                    "A detail Failure Attempt must cover the ordered stage "
                    "prefix through its Failure stage."
                )
            if any(
                item.outcome != "SUCCESS" for item in ordered[:-1]
            ):
                raise DetailStageAnnotationContractError(
                    "Stages before a detail Failure must be SUCCESS."
                )
        elif base_has_truncation:
            if (
                ordered[-1].outcome != "TRUNCATED_BY_ABNORMAL"
                or any(
                    item.outcome != "SUCCESS" for item in ordered[:-1]
                )
            ):
                raise DetailStageAnnotationContractError(
                    "A truncated detail Attempt must end with one "
                    "TRUNCATED_BY_ABNORMAL occurrence."
                )
        else:
            if any(item.outcome != "SUCCESS" for item in ordered):
                raise DetailStageAnnotationContractError(
                    "A successful coarse Attempt cannot gain a detail "
                    "Failure or truncation."
                )
            if observed_positions != expected_positions:
                raise DetailStageAnnotationContractError(
                    "A successful detail Attempt must cover every target "
                    "fine stage."
                )


__all__ = [
    "DETAIL_STAGE_ANNOTATION_BINDING_FILE_DOMAIN_V1",
    "DETAIL_STAGE_ANNOTATION_BUNDLE_FILE_DOMAIN_V1",
    "DETAIL_STAGE_ANNOTATION_BUNDLE_SCHEMA_V1",
    "DETAIL_STAGE_ANNOTATION_MANIFEST_FILE_DOMAIN_V1",
    "DETAIL_STAGE_ANNOTATION_MANIFEST_SCHEMA_V1",
    "DETAIL_STAGE_ANNOTATION_SET_BINDING_SCHEMA_V1",
    "DetailStageAnnotationBundleV1",
    "DetailStageAnnotationContractError",
    "DetailStageAnnotationManifestV1",
    "DetailStageAnnotationSetBindingV1",
    "validate_detail_stage_annotation_set",
    "validate_detail_stage_profile_compatibility",
]
