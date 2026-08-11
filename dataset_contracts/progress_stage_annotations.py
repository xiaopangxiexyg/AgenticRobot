"""Immutable post-capture stage timeline annotation contracts.

These contracts intentionally preserve capture-time Failure and Attempt facts.
Annotations identify ordered stage occurrences, explicit abnormal intervals,
optional whole-Episode exclusion, and privileged coarse-boundary time
overrides.  They never rewrite raw samples, video, or the source progress log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from .canonical_encoding import canonical_content_hash


STAGE_OCCURRENCE_ANNOTATION_SCHEMA_V1 = (
    "dexteleop.stage_occurrence_annotation.v1"
)
ABNORMAL_INTERVAL_ANNOTATION_SCHEMA_V1 = (
    "dexteleop.abnormal_interval_annotation.v1"
)
EPISODE_ABNORMAL_ANNOTATION_SCHEMA_V1 = (
    "dexteleop.episode_abnormal_annotation.v1"
)
EPISODE_REVIEW_REQUEST_ANNOTATION_SCHEMA_V1 = (
    "dexteleop.episode_review_request_annotation.v1"
)
PROGRESS_BOUNDARY_OVERRIDE_SCHEMA_V1 = (
    "dexteleop.progress_boundary_override_annotation.v1"
)
CAPTURE_END_STEP_ANNOTATION_SCHEMA_V1 = (
    "dexteleop.capture_end_step_annotation.v1"
)
STAGE_ANNOTATION_SET_BINDING_SCHEMA_V1 = (
    "dexteleop.stage_annotation_set_binding.v1"
)
STAGE_ANNOTATION_BUNDLE_SCHEMA_V1 = (
    "dexteleop.stage_annotation_bundle.v1"
)
STAGE_ANNOTATION_MANIFEST_SCHEMA_V1 = (
    "dexteleop.stage_annotation_manifest.v1"
)

STAGE_ANNOTATION_BINDING_FILE_DOMAIN_V1 = (
    "dexteleop.stage_annotation_binding.file.v1"
)
STAGE_ANNOTATION_BUNDLE_FILE_DOMAIN_V1 = (
    "dexteleop.stage_annotation_bundle.file.v1"
)
STAGE_ANNOTATION_MANIFEST_FILE_DOMAIN_V1 = (
    "dexteleop.stage_annotation_manifest.file.v1"
)

STAGE_OCCURRENCE_OUTCOMES_V1 = frozenset(
    {"SUCCESS", "FAILURE", "TRUNCATED_BY_ABNORMAL"}
)
ABNORMAL_REASON_CODES_V1 = frozenset(
    {
        "BAD_CAPTURE",
        "WRONG_TASK_OR_SETUP",
        "DATA_OR_DEVICE_FAILURE",
        "OTHER",
    }
)
STAGE_ANNOTATION_MODES_V1 = frozenset({"STANDARD", "SECOND_REVIEW"})

_STAGE_OCCURRENCE_HASH_DOMAIN = (
    "dexteleop.stage_occurrence_annotation.semantic.v1"
)
_ABNORMAL_INTERVAL_HASH_DOMAIN = (
    "dexteleop.abnormal_interval_annotation.semantic.v1"
)
_EPISODE_ABNORMAL_HASH_DOMAIN = (
    "dexteleop.episode_abnormal_annotation.semantic.v1"
)
_EPISODE_REVIEW_REQUEST_HASH_DOMAIN = (
    "dexteleop.episode_review_request_annotation.semantic.v1"
)
_PROGRESS_BOUNDARY_OVERRIDE_HASH_DOMAIN = (
    "dexteleop.progress_boundary_override_annotation.semantic.v1"
)
_CAPTURE_END_STEP_HASH_DOMAIN = (
    "dexteleop.capture_end_step_annotation.semantic.v1"
)
_BINDING_HASH_DOMAIN = "dexteleop.stage_annotation_set_binding.semantic.v1"
_BUNDLE_HASH_DOMAIN = "dexteleop.stage_annotation_bundle.semantic.v1"
_MANIFEST_HASH_DOMAIN = "dexteleop.stage_annotation_manifest.semantic.v1"


class ProgressStageAnnotationContractError(ValueError):
    """Raised when a stage annotation artifact is ambiguous or malformed."""


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProgressStageAnnotationContractError(
            f"{name} must be a non-empty string."
        )
    return value.strip()


def _optional_text(value: object | None, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProgressStageAnnotationContractError(
            f"{name} must be a string or null."
        )
    text = value.strip()
    return text or None


def _sha256(value: object, *, name: str) -> str:
    result = _required(value, name=name).lower()
    if len(result) != 64 or any(
        char not in "0123456789abcdef" for char in result
    ):
        raise ProgressStageAnnotationContractError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return result


def _finite(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProgressStageAnnotationContractError(
            f"{name} must be numeric."
        )
    result = float(value)
    if not math.isfinite(result):
        raise ProgressStageAnnotationContractError(
            f"{name} must be finite."
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ProgressStageAnnotationContractError(
            f"{name} must be a positive integer."
        )
    return value


def _confidence(value: object) -> float:
    result = _finite(value, name="confidence")
    if result < 0.0 or result > 1.0:
        raise ProgressStageAnnotationContractError(
            "confidence must lie in [0, 1]."
        )
    return result


def _verify_declared_hash(
    payload: Mapping[str, Any],
    *,
    actual: str,
    name: str,
) -> None:
    declared = payload.get("semantic_content_hash")
    if declared is not None and str(declared) != actual:
        raise ProgressStageAnnotationContractError(
            f"{name} semantic hash mismatch."
        )


def capture_end_projected_attempt_id(
    *,
    source_attempt_id: str,
    source_progress_step_id: str,
    progress_step_id: str,
) -> str:
    """Return the deterministic annotation-only Attempt identity for a Step."""

    source_attempt = _required(source_attempt_id, name="source_attempt_id")
    source_step = _required(
        source_progress_step_id,
        name="source_progress_step_id",
    )
    step = _required(progress_step_id, name="progress_step_id")
    if step == source_step:
        return source_attempt
    return f"{source_attempt}:capture-end-step:{step}"


@dataclass(frozen=True)
class StageOccurrenceAnnotationV1:
    """One contiguous stage occurrence contained by one capture Attempt."""

    annotation_id: str
    coarse_attempt_id: str
    progress_step_id: str
    stage_id: str
    occurrence_index_in_attempt: int
    start_at: float
    end_at: float
    clock_domain_id: str
    outcome: str
    confidence: float = 1.0
    source_failure_fact_id: str | None = None
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=STAGE_OCCURRENCE_ANNOTATION_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "annotation_id",
            "coarse_attempt_id",
            "progress_step_id",
            "stage_id",
            "clock_domain_id",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        start = _finite(self.start_at, name="start_at")
        end = _finite(self.end_at, name="end_at")
        if end <= start:
            raise ProgressStageAnnotationContractError(
                "A stage occurrence must have positive duration."
            )
        outcome = _required(self.outcome, name="outcome").upper()
        if outcome not in STAGE_OCCURRENCE_OUTCOMES_V1:
            raise ProgressStageAnnotationContractError(
                f"Unsupported stage occurrence outcome {outcome!r}."
            )
        source_failure = (
            None
            if self.source_failure_fact_id is None
            else _required(
                self.source_failure_fact_id,
                name="source_failure_fact_id",
            )
        )
        if outcome != "FAILURE" and source_failure is not None:
            raise ProgressStageAnnotationContractError(
                "Only a FAILURE occurrence may reference a Failure fact."
            )
        object.__setattr__(
            self,
            "occurrence_index_in_attempt",
            _positive_int(
                self.occurrence_index_in_attempt,
                name="occurrence_index_in_attempt",
            ),
        )
        object.__setattr__(self, "start_at", start)
        object.__setattr__(self, "end_at", end)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "source_failure_fact_id", source_failure)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_STAGE_OCCURRENCE_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "annotation_id": self.annotation_id,
            "coarse_attempt_id": self.coarse_attempt_id,
            "progress_step_id": self.progress_step_id,
            "stage_id": self.stage_id,
            "occurrence_index_in_attempt": self.occurrence_index_in_attempt,
            "start_at": self.start_at,
            "end_at": self.end_at,
            "clock_domain_id": self.clock_domain_id,
            "outcome": self.outcome,
            "confidence": self.confidence,
            "source_failure_fact_id": self.source_failure_fact_id,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "StageOccurrenceAnnotationV1":
        if payload.get("schema") != STAGE_OCCURRENCE_ANNOTATION_SCHEMA_V1:
            raise ProgressStageAnnotationContractError(
                "Unsupported stage occurrence annotation schema."
            )
        result = cls(
            annotation_id=payload.get("annotation_id", ""),
            coarse_attempt_id=payload.get("coarse_attempt_id", ""),
            progress_step_id=payload.get("progress_step_id", ""),
            stage_id=payload.get("stage_id", ""),
            occurrence_index_in_attempt=payload.get(
                "occurrence_index_in_attempt",
                0,
            ),
            start_at=payload.get("start_at", float("nan")),
            end_at=payload.get("end_at", float("nan")),
            clock_domain_id=payload.get("clock_domain_id", ""),
            outcome=payload.get("outcome", ""),
            confidence=payload.get("confidence", float("nan")),
            source_failure_fact_id=payload.get("source_failure_fact_id"),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="stage occurrence",
        )
        return result


@dataclass(frozen=True)
class AbnormalIntervalAnnotationV1:
    """One excluded interval that splits normal ordering validation."""

    annotation_id: str
    start_at: float
    end_at: float
    clock_domain_id: str
    reason_code: str
    note: str | None = None
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=ABNORMAL_INTERVAL_ANNOTATION_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "annotation_id",
            _required(self.annotation_id, name="annotation_id"),
        )
        object.__setattr__(
            self,
            "clock_domain_id",
            _required(self.clock_domain_id, name="clock_domain_id"),
        )
        start = _finite(self.start_at, name="start_at")
        end = _finite(self.end_at, name="end_at")
        if end <= start:
            raise ProgressStageAnnotationContractError(
                "An abnormal interval must have positive duration."
            )
        reason = _required(self.reason_code, name="reason_code").upper()
        if reason not in ABNORMAL_REASON_CODES_V1:
            raise ProgressStageAnnotationContractError(
                f"Unsupported abnormal reason {reason!r}."
            )
        object.__setattr__(self, "start_at", start)
        object.__setattr__(self, "end_at", end)
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(
            self,
            "note",
            _optional_text(self.note, name="note"),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_ABNORMAL_INTERVAL_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "annotation_id": self.annotation_id,
            "start_at": self.start_at,
            "end_at": self.end_at,
            "clock_domain_id": self.clock_domain_id,
            "reason_code": self.reason_code,
            "note": self.note,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "AbnormalIntervalAnnotationV1":
        if payload.get("schema") != ABNORMAL_INTERVAL_ANNOTATION_SCHEMA_V1:
            raise ProgressStageAnnotationContractError(
                "Unsupported abnormal interval annotation schema."
            )
        result = cls(
            annotation_id=payload.get("annotation_id", ""),
            start_at=payload.get("start_at", float("nan")),
            end_at=payload.get("end_at", float("nan")),
            clock_domain_id=payload.get("clock_domain_id", ""),
            reason_code=payload.get("reason_code", ""),
            note=payload.get("note"),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="abnormal interval",
        )
        return result


@dataclass(frozen=True)
class EpisodeAbnormalAnnotationV1:
    """Exclude one complete Episode from default training views."""

    annotation_id: str
    reason_code: str
    note: str | None = None
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=EPISODE_ABNORMAL_ANNOTATION_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "annotation_id",
            _required(self.annotation_id, name="annotation_id"),
        )
        reason = _required(self.reason_code, name="reason_code").upper()
        if reason not in ABNORMAL_REASON_CODES_V1:
            raise ProgressStageAnnotationContractError(
                f"Unsupported abnormal reason {reason!r}."
            )
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(
            self,
            "note",
            _optional_text(self.note, name="note"),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_EPISODE_ABNORMAL_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "annotation_id": self.annotation_id,
            "reason_code": self.reason_code,
            "note": self.note,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "EpisodeAbnormalAnnotationV1":
        if payload.get("schema") != EPISODE_ABNORMAL_ANNOTATION_SCHEMA_V1:
            raise ProgressStageAnnotationContractError(
                "Unsupported Episode abnormal annotation schema."
            )
        result = cls(
            annotation_id=payload.get("annotation_id", ""),
            reason_code=payload.get("reason_code", ""),
            note=payload.get("note"),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="Episode abnormal annotation",
        )
        return result


@dataclass(frozen=True)
class EpisodeReviewRequestAnnotationV1:
    """Mark one Episode as excluded until a second-review set supersedes it.

    This is deliberately a pending label, not an abnormal conclusion.  A
    later ``SECOND_REVIEW`` annotation set clears it by omitting the label;
    immutable earlier sets remain auditable.
    """

    annotation_id: str
    note: str | None = None
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=EPISODE_REVIEW_REQUEST_ANNOTATION_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "annotation_id",
            _required(self.annotation_id, name="annotation_id"),
        )
        object.__setattr__(
            self,
            "note",
            _optional_text(self.note, name="note"),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_EPISODE_REVIEW_REQUEST_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "annotation_id": self.annotation_id,
            "note": self.note,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "EpisodeReviewRequestAnnotationV1":
        if (
            payload.get("schema")
            != EPISODE_REVIEW_REQUEST_ANNOTATION_SCHEMA_V1
        ):
            raise ProgressStageAnnotationContractError(
                "Unsupported Episode review-request annotation schema."
            )
        result = cls(
            annotation_id=payload.get("annotation_id", ""),
            note=payload.get("note"),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="Episode review-request annotation",
        )
        return result


@dataclass(frozen=True)
class ProgressBoundaryOverrideAnnotationV1:
    """Privileged retrospective time override for one immutable Advance fact."""

    annotation_id: str
    source_advance_fact_id: str
    corrected_effective_at: float
    clock_domain_id: str
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=PROGRESS_BOUNDARY_OVERRIDE_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "annotation_id",
            "source_advance_fact_id",
            "clock_domain_id",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "corrected_effective_at",
            _finite(
                self.corrected_effective_at,
                name="corrected_effective_at",
            ),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_PROGRESS_BOUNDARY_OVERRIDE_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "annotation_id": self.annotation_id,
            "source_advance_fact_id": self.source_advance_fact_id,
            "corrected_effective_at": self.corrected_effective_at,
            "clock_domain_id": self.clock_domain_id,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "ProgressBoundaryOverrideAnnotationV1":
        if payload.get("schema") != PROGRESS_BOUNDARY_OVERRIDE_SCHEMA_V1:
            raise ProgressStageAnnotationContractError(
                "Unsupported progress boundary override schema."
            )
        result = cls(
            annotation_id=payload.get("annotation_id", ""),
            source_advance_fact_id=payload.get(
                "source_advance_fact_id",
                "",
            ),
            corrected_effective_at=payload.get(
                "corrected_effective_at",
                float("nan"),
            ),
            clock_domain_id=payload.get("clock_domain_id", ""),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="progress boundary override",
        )
        return result


@dataclass(frozen=True)
class CaptureEndStepAnnotationV1:
    """Post-capture interpretation of an ambiguous final capture stop.

    The referenced terminal fact and source Attempt remain immutable.  This
    annotation only declares which linear Progress Step contains the final
    annotated stage occurrence so fine-stage annotation can project a manually
    stopped capture over the corresponding Step prefix.  The generated draft
    defaults that tail to SUCCESS; later tail-outcome editing remains explicit.
    """

    annotation_id: str
    source_terminal_fact_id: str
    source_attempt_id: str
    end_progress_step_id: str
    clock_domain_id: str
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=CAPTURE_END_STEP_ANNOTATION_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "annotation_id",
            "source_terminal_fact_id",
            "source_attempt_id",
            "end_progress_step_id",
            "clock_domain_id",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_CAPTURE_END_STEP_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "annotation_id": self.annotation_id,
            "source_terminal_fact_id": self.source_terminal_fact_id,
            "source_attempt_id": self.source_attempt_id,
            "end_progress_step_id": self.end_progress_step_id,
            "clock_domain_id": self.clock_domain_id,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "CaptureEndStepAnnotationV1":
        if payload.get("schema") != CAPTURE_END_STEP_ANNOTATION_SCHEMA_V1:
            raise ProgressStageAnnotationContractError(
                "Unsupported capture-end Step annotation schema."
            )
        result = cls(
            annotation_id=payload.get("annotation_id", ""),
            source_terminal_fact_id=payload.get(
                "source_terminal_fact_id",
                "",
            ),
            source_attempt_id=payload.get("source_attempt_id", ""),
            end_progress_step_id=payload.get(
                "end_progress_step_id",
                "",
            ),
            clock_domain_id=payload.get("clock_domain_id", ""),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="capture-end Step annotation",
        )
        return result


@dataclass(frozen=True)
class StageAnnotationSetBindingV1:
    annotation_set_id: str
    episode_capture_identity_hash: str
    source_progress_manifest_hash: str
    task_stage_profile_hash: str
    bound_clock_domain_id: str
    annotator_id: str
    created_at: float
    created_clock_domain_id: str
    annotation_mode: str = "STANDARD"
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=STAGE_ANNOTATION_SET_BINDING_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "annotation_set_id",
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
            "task_stage_profile_hash",
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
        mode = _required(
            self.annotation_mode,
            name="annotation_mode",
        ).upper()
        if mode not in STAGE_ANNOTATION_MODES_V1:
            raise ProgressStageAnnotationContractError(
                f"Unsupported stage annotation mode {mode!r}."
            )
        object.__setattr__(self, "annotation_mode", mode)
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
            "task_stage_profile_hash": self.task_stage_profile_hash,
            "bound_clock_domain_id": self.bound_clock_domain_id,
            "annotator_id": self.annotator_id,
            "created_at": self.created_at,
            "created_clock_domain_id": self.created_clock_domain_id,
        }
        # Omit the default so existing STANDARD V1 artifacts retain their
        # exact semantic hash and remain readable without migration.
        if self.annotation_mode != "STANDARD":
            payload["annotation_mode"] = self.annotation_mode
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "StageAnnotationSetBindingV1":
        if payload.get("schema") != STAGE_ANNOTATION_SET_BINDING_SCHEMA_V1:
            raise ProgressStageAnnotationContractError(
                "Unsupported stage annotation-set binding schema."
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
            task_stage_profile_hash=payload.get(
                "task_stage_profile_hash",
                "",
            ),
            bound_clock_domain_id=payload.get(
                "bound_clock_domain_id",
                "",
            ),
            annotator_id=payload.get("annotator_id", ""),
            created_at=payload.get("created_at", float("nan")),
            created_clock_domain_id=payload.get(
                "created_clock_domain_id",
                "",
            ),
            annotation_mode=payload.get("annotation_mode", "STANDARD"),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="stage annotation-set binding",
        )
        return result


@dataclass(frozen=True)
class StageAnnotationBundleV1:
    """One clean immutable result compiled from an editable UI draft."""

    occurrences: tuple[StageOccurrenceAnnotationV1, ...] = ()
    abnormal_intervals: tuple[AbnormalIntervalAnnotationV1, ...] = ()
    progress_boundary_overrides: tuple[
        ProgressBoundaryOverrideAnnotationV1, ...
    ] = ()
    capture_end_step: CaptureEndStepAnnotationV1 | None = None
    episode_abnormal: EpisodeAbnormalAnnotationV1 | None = None
    episode_review_request: EpisodeReviewRequestAnnotationV1 | None = None
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=STAGE_ANNOTATION_BUNDLE_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        occurrences = tuple(self.occurrences)
        intervals = tuple(self.abnormal_intervals)
        overrides = tuple(self.progress_boundary_overrides)
        if any(
            not isinstance(item, StageOccurrenceAnnotationV1)
            for item in occurrences
        ):
            raise TypeError("occurrences contain an unsupported value.")
        if any(
            not isinstance(item, AbnormalIntervalAnnotationV1)
            for item in intervals
        ):
            raise TypeError("abnormal_intervals contain an unsupported value.")
        if any(
            not isinstance(item, ProgressBoundaryOverrideAnnotationV1)
            for item in overrides
        ):
            raise TypeError(
                "progress_boundary_overrides contain an unsupported value."
            )
        if self.capture_end_step is not None and not isinstance(
            self.capture_end_step,
            CaptureEndStepAnnotationV1,
        ):
            raise TypeError("capture_end_step has an unsupported value.")
        if self.episode_abnormal is not None and not isinstance(
            self.episode_abnormal,
            EpisodeAbnormalAnnotationV1,
        ):
            raise TypeError("episode_abnormal has an unsupported value.")
        if self.episode_review_request is not None and not isinstance(
            self.episode_review_request,
            EpisodeReviewRequestAnnotationV1,
        ):
            raise TypeError(
                "episode_review_request has an unsupported value."
            )
        annotation_ids = tuple(
            item.annotation_id
            for item in (
                occurrences
                + intervals
                + overrides
                + (
                    ()
                    if self.capture_end_step is None
                    else (self.capture_end_step,)
                )
                + (
                    ()
                    if self.episode_abnormal is None
                    else (self.episode_abnormal,)
                )
                + (
                    ()
                    if self.episode_review_request is None
                    else (self.episode_review_request,)
                )
            )
        )
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ProgressStageAnnotationContractError(
                "annotation_id values must be unique within a stage bundle."
            )
        intervals = tuple(sorted(intervals, key=lambda item: item.start_at))
        for earlier, later in zip(intervals, intervals[1:]):
            if later.start_at < earlier.end_at:
                raise ProgressStageAnnotationContractError(
                    "Abnormal intervals cannot overlap."
                )
        object.__setattr__(
            self,
            "occurrences",
            tuple(
                sorted(
                    occurrences,
                    key=lambda item: (
                        item.start_at,
                        item.end_at,
                        item.annotation_id,
                    ),
                )
            ),
        )
        object.__setattr__(self, "abnormal_intervals", intervals)
        object.__setattr__(
            self,
            "progress_boundary_overrides",
            tuple(
                sorted(
                    overrides,
                    key=lambda item: (
                        item.corrected_effective_at,
                        item.source_advance_fact_id,
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
            "abnormal_intervals": tuple(
                item.to_payload() for item in self.abnormal_intervals
            ),
            "progress_boundary_overrides": tuple(
                item.to_payload()
                for item in self.progress_boundary_overrides
            ),
            "episode_abnormal": (
                None
                if self.episode_abnormal is None
                else self.episode_abnormal.to_payload()
            ),
        }
        # Optional and omitted for backwards hash compatibility with existing
        # V1 bundles.
        if self.episode_review_request is not None:
            payload["episode_review_request"] = (
                self.episode_review_request.to_payload()
            )
        if self.capture_end_step is not None:
            payload["capture_end_step"] = self.capture_end_step.to_payload()
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "StageAnnotationBundleV1":
        if payload.get("schema") != STAGE_ANNOTATION_BUNDLE_SCHEMA_V1:
            raise ProgressStageAnnotationContractError(
                "Unsupported stage annotation bundle schema."
            )
        episode_payload = payload.get("episode_abnormal")
        review_payload = payload.get("episode_review_request")
        capture_end_payload = payload.get("capture_end_step")
        result = cls(
            occurrences=tuple(
                StageOccurrenceAnnotationV1.from_payload(item)
                for item in payload.get("occurrences", ())
            ),
            abnormal_intervals=tuple(
                AbnormalIntervalAnnotationV1.from_payload(item)
                for item in payload.get("abnormal_intervals", ())
            ),
            progress_boundary_overrides=tuple(
                ProgressBoundaryOverrideAnnotationV1.from_payload(item)
                for item in payload.get(
                    "progress_boundary_overrides",
                    (),
                )
            ),
            capture_end_step=(
                None
                if capture_end_payload is None
                else CaptureEndStepAnnotationV1.from_payload(
                    capture_end_payload
                )
            ),
            episode_abnormal=(
                None
                if episode_payload is None
                else EpisodeAbnormalAnnotationV1.from_payload(
                    episode_payload
                )
            ),
            episode_review_request=(
                None
                if review_payload is None
                else EpisodeReviewRequestAnnotationV1.from_payload(
                    review_payload
                )
            ),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="stage annotation bundle",
        )
        return result


@dataclass(frozen=True)
class StageAnnotationManifestV1:
    annotation_set_binding_hash: str
    annotation_bundle_hash: str
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=STAGE_ANNOTATION_MANIFEST_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "annotation_set_binding_hash",
            "annotation_bundle_hash",
        ):
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
            "annotation_set_binding_hash": (
                self.annotation_set_binding_hash
            ),
            "annotation_bundle_hash": self.annotation_bundle_hash,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "StageAnnotationManifestV1":
        if payload.get("schema") != STAGE_ANNOTATION_MANIFEST_SCHEMA_V1:
            raise ProgressStageAnnotationContractError(
                "Unsupported stage annotation manifest schema."
            )
        result = cls(
            annotation_set_binding_hash=payload.get(
                "annotation_set_binding_hash",
                "",
            ),
            annotation_bundle_hash=payload.get(
                "annotation_bundle_hash",
                "",
            ),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="stage annotation manifest",
        )
        return result


__all__ = [
    "ABNORMAL_INTERVAL_ANNOTATION_SCHEMA_V1",
    "ABNORMAL_REASON_CODES_V1",
    "CAPTURE_END_STEP_ANNOTATION_SCHEMA_V1",
    "EPISODE_ABNORMAL_ANNOTATION_SCHEMA_V1",
    "PROGRESS_BOUNDARY_OVERRIDE_SCHEMA_V1",
    "STAGE_ANNOTATION_BINDING_FILE_DOMAIN_V1",
    "STAGE_ANNOTATION_BUNDLE_FILE_DOMAIN_V1",
    "STAGE_ANNOTATION_MANIFEST_FILE_DOMAIN_V1",
    "STAGE_OCCURRENCE_ANNOTATION_SCHEMA_V1",
    "STAGE_OCCURRENCE_OUTCOMES_V1",
    "AbnormalIntervalAnnotationV1",
    "CaptureEndStepAnnotationV1",
    "EpisodeAbnormalAnnotationV1",
    "ProgressBoundaryOverrideAnnotationV1",
    "ProgressStageAnnotationContractError",
    "StageAnnotationBundleV1",
    "StageAnnotationManifestV1",
    "StageAnnotationSetBindingV1",
    "StageOccurrenceAnnotationV1",
    "capture_end_projected_attempt_id",
]
