"""Additive contracts for capture-failure onset refinement.

The original Progress Failure fact and stage annotation bundle remain the
source of truth for capture outcome and stage coverage.  These contracts only
record a discrete, post-capture refinement of the onset of a captured Failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from .canonical_encoding import canonical_content_hash


FAILURE_TRIGGER_ANNOTATION_SCHEMA_V1 = (
    "dexteleop.failure_trigger_annotation.v1"
)
FAILURE_TRIGGER_SET_SCHEMA_V1 = "dexteleop.failure_trigger_annotation_set.v1"
FAILURE_TRIGGER_FILE_DOMAIN_V1 = "dexteleop.failure_trigger_annotation.file.v1"
FAILURE_TRIGGER_CAPTURE_RECORD_SCHEMA_V1 = (
    "dexteleop.failure_trigger_capture_record.v1"
)
FAILURE_TRIGGER_CAPTURE_FILE_DOMAIN_V1 = (
    "dexteleop.failure_trigger_capture_record.file.v1"
)
FAILURE_TRIGGER_CAPTURE_PATH_DOMAIN_V1 = (
    "dexteleop.failure_trigger_capture_record.path.v1"
)
FAILURE_TRIGGER_SOURCES_V1 = frozenset(
    {
        "CAPTURE_DEFAULT",
        "CAPTURE_EXPLICIT",
        "POST_CAPTURE_REFINED",
    }
)
FAILURE_TRIGGER_CAPTURE_MODES_V1 = frozenset({"DEFAULT_FAILURE_TIME"})

_ANNOTATION_HASH_DOMAIN = (
    "dexteleop.failure_trigger_annotation.semantic.v1"
)
_SET_HASH_DOMAIN = "dexteleop.failure_trigger_annotation_set.semantic.v1"
_CAPTURE_RECORD_HASH_DOMAIN = (
    "dexteleop.failure_trigger_capture_record.semantic.v1"
)


class FailureTriggerContractError(ValueError):
    """Raised when a failure-trigger artifact is ambiguous or malformed."""


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FailureTriggerContractError(
            f"{name} must be a non-empty string."
        )
    return value.strip()


def _sha256(value: object, *, name: str) -> str:
    result = _required(value, name=name).lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise FailureTriggerContractError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return result


def _finite(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise FailureTriggerContractError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise FailureTriggerContractError(f"{name} must be finite.")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FailureTriggerContractError(
            f"{name} must be a non-negative integer."
        )
    return value


def _confidence(value: object) -> float:
    result = _finite(value, name="confidence")
    if result < 0.0 or result > 1.0:
        raise FailureTriggerContractError(
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
        raise FailureTriggerContractError(
            f"{name} semantic hash mismatch."
        )


@dataclass(frozen=True)
class FailureTriggerAnnotationV1:
    """One discrete onset marker for one captured Failure fact.

    The three sample indices are the authoritative discrete coordinates.  The
    corresponding ``*_at`` values are retained for readable audit payloads and
    must be checked against the bound Episode sample axis by the dataset
    interface before publication.
    """

    annotation_id: str
    source_failure_fact_id: str
    coarse_attempt_id: str
    progress_step_id: str
    stage_id: str
    lower_bound_sample_index: int
    upper_bound_sample_index: int
    trigger_sample_index: int
    lower_bound_at: float
    upper_bound_at: float
    trigger_at: float
    clock_domain_id: str
    confidence: float = 1.0
    source: str = "CAPTURE_DEFAULT"
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=FAILURE_TRIGGER_ANNOTATION_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "annotation_id",
            "source_failure_fact_id",
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
        for name in (
            "lower_bound_sample_index",
            "upper_bound_sample_index",
            "trigger_sample_index",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        if self.lower_bound_sample_index > self.upper_bound_sample_index:
            raise FailureTriggerContractError(
                "Failure-trigger sample bounds must be ordered."
            )
        if not (
            self.lower_bound_sample_index
            <= self.trigger_sample_index
            <= self.upper_bound_sample_index
        ):
            raise FailureTriggerContractError(
                "trigger_sample_index must lie inside its eligible window."
            )
        lower = _finite(self.lower_bound_at, name="lower_bound_at")
        upper = _finite(self.upper_bound_at, name="upper_bound_at")
        trigger = _finite(self.trigger_at, name="trigger_at")
        if upper < lower:
            raise FailureTriggerContractError(
                "Failure-trigger time bounds must be ordered."
            )
        if not lower <= trigger <= upper:
            raise FailureTriggerContractError(
                "trigger_at must lie inside its eligible time window."
            )
        source = _required(self.source, name="source").upper()
        if source not in FAILURE_TRIGGER_SOURCES_V1:
            raise FailureTriggerContractError(
                f"Unsupported failure-trigger source {source!r}."
            )
        object.__setattr__(self, "lower_bound_at", lower)
        object.__setattr__(self, "upper_bound_at", upper)
        object.__setattr__(self, "trigger_at", trigger)
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_ANNOTATION_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "annotation_id": self.annotation_id,
            "source_failure_fact_id": self.source_failure_fact_id,
            "coarse_attempt_id": self.coarse_attempt_id,
            "progress_step_id": self.progress_step_id,
            "stage_id": self.stage_id,
            "lower_bound_sample_index": self.lower_bound_sample_index,
            "upper_bound_sample_index": self.upper_bound_sample_index,
            "trigger_sample_index": self.trigger_sample_index,
            "lower_bound_at": self.lower_bound_at,
            "upper_bound_at": self.upper_bound_at,
            "trigger_at": self.trigger_at,
            "clock_domain_id": self.clock_domain_id,
            "confidence": self.confidence,
            "source": self.source,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "FailureTriggerAnnotationV1":
        if payload.get("schema") != FAILURE_TRIGGER_ANNOTATION_SCHEMA_V1:
            raise FailureTriggerContractError(
                "Unsupported failure-trigger annotation schema."
            )
        result = cls(
            annotation_id=payload.get("annotation_id", ""),
            source_failure_fact_id=payload.get(
                "source_failure_fact_id",
                "",
            ),
            coarse_attempt_id=payload.get("coarse_attempt_id", ""),
            progress_step_id=payload.get("progress_step_id", ""),
            stage_id=payload.get("stage_id", ""),
            lower_bound_sample_index=payload.get(
                "lower_bound_sample_index",
                -1,
            ),
            upper_bound_sample_index=payload.get(
                "upper_bound_sample_index",
                -1,
            ),
            trigger_sample_index=payload.get("trigger_sample_index", -1),
            lower_bound_at=payload.get("lower_bound_at", float("nan")),
            upper_bound_at=payload.get("upper_bound_at", float("nan")),
            trigger_at=payload.get("trigger_at", float("nan")),
            clock_domain_id=payload.get("clock_domain_id", ""),
            confidence=payload.get("confidence", float("nan")),
            source=payload.get("source", ""),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="failure-trigger annotation",
        )
        return result


@dataclass(frozen=True)
class FailureTriggerAnnotationSetV1:
    """Immutable trigger sidecar bound to one stage annotation set."""

    annotation_set_id: str
    stage_annotation_set_id: str
    episode_capture_identity_hash: str
    source_progress_manifest_hash: str
    task_stage_profile_hash: str
    stage_annotation_bundle_hash: str
    sample_axis_hash: str
    bound_clock_domain_id: str
    annotator_id: str
    created_at: float
    created_clock_domain_id: str
    annotations: tuple[FailureTriggerAnnotationV1, ...] = ()
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=FAILURE_TRIGGER_SET_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "annotation_set_id",
            "stage_annotation_set_id",
            "bound_clock_domain_id",
            "annotator_id",
            "created_clock_domain_id",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        if self.annotation_set_id != self.stage_annotation_set_id:
            raise FailureTriggerContractError(
                "Failure-trigger annotation set IDs must agree."
            )
        for name in (
            "episode_capture_identity_hash",
            "source_progress_manifest_hash",
            "task_stage_profile_hash",
            "stage_annotation_bundle_hash",
            "sample_axis_hash",
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
        annotations = tuple(self.annotations)
        if any(
            not isinstance(item, FailureTriggerAnnotationV1)
            for item in annotations
        ):
            raise FailureTriggerContractError(
                "annotations must contain FailureTriggerAnnotationV1 values."
            )
        annotation_ids = tuple(item.annotation_id for item in annotations)
        if len(annotation_ids) != len(set(annotation_ids)):
            raise FailureTriggerContractError(
                "Failure-trigger annotation IDs must be unique."
            )
        failure_ids = tuple(
            item.source_failure_fact_id for item in annotations
        )
        if len(failure_ids) != len(set(failure_ids)):
            raise FailureTriggerContractError(
                "One trigger is allowed per captured Failure fact."
            )
        if any(
            item.clock_domain_id != self.bound_clock_domain_id
            for item in annotations
        ):
            raise FailureTriggerContractError(
                "Every failure-trigger annotation must use the bound clock."
            )
        object.__setattr__(
            self,
            "annotations",
            tuple(
                sorted(
                    annotations,
                    key=lambda item: (
                        item.upper_bound_at,
                        item.source_failure_fact_id,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_SET_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "annotation_set_id": self.annotation_set_id,
            "stage_annotation_set_id": self.stage_annotation_set_id,
            "episode_capture_identity_hash": self.episode_capture_identity_hash,
            "source_progress_manifest_hash": self.source_progress_manifest_hash,
            "task_stage_profile_hash": self.task_stage_profile_hash,
            "stage_annotation_bundle_hash": self.stage_annotation_bundle_hash,
            "sample_axis_hash": self.sample_axis_hash,
            "bound_clock_domain_id": self.bound_clock_domain_id,
            "annotator_id": self.annotator_id,
            "created_at": self.created_at,
            "created_clock_domain_id": self.created_clock_domain_id,
            "annotations": tuple(item.to_payload() for item in self.annotations),
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "FailureTriggerAnnotationSetV1":
        if payload.get("schema") != FAILURE_TRIGGER_SET_SCHEMA_V1:
            raise FailureTriggerContractError(
                "Unsupported failure-trigger annotation-set schema."
            )
        raw_annotations = payload.get("annotations", ())
        if not isinstance(raw_annotations, (tuple, list)):
            raise FailureTriggerContractError(
                "Failure-trigger annotations must be a sequence."
            )
        if any(not isinstance(item, Mapping) for item in raw_annotations):
            raise FailureTriggerContractError(
                "Failure-trigger entries must be mappings."
            )
        result = cls(
            annotation_set_id=payload.get("annotation_set_id", ""),
            stage_annotation_set_id=payload.get(
                "stage_annotation_set_id",
                "",
            ),
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
            stage_annotation_bundle_hash=payload.get(
                "stage_annotation_bundle_hash",
                "",
            ),
            sample_axis_hash=payload.get("sample_axis_hash", ""),
            bound_clock_domain_id=payload.get("bound_clock_domain_id", ""),
            annotator_id=payload.get("annotator_id", ""),
            created_at=payload.get("created_at", float("nan")),
            created_clock_domain_id=payload.get(
                "created_clock_domain_id",
                "",
            ),
            annotations=tuple(
                FailureTriggerAnnotationV1.from_payload(item)
                for item in raw_annotations
            ),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="failure-trigger annotation set",
        )
        return result


@dataclass(frozen=True)
class FailureTriggerCaptureRecordV1:
    """Optional capture-time receipt; it never changes the Failure fact."""

    source_failure_fact_id: str
    coarse_attempt_id: str
    progress_step_id: str
    recorded_at: float
    clock_domain_id: str
    capture_mode: str = "DEFAULT_FAILURE_TIME"
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=FAILURE_TRIGGER_CAPTURE_RECORD_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "source_failure_fact_id",
            "coarse_attempt_id",
            "progress_step_id",
            "clock_domain_id",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "recorded_at",
            _finite(self.recorded_at, name="recorded_at"),
        )
        mode = _required(self.capture_mode, name="capture_mode").upper()
        if mode not in FAILURE_TRIGGER_CAPTURE_MODES_V1:
            raise FailureTriggerContractError(
                f"Unsupported failure-trigger capture mode {mode!r}."
            )
        object.__setattr__(self, "capture_mode", mode)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_CAPTURE_RECORD_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "source_failure_fact_id": self.source_failure_fact_id,
            "coarse_attempt_id": self.coarse_attempt_id,
            "progress_step_id": self.progress_step_id,
            "recorded_at": self.recorded_at,
            "clock_domain_id": self.clock_domain_id,
            "capture_mode": self.capture_mode,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "FailureTriggerCaptureRecordV1":
        if payload.get("schema") != FAILURE_TRIGGER_CAPTURE_RECORD_SCHEMA_V1:
            raise FailureTriggerContractError(
                "Unsupported failure-trigger capture record schema."
            )
        result = cls(
            source_failure_fact_id=payload.get(
                "source_failure_fact_id",
                "",
            ),
            coarse_attempt_id=payload.get("coarse_attempt_id", ""),
            progress_step_id=payload.get("progress_step_id", ""),
            recorded_at=payload.get("recorded_at", float("nan")),
            clock_domain_id=payload.get("clock_domain_id", ""),
            capture_mode=payload.get("capture_mode", ""),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="failure-trigger capture record",
        )
        return result


__all__ = [
    "FAILURE_TRIGGER_ANNOTATION_SCHEMA_V1",
    "FAILURE_TRIGGER_CAPTURE_FILE_DOMAIN_V1",
    "FAILURE_TRIGGER_CAPTURE_MODES_V1",
    "FAILURE_TRIGGER_CAPTURE_PATH_DOMAIN_V1",
    "FAILURE_TRIGGER_CAPTURE_RECORD_SCHEMA_V1",
    "FAILURE_TRIGGER_FILE_DOMAIN_V1",
    "FAILURE_TRIGGER_SET_SCHEMA_V1",
    "FAILURE_TRIGGER_SOURCES_V1",
    "FailureTriggerAnnotationV1",
    "FailureTriggerAnnotationSetV1",
    "FailureTriggerCaptureRecordV1",
    "FailureTriggerContractError",
]
