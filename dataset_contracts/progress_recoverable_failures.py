"""Additive contracts for post-capture recoverable-failure observations.

These observations describe a temporary loss of task execution followed by
recovery inside one already-successful stage occurrence.  They do not change
the capture Failure fact, stage outcome, Retry structure, or training
exclusions.  Sample indices are authoritative; timestamps are bound audit
coordinates checked by the dataset interface before publication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial as _partial
from typing import Any, Mapping

from . import _progress_validation as _validation
from .canonical_encoding import canonical_content_hash


RECOVERABLE_FAILURE_ANNOTATION_SCHEMA_V1 = (
    "dexteleop.recoverable_failure_annotation.v1"
)
RECOVERABLE_FAILURE_SET_SCHEMA_V1 = (
    "dexteleop.recoverable_failure_annotation_set.v1"
)
RECOVERABLE_FAILURE_FILE_DOMAIN_V1 = (
    "dexteleop.recoverable_failure_annotation.file.v1"
)

_ANNOTATION_HASH_DOMAIN = (
    "dexteleop.recoverable_failure_annotation.semantic.v1"
)
_SET_HASH_DOMAIN = (
    "dexteleop.recoverable_failure_annotation_set.semantic.v1"
)


class RecoverableFailureContractError(ValueError):
    """Raised when a recoverable-failure artifact is ambiguous or malformed."""


_required = _partial(_validation.required_text, RecoverableFailureContractError)
_sha256 = _partial(_validation.sha256_digest, RecoverableFailureContractError)
_finite = _partial(_validation.finite_number, RecoverableFailureContractError)
_nonnegative_int = _partial(
    _validation.nonnegative_int, RecoverableFailureContractError
)


def _optional_note(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RecoverableFailureContractError("note must be a string or null.")
    normalized = value.strip()
    return normalized or None


def _verify_declared_hash(
    payload: Mapping[str, Any],
    *,
    actual: str,
    name: str,
) -> None:
    declared = payload.get("semantic_content_hash")
    if declared is not None and str(declared) != actual:
        raise RecoverableFailureContractError(f"{name} semantic hash mismatch.")


@dataclass(frozen=True)
class RecoverableFailureAnnotationV1:
    """One discrete temporary failure-and-recovery interval in a success stage."""

    annotation_id: str
    stage_occurrence_annotation_id: str
    coarse_attempt_id: str
    progress_step_id: str
    stage_id: str
    onset_sample_index: int
    recovered_sample_index: int
    onset_at: float
    recovered_at: float
    clock_domain_id: str
    note: str | None = None
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=RECOVERABLE_FAILURE_ANNOTATION_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "annotation_id",
            "stage_occurrence_annotation_id",
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
        onset_index = _nonnegative_int(
            self.onset_sample_index,
            name="onset_sample_index",
        )
        recovered_index = _nonnegative_int(
            self.recovered_sample_index,
            name="recovered_sample_index",
        )
        if recovered_index <= onset_index:
            raise RecoverableFailureContractError(
                "recovered_sample_index must be later than onset_sample_index."
            )
        onset_at = _finite(self.onset_at, name="onset_at")
        recovered_at = _finite(self.recovered_at, name="recovered_at")
        if recovered_at <= onset_at:
            raise RecoverableFailureContractError(
                "recovered_at must be later than onset_at."
            )
        object.__setattr__(self, "onset_sample_index", onset_index)
        object.__setattr__(self, "recovered_sample_index", recovered_index)
        object.__setattr__(self, "onset_at", onset_at)
        object.__setattr__(self, "recovered_at", recovered_at)
        object.__setattr__(self, "note", _optional_note(self.note))
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
            "stage_occurrence_annotation_id": (
                self.stage_occurrence_annotation_id
            ),
            "coarse_attempt_id": self.coarse_attempt_id,
            "progress_step_id": self.progress_step_id,
            "stage_id": self.stage_id,
            "onset_sample_index": self.onset_sample_index,
            "recovered_sample_index": self.recovered_sample_index,
            "onset_at": self.onset_at,
            "recovered_at": self.recovered_at,
            "clock_domain_id": self.clock_domain_id,
            "note": self.note,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "RecoverableFailureAnnotationV1":
        if payload.get("schema") != RECOVERABLE_FAILURE_ANNOTATION_SCHEMA_V1:
            raise RecoverableFailureContractError(
                "Unsupported recoverable-failure annotation schema."
            )
        result = cls(
            annotation_id=payload.get("annotation_id", ""),
            stage_occurrence_annotation_id=payload.get(
                "stage_occurrence_annotation_id",
                "",
            ),
            coarse_attempt_id=payload.get("coarse_attempt_id", ""),
            progress_step_id=payload.get("progress_step_id", ""),
            stage_id=payload.get("stage_id", ""),
            onset_sample_index=payload.get("onset_sample_index", -1),
            recovered_sample_index=payload.get("recovered_sample_index", -1),
            onset_at=payload.get("onset_at", float("nan")),
            recovered_at=payload.get("recovered_at", float("nan")),
            clock_domain_id=payload.get("clock_domain_id", ""),
            note=payload.get("note"),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="recoverable-failure annotation",
        )
        return result


@dataclass(frozen=True)
class RecoverableFailureAnnotationSetV1:
    """Immutable optional sidecar bound to one sealed stage annotation set."""

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
    annotations: tuple[RecoverableFailureAnnotationV1, ...] = ()
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=RECOVERABLE_FAILURE_SET_SCHEMA_V1,
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
            raise RecoverableFailureContractError(
                "Recoverable-failure annotation set IDs must agree."
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
            not isinstance(item, RecoverableFailureAnnotationV1)
            for item in annotations
        ):
            raise RecoverableFailureContractError(
                "annotations must contain RecoverableFailureAnnotationV1 values."
            )
        if any(
            item.clock_domain_id != self.bound_clock_domain_id
            for item in annotations
        ):
            raise RecoverableFailureContractError(
                "Every recoverable-failure annotation must use the bound clock."
            )
        annotation_ids = tuple(item.annotation_id for item in annotations)
        if len(annotation_ids) != len(set(annotation_ids)):
            raise RecoverableFailureContractError(
                "Recoverable-failure annotation IDs must be unique."
            )
        ordered = tuple(
            sorted(
                annotations,
                key=lambda item: (
                    item.onset_sample_index,
                    item.recovered_sample_index,
                    item.annotation_id,
                ),
            )
        )
        by_occurrence: dict[str, list[RecoverableFailureAnnotationV1]] = {}
        for item in ordered:
            by_occurrence.setdefault(
                item.stage_occurrence_annotation_id,
                [],
            ).append(item)
        for values in by_occurrence.values():
            for previous, current in zip(values, values[1:]):
                if (
                    current.onset_sample_index
                    < previous.recovered_sample_index
                ):
                    raise RecoverableFailureContractError(
                        "Recoverable-failure intervals cannot overlap in one "
                        "stage occurrence."
                    )
        object.__setattr__(self, "annotations", ordered)
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
    ) -> "RecoverableFailureAnnotationSetV1":
        if payload.get("schema") != RECOVERABLE_FAILURE_SET_SCHEMA_V1:
            raise RecoverableFailureContractError(
                "Unsupported recoverable-failure annotation-set schema."
            )
        raw_annotations = payload.get("annotations", ())
        if not isinstance(raw_annotations, (tuple, list)) or any(
            not isinstance(item, Mapping) for item in raw_annotations
        ):
            raise RecoverableFailureContractError(
                "Recoverable-failure annotations must be a sequence of objects."
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
                RecoverableFailureAnnotationV1.from_payload(item)
                for item in raw_annotations
            ),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="recoverable-failure annotation set",
        )
        return result


__all__ = [
    "RECOVERABLE_FAILURE_ANNOTATION_SCHEMA_V1",
    "RECOVERABLE_FAILURE_FILE_DOMAIN_V1",
    "RECOVERABLE_FAILURE_SET_SCHEMA_V1",
    "RecoverableFailureAnnotationSetV1",
    "RecoverableFailureAnnotationV1",
    "RecoverableFailureContractError",
]
