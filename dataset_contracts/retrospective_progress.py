"""Pure contracts for progress windows migrated after capture.

These contracts describe task semantics asserted over an immutable legacy
Episode.  They deliberately do not manufacture capture-time progress facts:
the source binding, full-sample window, and migration seal remain visibly
retrospective and are hash-bound to the source artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial as _partial
from typing import Any, Mapping

from . import _progress_validation as _validation
from .canonical_encoding import canonical_content_hash


RETROSPECTIVE_VIDEO_BINDING_SCHEMA_V1 = (
    "dexteleop.retrospective_video_binding.v1"
)
RETROSPECTIVE_SOURCE_BINDING_SCHEMA_V1 = (
    "dexteleop.retrospective_source_binding.v1"
)
RETROSPECTIVE_PROGRESS_WINDOW_SCHEMA_V1 = (
    "dexteleop.retrospective_progress_window.v1"
)
RETROSPECTIVE_PROGRESS_MIGRATION_MANIFEST_SCHEMA_V1 = (
    "dexteleop.retrospective_progress_migration_manifest.v1"
)

RETROSPECTIVE_SOURCE_BINDING_FILE_DOMAIN_V1 = (
    "dexteleop.retrospective_source_binding.file.v1"
)
RETROSPECTIVE_PROGRESS_WINDOW_FILE_DOMAIN_V1 = (
    "dexteleop.retrospective_progress_window.file.v1"
)
RETROSPECTIVE_PROGRESS_MIGRATION_MANIFEST_FILE_DOMAIN_V1 = (
    "dexteleop.retrospective_progress_migration_manifest.file.v1"
)

RETROSPECTIVE_PROGRESS_RESOLUTION_KIND_V1 = "RETROSPECTIVE_MIGRATED"
RETROSPECTIVE_DATASET_LOCAL_SOURCE_LOCATOR_V1 = "."
RETROSPECTIVE_ASSERTION_PROVENANCES_V1 = frozenset(
    {"DATASET_OWNER_CONFIRMED"}
)

_VIDEO_HASH_DOMAIN = "dexteleop.retrospective_video_binding.semantic.v1"
_SOURCE_HASH_DOMAIN = "dexteleop.retrospective_source_binding.semantic.v1"
_WINDOW_HASH_DOMAIN = "dexteleop.retrospective_progress_window.semantic.v1"
_MANIFEST_HASH_DOMAIN = (
    "dexteleop.retrospective_progress_migration_manifest.semantic.v1"
)


class RetrospectiveProgressContractError(ValueError):
    """Raised when a migrated progress window is incomplete or ambiguous."""


_required = _partial(_validation.required_text, RetrospectiveProgressContractError)
_sha256 = _partial(_validation.sha256_digest, RetrospectiveProgressContractError)
_non_negative_int = _partial(
    _validation.nonnegative_int, RetrospectiveProgressContractError
)


def _positive_int(value: object, *, name: str) -> int:
    result = _non_negative_int(value, name=name)
    if result == 0:
        raise RetrospectiveProgressContractError(
            f"{name} must be a positive integer."
        )
    return result


_finite = _partial(_validation.finite_number, RetrospectiveProgressContractError)


def _optional_source_index(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, name=name)


def _verify_declared_hash(
    payload: Mapping[str, Any],
    *,
    actual: str,
    name: str,
) -> None:
    declared = payload.get("semantic_content_hash")
    if declared is not None and str(declared) != actual:
        raise RetrospectiveProgressContractError(f"{name} hash mismatch.")


@dataclass(frozen=True)
class RetrospectiveVideoBindingV1:
    """One admitted RGB stream with decoded-tail integrity evidence."""

    modality_id: str
    source_relative_path: str
    content_hash: str
    byte_count: int
    declared_frame_count: int
    decoded_frame_count: int
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=RETROSPECTIVE_VIDEO_BINDING_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in ("modality_id", "source_relative_path"):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "content_hash",
            _sha256(self.content_hash, name="content_hash"),
        )
        object.__setattr__(
            self,
            "byte_count",
            _positive_int(self.byte_count, name="byte_count"),
        )
        declared = _positive_int(
            self.declared_frame_count,
            name="declared_frame_count",
        )
        decoded = _positive_int(
            self.decoded_frame_count,
            name="decoded_frame_count",
        )
        if declared != decoded:
            raise RetrospectiveProgressContractError(
                "An admitted video must have equal declared and decoded "
                "frame counts."
            )
        object.__setattr__(self, "declared_frame_count", declared)
        object.__setattr__(self, "decoded_frame_count", decoded)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_VIDEO_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "modality_id": self.modality_id,
            "source_relative_path": self.source_relative_path,
            "content_hash": self.content_hash,
            "byte_count": self.byte_count,
            "declared_frame_count": self.declared_frame_count,
            "decoded_frame_count": self.decoded_frame_count,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "RetrospectiveVideoBindingV1":
        if payload.get("schema") != RETROSPECTIVE_VIDEO_BINDING_SCHEMA_V1:
            raise RetrospectiveProgressContractError(
                "Unsupported retrospective video-binding schema."
            )
        result = cls(
            modality_id=payload.get("modality_id", ""),
            source_relative_path=payload.get("source_relative_path", ""),
            content_hash=payload.get("content_hash", ""),
            byte_count=payload.get("byte_count", 0),
            declared_frame_count=payload.get("declared_frame_count", 0),
            decoded_frame_count=payload.get("decoded_frame_count", 0),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="Retrospective video binding",
        )
        return result


@dataclass(frozen=True)
class RetrospectiveSourceBindingV1:
    """Bind one external legacy Episode without copying or rewriting it."""

    episode_capture_identity_hash: str
    source_dataset_base_fingerprint: str
    source_dataset_locator: str
    source_schema: str
    source_batch_id: str
    source_episode_index: int
    provenance_source_episode_index: int | None
    source_info_hash: str
    source_episode_record_hash: str
    source_parquet_relative_path: str
    source_parquet_hash: str
    source_parquet_byte_count: int
    admitted_videos: tuple[RetrospectiveVideoBindingV1, ...]
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=RETROSPECTIVE_SOURCE_BINDING_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "episode_capture_identity_hash",
            "source_dataset_base_fingerprint",
            "source_info_hash",
            "source_episode_record_hash",
            "source_parquet_hash",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        for name in (
            "source_dataset_locator",
            "source_schema",
            "source_batch_id",
            "source_parquet_relative_path",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "source_episode_index",
            _non_negative_int(
                self.source_episode_index,
                name="source_episode_index",
            ),
        )
        object.__setattr__(
            self,
            "provenance_source_episode_index",
            _optional_source_index(
                self.provenance_source_episode_index,
                name="provenance_source_episode_index",
            ),
        )
        object.__setattr__(
            self,
            "source_parquet_byte_count",
            _positive_int(
                self.source_parquet_byte_count,
                name="source_parquet_byte_count",
            ),
        )
        videos = tuple(self.admitted_videos)
        if not videos or any(
            not isinstance(item, RetrospectiveVideoBindingV1)
            for item in videos
        ):
            raise RetrospectiveProgressContractError(
                "admitted_videos must contain verified video bindings."
            )
        modality_ids = tuple(item.modality_id for item in videos)
        relative_paths = tuple(item.source_relative_path for item in videos)
        if len(modality_ids) != len(set(modality_ids)):
            raise RetrospectiveProgressContractError(
                "admitted video modality_id values must be unique."
            )
        if len(relative_paths) != len(set(relative_paths)):
            raise RetrospectiveProgressContractError(
                "admitted video source paths must be unique."
            )
        videos = tuple(sorted(videos, key=lambda item: item.modality_id))
        object.__setattr__(self, "admitted_videos", videos)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_SOURCE_HASH_DOMAIN,
            ),
        )

    @property
    def admitted_modality_ids(self) -> tuple[str, ...]:
        return tuple(item.modality_id for item in self.admitted_videos)

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "episode_capture_identity_hash": (
                self.episode_capture_identity_hash
            ),
            "source_dataset_base_fingerprint": (
                self.source_dataset_base_fingerprint
            ),
            "source_dataset_locator": self.source_dataset_locator,
            "source_schema": self.source_schema,
            "source_batch_id": self.source_batch_id,
            "source_episode_index": self.source_episode_index,
            "provenance_source_episode_index": (
                self.provenance_source_episode_index
            ),
            "source_info_hash": self.source_info_hash,
            "source_episode_record_hash": self.source_episode_record_hash,
            "source_parquet_relative_path": self.source_parquet_relative_path,
            "source_parquet_hash": self.source_parquet_hash,
            "source_parquet_byte_count": self.source_parquet_byte_count,
            "admitted_videos": tuple(
                item.to_payload() for item in self.admitted_videos
            ),
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "RetrospectiveSourceBindingV1":
        if payload.get("schema") != RETROSPECTIVE_SOURCE_BINDING_SCHEMA_V1:
            raise RetrospectiveProgressContractError(
                "Unsupported retrospective source-binding schema."
            )
        result = cls(
            episode_capture_identity_hash=payload.get(
                "episode_capture_identity_hash", ""
            ),
            source_dataset_base_fingerprint=payload.get(
                "source_dataset_base_fingerprint", ""
            ),
            source_dataset_locator=payload.get("source_dataset_locator", ""),
            source_schema=payload.get("source_schema", ""),
            source_batch_id=payload.get("source_batch_id", ""),
            source_episode_index=payload.get("source_episode_index", -1),
            provenance_source_episode_index=payload.get(
                "provenance_source_episode_index"
            ),
            source_info_hash=payload.get("source_info_hash", ""),
            source_episode_record_hash=payload.get(
                "source_episode_record_hash", ""
            ),
            source_parquet_relative_path=payload.get(
                "source_parquet_relative_path", ""
            ),
            source_parquet_hash=payload.get("source_parquet_hash", ""),
            source_parquet_byte_count=payload.get(
                "source_parquet_byte_count", 0
            ),
            admitted_videos=tuple(
                RetrospectiveVideoBindingV1.from_payload(item)
                for item in payload.get("admitted_videos", ())
            ),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="Retrospective source binding",
        )
        return result


@dataclass(frozen=True)
class RetrospectiveProgressWindowV1:
    """One owner-confirmed, full-sample successful Progress Step window."""

    episode_capture_identity_hash: str
    source_binding_hash: str
    task_progress_profile_hash: str
    progress_episode_binding_hash: str
    task_stage_profile_hash: str
    sample_axis_hash: str
    sample_count: int
    clock_domain_id: str
    progress_step_id: str
    attempt_id: str
    start_at: float
    end_at: float
    assertion_provenance: str = "DATASET_OWNER_CONFIRMED"
    termination_cause: str = "WINDOW_COMPLETED"
    active_attempt_disposition: str = "SUCCESS"
    full_task_outcome: str = "NOT_APPLICABLE"
    resolution_kind: str = field(
        default=RETROSPECTIVE_PROGRESS_RESOLUTION_KIND_V1,
        init=False,
    )
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=RETROSPECTIVE_PROGRESS_WINDOW_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "episode_capture_identity_hash",
            "source_binding_hash",
            "task_progress_profile_hash",
            "progress_episode_binding_hash",
            "task_stage_profile_hash",
            "sample_axis_hash",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        for name in ("clock_domain_id", "progress_step_id", "attempt_id"):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "sample_count",
            _positive_int(self.sample_count, name="sample_count"),
        )
        start_at = _finite(self.start_at, name="start_at")
        end_at = _finite(self.end_at, name="end_at")
        if end_at < start_at:
            raise RetrospectiveProgressContractError(
                "end_at cannot precede start_at."
            )
        object.__setattr__(self, "start_at", start_at)
        object.__setattr__(self, "end_at", end_at)
        provenance = _required(
            self.assertion_provenance,
            name="assertion_provenance",
        ).upper()
        if provenance not in RETROSPECTIVE_ASSERTION_PROVENANCES_V1:
            raise RetrospectiveProgressContractError(
                f"Unsupported assertion_provenance {provenance!r}."
            )
        object.__setattr__(self, "assertion_provenance", provenance)
        expected = {
            "termination_cause": "WINDOW_COMPLETED",
            "active_attempt_disposition": "SUCCESS",
            "full_task_outcome": "NOT_APPLICABLE",
        }
        for name, required_value in expected.items():
            value = _required(getattr(self, name), name=name).upper()
            if value != required_value:
                raise RetrospectiveProgressContractError(
                    f"{name} must be {required_value!r} for a migrated "
                    "successful Step window."
                )
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_WINDOW_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "resolution_kind": self.resolution_kind,
            "episode_capture_identity_hash": (
                self.episode_capture_identity_hash
            ),
            "source_binding_hash": self.source_binding_hash,
            "task_progress_profile_hash": self.task_progress_profile_hash,
            "progress_episode_binding_hash": (
                self.progress_episode_binding_hash
            ),
            "task_stage_profile_hash": self.task_stage_profile_hash,
            "sample_axis_hash": self.sample_axis_hash,
            "sample_count": self.sample_count,
            "clock_domain_id": self.clock_domain_id,
            "progress_step_id": self.progress_step_id,
            "attempt_id": self.attempt_id,
            "start_at": self.start_at,
            "end_at": self.end_at,
            "assertion_provenance": self.assertion_provenance,
            "termination_cause": self.termination_cause,
            "active_attempt_disposition": (
                self.active_attempt_disposition
            ),
            "full_task_outcome": self.full_task_outcome,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "RetrospectiveProgressWindowV1":
        if payload.get("schema") != RETROSPECTIVE_PROGRESS_WINDOW_SCHEMA_V1:
            raise RetrospectiveProgressContractError(
                "Unsupported retrospective progress-window schema."
            )
        if (
            payload.get("resolution_kind")
            != RETROSPECTIVE_PROGRESS_RESOLUTION_KIND_V1
        ):
            raise RetrospectiveProgressContractError(
                "Retrospective progress window must remain visibly migrated."
            )
        result = cls(
            episode_capture_identity_hash=payload.get(
                "episode_capture_identity_hash", ""
            ),
            source_binding_hash=payload.get("source_binding_hash", ""),
            task_progress_profile_hash=payload.get(
                "task_progress_profile_hash", ""
            ),
            progress_episode_binding_hash=payload.get(
                "progress_episode_binding_hash", ""
            ),
            task_stage_profile_hash=payload.get(
                "task_stage_profile_hash", ""
            ),
            sample_axis_hash=payload.get("sample_axis_hash", ""),
            sample_count=payload.get("sample_count", 0),
            clock_domain_id=payload.get("clock_domain_id", ""),
            progress_step_id=payload.get("progress_step_id", ""),
            attempt_id=payload.get("attempt_id", ""),
            start_at=payload.get("start_at", float("nan")),
            end_at=payload.get("end_at", float("nan")),
            assertion_provenance=payload.get("assertion_provenance", ""),
            termination_cause=payload.get("termination_cause", ""),
            active_attempt_disposition=payload.get(
                "active_attempt_disposition", ""
            ),
            full_task_outcome=payload.get("full_task_outcome", ""),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="Retrospective progress window",
        )
        return result


@dataclass(frozen=True)
class RetrospectiveProgressMigrationManifestV1:
    """Immutable seal joining source truth and one migrated window."""

    episode_capture_identity_hash: str
    source_binding_hash: str
    retrospective_window_hash: str
    task_progress_profile_hash: str
    progress_episode_binding_hash: str
    task_stage_profile_hash: str
    sample_axis_hash: str
    migration_actor_id: str
    migration_created_at: float
    migration_clock_domain_id: str
    resolution_kind: str = field(
        default=RETROSPECTIVE_PROGRESS_RESOLUTION_KIND_V1,
        init=False,
    )
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=RETROSPECTIVE_PROGRESS_MIGRATION_MANIFEST_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "episode_capture_identity_hash",
            "source_binding_hash",
            "retrospective_window_hash",
            "task_progress_profile_hash",
            "progress_episode_binding_hash",
            "task_stage_profile_hash",
            "sample_axis_hash",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        for name in ("migration_actor_id", "migration_clock_domain_id"):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "migration_created_at",
            _finite(self.migration_created_at, name="migration_created_at"),
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
            "resolution_kind": self.resolution_kind,
            "episode_capture_identity_hash": (
                self.episode_capture_identity_hash
            ),
            "source_binding_hash": self.source_binding_hash,
            "retrospective_window_hash": self.retrospective_window_hash,
            "task_progress_profile_hash": self.task_progress_profile_hash,
            "progress_episode_binding_hash": (
                self.progress_episode_binding_hash
            ),
            "task_stage_profile_hash": self.task_stage_profile_hash,
            "sample_axis_hash": self.sample_axis_hash,
            "migration_actor_id": self.migration_actor_id,
            "migration_created_at": self.migration_created_at,
            "migration_clock_domain_id": self.migration_clock_domain_id,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "RetrospectiveProgressMigrationManifestV1":
        if (
            payload.get("schema")
            != RETROSPECTIVE_PROGRESS_MIGRATION_MANIFEST_SCHEMA_V1
        ):
            raise RetrospectiveProgressContractError(
                "Unsupported retrospective migration-manifest schema."
            )
        if (
            payload.get("resolution_kind")
            != RETROSPECTIVE_PROGRESS_RESOLUTION_KIND_V1
        ):
            raise RetrospectiveProgressContractError(
                "Retrospective migration manifest must remain visibly migrated."
            )
        result = cls(
            episode_capture_identity_hash=payload.get(
                "episode_capture_identity_hash", ""
            ),
            source_binding_hash=payload.get("source_binding_hash", ""),
            retrospective_window_hash=payload.get(
                "retrospective_window_hash", ""
            ),
            task_progress_profile_hash=payload.get(
                "task_progress_profile_hash", ""
            ),
            progress_episode_binding_hash=payload.get(
                "progress_episode_binding_hash", ""
            ),
            task_stage_profile_hash=payload.get(
                "task_stage_profile_hash", ""
            ),
            sample_axis_hash=payload.get("sample_axis_hash", ""),
            migration_actor_id=payload.get("migration_actor_id", ""),
            migration_created_at=payload.get(
                "migration_created_at", float("nan")
            ),
            migration_clock_domain_id=payload.get(
                "migration_clock_domain_id", ""
            ),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="Retrospective migration manifest",
        )
        return result


__all__ = [
    "RETROSPECTIVE_DATASET_LOCAL_SOURCE_LOCATOR_V1",
    "RETROSPECTIVE_ASSERTION_PROVENANCES_V1",
    "RETROSPECTIVE_PROGRESS_MIGRATION_MANIFEST_FILE_DOMAIN_V1",
    "RETROSPECTIVE_PROGRESS_MIGRATION_MANIFEST_SCHEMA_V1",
    "RETROSPECTIVE_PROGRESS_RESOLUTION_KIND_V1",
    "RETROSPECTIVE_PROGRESS_WINDOW_FILE_DOMAIN_V1",
    "RETROSPECTIVE_PROGRESS_WINDOW_SCHEMA_V1",
    "RETROSPECTIVE_SOURCE_BINDING_FILE_DOMAIN_V1",
    "RETROSPECTIVE_SOURCE_BINDING_SCHEMA_V1",
    "RETROSPECTIVE_VIDEO_BINDING_SCHEMA_V1",
    "RetrospectiveProgressContractError",
    "RetrospectiveProgressMigrationManifestV1",
    "RetrospectiveProgressWindowV1",
    "RetrospectiveSourceBindingV1",
    "RetrospectiveVideoBindingV1",
]
