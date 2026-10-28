"""Minimal immutable contracts for sealed progress-refinement annotations."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial as _partial
from typing import Any, Mapping, Sequence

from . import _progress_validation as _validation
from .canonical_encoding import canonical_content_hash
from .progress_facts import ProgressEpisodeManifestV1
from .progress_refinement import TaskRefinementProfileV1


REFINEMENT_BOUNDARY_ANNOTATION_SCHEMA_V1 = (
    "dexteleop.refinement_boundary_annotation.v1"
)
REFINEMENT_ANNOTATION_SET_BINDING_SCHEMA_V1 = (
    "dexteleop.refinement_annotation_set_binding.v1"
)
REFINEMENT_ANNOTATION_CHUNK_SCHEMA_V1 = (
    "dexteleop.refinement_annotation_chunk.v1"
)
REFINEMENT_ANNOTATION_MANIFEST_SCHEMA_V1 = (
    "dexteleop.refinement_annotation_manifest.v1"
)

REFINEMENT_ANNOTATION_BINDING_FILE_DOMAIN_V1 = (
    "dexteleop.refinement_annotation_binding.file.v1"
)
REFINEMENT_ANNOTATION_CHUNK_FILE_DOMAIN_V1 = (
    "dexteleop.refinement_annotation_chunk.file.v1"
)
REFINEMENT_ANNOTATION_MANIFEST_FILE_DOMAIN_V1 = (
    "dexteleop.refinement_annotation_manifest.file.v1"
)

_ANNOTATION_HASH_DOMAIN = (
    "dexteleop.refinement_boundary_annotation.semantic.v1"
)
_BINDING_HASH_DOMAIN = (
    "dexteleop.refinement_annotation_set_binding.semantic.v1"
)
_CHUNK_HASH_DOMAIN = "dexteleop.refinement_annotation_chunk.semantic.v1"
_MANIFEST_HASH_DOMAIN = (
    "dexteleop.refinement_annotation_manifest.semantic.v1"
)
_ANNOTATION_BUNDLE_HASH_DOMAIN = (
    "dexteleop.refinement_annotation_bundle.semantic.v1"
)


class ProgressAnnotationContractError(ValueError):
    """Raised when a refinement annotation artifact is ambiguous or invalid."""


_required = _partial(_validation.required_text, ProgressAnnotationContractError)
_sha256 = _partial(_validation.sha256_digest, ProgressAnnotationContractError)
_finite = _partial(_validation.finite_number, ProgressAnnotationContractError)
_confidence = _partial(_validation.confidence, ProgressAnnotationContractError)
_nonnegative_int = _partial(
    _validation.nonnegative_int, ProgressAnnotationContractError
)


def _verify_declared_hash(
    payload: Mapping[str, Any],
    *,
    actual: str,
    name: str,
) -> None:
    declared = payload.get("semantic_content_hash")
    if declared is not None and str(declared) != actual:
        raise ProgressAnnotationContractError(
            f"{name} semantic hash mismatch."
        )


@dataclass(frozen=True)
class RefinementBoundaryAnnotationV1:
    """One immutable observed boundary inside a coarse Attempt."""

    annotation_id: str
    coarse_attempt_id: str
    refinement_boundary_id: str
    effective_at: float
    clock_domain_id: str
    confidence: float
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=REFINEMENT_BOUNDARY_ANNOTATION_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "annotation_id",
            "coarse_attempt_id",
            "refinement_boundary_id",
            "clock_domain_id",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "effective_at",
            _finite(self.effective_at, name="effective_at"),
        )
        object.__setattr__(
            self,
            "confidence",
            _confidence(self.confidence),
        )
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
            "coarse_attempt_id": self.coarse_attempt_id,
            "refinement_boundary_id": self.refinement_boundary_id,
            "effective_at": self.effective_at,
            "clock_domain_id": self.clock_domain_id,
            "confidence": self.confidence,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "RefinementBoundaryAnnotationV1":
        if payload.get("schema") != REFINEMENT_BOUNDARY_ANNOTATION_SCHEMA_V1:
            raise ProgressAnnotationContractError(
                "Unsupported refinement-boundary annotation schema."
            )
        result = cls(
            annotation_id=payload.get("annotation_id", ""),
            coarse_attempt_id=payload.get("coarse_attempt_id", ""),
            refinement_boundary_id=payload.get(
                "refinement_boundary_id",
                "",
            ),
            effective_at=payload.get("effective_at", float("nan")),
            clock_domain_id=payload.get("clock_domain_id", ""),
            confidence=payload.get("confidence", float("nan")),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="refinement-boundary annotation",
        )
        return result


@dataclass(frozen=True)
class RefinementAnnotationSetBindingV1:
    """Immutable identity binding for one post-capture annotation set."""

    annotation_set_id: str
    source_progress_manifest_hash: str
    task_refinement_profile_hash: str
    bound_clock_domain_id: str
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=REFINEMENT_ANNOTATION_SET_BINDING_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "annotation_set_id",
            _required(self.annotation_set_id, name="annotation_set_id"),
        )
        for name in (
            "source_progress_manifest_hash",
            "task_refinement_profile_hash",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "bound_clock_domain_id",
            _required(
                self.bound_clock_domain_id,
                name="bound_clock_domain_id",
            ),
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
            "source_progress_manifest_hash": (
                self.source_progress_manifest_hash
            ),
            "task_refinement_profile_hash": (
                self.task_refinement_profile_hash
            ),
            "bound_clock_domain_id": self.bound_clock_domain_id,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "RefinementAnnotationSetBindingV1":
        if (
            payload.get("schema")
            != REFINEMENT_ANNOTATION_SET_BINDING_SCHEMA_V1
        ):
            raise ProgressAnnotationContractError(
                "Unsupported refinement annotation-set binding schema."
            )
        result = cls(
            annotation_set_id=payload.get("annotation_set_id", ""),
            source_progress_manifest_hash=payload.get(
                "source_progress_manifest_hash",
                "",
            ),
            task_refinement_profile_hash=payload.get(
                "task_refinement_profile_hash",
                "",
            ),
            bound_clock_domain_id=payload.get(
                "bound_clock_domain_id",
                "",
            ),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="refinement annotation-set binding",
        )
        return result


@dataclass(frozen=True)
class RefinementAnnotationChunkV1:
    """One immutable, append-ordered physical annotation chunk."""

    chunk_index: int
    first_annotation_index: int
    annotations: tuple[RefinementBoundaryAnnotationV1, ...]
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=REFINEMENT_ANNOTATION_CHUNK_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "chunk_index",
            _nonnegative_int(self.chunk_index, name="chunk_index"),
        )
        object.__setattr__(
            self,
            "first_annotation_index",
            _nonnegative_int(
                self.first_annotation_index,
                name="first_annotation_index",
            ),
        )
        annotations = tuple(self.annotations)
        if not annotations or any(
            not isinstance(item, RefinementBoundaryAnnotationV1)
            for item in annotations
        ):
            raise ProgressAnnotationContractError(
                "Annotation chunks must contain at least one typed annotation."
            )
        annotation_ids = tuple(item.annotation_id for item in annotations)
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ProgressAnnotationContractError(
                "annotation_id values must be unique within a chunk."
            )
        object.__setattr__(self, "annotations", annotations)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_CHUNK_HASH_DOMAIN,
            ),
        )

    @property
    def annotation_count(self) -> int:
        return len(self.annotations)

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "chunk_index": self.chunk_index,
            "first_annotation_index": self.first_annotation_index,
            "annotations": tuple(
                item.to_payload() for item in self.annotations
            ),
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "RefinementAnnotationChunkV1":
        if payload.get("schema") != REFINEMENT_ANNOTATION_CHUNK_SCHEMA_V1:
            raise ProgressAnnotationContractError(
                "Unsupported refinement annotation chunk schema."
            )
        raw_annotations = payload.get("annotations", ())
        if not isinstance(raw_annotations, (tuple, list)):
            raise ProgressAnnotationContractError(
                "Chunk annotations must be a sequence."
            )
        if any(not isinstance(item, Mapping) for item in raw_annotations):
            raise ProgressAnnotationContractError(
                "Chunk annotation entries must be mappings."
            )
        result = cls(
            chunk_index=payload.get("chunk_index", -1),
            first_annotation_index=payload.get(
                "first_annotation_index",
                -1,
            ),
            annotations=tuple(
                RefinementBoundaryAnnotationV1.from_payload(item)
                for item in raw_annotations
            ),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="refinement annotation chunk",
        )
        return result


@dataclass(frozen=True)
class RefinementAnnotationManifestV1:
    """Seal binding an ordered, immutable set of annotation chunks."""

    annotation_set_binding_hash: str
    annotation_count: int
    annotation_bundle_hash: str
    chunk_hashes: tuple[str, ...]
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=REFINEMENT_ANNOTATION_MANIFEST_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "annotation_set_binding_hash",
            _sha256(
                self.annotation_set_binding_hash,
                name="annotation_set_binding_hash",
            ),
        )
        object.__setattr__(
            self,
            "annotation_count",
            _nonnegative_int(
                self.annotation_count,
                name="annotation_count",
            ),
        )
        object.__setattr__(
            self,
            "annotation_bundle_hash",
            _sha256(
                self.annotation_bundle_hash,
                name="annotation_bundle_hash",
            ),
        )
        chunk_hashes = tuple(
            _sha256(value, name="chunk_hashes")
            for value in self.chunk_hashes
        )
        if len(chunk_hashes) != len(set(chunk_hashes)):
            raise ProgressAnnotationContractError(
                "chunk_hashes entries must be unique."
            )
        if self.annotation_count == 0 and chunk_hashes:
            raise ProgressAnnotationContractError(
                "An empty annotation set cannot reference chunks."
            )
        if self.annotation_count > 0 and not chunk_hashes:
            raise ProgressAnnotationContractError(
                "A non-empty annotation set must reference chunks."
            )
        object.__setattr__(self, "chunk_hashes", chunk_hashes)
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
            "annotation_count": self.annotation_count,
            "annotation_bundle_hash": self.annotation_bundle_hash,
            "chunk_hashes": self.chunk_hashes,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "RefinementAnnotationManifestV1":
        if payload.get("schema") != REFINEMENT_ANNOTATION_MANIFEST_SCHEMA_V1:
            raise ProgressAnnotationContractError(
                "Unsupported refinement annotation manifest schema."
            )
        result = cls(
            annotation_set_binding_hash=payload.get(
                "annotation_set_binding_hash",
                "",
            ),
            annotation_count=payload.get("annotation_count", -1),
            annotation_bundle_hash=payload.get(
                "annotation_bundle_hash",
                "",
            ),
            chunk_hashes=tuple(payload.get("chunk_hashes", ())),
        )
        _verify_declared_hash(
            payload,
            actual=result.semantic_content_hash,
            name="refinement annotation manifest",
        )
        return result

    @staticmethod
    def annotation_bundle_hash_for(
        annotations: Sequence[RefinementBoundaryAnnotationV1],
    ) -> str:
        return canonical_content_hash(
            tuple(item.semantic_content_hash for item in annotations),
            domain=_ANNOTATION_BUNDLE_HASH_DOMAIN,
        )


def validate_refinement_annotation_binding(
    *,
    source_progress_manifest: ProgressEpisodeManifestV1,
    refinement_profile: TaskRefinementProfileV1,
    binding: RefinementAnnotationSetBindingV1,
) -> None:
    """Require annotations to bind one sealed progress truth and one profile."""

    if not isinstance(source_progress_manifest, ProgressEpisodeManifestV1):
        raise TypeError(
            "source_progress_manifest must be ProgressEpisodeManifestV1."
        )
    if not isinstance(refinement_profile, TaskRefinementProfileV1):
        raise TypeError("refinement_profile must be TaskRefinementProfileV1.")
    if not isinstance(binding, RefinementAnnotationSetBindingV1):
        raise TypeError(
            "binding must be RefinementAnnotationSetBindingV1."
        )
    if (
        binding.source_progress_manifest_hash
        != source_progress_manifest.semantic_content_hash
    ):
        raise ProgressAnnotationContractError(
            "Annotation binding/source progress manifest hash mismatch."
        )
    if (
        binding.task_refinement_profile_hash
        != refinement_profile.semantic_content_hash
    ):
        raise ProgressAnnotationContractError(
            "Annotation binding/refinement profile hash mismatch."
        )
    if (
        refinement_profile.task_progress_profile_hash
        != source_progress_manifest.task_progress_profile_hash
    ):
        raise ProgressAnnotationContractError(
            "Refinement profile does not describe the sealed progress profile."
        )
    if (
        binding.bound_clock_domain_id
        != source_progress_manifest.sealed_clock_domain_id
    ):
        raise ProgressAnnotationContractError(
            "Annotation binding must use the sealed Episode clock."
        )


__all__ = [
    "REFINEMENT_ANNOTATION_BINDING_FILE_DOMAIN_V1",
    "REFINEMENT_ANNOTATION_CHUNK_FILE_DOMAIN_V1",
    "REFINEMENT_ANNOTATION_CHUNK_SCHEMA_V1",
    "REFINEMENT_ANNOTATION_MANIFEST_FILE_DOMAIN_V1",
    "REFINEMENT_ANNOTATION_MANIFEST_SCHEMA_V1",
    "REFINEMENT_ANNOTATION_SET_BINDING_SCHEMA_V1",
    "REFINEMENT_BOUNDARY_ANNOTATION_SCHEMA_V1",
    "ProgressAnnotationContractError",
    "RefinementAnnotationChunkV1",
    "RefinementAnnotationManifestV1",
    "RefinementAnnotationSetBindingV1",
    "RefinementBoundaryAnnotationV1",
    "validate_refinement_annotation_binding",
]
