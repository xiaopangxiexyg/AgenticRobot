"""Append-only refinement annotation persistence for a sealed progress Episode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Sequence

from dataset_contracts.canonical_encoding import canonical_loads
from dataset_contracts.progress_annotations import (
    REFINEMENT_ANNOTATION_BINDING_FILE_DOMAIN_V1,
    REFINEMENT_ANNOTATION_CHUNK_FILE_DOMAIN_V1,
    REFINEMENT_ANNOTATION_MANIFEST_FILE_DOMAIN_V1,
    RefinementAnnotationChunkV1,
    RefinementAnnotationManifestV1,
    RefinementAnnotationSetBindingV1,
    RefinementBoundaryAnnotationV1,
    validate_refinement_annotation_binding,
)
from dataset_contracts.progress_facts import ProgressEpisodeManifestV1
from dataset_contracts.progress_refinement import TaskRefinementProfileV1

from ..immutable_canonical_store import (
    fsync_directory,
    publish_immutable_canonical,
)


class RefinementAnnotationStoreError(RuntimeError):
    """Base error for refinement annotation persistence."""


class RefinementAnnotationStoreIncompleteError(
    RefinementAnnotationStoreError
):
    """A malformed or contradictory annotation set cannot be continued."""


@dataclass(frozen=True)
class RefinementAnnotationAppendReport:
    accepted_count: int
    idempotent_count: int
    published_chunk_count: int
    chunk_relative_paths: tuple[str, ...]


@dataclass(frozen=True)
class RefinementAnnotationSealReport:
    manifest: RefinementAnnotationManifestV1
    manifest_relative_path: str
    idempotent: bool


class RefinementAnnotationStore:
    """One append owner for one immutable post-capture annotation set.

    The caller supplies the set directory. No repository-global dataset
    topology is frozen here. Every append batch is split into bounded,
    atomically published chunks; a crash can therefore lose only an
    unpublished temporary file, never a previously committed annotation.
    """

    def __init__(
        self,
        *,
        annotation_dir: str | Path,
        source_progress_manifest: ProgressEpisodeManifestV1,
        refinement_profile: TaskRefinementProfileV1,
        binding: RefinementAnnotationSetBindingV1,
        max_annotations_per_chunk: int = 128,
    ) -> None:
        validate_refinement_annotation_binding(
            source_progress_manifest=source_progress_manifest,
            refinement_profile=refinement_profile,
            binding=binding,
        )
        if (
            not isinstance(max_annotations_per_chunk, int)
            or isinstance(max_annotations_per_chunk, bool)
            or max_annotations_per_chunk < 1
        ):
            raise ValueError(
                "max_annotations_per_chunk must be a positive integer."
            )
        self.annotation_dir = Path(annotation_dir).expanduser().resolve()
        self.source_progress_manifest = source_progress_manifest
        self.refinement_profile = refinement_profile
        self.binding = binding
        self.max_annotations_per_chunk = max_annotations_per_chunk
        self.binding_path = self.annotation_dir / "binding.cjson"
        self.chunk_dir = self.annotation_dir / "chunks"
        self.manifest_path = self.annotation_dir / "manifest.cjson"
        self._lock = threading.RLock()
        self._closed = False
        self._sealed = False
        self._manifest: RefinementAnnotationManifestV1 | None = None
        self._chunks: list[RefinementAnnotationChunkV1] = []
        self._annotations: list[RefinementBoundaryAnnotationV1] = []
        self._annotation_by_id: dict[
            str,
            RefinementBoundaryAnnotationV1,
        ] = {}

        self.annotation_dir.mkdir(parents=True, exist_ok=True)
        publish_immutable_canonical(
            self.binding_path,
            binding.to_payload(),
            domain=REFINEMENT_ANNOTATION_BINDING_FILE_DOMAIN_V1,
        )
        chunk_dir_existed = self.chunk_dir.exists()
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        if not chunk_dir_existed:
            fsync_directory(self.annotation_dir)
        self._load_chunks()
        self._load_manifest_if_present()

    @property
    def annotations(self) -> tuple[RefinementBoundaryAnnotationV1, ...]:
        with self._lock:
            return tuple(self._annotations)

    @property
    def chunks(self) -> tuple[RefinementAnnotationChunkV1, ...]:
        with self._lock:
            return tuple(self._chunks)

    @property
    def sealed(self) -> bool:
        with self._lock:
            return self._sealed

    @property
    def manifest(self) -> RefinementAnnotationManifestV1 | None:
        with self._lock:
            return self._manifest

    def annotation_by_id(
        self,
        annotation_id: str,
    ) -> RefinementBoundaryAnnotationV1 | None:
        with self._lock:
            return self._annotation_by_id.get(str(annotation_id))

    def append_annotations(
        self,
        annotations: Sequence[RefinementBoundaryAnnotationV1],
    ) -> RefinementAnnotationAppendReport:
        batch = tuple(annotations)
        if not batch:
            raise ValueError("append_annotations requires a non-empty batch.")
        if any(
            not isinstance(item, RefinementBoundaryAnnotationV1)
            for item in batch
        ):
            raise TypeError(
                "annotations must contain RefinementBoundaryAnnotationV1."
            )
        batch_ids = tuple(item.annotation_id for item in batch)
        if len(batch_ids) != len(set(batch_ids)):
            raise RefinementAnnotationStoreError(
                "annotation_id values must be unique within an append batch."
            )

        with self._lock:
            self._require_open()
            new_annotations: list[RefinementBoundaryAnnotationV1] = []
            idempotent_count = 0
            for annotation in batch:
                if (
                    annotation.clock_domain_id
                    != self.binding.bound_clock_domain_id
                ):
                    raise RefinementAnnotationStoreError(
                        "Annotation must use the bound Episode clock."
                    )
                existing = self._annotation_by_id.get(
                    annotation.annotation_id
                )
                if existing is None:
                    new_annotations.append(annotation)
                    continue
                if (
                    existing.semantic_content_hash
                    != annotation.semantic_content_hash
                ):
                    raise RefinementAnnotationStoreError(
                        "Annotation ID conflict for "
                        f"{annotation.annotation_id!r}."
                    )
                idempotent_count += 1

            relative_paths: list[str] = []
            for offset in range(
                0,
                len(new_annotations),
                self.max_annotations_per_chunk,
            ):
                group = tuple(
                    new_annotations[
                        offset : offset + self.max_annotations_per_chunk
                    ]
                )
                chunk = RefinementAnnotationChunkV1(
                    chunk_index=len(self._chunks),
                    first_annotation_index=len(self._annotations),
                    annotations=group,
                )
                path = self._chunk_path(chunk.chunk_index)
                try:
                    publish_immutable_canonical(
                        path,
                        chunk.to_payload(),
                        domain=REFINEMENT_ANNOTATION_CHUNK_FILE_DOMAIN_V1,
                    )
                except ValueError as exc:
                    raise RefinementAnnotationStoreError(
                        "Annotation chunk publication conflicted with another "
                        "writer or existing content."
                    ) from exc
                self._accept_loaded_chunk(chunk)
                relative_paths.append(
                    path.relative_to(self.annotation_dir).as_posix()
                )

            return RefinementAnnotationAppendReport(
                accepted_count=len(new_annotations),
                idempotent_count=idempotent_count,
                published_chunk_count=len(relative_paths),
                chunk_relative_paths=tuple(relative_paths),
            )

    def seal(self) -> RefinementAnnotationSealReport:
        with self._lock:
            if self._manifest is not None:
                return RefinementAnnotationSealReport(
                    manifest=self._manifest,
                    manifest_relative_path=self.manifest_path.name,
                    idempotent=True,
                )
            self._require_open()
            manifest = RefinementAnnotationManifestV1(
                annotation_set_binding_hash=(
                    self.binding.semantic_content_hash
                ),
                annotation_count=len(self._annotations),
                annotation_bundle_hash=(
                    RefinementAnnotationManifestV1.annotation_bundle_hash_for(
                        self._annotations
                    )
                ),
                chunk_hashes=tuple(
                    item.semantic_content_hash for item in self._chunks
                ),
            )
            idempotent = publish_immutable_canonical(
                self.manifest_path,
                manifest.to_payload(),
                domain=REFINEMENT_ANNOTATION_MANIFEST_FILE_DOMAIN_V1,
            )
            self._manifest = manifest
            self._sealed = True
            return RefinementAnnotationSealReport(
                manifest=manifest,
                manifest_relative_path=self.manifest_path.name,
                idempotent=idempotent,
            )

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def __enter__(self) -> "RefinementAnnotationStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RefinementAnnotationStoreError(
                "Refinement annotation store is closed."
            )
        if not self._sealed and self.manifest_path.exists():
            self._load_manifest_if_present()
        if self._sealed:
            raise RefinementAnnotationStoreError(
                "Refinement annotation store is sealed."
            )

    def _chunk_path(self, chunk_index: int) -> Path:
        return self.chunk_dir / f"chunk-{chunk_index:06d}.cjson"

    def _load_chunks(self) -> None:
        entries = tuple(
            sorted(
                path
                for path in self.chunk_dir.iterdir()
                if not path.name.startswith(".")
            )
        )
        expected_names = tuple(
            f"chunk-{index:06d}.cjson" for index in range(len(entries))
        )
        actual_names = tuple(path.name for path in entries)
        if actual_names != expected_names:
            raise RefinementAnnotationStoreIncompleteError(
                "Annotation chunk files are not a contiguous canonical sequence."
            )
        for expected_index, path in enumerate(entries):
            try:
                payload = canonical_loads(
                    path.read_bytes(),
                    expected_domain=(
                        REFINEMENT_ANNOTATION_CHUNK_FILE_DOMAIN_V1
                    ),
                )
                chunk = RefinementAnnotationChunkV1.from_payload(payload)
            except (OSError, TypeError, ValueError) as exc:
                raise RefinementAnnotationStoreIncompleteError(
                    f"Invalid annotation chunk {path.name}: {exc}"
                ) from exc
            if chunk.chunk_index != expected_index:
                raise RefinementAnnotationStoreIncompleteError(
                    "Annotation chunk index does not match its path."
                )
            self._accept_loaded_chunk(chunk)

    def _accept_loaded_chunk(
        self,
        chunk: RefinementAnnotationChunkV1,
    ) -> None:
        if chunk.chunk_index != len(self._chunks):
            raise RefinementAnnotationStoreIncompleteError(
                "Annotation chunk indices are not contiguous."
            )
        if chunk.first_annotation_index != len(self._annotations):
            raise RefinementAnnotationStoreIncompleteError(
                "Annotation indices are not contiguous across chunks."
            )
        for annotation in chunk.annotations:
            if (
                annotation.clock_domain_id
                != self.binding.bound_clock_domain_id
            ):
                raise RefinementAnnotationStoreIncompleteError(
                    "Durable annotation uses a foreign clock."
                )
            if annotation.annotation_id in self._annotation_by_id:
                raise RefinementAnnotationStoreIncompleteError(
                    "Duplicate durable annotation ID "
                    f"{annotation.annotation_id!r}."
                )
        self._chunks.append(chunk)
        self._annotations.extend(chunk.annotations)
        self._annotation_by_id.update(
            (item.annotation_id, item) for item in chunk.annotations
        )

    def _load_manifest_if_present(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            payload = canonical_loads(
                self.manifest_path.read_bytes(),
                expected_domain=(
                    REFINEMENT_ANNOTATION_MANIFEST_FILE_DOMAIN_V1
                ),
            )
            manifest = RefinementAnnotationManifestV1.from_payload(payload)
        except (OSError, TypeError, ValueError) as exc:
            raise RefinementAnnotationStoreIncompleteError(
                f"Invalid refinement annotation manifest: {exc}"
            ) from exc
        if (
            manifest.annotation_set_binding_hash
            != self.binding.semantic_content_hash
            or manifest.annotation_count != len(self._annotations)
            or manifest.annotation_bundle_hash
            != RefinementAnnotationManifestV1.annotation_bundle_hash_for(
                self._annotations
            )
            or manifest.chunk_hashes
            != tuple(item.semantic_content_hash for item in self._chunks)
        ):
            raise RefinementAnnotationStoreIncompleteError(
                "Annotation manifest does not bind the complete chunk set."
            )
        self._manifest = manifest
        self._sealed = True


__all__ = [
    "RefinementAnnotationAppendReport",
    "RefinementAnnotationSealReport",
    "RefinementAnnotationStore",
    "RefinementAnnotationStoreError",
    "RefinementAnnotationStoreIncompleteError",
]
