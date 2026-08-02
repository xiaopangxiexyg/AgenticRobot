"""Read and verify sealed refinement annotation sets."""

from __future__ import annotations

from pathlib import Path

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

from .progress_refinement import (
    SelectedRefinementAnnotationSet,
)


class ProgressAnnotationReadError(ValueError):
    """Raised when a sealed annotation set cannot be verified exactly."""


def load_selected_refinement_annotation_set(
    *,
    annotation_dir: str | Path,
    source_progress_manifest: ProgressEpisodeManifestV1,
    refinement_profile: TaskRefinementProfileV1,
) -> SelectedRefinementAnnotationSet:
    """Load one immutable set as the caller-selected refinement read view."""

    root = Path(annotation_dir).expanduser().resolve()
    binding_path = root / "binding.cjson"
    manifest_path = root / "manifest.cjson"
    chunk_dir = root / "chunks"
    if not binding_path.is_file():
        raise ProgressAnnotationReadError(
            "Refinement annotation binding is missing."
        )
    if not manifest_path.is_file():
        raise ProgressAnnotationReadError(
            "Refinement annotation set is not sealed."
        )
    if not chunk_dir.is_dir():
        raise ProgressAnnotationReadError(
            "Refinement annotation chunk directory is missing."
        )

    try:
        binding = RefinementAnnotationSetBindingV1.from_payload(
            canonical_loads(
                binding_path.read_bytes(),
                expected_domain=(
                    REFINEMENT_ANNOTATION_BINDING_FILE_DOMAIN_V1
                ),
            )
        )
        validate_refinement_annotation_binding(
            source_progress_manifest=source_progress_manifest,
            refinement_profile=refinement_profile,
            binding=binding,
        )
        manifest = RefinementAnnotationManifestV1.from_payload(
            canonical_loads(
                manifest_path.read_bytes(),
                expected_domain=(
                    REFINEMENT_ANNOTATION_MANIFEST_FILE_DOMAIN_V1
                ),
            )
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ProgressAnnotationReadError(
            f"Invalid refinement annotation binding or manifest: {exc}"
        ) from exc

    entries = tuple(
        sorted(
            path
            for path in chunk_dir.iterdir()
            if not path.name.startswith(".")
        )
    )
    expected_names = tuple(
        f"chunk-{index:06d}.cjson" for index in range(len(entries))
    )
    if tuple(path.name for path in entries) != expected_names:
        raise ProgressAnnotationReadError(
            "Annotation chunk files are not a contiguous canonical sequence."
        )

    chunks: list[RefinementAnnotationChunkV1] = []
    annotations: list[RefinementBoundaryAnnotationV1] = []
    annotation_ids: set[str] = set()
    for expected_index, path in enumerate(entries):
        try:
            chunk = RefinementAnnotationChunkV1.from_payload(
                canonical_loads(
                    path.read_bytes(),
                    expected_domain=(
                        REFINEMENT_ANNOTATION_CHUNK_FILE_DOMAIN_V1
                    ),
                )
            )
        except (OSError, TypeError, ValueError) as exc:
            raise ProgressAnnotationReadError(
                f"Invalid refinement annotation chunk {path.name}: {exc}"
            ) from exc
        if (
            chunk.chunk_index != expected_index
            or chunk.first_annotation_index != len(annotations)
        ):
            raise ProgressAnnotationReadError(
                "Annotation chunk indices are not contiguous."
            )
        for annotation in chunk.annotations:
            if annotation.clock_domain_id != binding.bound_clock_domain_id:
                raise ProgressAnnotationReadError(
                    "Refinement annotation uses a foreign clock."
                )
            if annotation.annotation_id in annotation_ids:
                raise ProgressAnnotationReadError(
                    "Duplicate annotation ID in sealed chunks."
                )
            annotation_ids.add(annotation.annotation_id)
        chunks.append(chunk)
        annotations.extend(chunk.annotations)

    if (
        manifest.annotation_set_binding_hash
        != binding.semantic_content_hash
        or manifest.annotation_count != len(annotations)
        or manifest.annotation_bundle_hash
        != RefinementAnnotationManifestV1.annotation_bundle_hash_for(
            annotations
        )
        or manifest.chunk_hashes
        != tuple(item.semantic_content_hash for item in chunks)
    ):
        raise ProgressAnnotationReadError(
            "Annotation manifest does not bind the complete chunk set."
        )

    return SelectedRefinementAnnotationSet(
        annotation_set_id=binding.annotation_set_id,
        source_annotation_manifest_hash=manifest.semantic_content_hash,
        source_progress_manifest_hash=(
            binding.source_progress_manifest_hash
        ),
        task_refinement_profile_hash=(
            binding.task_refinement_profile_hash
        ),
        sealed=True,
        observations=tuple(annotations),
    )


__all__ = [
    "ProgressAnnotationReadError",
    "load_selected_refinement_annotation_set",
]
