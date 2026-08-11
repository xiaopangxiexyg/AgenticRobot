"""Read-only loading of sealed ordered-stage annotation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_loads
from dataset_contracts.progress_stage_annotations import (
    STAGE_ANNOTATION_BINDING_FILE_DOMAIN_V1,
    STAGE_ANNOTATION_BUNDLE_FILE_DOMAIN_V1,
    STAGE_ANNOTATION_MANIFEST_FILE_DOMAIN_V1,
    StageAnnotationBundleV1,
    StageAnnotationManifestV1,
    StageAnnotationSetBindingV1,
)


STAGE_ANNOTATION_RELATIVE_ROOT = Path("meta/progress/stage_annotations")


class StageAnnotationArtifactReadError(ValueError):
    """Raised when a sealed stage-annotation set is incomplete or invalid."""


@dataclass(frozen=True)
class LoadedStageAnnotationSet:
    annotation_dir: Path
    binding: StageAnnotationSetBindingV1
    bundle: StageAnnotationBundleV1
    manifest: StageAnnotationManifestV1


def load_stage_annotation_set(
    annotation_dir: str | Path,
) -> LoadedStageAnnotationSet:
    """Load one complete immutable set without creating or repairing files."""

    root = Path(annotation_dir).expanduser().resolve()
    try:
        binding_payload = canonical_loads(
            (root / "binding.cjson").read_bytes(),
            expected_domain=STAGE_ANNOTATION_BINDING_FILE_DOMAIN_V1,
        )
        bundle_payload = canonical_loads(
            (root / "annotations.cjson").read_bytes(),
            expected_domain=STAGE_ANNOTATION_BUNDLE_FILE_DOMAIN_V1,
        )
        manifest_payload = canonical_loads(
            (root / "manifest.cjson").read_bytes(),
            expected_domain=STAGE_ANNOTATION_MANIFEST_FILE_DOMAIN_V1,
        )
        if not all(
            isinstance(item, Mapping)
            for item in (binding_payload, bundle_payload, manifest_payload)
        ):
            raise TypeError("Canonical stage annotation payload is not an object.")
        binding = StageAnnotationSetBindingV1.from_payload(binding_payload)
        bundle = StageAnnotationBundleV1.from_payload(bundle_payload)
        manifest = StageAnnotationManifestV1.from_payload(manifest_payload)
    except (OSError, TypeError, ValueError) as exc:
        raise StageAnnotationArtifactReadError(
            f"Incomplete or invalid stage annotation set {root}: {exc}"
        ) from exc
    if (
        manifest.annotation_set_binding_hash != binding.semantic_content_hash
        or manifest.annotation_bundle_hash != bundle.semantic_content_hash
    ):
        raise StageAnnotationArtifactReadError(
            "Stage annotation manifest does not bind the loaded content."
        )
    return LoadedStageAnnotationSet(
        annotation_dir=root,
        binding=binding,
        bundle=bundle,
        manifest=manifest,
    )


__all__ = [
    "LoadedStageAnnotationSet",
    "STAGE_ANNOTATION_RELATIVE_ROOT",
    "StageAnnotationArtifactReadError",
    "load_stage_annotation_set",
]
