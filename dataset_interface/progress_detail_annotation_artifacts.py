"""Read-only loading of immutable detail-stage annotation overlays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_loads
from dataset_contracts.progress_detail_annotations import (
    DETAIL_STAGE_ANNOTATION_BINDING_FILE_DOMAIN_V1,
    DETAIL_STAGE_ANNOTATION_BUNDLE_FILE_DOMAIN_V1,
    DETAIL_STAGE_ANNOTATION_MANIFEST_FILE_DOMAIN_V1,
    DetailStageAnnotationBundleV1,
    DetailStageAnnotationManifestV1,
    DetailStageAnnotationSetBindingV1,
)


DETAIL_STAGE_ANNOTATION_RELATIVE_ROOT = Path(
    "meta/progress/detail_stage_annotations"
)


class DetailStageAnnotationArtifactReadError(ValueError):
    """Raised when a detail overlay is incomplete or internally invalid."""


@dataclass(frozen=True)
class LoadedDetailStageAnnotationSet:
    annotation_dir: Path
    binding: DetailStageAnnotationSetBindingV1
    bundle: DetailStageAnnotationBundleV1
    manifest: DetailStageAnnotationManifestV1


def load_detail_stage_annotation_set(
    annotation_dir: str | Path,
) -> LoadedDetailStageAnnotationSet:
    """Load one complete detail overlay without repairing or mutating it."""

    root = Path(annotation_dir).expanduser().resolve()
    try:
        binding_payload = canonical_loads(
            (root / "binding.cjson").read_bytes(),
            expected_domain=DETAIL_STAGE_ANNOTATION_BINDING_FILE_DOMAIN_V1,
        )
        bundle_payload = canonical_loads(
            (root / "annotations.cjson").read_bytes(),
            expected_domain=DETAIL_STAGE_ANNOTATION_BUNDLE_FILE_DOMAIN_V1,
        )
        manifest_payload = canonical_loads(
            (root / "manifest.cjson").read_bytes(),
            expected_domain=DETAIL_STAGE_ANNOTATION_MANIFEST_FILE_DOMAIN_V1,
        )
        if not all(
            isinstance(item, Mapping)
            for item in (binding_payload, bundle_payload, manifest_payload)
        ):
            raise TypeError(
                "Canonical detail annotation payload is not an object."
            )
        binding = DetailStageAnnotationSetBindingV1.from_payload(
            binding_payload
        )
        bundle = DetailStageAnnotationBundleV1.from_payload(bundle_payload)
        manifest = DetailStageAnnotationManifestV1.from_payload(
            manifest_payload
        )
    except (OSError, TypeError, ValueError) as exc:
        raise DetailStageAnnotationArtifactReadError(
            f"Incomplete or invalid detail annotation set {root}: {exc}"
        ) from exc
    if (
        manifest.annotation_set_binding_hash != binding.semantic_content_hash
        or manifest.annotation_bundle_hash != bundle.semantic_content_hash
    ):
        raise DetailStageAnnotationArtifactReadError(
            "Detail annotation manifest does not bind the loaded content."
        )
    return LoadedDetailStageAnnotationSet(
        annotation_dir=root,
        binding=binding,
        bundle=bundle,
        manifest=manifest,
    )


__all__ = [
    "DETAIL_STAGE_ANNOTATION_RELATIVE_ROOT",
    "DetailStageAnnotationArtifactReadError",
    "LoadedDetailStageAnnotationSet",
    "load_detail_stage_annotation_set",
]
