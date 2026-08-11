"""Read and validate completed retrospective annotation campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_loads
from dataset_contracts.retrospective_campaign import (
    RETROSPECTIVE_ANNOTATION_CAMPAIGN_FILE_DOMAIN_V1,
    RETROSPECTIVE_CAMPAIGN_PUBLICATION_RECEIPT_FILE_DOMAIN_V1,
    RetrospectiveAnnotationCampaignV1,
    RetrospectiveCampaignEpisodeSealV1,
    RetrospectiveCampaignPublicationReceiptV1,
)
from dataset_contracts.retrospective_progress import (
    RETROSPECTIVE_DATASET_LOCAL_SOURCE_LOCATOR_V1,
    RETROSPECTIVE_PROGRESS_MIGRATION_MANIFEST_FILE_DOMAIN_V1,
    RETROSPECTIVE_SOURCE_BINDING_FILE_DOMAIN_V1,
    RetrospectiveProgressMigrationManifestV1,
    RetrospectiveSourceBindingV1,
)

from .episode_curation import source_dataset_base_fingerprint


class RetrospectiveCampaignArtifactReadError(ValueError):
    """Raised when a completed campaign is missing or contradictory."""


@dataclass(frozen=True)
class LoadedRetrospectiveCampaign:
    campaign_dir: Path
    definition: RetrospectiveAnnotationCampaignV1
    receipt: RetrospectiveCampaignPublicationReceiptV1


@dataclass(frozen=True)
class LoadedRetrospectiveCampaignIndex:
    semantic_layer_root: Path
    campaigns: tuple[LoadedRetrospectiveCampaign, ...]
    episode_seals: tuple[RetrospectiveCampaignEpisodeSealV1, ...]

    @property
    def admitted_episode_identity_hashes(self) -> frozenset[str]:
        return frozenset(
            value.episode_capture_identity_hash for value in self.episode_seals
        )


def _mapping(path: Path, *, domain: str) -> Mapping[str, Any]:
    try:
        payload = canonical_loads(path.read_bytes(), expected_domain=domain)
    except (OSError, TypeError, ValueError) as exc:
        raise RetrospectiveCampaignArtifactReadError(
            f"Cannot read campaign artifact {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RetrospectiveCampaignArtifactReadError(
            f"Campaign artifact must contain an object: {path}"
        )
    return payload


def _require_dataset_local_layer(root: Path) -> Path:
    if root.parent.name != "semantic_layers" or root.parent.parent == root.parent:
        raise RetrospectiveCampaignArtifactReadError(
            "Campaign-backed annotation root is not a dataset-local semantic layer."
        )
    owner = root.parent.parent.resolve()
    if root.parent.resolve().parent != owner:
        raise RetrospectiveCampaignArtifactReadError(
            "Semantic layer owner cannot be resolved safely."
        )
    return owner


def load_retrospective_campaign_index(
    semantic_layer_root: str | Path,
) -> LoadedRetrospectiveCampaignIndex:
    """Load only campaigns whose publication receipt is present and valid."""

    root = Path(semantic_layer_root).expanduser().resolve()
    owner = _require_dataset_local_layer(root)
    campaigns_root = root / "campaigns"
    if not campaigns_root.is_dir():
        raise RetrospectiveCampaignArtifactReadError(
            f"Dataset-local semantic layer has no campaigns directory: {root}"
        )
    owner_fingerprint = source_dataset_base_fingerprint(owner)
    loaded: list[LoadedRetrospectiveCampaign] = []
    seals_by_identity: dict[str, RetrospectiveCampaignEpisodeSealV1] = {}
    for campaign_dir in sorted(campaigns_root.iterdir(), key=lambda item: item.name):
        if not campaign_dir.is_dir():
            continue
        receipt_path = campaign_dir / "publication_receipt.cjson"
        if not receipt_path.is_file():
            # A definition without a receipt is a resumable partial publish,
            # not an annotation-visible campaign.
            continue
        definition_path = campaign_dir / "definition.cjson"
        if not definition_path.is_file():
            raise RetrospectiveCampaignArtifactReadError(
                f"Campaign receipt has no definition: {campaign_dir}"
            )
        try:
            definition = RetrospectiveAnnotationCampaignV1.from_payload(
                _mapping(
                    definition_path,
                    domain=RETROSPECTIVE_ANNOTATION_CAMPAIGN_FILE_DOMAIN_V1,
                )
            )
            receipt = RetrospectiveCampaignPublicationReceiptV1.from_payload(
                _mapping(
                    receipt_path,
                    domain=(
                        RETROSPECTIVE_CAMPAIGN_PUBLICATION_RECEIPT_FILE_DOMAIN_V1
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise RetrospectiveCampaignArtifactReadError(
                f"Invalid campaign under {campaign_dir}: {exc}"
            ) from exc
        if definition.campaign_id != campaign_dir.name:
            raise RetrospectiveCampaignArtifactReadError(
                "Campaign directory and definition IDs differ."
            )
        if definition.semantic_layer_id != root.name:
            raise RetrospectiveCampaignArtifactReadError(
                "Campaign definition names a different semantic layer."
            )
        if definition.source_dataset_base_fingerprint != owner_fingerprint:
            raise RetrospectiveCampaignArtifactReadError(
                "Campaign definition and owning dataset fingerprints differ."
            )
        if receipt.campaign_definition_hash != definition.semantic_content_hash:
            raise RetrospectiveCampaignArtifactReadError(
                "Campaign receipt does not bind the stored definition."
            )
        selected = set(definition.selected_episode_identity_hashes)
        observed = {
            value.episode_capture_identity_hash for value in receipt.episode_seals
        }
        if observed != selected or len(receipt.episode_seals) != len(selected):
            raise RetrospectiveCampaignArtifactReadError(
                "Campaign receipt does not seal exactly the selected Episodes."
            )

        for seal in receipt.episode_seals:
            episode_dir = root / "episodes" / seal.episode_capture_identity_hash
            try:
                manifest = RetrospectiveProgressMigrationManifestV1.from_payload(
                    _mapping(
                        episode_dir / "migration_manifest.cjson",
                        domain=(
                            RETROSPECTIVE_PROGRESS_MIGRATION_MANIFEST_FILE_DOMAIN_V1
                        ),
                    )
                )
                source = RetrospectiveSourceBindingV1.from_payload(
                    _mapping(
                        episode_dir / "source_binding.cjson",
                        domain=RETROSPECTIVE_SOURCE_BINDING_FILE_DOMAIN_V1,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise RetrospectiveCampaignArtifactReadError(
                    f"Invalid campaign Episode {episode_dir}: {exc}"
                ) from exc
            identity = seal.episode_capture_identity_hash
            if (
                manifest.episode_capture_identity_hash != identity
                or source.episode_capture_identity_hash != identity
            ):
                raise RetrospectiveCampaignArtifactReadError(
                    "Campaign Episode identity bindings differ."
                )
            if manifest.semantic_content_hash != seal.migration_manifest_hash:
                raise RetrospectiveCampaignArtifactReadError(
                    "Campaign Episode seal and migration manifest differ."
                )
            if manifest.source_binding_hash != source.semantic_content_hash:
                raise RetrospectiveCampaignArtifactReadError(
                    "Migration manifest and source binding differ."
                )
            if (
                source.source_dataset_locator
                != RETROSPECTIVE_DATASET_LOCAL_SOURCE_LOCATOR_V1
            ):
                raise RetrospectiveCampaignArtifactReadError(
                    "Dataset-local campaign contains a non-local source locator."
                )
            if source.source_dataset_base_fingerprint != owner_fingerprint:
                raise RetrospectiveCampaignArtifactReadError(
                    "Campaign Episode belongs to a different source dataset."
                )
            if (
                manifest.migration_actor_id != definition.migration_actor_id
                or manifest.migration_created_at
                != definition.migration_created_at
            ):
                raise RetrospectiveCampaignArtifactReadError(
                    "Campaign and Episode migration identities differ."
                )
            previous = seals_by_identity.get(identity)
            if previous is not None and previous != seal:
                raise RetrospectiveCampaignArtifactReadError(
                    "Campaigns bind one Episode to different migration seals."
                )
            seals_by_identity[identity] = seal
        loaded.append(
            LoadedRetrospectiveCampaign(
                campaign_dir=campaign_dir,
                definition=definition,
                receipt=receipt,
            )
        )
    return LoadedRetrospectiveCampaignIndex(
        semantic_layer_root=root,
        campaigns=tuple(loaded),
        episode_seals=tuple(
            seals_by_identity[key] for key in sorted(seals_by_identity)
        ),
    )


__all__ = [
    "LoadedRetrospectiveCampaign",
    "LoadedRetrospectiveCampaignIndex",
    "RetrospectiveCampaignArtifactReadError",
    "load_retrospective_campaign_index",
]
