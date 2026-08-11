"""Atomic persistence for externally bound retrospective Progress windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time

from dataset_contracts.progress_facts import (
    PROGRESS_BINDING_FILE_DOMAIN_V1,
    PROGRESS_PROFILE_FILE_DOMAIN_V1,
)
from dataset_contracts.progress_profile import (
    ProgressEpisodeBindingV1,
    TaskProgressProfileV1,
)
from dataset_contracts.progress_stages import (
    TASK_STAGE_PROFILE_FILE_DOMAIN_V1,
    TaskStageProfileV1,
)
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
    RETROSPECTIVE_PROGRESS_WINDOW_FILE_DOMAIN_V1,
    RETROSPECTIVE_SOURCE_BINDING_FILE_DOMAIN_V1,
    RetrospectiveProgressMigrationManifestV1,
    RetrospectiveProgressWindowV1,
    RetrospectiveSourceBindingV1,
)
from dataset_interface.retrospective_progress import (
    resolve_retrospective_progress_window,
)
from dataset_interface.sample_axis import EpisodeSampleAxis

from ..immutable_canonical_store import publish_immutable_canonical


RETROSPECTIVE_EPISODES_RELATIVE_ROOT = Path("episodes")
RETROSPECTIVE_CAMPAIGNS_RELATIVE_ROOT = Path("campaigns")
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class RetrospectiveProgressStoreError(RuntimeError):
    """Raised when an immutable migration workspace cannot be published."""


@dataclass(frozen=True)
class RetrospectiveProgressCommitReport:
    episode_dir: Path
    migration_manifest: RetrospectiveProgressMigrationManifestV1
    idempotent: bool


@dataclass(frozen=True)
class RetrospectiveMigrationIdentity:
    actor_id: str
    created_at: float
    recovered: bool


@dataclass(frozen=True)
class RetrospectiveCampaignDefinitionReport:
    campaign_dir: Path
    campaign: RetrospectiveAnnotationCampaignV1
    idempotent: bool
    identity_recovered: bool


@dataclass(frozen=True)
class RetrospectiveCampaignReceiptReport:
    receipt_path: Path
    receipt: RetrospectiveCampaignPublicationReceiptV1
    idempotent: bool


def _campaign_directory(root: Path, campaign_id: str) -> Path:
    value = str(campaign_id).strip()
    if not _SAFE_ID_PATTERN.fullmatch(value):
        raise RetrospectiveProgressStoreError(
            "campaign_id must use only letters, digits, '.', '_' or '-'."
        )
    parent = (root / RETROSPECTIVE_CAMPAIGNS_RELATIVE_ROOT).resolve()
    result = (parent / value).resolve()
    if result.parent != parent:
        raise RetrospectiveProgressStoreError(
            "Campaign path escaped the semantic layer root."
        )
    return result


def prepare_retrospective_campaign_definition(
    *,
    semantic_layer_root: str | Path,
    semantic_layer_id: str,
    campaign_id: str,
    campaign_kind: str,
    source_dataset_base_fingerprint: str,
    cohort_plan_hash: str,
    eligible_episode_identity_hashes: tuple[str, ...],
    selected_episode_identity_hashes: tuple[str, ...],
    requested_actor_id: str = "",
    requested_created_at: float | None = None,
) -> RetrospectiveCampaignDefinitionReport:
    """Resolve one campaign definition without writing any artifact."""

    root = Path(semantic_layer_root).expanduser().resolve()
    if root.name != str(semantic_layer_id).strip():
        raise RetrospectiveProgressStoreError(
            "semantic_layer_id differs from the semantic layer directory."
        )
    campaign_dir = _campaign_directory(root, campaign_id)
    definition_path = campaign_dir / "definition.cjson"
    existing: RetrospectiveAnnotationCampaignV1 | None = None
    if definition_path.is_file():
        try:
            payload = canonical_loads(
                definition_path.read_bytes(),
                expected_domain=RETROSPECTIVE_ANNOTATION_CAMPAIGN_FILE_DOMAIN_V1,
            )
            if not isinstance(payload, dict):
                raise TypeError("campaign definition root is not an object")
            existing = RetrospectiveAnnotationCampaignV1.from_payload(payload)
        except (OSError, TypeError, ValueError) as exc:
            raise RetrospectiveProgressStoreError(
                f"Cannot recover campaign definition from {definition_path}: {exc}"
            ) from exc

    actor = str(requested_actor_id).strip()
    identity_recovered = existing is not None
    if existing is not None:
        if actor and actor != existing.migration_actor_id:
            raise RetrospectiveProgressStoreError(
                "Requested migration actor conflicts with the campaign: "
                f"requested={actor!r} stored={existing.migration_actor_id!r}"
            )
        if (
            requested_created_at is not None
            and float(requested_created_at) != existing.migration_created_at
        ):
            raise RetrospectiveProgressStoreError(
                "Requested migration timestamp conflicts with the campaign: "
                f"requested={requested_created_at} "
                f"stored={existing.migration_created_at}"
            )
        actor = existing.migration_actor_id
        created_at = existing.migration_created_at
    else:
        if not actor:
            raise RetrospectiveProgressStoreError(
                "The first campaign publication requires migration_actor_id."
            )
        created_at = (
            time.time()
            if requested_created_at is None
            else float(requested_created_at)
        )

    campaign = RetrospectiveAnnotationCampaignV1(
        semantic_layer_id=semantic_layer_id,
        campaign_id=campaign_id,
        campaign_kind=campaign_kind,
        source_dataset_base_fingerprint=source_dataset_base_fingerprint,
        cohort_plan_hash=cohort_plan_hash,
        eligible_episode_identity_hashes=eligible_episode_identity_hashes,
        selected_episode_identity_hashes=selected_episode_identity_hashes,
        migration_actor_id=actor,
        migration_created_at=created_at,
        migration_clock_domain_id="annotation_wall_clock",
    )
    if existing is not None and existing != campaign:
        raise RetrospectiveProgressStoreError(
            "Requested campaign content conflicts with the stored definition."
        )
    return RetrospectiveCampaignDefinitionReport(
        campaign_dir=campaign_dir,
        campaign=campaign,
        idempotent=existing is not None,
        identity_recovered=identity_recovered,
    )


def publish_retrospective_campaign_definition(
    *,
    semantic_layer_root: str | Path,
    semantic_layer_id: str,
    campaign_id: str,
    campaign_kind: str,
    source_dataset_base_fingerprint: str,
    cohort_plan_hash: str,
    eligible_episode_identity_hashes: tuple[str, ...],
    selected_episode_identity_hashes: tuple[str, ...],
    requested_actor_id: str = "",
    requested_created_at: float | None = None,
) -> RetrospectiveCampaignDefinitionReport:
    """Create once or recover one campaign-scoped migration identity."""

    prepared = prepare_retrospective_campaign_definition(
        semantic_layer_root=semantic_layer_root,
        semantic_layer_id=semantic_layer_id,
        campaign_id=campaign_id,
        campaign_kind=campaign_kind,
        source_dataset_base_fingerprint=source_dataset_base_fingerprint,
        cohort_plan_hash=cohort_plan_hash,
        eligible_episode_identity_hashes=eligible_episode_identity_hashes,
        selected_episode_identity_hashes=selected_episode_identity_hashes,
        requested_actor_id=requested_actor_id,
        requested_created_at=requested_created_at,
    )
    definition_path = prepared.campaign_dir / "definition.cjson"
    try:
        idempotent = publish_immutable_canonical(
            definition_path,
            prepared.campaign.to_payload(),
            domain=RETROSPECTIVE_ANNOTATION_CAMPAIGN_FILE_DOMAIN_V1,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RetrospectiveProgressStoreError(
            f"Cannot publish retrospective campaign definition: {exc}"
        ) from exc
    return RetrospectiveCampaignDefinitionReport(
        campaign_dir=prepared.campaign_dir,
        campaign=prepared.campaign,
        idempotent=idempotent,
        identity_recovered=prepared.identity_recovered,
    )


def seal_retrospective_campaign_publication(
    *,
    semantic_layer_root: str | Path,
    campaign: RetrospectiveAnnotationCampaignV1,
    episode_seals: tuple[RetrospectiveCampaignEpisodeSealV1, ...],
) -> RetrospectiveCampaignReceiptReport:
    """Seal a campaign only after every selected Episode seal is present."""

    root = Path(semantic_layer_root).expanduser().resolve()
    campaign_dir = _campaign_directory(root, campaign.campaign_id)
    definition_path = campaign_dir / "definition.cjson"
    try:
        definition_payload = canonical_loads(
            definition_path.read_bytes(),
            expected_domain=RETROSPECTIVE_ANNOTATION_CAMPAIGN_FILE_DOMAIN_V1,
        )
        if not isinstance(definition_payload, dict):
            raise TypeError("campaign definition root is not an object")
        stored_campaign = RetrospectiveAnnotationCampaignV1.from_payload(
            definition_payload
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RetrospectiveProgressStoreError(
            f"Cannot verify stored campaign definition {definition_path}: {exc}"
        ) from exc
    if stored_campaign != campaign:
        raise RetrospectiveProgressStoreError(
            "Campaign receipt does not use the stored campaign definition."
        )
    selected = set(campaign.selected_episode_identity_hashes)
    observed = {
        value.episode_capture_identity_hash for value in episode_seals
    }
    if observed != selected or len(episode_seals) != len(selected):
        raise RetrospectiveProgressStoreError(
            "Campaign receipt must seal exactly the selected Episodes."
        )
    for seal in episode_seals:
        manifest_path = (
            _episode_directory(root, seal.episode_capture_identity_hash)
            / "migration_manifest.cjson"
        )
        try:
            payload = canonical_loads(
                manifest_path.read_bytes(),
                expected_domain=(
                    RETROSPECTIVE_PROGRESS_MIGRATION_MANIFEST_FILE_DOMAIN_V1
                ),
            )
            if not isinstance(payload, dict):
                raise TypeError("migration manifest root is not an object")
            manifest = RetrospectiveProgressMigrationManifestV1.from_payload(payload)
            source_path = manifest_path.parent / "source_binding.cjson"
            source_payload = canonical_loads(
                source_path.read_bytes(),
                expected_domain=RETROSPECTIVE_SOURCE_BINDING_FILE_DOMAIN_V1,
            )
            if not isinstance(source_payload, dict):
                raise TypeError("source binding root is not an object")
            source = RetrospectiveSourceBindingV1.from_payload(source_payload)
        except (OSError, TypeError, ValueError) as exc:
            raise RetrospectiveProgressStoreError(
                f"Cannot verify campaign Episode seal {manifest_path}: {exc}"
            ) from exc
        if manifest.semantic_content_hash != seal.migration_manifest_hash:
            raise RetrospectiveProgressStoreError(
                f"Campaign Episode seal differs from {manifest_path}."
            )
        if (
            source.episode_capture_identity_hash
            != seal.episode_capture_identity_hash
            or manifest.source_binding_hash != source.semantic_content_hash
        ):
            raise RetrospectiveProgressStoreError(
                "Campaign Episode source binding differs from its seal."
            )
        if (
            source.source_dataset_locator
            != RETROSPECTIVE_DATASET_LOCAL_SOURCE_LOCATOR_V1
        ):
            raise RetrospectiveProgressStoreError(
                "Dataset-local campaign requires a dataset-local source locator."
            )
        if (
            source.source_dataset_base_fingerprint
            != campaign.source_dataset_base_fingerprint
        ):
            raise RetrospectiveProgressStoreError(
                "Campaign and Episode source dataset fingerprints differ."
            )
        if manifest.migration_actor_id != campaign.migration_actor_id:
            raise RetrospectiveProgressStoreError(
                "Campaign and Episode migration actors differ."
            )
        if manifest.migration_created_at != campaign.migration_created_at:
            raise RetrospectiveProgressStoreError(
                "Campaign and Episode migration timestamps differ."
            )
    receipt = RetrospectiveCampaignPublicationReceiptV1(
        campaign_definition_hash=campaign.semantic_content_hash,
        episode_seals=episode_seals,
    )
    receipt_path = campaign_dir / "publication_receipt.cjson"
    try:
        idempotent = publish_immutable_canonical(
            receipt_path,
            receipt.to_payload(),
            domain=RETROSPECTIVE_CAMPAIGN_PUBLICATION_RECEIPT_FILE_DOMAIN_V1,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RetrospectiveProgressStoreError(
            f"Cannot publish retrospective campaign receipt: {exc}"
        ) from exc
    return RetrospectiveCampaignReceiptReport(
        receipt_path=receipt_path,
        receipt=receipt,
        idempotent=idempotent,
    )


def resolve_retrospective_migration_identity(
    workspace_root: str | Path,
    *,
    requested_actor_id: str = "",
    requested_created_at: float | None = None,
) -> RetrospectiveMigrationIdentity:
    """Create once or recover the workspace-wide migration identity.

    Existing immutable Episode seals are the authority, so no mutable side
    file is needed.  A partial publish can therefore be resumed with the
    exact original actor and annotation-clock timestamp.
    """

    root = Path(workspace_root).expanduser().resolve()
    existing: set[tuple[str, float]] = set()
    episodes_root = root / RETROSPECTIVE_EPISODES_RELATIVE_ROOT
    if episodes_root.is_dir():
        for episode_dir in sorted(episodes_root.iterdir()):
            path = episode_dir / "migration_manifest.cjson"
            if not episode_dir.is_dir() or not path.is_file():
                continue
            try:
                payload = canonical_loads(
                    path.read_bytes(),
                    expected_domain=(
                        RETROSPECTIVE_PROGRESS_MIGRATION_MANIFEST_FILE_DOMAIN_V1
                    ),
                )
                if not isinstance(payload, dict):
                    raise TypeError("migration manifest root is not an object")
                manifest = RetrospectiveProgressMigrationManifestV1.from_payload(
                    payload
                )
            except (OSError, TypeError, ValueError) as exc:
                raise RetrospectiveProgressStoreError(
                    f"Cannot recover migration identity from {path}: {exc}"
                ) from exc
            existing.add(
                (manifest.migration_actor_id, manifest.migration_created_at)
            )
    if len(existing) > 1:
        raise RetrospectiveProgressStoreError(
            "Workspace contains multiple migration actor/timestamp pairs."
        )

    actor = str(requested_actor_id).strip()
    if existing:
        stored_actor, stored_created_at = next(iter(existing))
        if actor and actor != stored_actor:
            raise RetrospectiveProgressStoreError(
                "Requested migration actor conflicts with the workspace: "
                f"requested={actor!r} stored={stored_actor!r}"
            )
        if (
            requested_created_at is not None
            and float(requested_created_at) != stored_created_at
        ):
            raise RetrospectiveProgressStoreError(
                "Requested migration timestamp conflicts with the workspace: "
                f"requested={requested_created_at} stored={stored_created_at}"
            )
        return RetrospectiveMigrationIdentity(
            actor_id=stored_actor,
            created_at=stored_created_at,
            recovered=True,
        )

    if not actor:
        raise RetrospectiveProgressStoreError(
            "The first workspace publication requires migration_actor_id."
        )
    created_at = (
        time.time()
        if requested_created_at is None
        else float(requested_created_at)
    )
    return RetrospectiveMigrationIdentity(
        actor_id=actor,
        created_at=created_at,
        recovered=False,
    )


def _episode_directory(workspace_root: Path, episode_identity_hash: str) -> Path:
    identity = str(episode_identity_hash).strip().lower()
    if len(identity) != 64 or any(
        character not in "0123456789abcdef" for character in identity
    ):
        raise ValueError("episode_identity_hash must be a lowercase SHA-256 digest.")
    episode_dir = (workspace_root / RETROSPECTIVE_EPISODES_RELATIVE_ROOT / identity).resolve()
    expected_parent = (workspace_root / RETROSPECTIVE_EPISODES_RELATIVE_ROOT).resolve()
    if episode_dir.parent != expected_parent:
        raise ValueError("Retrospective Episode path escaped its owned root.")
    return episode_dir


def commit_retrospective_progress_episode(
    *,
    workspace_root: str | Path,
    progress_profile: TaskProgressProfileV1,
    progress_binding: ProgressEpisodeBindingV1,
    stage_profile: TaskStageProfileV1,
    sample_axis: EpisodeSampleAxis,
    source_binding: RetrospectiveSourceBindingV1,
    window: RetrospectiveProgressWindowV1,
    migration_manifest: RetrospectiveProgressMigrationManifestV1,
) -> RetrospectiveProgressCommitReport:
    """Publish one migrated Episode; the migration manifest is written last."""

    resolve_retrospective_progress_window(
        progress_profile=progress_profile,
        progress_binding=progress_binding,
        stage_profile=stage_profile,
        sample_axis=sample_axis,
        source_binding=source_binding,
        window=window,
        manifest=migration_manifest,
    )
    root = Path(workspace_root).expanduser().resolve()
    episode_dir = _episode_directory(
        root,
        source_binding.episode_capture_identity_hash,
    )
    try:
        idempotent = (
            publish_immutable_canonical(
                root / "profile.cjson",
                progress_profile.to_payload(),
                domain=PROGRESS_PROFILE_FILE_DOMAIN_V1,
            ),
            publish_immutable_canonical(
                root / "stages.cjson",
                stage_profile.to_payload(),
                domain=TASK_STAGE_PROFILE_FILE_DOMAIN_V1,
            ),
            publish_immutable_canonical(
                episode_dir / "source_binding.cjson",
                source_binding.to_payload(),
                domain=RETROSPECTIVE_SOURCE_BINDING_FILE_DOMAIN_V1,
            ),
            publish_immutable_canonical(
                episode_dir / "progress_binding.cjson",
                progress_binding.to_payload(),
                domain=PROGRESS_BINDING_FILE_DOMAIN_V1,
            ),
            publish_immutable_canonical(
                episode_dir / "retrospective_window.cjson",
                window.to_payload(),
                domain=RETROSPECTIVE_PROGRESS_WINDOW_FILE_DOMAIN_V1,
            ),
            publish_immutable_canonical(
                episode_dir / "migration_manifest.cjson",
                migration_manifest.to_payload(),
                domain=(
                    RETROSPECTIVE_PROGRESS_MIGRATION_MANIFEST_FILE_DOMAIN_V1
                ),
            ),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RetrospectiveProgressStoreError(
            f"Cannot publish retrospective Progress Episode: {exc}"
        ) from exc
    return RetrospectiveProgressCommitReport(
        episode_dir=episode_dir,
        migration_manifest=migration_manifest,
        idempotent=all(idempotent),
    )


__all__ = [
    "RETROSPECTIVE_CAMPAIGNS_RELATIVE_ROOT",
    "RETROSPECTIVE_EPISODES_RELATIVE_ROOT",
    "RetrospectiveCampaignDefinitionReport",
    "RetrospectiveCampaignReceiptReport",
    "RetrospectiveProgressCommitReport",
    "RetrospectiveMigrationIdentity",
    "RetrospectiveProgressStoreError",
    "commit_retrospective_progress_episode",
    "prepare_retrospective_campaign_definition",
    "publish_retrospective_campaign_definition",
    "resolve_retrospective_migration_identity",
    "seal_retrospective_campaign_publication",
]
