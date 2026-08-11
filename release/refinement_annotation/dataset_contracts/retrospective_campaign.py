"""Pure contracts for auditable retrospective-annotation campaigns."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from .canonical_encoding import canonical_content_hash


RETROSPECTIVE_ANNOTATION_CAMPAIGN_SCHEMA_V1 = (
    "dexteleop.retrospective_annotation_campaign.v1"
)
RETROSPECTIVE_ANNOTATION_CAMPAIGN_FILE_DOMAIN_V1 = (
    "dexteleop.retrospective_annotation_campaign.file.v1"
)
RETROSPECTIVE_CAMPAIGN_EPISODE_SEAL_SCHEMA_V1 = (
    "dexteleop.retrospective_campaign_episode_seal.v1"
)
RETROSPECTIVE_CAMPAIGN_PUBLICATION_RECEIPT_SCHEMA_V1 = (
    "dexteleop.retrospective_campaign_publication_receipt.v1"
)
RETROSPECTIVE_CAMPAIGN_PUBLICATION_RECEIPT_FILE_DOMAIN_V1 = (
    "dexteleop.retrospective_campaign_publication_receipt.file.v1"
)

_CAMPAIGN_HASH_DOMAIN = "dexteleop.retrospective_annotation_campaign.semantic.v1"
_RECEIPT_HASH_DOMAIN = (
    "dexteleop.retrospective_campaign_publication_receipt.semantic.v1"
)
_CAMPAIGN_KINDS = frozenset({"PILOT_QA", "EXPANSION", "FULL"})


class RetrospectiveCampaignContractError(ValueError):
    """Raised when a campaign cannot unambiguously bind its source cohort."""


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetrospectiveCampaignContractError(
            f"{name} must be a non-empty string."
        )
    return value.strip()


def _sha256(value: object, *, name: str) -> str:
    result = _required(value, name=name).lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise RetrospectiveCampaignContractError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return result


def _hashes(values: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise RetrospectiveCampaignContractError(
            f"{name} must be a tuple or list."
        )
    result = tuple(sorted(_sha256(value, name=name) for value in values))
    if not result or len(result) != len(set(result)):
        raise RetrospectiveCampaignContractError(
            f"{name} must be non-empty and unique."
        )
    return result


def _finite(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RetrospectiveCampaignContractError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise RetrospectiveCampaignContractError(f"{name} must be finite.")
    return result


def _verify_hash(payload: Mapping[str, Any], actual: str, *, name: str) -> None:
    declared = payload.get("semantic_content_hash")
    if declared is not None and str(declared) != actual:
        raise RetrospectiveCampaignContractError(f"{name} hash mismatch.")


@dataclass(frozen=True)
class RetrospectiveAnnotationCampaignV1:
    """One immutable selection event inside a dataset-local semantic layer."""

    semantic_layer_id: str
    campaign_id: str
    campaign_kind: str
    source_dataset_base_fingerprint: str
    cohort_plan_hash: str
    eligible_episode_identity_hashes: tuple[str, ...]
    selected_episode_identity_hashes: tuple[str, ...]
    migration_actor_id: str
    migration_created_at: float
    migration_clock_domain_id: str
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=RETROSPECTIVE_ANNOTATION_CAMPAIGN_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "semantic_layer_id",
            "campaign_id",
            "migration_actor_id",
            "migration_clock_domain_id",
        ):
            object.__setattr__(
                self, name, _required(getattr(self, name), name=name)
            )
        kind = _required(self.campaign_kind, name="campaign_kind").upper()
        if kind not in _CAMPAIGN_KINDS:
            raise RetrospectiveCampaignContractError(
                f"Unsupported campaign_kind: {kind!r}."
            )
        object.__setattr__(self, "campaign_kind", kind)
        for name in ("source_dataset_base_fingerprint", "cohort_plan_hash"):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), name=name)
            )
        eligible = _hashes(
            self.eligible_episode_identity_hashes,
            name="eligible_episode_identity_hashes",
        )
        selected = _hashes(
            self.selected_episode_identity_hashes,
            name="selected_episode_identity_hashes",
        )
        if not set(selected).issubset(eligible):
            raise RetrospectiveCampaignContractError(
                "Every selected Episode must belong to the eligible cohort."
            )
        object.__setattr__(self, "eligible_episode_identity_hashes", eligible)
        object.__setattr__(self, "selected_episode_identity_hashes", selected)
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
                domain=_CAMPAIGN_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "semantic_layer_id": self.semantic_layer_id,
            "campaign_id": self.campaign_id,
            "campaign_kind": self.campaign_kind,
            "source_dataset_base_fingerprint": (
                self.source_dataset_base_fingerprint
            ),
            "cohort_plan_hash": self.cohort_plan_hash,
            "eligible_episode_identity_hashes": (
                self.eligible_episode_identity_hashes
            ),
            "selected_episode_identity_hashes": (
                self.selected_episode_identity_hashes
            ),
            "migration_actor_id": self.migration_actor_id,
            "migration_created_at": self.migration_created_at,
            "migration_clock_domain_id": self.migration_clock_domain_id,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> "RetrospectiveAnnotationCampaignV1":
        if payload.get("schema") != RETROSPECTIVE_ANNOTATION_CAMPAIGN_SCHEMA_V1:
            raise RetrospectiveCampaignContractError(
                "Unsupported retrospective campaign schema."
            )
        result = cls(
            semantic_layer_id=payload.get("semantic_layer_id", ""),
            campaign_id=payload.get("campaign_id", ""),
            campaign_kind=payload.get("campaign_kind", ""),
            source_dataset_base_fingerprint=payload.get(
                "source_dataset_base_fingerprint", ""
            ),
            cohort_plan_hash=payload.get("cohort_plan_hash", ""),
            eligible_episode_identity_hashes=tuple(
                payload.get("eligible_episode_identity_hashes", ())
            ),
            selected_episode_identity_hashes=tuple(
                payload.get("selected_episode_identity_hashes", ())
            ),
            migration_actor_id=payload.get("migration_actor_id", ""),
            migration_created_at=payload.get("migration_created_at", 0.0),
            migration_clock_domain_id=payload.get(
                "migration_clock_domain_id", ""
            ),
        )
        _verify_hash(payload, result.semantic_content_hash, name="Campaign")
        return result


@dataclass(frozen=True)
class RetrospectiveCampaignEpisodeSealV1:
    episode_capture_identity_hash: str
    migration_manifest_hash: str
    schema: str = field(
        default=RETROSPECTIVE_CAMPAIGN_EPISODE_SEAL_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "episode_capture_identity_hash",
            "migration_manifest_hash",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), name=name)
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "episode_capture_identity_hash": self.episode_capture_identity_hash,
            "migration_manifest_hash": self.migration_manifest_hash,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> "RetrospectiveCampaignEpisodeSealV1":
        if payload.get("schema") != RETROSPECTIVE_CAMPAIGN_EPISODE_SEAL_SCHEMA_V1:
            raise RetrospectiveCampaignContractError(
                "Unsupported campaign Episode-seal schema."
            )
        return cls(
            episode_capture_identity_hash=payload.get(
                "episode_capture_identity_hash", ""
            ),
            migration_manifest_hash=payload.get("migration_manifest_hash", ""),
        )


@dataclass(frozen=True)
class RetrospectiveCampaignPublicationReceiptV1:
    campaign_definition_hash: str
    episode_seals: tuple[RetrospectiveCampaignEpisodeSealV1, ...]
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=RETROSPECTIVE_CAMPAIGN_PUBLICATION_RECEIPT_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "campaign_definition_hash",
            _sha256(
                self.campaign_definition_hash,
                name="campaign_definition_hash",
            ),
        )
        seals = tuple(
            sorted(
                self.episode_seals,
                key=lambda value: value.episode_capture_identity_hash,
            )
        )
        if not seals or any(
            not isinstance(value, RetrospectiveCampaignEpisodeSealV1)
            for value in seals
        ):
            raise RetrospectiveCampaignContractError(
                "episode_seals must contain at least one Episode seal."
            )
        identities = tuple(
            value.episode_capture_identity_hash for value in seals
        )
        if len(identities) != len(set(identities)):
            raise RetrospectiveCampaignContractError(
                "Campaign receipt contains duplicate Episode identities."
            )
        object.__setattr__(self, "episode_seals", seals)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=_RECEIPT_HASH_DOMAIN,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "campaign_definition_hash": self.campaign_definition_hash,
            "episode_seals": tuple(
                value.to_payload() for value in self.episode_seals
            ),
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> "RetrospectiveCampaignPublicationReceiptV1":
        if (
            payload.get("schema")
            != RETROSPECTIVE_CAMPAIGN_PUBLICATION_RECEIPT_SCHEMA_V1
        ):
            raise RetrospectiveCampaignContractError(
                "Unsupported campaign publication-receipt schema."
            )
        raw_seals = payload.get("episode_seals", ())
        if not isinstance(raw_seals, (tuple, list)):
            raise RetrospectiveCampaignContractError(
                "episode_seals must be a tuple or list."
            )
        result = cls(
            campaign_definition_hash=payload.get(
                "campaign_definition_hash", ""
            ),
            episode_seals=tuple(
                RetrospectiveCampaignEpisodeSealV1.from_payload(value)
                for value in raw_seals
                if isinstance(value, Mapping)
            ),
        )
        if len(result.episode_seals) != len(raw_seals):
            raise RetrospectiveCampaignContractError(
                "Every episode_seal must be an object."
            )
        _verify_hash(
            payload,
            result.semantic_content_hash,
            name="Campaign publication receipt",
        )
        return result


__all__ = [
    "RETROSPECTIVE_ANNOTATION_CAMPAIGN_FILE_DOMAIN_V1",
    "RETROSPECTIVE_ANNOTATION_CAMPAIGN_SCHEMA_V1",
    "RETROSPECTIVE_CAMPAIGN_EPISODE_SEAL_SCHEMA_V1",
    "RETROSPECTIVE_CAMPAIGN_PUBLICATION_RECEIPT_FILE_DOMAIN_V1",
    "RETROSPECTIVE_CAMPAIGN_PUBLICATION_RECEIPT_SCHEMA_V1",
    "RetrospectiveAnnotationCampaignV1",
    "RetrospectiveCampaignContractError",
    "RetrospectiveCampaignEpisodeSealV1",
    "RetrospectiveCampaignPublicationReceiptV1",
]
