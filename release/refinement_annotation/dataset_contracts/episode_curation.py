"""Contract for the optional DexTeleop Plus episode curation overlay.

The overlay is deliberately episode scoped and lives outside the raw dataset.
It can therefore be added to a v2.3 or v2.4 dataset without changing parquet
columns, writer lifecycle, or the interpretation of an existing row.  Missing
overlay records are valid and mean that the curation value is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from .canonical_encoding import canonical_content_hash


PLUS_EPISODE_CURATION_SCHEMA_V1 = "dexteleop_plus_episode_curation_v1"
PLUS_ANNOTATION_SET_SCHEMA_V1 = "dexteleop_plus_annotation_set_v1"
PLUS_TASK_PROFILE_SCHEMA_V1 = "dexteleop_plus_task_profile_v1"
PLUS_SOURCE_FINGERPRINT_SCHEMA_V1 = "dexteleop_plus_source_fingerprint_v1"

PLUS_DISPOSITIONS = ("keep_nominal", "keep_recovery", "exclude", "unknown")
PLUS_REASON_CODES = (
    "task_failure",
    "operator_recovery",
    "system_issue",
    "safety_abort",
    "other",
    "unknown",
)
PLUS_REVIEW_STATUSES = (
    "confirmed",
    "provisional",
    "conflicted",
    "rejected",
    "unknown",
)
PLUS_ANNOTATION_SET_STATUSES = (
    "draft",
    "complete",
    "partial",
    "invalid",
    "superseded",
)
PLUS_SOURCE_BINDINGS = (
    "v23_compat_content_locator_v1",
    "v24_committed_capture_membership_v1",
    "v24_compatibility_content_locator_v1",
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EpisodeCurationContractError(ValueError):
    """Raised when an optional Plus record is malformed."""


def _clean_text(value: Any, *, field: str, allow_empty: bool = False) -> str:
    text = str(value or "").strip()
    if not text and not allow_empty:
        raise EpisodeCurationContractError(f"{field} must be non-empty")
    return text


def _validate_id(value: str, *, field: str) -> str:
    if not _ID_RE.fullmatch(value):
        raise EpisodeCurationContractError(
            f"{field} must match {_ID_RE.pattern!r}; free-form text is not allowed"
        )
    return value


def _validate_sha256(value: str, *, field: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise EpisodeCurationContractError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class TaskVariant:
    """One pre-registered, finite task variant shown by the collection GUI."""

    variant_id: str
    label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant_id", _validate_id(str(self.variant_id).strip(), field="variant_id"))
        label = _clean_text(self.label, field="label")
        object.__setattr__(self, "label", label)

    def to_dict(self) -> dict[str, str]:
        return {"variant_id": self.variant_id, "label": self.label}


@dataclass(frozen=True)
class TaskVariantProfile:
    """Small allow-list used to prevent free-form GUI annotations."""

    task_profile_id: str
    variants: tuple[TaskVariant, ...]
    task_id: str = ""
    schema_version: str = PLUS_TASK_PROFILE_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema_version != PLUS_TASK_PROFILE_SCHEMA_V1:
            raise EpisodeCurationContractError(
                f"unsupported task profile schema: {self.schema_version!r}"
            )
        object.__setattr__(
            self,
            "task_profile_id",
            _validate_id(str(self.task_profile_id).strip(), field="task_profile_id"),
        )
        object.__setattr__(self, "task_id", str(self.task_id or "").strip())
        variants = tuple(self.variants)
        if not variants:
            raise EpisodeCurationContractError("task profile must declare at least one variant")
        ids = [item.variant_id for item in variants]
        if len(set(ids)) != len(ids):
            raise EpisodeCurationContractError("task profile variant_id values must be unique")
        object.__setattr__(self, "variants", variants)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskVariantProfile":
        if not isinstance(payload, Mapping):
            raise EpisodeCurationContractError("task profile must be an object")
        variants_payload = payload.get("variants", ())
        if not isinstance(variants_payload, (list, tuple)):
            raise EpisodeCurationContractError("task profile variants must be a list")
        variants = tuple(
            TaskVariant(
                variant_id=item.get("variant_id", item.get("id", "")),
                label=item.get("label", item.get("name", "")),
            )
            for item in variants_payload
            if isinstance(item, Mapping)
        )
        if len(variants) != len(variants_payload):
            raise EpisodeCurationContractError("each task profile variant must be an object")
        return cls(
            task_profile_id=payload.get("task_profile_id", ""),
            task_id=payload.get("task_id", ""),
            variants=variants,
            schema_version=str(payload.get("schema_version", PLUS_TASK_PROFILE_SCHEMA_V1)),
        )

    def has_variant(self, variant_id: str | None) -> bool:
        if variant_id is None or not str(variant_id).strip():
            return False
        return str(variant_id).strip() in {item.variant_id for item in self.variants}

    def require_variant(self, variant_id: str | None) -> str:
        value = _clean_text(variant_id, field="variant_id")
        if not self.has_variant(value):
            raise EpisodeCurationContractError(
                f"variant_id {value!r} is not registered in task profile {self.task_profile_id!r}"
            )
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_profile_id": self.task_profile_id,
            "task_id": self.task_id,
            "variants": [item.to_dict() for item in self.variants],
        }


@dataclass(frozen=True)
class EpisodeCurationRecord:
    """One immutable episode-level curation fact.

    ``legacy_episode_index`` is intentionally retained: it is the stable
    locator available in existing v2.3 datasets and in explicit v2.4
    compatibility views.  The hash fields prevent a record from silently
    attaching to a different source dataset after a copy or re-export.
    """

    annotation_episode_id: str
    legacy_episode_index: int
    source_dataset_base_fingerprint: str
    source_episode_manifest_hash: str | None
    source_schema: str
    source_binding: str
    disposition: str = "unknown"
    reason_code: str = "unknown"
    task_profile_id: str | None = None
    variant_id: str | None = None
    review_status: str = "confirmed"
    schema_version: str = PLUS_EPISODE_CURATION_SCHEMA_V1
    annotation_set_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != PLUS_EPISODE_CURATION_SCHEMA_V1:
            raise EpisodeCurationContractError(
                f"unsupported episode curation schema: {self.schema_version!r}"
            )
        object.__setattr__(
            self,
            "annotation_episode_id",
            _validate_sha256(str(self.annotation_episode_id).strip(), field="annotation_episode_id"),
        )
        index = int(self.legacy_episode_index)
        if index < 0:
            raise EpisodeCurationContractError("legacy_episode_index must be non-negative")
        object.__setattr__(self, "legacy_episode_index", index)
        object.__setattr__(
            self,
            "source_dataset_base_fingerprint",
            _validate_sha256(
                str(self.source_dataset_base_fingerprint).strip(),
                field="source_dataset_base_fingerprint",
            ),
        )
        manifest_hash = self.source_episode_manifest_hash
        if manifest_hash is not None:
            manifest_hash = _validate_sha256(str(manifest_hash).strip(), field="source_episode_manifest_hash")
        object.__setattr__(self, "source_episode_manifest_hash", manifest_hash)
        source_schema = _clean_text(self.source_schema, field="source_schema")
        source_binding = _clean_text(self.source_binding, field="source_binding")
        if source_binding not in PLUS_SOURCE_BINDINGS:
            raise EpisodeCurationContractError(f"unsupported source_binding: {source_binding!r}")
        object.__setattr__(self, "source_schema", source_schema)
        object.__setattr__(self, "source_binding", source_binding)
        disposition = str(self.disposition or "unknown").strip()
        reason_code = str(self.reason_code or "unknown").strip()
        review_status = str(self.review_status or "unknown").strip()
        if disposition not in PLUS_DISPOSITIONS:
            raise EpisodeCurationContractError(f"unsupported disposition: {disposition!r}")
        if reason_code not in PLUS_REASON_CODES:
            raise EpisodeCurationContractError(f"unsupported reason_code: {reason_code!r}")
        if review_status not in PLUS_REVIEW_STATUSES:
            raise EpisodeCurationContractError(f"unsupported review_status: {review_status!r}")
        if disposition in {"keep_nominal", "unknown"} and reason_code != "unknown":
            raise EpisodeCurationContractError(
                "reason_code must be unknown for keep_nominal or unknown disposition"
            )
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "review_status", review_status)
        for field_name in ("task_profile_id", "variant_id", "annotation_set_id"):
            value = getattr(self, field_name)
            if value is not None and str(value).strip():
                object.__setattr__(self, field_name, _validate_id(str(value).strip(), field=field_name))
            elif field_name != "annotation_set_id":
                object.__setattr__(self, field_name, None)
            else:
                object.__setattr__(self, field_name, "")
        if self.variant_id is not None and self.task_profile_id is None:
            raise EpisodeCurationContractError("variant_id requires task_profile_id")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EpisodeCurationRecord":
        if not isinstance(payload, Mapping):
            raise EpisodeCurationContractError("episode curation record must be an object")
        return cls(
            schema_version=str(payload.get("schema_version", PLUS_EPISODE_CURATION_SCHEMA_V1)),
            annotation_episode_id=payload.get("annotation_episode_id", ""),
            annotation_set_id=payload.get("annotation_set_id", ""),
            legacy_episode_index=payload.get("legacy_episode_index", -1),
            source_dataset_base_fingerprint=payload.get("source_dataset_base_fingerprint", ""),
            source_episode_manifest_hash=payload.get("source_episode_manifest_hash"),
            source_schema=payload.get("source_schema", ""),
            source_binding=payload.get("source_binding", ""),
            disposition=payload.get("disposition", "unknown"),
            reason_code=payload.get("reason_code", "unknown"),
            task_profile_id=payload.get("task_profile_id"),
            variant_id=payload.get("variant_id"),
            review_status=payload.get("review_status", "confirmed"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "annotation_set_id": self.annotation_set_id or None,
            "annotation_episode_id": self.annotation_episode_id,
            "legacy_episode_index": self.legacy_episode_index,
            "source_dataset_base_fingerprint": self.source_dataset_base_fingerprint,
            "source_episode_manifest_hash": self.source_episode_manifest_hash,
            "source_schema": self.source_schema,
            "source_binding": self.source_binding,
            "disposition": self.disposition,
            "reason_code": self.reason_code,
            "task_profile_id": self.task_profile_id,
            "variant_id": self.variant_id,
            "review_status": self.review_status,
        }


def annotation_episode_id(
    *,
    source_dataset_base_fingerprint: str,
    source_episode_manifest_hash: str | None,
    legacy_episode_index: int,
) -> str:
    """Derive an external identity without changing the raw dataset contract."""

    return canonical_content_hash(
        {
            "source_dataset_base_fingerprint": source_dataset_base_fingerprint,
            "source_episode_manifest_hash": source_episode_manifest_hash,
            "legacy_episode_index": int(legacy_episode_index),
        },
        domain="dexteleop_plus_annotation_episode_identity_v1",
    )


def annotation_set_id(
    *,
    source_dataset_base_fingerprint: str,
    source_binding: str,
    task_profile_id: str | None,
) -> str:
    return canonical_content_hash(
        {
            "source_dataset_base_fingerprint": source_dataset_base_fingerprint,
            "source_binding": source_binding,
            "task_profile_id": task_profile_id,
        },
        domain="dexteleop_plus_annotation_set_identity_v1",
    )


__all__ = [
    "EpisodeCurationContractError",
    "EpisodeCurationRecord",
    "PLUS_ANNOTATION_SET_SCHEMA_V1",
    "PLUS_ANNOTATION_SET_STATUSES",
    "PLUS_DISPOSITIONS",
    "PLUS_EPISODE_CURATION_SCHEMA_V1",
    "PLUS_REASON_CODES",
    "PLUS_REVIEW_STATUSES",
    "PLUS_SOURCE_BINDINGS",
    "PLUS_SOURCE_FINGERPRINT_SCHEMA_V1",
    "PLUS_TASK_PROFILE_SCHEMA_V1",
    "TaskVariant",
    "TaskVariantProfile",
    "annotation_episode_id",
    "annotation_set_id",
]
