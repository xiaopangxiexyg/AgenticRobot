"""Shared episode identity for additive capture capabilities.

The dataset episode lifecycle owner creates this identity once and passes it
to subordinate writers such as ControlTrace, native streams, embodiment
bindings, and task-semantics capture.  This contract does not create, begin,
finalize, abort, or commit an episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from .canonical_encoding import canonical_content_hash


EPISODE_CAPTURE_IDENTITY_SCHEMA_V1 = "dexteleop.episode_capture_identity.v1"
EPISODE_CAPTURE_IDENTITY_HASH_DOMAIN_V1 = (
    "dexteleop.episode_capture_identity.semantic.v1"
)


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _safe_episode_uuid(value: object) -> str:
    text = _required(value, name="episode_uuid")
    if text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError("episode_uuid must be safe for dataset-relative storage.")
    return text


def _finite_time(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class EpisodeCaptureIdentity:
    """One immutable identity minted by the authoritative episode owner."""

    episode_uuid: str
    legacy_episode_index: int
    lifecycle_owner_id: str
    prepared_at: float
    prepared_clock_domain_id: str
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=EPISODE_CAPTURE_IDENTITY_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        episode_uuid = _safe_episode_uuid(self.episode_uuid)
        index = int(self.legacy_episode_index)
        if isinstance(self.legacy_episode_index, bool) or index < 0:
            raise ValueError("legacy_episode_index must be a non-negative integer.")
        owner = _required(self.lifecycle_owner_id, name="lifecycle_owner_id")
        prepared_at = _finite_time(self.prepared_at, name="prepared_at")
        clock = _required(
            self.prepared_clock_domain_id,
            name="prepared_clock_domain_id",
        )
        object.__setattr__(self, "episode_uuid", episode_uuid)
        object.__setattr__(self, "legacy_episode_index", index)
        object.__setattr__(self, "lifecycle_owner_id", owner)
        object.__setattr__(self, "prepared_at", prepared_at)
        object.__setattr__(self, "prepared_clock_domain_id", clock)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=EPISODE_CAPTURE_IDENTITY_HASH_DOMAIN_V1,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "episode_uuid": self.episode_uuid,
            "legacy_episode_index": self.legacy_episode_index,
            "lifecycle_owner_id": self.lifecycle_owner_id,
            "prepared_at": self.prepared_at,
            "prepared_clock_domain_id": self.prepared_clock_domain_id,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EpisodeCaptureIdentity":
        if payload.get("schema") != EPISODE_CAPTURE_IDENTITY_SCHEMA_V1:
            raise ValueError("Unsupported episode capture identity schema.")
        result = cls(
            episode_uuid=payload.get("episode_uuid", ""),
            legacy_episode_index=payload.get("legacy_episode_index", -1),
            lifecycle_owner_id=payload.get("lifecycle_owner_id", ""),
            prepared_at=payload.get("prepared_at", float("nan")),
            prepared_clock_domain_id=payload.get(
                "prepared_clock_domain_id",
                "",
            ),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ValueError("Episode capture identity semantic hash mismatch.")
        return result


__all__ = [
    "EPISODE_CAPTURE_IDENTITY_HASH_DOMAIN_V1",
    "EPISODE_CAPTURE_IDENTITY_SCHEMA_V1",
    "EpisodeCaptureIdentity",
]
