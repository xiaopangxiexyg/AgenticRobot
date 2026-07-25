"""Protocol-neutral domain events emitted by control composition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash


COMPOSITION_ACTIVATION_BINDING_HASH_DOMAIN_V2 = (
    "dexteleop.composition_activation_binding.v2"
)
COMPOSITION_SOURCE_ELIGIBILITY_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.composition_source_eligibility_binding.v1"
)


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


@dataclass(frozen=True)
class CompositionActivationBinding:
    """Domain lineage binding kept separate from activation-v2 identity."""

    subject_kind: str
    subject_id: str
    subject_hash: str
    activation_id: str
    activation_hash: str
    semantic_content_hash: str = ""

    def __post_init__(self) -> None:
        kind = _required(self.subject_kind, name="subject_kind")
        if kind not in {"input", "snapshot", "result"}:
            raise ValueError("Unsupported composition activation subject kind.")
        payload = {
            "subject_kind": kind,
            "subject_id": _required(self.subject_id, name="subject_id"),
            "subject_hash": _required(self.subject_hash, name="subject_hash"),
            "activation_id": _required(self.activation_id, name="activation_id"),
            "activation_hash": _required(
                self.activation_hash,
                name="activation_hash",
            ),
        }
        digest = canonical_content_hash(
            payload,
            domain=COMPOSITION_ACTIVATION_BINDING_HASH_DOMAIN_V2,
        )
        if self.semantic_content_hash and self.semantic_content_hash != digest:
            raise ValueError("Composition activation binding hash mismatch.")
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "semantic_content_hash", digest)

    @property
    def binding_id(self) -> str:
        return f"composition-activation-binding:{self.semantic_content_hash}"


@dataclass(frozen=True)
class CompositionSourceEligibilityBinding:
    """Bind one input snapshot to the exact same-cut eligibility snapshot."""

    composition_input_snapshot_id: str
    composition_input_snapshot_hash: str
    source_eligibility_snapshot_id: str
    source_eligibility_snapshot_hash: str
    semantic_content_hash: str = ""

    def __post_init__(self) -> None:
        payload = {
            "composition_input_snapshot_id": _required(
                self.composition_input_snapshot_id,
                name="composition_input_snapshot_id",
            ),
            "composition_input_snapshot_hash": _required(
                self.composition_input_snapshot_hash,
                name="composition_input_snapshot_hash",
            ),
            "source_eligibility_snapshot_id": _required(
                self.source_eligibility_snapshot_id,
                name="source_eligibility_snapshot_id",
            ),
            "source_eligibility_snapshot_hash": _required(
                self.source_eligibility_snapshot_hash,
                name="source_eligibility_snapshot_hash",
            ),
        }
        digest = canonical_content_hash(
            payload,
            domain=COMPOSITION_SOURCE_ELIGIBILITY_BINDING_HASH_DOMAIN_V1,
        )
        if self.semantic_content_hash and self.semantic_content_hash != digest:
            raise ValueError("Composition source eligibility binding hash mismatch.")
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "semantic_content_hash", digest)

    @property
    def binding_id(self) -> str:
        return (
            "composition-source-eligibility-binding:"
            f"{self.semantic_content_hash}"
        )


@dataclass(frozen=True)
class CompositionDomainEvent:
    """One immutable domain occurrence, before any recording projection."""

    event_kind: str
    subject: Any
    occurred_at: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_kind",
            _required(self.event_kind, name="event_kind"),
        )
        occurred_at = float(self.occurred_at)
        if not math.isfinite(occurred_at):
            raise ValueError("occurred_at must be finite.")
        object.__setattr__(self, "occurred_at", occurred_at)
        if self.subject is None:
            raise ValueError("Composition domain event subject is required.")


__all__ = [
    "COMPOSITION_ACTIVATION_BINDING_HASH_DOMAIN_V2",
    "COMPOSITION_SOURCE_ELIGIBILITY_BINDING_HASH_DOMAIN_V1",
    "CompositionActivationBinding",
    "CompositionDomainEvent",
    "CompositionSourceEligibilityBinding",
]
