"""Protocol-neutral domain events emitted by control composition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash


COMPOSITION_ACTIVATION_BINDING_HASH_DOMAIN_V2 = (
    "dexteleop.composition_activation_binding.v2"
)
COMPOSITION_SOURCE_ELIGIBILITY_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.composition_source_eligibility_binding.v1"
)
COMPOSITION_INPUT_SEMANTIC_BINDING_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.composition_input_semantic_binding.development"
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
        if kind not in {"input", "snapshot", "evaluation"}:
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
class CompositionInputSemanticBinding:
    """Record non-debug semantic identities carried by one admitted input.

    Bindings stay separate from the established composition-input trace record,
    while the input's own semantic hash still commits to them.
    """

    composition_input_id: str
    composition_input_hash: str
    binding_id_by_kind: Mapping[str, str]
    binding_hash_by_kind: Mapping[str, str]
    semantic_content_hash: str = ""

    def __post_init__(self) -> None:
        binding_ids = {
            _required(kind, name="binding_kind"): _required(
                value,
                name="binding_id",
            )
            for kind, value in dict(self.binding_id_by_kind).items()
        }
        binding_hashes = {
            _required(kind, name="binding_kind"): _required(
                value,
                name="binding_hash",
            )
            for kind, value in dict(self.binding_hash_by_kind).items()
        }
        if not binding_ids:
            raise ValueError("Composition input semantic binding must be non-empty.")
        if set(binding_ids) != set(binding_hashes):
            raise ValueError("Composition input semantic binding kinds must match.")
        payload = {
            "composition_input_id": _required(
                self.composition_input_id,
                name="composition_input_id",
            ),
            "composition_input_hash": _required(
                self.composition_input_hash,
                name="composition_input_hash",
            ),
            "binding_id_by_kind": dict(sorted(binding_ids.items())),
            "binding_hash_by_kind": dict(sorted(binding_hashes.items())),
        }
        digest = canonical_content_hash(
            payload,
            domain=COMPOSITION_INPUT_SEMANTIC_BINDING_HASH_DOMAIN_DEVELOPMENT,
        )
        if self.semantic_content_hash and self.semantic_content_hash != digest:
            raise ValueError("Composition input semantic binding hash mismatch.")
        object.__setattr__(self, "composition_input_id", payload["composition_input_id"])
        object.__setattr__(
            self,
            "composition_input_hash",
            payload["composition_input_hash"],
        )
        object.__setattr__(
            self,
            "binding_id_by_kind",
            MappingProxyType(payload["binding_id_by_kind"]),
        )
        object.__setattr__(
            self,
            "binding_hash_by_kind",
            MappingProxyType(payload["binding_hash_by_kind"]),
        )
        object.__setattr__(self, "semantic_content_hash", digest)

    @property
    def binding_id(self) -> str:
        return f"composition-input-semantic-binding:{self.semantic_content_hash}"


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


@dataclass(frozen=True)
class CompositionEventEmissionFailure:
    """Bounded diagnostic for a passive domain-event observer failure.

    Event observation is intentionally outside composition authority.  A
    failing trace/WAL adapter therefore cannot rewrite or veto an already
    admitted composition fact, but the failure must remain inspectable by an
    application profile that independently requires capture completeness.
    """

    event_kinds: tuple[str, ...]
    error_type: str
    message: str

    def __post_init__(self) -> None:
        kinds = tuple(
            _required(value, name="event_kind") for value in self.event_kinds
        )
        if not kinds:
            raise ValueError("An emission failure must identify at least one event.")
        object.__setattr__(self, "event_kinds", kinds)
        object.__setattr__(
            self,
            "error_type",
            _required(self.error_type, name="error_type"),
        )
        object.__setattr__(self, "message", str(self.message))


__all__ = [
    "COMPOSITION_ACTIVATION_BINDING_HASH_DOMAIN_V2",
    "COMPOSITION_INPUT_SEMANTIC_BINDING_HASH_DOMAIN_DEVELOPMENT",
    "COMPOSITION_SOURCE_ELIGIBILITY_BINDING_HASH_DOMAIN_V1",
    "CompositionActivationBinding",
    "CompositionDomainEvent",
    "CompositionEventEmissionFailure",
    "CompositionInputSemanticBinding",
    "CompositionSourceEligibilityBinding",
]
