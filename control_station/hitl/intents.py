"""Typed operator intents for exclusive Human/Policy source selection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from dataset_contracts.canonical_encoding import canonical_content_hash


EXCLUSIVE_HITL_SOURCES = ("human", "policy")
EXCLUSIVE_HITL_INTENT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.exclusive_hitl_intent.v1"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class ExclusiveHITLSourceIntent:
    """One edge-triggered request; it carries no control authority."""

    requested_source: str
    requested_at: float
    clock_domain_id: str
    input_source: str
    input_event: str
    sequence: int
    intent_id: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        source = _required(self.requested_source, name="requested_source")
        if source not in EXCLUSIVE_HITL_SOURCES:
            raise ValueError("requested_source must be 'human' or 'policy'.")
        requested_at = float(self.requested_at)
        if not math.isfinite(requested_at):
            raise ValueError("requested_at must be finite.")
        sequence = int(self.sequence)
        if sequence < 0 or sequence != self.sequence:
            raise ValueError("sequence must be a non-negative integer.")
        payload = {
            "requested_source": source,
            "requested_at": requested_at,
            "clock_domain_id": _required(
                self.clock_domain_id,
                name="clock_domain_id",
            ),
            "input_source": _required(self.input_source, name="input_source"),
            "input_event": _required(self.input_event, name="input_event"),
            "sequence": sequence,
        }
        digest = canonical_content_hash(
            payload,
            domain=EXCLUSIVE_HITL_INTENT_HASH_DOMAIN_V1,
        )
        object.__setattr__(self, "requested_source", source)
        object.__setattr__(self, "requested_at", requested_at)
        object.__setattr__(self, "clock_domain_id", payload["clock_domain_id"])
        object.__setattr__(self, "input_source", payload["input_source"])
        object.__setattr__(self, "input_event", payload["input_event"])
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "intent_id", f"exclusive-hitl-intent:{digest}")
        object.__setattr__(self, "semantic_content_hash", digest)

    @property
    def authoritative(self) -> bool:
        return False


__all__ = [
    "EXCLUSIVE_HITL_INTENT_HASH_DOMAIN_V1",
    "EXCLUSIVE_HITL_SOURCES",
    "ExclusiveHITLSourceIntent",
]
