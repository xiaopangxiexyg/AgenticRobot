"""Declared information demand for one control-composition runtime.

The plan is independent of activation identity and dataset recording.  It lets
source adapters know which immutable context channels may affect a decision
before they acquire, copy, freeze, or admit optional data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from dataset_contracts.canonical_encoding import canonical_content_hash


COMPOSITION_INFORMATION_PLAN_HASH_DOMAIN_V1 = (
    "dexteleop.composition_information_plan.v1"
)
CONTEXT_PAYLOAD_ACCESS_MODES_V1 = frozenset(
    {"metadata_only", "inline", "blob_ref"}
)
COMPOSITION_DIAGNOSTIC_LEVELS_V1 = frozenset({"core", "debug"})


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


@dataclass(frozen=True)
class CompositionContextRequirement:
    """One latest-value context channel admitted for composition use.

    Version 1 deliberately supports the existing latest-value snapshot
    semantics only.  History/window selection will require a later additive
    contract rather than silently changing this plan.
    """

    channel: str
    accepted_payload_contract_hashes: tuple[str, ...]
    payload_access: str = "inline"
    maximum_age_sec: float | None = None

    def __post_init__(self) -> None:
        channel = _required(self.channel, name="channel")
        hashes = tuple(
            _required(value, name="accepted_payload_contract_hash")
            for value in self.accepted_payload_contract_hashes
        )
        if not hashes or len(hashes) != len(set(hashes)):
            raise ValueError(
                "accepted_payload_contract_hashes must be non-empty and unique."
            )
        access = _required(self.payload_access, name="payload_access")
        if access not in CONTEXT_PAYLOAD_ACCESS_MODES_V1:
            raise ValueError("Unsupported context payload access mode.")
        maximum_age = self.maximum_age_sec
        if maximum_age is not None:
            maximum_age = float(maximum_age)
            if not math.isfinite(maximum_age) or maximum_age < 0.0:
                raise ValueError("maximum_age_sec must be finite and non-negative.")
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "accepted_payload_contract_hashes", hashes)
        object.__setattr__(self, "payload_access", access)
        object.__setattr__(self, "maximum_age_sec", maximum_age)

    def accepts_payload_contract(self, payload_contract_hash: str) -> bool:
        return (
            _required(payload_contract_hash, name="payload_contract_hash")
            in self.accepted_payload_contract_hashes
        )


@dataclass(frozen=True)
class CompositionInformationPlan:
    """Immutable pre-decision information demand.

    The plan controls composition acquisition/admission and optional derived
    computation.  It never describes dataset streams, media persistence, WAL,
    episode membership, or any recording schema.
    """

    plan_id: str
    context_requirements: tuple[CompositionContextRequirement, ...] = ()
    source_difference_spec_hash: str | None = None
    diagnostic_level: str = "core"
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        plan_id = _required(self.plan_id, name="plan_id")
        requirements = tuple(self.context_requirements)
        if any(
            not isinstance(item, CompositionContextRequirement)
            for item in requirements
        ):
            raise TypeError(
                "context_requirements must contain CompositionContextRequirement."
            )
        channels = tuple(item.channel for item in requirements)
        if len(channels) != len(set(channels)):
            raise ValueError("Composition information channels must be unique.")
        difference_hash = (
            None
            if self.source_difference_spec_hash is None
            else _required(
                self.source_difference_spec_hash,
                name="source_difference_spec_hash",
            )
        )
        diagnostic_level = _required(
            self.diagnostic_level,
            name="diagnostic_level",
        )
        if diagnostic_level not in COMPOSITION_DIAGNOSTIC_LEVELS_V1:
            raise ValueError("Unsupported composition diagnostic level.")
        payload = {
            "plan_id": plan_id,
            "context_requirements": tuple(
                {
                    "channel": item.channel,
                    "accepted_payload_contract_hashes": (
                        item.accepted_payload_contract_hashes
                    ),
                    "payload_access": item.payload_access,
                    "maximum_age_sec": item.maximum_age_sec,
                }
                for item in requirements
            ),
            "source_difference_spec_hash": difference_hash,
            "diagnostic_level": diagnostic_level,
        }
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "context_requirements", requirements)
        object.__setattr__(
            self,
            "source_difference_spec_hash",
            difference_hash,
        )
        object.__setattr__(self, "diagnostic_level", diagnostic_level)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=COMPOSITION_INFORMATION_PLAN_HASH_DOMAIN_V1,
            ),
        )

    @property
    def required_context_channels(self) -> tuple[str, ...]:
        return tuple(item.channel for item in self.context_requirements)

    def requirement_for_channel(
        self,
        channel: str,
    ) -> CompositionContextRequirement | None:
        requested = _required(channel, name="channel")
        return next(
            (
                item
                for item in self.context_requirements
                if item.channel == requested
            ),
            None,
        )


__all__ = [
    "COMPOSITION_DIAGNOSTIC_LEVELS_V1",
    "COMPOSITION_INFORMATION_PLAN_HASH_DOMAIN_V1",
    "CONTEXT_PAYLOAD_ACCESS_MODES_V1",
    "CompositionContextRequirement",
    "CompositionInformationPlan",
]
