"""Immutable input/state and direct-event evidence for trigger providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.control_admission import ControlCut
from robot_motion.runtime.immutable import deep_freeze

from control_composition.activation import CompositionActivationScope

from ._validation import finite, lineage, nonnegative_int, required


COMPOSITION_TRIGGER_ORIGIN_CONDITION = "condition"
COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT = "direct_event"
COMPOSITION_TRIGGER_ORIGINS_DEVELOPMENT = frozenset(
    {
        COMPOSITION_TRIGGER_ORIGIN_CONDITION,
        COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT,
    }
)

COMPOSITION_TRIGGER_EVALUATION_BASIS_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.composition_trigger_evaluation_basis.development"
)
COMPOSITION_TRIGGER_EVIDENCE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.composition_trigger_evidence.development"
)
COMPOSITION_TRIGGER_CONTROL_CUT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.composition_trigger_control_cut.development"
)


def composition_trigger_control_cut_hash(cut: ControlCut) -> str:
    """Return the exact receiver cut identity used by one trigger decision."""

    if not isinstance(cut, ControlCut):
        raise TypeError("cut must be a ControlCut.")
    return canonical_content_hash(
        {
            "runtime_session_id": cut.runtime_session_id,
            "control_domain_id": cut.control_domain_id,
            "clock_domain_id": cut.clock_domain_id,
            "clock_reset_epoch": cut.clock_reset_epoch,
            "cutoff_at": cut.cutoff_at,
            "cutoff_admission_sequence": cut.cutoff_admission_sequence,
        },
        domain=COMPOSITION_TRIGGER_CONTROL_CUT_HASH_DOMAIN_DEVELOPMENT,
    )


@dataclass(frozen=True)
class CompositionTriggerEvaluationBasis:
    """Immutable information exposed to a replaceable trigger provider.

    ``semantic_payload`` is profile-declared selected input/state information,
    never a live device, model, simulator, or mutable store handle. Providers
    return immutable evidence before the reducer is called.
    """

    control_cut: ControlCut
    activation_scope: CompositionActivationScope
    payload_contract_hash: str
    semantic_payload: Mapping[str, Any]
    source_fact_ids: tuple[str, ...]
    source_fact_hashes: tuple[str, ...]
    valid_until: float
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.control_cut, ControlCut):
            raise TypeError("control_cut must be a ControlCut.")
        if not isinstance(self.activation_scope, CompositionActivationScope):
            raise TypeError("activation_scope must be CompositionActivationScope.")
        cut = self.control_cut
        activation = self.activation_scope
        if cut.clock_domain_id != activation.clock_domain_id:
            raise ValueError("Trigger basis and activation use another clock.")
        if cut.cutoff_at < activation.activated_at:
            raise ValueError("Trigger basis cut predates its activation.")
        contract_hash = required(
            self.payload_contract_hash,
            name="payload_contract_hash",
        )
        valid_until = finite(self.valid_until, name="valid_until")
        if valid_until <= cut.cutoff_at:
            raise ValueError("Trigger evaluation basis has no future support.")
        fact_ids, fact_hashes = lineage(
            tuple(self.source_fact_ids),
            tuple(self.source_fact_hashes),
        )
        payload = deep_freeze(dict(self.semantic_payload))
        semantic = {
            "control_cut_hash": composition_trigger_control_cut_hash(cut),
            "composition_activation_id": activation.activation_id,
            "composition_activation_hash": activation.semantic_content_hash,
            "payload_contract_hash": contract_hash,
            "semantic_payload": payload,
            "source_fact_ids": fact_ids,
            "source_fact_hashes": fact_hashes,
            "valid_until": valid_until,
        }
        object.__setattr__(self, "payload_contract_hash", contract_hash)
        object.__setattr__(self, "semantic_payload", payload)
        object.__setattr__(self, "source_fact_ids", fact_ids)
        object.__setattr__(self, "source_fact_hashes", fact_hashes)
        object.__setattr__(self, "valid_until", valid_until)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                semantic,
                domain=(
                    COMPOSITION_TRIGGER_EVALUATION_BASIS_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def basis_id(self) -> str:
        return f"composition-trigger-basis:{self.semantic_content_hash}"


@dataclass(frozen=True)
class CompositionTriggerEvidence:
    """One receiver-visible condition level or discrete event edge."""

    runtime_session_id: str
    control_domain_id: str
    clock_reset_epoch: int
    composition_activation_id: str
    composition_activation_hash: str
    trigger_source_id: str
    trigger_source_spec_hash: str
    origin_kind: str
    signal_name: str
    producer_epoch: int
    sequence: int
    active: bool
    occurred_at: float
    available_at: float
    valid_until: float
    clock_domain_id: str
    control_admission_sequence: int
    payload_contract_hash: str
    semantic_payload: Mapping[str, Any]
    source_fact_ids: tuple[str, ...]
    source_fact_hashes: tuple[str, ...]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        runtime_session_id = required(
            self.runtime_session_id,
            name="runtime_session_id",
        )
        control_domain_id = required(
            self.control_domain_id,
            name="control_domain_id",
        )
        reset_epoch = nonnegative_int(
            self.clock_reset_epoch,
            name="clock_reset_epoch",
        )
        activation_id = required(
            self.composition_activation_id,
            name="composition_activation_id",
        )
        activation_hash = required(
            self.composition_activation_hash,
            name="composition_activation_hash",
        )
        source_id = required(self.trigger_source_id, name="trigger_source_id")
        source_hash = required(
            self.trigger_source_spec_hash,
            name="trigger_source_spec_hash",
        )
        origin = required(self.origin_kind, name="origin_kind")
        if origin not in COMPOSITION_TRIGGER_ORIGINS_DEVELOPMENT:
            raise ValueError("Unsupported composition trigger origin.")
        signal_name = required(self.signal_name, name="signal_name")
        producer_epoch = nonnegative_int(
            self.producer_epoch,
            name="producer_epoch",
        )
        sequence = nonnegative_int(self.sequence, name="sequence")
        if not isinstance(self.active, bool):
            raise TypeError("active must be a bool.")
        if origin == COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT and not self.active:
            raise ValueError("A direct event is an edge and must be active.")
        occurred_at = finite(self.occurred_at, name="occurred_at")
        available_at = finite(self.available_at, name="available_at")
        valid_until = finite(self.valid_until, name="valid_until")
        if not occurred_at <= available_at < valid_until:
            raise ValueError(
                "Trigger timing must satisfy occurred_at <= available_at < "
                "valid_until."
            )
        clock_domain_id = required(
            self.clock_domain_id,
            name="clock_domain_id",
        )
        admission_sequence = nonnegative_int(
            self.control_admission_sequence,
            name="control_admission_sequence",
        )
        contract_hash = required(
            self.payload_contract_hash,
            name="payload_contract_hash",
        )
        payload = deep_freeze(dict(self.semantic_payload))
        fact_ids, fact_hashes = lineage(
            tuple(self.source_fact_ids),
            tuple(self.source_fact_hashes),
        )
        semantic = {
            "runtime_session_id": runtime_session_id,
            "control_domain_id": control_domain_id,
            "clock_reset_epoch": reset_epoch,
            "composition_activation_id": activation_id,
            "composition_activation_hash": activation_hash,
            "trigger_source_id": source_id,
            "trigger_source_spec_hash": source_hash,
            "origin_kind": origin,
            "signal_name": signal_name,
            "producer_epoch": producer_epoch,
            "sequence": sequence,
            "active": self.active,
            "occurred_at": occurred_at,
            "available_at": available_at,
            "valid_until": valid_until,
            "clock_domain_id": clock_domain_id,
            "control_admission_sequence": admission_sequence,
            "payload_contract_hash": contract_hash,
            "semantic_payload": payload,
            "source_fact_ids": fact_ids,
            "source_fact_hashes": fact_hashes,
        }
        for name, value in semantic.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                semantic,
                domain=COMPOSITION_TRIGGER_EVIDENCE_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @property
    def evidence_id(self) -> str:
        return f"composition-trigger-evidence:{self.semantic_content_hash}"


@runtime_checkable
class CompositionTriggerEvidenceProvider(Protocol):
    """Replaceable input/state classifier or discrete-event adapter."""

    @property
    def trigger_source_id(self) -> str: ...

    @property
    def trigger_source_spec_hash(self) -> str: ...

    @property
    def origin_kind(self) -> str: ...

    def evaluate(
        self,
        basis: CompositionTriggerEvaluationBasis,
    ) -> CompositionTriggerEvidence | None: ...


__all__ = [
    "COMPOSITION_TRIGGER_ORIGIN_CONDITION",
    "COMPOSITION_TRIGGER_ORIGIN_DIRECT_EVENT",
    "COMPOSITION_TRIGGER_ORIGINS_DEVELOPMENT",
    "CompositionTriggerEvaluationBasis",
    "CompositionTriggerEvidence",
    "CompositionTriggerEvidenceProvider",
    "composition_trigger_control_cut_hash",
]
