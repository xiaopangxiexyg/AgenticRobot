"""Non-authoritative composition over immutable, non-admitted projections.

Projected inputs are deliberately distinct from :class:`CompositionInputRecord`.
They carry no receiver admission sequence or receipt and can never be promoted
to an action proposal.  Their only purpose is deterministic, read-only
evaluation of an already-frozen causal basis (for example a future trajectory
preview).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.component_action import ComponentAction
from robot_motion.runtime.control_admission import ControlCut
from robot_motion.runtime.immutable import deep_freeze


PROJECTED_COMPOSITION_INPUT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.projected_composition_input.development"
)
FROZEN_COMPOSITION_CONTROL_CUT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.frozen_composition_control_cut.development"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _lineage(
    ids: tuple[str, ...],
    hashes: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    normalized_ids = tuple(
        _required(value, name="source_fact_id") for value in ids
    )
    normalized_hashes = tuple(
        _required(value, name="source_fact_hash") for value in hashes
    )
    if not normalized_ids or len(normalized_ids) != len(normalized_hashes):
        raise ValueError(
            "Projected source fact ids/hashes must be non-empty and aligned."
        )
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("Projected source fact ids must be unique.")
    return normalized_ids, normalized_hashes


def frozen_control_cut_semantic_hash(cut: ControlCut) -> str:
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
            "closing": False,
        },
        domain=FROZEN_COMPOSITION_CONTROL_CUT_HASH_DOMAIN_DEVELOPMENT,
    )


@dataclass(frozen=True)
class ProjectedCompositionInput:
    """One role input derived from immutable facts without receiver admission."""

    projection_basis_hash: str
    runtime_session_id: str
    control_domain_id: str
    input_role: str
    action: ComponentAction
    producer_id: str
    producer_epoch: int
    produced_at: float
    produced_clock_domain_id: str
    effective_from: float
    effective_until: float
    effective_clock_domain_id: str
    projection_sequence: int
    source_kind: str
    source_fact_ids: tuple[str, ...]
    source_fact_hashes: tuple[str, ...]
    context_ids: tuple[str, ...] = ()
    semantic_binding_id_by_kind: Mapping[str, str] = field(default_factory=dict)
    semantic_binding_hash_by_kind: Mapping[str, str] = field(default_factory=dict)
    composition_activation_id: str | None = None
    composition_activation_hash: str | None = None
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        basis_hash = _required(
            self.projection_basis_hash,
            name="projection_basis_hash",
        )
        session = _required(
            self.runtime_session_id,
            name="runtime_session_id",
        )
        domain = _required(self.control_domain_id, name="control_domain_id")
        role = _required(self.input_role, name="input_role")
        if not isinstance(self.action, ComponentAction):
            raise TypeError("action must be a ComponentAction.")
        producer = _required(self.producer_id, name="producer_id")
        producer_epoch = _nonnegative_int(
            self.producer_epoch,
            name="producer_epoch",
        )
        produced_at = _finite(self.produced_at, name="produced_at")
        produced_clock = _required(
            self.produced_clock_domain_id,
            name="produced_clock_domain_id",
        )
        effective_from = _finite(
            self.effective_from,
            name="effective_from",
        )
        effective_until = _finite(
            self.effective_until,
            name="effective_until",
        )
        if effective_until <= effective_from:
            raise ValueError("Projected input interval must be non-empty.")
        effective_clock = _required(
            self.effective_clock_domain_id,
            name="effective_clock_domain_id",
        )
        projection_sequence = _nonnegative_int(
            self.projection_sequence,
            name="projection_sequence",
        )
        source_kind = _required(self.source_kind, name="source_kind")
        fact_ids, fact_hashes = _lineage(
            tuple(self.source_fact_ids),
            tuple(self.source_fact_hashes),
        )
        context_ids = tuple(
            _required(value, name="context_id") for value in self.context_ids
        )
        if len(context_ids) != len(set(context_ids)):
            raise ValueError("context_ids must be unique.")
        binding_ids = {
            _required(kind, name="semantic_binding_kind"): _required(
                binding_id,
                name="semantic_binding_id",
            )
            for kind, binding_id in dict(
                self.semantic_binding_id_by_kind
            ).items()
        }
        binding_hashes = {
            _required(kind, name="semantic_binding_kind"): _required(
                binding_hash,
                name="semantic_binding_hash",
            )
            for kind, binding_hash in dict(
                self.semantic_binding_hash_by_kind
            ).items()
        }
        if set(binding_ids) != set(binding_hashes):
            raise ValueError(
                "Projected semantic binding id/hash kinds must match."
            )
        activation_id = self.composition_activation_id
        activation_hash = self.composition_activation_hash
        if (activation_id is None) != (activation_hash is None):
            raise ValueError("Composition activation identity must be complete.")
        if activation_id is not None:
            activation_id = _required(
                activation_id,
                name="composition_activation_id",
            )
            activation_hash = _required(
                activation_hash,
                name="composition_activation_hash",
            )
        metadata = deep_freeze(dict(self.audit_metadata))
        payload = {
            "projection_basis_hash": basis_hash,
            "runtime_session_id": session,
            "control_domain_id": domain,
            "input_role": role,
            "action_hash": self.action.semantic_content_hash,
            "action_schema_hash": self.action.action_schema.semantic_content_hash,
            "component_coverage": self.action.component_coverage,
            "producer_id": producer,
            "producer_epoch": producer_epoch,
            "produced_at": produced_at,
            "produced_clock_domain_id": produced_clock,
            "effective_from": effective_from,
            "effective_until": effective_until,
            "effective_clock_domain_id": effective_clock,
            "projection_sequence": projection_sequence,
            "source_kind": source_kind,
            "source_fact_ids": fact_ids,
            "source_fact_hashes": fact_hashes,
            "context_ids": context_ids,
            **(
                {
                    "semantic_binding_id_by_kind": dict(
                        sorted(binding_ids.items())
                    ),
                    "semantic_binding_hash_by_kind": dict(
                        sorted(binding_hashes.items())
                    ),
                }
                if binding_ids
                else {}
            ),
            "composition_activation_id": activation_id,
            "composition_activation_hash": activation_hash,
            "admitted": False,
            "authoritative": False,
            "promotable": False,
        }
        object.__setattr__(self, "projection_basis_hash", basis_hash)
        object.__setattr__(self, "runtime_session_id", session)
        object.__setattr__(self, "control_domain_id", domain)
        object.__setattr__(self, "input_role", role)
        object.__setattr__(self, "producer_id", producer)
        object.__setattr__(self, "producer_epoch", producer_epoch)
        object.__setattr__(self, "produced_at", produced_at)
        object.__setattr__(self, "produced_clock_domain_id", produced_clock)
        object.__setattr__(self, "effective_from", effective_from)
        object.__setattr__(self, "effective_until", effective_until)
        object.__setattr__(self, "effective_clock_domain_id", effective_clock)
        object.__setattr__(self, "projection_sequence", projection_sequence)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "source_fact_ids", fact_ids)
        object.__setattr__(self, "source_fact_hashes", fact_hashes)
        object.__setattr__(self, "context_ids", context_ids)
        object.__setattr__(
            self,
            "semantic_binding_id_by_kind",
            MappingProxyType(dict(sorted(binding_ids.items()))),
        )
        object.__setattr__(
            self,
            "semantic_binding_hash_by_kind",
            MappingProxyType(dict(sorted(binding_hashes.items()))),
        )
        object.__setattr__(self, "composition_activation_id", activation_id)
        object.__setattr__(self, "composition_activation_hash", activation_hash)
        object.__setattr__(self, "audit_metadata", metadata)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=PROJECTED_COMPOSITION_INPUT_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @property
    def projected_input_id(self) -> str:
        return f"projected-composition-input:{self.semantic_content_hash}"

    @property
    def admitted(self) -> bool:
        return False

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def promotable(self) -> bool:
        return False


__all__ = [
    "FROZEN_COMPOSITION_CONTROL_CUT_HASH_DOMAIN_DEVELOPMENT",
    "PROJECTED_COMPOSITION_INPUT_HASH_DOMAIN_DEVELOPMENT",
    "ProjectedCompositionInput",
    "frozen_control_cut_semantic_hash",
]
