"""Pure construction of one same-source multi-role input batch.

The builder is source-, backend-, embodiment-, and representation-neutral. It
does not poll a source or admit the resulting batch; callers provide canonical
``ComponentAction`` values and exact source/time lineage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from robot_motion.action import ComponentAction

from ..activation import CompositionActivationScope
from .core import CompositionInputBatchDraft, CompositionInputDraft


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class DerivedCompositionInputRoleBinding:
    input_role: str
    producer_id: str
    producer_epoch: int
    action_schema_hash: str

    def __post_init__(self) -> None:
        for name in ("input_role", "producer_id", "action_schema_hash"):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        epoch = int(self.producer_epoch)
        if (
            isinstance(self.producer_epoch, bool)
            or epoch != self.producer_epoch
            or epoch < 0
        ):
            raise ValueError("producer_epoch must be a non-negative integer.")
        object.__setattr__(self, "producer_epoch", epoch)


def build_same_source_composition_input_batch(
    *,
    runtime_session_id: str,
    control_domain_id: str,
    role_bindings: Sequence[DerivedCompositionInputRoleBinding],
    action_by_role: Mapping[str, ComponentAction],
    source_sequence: int,
    produced_at: float,
    produced_clock_domain_id: str,
    effective_from: float,
    effective_until: float,
    effective_clock_domain_id: str,
    source_fact_ids: Sequence[str],
    source_fact_hashes: Sequence[str],
    activation_scope: CompositionActivationScope,
    semantic_binding_id_by_kind: Mapping[str, str] | None = None,
    semantic_binding_hash_by_kind: Mapping[str, str] | None = None,
    context_ids: Sequence[str] = (),
    observation_cut_id: str | None = None,
    audit_metadata: Mapping[str, Any] | None = None,
) -> CompositionInputBatchDraft:
    """Build fixed-role drafts that the receiver can admit all-or-none."""

    if not isinstance(activation_scope, CompositionActivationScope):
        raise TypeError("activation_scope must be CompositionActivationScope.")
    bindings = tuple(role_bindings)
    if len(bindings) < 2 or not all(
        isinstance(binding, DerivedCompositionInputRoleBinding)
        for binding in bindings
    ):
        raise TypeError("At least two typed derived-role bindings are required.")
    roles = tuple(binding.input_role for binding in bindings)
    if len(roles) != len(set(roles)):
        raise ValueError("Derived input roles must be unique.")
    producer_keys = tuple(
        (binding.producer_id, binding.producer_epoch) for binding in bindings
    )
    if len(producer_keys) != len(set(producer_keys)):
        raise ValueError(
            "Each same-sequence derived role needs a distinct producer identity."
        )
    actions = dict(action_by_role)
    if set(actions) != set(roles):
        raise ValueError("action_by_role must exactly cover derived role bindings.")
    source_ids = tuple(
        _required(value, name="source_fact_id") for value in source_fact_ids
    )
    source_hashes = tuple(
        _required(value, name="source_fact_hash")
        for value in source_fact_hashes
    )
    if not source_ids or len(source_ids) != len(source_hashes):
        raise ValueError(
            "Derived input batch requires non-empty paired source fact lineage."
        )
    metadata = {
        **dict(audit_metadata or {}),
        "same_source_derived_batch": True,
        "derived_input_roles": roles,
    }
    drafts = []
    for binding in bindings:
        action = actions[binding.input_role]
        if not isinstance(action, ComponentAction):
            raise TypeError("action_by_role values must be ComponentAction.")
        if action.action_schema.semantic_content_hash != binding.action_schema_hash:
            raise ValueError(
                f"Derived role {binding.input_role!r} action schema mismatch."
            )
        drafts.append(
            CompositionInputDraft(
                runtime_session_id=_required(
                    runtime_session_id,
                    name="runtime_session_id",
                ),
                control_domain_id=_required(
                    control_domain_id,
                    name="control_domain_id",
                ),
                producer_id=binding.producer_id,
                producer_epoch=binding.producer_epoch,
                sequence=int(source_sequence),
                input_role=binding.input_role,
                action=action,
                produced_at=float(produced_at),
                produced_clock_domain_id=_required(
                    produced_clock_domain_id,
                    name="produced_clock_domain_id",
                ),
                effective_from=float(effective_from),
                effective_until=float(effective_until),
                effective_clock_domain_id=_required(
                    effective_clock_domain_id,
                    name="effective_clock_domain_id",
                ),
                context_ids=tuple(context_ids),
                observation_cut_id=observation_cut_id,
                source_fact_ids=source_ids,
                source_fact_hashes=source_hashes,
                semantic_binding_id_by_kind=dict(
                    semantic_binding_id_by_kind or {}
                ),
                semantic_binding_hash_by_kind=dict(
                    semantic_binding_hash_by_kind or {}
                ),
                composition_activation_id=activation_scope.activation_id,
                composition_activation_hash=(
                    activation_scope.semantic_content_hash
                ),
                audit_metadata={
                    **metadata,
                    "derived_input_role": binding.input_role,
                },
            )
        )
    return CompositionInputBatchDraft(inputs=tuple(drafts))


__all__ = [
    "DerivedCompositionInputRoleBinding",
    "build_same_source_composition_input_batch",
]
