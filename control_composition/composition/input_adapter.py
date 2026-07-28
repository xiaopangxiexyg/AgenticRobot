"""Pure adapters from canonical contributor emissions to composition inputs.

The adapter in this module only freezes the application-neutral facts needed by
``CompositionInputStore``.  It does not know whether the facts came from a
policy, a human, replay, a simulator, or hardware, and it never creates an
authority candidate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from robot_motion.action import RobotAction
from robot_motion.action.component_action import ComponentAction, ComponentActionSchema
from robot_motion.action.component_action_v1 import (
    project_robot_action_v1_to_component_action,
)
from robot_motion.runtime.immutable import deep_freeze

from ..temporal_sampling import (
    ComponentActionTemporalSample,
    GuaranteedNextComponentAction,
    sample_component_action_source,
)

from .core import CompositionInputDraft


COMPOSITION_INPUT_DRAFT_ADAPTER_CONTRACT_ID_V1 = (
    "control_composition_canonical_action_to_input_draft_v1"
)


@dataclass(frozen=True)
class TemporallySampledCompositionInput:
    """A source-time sample and the exact non-authoritative draft built from it."""

    sample: ComponentActionTemporalSample
    draft: CompositionInputDraft

    def __post_init__(self) -> None:
        if not isinstance(self.sample, ComponentActionTemporalSample):
            raise TypeError("sample must be a ComponentActionTemporalSample.")
        if not isinstance(self.draft, CompositionInputDraft):
            raise TypeError("draft must be a CompositionInputDraft.")
        if self.draft.action.semantic_content_hash != self.sample.action_hash:
            raise ValueError("Temporal sample and composition draft action mismatch.")
        if (
            self.draft.source_fact_ids != self.sample.source_fact_ids
            or self.draft.source_fact_hashes != self.sample.source_fact_hashes
        ):
            raise ValueError(
                "Temporal sample and composition draft source lineage mismatch."
            )


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _identifiers(
    values: tuple[str, ...] | list[str],
    *,
    name: str,
) -> tuple[str, ...]:
    normalized = tuple(_required(value, name=name) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be unique.")
    return normalized


def build_composition_input_draft(
    *,
    runtime_session_id: str,
    control_domain_id: str,
    producer_id: str,
    producer_epoch: int,
    sequence: int,
    input_role: str,
    action: ComponentAction,
    produced_at: float,
    produced_clock_domain_id: str,
    effective_from: float,
    effective_until: float,
    effective_clock_domain_id: str,
    context_ids: tuple[str, ...] = (),
    observation_cut_id: str | None = None,
    policy_output_id: str | None = None,
    source_fact_ids: tuple[str, ...] = (),
    source_fact_hashes: tuple[str, ...] = (),
    composition_activation_id: str | None = None,
    composition_activation_hash: str | None = None,
    audit_metadata: Mapping[str, Any] | None = None,
) -> CompositionInputDraft:
    """Build one non-authoritative input draft from a canonical action.

    Source-specific adapters should use this function after validating their
    own immutable emission.  Keeping this seam here prevents policy and human
    bindings from copying the input-draft contract or accidentally setting
    ``authority_candidate``.
    """

    if not isinstance(action, ComponentAction):
        raise TypeError("action must be a ComponentAction.")
    session = _required(runtime_session_id, name="runtime_session_id")
    domain = _required(control_domain_id, name="control_domain_id")
    producer = _required(producer_id, name="producer_id")
    role = _required(input_role, name="input_role")
    produced_clock = _required(
        produced_clock_domain_id,
        name="produced_clock_domain_id",
    )
    effective_clock = _required(
        effective_clock_domain_id,
        name="effective_clock_domain_id",
    )
    epoch = int(producer_epoch)
    sequence_value = int(sequence)
    if isinstance(producer_epoch, bool) or epoch != producer_epoch or epoch < 0:
        raise ValueError("producer_epoch must be a non-negative integer.")
    if isinstance(sequence, bool) or sequence_value != sequence or sequence_value < 0:
        raise ValueError("sequence must be a non-negative integer.")
    produced = _finite(produced_at, name="produced_at")
    effective_start = _finite(effective_from, name="effective_from")
    effective_end = _finite(effective_until, name="effective_until")
    if effective_end <= effective_start:
        raise ValueError("effective_until must be greater than effective_from.")

    normalized_context_ids = _identifiers(tuple(context_ids), name="context_id")
    normalized_source_ids = _identifiers(
        tuple(source_fact_ids),
        name="source_fact_id",
    )
    normalized_source_hashes = tuple(
        _required(value, name="source_fact_hash") for value in source_fact_hashes
    )
    if len(normalized_source_ids) != len(normalized_source_hashes):
        raise ValueError("source_fact_ids and source_fact_hashes must have equal length.")
    activation_id = (
        None
        if composition_activation_id is None
        else _required(composition_activation_id, name="composition_activation_id")
    )
    activation_hash = (
        None
        if composition_activation_hash is None
        else _required(
            composition_activation_hash,
            name="composition_activation_hash",
        )
    )
    if (activation_id is None) != (activation_hash is None):
        raise ValueError(
            "composition_activation_id and composition_activation_hash "
            "must be supplied together."
        )
    observation_id = (
        None
        if observation_cut_id is None
        else _required(observation_cut_id, name="observation_cut_id")
    )
    policy_id = (
        None
        if policy_output_id is None
        else _required(policy_output_id, name="policy_output_id")
    )
    metadata = deep_freeze(dict(audit_metadata or {}))
    return CompositionInputDraft(
        runtime_session_id=session,
        control_domain_id=domain,
        producer_id=producer,
        producer_epoch=epoch,
        sequence=sequence_value,
        input_role=role,
        action=action,
        produced_at=produced,
        produced_clock_domain_id=produced_clock,
        effective_from=effective_start,
        effective_until=effective_end,
        effective_clock_domain_id=effective_clock,
        context_ids=normalized_context_ids,
        observation_cut_id=observation_id,
        policy_output_id=policy_id,
        source_fact_ids=normalized_source_ids,
        source_fact_hashes=normalized_source_hashes,
        composition_activation_id=activation_id,
        composition_activation_hash=activation_hash,
        authority_candidate=False,
        audit_metadata=metadata,
    )


def build_robot_action_v1_composition_input_draft(
    *,
    action: RobotAction,
    action_schema: ComponentActionSchema,
    **kwargs: Any,
) -> CompositionInputDraft:
    """Explicit compatibility bridge for legacy RobotAction v1 sources."""

    if not isinstance(action, RobotAction):
        raise TypeError("action must be a RobotAction.")
    if not isinstance(action_schema, ComponentActionSchema):
        raise TypeError("action_schema must be a ComponentActionSchema.")
    return build_composition_input_draft(
        action=project_robot_action_v1_to_component_action(
            action,
            action_schema=action_schema,
        ),
        **kwargs,
    )


def build_temporally_sampled_composition_input(
    *,
    current_action: ComponentAction,
    current_target_effective_at: float,
    current_source_fact_id: str,
    current_source_fact_hash: str,
    target_effective_at: float,
    effective_until: float,
    guaranteed_next_step: GuaranteedNextComponentAction | None = None,
    runtime_session_id: str,
    control_domain_id: str,
    producer_id: str,
    producer_epoch: int,
    sequence: int,
    input_role: str,
    produced_at: float,
    clock_domain_id: str,
    context_ids: tuple[str, ...] = (),
    observation_cut_id: str | None = None,
    policy_output_id: str | None = None,
    composition_activation_id: str | None = None,
    composition_activation_hash: str | None = None,
    audit_metadata: Mapping[str, Any] | None = None,
) -> TemporallySampledCompositionInput:
    """Build one composition input at a target time from an optional next step.

    The typed ``guaranteed_next_step`` is the only interpolation opt-in.
    ``None`` preserves current-step behavior.  The output validity begins at the
    sampled target; callers choose its finite end according to their source
    cadence/TTL and may submit another sample for a later composition cut.
    """

    sample = sample_component_action_source(
        current_action=current_action,
        current_target_effective_at=current_target_effective_at,
        current_source_fact_id=current_source_fact_id,
        current_source_fact_hash=current_source_fact_hash,
        target_effective_at=target_effective_at,
        clock_domain_id=clock_domain_id,
        guaranteed_next_step=guaranteed_next_step,
    )
    metadata = {
        **dict(audit_metadata or {}),
        "source_temporal_sample_hash": sample.semantic_content_hash,
        "source_temporal_derivation_kind": sample.derivation_kind,
        "source_interpolation_alpha": sample.interpolation_alpha,
        "source_interpolation_contract_hash": (
            sample.interpolation_contract_hash
        ),
        "next_step_guarantee_id": sample.next_step_guarantee_id,
        "next_step_guarantee_hash": sample.next_step_guarantee_hash,
    }
    draft = build_composition_input_draft(
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        producer_id=producer_id,
        producer_epoch=producer_epoch,
        sequence=sequence,
        input_role=input_role,
        action=sample.action,
        produced_at=produced_at,
        produced_clock_domain_id=clock_domain_id,
        effective_from=target_effective_at,
        effective_until=effective_until,
        effective_clock_domain_id=clock_domain_id,
        context_ids=context_ids,
        observation_cut_id=observation_cut_id,
        policy_output_id=policy_output_id,
        source_fact_ids=sample.source_fact_ids,
        source_fact_hashes=sample.source_fact_hashes,
        composition_activation_id=composition_activation_id,
        composition_activation_hash=composition_activation_hash,
        audit_metadata=metadata,
    )
    return TemporallySampledCompositionInput(sample=sample, draft=draft)


__all__ = [
    "COMPOSITION_INPUT_DRAFT_ADAPTER_CONTRACT_ID_V1",
    "TemporallySampledCompositionInput",
    "build_composition_input_draft",
    "build_robot_action_v1_composition_input_draft",
    "build_temporally_sampled_composition_input",
]
