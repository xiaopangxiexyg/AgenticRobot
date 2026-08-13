"""Pure adaptation of one passive Human emission to composition carriers.

The adapter preserves the existing teleoperation source receipt and trace
facts.  It does not poll a device, admit a composition input or proposal,
select authority, apply safety, or execute a target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from control_composition.activation import CompositionActivationScope
from control_composition.composition import (
    CompositionInputDraft,
    build_composition_input_draft,
)
from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.action import ComponentAction, SourceValidationResult
from teleop_runtime.action.human_proposal import (
    HumanCommandEmission,
    HumanProposalAdapter,
    HumanProposalAdapterResult,
    SourceAdapterReceipt,
)
from teleop_runtime.control_trace.projection import (
    project_human_proposal_adapter_result,
)


@dataclass(frozen=True)
class HumanComponentActionAdaptation:
    """One immutable, non-authoritative human source adaptation."""

    emission: HumanCommandEmission
    adapter_result: HumanProposalAdapterResult
    trace_facts: tuple[ControlTraceFact, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.emission, HumanCommandEmission):
            raise TypeError("emission must be HumanCommandEmission.")
        if not isinstance(self.adapter_result, HumanProposalAdapterResult):
            raise TypeError("adapter_result must be HumanProposalAdapterResult.")
        if any(not isinstance(fact, ControlTraceFact) for fact in self.trace_facts):
            raise TypeError("trace_facts must contain ControlTraceFact values.")
        receipt = self.adapter_result.source_adapter_receipt
        if receipt.source_emission_semantic_hash != (
            self.emission.semantic_content_hash
        ):
            raise ValueError("Human adaptation belongs to another emission.")
        # ``output_action_semantic_hash`` identifies the compatibility
        # RobotAction produced by HumanProposalAdapter.  The ComponentAction
        # is a later typed projection carried by proposal_draft, so the two
        # hashes are intentionally in different domains.
        self.action

    @property
    def source_adapter_receipt(self) -> SourceAdapterReceipt:
        return self.adapter_result.source_adapter_receipt

    @property
    def source_validation(self) -> SourceValidationResult:
        return self.adapter_result.source_validation

    @property
    def action(self) -> ComponentAction | None:
        draft = self.adapter_result.proposal_draft
        if draft is None or not self.adapter_result.source_validation.accepted:
            return None
        action = draft.action
        if not isinstance(action, ComponentAction):
            raise TypeError("Human adapter did not produce ComponentAction.")
        return action


@dataclass(frozen=True)
class HumanCompositionInputAdaptation:
    """Human source adaptation plus an optional composition contributor.

    A rejected or empty source still carries its typed source adaptation and
    trace facts, but deliberately has no ``composition_input``.  This object
    owns no receiver admission, authority, safety, target, or dispatch step.
    """

    source_adaptation: HumanComponentActionAdaptation
    input_role: str
    composition_input: CompositionInputDraft | None

    def __post_init__(self) -> None:
        if not isinstance(
            self.source_adaptation,
            HumanComponentActionAdaptation,
        ):
            raise TypeError(
                "source_adaptation must be HumanComponentActionAdaptation."
            )
        role = str(self.input_role).strip()
        if not role:
            raise ValueError("input_role must be non-empty.")
        if self.composition_input is not None:
            if not isinstance(self.composition_input, CompositionInputDraft):
                raise TypeError(
                    "composition_input must be CompositionInputDraft or None."
                )
            if self.composition_input.input_role != role:
                raise ValueError("Human composition input uses another role.")
        object.__setattr__(self, "input_role", role)

    @property
    def source_emission_id(self) -> str:
        return self.source_adaptation.source_adapter_receipt.source_emission_id


def adapt_human_emission_to_component_action(
    *,
    adapter: HumanProposalAdapter,
    emission: HumanCommandEmission,
    effective_from: float,
    effective_until: float,
    target_effective_at: float,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
    observation_cut_id: str | None = None,
) -> HumanComponentActionAdaptation:
    """Adapt once and project the source facts without admitting anything."""

    if not isinstance(adapter, HumanProposalAdapter):
        raise TypeError("adapter must be HumanProposalAdapter.")
    if not isinstance(emission, HumanCommandEmission):
        raise TypeError("emission must be HumanCommandEmission.")
    result = adapter.adapt(
        emission,
        effective_from=effective_from,
        effective_until=effective_until,
        target_effective_at=target_effective_at,
        observation_cut_id=observation_cut_id,
    )
    facts = project_human_proposal_adapter_result(
        emission,
        result,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=occurred_at,
        clock_domain_id=clock_domain_id,
    )
    return HumanComponentActionAdaptation(
        emission=emission,
        adapter_result=result,
        trace_facts=facts,
    )


def adapt_human_emission_to_composition_input(
    *,
    adapter: HumanProposalAdapter,
    emission: HumanCommandEmission,
    input_role: str,
    effective_from: float,
    effective_until: float,
    target_effective_at: float,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    occurred_at: float,
    clock_domain_id: str,
    activation_scope: CompositionActivationScope | None = None,
    observation_cut_id: str | None = None,
    audit_metadata: Mapping[str, object] | None = None,
) -> HumanCompositionInputAdaptation:
    """Build, but do not admit, one human composition contributor.

    Device decoding and representation projection remain in
    :class:`HumanProposalAdapter`.  This function only preserves that result
    in the backend-neutral ``CompositionInputDraft`` carrier.
    """

    role = str(input_role).strip()
    if not role:
        raise ValueError("input_role must be non-empty.")
    if activation_scope is not None and not isinstance(
        activation_scope,
        CompositionActivationScope,
    ):
        raise TypeError(
            "activation_scope must be CompositionActivationScope or None."
        )
    adaptation = adapt_human_emission_to_component_action(
        adapter=adapter,
        emission=emission,
        effective_from=effective_from,
        effective_until=effective_until,
        target_effective_at=target_effective_at,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        occurred_at=occurred_at,
        clock_domain_id=clock_domain_id,
        observation_cut_id=observation_cut_id,
    )
    result = adaptation.adapter_result
    proposal = result.proposal_draft
    action = adaptation.action
    if proposal is None or action is None:
        return HumanCompositionInputAdaptation(
            source_adaptation=adaptation,
            input_role=role,
            composition_input=None,
        )
    source_receipt = adaptation.source_adapter_receipt
    facts = adaptation.trace_facts
    draft = build_composition_input_draft(
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        producer_id=proposal.producer_id,
        producer_epoch=proposal.producer_epoch,
        sequence=proposal.sequence,
        input_role=role,
        action=action,
        produced_at=proposal.produced_at,
        produced_clock_domain_id=proposal.produced_clock_domain_id,
        effective_from=proposal.effective_from,
        effective_until=proposal.effective_until,
        effective_clock_domain_id=proposal.effective_clock_domain_id,
        context_ids=proposal.context_ids,
        observation_cut_id=proposal.observation_cut_id,
        source_fact_ids=tuple(fact.fact_id for fact in facts),
        source_fact_hashes=tuple(
            fact.semantic_content_hash for fact in facts
        ),
        composition_activation_id=(
            None if activation_scope is None else activation_scope.activation_id
        ),
        composition_activation_hash=(
            None
            if activation_scope is None
            else activation_scope.semantic_content_hash
        ),
        audit_metadata={
            **dict(proposal.audit_metadata),
            "composition_contributor": True,
            "contributor_kind": "human",
            "human_emission_id": source_receipt.source_emission_id,
            "human_emission_hash": emission.semantic_content_hash,
            "representation_trace_id": source_receipt.representation_trace_id,
            "source_adapter_receipt_hash": (
                source_receipt.semantic_content_hash
            ),
            **dict(audit_metadata or {}),
        },
    )
    return HumanCompositionInputAdaptation(
        source_adaptation=adaptation,
        input_role=role,
        composition_input=draft,
    )


__all__ = [
    "HumanComponentActionAdaptation",
    "HumanCompositionInputAdaptation",
    "adapt_human_emission_to_component_action",
    "adapt_human_emission_to_composition_input",
]
