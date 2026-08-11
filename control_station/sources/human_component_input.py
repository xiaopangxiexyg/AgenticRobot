"""Pure adaptation of one passive human emission to a component action.

The adapter preserves the existing teleoperation source receipt and trace
facts.  It does not poll a device, admit a composition input or proposal,
select authority, apply safety, or execute a target.
"""

from __future__ import annotations

from dataclasses import dataclass

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


__all__ = [
    "HumanComponentActionAdaptation",
    "adapt_human_emission_to_component_action",
]
