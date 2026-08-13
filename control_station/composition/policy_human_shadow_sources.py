"""Source-emission to admitted-input bridge for composition shadowing."""

from __future__ import annotations

from typing import Callable

from control_composition.activation import CompositionActivationScope
from control_composition.runtime import ControlCompositionRuntime
from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.deploy import (
    PolicyActionEmission,
    project_policy_action_emission,
)
from teleop_runtime.action.human_proposal import (
    HumanCommandEmission,
    HumanProposalAdapter,
)

from ..sources.human_component_input import (
    adapt_human_emission_to_composition_input,
)
from ..sources.policy import PolicyCompositionInputAdapter
from .policy_human_shadow_contracts import CompositionShadowInputSubmission


class PolicyHumanShadowInputBinder:
    """Adapt and admit sources without evaluating or publishing composition."""

    def __init__(
        self,
        *,
        runtime: ControlCompositionRuntime,
        policy_input_adapter: PolicyCompositionInputAdapter,
        human_input_adapter: HumanProposalAdapter,
        human_role: str,
        human_effective_horizon_sec: float,
        clock_domain_id: str,
        activation_scope: CompositionActivationScope | None,
        source_fact_sink: Callable[[ControlTraceFact], object] | None,
    ) -> None:
        self.runtime = runtime
        self.policy_input_adapter = policy_input_adapter
        self.human_input_adapter = human_input_adapter
        self.policy_role = policy_input_adapter.binding.role
        self.human_role = human_role
        self.human_effective_horizon_sec = human_effective_horizon_sec
        self.clock_domain_id = clock_domain_id
        self.activation_scope = activation_scope
        self.source_fact_sink = source_fact_sink

    def _record_source_facts(
        self,
        facts: tuple[ControlTraceFact, ...],
    ) -> tuple[str, ...]:
        sink = self.source_fact_sink
        if sink is None:
            return ()
        failures: list[str] = []
        for fact in facts:
            try:
                receipt = sink(fact)
                if receipt is False or getattr(receipt, "accepted", True) is False:
                    reason = (
                        getattr(receipt, "rejection_code", None)
                        or getattr(receipt, "message", None)
                        or "trace_sink_rejected"
                    )
                    failures.append(f"{fact.fact_id}:{reason}")
            except Exception as error:  # observation cannot gate composition
                failures.append(
                    f"{fact.fact_id}:{type(error).__name__}:{error}"
                )
        return tuple(failures)

    def submit_policy(
        self,
        emission: PolicyActionEmission,
        *,
        observed_at: float,
    ) -> CompositionShadowInputSubmission:
        try:
            facts = project_policy_action_emission(
                emission,
                runtime_session_id=self.runtime.runtime_session_id,
                control_domain_id=self.runtime.control_domain_id,
                embodiment_instance_id=self.runtime.embodiment_instance_id,
                clock_domain_id=self.clock_domain_id,
            )
            trace_failures = self._record_source_facts(facts)
            draft = self.policy_input_adapter.adapt(
                emission,
                activation_scope=self.activation_scope,
                source_fact_ids=tuple(fact.fact_id for fact in facts),
                source_fact_hashes=tuple(
                    fact.semantic_content_hash for fact in facts
                ),
            )
            receipt = self.runtime.admit_composition_input(
                draft,
                activation_scope=self.activation_scope,
            )
            error = None
            if not receipt.accepted:
                error = receipt.message or receipt.rejection_code or (
                    "composition_input_rejected"
                )
            return CompositionShadowInputSubmission(
                source_kind="policy",
                input_role=self.policy_role,
                source_emission_id=emission.emission_id,
                observed_at=observed_at,
                input_receipt=receipt,
                source_fact_ids=tuple(fact.fact_id for fact in facts),
                trace_failures=trace_failures,
                error=error,
            )
        except Exception as error:  # passive contributor failure remains visible
            return CompositionShadowInputSubmission(
                source_kind="policy",
                input_role=self.policy_role,
                source_emission_id=emission.emission_id,
                observed_at=observed_at,
                error=f"{type(error).__name__}:{error}",
            )

    def submit_human(
        self,
        emission: HumanCommandEmission,
        *,
        observed_at: float,
    ) -> CompositionShadowInputSubmission:
        source_id = f"human-emission:{emission.emission_sequence}"
        try:
            adaptation = adapt_human_emission_to_composition_input(
                adapter=self.human_input_adapter,
                emission=emission,
                input_role=self.human_role,
                effective_from=observed_at,
                effective_until=(
                    observed_at + self.human_effective_horizon_sec
                ),
                target_effective_at=observed_at,
                runtime_session_id=self.runtime.runtime_session_id,
                control_domain_id=self.runtime.control_domain_id,
                embodiment_instance_id=self.runtime.embodiment_instance_id,
                occurred_at=observed_at,
                clock_domain_id=self.clock_domain_id,
                activation_scope=self.activation_scope,
            )
            source_id = adaptation.source_emission_id
            source_adaptation = adaptation.source_adaptation
            facts = source_adaptation.trace_facts
            trace_failures = self._record_source_facts(facts)
            draft = adaptation.composition_input
            if draft is None:
                reasons = source_adaptation.source_validation.reasons
                return CompositionShadowInputSubmission(
                    source_kind="human",
                    input_role=self.human_role,
                    source_emission_id=source_id,
                    observed_at=observed_at,
                    source_fact_ids=tuple(fact.fact_id for fact in facts),
                    trace_failures=trace_failures,
                    error="source_validation_failed:"
                    + (",".join(reasons) if reasons else "no_action"),
                )
            receipt = self.runtime.admit_composition_input(
                draft,
                activation_scope=self.activation_scope,
            )
            error = None
            if not receipt.accepted:
                error = receipt.message or receipt.rejection_code or (
                    "composition_input_rejected"
                )
            return CompositionShadowInputSubmission(
                source_kind="human",
                input_role=self.human_role,
                source_emission_id=source_id,
                observed_at=observed_at,
                input_receipt=receipt,
                source_fact_ids=tuple(fact.fact_id for fact in facts),
                trace_failures=trace_failures,
                error=error,
            )
        except Exception as error:  # passive contributor failure remains visible
            return CompositionShadowInputSubmission(
                source_kind="human",
                input_role=self.human_role,
                source_emission_id=source_id,
                observed_at=observed_at,
                error=f"{type(error).__name__}:{error}",
            )


__all__ = ["PolicyHumanShadowInputBinder"]
