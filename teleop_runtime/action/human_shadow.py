"""Observe-only coordinator connecting TeleopSession hooks to shared arbitration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from teleop_runtime.control_trace.projection import (
    make_runtime_trace_terminal,
    project_human_proposal_adapter_result,
    project_legacy_execution,
    project_shadow_startup,
    project_shadow_step,
)
from teleop_runtime.control_trace.runtime import ControlTraceRuntime
from teleop_runtime.deploy.arbiter_shadow import (
    ArbiterShadowStep,
    LegacyExecutionObservationEvent,
    PassivePolicyArbiterShadow,
)

from .human_proposal import (
    HumanCommandEmission,
    HumanCommandFilterObservation,
    HumanLegacyActionAdapterResult,
    HumanLegacyCommandObservation,
    HumanProposalAdapter,
    HumanProposalAdapterResult,
)
from .human_safety_shadow import (
    HumanLegacySafetyBindingEvidence,
    bind_human_legacy_safety_evidence,
)
from teleop_runtime.real_teleop.temporary_dual_arm_safety import (
    TemporaryDualArmSafetyEvaluation,
)


@dataclass(frozen=True)
class HumanProposalShadowObservation:
    emission: HumanCommandEmission
    adaptation: HumanProposalAdapterResult
    shadow_step: ArbiterShadowStep


@dataclass(frozen=True)
class HumanLegacyShadowObservation:
    observation: HumanLegacyCommandObservation
    action_adaptation: HumanLegacyActionAdapterResult
    legacy_event: LegacyExecutionObservationEvent


@dataclass(frozen=True)
class HumanFilterSafetyShadowObservation:
    observation: HumanCommandFilterObservation
    binding_evidence: HumanLegacySafetyBindingEvidence


class HumanProposalShadowRuntime:
    """Default-off hook target; owns no Rig, dispatcher, or episode lifecycle."""

    def __init__(
        self,
        *,
        adapter: HumanProposalAdapter,
        shadow: PassivePolicyArbiterShadow,
        trace_runtime: ControlTraceRuntime | None = None,
        proposal_validity_sec: float | None = None,
        pending_capacity: int = 4096,
    ) -> None:
        self.adapter = adapter
        self.shadow = shadow
        self.trace_runtime = trace_runtime
        self.proposal_validity_sec = float(
            shadow.scheduler.spec.period_sec
            if proposal_validity_sec is None
            else proposal_validity_sec
        )
        if self.proposal_validity_sec <= 0.0:
            raise ValueError("proposal_validity_sec must be positive.")
        if int(pending_capacity) <= 0:
            raise ValueError("pending_capacity must be positive.")
        self.pending_capacity = int(pending_capacity)
        self._pending: dict[int, HumanProposalShadowObservation] = {}
        self._started = False
        self._closed = False
        self.trace_rejection_count = 0
        self.last_trace_rejection: Any | None = None
        self.last_source_observation: HumanProposalShadowObservation | None = None
        self.last_filter_safety_observation: HumanFilterSafetyShadowObservation | None = None
        self.last_legacy_observation: HumanLegacyShadowObservation | None = None
        self._validate_bindings()

    def _validate_bindings(self) -> None:
        adapter = self.adapter.config
        shadow = self.shadow.config
        for name in (
            "runtime_session_id",
            "control_domain_id",
            "embodiment_instance_id",
            "producer_id",
            "producer_epoch",
        ):
            if getattr(adapter, name) != getattr(shadow, name):
                raise ValueError(f"Human adapter and shadow disagree on {name}.")
        if (
            adapter.component_catalog.semantic_content_hash
            != self.shadow.component_catalog.semantic_content_hash
        ):
            raise ValueError("Human adapter and shadow component catalogs disagree.")
        if adapter.effective_clock_domain_id != shadow.clock_domain_id:
            raise ValueError("Human adapter effective clock must match shadow control clock.")
        if self.trace_runtime is not None:
            session = self.trace_runtime.session
            if session.runtime_session_id != shadow.runtime_session_id:
                raise ValueError("Trace and human shadow runtime sessions disagree.")
            if session.control_domain_id != shadow.control_domain_id:
                raise ValueError("Trace and human shadow control domains disagree.")

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("Human proposal shadow runtime is closed.")
        if self._started:
            return
        self._started = True
        self._issue_all(project_shadow_startup(self.shadow))

    def observe_emission(
        self,
        emission: HumanCommandEmission,
    ) -> HumanProposalShadowObservation:
        if not self._started:
            self.start()
        effective_from = float(emission.emitted_at)
        adaptation = self.adapter.adapt(
            emission,
            effective_from=effective_from,
            effective_until=effective_from + self.proposal_validity_sec,
            target_effective_at=effective_from,
        )
        if adaptation.proposal_draft is None:
            reason = (
                adaptation.source_validation.reasons[0]
                if adaptation.source_validation.reasons
                else "human_source_adapter_rejected"
            )
            step = self.shadow.status(
                "emission_rejected",
                reason=reason,
                emission_id=adaptation.source_adapter_receipt.source_emission_id,
                occurred_at=emission.emitted_at,
            )
        else:
            step = self.shadow.evaluate_proposal_draft(
                adaptation.proposal_draft,
                source_event_id=adaptation.source_adapter_receipt.source_emission_id,
            )
        observed = HumanProposalShadowObservation(
            emission=emission,
            adaptation=adaptation,
            shadow_step=step,
        )
        self.last_source_observation = observed
        self._pending[emission.emission_sequence] = observed
        while len(self._pending) > self.pending_capacity:
            oldest = next(iter(self._pending))
            del self._pending[oldest]
        self._issue_all(
            (
                *project_human_proposal_adapter_result(
                    emission,
                    adaptation,
                    runtime_session_id=self.shadow.config.runtime_session_id,
                    control_domain_id=self.shadow.config.control_domain_id,
                    embodiment_instance_id=self.shadow.config.embodiment_instance_id,
                    occurred_at=emission.emitted_at,
                    clock_domain_id=self.shadow.config.clock_domain_id,
                ),
                *project_shadow_step(self.shadow, step),
            )
        )
        return observed

    def observe_filtered_command(
        self,
        observation: HumanCommandFilterObservation,
        limiter_evaluation: TemporaryDualArmSafetyEvaluation,
    ) -> HumanFilterSafetyShadowObservation:
        source = self._pending.get(observation.emission_sequence)
        if source is None:
            raise ValueError("Filtered human command has no matching source emission.")
        evidence = bind_human_legacy_safety_evidence(
            emission=source.emission,
            filter_observation=observation,
            limiter_evaluation=limiter_evaluation,
        )
        observed = HumanFilterSafetyShadowObservation(
            observation=observation,
            binding_evidence=evidence,
        )
        self.last_filter_safety_observation = observed
        return observed

    def observe_legacy_execution(
        self,
        observation: HumanLegacyCommandObservation,
    ) -> HumanLegacyShadowObservation:
        source = self._pending.pop(observation.emission_sequence, None)
        if source is None:
            raise ValueError("Legacy human observation has no matching source emission.")
        action_adaptation = self.adapter.adapt_legacy_observation(observation)
        receipt = source.adaptation.source_adapter_receipt
        legacy_event = self.shadow.observe_legacy_action(
            source_event_id=receipt.source_emission_id,
            shadow_step=source.shadow_step,
            observed_action=action_adaptation.robot_action,
            command_present=bool(observation.final_commands),
            execution_accepted=observation.command_dispatched,
            execution_mode=observation.outcome,
            dispatch_attempted=observation.command_dispatched,
            dispatch_completed=observation.command_dispatched,
            observed_at=observation.observed_at,
        )
        observed = HumanLegacyShadowObservation(
            observation=observation,
            action_adaptation=action_adaptation,
            legacy_event=legacy_event,
        )
        self.last_legacy_observation = observed
        self._issue_all((project_legacy_execution(self.shadow, legacy_event),))
        return observed

    def close(self, *, reason: str = "human_shadow_runtime_closed") -> None:
        if self._closed:
            return
        if self._started:
            self._issue_all(
                (
                    make_runtime_trace_terminal(
                        self.shadow,
                        terminal_status="completed",
                        reason=reason,
                    ),
                )
            )
        self._pending.clear()
        self._closed = True

    def _issue_all(self, facts: tuple[Any, ...]) -> None:
        if self.trace_runtime is None:
            return
        for fact in facts:
            receipt = self.trace_runtime.issue(fact)
            if not receipt.accepted:
                self.trace_rejection_count += 1
                self.last_trace_rejection = receipt


__all__ = [
    "HumanFilterSafetyShadowObservation",
    "HumanLegacyShadowObservation",
    "HumanProposalShadowObservation",
    "HumanProposalShadowRuntime",
]
