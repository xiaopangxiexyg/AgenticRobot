"""Transport-neutral policy topology runtime above action authority.

This runtime owns context and deterministic composition facts.  It shares the
control-domain admission prefix with the authoritative core, but owns no
authority policy, execution state, dispatcher, or backend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.action.authoritative_trace import (
    AuthoritativeTraceContext,
    project_composition_input,
    project_composition_input_admission,
    project_composition_input_snapshot,
    project_composition_result,
    project_policy_context,
    project_policy_context_admission,
    project_policy_context_snapshot,
)
from robot_motion.action.components import ActionComponentCatalog
from robot_motion.action.composition import (
    CompositionInputAdmissionReceipt,
    CompositionInputDraft,
    CompositionInputSnapshot,
    CompositionInputStore,
    CompositionResult,
    CompositionSpec,
    DeterministicComposer,
    build_blocked_composition_result,
    build_composite_action_proposal_draft,
    build_composition_input_snapshot,
    project_composition_inputs,
)
from robot_motion.action.proposal import ProposalAdmissionReceipt
from robot_motion.runtime.control_admission import (
    ControlAdmissionCoordinator,
    ControlCut,
)
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry

from .context import (
    PolicyContextAdmissionReceipt,
    PolicyContextDraft,
    PolicyContextSnapshot,
    PolicyContextStore,
    build_policy_context_snapshot,
    project_policy_contexts,
)


TraceEmitter = Callable[[tuple[ControlTraceFact, ...]], None]
ProposalAdmitter = Callable[..., ProposalAdmissionReceipt]


@dataclass(frozen=True)
class CompositeProposalSubmission:
    input_snapshot: CompositionInputSnapshot
    composition_result: CompositionResult
    proposal_admission: ProposalAdmissionReceipt | None
    context_snapshot: PolicyContextSnapshot | None = None


class PolicyTopologyRuntime:
    """Context/composition owner that can only publish proposal drafts."""

    def __init__(
        self,
        *,
        runtime_session_id: str,
        control_domain_id: str,
        embodiment_instance_id: str,
        coordinator: ControlAdmissionCoordinator,
        producer_epochs: ProducerEpochRegistry,
        component_catalog: ActionComponentCatalog,
        trace_context: AuthoritativeTraceContext,
        trace_emitter: TraceEmitter,
        proposal_admitter: ProposalAdmitter,
        context_capacity: int = 4096,
        composition_input_capacity: int = 4096,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if coordinator.control_domain_id != control_domain_id:
            raise ValueError("Topology runtime and coordinator domains must match.")
        if coordinator.clock_domain.runtime_session_id != runtime_session_id:
            raise ValueError("Topology runtime and coordinator sessions must match.")
        if producer_epochs.coordinator is not coordinator:
            raise ValueError("Topology runtime must share the producer epoch registry.")
        if trace_context.embodiment_instance_id != embodiment_instance_id:
            raise ValueError("Topology trace context belongs to another embodiment.")
        if not callable(trace_emitter) or not callable(proposal_admitter):
            raise TypeError("Topology trace/proposal sinks must be callable.")
        self.runtime_session_id = runtime_session_id
        self.control_domain_id = control_domain_id
        self.embodiment_instance_id = embodiment_instance_id
        self.coordinator = coordinator
        self.producer_epochs = producer_epochs
        self.component_catalog = component_catalog
        self.trace_context = trace_context
        self.trace_emitter = trace_emitter
        self.proposal_admitter = proposal_admitter
        self.clock = clock
        self.contexts = PolicyContextStore(
            runtime_session_id=runtime_session_id,
            coordinator=coordinator,
            producer_epochs=producer_epochs,
            capacity=context_capacity,
        )
        self.composition_inputs = CompositionInputStore(
            runtime_session_id=runtime_session_id,
            coordinator=coordinator,
            producer_epochs=producer_epochs,
            capacity=composition_input_capacity,
        )
        self.composer = DeterministicComposer()

    def _emit(self, *facts: ControlTraceFact) -> None:
        self.trace_emitter(tuple(facts))

    def admit_policy_context(
        self,
        draft: PolicyContextDraft,
    ) -> PolicyContextAdmissionReceipt:
        receipt = self.contexts.admit(draft)
        facts = [
            project_policy_context_admission(
                self.trace_context,
                receipt,
                occurred_at=float(self.clock()),
            )
        ]
        if receipt.context is not None:
            facts.append(project_policy_context(self.trace_context, receipt.context))
        self.trace_emitter(tuple(facts))
        return receipt

    def capture_policy_context_snapshot(
        self,
        *,
        target_effective_at: float,
        required_channels: tuple[str, ...],
        cutoff_at: float | None = None,
    ) -> PolicyContextSnapshot:
        cut = self.coordinator.capture_cut(
            cutoff_at=float(self.clock() if cutoff_at is None else cutoff_at)
        ).cut
        snapshot = build_policy_context_snapshot(
            project_policy_contexts(self.contexts, cut),
            producer_epochs=self.producer_epochs.snapshot(cut),
            target_effective_at=target_effective_at,
            required_channels=required_channels,
        )
        self._emit(project_policy_context_snapshot(self.trace_context, snapshot))
        return snapshot

    def admit_composition_input(
        self,
        draft: CompositionInputDraft,
    ) -> CompositionInputAdmissionReceipt:
        receipt = self.composition_inputs.admit(draft)
        facts = [
            project_composition_input_admission(
                self.trace_context,
                receipt,
                occurred_at=float(self.clock()),
            )
        ]
        if receipt.composition_input is not None:
            facts.append(
                project_composition_input(
                    self.trace_context,
                    receipt.composition_input,
                )
            )
        self.trace_emitter(tuple(facts))
        return receipt

    def compose_and_admit_proposal(
        self,
        *,
        spec: CompositionSpec,
        target_effective_at: float,
        producer_id: str,
        producer_epoch: int,
        sequence: int,
        effective_until: float,
        component_coverage: tuple[str, ...],
        selection_groups: tuple[str, ...],
        policy_node_id: str | None = None,
        required_context_channels: tuple[str, ...] = (),
        cutoff_at: float | None = None,
    ) -> CompositeProposalSubmission:
        evaluation = self.evaluate_composition(
            spec=spec,
            target_effective_at=target_effective_at,
            required_context_channels=required_context_channels,
            cutoff_at=cutoff_at,
        )
        result = evaluation.composition_result
        if result.output_status != "ready" or result.composed_action is None:
            return evaluation
        draft = build_composite_action_proposal_draft(
            result,
            runtime_session_id=self.runtime_session_id,
            control_domain_id=self.control_domain_id,
            embodiment_instance_id=self.embodiment_instance_id,
            producer_id=producer_id,
            producer_epoch=producer_epoch,
            sequence=sequence,
            produced_at=float(self.clock()),
            produced_clock_domain_id=self.coordinator.clock_domain.clock_domain_id,
            effective_from=float(target_effective_at),
            effective_until=float(effective_until),
            effective_clock_domain_id=self.coordinator.clock_domain.clock_domain_id,
            component_schema_hash=self.component_catalog.semantic_content_hash,
            component_coverage=component_coverage,
            selection_groups=selection_groups,
            policy_node_id=policy_node_id,
        )
        receipt = self.proposal_admitter(
            draft,
            source_event_id=result.composition_result_id,
        )
        return CompositeProposalSubmission(
            evaluation.input_snapshot,
            result,
            receipt,
            evaluation.context_snapshot,
        )

    def evaluate_composition(
        self,
        *,
        spec: CompositionSpec,
        target_effective_at: float,
        required_context_channels: tuple[str, ...] = (),
        cutoff_at: float | None = None,
        control_cut: ControlCut | None = None,
    ) -> CompositeProposalSubmission:
        """Evaluate and trace one composition cut without publishing a proposal."""

        if control_cut is not None and cutoff_at is not None:
            raise ValueError("Provide control_cut or cutoff_at, not both.")
        if control_cut is None:
            timestamp = float(self.clock() if cutoff_at is None else cutoff_at)
            cut = self.coordinator.capture_cut(cutoff_at=timestamp).cut
        else:
            cut = control_cut
            descriptor = self.coordinator.clock_domain
            if (
                cut.runtime_session_id != self.runtime_session_id
                or cut.control_domain_id != self.control_domain_id
                or cut.clock_domain_id != descriptor.clock_domain_id
                or cut.clock_reset_epoch != descriptor.reset_epoch
                or cut.cutoff_admission_sequence
                > self.coordinator.committed_high_watermark
            ):
                raise ValueError(
                    "Composition control_cut is incompatible with this receiver."
                )
        epoch_snapshot = self.producer_epochs.snapshot(cut)
        input_snapshot = build_composition_input_snapshot(
            project_composition_inputs(self.composition_inputs, cut),
            producer_epochs=epoch_snapshot,
            spec=spec,
            target_effective_at=target_effective_at,
        )
        context_snapshot: PolicyContextSnapshot | None = None
        blocked_reason: str | None = None
        if required_context_channels:
            context_snapshot = build_policy_context_snapshot(
                project_policy_contexts(self.contexts, cut),
                producer_epochs=epoch_snapshot,
                target_effective_at=target_effective_at,
                required_channels=required_context_channels,
            )
            if context_snapshot.missing_channels:
                blocked_reason = "missing_context:" + ",".join(
                    context_snapshot.missing_channels
                )
            else:
                required_context_ids = set(
                    context_snapshot.selected_context_id_by_channel.values()
                )
                mismatched_roles = tuple(
                    role
                    for role, item in input_snapshot.selected_input_by_role.items()
                    if not required_context_ids.issubset(item.context_ids)
                )
                if mismatched_roles:
                    blocked_reason = "context_dependency_mismatch:" + ",".join(
                        sorted(mismatched_roles)
                    )
        result = (
            self.composer.compose(input_snapshot, spec=spec)
            if blocked_reason is None
            else build_blocked_composition_result(
                input_snapshot,
                spec=spec,
                status="no_action",
                reason=blocked_reason,
            )
        )
        facts = [project_composition_input_snapshot(self.trace_context, input_snapshot)]
        if context_snapshot is not None:
            facts.append(
                project_policy_context_snapshot(self.trace_context, context_snapshot)
            )
        facts.append(
            project_composition_result(
                self.trace_context,
                result,
                occurred_at=float(self.clock()),
            )
        )
        self._emit(*facts)
        return CompositeProposalSubmission(
            input_snapshot,
            result,
            None,
            context_snapshot,
        )


__all__ = ["CompositeProposalSubmission", "PolicyTopologyRuntime"]
