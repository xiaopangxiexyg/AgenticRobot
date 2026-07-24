"""Transport-neutral composition runtime above canonical action authority.

This runtime owns context and deterministic composition facts.  It shares the
control-domain admission prefix with the authoritative core, but owns no
authority policy, execution state, dispatcher, or backend.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext
from control_composition.trace import (
    project_composition_context,
    project_composition_context_admission,
    project_composition_context_snapshot,
    project_composition_input,
    project_composition_input_admission,
    project_composition_input_snapshot,
    project_composition_result,
    project_composition_spec,
    project_source_difference_observation,
    project_source_difference_spec,
)
from control_composition.evaluation import (
    CompositionDiagnostic,
    CompositionEvaluationRecord,
)
from control_composition.ports import (
    CompositionClock,
    CompositionProposalAdmitter,
    CompositionTraceEmitter,
)
from robot_motion.action.components import ActionComponentCatalog
from control_composition.composition import (
    CompositionInputAdmissionReceipt,
    CompositionInputDraft,
    CompositionInputSnapshot,
    CompositionInputStore,
    CompositionResult,
    CompositionSpec,
    DeterministicComposer,
    build_blocked_composition_result,
    build_composite_action_proposal_draft,
    CompositionContextAdmissionReceipt,
    CompositionContextDraft,
    CompositionContextSnapshot,
    CompositionContextStore,
)
from robot_motion.action.proposal import ProposalAdmissionReceipt
from control_composition.observability import (
    SourceDifferenceObservation,
    SourceDifferenceSpec,
    evaluate_composition_source_differences,
)
from robot_motion.runtime.control_admission import (
    ControlAdmissionCoordinator,
    ControlCut,
)
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry

from control_composition.retention import TimeWindowRetentionSpec


@dataclass(frozen=True)
class CompositeProposalSubmission:
    input_snapshot: CompositionInputSnapshot
    composition_result: CompositionResult
    proposal_admission: ProposalAdmissionReceipt | None
    context_snapshot: CompositionContextSnapshot | None = None
    source_difference_observation: SourceDifferenceObservation | None = None
    source_difference_error: str | None = None
    evaluation_record: CompositionEvaluationRecord | None = None

    @property
    def diagnostics(self) -> tuple[CompositionDiagnostic, ...]:
        """Typed optional-stage diagnostics for this exact evaluation."""

        if self.evaluation_record is None:
            return ()
        return self.evaluation_record.diagnostics

    @property
    def trace_facts_emitted(self) -> tuple[ControlTraceFact, ...]:
        """Exact trace batch emitted while computing this evaluation."""

        if self.evaluation_record is None:
            return ()
        return self.evaluation_record.trace_facts_emitted


class ControlCompositionRuntime:
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
        trace_emitter: CompositionTraceEmitter,
        proposal_admitter: CompositionProposalAdmitter,
        context_retention_spec: TimeWindowRetentionSpec,
        composition_input_retention_spec: TimeWindowRetentionSpec,
        clock: CompositionClock = time.monotonic,
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
        self.contexts = CompositionContextStore(
            runtime_session_id=runtime_session_id,
            coordinator=coordinator,
            producer_epochs=producer_epochs,
            retention_spec=context_retention_spec,
        )
        self.composition_inputs = CompositionInputStore(
            runtime_session_id=runtime_session_id,
            coordinator=coordinator,
            producer_epochs=producer_epochs,
            retention_spec=composition_input_retention_spec,
        )
        self.composer = DeterministicComposer()
        self._evaluation_snapshot_lock = threading.RLock()
        self._composition_promotion_lock = threading.RLock()
        self._trace_spec_lock = threading.RLock()
        self._emitted_composition_spec_hashes: set[str] = set()
        self._emitted_source_difference_spec_hashes: set[str] = set()
        self._proposal_admission_by_composition_result_id: dict[
            str,
            tuple[str, int, ProposalAdmissionReceipt],
        ] = {}

    def _emit(self, *facts: ControlTraceFact) -> None:
        self.trace_emitter(tuple(facts))

    def admit_composition_context(
        self,
        draft: CompositionContextDraft,
    ) -> CompositionContextAdmissionReceipt:
        receipt = self.contexts.admit(draft)
        facts = [
            project_composition_context_admission(
                self.trace_context,
                receipt,
                occurred_at=float(self.clock()),
            )
        ]
        if receipt.context is not None:
            facts.append(
                project_composition_context(self.trace_context, receipt.context)
            )
        self.trace_emitter(tuple(facts))
        return receipt

    def capture_composition_context_snapshot(
        self,
        *,
        target_effective_at: float,
        required_channels: tuple[str, ...],
        cutoff_at: float | None = None,
    ) -> CompositionContextSnapshot:
        cut = self.coordinator.capture_cut(
            cutoff_at=float(self.clock() if cutoff_at is None else cutoff_at)
        ).cut
        with self._evaluation_snapshot_lock:
            snapshot = self.contexts.capture_snapshot(
                cut,
                producer_epochs=self.producer_epochs.snapshot(cut),
                target_effective_at=target_effective_at,
                required_channels=required_channels,
            )
        self._emit(project_composition_context_snapshot(self.trace_context, snapshot))
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
        source_difference_spec: SourceDifferenceSpec | None = None,
    ) -> CompositeProposalSubmission:
        evaluation = self.evaluate_composition(
            spec=spec,
            target_effective_at=target_effective_at,
            required_context_channels=required_context_channels,
            cutoff_at=cutoff_at,
            source_difference_spec=source_difference_spec,
        )
        if (
            evaluation.composition_result.output_status != "ready"
            or evaluation.composition_result.composed_action is None
        ):
            return evaluation
        return self.admit_evaluated_composition(
            evaluation,
            runtime_session_id=self.runtime_session_id,
            producer_id=producer_id,
            producer_epoch=producer_epoch,
            sequence=sequence,
            effective_until=float(effective_until),
            component_coverage=component_coverage,
            selection_groups=selection_groups,
            policy_node_id=policy_node_id,
        )

    def admit_evaluated_composition(
        self,
        evaluation: CompositeProposalSubmission,
        *,
        runtime_session_id: str,
        producer_id: str,
        producer_epoch: int,
        sequence: int,
        effective_until: float,
        component_coverage: tuple[str, ...],
        selection_groups: tuple[str, ...],
        policy_node_id: str | None = None,
    ) -> CompositeProposalSubmission:
        """Promote one exact ready result at most once.

        Evaluation and proposal admission are separate so a formal authority
        gate can inspect the immutable result without recapturing a later cut.
        """

        if not isinstance(evaluation, CompositeProposalSubmission):
            raise TypeError("evaluation must be a CompositeProposalSubmission.")
        if runtime_session_id != self.runtime_session_id:
            raise ValueError("Composition promotion belongs to another session.")
        if evaluation.proposal_admission is not None:
            raise ValueError("Composition evaluation was already promoted.")
        result = evaluation.composition_result
        if result.output_status != "ready" or result.composed_action is None:
            raise ValueError("Only a ready composition evaluation may be promoted.")
        if (
            result.composition_input_snapshot_id
            != evaluation.input_snapshot.composition_input_snapshot_id
            or result.composition_input_snapshot_hash
            != evaluation.input_snapshot.semantic_content_hash
        ):
            raise ValueError("Composition result does not match its input snapshot.")
        with self._composition_promotion_lock:
            cached = self._proposal_admission_by_composition_result_id.get(
                result.composition_result_id
            )
            if cached is not None:
                cached_producer, cached_epoch, receipt = cached
                if (
                    cached_producer != producer_id
                    or cached_epoch != int(producer_epoch)
                ):
                    raise ValueError(
                        "Composition result was already promoted by another producer."
                    )
                return CompositeProposalSubmission(
                    evaluation.input_snapshot,
                    result,
                    receipt,
                    evaluation.context_snapshot,
                    evaluation.source_difference_observation,
                    evaluation.source_difference_error,
                    evaluation.evaluation_record,
                )
            draft = build_composite_action_proposal_draft(
                result,
                runtime_session_id=self.runtime_session_id,
                control_domain_id=self.control_domain_id,
                embodiment_instance_id=self.embodiment_instance_id,
                producer_id=producer_id,
                producer_epoch=producer_epoch,
                sequence=sequence,
                produced_at=float(self.clock()),
                produced_clock_domain_id=(
                    self.coordinator.clock_domain.clock_domain_id
                ),
                effective_from=float(
                    evaluation.input_snapshot.target_effective_at
                ),
                effective_until=float(effective_until),
                effective_clock_domain_id=(
                    self.coordinator.clock_domain.clock_domain_id
                ),
                component_schema_hash=self.component_catalog.semantic_content_hash,
                component_coverage=component_coverage,
                selection_groups=selection_groups,
                policy_node_id=policy_node_id,
            )
            receipt = self.proposal_admitter(
                draft,
                source_event_id=result.composition_result_id,
            )
            if receipt.accepted:
                self._proposal_admission_by_composition_result_id[
                    result.composition_result_id
                ] = (producer_id, int(producer_epoch), receipt)
            return CompositeProposalSubmission(
                evaluation.input_snapshot,
                result,
                receipt,
                evaluation.context_snapshot,
                evaluation.source_difference_observation,
                evaluation.source_difference_error,
                evaluation.evaluation_record,
            )

    def evaluate_composition(
        self,
        *,
        spec: CompositionSpec,
        target_effective_at: float,
        required_context_channels: tuple[str, ...] = (),
        cutoff_at: float | None = None,
        control_cut: ControlCut | None = None,
        source_difference_spec: SourceDifferenceSpec | None = None,
    ) -> CompositeProposalSubmission:
        """Evaluate and trace one composition cut without publishing a proposal."""

        if source_difference_spec is not None and not isinstance(
            source_difference_spec,
            SourceDifferenceSpec,
        ):
            raise TypeError(
                "source_difference_spec must be a SourceDifferenceSpec or None."
            )
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
        with self._evaluation_snapshot_lock:
            self.composition_inputs.validate_target(target_effective_at)
            self.contexts.validate_target(target_effective_at)
            epoch_snapshot = self.producer_epochs.snapshot(cut)
            input_snapshot = self.composition_inputs.capture_snapshot(
                cut,
                producer_epochs=epoch_snapshot,
                spec=spec,
                target_effective_at=target_effective_at,
            )
            context_snapshot: CompositionContextSnapshot | None = None
            if required_context_channels:
                context_snapshot = self.contexts.capture_snapshot(
                    cut,
                    producer_epochs=epoch_snapshot,
                    target_effective_at=target_effective_at,
                    required_channels=required_context_channels,
                )
            else:
                self.contexts.advance_closed_target(target_effective_at)
        blocked_reason: str | None = None
        if context_snapshot is not None:
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
                project_composition_context_snapshot(
                    self.trace_context, context_snapshot
                )
            )
        facts.append(
            project_composition_result(
                self.trace_context,
                result,
                occurred_at=float(self.clock()),
            )
        )
        difference_observation: SourceDifferenceObservation | None = None
        difference_error: str | None = None
        diagnostics: list[CompositionDiagnostic] = []
        difference_observation_fact: ControlTraceFact | None = None
        emitted_facts: tuple[ControlTraceFact, ...]
        difference_evaluated_at: float | None = None
        if source_difference_spec is not None:
            difference_evaluated_at = float(self.clock())
            try:
                difference_observation = (
                    evaluate_composition_source_differences(
                        input_snapshot,
                        result,
                        spec=spec,
                        difference_spec=source_difference_spec,
                        evaluated_at=difference_evaluated_at,
                        clock_domain_id=(
                            self.coordinator.clock_domain.clock_domain_id
                        ),
                    )
                )
            except Exception as error:
                difference_error = f"{type(error).__name__}:{error}"
                diagnostics.append(
                    CompositionDiagnostic(
                        stage="source_difference",
                        code=type(error).__name__,
                        message=str(error),
                        recoverable=True,
                    )
                )
            else:
                difference_observation_fact = (
                    project_source_difference_observation(
                        self.trace_context,
                        difference_observation,
                    )
                )
        with self._trace_spec_lock:
            composition_spec_hash = spec.semantic_content_hash
            emit_composition_spec = (
                composition_spec_hash
                not in self._emitted_composition_spec_hashes
            )
            emit_difference_spec = (
                source_difference_spec is not None
                and source_difference_spec.semantic_content_hash
                not in self._emitted_source_difference_spec_hashes
            )
            evaluation_facts: list[ControlTraceFact] = []
            if emit_composition_spec:
                evaluation_facts.append(
                    project_composition_spec(
                        self.trace_context,
                        spec,
                        occurred_at=input_snapshot.cutoff_at,
                    )
                )
            evaluation_facts.extend(facts)
            if emit_difference_spec:
                evaluation_facts.append(
                    project_source_difference_spec(
                        self.trace_context,
                        source_difference_spec,
                        occurred_at=float(difference_evaluated_at),
                    )
                )
            if difference_observation_fact is not None:
                evaluation_facts.append(difference_observation_fact)
            self._emit(*evaluation_facts)
            if emit_composition_spec:
                self._emitted_composition_spec_hashes.add(
                    composition_spec_hash
                )
            if emit_difference_spec:
                self._emitted_source_difference_spec_hashes.add(
                    source_difference_spec.semantic_content_hash
                )
            emitted_facts = tuple(evaluation_facts)
        evaluation_record = CompositionEvaluationRecord(
            composition_spec=spec,
            control_cut=cut,
            producer_epoch_snapshot=epoch_snapshot,
            input_snapshot=input_snapshot,
            composition_result=result,
            required_context_channels=tuple(required_context_channels),
            context_snapshot=context_snapshot,
            source_difference_spec=source_difference_spec,
            source_difference_observation=difference_observation,
            diagnostics=tuple(diagnostics),
            trace_facts_emitted=emitted_facts,
        )
        return CompositeProposalSubmission(
            input_snapshot=input_snapshot,
            composition_result=result,
            proposal_admission=None,
            context_snapshot=context_snapshot,
            source_difference_observation=difference_observation,
            source_difference_error=difference_error,
            evaluation_record=evaluation_record,
        )


__all__ = ["CompositeProposalSubmission", "ControlCompositionRuntime"]
