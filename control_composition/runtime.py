"""Transport-neutral composition runtime above canonical action authority.

This runtime owns context and deterministic composition facts.  It shares the
control-domain admission prefix with the authoritative core, but owns no
authority policy, execution state, dispatcher, or backend.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from control_composition.activation import CompositionActivationScope
from control_composition.events import (
    CompositionActivationBinding,
    CompositionDomainEvent,
    CompositionSourceEligibilityBinding,
)
from control_composition.information import (
    CompositionContextRequirement,
    CompositionInformationPlan,
)
from control_composition.eligibility import (
    CompositionSourceEligibilityAdmissionReceipt,
    CompositionSourceEligibilityDraft,
    CompositionSourceEligibilitySnapshot,
    CompositionSourceEligibilitySpec,
    CompositionSourceEligibilityStore,
)
from control_composition.evaluation import (
    CompositionDiagnostic,
    CompositionEvaluationRecord,
)
from control_composition.ports import (
    CompositionClock,
    CompositionEventEmitter,
    CompositionProposalAdmitter,
)
from robot_motion.action.components import ActionComponentCatalog
from control_composition.composition import (
    CompositionInputAdmissionReceipt,
    CompositionInputDraft,
    CompositionInputSnapshot,
    CompositionInputStore,
    CompositionResult,
    CompositionSpec,
    CompositionStrategyDecision,
    DeterministicComposer,
    FormalCompositionStrategy,
    FormalCompositionStrategyInput,
    FormalCompositionStrategyResultBinding,
    FormalCompositionStrategySpec,
    build_blocked_composition_result,
    build_composite_action_proposal_draft,
    build_formal_strategy_composition_result,
    build_formal_strategy_result_binding,
    CompositionContextAdmissionReceipt,
    CompositionContextDraft,
    CompositionContextSnapshot,
    CompositionContextStore,
)
from robot_motion.action.proposal import ProposalAdmissionReceipt
from control_composition.composition.component_contract import (
    COMPONENT_SEMANTIC_PARAMETER_KEYS_V1,
    FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
    ComponentCompositionRuleSet,
    FormalComponentCompositionBinding,
)
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
from robot_motion.runtime.identity import RuntimeFactIdentity

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
    source_eligibility_snapshot: CompositionSourceEligibilitySnapshot | None = None

    @property
    def diagnostics(self) -> tuple[CompositionDiagnostic, ...]:
        """Typed optional-stage diagnostics for this exact evaluation."""

        if self.evaluation_record is None:
            return ()
        return self.evaluation_record.diagnostics

    @property
    def domain_events_emitted(self) -> tuple[CompositionDomainEvent, ...]:
        """Exact protocol-neutral event batch emitted for this evaluation."""

        if self.evaluation_record is None:
            return ()
        return self.evaluation_record.domain_events_emitted


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
        event_emitter: CompositionEventEmitter,
        proposal_admitter: CompositionProposalAdmitter,
        context_retention_spec: TimeWindowRetentionSpec,
        composition_input_retention_spec: TimeWindowRetentionSpec,
        source_eligibility_retention_spec: TimeWindowRetentionSpec | None = None,
        information_plan: CompositionInformationPlan | None = None,
        composition_strategy: FormalCompositionStrategy | None = None,
        clock: CompositionClock = time.monotonic,
    ) -> None:
        if coordinator.control_domain_id != control_domain_id:
            raise ValueError("Topology runtime and coordinator domains must match.")
        if coordinator.clock_domain.runtime_session_id != runtime_session_id:
            raise ValueError("Topology runtime and coordinator sessions must match.")
        if producer_epochs.coordinator is not coordinator:
            raise ValueError("Topology runtime must share the producer epoch registry.")
        if not callable(event_emitter) or not callable(proposal_admitter):
            raise TypeError("Topology event/proposal sinks must be callable.")
        if information_plan is not None and not isinstance(
            information_plan,
            CompositionInformationPlan,
        ):
            raise TypeError(
                "information_plan must be a CompositionInformationPlan or None."
            )
        if composition_strategy is not None:
            strategy_spec = getattr(composition_strategy, "spec", None)
            if not isinstance(strategy_spec, FormalCompositionStrategySpec):
                raise TypeError(
                    "composition_strategy must expose a "
                    "FormalCompositionStrategySpec."
                )
            if not callable(getattr(composition_strategy, "decide", None)):
                raise TypeError("composition_strategy must implement decide().")
            declared_channels = (
                ()
                if information_plan is None
                else information_plan.required_context_channels
            )
            if strategy_spec.required_context_channels != declared_channels:
                raise ValueError(
                    "Formal strategy channels must match the runtime "
                    "CompositionInformationPlan."
                )
        self.runtime_session_id = runtime_session_id
        self.control_domain_id = control_domain_id
        self.embodiment_instance_id = embodiment_instance_id
        self.coordinator = coordinator
        self.producer_epochs = producer_epochs
        self.component_catalog = component_catalog
        self.event_emitter = event_emitter
        self.proposal_admitter = proposal_admitter
        self._information_plan = information_plan
        self.composition_strategy = composition_strategy
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
        self.source_eligibility = (
            None
            if source_eligibility_retention_spec is None
            else CompositionSourceEligibilityStore(
                runtime_session_id=runtime_session_id,
                coordinator=coordinator,
                producer_epochs=producer_epochs,
                retention_spec=source_eligibility_retention_spec,
            )
        )
        self.composer = DeterministicComposer()
        self._evaluation_snapshot_lock = threading.RLock()
        self._composition_promotion_lock = threading.RLock()
        self._event_spec_lock = threading.RLock()
        self._emitted_composition_spec_hashes: set[str] = set()
        self._emitted_source_eligibility_spec_hashes: set[str] = set()
        self._emitted_source_difference_spec_hashes: set[str] = set()
        self._emitted_formal_strategy_spec_hashes: set[str] = set()
        self._emitted_component_rule_set_hashes: set[str] = set()
        self._emitted_formal_component_binding_hashes: set[str] = set()
        self._information_plan_emitted = False
        self._proposal_admission_by_composition_result_id: dict[
            str,
            tuple[str, int, ProposalAdmissionReceipt],
        ] = {}

    def _emit(self, *events: CompositionDomainEvent) -> None:
        self.event_emitter(tuple(events))

    @property
    def information_plan(self) -> CompositionInformationPlan | None:
        """Return the immutable runtime-bound demand plan, if configured."""

        return self._information_plan

    def _information_plan_event(
        self,
        *,
        occurred_at: float,
    ) -> CompositionDomainEvent | None:
        if self.information_plan is None or self._information_plan_emitted:
            return None
        return CompositionDomainEvent(
            event_kind="composition_information_plan",
            subject=self.information_plan,
            occurred_at=occurred_at,
        )

    def _emit_with_information_plan(
        self,
        *events: CompositionDomainEvent,
    ) -> None:
        if not events:
            return
        with self._event_spec_lock:
            plan_event = self._information_plan_event(
                occurred_at=events[0].occurred_at
            )
            batch = (
                tuple(events)
                if plan_event is None
                else (plan_event, *events)
            )
            self._emit(*batch)
            if plan_event is not None:
                self._information_plan_emitted = True

    def context_requirement_for_channel(
        self,
        channel: str,
    ) -> CompositionContextRequirement | None:
        """Expose predeclared demand so adapters can avoid optional acquisition."""

        if self.information_plan is None:
            return None
        return self.information_plan.requirement_for_channel(channel)

    @staticmethod
    def _draft_context_id(draft: CompositionContextDraft) -> str | None:
        try:
            return RuntimeFactIdentity(
                runtime_session_id=draft.runtime_session_id,
                control_domain_id=draft.control_domain_id,
                producer_id=draft.producer_id,
                producer_epoch=draft.producer_epoch,
                sequence=draft.sequence,
                namespace="composition_context",
            ).fact_id
        except (TypeError, ValueError):
            return None

    def admit_composition_context(
        self,
        draft: CompositionContextDraft,
    ) -> CompositionContextAdmissionReceipt:
        requirement = self.context_requirement_for_channel(draft.context_channel)
        if self.information_plan is not None and requirement is None:
            receipt = CompositionContextAdmissionReceipt(
                accepted=False,
                rejection_code="information_channel_not_requested",
                message=(
                    "Context channel is not declared by CompositionInformationPlan."
                ),
                context_id=self._draft_context_id(draft),
            )
            self._emit_with_information_plan(
                CompositionDomainEvent(
                    event_kind="composition_context_admission",
                    subject=receipt,
                    occurred_at=float(self.clock()),
                )
            )
            return receipt
        if requirement is not None:
            if not requirement.accepts_payload_contract(
                draft.payload_contract_hash
            ):
                receipt = CompositionContextAdmissionReceipt(
                    accepted=False,
                    rejection_code="information_payload_contract_mismatch",
                    message=(
                        "Context payload contract is not accepted by the "
                        "CompositionInformationPlan."
                    ),
                    context_id=self._draft_context_id(draft),
                )
                self._emit_with_information_plan(
                    CompositionDomainEvent(
                        event_kind="composition_context_admission",
                        subject=receipt,
                        occurred_at=float(self.clock()),
                    )
                )
                return receipt
            if requirement.payload_access == "metadata_only" and (
                draft.semantic_payload or draft.blob_refs
            ):
                receipt = CompositionContextAdmissionReceipt(
                    accepted=False,
                    rejection_code="information_payload_access_violation",
                    message="metadata_only context must not carry payload or blob refs.",
                    context_id=self._draft_context_id(draft),
                )
                self._emit_with_information_plan(
                    CompositionDomainEvent(
                        event_kind="composition_context_admission",
                        subject=receipt,
                        occurred_at=float(self.clock()),
                    )
                )
                return receipt
            if requirement.payload_access == "blob_ref" and not draft.blob_refs:
                receipt = CompositionContextAdmissionReceipt(
                    accepted=False,
                    rejection_code="information_blob_ref_required",
                    message="blob_ref context requires at least one stable blob ref.",
                    context_id=self._draft_context_id(draft),
                )
                self._emit_with_information_plan(
                    CompositionDomainEvent(
                        event_kind="composition_context_admission",
                        subject=receipt,
                        occurred_at=float(self.clock()),
                    )
                )
                return receipt
            if requirement.payload_access == "inline" and draft.blob_refs:
                receipt = CompositionContextAdmissionReceipt(
                    accepted=False,
                    rejection_code="information_payload_access_violation",
                    message="inline context must not carry blob refs.",
                    context_id=self._draft_context_id(draft),
                )
                self._emit_with_information_plan(
                    CompositionDomainEvent(
                        event_kind="composition_context_admission",
                        subject=receipt,
                        occurred_at=float(self.clock()),
                    )
                )
                return receipt
        receipt = self.contexts.admit(draft)
        events = [
            CompositionDomainEvent(
                event_kind="composition_context_admission",
                subject=receipt,
                occurred_at=float(self.clock()),
            )
        ]
        if receipt.context is not None:
            events.append(
                CompositionDomainEvent(
                    event_kind="composition_context",
                    subject=receipt.context,
                    occurred_at=receipt.context.available_at,
                )
            )
        self._emit_with_information_plan(*events)
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
        self._emit_with_information_plan(
            CompositionDomainEvent(
                event_kind="composition_context_snapshot",
                subject=snapshot,
                occurred_at=snapshot.cutoff_at,
            )
        )
        return snapshot

    def admit_composition_input(
        self,
        draft: CompositionInputDraft,
        *,
        activation_scope: CompositionActivationScope | None = None,
    ) -> CompositionInputAdmissionReceipt:
        receipt = self.composition_inputs.admit(
            draft,
            activation_scope=activation_scope,
        )
        events = [
            CompositionDomainEvent(
                event_kind="composition_input_admission",
                subject=receipt,
                occurred_at=float(self.clock()),
            )
        ]
        if receipt.composition_input is not None:
            input_record = receipt.composition_input
            events.append(
                CompositionDomainEvent(
                    event_kind=(
                        "composition_contributor_input"
                        if input_record.source_fact_ids
                        else "composition_input"
                    ),
                    subject=input_record,
                    occurred_at=input_record.available_at,
                )
            )
            if input_record.composition_activation_id is not None:
                events.append(
                    CompositionDomainEvent(
                        event_kind="composition_activation_binding",
                        subject=CompositionActivationBinding(
                            subject_kind="input",
                            subject_id=input_record.composition_input_id,
                            subject_hash=input_record.semantic_content_hash,
                            activation_id=input_record.composition_activation_id,
                            activation_hash=str(
                                input_record.composition_activation_hash
                            ),
                        ),
                        occurred_at=input_record.available_at,
                    )
                )
        self._emit_with_information_plan(*events)
        return receipt

    def admit_source_eligibility(
        self,
        draft: CompositionSourceEligibilityDraft,
    ) -> CompositionSourceEligibilityAdmissionReceipt:
        """Admit one opt-in source-liveness fact without granting authority."""

        if self.source_eligibility is None:
            receipt = CompositionSourceEligibilityAdmissionReceipt(
                accepted=False,
                rejection_code="source_eligibility_disabled",
                message=(
                    "This composition runtime has no source eligibility store."
                ),
            )
        else:
            receipt = self.source_eligibility.admit(draft)
        events = [
            CompositionDomainEvent(
                event_kind="composition_source_eligibility_admission",
                subject=receipt,
                occurred_at=float(self.clock()),
            )
        ]
        if receipt.source_state is not None:
            events.append(
                CompositionDomainEvent(
                    event_kind="composition_source_eligibility_state",
                    subject=receipt.source_state,
                    occurred_at=receipt.source_state.available_at,
                )
            )
        self._emit_with_information_plan(*events)
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
        activation_scope: CompositionActivationScope | None = None,
        source_eligibility_spec: CompositionSourceEligibilitySpec | None = None,
    ) -> CompositeProposalSubmission:
        evaluation = self.evaluate_composition(
            spec=spec,
            target_effective_at=target_effective_at,
            required_context_channels=required_context_channels,
            cutoff_at=cutoff_at,
            source_difference_spec=source_difference_spec,
            activation_scope=activation_scope,
            source_eligibility_spec=source_eligibility_spec,
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
                    evaluation.source_eligibility_snapshot,
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
                evaluation.source_eligibility_snapshot,
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
        activation_scope: CompositionActivationScope | None = None,
        source_eligibility_spec: CompositionSourceEligibilitySpec | None = None,
    ) -> CompositeProposalSubmission:
        """Evaluate and observe one composition cut without publishing a proposal."""

        if source_difference_spec is not None and not isinstance(
            source_difference_spec,
            SourceDifferenceSpec,
        ):
            raise TypeError(
                "source_difference_spec must be a SourceDifferenceSpec or None."
            )
        if source_eligibility_spec is not None:
            if not isinstance(
                source_eligibility_spec,
                CompositionSourceEligibilitySpec,
            ):
                raise TypeError(
                    "source_eligibility_spec must be a "
                    "CompositionSourceEligibilitySpec or None."
                )
            source_eligibility_spec.validate_composition_spec(spec)
            if self.source_eligibility is None:
                raise ValueError(
                    "source_eligibility_spec requires an enabled source "
                    "eligibility store."
                )
        requested_context_channels = tuple(required_context_channels)
        if self.information_plan is not None:
            declared_channels = self.information_plan.required_context_channels
            if (
                requested_context_channels
                and requested_context_channels != declared_channels
            ):
                raise ValueError(
                    "required_context_channels must match CompositionInformationPlan."
                )
            requested_context_channels = declared_channels
            declared_difference_hash = (
                self.information_plan.source_difference_spec_hash
            )
            actual_difference_hash = (
                None
                if source_difference_spec is None
                else source_difference_spec.semantic_content_hash
            )
            if actual_difference_hash != declared_difference_hash:
                raise ValueError(
                    "source_difference_spec must match CompositionInformationPlan."
                )
        strategy = self.composition_strategy
        component_rule_set: ComponentCompositionRuleSet | None = None
        formal_component_binding: (
            FormalComponentCompositionBinding | None
        ) = None
        if strategy is not None:
            strategy_spec = strategy.spec
            if strategy_spec.composition_spec_hash != spec.semantic_content_hash:
                raise ValueError(
                    "Formal strategy is bound to another CompositionSpec."
                )
            if (
                requested_context_channels
                != strategy_spec.required_context_channels
            ):
                raise ValueError(
                    "Formal strategy context channels do not match evaluation."
                )
            if (
                spec.algebra == FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2
                and strategy_spec.implementation_contract_hashes
                != (spec.component_rule_set_hash,)
            ):
                raise ValueError(
                    "Formal component strategy must bind exactly the "
                    "CompositionSpec component rule set."
                )
            if spec.algebra == FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2:
                duplicated_semantic_keys = tuple(
                    sorted(
                        set(strategy_spec.parameter_payload)
                        & COMPONENT_SEMANTIC_PARAMETER_KEYS_V1
                    )
                )
                if duplicated_semantic_keys:
                    raise ValueError(
                        "Formal strategy parameter_payload must not redefine "
                        "CompositionSpec component semantics: "
                        + ",".join(duplicated_semantic_keys)
                    )
                component_rule_set = getattr(
                    strategy,
                    "component_rule_set",
                    None,
                )
                if not isinstance(
                    component_rule_set,
                    ComponentCompositionRuleSet,
                ):
                    raise ValueError(
                        "Formal component strategy must expose its typed "
                        "ComponentCompositionRuleSet."
                    )
                if (
                    component_rule_set.semantic_content_hash
                    != spec.component_rule_set_hash
                ):
                    raise ValueError(
                        "Formal component strategy exposes another component "
                        "rule set."
                    )
                formal_component_binding = (
                    FormalComponentCompositionBinding(
                        composition_spec_hash=spec.semantic_content_hash,
                        component_rule_set_hash=(
                            component_rule_set.semantic_content_hash
                        ),
                        formal_strategy_spec_hash=(
                            strategy_spec.semantic_content_hash
                        ),
                    )
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
            if self.source_eligibility is not None:
                self.source_eligibility.validate_target(target_effective_at)
            epoch_snapshot = self.producer_epochs.snapshot(cut)
            source_eligibility_snapshot: (
                CompositionSourceEligibilitySnapshot | None
            ) = None
            if source_eligibility_spec is not None:
                source_eligibility_snapshot = (
                    self.source_eligibility.capture_snapshot(
                        cut,
                        producer_epochs=epoch_snapshot,
                        spec=source_eligibility_spec,
                        composition_spec=spec,
                        target_effective_at=target_effective_at,
                    )
                )
            input_snapshot = self.composition_inputs.capture_snapshot(
                cut,
                producer_epochs=epoch_snapshot,
                spec=spec,
                target_effective_at=target_effective_at,
                activation_scope=activation_scope,
                source_eligibility_snapshot=source_eligibility_snapshot,
            )
            context_snapshot: CompositionContextSnapshot | None = None
            if requested_context_channels:
                context_snapshot = self.contexts.capture_snapshot(
                    cut,
                    producer_epochs=epoch_snapshot,
                    target_effective_at=target_effective_at,
                    required_channels=requested_context_channels,
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
            if blocked_reason is None and self.information_plan is not None:
                receiver_clock_domain_id = (
                    self.coordinator.clock_domain.clock_domain_id
                )
                age_clock_mismatch_channels = tuple(
                    requirement.channel
                    for requirement in self.information_plan.context_requirements
                    if (
                        requirement.maximum_age_sec is not None
                        and requirement.channel
                        in context_snapshot.selected_context_by_channel
                        and context_snapshot.selected_context_by_channel[
                            requirement.channel
                        ].produced_clock_domain_id
                        != receiver_clock_domain_id
                    )
                )
                if age_clock_mismatch_channels:
                    blocked_reason = (
                        "context_age_clock_mismatch:"
                        + ",".join(age_clock_mismatch_channels)
                    )
                else:
                    stale_channels = tuple(
                        requirement.channel
                        for requirement in self.information_plan.context_requirements
                        if (
                            requirement.maximum_age_sec is not None
                            and requirement.channel
                            in context_snapshot.selected_context_by_channel
                            and (
                                float(target_effective_at)
                                - context_snapshot.selected_context_by_channel[
                                    requirement.channel
                                ].produced_at
                            )
                            > requirement.maximum_age_sec
                        )
                    )
                    if stale_channels:
                        blocked_reason = "stale_context:" + ",".join(
                            stale_channels
                        )
        strategy_input: FormalCompositionStrategyInput | None = None
        strategy_result_binding: (
            FormalCompositionStrategyResultBinding | None
        ) = None
        strategy_diagnostic: CompositionDiagnostic | None = None
        if strategy is None:
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
        else:
            strategy_input = FormalCompositionStrategyInput(
                runtime_session_id=self.runtime_session_id,
                control_domain_id=self.control_domain_id,
                clock_domain_id=(
                    self.coordinator.clock_domain.clock_domain_id
                ),
                strategy_spec=strategy.spec,
                composition_spec=spec,
                input_snapshot=input_snapshot,
                context_snapshot=context_snapshot,
                information_plan=self.information_plan,
            )
            if blocked_reason is not None:
                strategy_decision = CompositionStrategyDecision(
                    output_status="no_action",
                    reason=blocked_reason,
                    composed_action=None,
                )
            else:
                started_at = time.perf_counter()
                try:
                    strategy_decision = strategy.decide(strategy_input)
                    if not isinstance(
                        strategy_decision,
                        CompositionStrategyDecision,
                    ):
                        raise TypeError(
                            "Formal strategy decide() must return "
                            "CompositionStrategyDecision."
                        )
                except Exception as error:
                    strategy_decision = CompositionStrategyDecision(
                        output_status="invalid",
                        reason=f"strategy_error:{type(error).__name__}",
                        composed_action=None,
                    )
                    strategy_diagnostic = CompositionDiagnostic(
                        stage="formal_strategy",
                        code=type(error).__name__,
                        message=str(error),
                        recoverable=True,
                    )
                else:
                    elapsed = time.perf_counter() - started_at
                    maximum_latency = (
                        strategy.spec.maximum_decision_latency_sec
                    )
                    if (
                        maximum_latency is not None
                        and elapsed > maximum_latency
                    ):
                        strategy_decision = CompositionStrategyDecision(
                            output_status="no_action",
                            reason="strategy_latency_exceeded",
                            composed_action=None,
                        )
                        strategy_diagnostic = CompositionDiagnostic(
                            stage="formal_strategy",
                            code="strategy_latency_exceeded",
                            message=(
                                "Formal inline strategy exceeded its declared "
                                "decision latency."
                            ),
                            recoverable=True,
                        )
            result = build_formal_strategy_composition_result(
                strategy_input,
                strategy_decision,
            )
            strategy_result_binding = build_formal_strategy_result_binding(
                strategy_input,
                result,
            )
        events = [
            CompositionDomainEvent(
                event_kind="composition_input_snapshot",
                subject=input_snapshot,
                occurred_at=input_snapshot.cutoff_at,
            )
        ]
        if input_snapshot.expected_activation_id is not None:
            events.append(
                CompositionDomainEvent(
                    event_kind="composition_activation_binding",
                    subject=CompositionActivationBinding(
                        subject_kind="snapshot",
                        subject_id=input_snapshot.composition_input_snapshot_id,
                        subject_hash=input_snapshot.semantic_content_hash,
                        activation_id=input_snapshot.expected_activation_id,
                        activation_hash=str(
                            input_snapshot.expected_activation_hash
                        ),
                    ),
                    occurred_at=input_snapshot.cutoff_at,
                )
            )
        if source_eligibility_snapshot is not None:
            events.append(
                CompositionDomainEvent(
                    event_kind="composition_source_eligibility_snapshot",
                    subject=source_eligibility_snapshot,
                    occurred_at=source_eligibility_snapshot.cutoff_at,
                )
            )
            events.append(
                CompositionDomainEvent(
                    event_kind="composition_source_eligibility_binding",
                    subject=CompositionSourceEligibilityBinding(
                        composition_input_snapshot_id=(
                            input_snapshot.composition_input_snapshot_id
                        ),
                        composition_input_snapshot_hash=(
                            input_snapshot.semantic_content_hash
                        ),
                        source_eligibility_snapshot_id=(
                            source_eligibility_snapshot
                            .source_eligibility_snapshot_id
                        ),
                        source_eligibility_snapshot_hash=(
                            source_eligibility_snapshot.semantic_content_hash
                        ),
                    ),
                    occurred_at=input_snapshot.cutoff_at,
                )
            )
        if context_snapshot is not None:
            events.append(
                CompositionDomainEvent(
                    event_kind="composition_context_snapshot",
                    subject=context_snapshot,
                    occurred_at=context_snapshot.cutoff_at,
                )
            )
        if strategy_input is not None:
            events.append(
                CompositionDomainEvent(
                    event_kind="formal_composition_strategy_input",
                    subject=strategy_input,
                    occurred_at=input_snapshot.cutoff_at,
                )
            )
        result_observed_at = float(self.clock())
        events.append(
            CompositionDomainEvent(
                event_kind="composition_result",
                subject=result,
                occurred_at=result_observed_at,
            )
        )
        if strategy_result_binding is not None:
            events.append(
                CompositionDomainEvent(
                    event_kind="formal_composition_strategy_result_binding",
                    subject=strategy_result_binding,
                    occurred_at=result_observed_at,
                )
            )
        if result.composition_activation_id is not None:
            events.append(
                CompositionDomainEvent(
                    event_kind="composition_activation_binding",
                    subject=CompositionActivationBinding(
                        subject_kind="result",
                        subject_id=result.composition_result_id,
                        subject_hash=result.semantic_content_hash,
                        activation_id=result.composition_activation_id,
                        activation_hash=str(
                            result.composition_activation_hash
                        ),
                    ),
                    occurred_at=float(self.clock()),
                )
            )
        difference_observation: SourceDifferenceObservation | None = None
        difference_error: str | None = None
        diagnostics: list[CompositionDiagnostic] = []
        if strategy_diagnostic is not None:
            diagnostics.append(strategy_diagnostic)
        difference_observation_event: CompositionDomainEvent | None = None
        emitted_events: tuple[CompositionDomainEvent, ...]
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
                difference_observation_event = CompositionDomainEvent(
                    event_kind="source_difference_observation",
                    subject=difference_observation,
                    occurred_at=difference_observation.evaluated_at,
                )
        with self._event_spec_lock:
            composition_spec_hash = spec.semantic_content_hash
            emit_composition_spec = (
                composition_spec_hash
                not in self._emitted_composition_spec_hashes
            )
            emit_source_eligibility_spec = (
                source_eligibility_spec is not None
                and source_eligibility_spec.semantic_content_hash
                not in self._emitted_source_eligibility_spec_hashes
            )
            emit_difference_spec = (
                source_difference_spec is not None
                and source_difference_spec.semantic_content_hash
                not in self._emitted_source_difference_spec_hashes
            )
            emit_formal_strategy_spec = (
                strategy is not None
                and strategy.spec.semantic_content_hash
                not in self._emitted_formal_strategy_spec_hashes
            )
            emit_component_rule_set = (
                component_rule_set is not None
                and component_rule_set.semantic_content_hash
                not in self._emitted_component_rule_set_hashes
            )
            emit_formal_component_binding = (
                formal_component_binding is not None
                and formal_component_binding.semantic_content_hash
                not in self._emitted_formal_component_binding_hashes
            )
            evaluation_events: list[CompositionDomainEvent] = []
            plan_event = self._information_plan_event(
                occurred_at=input_snapshot.cutoff_at
            )
            if plan_event is not None:
                evaluation_events.append(plan_event)
            if emit_composition_spec:
                evaluation_events.append(
                    CompositionDomainEvent(
                        event_kind="composition_spec",
                        subject=spec,
                        occurred_at=input_snapshot.cutoff_at,
                    )
                )
            if emit_component_rule_set:
                evaluation_events.append(
                    CompositionDomainEvent(
                        event_kind="component_composition_rule_set",
                        subject=component_rule_set,
                        occurred_at=input_snapshot.cutoff_at,
                    )
                )
            if emit_formal_strategy_spec:
                evaluation_events.append(
                    CompositionDomainEvent(
                        event_kind="formal_composition_strategy_spec",
                        subject=strategy.spec,
                        occurred_at=input_snapshot.cutoff_at,
                    )
                )
            if emit_formal_component_binding:
                evaluation_events.append(
                    CompositionDomainEvent(
                        event_kind="formal_component_composition_binding",
                        subject=formal_component_binding,
                        occurred_at=input_snapshot.cutoff_at,
                    )
                )
            if emit_source_eligibility_spec:
                evaluation_events.append(
                    CompositionDomainEvent(
                        event_kind="composition_source_eligibility_spec",
                        subject=source_eligibility_spec,
                        occurred_at=input_snapshot.cutoff_at,
                    )
                )
            evaluation_events.extend(events)
            if emit_difference_spec:
                evaluation_events.append(
                    CompositionDomainEvent(
                        event_kind="source_difference_spec",
                        subject=source_difference_spec,
                        occurred_at=float(difference_evaluated_at),
                    )
                )
            if difference_observation_event is not None:
                evaluation_events.append(difference_observation_event)
            self._emit(*evaluation_events)
            if plan_event is not None:
                self._information_plan_emitted = True
            if emit_composition_spec:
                self._emitted_composition_spec_hashes.add(
                    composition_spec_hash
                )
            if emit_source_eligibility_spec:
                self._emitted_source_eligibility_spec_hashes.add(
                    source_eligibility_spec.semantic_content_hash
                )
            if emit_difference_spec:
                self._emitted_source_difference_spec_hashes.add(
                    source_difference_spec.semantic_content_hash
                )
            if emit_formal_strategy_spec:
                self._emitted_formal_strategy_spec_hashes.add(
                    strategy.spec.semantic_content_hash
                )
            if emit_component_rule_set:
                self._emitted_component_rule_set_hashes.add(
                    component_rule_set.semantic_content_hash
                )
            if emit_formal_component_binding:
                self._emitted_formal_component_binding_hashes.add(
                    formal_component_binding.semantic_content_hash
                )
            emitted_events = tuple(evaluation_events)
        evaluation_record = CompositionEvaluationRecord(
            information_plan=self.information_plan,
            composition_spec=spec,
            control_cut=cut,
            producer_epoch_snapshot=epoch_snapshot,
            input_snapshot=input_snapshot,
            composition_result=result,
            required_context_channels=requested_context_channels,
            context_snapshot=context_snapshot,
            source_eligibility_spec=source_eligibility_spec,
            source_eligibility_snapshot=source_eligibility_snapshot,
            strategy_spec=None if strategy is None else strategy.spec,
            strategy_input=strategy_input,
            strategy_result_binding=strategy_result_binding,
            source_difference_spec=source_difference_spec,
            source_difference_observation=difference_observation,
            diagnostics=tuple(diagnostics),
            domain_events_emitted=emitted_events,
        )
        return CompositeProposalSubmission(
            input_snapshot=input_snapshot,
            composition_result=result,
            proposal_admission=None,
            context_snapshot=context_snapshot,
            source_difference_observation=difference_observation,
            source_difference_error=difference_error,
            evaluation_record=evaluation_record,
            source_eligibility_snapshot=source_eligibility_snapshot,
        )


__all__ = ["CompositeProposalSubmission", "ControlCompositionRuntime"]
