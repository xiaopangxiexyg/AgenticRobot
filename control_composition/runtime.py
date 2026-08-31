"""Transport-neutral composition runtime above canonical action authority.

This runtime owns context and deterministic composition facts.  It shares the
control-domain admission prefix with the authoritative core, but owns no
authority policy, execution state, dispatcher, or backend.
"""

from __future__ import annotations

from collections import deque
import threading
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from control_composition.activation import CompositionActivationScope
from control_composition.events import (
    CompositionActivationBinding,
    CompositionDomainEvent,
    CompositionEventEmissionFailure,
    CompositionInputSemanticBinding,
    CompositionSourceEligibilityBinding,
)
from control_composition.information import (
    CompositionContextRequirement,
    CompositionInformationPlan,
)
from control_composition.temporal import CompositionTemporalSelectionSpec
from control_composition.eligibility import (
    CompositionSourceEligibilityAdmissionReceipt,
    CompositionSourceEligibilityDraft,
    CompositionSourceEligibilitySnapshot,
    CompositionSourceEligibilitySpec,
    CompositionSourceEligibilityStore,
    project_composition_source_eligibility,
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
from robot_motion.action.component_action import ComponentActionSchema
from control_composition.composition import (
    CompositionInputAdmissionReceipt,
    CompositionInputBatchAdmissionReceipt,
    CompositionInputBatchDraft,
    CompositionInputDraft,
    CompositionInputSnapshot,
    CompositionInputStore,
    CompositionEvaluation,
    CompositionSpec,
    CompositionStrategyDecision,
    DeterministicComposer,
    FormalCompositionStrategy,
    FormalCompositionStrategyInput,
    FormalCompositionPublicationTransaction,
    FormalCompositionStrategyResultBinding,
    FormalCompositionStrategySpec,
    build_blocked_composition_evaluation,
    build_composite_action_proposal_draft,
    build_formal_strategy_composition_evaluation,
    build_formal_strategy_result_binding,
    CompositionContextAdmissionReceipt,
    CompositionContextDraft,
    CompositionContextRecord,
    CompositionContextSnapshot,
    CompositionContextStore,
    CompositionPreMixPolicy,
    CompositionPreMixPolicyDecision,
    CompositionPreMixPolicyInput,
    CompositionPreMixPolicySpec,
    CompositionDynamicParameterAdmissionReceipt,
    CompositionDynamicParameterDraft,
    CompositionStrategyControl,
    CompositionStrategyControlBinding,
    CompositionStrategyRevisionAdmissionReceipt,
    CompositionStrategyRevisionDraft,
    project_composition_contexts,
    project_composition_inputs,
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
    ControlCutCapture,
)
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry
from robot_motion.runtime.identity import RuntimeFactIdentity

from control_composition.retention import TimeWindowRetentionSpec
from control_composition.residual_anchor_lifecycle import (
    ResidualAnchorLifecycleAdmissionReceipt,
    ResidualAnchorLifecycleDraft,
    ResidualAnchorLifecycleSnapshot,
    ResidualAnchorLifecycleSpec,
    ResidualAnchorLifecycleStore,
    capture_residual_anchor_lifecycle_snapshot,
    project_residual_anchor_lifecycle,
)


COMPOSITION_PROMOTION_REJECTION_SUPPORT_EXPIRED = (
    "composition_support_expired_before_proposal_publication"
)


@dataclass(frozen=True)
class CompositeProposalSubmission:
    input_snapshot: CompositionInputSnapshot
    composition_evaluation: CompositionEvaluation
    proposal_admission: ProposalAdmissionReceipt | None
    context_snapshot: CompositionContextSnapshot | None = None
    source_difference_observation: SourceDifferenceObservation | None = None
    source_difference_error: str | None = None
    evaluation_record: CompositionEvaluationRecord | None = None
    source_eligibility_snapshot: CompositionSourceEligibilitySnapshot | None = None
    residual_anchor_lifecycle_snapshot: (
        ResidualAnchorLifecycleSnapshot | None
    ) = None

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


def _composition_supported_until(
    *,
    input_snapshot: CompositionInputSnapshot,
    context_snapshot: CompositionContextSnapshot | None,
    source_eligibility_snapshot: (
        CompositionSourceEligibilitySnapshot | None
    ),
    information_plan: CompositionInformationPlan | None,
) -> float | None:
    """Return the earliest known end of support for this exact cut."""

    limits = [
        record.effective_until
        for record in input_snapshot.selected_input_by_role.values()
    ]
    if source_eligibility_snapshot is not None:
        limits.extend(
            source_eligibility_snapshot.selected_state_by_role[
                role
            ].lease_expires_at
            for role in input_snapshot.selected_input_by_role
            if (
                role
                in source_eligibility_snapshot.selected_state_by_role
            )
        )
    if context_snapshot is not None:
        limits.extend(
            record.effective_until
            for record in context_snapshot.selected_context_by_channel.values()
        )
        if information_plan is not None:
            for requirement in information_plan.context_requirements:
                if requirement.maximum_age_sec is None:
                    continue
                record = context_snapshot.selected_context_by_channel.get(
                    requirement.channel
                )
                if record is not None:
                    limits.append(
                        record.produced_at + requirement.maximum_age_sec
                    )
    return None if not limits else min(float(value) for value in limits)


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
        action_schema: ComponentActionSchema,
        input_action_schema_by_role: Mapping[
            str, ComponentActionSchema
        ] | None = None,
        event_emitter: CompositionEventEmitter,
        proposal_admitter: CompositionProposalAdmitter,
        context_retention_spec: TimeWindowRetentionSpec,
        composition_input_retention_spec: TimeWindowRetentionSpec,
        source_eligibility_retention_spec: TimeWindowRetentionSpec | None = None,
        temporal_selection_spec: CompositionTemporalSelectionSpec | None = None,
        information_plan: CompositionInformationPlan | None = None,
        composition_strategy: FormalCompositionStrategy | None = None,
        strategy_control: CompositionStrategyControl | None = None,
        pre_mix_policy: CompositionPreMixPolicy | None = None,
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
                else information_plan.context_channels
            )
            if strategy_spec.context_channels != declared_channels:
                raise ValueError(
                    "Formal strategy channels must match the runtime "
                    "CompositionInformationPlan."
                )
        if strategy_control is not None:
            if composition_strategy is None:
                raise ValueError(
                    "strategy_control requires a formal composition strategy."
                )
            if not isinstance(
                strategy_control,
                CompositionStrategyControl,
            ):
                raise TypeError(
                    "strategy_control must be a CompositionStrategyControl."
                )
            if (
                strategy_control.runtime_session_id != runtime_session_id
                or strategy_control.control_domain_id != control_domain_id
                or strategy_control.coordinator is not coordinator
                or strategy_control.producer_epochs is not producer_epochs
            ):
                raise ValueError(
                    "strategy_control must share this runtime receiver."
                )
            initial_revision = strategy_control.latest_revision
            if (
                initial_revision.strategy_spec != composition_strategy.spec
                or initial_revision.composition_spec.semantic_content_hash
                != composition_strategy.spec.composition_spec_hash
            ):
                raise ValueError(
                    "strategy_control initial revision must bind the installed "
                    "formal strategy."
                )
        if pre_mix_policy is not None:
            policy_spec = getattr(pre_mix_policy, "spec", None)
            if not isinstance(policy_spec, CompositionPreMixPolicySpec):
                raise TypeError(
                    "pre_mix_policy must expose a "
                    "CompositionPreMixPolicySpec."
                )
            if policy_spec.maximum_decision_latency_sec is None:
                raise ValueError(
                    "A synchronous pre_mix_policy must declare "
                    "maximum_decision_latency_sec."
                )
            if not callable(getattr(pre_mix_policy, "decide", None)):
                raise TypeError("pre_mix_policy must implement decide().")
        if temporal_selection_spec is not None and not isinstance(
            temporal_selection_spec,
            CompositionTemporalSelectionSpec,
        ):
            raise TypeError(
                "temporal_selection_spec must be a "
                "CompositionTemporalSelectionSpec or None."
            )
        self.runtime_session_id = runtime_session_id
        self.control_domain_id = control_domain_id
        self.embodiment_instance_id = embodiment_instance_id
        self.coordinator = coordinator
        self.producer_epochs = producer_epochs
        if not isinstance(action_schema, ComponentActionSchema):
            raise TypeError("action_schema must be a ComponentActionSchema.")
        self.action_schema = action_schema
        input_schemas = dict(input_action_schema_by_role or {})
        if any(
            not isinstance(schema, ComponentActionSchema)
            for schema in input_schemas.values()
        ):
            raise TypeError(
                "input_action_schema_by_role values must be "
                "ComponentActionSchema."
            )
        self.input_action_schema_by_role = MappingProxyType(input_schemas)
        self.event_emitter = event_emitter
        self.proposal_admitter = proposal_admitter
        self._information_plan = information_plan
        self.composition_strategy = composition_strategy
        self.strategy_control = strategy_control
        self.pre_mix_policy = pre_mix_policy
        self.temporal_selection_spec = (
            temporal_selection_spec
            if temporal_selection_spec is not None
            else CompositionTemporalSelectionSpec(
                spec_id=(
                    "runtime-bounded-same-target:"
                    f"{composition_input_retention_spec.spec_id}"
                ),
                maximum_target_lead_sec=(
                    composition_input_retention_spec.max_future_lead_sec
                    + composition_input_retention_spec.max_effective_duration_sec
                ),
                maximum_target_lag_sec=(
                    composition_input_retention_spec.history_window_sec
                ),
            )
        )
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
            action_schema=action_schema,
            input_action_schema_by_role=self.input_action_schema_by_role,
        )
        self.residual_anchor_lifecycle = ResidualAnchorLifecycleStore(
            runtime_session_id=runtime_session_id,
            coordinator=coordinator,
            producer_epochs=producer_epochs,
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
        self._strategy_publication_lock = threading.RLock()
        self._event_spec_lock = threading.RLock()
        self._event_failure_lock = threading.RLock()
        self._event_emission_failures: deque[
            CompositionEventEmissionFailure
        ] = deque(maxlen=128)
        self._event_emission_failure_count = 0
        self._emitted_composition_spec_hashes: set[str] = set()
        self._emitted_action_payload_schema_hashes: set[str] = set()
        self._emitted_component_action_schema_hashes: set[str] = set()
        self._emitted_source_eligibility_spec_hashes: set[str] = set()
        self._emitted_source_difference_spec_hashes: set[str] = set()
        self._emitted_formal_strategy_spec_hashes: set[str] = set()
        self._emitted_component_rule_set_hashes: set[str] = set()
        self._emitted_formal_component_binding_hashes: set[str] = set()
        self._emitted_strategy_revision_hashes: set[str] = set()
        self._emitted_dynamic_parameter_contract_hashes: set[str] = set()
        self._emitted_residual_anchor_lifecycle_spec_hashes: set[str] = set()
        self._information_plan_emitted = False
        self._temporal_selection_spec_emitted = False
        self._proposal_admission_by_composition_evaluation_id: dict[
            str,
            tuple[
                str,
                int,
                int,
                float,
                str | None,
                ProposalAdmissionReceipt,
            ],
        ] = {}
        self._accepted_publication_target_by_scope: dict[
            tuple[str, int, str], float
        ] = {}
        self._pending_publication_commit_by_evaluation_id: dict[
            str, CompositionEvaluation
        ] = {}

    @property
    def event_emission_failures(
        self,
    ) -> tuple[CompositionEventEmissionFailure, ...]:
        """Return the bounded passive-observer failure history."""

        with self._event_failure_lock:
            return tuple(self._event_emission_failures)

    @property
    def event_emission_failure_count(self) -> int:
        """Return the monotonic count, including failures evicted from history."""

        with self._event_failure_lock:
            return self._event_emission_failure_count

    def _emit(self, *events: CompositionDomainEvent) -> bool:
        """Best-effort event observation that never gates composition state."""

        batch = tuple(events)
        try:
            self.event_emitter(batch)
        except Exception as error:
            failure = CompositionEventEmissionFailure(
                event_kinds=tuple(event.event_kind for event in batch),
                error_type=type(error).__name__,
                message=str(error),
            )
            with self._event_failure_lock:
                self._event_emission_failure_count += 1
                self._event_emission_failures.append(failure)
            return False
        return True

    @property
    def information_plan(self) -> CompositionInformationPlan | None:
        """Return the immutable runtime-bound demand plan, if configured."""

        return self._information_plan

    def capture_non_closing_projection_cut(
        self,
        *,
        cutoff_at: float | None = None,
        include_composition_inputs: bool = False,
        include_contexts: bool = True,
        include_source_eligibility: bool = True,
        include_residual_anchor_lifecycle: bool = False,
    ) -> ControlCutCapture:
        """Freeze read-only store projections without closing a target window.

        This is the supported observer/preview seam.  It captures one control
        cut and immutable record tuples while holding the same snapshot lock as
        formal evaluation, but never calls a store ``capture_snapshot()`` or
        advances retention watermarks.
        """

        projectors = {"producer_epochs": self.producer_epochs.snapshot}
        if include_composition_inputs:
            projectors["composition_inputs"] = (
                lambda cut: project_composition_inputs(
                    self.composition_inputs,
                    cut,
                )
            )
        if include_contexts:
            projectors["contexts"] = lambda cut: project_composition_contexts(
                self.contexts,
                cut,
            )
        if include_source_eligibility and self.source_eligibility is not None:
            projectors["source_eligibility"] = (
                lambda cut: project_composition_source_eligibility(
                    self.source_eligibility,
                    cut,
                )
            )
        if include_residual_anchor_lifecycle:
            projectors["residual_anchor_lifecycle"] = (
                lambda cut: project_residual_anchor_lifecycle(
                    self.residual_anchor_lifecycle,
                    cut,
                )
            )
        with self._evaluation_snapshot_lock:
            return self.coordinator.capture_cut(
                cutoff_at=cutoff_at,
                projectors=projectors,
            )

    def admit_strategy_revision(
        self,
        draft: CompositionStrategyRevisionDraft,
    ) -> CompositionStrategyRevisionAdmissionReceipt:
        """Admit one complete low-frequency revision without mutating strategy."""

        if self.strategy_control is None:
            raise RuntimeError("This runtime has no strategy hot-control path.")
        receipt = self.strategy_control.admit_revision(draft)
        if receipt.revision is not None:
            revision = receipt.revision
            contract = revision.dynamic_parameter_contract
            with self._event_spec_lock:
                events: list[CompositionDomainEvent] = []
                if (
                    contract is not None
                    and contract.semantic_content_hash
                    not in self._emitted_dynamic_parameter_contract_hashes
                ):
                    events.append(
                        CompositionDomainEvent(
                            event_kind=(
                                "composition_dynamic_parameter_contract"
                            ),
                            subject=contract,
                            occurred_at=revision.available_at,
                        )
                    )
                if (
                    revision.semantic_content_hash
                    not in self._emitted_strategy_revision_hashes
                ):
                    events.append(
                        CompositionDomainEvent(
                            event_kind="composition_strategy_revision",
                            subject=revision,
                            occurred_at=revision.available_at,
                        )
                    )
                emitted = not events or self._emit(*events)
                if emitted and contract is not None:
                    self._emitted_dynamic_parameter_contract_hashes.add(
                        contract.semantic_content_hash
                    )
                if emitted:
                    self._emitted_strategy_revision_hashes.add(
                        revision.semantic_content_hash
                    )
        return receipt

    def admit_dynamic_parameters(
        self,
        draft: CompositionDynamicParameterDraft,
    ) -> CompositionDynamicParameterAdmissionReceipt:
        """Admit one high-frequency complete alpha vector."""

        if self.strategy_control is None:
            raise RuntimeError("This runtime has no strategy hot-control path.")
        receipt = self.strategy_control.admit_parameter(draft)
        if receipt.parameter_record is not None:
            record = receipt.parameter_record
            revision = self.strategy_control.revision_by_hash(
                record.strategy_revision_hash
            )
            assert revision is not None
            contract = revision.dynamic_parameter_contract
            with self._event_spec_lock:
                events: list[CompositionDomainEvent] = []
                if (
                    contract is not None
                    and contract.semantic_content_hash
                    not in self._emitted_dynamic_parameter_contract_hashes
                ):
                    events.append(
                        CompositionDomainEvent(
                            event_kind=(
                                "composition_dynamic_parameter_contract"
                            ),
                            subject=contract,
                            occurred_at=record.available_at,
                        )
                    )
                if (
                    revision.semantic_content_hash
                    not in self._emitted_strategy_revision_hashes
                ):
                    events.append(
                        CompositionDomainEvent(
                            event_kind="composition_strategy_revision",
                            subject=revision,
                            occurred_at=record.available_at,
                        )
                    )
                events.append(
                    CompositionDomainEvent(
                        event_kind="composition_dynamic_parameter",
                        subject=record,
                        occurred_at=record.available_at,
                    )
                )
                emitted = self._emit(*events)
                if emitted and contract is not None:
                    self._emitted_dynamic_parameter_contract_hashes.add(
                        contract.semantic_content_hash
                    )
                if emitted:
                    self._emitted_strategy_revision_hashes.add(
                        revision.semantic_content_hash
                    )
        return receipt

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
    ) -> bool:
        if not events:
            return True
        with self._event_spec_lock:
            plan_event = self._information_plan_event(
                occurred_at=events[0].occurred_at
            )
            batch = (
                tuple(events)
                if plan_event is None
                else (plan_event, *events)
            )
            emitted = self._emit(*batch)
            if emitted and plan_event is not None:
                self._information_plan_emitted = True
            return emitted

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

    def retained_composition_context_by_id(
        self,
        context_id: str,
    ) -> CompositionContextRecord | None:
        """Resolve a retained fact for exact producer retry reconciliation."""

        return self.contexts.record_by_id(context_id)

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
        self._emit_composition_input_receipts((receipt,))
        return receipt

    def _emit_composition_input_receipts(
        self,
        receipts: tuple[CompositionInputAdmissionReceipt, ...],
        *,
        batch_receipt: CompositionInputBatchAdmissionReceipt | None = None,
    ) -> None:
        events: list[CompositionDomainEvent] = []
        if batch_receipt is not None:
            events.append(
                CompositionDomainEvent(
                    event_kind="composition_input_batch_admission",
                    subject=batch_receipt,
                    occurred_at=float(self.clock()),
                )
            )
        for receipt in receipts:
            events.append(
                CompositionDomainEvent(
                    event_kind="composition_input_admission",
                    subject=receipt,
                    occurred_at=float(self.clock()),
                )
            )
            if receipt.composition_input is None:
                continue
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
            if input_record.semantic_binding_id_by_kind:
                events.append(
                    CompositionDomainEvent(
                        event_kind="composition_input_semantic_binding",
                        subject=CompositionInputSemanticBinding(
                            composition_input_id=(
                                input_record.composition_input_id
                            ),
                            composition_input_hash=(
                                input_record.semantic_content_hash
                            ),
                            binding_id_by_kind=(
                                input_record.semantic_binding_id_by_kind
                            ),
                            binding_hash_by_kind=(
                                input_record.semantic_binding_hash_by_kind
                            ),
                        ),
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

    def admit_composition_inputs(
        self,
        batch: CompositionInputBatchDraft,
        *,
        activation_scope: CompositionActivationScope | None = None,
    ) -> CompositionInputBatchAdmissionReceipt:
        """Admit one same-source multi-role batch without partial visibility."""

        receipt = self.composition_inputs.admit_batch(
            batch,
            activation_scope=activation_scope,
        )
        self._emit_composition_input_receipts(
            receipt.input_receipts,
            batch_receipt=receipt,
        )
        return receipt

    def admit_residual_anchor_lifecycle(
        self,
        draft: ResidualAnchorLifecycleDraft,
        *,
        spec: ResidualAnchorLifecycleSpec,
        activation_scope: CompositionActivationScope,
    ) -> ResidualAnchorLifecycleAdmissionReceipt:
        """Admit one low-rate anchor pending/ready fact on this receiver."""

        receipt = self.residual_anchor_lifecycle.admit(
            draft,
            spec=spec,
            activation_scope=activation_scope,
        )
        events: list[CompositionDomainEvent] = []
        with self._event_spec_lock:
            emit_spec = (
                spec.semantic_content_hash
                not in self._emitted_residual_anchor_lifecycle_spec_hashes
            )
            if emit_spec:
                events.append(
                    CompositionDomainEvent(
                        event_kind="residual_anchor_lifecycle_spec",
                        subject=spec,
                        occurred_at=float(self.clock()),
                    )
                )
            events.append(
                CompositionDomainEvent(
                    event_kind="residual_anchor_lifecycle_admission",
                    subject=receipt,
                    occurred_at=float(self.clock()),
                )
            )
            if receipt.lifecycle_record is not None:
                events.append(
                    CompositionDomainEvent(
                        event_kind="residual_anchor_lifecycle",
                        subject=receipt.lifecycle_record,
                        occurred_at=receipt.lifecycle_record.available_at,
                    )
                )
            emitted = self._emit_with_information_plan(*events)
            if emitted and emit_spec:
                self._emitted_residual_anchor_lifecycle_spec_hashes.add(
                    spec.semantic_content_hash
                )
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
        policy_node_id: str | None = None,
        required_context_channels: tuple[str, ...] = (),
        cutoff_at: float | None = None,
        source_difference_spec: SourceDifferenceSpec | None = None,
        activation_scope: CompositionActivationScope | None = None,
        source_eligibility_spec: CompositionSourceEligibilitySpec | None = None,
        residual_anchor_lifecycle_scope_id: str | None = None,
    ) -> CompositeProposalSubmission:
        evaluation = self.evaluate_composition(
            spec=spec,
            target_effective_at=target_effective_at,
            required_context_channels=required_context_channels,
            cutoff_at=cutoff_at,
            source_difference_spec=source_difference_spec,
            activation_scope=activation_scope,
            source_eligibility_spec=source_eligibility_spec,
            residual_anchor_lifecycle_scope_id=(
                residual_anchor_lifecycle_scope_id
            ),
        )
        if (
            evaluation.composition_evaluation.output_status != "ready"
            or evaluation.composition_evaluation.composed_candidate is None
        ):
            return evaluation
        return self.admit_evaluated_composition(
            evaluation,
            runtime_session_id=self.runtime_session_id,
            producer_id=producer_id,
            producer_epoch=producer_epoch,
            sequence=sequence,
            effective_until=float(effective_until),
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
        evaluation_record = evaluation.evaluation_record
        if evaluation_record is None:
            raise ValueError(
                "Composition promotion requires its causal evaluation record."
            )
        if (
            evaluation_record.control_cut.runtime_session_id
            != self.runtime_session_id
        ):
            raise ValueError(
                "Composition evaluation belongs to another runtime session."
            )
        snapshot = evaluation.input_snapshot
        result = evaluation.composition_evaluation
        if result.output_status != "ready" or result.composed_candidate is None:
            raise ValueError("Only a ready composition evaluation may be promoted.")
        if snapshot.control_domain_id != self.control_domain_id:
            raise ValueError(
                "Composition evaluation belongs to another control domain."
            )
        receiver_clock_domain_id = self.coordinator.clock_domain.clock_domain_id
        if (
            snapshot.target_clock_domain_id != receiver_clock_domain_id
            or result.target_clock_domain_id != receiver_clock_domain_id
        ):
            raise ValueError(
                "Composition evaluation belongs to another target clock domain."
            )
        receiver_action_schema_hash = self.action_schema.semantic_content_hash
        if (
            snapshot.action_schema_hash != receiver_action_schema_hash
            or result.action_schema_hash != receiver_action_schema_hash
            or result.composed_candidate.action_schema.semantic_content_hash
            != receiver_action_schema_hash
        ):
            raise ValueError(
                "Composition evaluation uses another receiver action schema."
            )
        missing_required = tuple(
            role
            for role in snapshot.required_roles
            if role not in snapshot.selected_input_by_role
        )
        if missing_required:
            raise ValueError(
                "A composition with missing required roles cannot be promoted: "
                f"{missing_required!r}."
            )
        if (
            result.composition_input_snapshot_id
            != snapshot.composition_input_snapshot_id
            or result.composition_input_snapshot_hash
            != snapshot.semantic_content_hash
        ):
            raise ValueError("Composition result does not match its input snapshot.")
        normalized_epoch = int(producer_epoch)
        normalized_sequence = int(sequence)
        normalized_until = float(effective_until)
        activation_id = result.composition_activation_id or "unscoped"
        publication_scope = (
            str(producer_id),
            normalized_epoch,
            activation_id,
        )
        with self._composition_promotion_lock, self._strategy_publication_lock:
            publication_transaction = (
                self.composition_strategy
                if isinstance(
                    self.composition_strategy,
                    FormalCompositionPublicationTransaction,
                )
                else None
            )
            cached = self._proposal_admission_by_composition_evaluation_id.get(
                result.composition_evaluation_id
            )
            if cached is not None:
                (
                    cached_producer,
                    cached_epoch,
                    cached_sequence,
                    cached_until,
                    cached_policy_node_id,
                    receipt,
                ) = cached
                if (
                    cached_producer != producer_id
                    or cached_epoch != normalized_epoch
                    or cached_sequence != normalized_sequence
                    or cached_until != normalized_until
                    or cached_policy_node_id != policy_node_id
                ):
                    raise ValueError(
                        "Composition result promotion retry changed its "
                        "publication envelope."
                    )
                pending_commit = (
                    self._pending_publication_commit_by_evaluation_id.get(
                        result.composition_evaluation_id
                    )
                )
                if pending_commit is not None:
                    if (
                        publication_transaction is None
                        or pending_commit.semantic_content_hash
                        != result.semantic_content_hash
                    ):
                        raise RuntimeError(
                            "Pending strategy publication commit lost its "
                            "exact transaction owner."
                        )
                    publication_transaction.commit_publication_transaction(
                        pending_commit
                    )
                    self._pending_publication_commit_by_evaluation_id.pop(
                        result.composition_evaluation_id,
                        None,
                    )
                return CompositeProposalSubmission(
                    input_snapshot=evaluation.input_snapshot,
                    composition_evaluation=result,
                    proposal_admission=receipt,
                    context_snapshot=evaluation.context_snapshot,
                    source_difference_observation=(
                        evaluation.source_difference_observation
                    ),
                    source_difference_error=evaluation.source_difference_error,
                    evaluation_record=evaluation.evaluation_record,
                    source_eligibility_snapshot=(
                        evaluation.source_eligibility_snapshot
                    ),
                    residual_anchor_lifecycle_snapshot=(
                        evaluation.residual_anchor_lifecycle_snapshot
                    ),
                )
            last_target = self._accepted_publication_target_by_scope.get(
                publication_scope
            )
            if (
                last_target is not None
                and result.target_effective_at <= last_target
            ):
                raise ValueError(
                    "Production composition targets must increase strictly "
                    "within one producer activation."
                )
            if publication_transaction is not None:
                publication_transaction.validate_publication_transaction(result)
            produced_at = float(self.clock())
            if result.composition_supported_until is None:
                raise ValueError(
                    "Ready composition evaluation has no support deadline."
                )
            if produced_at >= result.composition_supported_until:
                receipt = ProposalAdmissionReceipt(
                    accepted=False,
                    rejection_code=(
                        COMPOSITION_PROMOTION_REJECTION_SUPPORT_EXPIRED
                    ),
                    message=(
                        "Composition support expired before proposal publication."
                    ),
                )
                self._proposal_admission_by_composition_evaluation_id[
                    result.composition_evaluation_id
                ] = (
                    producer_id,
                    normalized_epoch,
                    normalized_sequence,
                    normalized_until,
                    policy_node_id,
                    receipt,
                )
                return CompositeProposalSubmission(
                    input_snapshot=evaluation.input_snapshot,
                    composition_evaluation=result,
                    proposal_admission=receipt,
                    context_snapshot=evaluation.context_snapshot,
                    source_difference_observation=(
                        evaluation.source_difference_observation
                    ),
                    source_difference_error=evaluation.source_difference_error,
                    evaluation_record=evaluation.evaluation_record,
                    source_eligibility_snapshot=(
                        evaluation.source_eligibility_snapshot
                    ),
                    residual_anchor_lifecycle_snapshot=(
                        evaluation.residual_anchor_lifecycle_snapshot
                    ),
                )
            draft = build_composite_action_proposal_draft(
                result,
                runtime_session_id=self.runtime_session_id,
                control_domain_id=self.control_domain_id,
                embodiment_instance_id=self.embodiment_instance_id,
                producer_id=producer_id,
                producer_epoch=normalized_epoch,
                sequence=normalized_sequence,
                produced_at=produced_at,
                produced_clock_domain_id=(
                    self.coordinator.clock_domain.clock_domain_id
                ),
                effective_from=float(
                    evaluation.input_snapshot.target_effective_at
                ),
                effective_until=normalized_until,
                effective_clock_domain_id=(
                    self.coordinator.clock_domain.clock_domain_id
                ),
                policy_node_id=policy_node_id,
            )
            receipt = self.proposal_admitter(
                draft,
                source_event_id=result.composition_evaluation_id,
            )
            if receipt.accepted:
                self._proposal_admission_by_composition_evaluation_id[
                    result.composition_evaluation_id
                ] = (
                    producer_id,
                    normalized_epoch,
                    normalized_sequence,
                    normalized_until,
                    policy_node_id,
                    receipt,
                )
                self._accepted_publication_target_by_scope[
                    publication_scope
                ] = result.target_effective_at
                if publication_transaction is not None:
                    self._pending_publication_commit_by_evaluation_id[
                        result.composition_evaluation_id
                    ] = result
                    publication_transaction.commit_publication_transaction(result)
                    self._pending_publication_commit_by_evaluation_id.pop(
                        result.composition_evaluation_id,
                        None,
                    )
            return CompositeProposalSubmission(
                input_snapshot=evaluation.input_snapshot,
                composition_evaluation=result,
                proposal_admission=receipt,
                context_snapshot=evaluation.context_snapshot,
                source_difference_observation=(
                    evaluation.source_difference_observation
                ),
                source_difference_error=evaluation.source_difference_error,
                evaluation_record=evaluation.evaluation_record,
                source_eligibility_snapshot=(
                    evaluation.source_eligibility_snapshot
                ),
                residual_anchor_lifecycle_snapshot=(
                    evaluation.residual_anchor_lifecycle_snapshot
                ),
            )

    def evaluate_hot_composition(
        self,
        *,
        target_effective_at: float,
        required_context_channels: tuple[str, ...] = (),
        cutoff_at: float | None = None,
        control_cut: ControlCut | None = None,
        source_difference_spec: SourceDifferenceSpec | None = None,
        activation_scope: CompositionActivationScope | None = None,
        source_eligibility_spec: CompositionSourceEligibilitySpec | None = None,
        residual_anchor_lifecycle_scope_id: str | None = None,
    ) -> CompositeProposalSubmission:
        """Evaluate one cut using its exact admitted revision and parameters."""

        with self._strategy_publication_lock:
            if self.strategy_control is None:
                raise RuntimeError("This runtime has no strategy hot-control path.")
            if control_cut is not None and cutoff_at is not None:
                raise ValueError("Provide control_cut or cutoff_at, not both.")
            if control_cut is None:
                timestamp = float(self.clock() if cutoff_at is None else cutoff_at)
                cut = self.coordinator.capture_cut(cutoff_at=timestamp).cut
            else:
                cut = control_cut
            producer_epochs = self.producer_epochs.snapshot(cut)
            binding = self.strategy_control.capture_binding(
                cut,
                producer_epochs=producer_epochs,
                target_effective_at=target_effective_at,
            )
            return self.evaluate_composition(
                spec=binding.revision.composition_spec,
                target_effective_at=target_effective_at,
                required_context_channels=required_context_channels,
                control_cut=cut,
                source_difference_spec=source_difference_spec,
                activation_scope=activation_scope,
                source_eligibility_spec=source_eligibility_spec,
                residual_anchor_lifecycle_scope_id=(
                    residual_anchor_lifecycle_scope_id
                ),
                _strategy_control_binding=binding,
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
        residual_anchor_lifecycle_scope_id: str | None = None,
        _strategy_control_binding: CompositionStrategyControlBinding | None = None,
    ) -> CompositeProposalSubmission:
        """Evaluate one cut while fencing stateful publication transactions."""

        with self._strategy_publication_lock:
            if self._pending_publication_commit_by_evaluation_id:
                raise RuntimeError(
                    "A proposal was admitted while its strategy publication "
                    "commit is pending; retry that exact promotion first."
                )
            return self._evaluate_composition_locked(
                spec=spec,
                target_effective_at=target_effective_at,
                required_context_channels=required_context_channels,
                cutoff_at=cutoff_at,
                control_cut=control_cut,
                source_difference_spec=source_difference_spec,
                activation_scope=activation_scope,
                source_eligibility_spec=source_eligibility_spec,
                residual_anchor_lifecycle_scope_id=(
                    residual_anchor_lifecycle_scope_id
                ),
                _strategy_control_binding=_strategy_control_binding,
            )

    def _evaluate_composition_locked(
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
        residual_anchor_lifecycle_scope_id: str | None = None,
        _strategy_control_binding: CompositionStrategyControlBinding | None = None,
    ) -> CompositeProposalSubmission:
        """Implementation for one strategy/publication-locked evaluation."""

        if spec.action_schema_hash is not None and (
            spec.action_schema_hash
            != self.action_schema.semantic_content_hash
        ):
            raise ValueError(
                "CompositionSpec must bind the runtime ComponentActionSchema."
            )
        if spec.input_action_schema_hash_by_role:
            installed_hashes = {
                role: schema.semantic_content_hash
                for role, schema in self.input_action_schema_by_role.items()
            }
            if (
                installed_hashes
                != dict(spec.input_action_schema_hash_by_role)
            ):
                raise ValueError(
                    "CompositionSpec role-specific input schemas do not match "
                    "the runtime."
                )
        if _strategy_control_binding is not None:
            if self.strategy_control is None:
                raise ValueError(
                    "A strategy-control binding requires an enabled controller."
                )
            if (
                _strategy_control_binding.revision.composition_spec != spec
            ):
                raise ValueError(
                    "Strategy-control binding uses another CompositionSpec."
                )
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
            declared_channels = self.information_plan.context_channels
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
        debug_diagnostics_enabled = (
            self.information_plan is None
            or self.information_plan.diagnostic_level == "debug"
        )
        strategy = self.composition_strategy
        active_strategy_spec: FormalCompositionStrategySpec | None = None
        component_rule_set: ComponentCompositionRuleSet | None = None
        formal_component_binding: (
            FormalComponentCompositionBinding | None
        ) = None
        residual_anchor_lifecycle_spec: ResidualAnchorLifecycleSpec | None = None
        if strategy is not None:
            strategy_spec = (
                strategy.spec
                if _strategy_control_binding is None
                else _strategy_control_binding.revision.strategy_spec
            )
            active_strategy_spec = strategy_spec
            declared_lifecycle_hash = (
                strategy_spec.required_residual_anchor_lifecycle_spec_hash
            )
            candidate_lifecycle_spec = getattr(
                strategy,
                "residual_anchor_lifecycle_spec",
                None,
            )
            if declared_lifecycle_hash is not None:
                if not isinstance(
                    candidate_lifecycle_spec,
                    ResidualAnchorLifecycleSpec,
                ):
                    raise ValueError(
                        "Formal strategy declares residual anchor lifecycle but "
                        "does not expose its typed spec."
                    )
                if (
                    candidate_lifecycle_spec.semantic_content_hash
                    != declared_lifecycle_hash
                ):
                    raise ValueError(
                        "Formal strategy exposes another residual anchor "
                        "lifecycle spec."
                    )
                if activation_scope is None:
                    raise ValueError(
                        "Residual anchor lifecycle evaluation requires an "
                        "activation scope."
                    )
                if not str(residual_anchor_lifecycle_scope_id or "").strip():
                    raise ValueError(
                        "Residual anchor lifecycle evaluation requires its "
                        "lifecycle scope ID."
                    )
                residual_anchor_lifecycle_spec = candidate_lifecycle_spec
            elif candidate_lifecycle_spec is not None:
                raise ValueError(
                    "Formal strategy exposes undeclared residual anchor lifecycle."
                )
            if strategy_spec.composition_spec_hash != spec.semantic_content_hash:
                raise ValueError(
                    "Formal strategy is bound to another CompositionSpec."
                )
            if (
                requested_context_channels
                != strategy_spec.context_channels
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
                if _strategy_control_binding is not None:
                    component_rule_set = (
                        _strategy_control_binding.revision.component_rule_set
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
            if _strategy_control_binding is not None:
                if (
                    _strategy_control_binding.parameter_snapshot.control_domain_id
                    != cut.control_domain_id
                    or _strategy_control_binding.parameter_snapshot.cutoff_at
                    != cut.cutoff_at
                    or (
                        _strategy_control_binding.parameter_snapshot
                        .cutoff_admission_sequence
                        != cut.cutoff_admission_sequence
                    )
                    or (
                        _strategy_control_binding.parameter_snapshot
                        .target_effective_at
                        != float(target_effective_at)
                    )
                ):
                    raise ValueError(
                        "Strategy-control binding does not match this causal cut."
                    )
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
                temporal_selection_spec=self.temporal_selection_spec,
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
            residual_anchor_lifecycle_snapshot: (
                ResidualAnchorLifecycleSnapshot | None
            ) = None
            if residual_anchor_lifecycle_spec is not None:
                assert activation_scope is not None
                residual_anchor_lifecycle_snapshot = (
                    capture_residual_anchor_lifecycle_snapshot(
                        self.residual_anchor_lifecycle,
                        cut,
                        producer_epochs=epoch_snapshot,
                        spec=residual_anchor_lifecycle_spec,
                        activation_scope=activation_scope,
                        lifecycle_scope_id=str(
                            residual_anchor_lifecycle_scope_id
                        ),
                        target_effective_at=target_effective_at,
                    )
                )
        blocked_reason: str | None = None
        blocked_result_status = "no_action"
        if context_snapshot is not None:
            missing_required_channels = tuple(
                channel
                for channel in context_snapshot.missing_channels
                if self.information_plan is None
                or channel in self.information_plan.required_context_channels
            )
            if missing_required_channels:
                blocked_reason = "missing_context:" + ",".join(
                    missing_required_channels
                )
            else:
                mismatches: list[str] = []
                if self.information_plan is not None:
                    selected_inputs = input_snapshot.selected_input_by_role
                    for requirement in (
                        self.information_plan.context_requirements
                    ):
                        if requirement.binding_mode != (
                            "selected_input_ancestor"
                        ) or requirement.channel not in (
                            context_snapshot.selected_context_id_by_channel
                        ):
                            continue
                        context_id = (
                            context_snapshot.selected_context_id_by_channel[
                                requirement.channel
                            ]
                        )
                        roles = (
                            requirement.bound_input_roles
                            or tuple(selected_inputs)
                        )
                        mismatches.extend(
                            f"{requirement.channel}@{role}"
                            for role in roles
                            if role not in selected_inputs
                            or context_id
                            not in selected_inputs[role].context_ids
                        )
                if mismatches:
                    blocked_reason = (
                        "context_dependency_mismatch:"
                        + ",".join(sorted(mismatches))
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
                            >= requirement.maximum_age_sec
                        )
                    )
                    if stale_channels:
                        blocked_reason = "stale_context:" + ",".join(
                            stale_channels
                        )
        pre_mix_policy_spec: CompositionPreMixPolicySpec | None = None
        pre_mix_policy_input: CompositionPreMixPolicyInput | None = None
        pre_mix_policy_decision: CompositionPreMixPolicyDecision | None = None
        pre_mix_policy_elapsed_sec: float | None = None
        pre_mix_policy_diagnostic: CompositionDiagnostic | None = None
        pre_mix_policy_failed = False
        if self.pre_mix_policy is not None:
            pre_mix_policy_spec = self.pre_mix_policy.spec
        if blocked_reason is None and self.pre_mix_policy is not None:
            policy_spec = self.pre_mix_policy.spec
            policy_input = CompositionPreMixPolicyInput(
                runtime_session_id=self.runtime_session_id,
                control_domain_id=self.control_domain_id,
                target_effective_at=target_effective_at,
                composition_spec=spec,
                input_snapshot=input_snapshot,
                context_snapshot=context_snapshot,
                information_plan=self.information_plan,
            )
            pre_mix_policy_input = policy_input
            started_at = time.perf_counter()
            try:
                candidate_decision = self.pre_mix_policy.decide(policy_input)
                if not isinstance(
                    candidate_decision,
                    CompositionPreMixPolicyDecision,
                ):
                    raise TypeError(
                        "Composition pre-mix policy decide() must return "
                        "CompositionPreMixPolicyDecision."
                    )
                if (
                    candidate_decision.policy_spec_hash
                    != policy_spec.semantic_content_hash
                ):
                    raise ValueError(
                        "Composition pre-mix policy decision uses another spec."
                    )
                if (
                    candidate_decision.policy_input_hash
                    != policy_input.semantic_content_hash
                ):
                    raise ValueError(
                        "Composition pre-mix policy decision uses another cut."
                    )
                pre_mix_policy_decision = candidate_decision
            except Exception as error:
                pre_mix_policy_failed = True
                pre_mix_policy_decision = CompositionPreMixPolicyDecision(
                    policy_spec_hash=policy_spec.semantic_content_hash,
                    policy_input_hash=policy_input.semantic_content_hash,
                    disposition="invalid",
                    reason=f"policy_error:{type(error).__name__}",
                )
                if debug_diagnostics_enabled:
                    pre_mix_policy_diagnostic = CompositionDiagnostic(
                        stage="pre_mix_policy",
                        code=type(error).__name__,
                        message=str(error),
                        recoverable=True,
                    )
            finally:
                pre_mix_policy_elapsed_sec = time.perf_counter() - started_at
            maximum_latency = policy_spec.maximum_decision_latency_sec
            if (
                not pre_mix_policy_failed
                and maximum_latency is not None
                and pre_mix_policy_elapsed_sec > maximum_latency
            ):
                pre_mix_policy_decision = CompositionPreMixPolicyDecision(
                    policy_spec_hash=policy_spec.semantic_content_hash,
                    policy_input_hash=policy_input.semantic_content_hash,
                    disposition="no_action",
                    reason="policy_latency_exceeded",
                )
                if debug_diagnostics_enabled:
                    pre_mix_policy_diagnostic = CompositionDiagnostic(
                        stage="pre_mix_policy",
                        code="latency_exceeded",
                        message=(
                            "Composition pre-mix policy exceeded its declared "
                            "decision latency."
                        ),
                        recoverable=True,
                    )
            if pre_mix_policy_decision.disposition != "pass":
                blocked_result_status = pre_mix_policy_decision.disposition
                blocked_reason = (
                    "pre_mix_policy:"
                    f"{policy_spec.policy_id}:"
                    f"{pre_mix_policy_decision.disposition}:"
                    f"{pre_mix_policy_decision.reason}"
                )
        strategy_input: FormalCompositionStrategyInput | None = None
        strategy_result_binding: (
            FormalCompositionStrategyResultBinding | None
        ) = None
        strategy_diagnostic: CompositionDiagnostic | None = None
        composition_supported_until = _composition_supported_until(
            input_snapshot=input_snapshot,
            context_snapshot=context_snapshot,
            source_eligibility_snapshot=source_eligibility_snapshot,
            information_plan=self.information_plan,
        )
        if strategy is None:
            result = (
                self.composer.compose(
                    input_snapshot,
                    spec=spec,
                    composition_supported_until=(
                        composition_supported_until
                    ),
                )
                if blocked_reason is None
                else build_blocked_composition_evaluation(
                    input_snapshot,
                    spec=spec,
                    status=blocked_result_status,
                    reason=blocked_reason,
                )
            )
        else:
            assert active_strategy_spec is not None
            strategy_input = FormalCompositionStrategyInput(
                runtime_session_id=self.runtime_session_id,
                control_domain_id=self.control_domain_id,
                clock_domain_id=(
                    self.coordinator.clock_domain.clock_domain_id
                ),
                strategy_spec=active_strategy_spec,
                composition_spec=spec,
                input_snapshot=input_snapshot,
                context_snapshot=context_snapshot,
                information_plan=self.information_plan,
                residual_anchor_lifecycle_snapshot=(
                    residual_anchor_lifecycle_snapshot
                ),
                strategy_control_binding_id=(
                    None
                    if _strategy_control_binding is None
                    else _strategy_control_binding.binding_id
                ),
                strategy_control_binding_hash=(
                    None
                    if _strategy_control_binding is None
                    else _strategy_control_binding.semantic_content_hash
                ),
            )
            if blocked_reason is not None:
                strategy_decision = CompositionStrategyDecision(
                    output_status=blocked_result_status,
                    reason=blocked_reason,
                    composed_candidate=None,
                )
            else:
                started_at = time.perf_counter()
                try:
                    with self._strategy_publication_lock:
                        if _strategy_control_binding is None:
                            strategy_decision = strategy.decide(strategy_input)
                        else:
                            decide_with_control = getattr(
                                strategy,
                                "decide_with_control",
                                None,
                            )
                            if not callable(decide_with_control):
                                raise TypeError(
                                    "Hot-controlled strategy must implement "
                                    "decide_with_control()."
                                )
                            strategy_decision = decide_with_control(
                                strategy_input,
                                _strategy_control_binding,
                            )
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
                        composed_candidate=None,
                    )
                    if debug_diagnostics_enabled:
                        strategy_diagnostic = CompositionDiagnostic(
                            stage="formal_strategy",
                            code=type(error).__name__,
                            message=str(error),
                            recoverable=True,
                        )
                else:
                    elapsed = time.perf_counter() - started_at
                    maximum_latency = (
                        active_strategy_spec.maximum_decision_latency_sec
                    )
                    if (
                        maximum_latency is not None
                        and elapsed > maximum_latency
                    ):
                        strategy_decision = CompositionStrategyDecision(
                            output_status="no_action",
                            reason="strategy_latency_exceeded",
                            composed_candidate=None,
                        )
                        if debug_diagnostics_enabled:
                            strategy_diagnostic = CompositionDiagnostic(
                                stage="formal_strategy",
                                code="strategy_latency_exceeded",
                                message=(
                                    "Formal inline strategy exceeded its "
                                    "declared decision latency."
                                ),
                                recoverable=True,
                            )
            result = build_formal_strategy_composition_evaluation(
                strategy_input,
                strategy_decision,
                composition_supported_until=composition_supported_until,
            )
            strategy_result_binding = build_formal_strategy_result_binding(
                strategy_input,
                result,
            )
        events = []
        if _strategy_control_binding is not None:
            events.extend(
                (
                    CompositionDomainEvent(
                        event_kind="composition_dynamic_parameter_snapshot",
                        subject=(
                            _strategy_control_binding.parameter_snapshot
                        ),
                        occurred_at=input_snapshot.cutoff_at,
                    ),
                    CompositionDomainEvent(
                        event_kind="composition_strategy_control_binding",
                        subject=_strategy_control_binding,
                        occurred_at=input_snapshot.cutoff_at,
                    ),
                )
            )
        events.append(
            CompositionDomainEvent(
                event_kind="composition_input_snapshot",
                subject=input_snapshot,
                occurred_at=input_snapshot.cutoff_at,
            )
        )
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
            if residual_anchor_lifecycle_snapshot is not None:
                events.append(
                    CompositionDomainEvent(
                        event_kind="residual_anchor_lifecycle_snapshot",
                        subject=residual_anchor_lifecycle_snapshot,
                        occurred_at=residual_anchor_lifecycle_snapshot.cutoff_at,
                    )
                )
            events.append(
                CompositionDomainEvent(
                    event_kind=(
                        "formal_composition_strategy_input"
                        if _strategy_control_binding is None
                        else "hot_composition_strategy_input"
                    ),
                    subject=strategy_input,
                    occurred_at=input_snapshot.cutoff_at,
                )
            )
        result_observed_at = float(self.clock())
        events.append(
            CompositionDomainEvent(
                event_kind="composition_evaluation",
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
                        subject_kind="evaluation",
                        subject_id=result.composition_evaluation_id,
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
        if pre_mix_policy_diagnostic is not None:
            diagnostics.append(pre_mix_policy_diagnostic)
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
                difference_error = (
                    f"{type(error).__name__}:{error}"
                    if debug_diagnostics_enabled
                    else type(error).__name__
                )
                if debug_diagnostics_enabled:
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
            action_schema_by_hash = {
                schema.semantic_content_hash: schema
                for schema in (
                    self.action_schema,
                    *self.input_action_schema_by_role.values(),
                )
            }
            payload_schema_by_hash = {
                payload_schema.semantic_content_hash: payload_schema
                for action_schema in action_schema_by_hash.values()
                for payload_schema in (
                    action_schema.payload_schema_by_component.values()
                )
            }
            action_payload_schemas_to_emit = tuple(
                payload_schema_by_hash[schema_hash]
                for schema_hash in sorted(payload_schema_by_hash)
                if schema_hash
                not in self._emitted_action_payload_schema_hashes
            )
            component_action_schemas_to_emit = tuple(
                action_schema_by_hash[schema_hash]
                for schema_hash in sorted(action_schema_by_hash)
                if schema_hash
                not in self._emitted_component_action_schema_hashes
            )
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
                active_strategy_spec is not None
                and active_strategy_spec.semantic_content_hash
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
            active_revision = (
                None
                if _strategy_control_binding is None
                else _strategy_control_binding.revision
            )
            emit_strategy_revision = (
                active_revision is not None
                and active_revision.semantic_content_hash
                not in self._emitted_strategy_revision_hashes
            )
            dynamic_parameter_contract = (
                None
                if active_revision is None
                else active_revision.dynamic_parameter_contract
            )
            emit_dynamic_parameter_contract = (
                dynamic_parameter_contract is not None
                and dynamic_parameter_contract.semantic_content_hash
                not in self._emitted_dynamic_parameter_contract_hashes
            )
            evaluation_events: list[CompositionDomainEvent] = []
            plan_event = self._information_plan_event(
                occurred_at=input_snapshot.cutoff_at
            )
            if plan_event is not None:
                evaluation_events.append(plan_event)
            if not self._temporal_selection_spec_emitted:
                evaluation_events.append(
                    CompositionDomainEvent(
                        event_kind="composition_temporal_selection_spec",
                        subject=self.temporal_selection_spec,
                        occurred_at=input_snapshot.cutoff_at,
                    )
                )
            evaluation_events.extend(
                CompositionDomainEvent(
                    event_kind="action_payload_schema",
                    subject=payload_schema,
                    occurred_at=input_snapshot.cutoff_at,
                )
                for payload_schema in action_payload_schemas_to_emit
            )
            evaluation_events.extend(
                CompositionDomainEvent(
                    event_kind="component_action_schema",
                    subject=action_schema,
                    occurred_at=input_snapshot.cutoff_at,
                )
                for action_schema in component_action_schemas_to_emit
            )
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
                        subject=active_strategy_spec,
                        occurred_at=input_snapshot.cutoff_at,
                    )
                )
            if emit_dynamic_parameter_contract:
                evaluation_events.append(
                    CompositionDomainEvent(
                        event_kind="composition_dynamic_parameter_contract",
                        subject=dynamic_parameter_contract,
                        occurred_at=input_snapshot.cutoff_at,
                    )
                )
            if emit_strategy_revision:
                evaluation_events.append(
                    CompositionDomainEvent(
                        event_kind="composition_strategy_revision",
                        subject=active_revision,
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
            emission_succeeded = self._emit(*evaluation_events)
            if emission_succeeded and plan_event is not None:
                self._information_plan_emitted = True
            if emission_succeeded:
                self._temporal_selection_spec_emitted = True
                self._emitted_action_payload_schema_hashes.update(
                    schema.semantic_content_hash
                    for schema in action_payload_schemas_to_emit
                )
                self._emitted_component_action_schema_hashes.update(
                    schema.semantic_content_hash
                    for schema in component_action_schemas_to_emit
                )
            if emission_succeeded and emit_composition_spec:
                self._emitted_composition_spec_hashes.add(
                    composition_spec_hash
                )
            if emission_succeeded and emit_source_eligibility_spec:
                self._emitted_source_eligibility_spec_hashes.add(
                    source_eligibility_spec.semantic_content_hash
                )
            if emission_succeeded and emit_difference_spec:
                self._emitted_source_difference_spec_hashes.add(
                    source_difference_spec.semantic_content_hash
                )
            if emission_succeeded and emit_formal_strategy_spec:
                self._emitted_formal_strategy_spec_hashes.add(
                    active_strategy_spec.semantic_content_hash
                )
            if emission_succeeded and emit_dynamic_parameter_contract:
                self._emitted_dynamic_parameter_contract_hashes.add(
                    dynamic_parameter_contract.semantic_content_hash
                )
            if emission_succeeded and emit_strategy_revision:
                self._emitted_strategy_revision_hashes.add(
                    active_revision.semantic_content_hash
                )
            if emission_succeeded and emit_component_rule_set:
                self._emitted_component_rule_set_hashes.add(
                    component_rule_set.semantic_content_hash
                )
            if emission_succeeded and emit_formal_component_binding:
                self._emitted_formal_component_binding_hashes.add(
                    formal_component_binding.semantic_content_hash
                )
            emitted_events = (
                tuple(evaluation_events) if emission_succeeded else ()
            )
        evaluation_record = CompositionEvaluationRecord(
            information_plan=self.information_plan,
            composition_spec=spec,
            control_cut=cut,
            producer_epoch_snapshot=epoch_snapshot,
            input_snapshot=input_snapshot,
            composition_evaluation=result,
            required_context_channels=requested_context_channels,
            context_snapshot=context_snapshot,
            source_eligibility_spec=source_eligibility_spec,
            source_eligibility_snapshot=source_eligibility_snapshot,
            residual_anchor_lifecycle_snapshot=(
                residual_anchor_lifecycle_snapshot
            ),
            strategy_spec=active_strategy_spec,
            strategy_input=strategy_input,
            strategy_result_binding=strategy_result_binding,
            strategy_control_binding=_strategy_control_binding,
            source_difference_spec=source_difference_spec,
            source_difference_observation=difference_observation,
            diagnostics=tuple(diagnostics),
            domain_events_emitted=emitted_events,
            pre_mix_policy_spec=pre_mix_policy_spec,
            pre_mix_policy_input=pre_mix_policy_input,
            pre_mix_policy_decision=pre_mix_policy_decision,
            pre_mix_policy_elapsed_sec=(
                pre_mix_policy_elapsed_sec
                if debug_diagnostics_enabled
                else None
            ),
        )
        return CompositeProposalSubmission(
            input_snapshot=input_snapshot,
            composition_evaluation=result,
            proposal_admission=None,
            context_snapshot=context_snapshot,
            source_difference_observation=difference_observation,
            source_difference_error=difference_error,
            evaluation_record=evaluation_record,
            source_eligibility_snapshot=source_eligibility_snapshot,
            residual_anchor_lifecycle_snapshot=(
                residual_anchor_lifecycle_snapshot
            ),
        )


__all__ = [
    "COMPOSITION_PROMOTION_REJECTION_SUPPORT_EXPIRED",
    "CompositeProposalSubmission",
    "ControlCompositionRuntime",
]
