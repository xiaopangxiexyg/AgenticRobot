"""Backend-neutral single-domain authoritative control orchestration.

The runtime assembles existing proposal, arbitration, persistence, safety,
dispatch-owner and trace contracts.  It does not own sensors, policy inference,
IK implementations, a dataset episode, or a concrete backend lifecycle.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

from dataset_contracts.control_trace import ControlTraceFact
from dataset_contracts.replay import (
    REPLAY_BINDING_USE_EXECUTION_CANDIDATE,
    REPLAY_BINDING_VERIFIED_COMPATIBLE,
    CandidateBindingEligibility,
    ReplayBindingAssessment,
    ReplayExecutionAuthorization,
)
from robot_motion.runtime import CommandDispatchReceipt, TeleopCommand
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.dispatch_owner import (
    DispatchOwnerRegistry,
    DispatchOwnerToken,
    DispatchPermitValidationReceipt,
)
from robot_motion.runtime.cancellation import (
    CancellationAcknowledgementTracker,
    ExecutionCancellation,
    ExecutionCancellationReceipt,
    ExecutionCancellationRequest,
    ExecutionEpochRegistry,
)
from robot_motion.runtime.execution_gate import (
    ExecutionGateEvent,
    ExecutionGateEventDraft,
    ExecutionGateSnapshot,
    ExecutionGateStore,
)
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.observation_cut import ObservationCutRecord
from robot_motion.runtime.producer_epoch import (
    ProducerEpochActivation,
    ProducerEpochRegistry,
)
from robot_motion.position_control.capability import ExecutionCapabilityProfile

from .arbiter import ActionArbiter, ArbitrationResult
from .arbitration_input import ArbitrationInputSnapshot, capture_arbitration_input
from .arbitration_runtime import (
    ArbitrationPublicationAttempt,
    ArbitrationRuntime,
    AuthoritativeArbitrationDecision,
    DecisionScheduleSpec,
    DecisionScheduler,
)
from .authority import AuthorityEvent, AuthorityEventDraft, AuthorityPolicySpec, AuthorityStore
from .authoritative_trace import (
    AuthoritativeTraceContext,
    project_arbitration_computation,
    project_arbitration_input,
    project_authority_event,
    project_candidate_authorization_admission,
    project_command,
    project_compound_authority_failure,
    project_conservative_compound_authority_profile,
    project_dispatch_permit_validation,
    project_dispatch_owner_validation,
    project_dispatch_receipts,
    project_execution_cancellation,
    project_execution_cancellation_ack,
    project_execution_directive,
    project_execution_gate_event,
    project_measurement_alignment,
    project_observation_cut,
    project_producer_epoch_activation,
    project_proposal_admission,
    project_publication,
    project_replay_execution_authorization,
    project_shared_safety_result,
)
from .components import ActionComponentCatalog
from .compound_authority import ConservativeCompoundAuthorityProfile
from .dispatch_context import (
    AuthoritativeCommandDispatchContext,
    AuthoritativeReceiverStateSnapshot,
    build_authoritative_receiver_state_snapshot,
    build_authoritative_command_dispatch_context,
)
from .candidate_authorization import (
    CandidateAuthorizationAdmissionReceipt,
    CandidateAuthorizationDraft,
    CandidateAuthorizationStore,
)
from .executor import RobotActionResolver
from .persistence import (
    ActionPersistencePolicy,
    ActiveExecutionSegment,
    ActiveExecutionSnapshot,
    ExecutionDirective,
    build_execution_directive,
)
from .proposal import ActionProposalDraft, ProposalAdmissionReceipt, ProposalBuffer
from .safety import SharedCanonicalSafetyResult
from .schema import RobotAction


TraceSink = Callable[[ControlTraceFact], None]
CommandDispatcher = Callable[[TeleopCommand], Mapping[str, CommandDispatchReceipt]]
ContextualCommandDispatcher = Callable[
    [TeleopCommand, AuthoritativeCommandDispatchContext],
    Mapping[str, CommandDispatchReceipt],
]
SharedSafety = Callable[[RobotAction, str], SharedCanonicalSafetyResult]
BackendHold = Callable[[ExecutionCancellation], Mapping[str, bool]]


@dataclass(frozen=True)
class AuthoritativeControlConfig:
    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str
    owner_runtime_id: str
    clock_domain: ClockDomainDescriptor
    component_catalog: ActionComponentCatalog
    active_selection_groups: tuple[str, ...]
    authority_policy: AuthorityPolicySpec
    decision_schedule: DecisionScheduleSpec
    persistence_policy: ActionPersistencePolicy
    execution_capability: ExecutionCapabilityProfile
    proposal_capacity: int = 4096
    candidate_authorization_capacity: int = 4096
    target_runtime_contract_hash: str | None = None
    required_authorization_producer_kinds: tuple[str, ...] = ("replay",)
    initial_authority_epoch: int = 0
    initial_execution_epoch: int = 0
    initial_control_mode: str = "policy"
    defer_selected_segment_activation: bool = False
    compound_authority_profiles: tuple[
        ConservativeCompoundAuthorityProfile, ...
    ] = ()

    def __post_init__(self) -> None:
        for name in (
            "runtime_session_id",
            "control_domain_id",
            "embodiment_instance_id",
            "owner_runtime_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        if self.clock_domain.runtime_session_id != self.runtime_session_id:
            raise ValueError("Control clock and runtime session must match.")
        groups = tuple(dict.fromkeys(str(value) for value in self.active_selection_groups))
        if not groups or not set(groups).issubset(self.component_catalog.selection_group_ids):
            raise ValueError("active_selection_groups must be a non-empty catalog subset.")
        object.__setattr__(self, "active_selection_groups", groups)
        if int(self.proposal_capacity) <= 0:
            raise ValueError("proposal_capacity must be positive.")
        if int(self.candidate_authorization_capacity) <= 0:
            raise ValueError("candidate_authorization_capacity must be positive.")
        if self.decision_schedule.decision_hz > (
            self.execution_capability.configured_selected_keyframe_hz
        ):
            raise ValueError(
                "Decision schedule exceeds the configured execution capability."
            )
        if not set(groups).issubset(
            self.execution_capability.validated_selection_groups
        ):
            raise ValueError(
                "Active selection groups exceed the execution capability evidence."
            )
        if self.target_runtime_contract_hash is not None and not str(
            self.target_runtime_contract_hash
        ).strip():
            raise ValueError("target_runtime_contract_hash must be non-empty when set.")
        kinds = tuple(
            dict.fromkeys(str(value).strip() for value in self.required_authorization_producer_kinds)
        )
        if any(not value for value in kinds):
            raise ValueError("required authorization producer kinds must be non-empty.")
        object.__setattr__(self, "required_authorization_producer_kinds", kinds)
        profiles = tuple(self.compound_authority_profiles)
        if any(
            not isinstance(profile, ConservativeCompoundAuthorityProfile)
            for profile in profiles
        ):
            raise TypeError(
                "compound_authority_profiles must contain only "
                "ConservativeCompoundAuthorityProfile values."
            )
        producer_ids = tuple(profile.producer_id for profile in profiles)
        if len(producer_ids) != len(set(producer_ids)):
            raise ValueError(
                "Only one compound authority profile may bind each producer."
            )
        for profile in profiles:
            profile.validate_runtime_binding(
                component_catalog=self.component_catalog,
                active_selection_groups=groups,
                authority_policy=self.authority_policy,
            )
        object.__setattr__(self, "compound_authority_profiles", profiles)


@dataclass(frozen=True)
class AuthoritativeControlStep:
    status: str
    snapshot: ArbitrationInputSnapshot | None = None
    computation: ArbitrationResult | None = None
    publication_attempt: ArbitrationPublicationAttempt | None = None
    decision: AuthoritativeArbitrationDecision | None = None
    directive: ExecutionDirective | None = None
    safety_result: SharedCanonicalSafetyResult | None = None
    command: TeleopCommand | None = None
    dispatch_context: AuthoritativeCommandDispatchContext | None = None
    dispatch_validation: DispatchPermitValidationReceipt | None = None
    dispatch_receipts: Mapping[str, CommandDispatchReceipt] | None = None
    command_id: str | None = None
    dispatch_receipt_id: str | None = None
    cancellation: ExecutionCancellation | None = None
    cancellation_receipt: ExecutionCancellationReceipt | None = None
    cancellation_dispatch_validation: DispatchPermitValidationReceipt | None = None
    trace_facts: tuple[ControlTraceFact, ...] = ()

    def __post_init__(self) -> None:
        if self.dispatch_receipts is not None:
            object.__setattr__(
                self,
                "dispatch_receipts",
                MappingProxyType(dict(self.dispatch_receipts)),
            )
        object.__setattr__(self, "trace_facts", tuple(self.trace_facts))


@dataclass(frozen=True)
class ExecutionGateTransitionOutcome:
    """Typed result of one receiver-admitted execution-gate transition.

    The outcome exposes the exact cancellation and trace evidence produced by
    this transition so application compositions never need to infer causality
    by slicing the runtime trace or reaching into acknowledgement stores.
    """

    event: ExecutionGateEvent
    snapshot: ExecutionGateSnapshot
    cancellation: ExecutionCancellation | None = None
    cancellation_receipt: ExecutionCancellationReceipt | None = None
    cancellation_dispatch_validation: DispatchPermitValidationReceipt | None = None
    trace_facts: tuple[ControlTraceFact, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_facts", tuple(self.trace_facts))

    @property
    def lower_event_ids(self) -> tuple[str, ...]:
        """Stable IDs for the gate event and all resulting lower facts."""

        return tuple(
            dict.fromkeys(
                (self.event.event_id, *(fact.fact_id for fact in self.trace_facts))
            )
        )


class AuthoritativeControlRuntime:
    """One authority/publication/dispatch chain for one control domain."""

    def __init__(
        self,
        config: AuthoritativeControlConfig,
        *,
        dispatcher: CommandDispatcher,
        backend_hold: BackendHold,
        backend_name: str,
        resolver: RobotActionResolver | None = None,
        shared_safety: SharedSafety | None = None,
        trace_sink: TraceSink | None = None,
        contextual_dispatcher: ContextualCommandDispatcher | None = None,
        dispatch_owners: DispatchOwnerRegistry | None = None,
        dispatch_owner_token: DispatchOwnerToken | None = None,
        execution_epochs: ExecutionEpochRegistry | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(dispatcher) or not callable(backend_hold) or not callable(clock):
            raise TypeError("dispatcher, backend_hold, and clock must be callable.")
        if not str(backend_name).strip():
            raise ValueError("backend_name must be non-empty.")
        self.config = config
        self.dispatcher = dispatcher
        self.contextual_dispatcher = contextual_dispatcher
        self.backend_hold = backend_hold
        self.backend_name = str(backend_name)
        self.resolver = resolver or RobotActionResolver()
        self.shared_safety = shared_safety or self._identity_safety
        self.trace_sink = trace_sink
        self.clock = clock
        self.trace_context = AuthoritativeTraceContext(
            runtime_session_id=config.runtime_session_id,
            control_domain_id=config.control_domain_id,
            embodiment_instance_id=config.embodiment_instance_id,
            clock_domain_id=config.clock_domain.clock_domain_id,
        )
        self.coordinator = ControlAdmissionCoordinator(
            config.control_domain_id,
            clock_domain=config.clock_domain,
            clock=clock,
        )
        self.producer_epochs = ProducerEpochRegistry(
            runtime_session_id=config.runtime_session_id,
            coordinator=self.coordinator,
        )
        self.proposals = ProposalBuffer(
            runtime_session_id=config.runtime_session_id,
            coordinator=self.coordinator,
            producer_epochs=self.producer_epochs,
            capacity=config.proposal_capacity,
        )
        self._candidate_authorizations = CandidateAuthorizationStore(
            runtime_session_id=config.runtime_session_id,
            coordinator=self.coordinator,
            proposals=self.proposals,
            capacity=config.candidate_authorization_capacity,
        )
        self.authority_store = AuthorityStore(
            runtime_session_id=config.runtime_session_id,
            coordinator=self.coordinator,
            policy=config.authority_policy,
            initial_mode=config.initial_control_mode,
        )
        self.execution_gate_store = ExecutionGateStore(
            runtime_session_id=config.runtime_session_id,
            coordinator=self.coordinator,
        )
        self.scheduler = DecisionScheduler(config.decision_schedule, clock=clock)
        self.arbiter = ActionArbiter()
        self.arbitration_runtime = ArbitrationRuntime(config.decision_schedule)
        injected_owner_state = (
            dispatch_owners is not None,
            dispatch_owner_token is not None,
            execution_epochs is not None,
        )
        if any(injected_owner_state) and not all(injected_owner_state):
            raise ValueError(
                "dispatch_owners, dispatch_owner_token, and execution_epochs "
                "must be injected together."
            )
        if dispatch_owners is None:
            self.dispatch_owners = DispatchOwnerRegistry(
                runtime_session_id=config.runtime_session_id,
                control_domain_id=config.control_domain_id,
                initial_authority_epoch=config.initial_authority_epoch,
                initial_execution_epoch=config.initial_execution_epoch,
                clock=clock,
            )
            self.dispatch_owner_token = self.dispatch_owners.grant_initial_owner(
                owner_runtime_id=config.owner_runtime_id,
                owner_kind="shared_arbitration_runtime",
            )
        else:
            self.dispatch_owners = dispatch_owners
            assert dispatch_owner_token is not None
            self.dispatch_owner_token = self._validate_injected_dispatch_owner(
                config,
                registry=dispatch_owners,
                token=dispatch_owner_token,
            )

        if execution_epochs is None:
            self.execution_epochs = ExecutionEpochRegistry(
                runtime_session_id=config.runtime_session_id,
                control_domain_id=config.control_domain_id,
                clock_domain_id=config.clock_domain.clock_domain_id,
                initial_execution_epoch=config.initial_execution_epoch,
                initial_segment_generation_by_group={
                    group: 0 for group in config.active_selection_groups
                },
                clock=clock,
            )
        else:
            self.execution_epochs = self._validate_injected_execution_epochs(
                config,
                registry=execution_epochs,
                dispatch_registry=self.dispatch_owners,
            )
        self.cancellation_acknowledgements = CancellationAcknowledgementTracker(
            clock=clock
        )
        self._active_segment_by_group: dict[str, ActiveExecutionSegment] = {}
        self._pending_segment_by_group: dict[str, ActiveExecutionSegment] = {}
        self._backend_hold_active = False
        self._authority_admission_lock = threading.RLock()
        self._next_authority_sequence = 0
        self._next_execution_gate_sequence = 0
        self._previous_decision_id: str | None = None
        self._previous_selection_by_group: dict[str, str] = {}
        self._facts: list[ControlTraceFact] = []
        self._terminal = False
        self._compound_authority_profile_by_producer = {
            profile.producer_id: profile
            for profile in config.compound_authority_profiles
        }
        self._compound_authority_profiles_recorded = False

    @staticmethod
    def _validate_injected_dispatch_owner(
        config: AuthoritativeControlConfig,
        *,
        registry: DispatchOwnerRegistry,
        token: DispatchOwnerToken,
    ) -> DispatchOwnerToken:
        if not isinstance(registry, DispatchOwnerRegistry):
            raise TypeError("dispatch_owners must be a DispatchOwnerRegistry.")
        if not isinstance(token, DispatchOwnerToken):
            raise TypeError("dispatch_owner_token must be a DispatchOwnerToken.")
        snapshot = registry.snapshot()
        if (
            snapshot.runtime_session_id != config.runtime_session_id
            or snapshot.control_domain_id != config.control_domain_id
        ):
            raise ValueError("Injected dispatch owner belongs to another session/domain.")
        if snapshot.current_token is None:
            raise ValueError("Injected dispatch owner registry has no current token.")
        if (
            snapshot.current_token.token_id != token.token_id
            or snapshot.current_token.semantic_content_hash
            != token.semantic_content_hash
        ):
            raise ValueError("Injected dispatch owner token is not current.")
        if token.owner_kind != "shared_arbitration_runtime":
            raise ValueError(
                "AuthoritativeControlRuntime requires a shared-arbitration owner token."
            )
        if token.owner_runtime_id != config.owner_runtime_id:
            raise ValueError("Injected dispatch owner runtime does not match config.")
        if snapshot.current_authority_epoch != config.initial_authority_epoch:
            raise ValueError("Injected authority epoch does not match config initial epoch.")
        if snapshot.current_execution_epoch != config.initial_execution_epoch:
            raise ValueError("Injected dispatch execution epoch does not match config.")
        return token

    @staticmethod
    def _validate_injected_execution_epochs(
        config: AuthoritativeControlConfig,
        *,
        registry: ExecutionEpochRegistry,
        dispatch_registry: DispatchOwnerRegistry,
    ) -> ExecutionEpochRegistry:
        if not isinstance(registry, ExecutionEpochRegistry):
            raise TypeError("execution_epochs must be an ExecutionEpochRegistry.")
        if (
            registry.runtime_session_id != config.runtime_session_id
            or registry.control_domain_id != config.control_domain_id
            or registry.clock_domain_id != config.clock_domain.clock_domain_id
        ):
            raise ValueError("Injected execution epochs belong to another runtime scope.")
        snapshot = registry.snapshot()
        dispatch_snapshot = dispatch_registry.snapshot()
        if snapshot.execution_epoch != config.initial_execution_epoch:
            raise ValueError("Injected execution epoch does not match config initial epoch.")
        if snapshot.execution_epoch != dispatch_snapshot.current_execution_epoch:
            raise ValueError("Dispatch-owner and execution registries disagree on epoch.")
        missing_groups = set(config.active_selection_groups).difference(
            snapshot.segment_generation_by_group
        )
        if missing_groups:
            raise ValueError(
                "Injected execution registry lacks active selection groups: "
                f"{sorted(missing_groups)!r}."
            )
        return registry

    @staticmethod
    def _identity_safety(action: RobotAction, input_action_id: str) -> SharedCanonicalSafetyResult:
        return SharedCanonicalSafetyResult(
            input_action_id=input_action_id,
            safe_action=action,
            status="safe",
        )

    @property
    def trace_facts(self) -> tuple[ControlTraceFact, ...]:
        return tuple(self._facts)

    @property
    def execution_epoch(self) -> int:
        return self.execution_epochs.snapshot().execution_epoch

    @property
    def active_execution_segment_by_group(
        self,
    ) -> Mapping[str, ActiveExecutionSegment]:
        return MappingProxyType(dict(self._active_segment_by_group))

    @property
    def pending_execution_segment_by_group(
        self,
    ) -> Mapping[str, ActiveExecutionSegment]:
        return MappingProxyType(dict(self._pending_segment_by_group))

    def capture_receiver_state(self) -> AuthoritativeReceiverStateSnapshot:
        """Capture live state for a receiver-side second validation.

        This is sufficient for the deterministic single-threaded MuJoCo c2
        slice.  It does not claim the stronger cross-thread worker locking
        required before M5.8 real-backend integration.
        """

        owner = self.dispatch_owners.snapshot()
        token = owner.current_token
        return build_authoritative_receiver_state_snapshot(
            runtime_session_id=self.config.runtime_session_id,
            control_domain_id=self.config.control_domain_id,
            current_owner_token_id=None if token is None else token.token_id,
            current_owner_generation=(
                None if token is None else token.owner_generation
            ),
            current_authority_epoch=owner.current_authority_epoch,
            current_execution_epoch=self.execution_epoch,
            expected_next_group_generation_by_group={
                group: (
                    1
                    if self._active_segment_by_group.get(group) is None
                    else self._active_segment_by_group[
                        group
                    ].group_segment_generation
                    + 1
                )
                for group in self.config.active_selection_groups
            },
            captured_at=float(self.clock()),
        )

    def activate_pending_execution_segment(
        self,
        *,
        selection_group_id: str,
        group_segment_generation: int,
    ) -> ActiveExecutionSegment:
        """Promote one receiver-accepted future segment at its actual START.

        This scoped seam keeps scheduled receiver activation separate from
        command admission.  A future segment is not active merely because its
        mailbox write succeeded.
        """

        group = str(selection_group_id).strip()
        activated = self.activate_pending_execution_segments(
            group_segment_generation_by_group={
                group: int(group_segment_generation)
            }
        )
        return activated[group]

    def activate_pending_execution_segments(
        self,
        *,
        group_segment_generation_by_group: Mapping[str, int],
    ) -> Mapping[str, ActiveExecutionSegment]:
        """Atomically promote an exact set of receiver-accepted group segments.

        Every pending group and generation is validated before any runtime
        state changes.  This is the activation seam used when independently
        arbitrated arm/hand groups share one compound backend target.
        """

        if not self.config.defer_selected_segment_activation:
            raise RuntimeError("This runtime does not defer segment activation.")
        requested = {
            str(group).strip(): int(generation)
            for group, generation in group_segment_generation_by_group.items()
        }
        if not requested or any(not group for group in requested):
            raise ValueError(
                "group_segment_generation_by_group must be non-empty."
            )
        with self._authority_admission_lock:
            pending_by_group: dict[str, ActiveExecutionSegment] = {}
            for group, generation in requested.items():
                pending = self._pending_segment_by_group.get(group)
                if pending is None:
                    raise RuntimeError(
                        f"No pending execution segment exists for {group!r}."
                    )
                if pending.group_segment_generation != generation:
                    raise RuntimeError(
                        f"Pending execution segment generation mismatch for {group!r}."
                    )
                pending_by_group[group] = pending
            for group, pending in pending_by_group.items():
                self._active_segment_by_group[group] = pending
            for group in pending_by_group:
                self._pending_segment_by_group.pop(group, None)
            return MappingProxyType(dict(sorted(pending_by_group.items())))

    def _emit(self, facts: tuple[ControlTraceFact, ...] | list[ControlTraceFact]) -> None:
        for fact in facts:
            self._facts.append(fact)
            if self.trace_sink is not None:
                self.trace_sink(fact)

    def activate_producer(
        self,
        *,
        producer_id: str,
        producer_epoch: int,
        produced_at: float | None = None,
        produced_clock_domain_id: str | None = None,
        reason: str = "startup",
    ) -> ProducerEpochActivation:
        timestamp = float(self.clock() if produced_at is None else produced_at)
        activation = self.producer_epochs.activate(
            producer_id=producer_id,
            producer_epoch=producer_epoch,
            produced_at=timestamp,
            produced_clock_domain_id=(
                self.config.clock_domain.clock_domain_id
                if produced_clock_domain_id is None
                else produced_clock_domain_id
            ),
            reason=reason,
        )
        self._emit((project_producer_epoch_activation(self.trace_context, activation),))
        return activation

    def record_observation_cut(self, record: ObservationCutRecord) -> ControlTraceFact:
        if (
            record.runtime_session_id != self.config.runtime_session_id
            or record.control_domain_id != self.config.control_domain_id
        ):
            raise ValueError("Observation cut belongs to another session/control domain.")
        fact = project_observation_cut(self.trace_context, record)
        self._emit((fact,))
        return fact

    def admit_authority_event(
        self,
        *,
        event_type: str,
        reason: str,
        selection_group_id: str | None = None,
        owner_producer_id: str | None = None,
        control_mode: str | None = None,
        expires_at: float | None = None,
        expires_clock_domain_id: str | None = None,
        produced_at: float | None = None,
    ) -> AuthorityEvent:
        """Admit and trace one typed authority transition.

        This is the authoritative runtime seam used by application-level
        handoff compositions.  It carries no action and grants no dispatch
        permit by itself; the next immutable arbitration cut consumes it.
        """

        with self._authority_admission_lock:
            event = self.authority_store.admit(
                AuthorityEventDraft(
                    runtime_session_id=self.config.runtime_session_id,
                    control_domain_id=self.config.control_domain_id,
                    sequence=self._next_authority_sequence,
                    event_type=event_type,
                    produced_at=float(
                        self.clock() if produced_at is None else produced_at
                    ),
                    produced_clock_domain_id=(
                        self.config.clock_domain.clock_domain_id
                    ),
                    selection_group_id=selection_group_id,
                    owner_producer_id=owner_producer_id,
                    control_mode=control_mode,
                    expires_at=expires_at,
                    expires_clock_domain_id=expires_clock_domain_id,
                    reason=reason,
                )
            )
            self._next_authority_sequence += 1
            self.dispatch_owners.observe_authority_epoch(
                authority_epoch=event.authority_epoch,
                evidence_event_id=event.event_id,
            )
            self._emit((project_authority_event(self.trace_context, event),))
            return event

    def _admit_execution_gate(
        self,
        *,
        status: str,
        reason: str,
        source_event_ids: tuple[str, ...] = (),
        produced_at: float | None = None,
        emit: bool,
    ) -> tuple[ExecutionGateEvent, ControlTraceFact]:
        event = self.execution_gate_store.admit(
            ExecutionGateEventDraft(
                runtime_session_id=self.config.runtime_session_id,
                control_domain_id=self.config.control_domain_id,
                sequence=self._next_execution_gate_sequence,
                status=status,
                produced_at=float(
                    self.clock() if produced_at is None else produced_at
                ),
                produced_clock_domain_id=self.config.clock_domain.clock_domain_id,
                reason=reason,
                source_event_ids=tuple(source_event_ids),
            )
        )
        self._next_execution_gate_sequence += 1
        fact = project_execution_gate_event(self.trace_context, event)
        if emit:
            self._emit((fact,))
        return event, fact

    def admit_execution_gate(
        self,
        *,
        status: str,
        reason: str,
        source_event_ids: tuple[str, ...] = (),
        produced_at: float | None = None,
    ) -> ExecutionGateEvent:
        """Compatibility seam returning only the admitted gate event."""

        return self.transition_execution_gate(
            status=status,
            reason=reason,
            source_event_ids=source_event_ids,
            produced_at=produced_at,
        ).event

    def transition_execution_gate(
        self,
        *,
        status: str,
        reason: str,
        source_event_ids: tuple[str, ...] = (),
        produced_at: float | None = None,
        force_cancellation: bool = False,
    ) -> ExecutionGateTransitionOutcome:
        """Admit one gate transition and return its exact causal outcome."""

        if not isinstance(force_cancellation, bool):
            raise TypeError("force_cancellation must be a bool.")
        normalized_status = str(status).strip().lower()
        if normalized_status == "ready" and self.execution_epochs.snapshot().fault_latched:
            raise RuntimeError(
                "A latched execution fault requires explicit recovery before READY."
            )
        event, gate_fact = self._admit_execution_gate(
            status=normalized_status,
            reason=reason,
            source_event_ids=source_event_ids,
            produced_at=produced_at,
            emit=True,
        )
        cancellation = None
        cancellation_receipt = None
        cancellation_validation = None
        transition_facts: list[ControlTraceFact] = [gate_fact]
        if normalized_status in {"hold", "fault", "transitioning"}:
            should_cancel = force_cancellation or not self._backend_hold_active or (
                normalized_status == "fault"
                and not self.execution_epochs.snapshot().fault_latched
            )
            if should_cancel:
                (
                    cancellation,
                    cancellation_receipt,
                    cancellation_validation,
                    facts,
                ) = self._cancel_execution(
                    reason="fault" if normalized_status == "fault" else "hold",
                    authority_epoch=(
                        self.dispatch_owners.snapshot().current_authority_epoch
                    ),
                    selection_group_ids=self.config.active_selection_groups,
                    source_event_ids=(event.event_id,),
                    force_domain_scope=True,
                )
                self._emit(facts)
                transition_facts.extend(facts)
        return ExecutionGateTransitionOutcome(
            event=event,
            snapshot=self.capture_execution_gate_snapshot(now=event.available_at),
            cancellation=cancellation,
            cancellation_receipt=cancellation_receipt,
            cancellation_dispatch_validation=cancellation_validation,
            trace_facts=tuple(transition_facts),
        )

    def record_external_facts(
        self,
        facts: tuple[ControlTraceFact, ...],
    ) -> None:
        """Attach already-projected pure source facts to this runtime trace."""

        existing = {fact.fact_id: fact.semantic_content_hash for fact in self._facts}
        admitted: list[ControlTraceFact] = []
        for fact in facts:
            if (
                fact.runtime_session_id != self.config.runtime_session_id
                or fact.control_domain_id != self.config.control_domain_id
                or fact.embodiment_instance_id != self.config.embodiment_instance_id
            ):
                raise ValueError("External trace fact belongs to another runtime target.")
            previous_hash = existing.get(fact.fact_id)
            if previous_hash is not None:
                if previous_hash != fact.semantic_content_hash:
                    raise ValueError("External trace fact ID was reused with new content.")
                continue
            existing[fact.fact_id] = fact.semantic_content_hash
            admitted.append(fact)
        self._emit(admitted)

    def record_compound_authority_profiles(self) -> tuple[ControlTraceFact, ...]:
        """Record configured profile identities once before authoritative use."""

        if self._compound_authority_profiles_recorded:
            return ()
        occurred_at = float(self.clock())
        facts = tuple(
            project_conservative_compound_authority_profile(
                self.trace_context,
                profile,
                occurred_at=occurred_at,
            )
            for profile in self.config.compound_authority_profiles
        )
        self._emit(facts)
        self._compound_authority_profiles_recorded = True
        return facts

    def admit_proposal(
        self,
        draft: ActionProposalDraft,
        *,
        source_event_id: str,
    ) -> ProposalAdmissionReceipt:
        compound_profile = self._compound_authority_profile_by_producer.get(
            draft.producer_id
        )
        compound_rejection = (
            None
            if compound_profile is None
            else compound_profile.validate_proposal_draft(
                draft,
                component_catalog=self.config.component_catalog,
            )
        )
        if compound_rejection is None:
            receipt = self.proposals.admit(draft)
        else:
            receipt = ProposalAdmissionReceipt(
                accepted=False,
                rejection_code="compound_authority_scope_mismatch",
                message=(
                    "Conservative compound authority rejected a non-atomic "
                    f"proposal: {compound_rejection}."
                ),
            )
        self._emit(
            project_proposal_admission(
                self.trace_context,
                receipt,
                source_event_id=source_event_id,
                occurred_at=float(self.clock()),
            )
        )
        return receipt

    def _compound_profile_for_decision(
        self,
        decision: AuthoritativeArbitrationDecision,
    ) -> ConservativeCompoundAuthorityProfile | None:
        selected_ids = set(dict(decision.selected_by_group).values())
        if len(selected_ids) != 1:
            return None
        proposal = self.proposals.get(next(iter(selected_ids)))
        if proposal is None:
            return None
        profile = self._compound_authority_profile_by_producer.get(
            proposal.producer_id
        )
        if profile is None:
            return None
        if set(decision.selected_by_group) != set(profile.selection_groups):
            return None
        return profile

    def _compound_failure_fact(
        self,
        profile: ConservativeCompoundAuthorityProfile,
        *,
        stage: str,
        source_event_ids: tuple[str, ...],
    ) -> ControlTraceFact:
        return project_compound_authority_failure(
            self.trace_context,
            profile,
            stage=stage,
            source_event_ids=source_event_ids,
            occurred_at=float(self.clock()),
        )

    def admit_replay_candidate_authorization(
        self,
        *,
        proposal_id: str,
        assessment: ReplayBindingAssessment,
        eligibility: CandidateBindingEligibility,
        execution_authorization: ReplayExecutionAuthorization,
    ) -> CandidateAuthorizationAdmissionReceipt:
        """Validate and admit exact-target replay publication evidence."""

        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            raise ValueError("Replay authorization requires an admitted proposal.")
        if proposal.producer_kind != "replay":
            raise ValueError("Replay authorization can only bind a replay proposal.")
        if self.config.target_runtime_contract_hash is None:
            raise ValueError("This runtime has no immutable target action contract.")
        if assessment.verdict != REPLAY_BINDING_VERIFIED_COMPATIBLE:
            raise ValueError("Replay authority requires VERIFIED_COMPATIBLE assessment.")
        if eligibility.requested_use != REPLAY_BINDING_USE_EXECUTION_CANDIDATE:
            raise ValueError("Replay authority requires execution-candidate eligibility.")
        if not eligibility.authoritative_allowed:
            raise ValueError("Replay binding eligibility denies authoritative use.")
        if (
            eligibility.assessment_id != assessment.assessment_id
            or eligibility.assessment_hash != assessment.semantic_content_hash
        ):
            raise ValueError("Replay eligibility does not reference the supplied assessment.")
        if (
            execution_authorization.assessment_id != assessment.assessment_id
            or execution_authorization.assessment_hash
            != assessment.semantic_content_hash
        ):
            raise ValueError(
                "Replay execution authorization does not reference the assessment."
            )
        if not execution_authorization.authorized:
            raise ValueError("Replay execution authorization denies motion authority.")
        if execution_authorization.dispatch_owner_state_id != (
            self.dispatch_owner_token.token_id
        ):
            raise ValueError("Replay execution authorization binds another dispatch owner.")
        if (
            eligibility.target_runtime_contract_hash
            != self.config.target_runtime_contract_hash
            or assessment.target_runtime_contract_hash
            != self.config.target_runtime_contract_hash
        ):
            raise ValueError("Replay authority evidence targets another runtime contract.")
        draft = CandidateAuthorizationDraft(
            proposal_id=proposal.proposal_id,
            proposal_content_hash=proposal.semantic_content_hash,
            producer_kind=proposal.producer_kind,
            evidence_id=execution_authorization.authorization_id,
            evidence_hash=execution_authorization.semantic_content_hash,
            binding_eligibility_id=eligibility.eligibility_id,
            binding_eligibility_hash=eligibility.semantic_content_hash,
            target_runtime_contract_hash=eligibility.target_runtime_contract_hash,
            authoritative_allowed=True,
            reasons=eligibility.reasons,
        )
        self.record_external_facts(
            (
                project_replay_execution_authorization(
                    self.trace_context,
                    execution_authorization,
                    occurred_at=float(self.clock()),
                ),
            )
        )
        receipt = self._candidate_authorizations.admit(draft)
        self._emit(
            (
                project_candidate_authorization_admission(
                    self.trace_context,
                    draft,
                    receipt,
                    occurred_at=float(self.clock()),
                ),
            )
        )
        return receipt

    def capture_execution_gate_snapshot(
        self,
        *,
        now: float | None = None,
    ) -> ExecutionGateSnapshot:
        """Capture the authoritative gate state at one causal control cut."""

        capture = self.coordinator.capture_cut(
            cutoff_at=None if now is None else float(now),
            projectors={"gate": self.execution_gate_store.records_at_or_before},
        )
        return self.execution_gate_store.build_snapshot(
            cut=capture.cut,
            records=capture.projections["gate"],
        )

    def _live_gate_snapshot(self, *, now: float) -> ExecutionGateSnapshot:
        """Compatibility alias for the decision path's existing private seam."""

        return self.capture_execution_gate_snapshot(now=now)

    def _cancel_execution(
        self,
        *,
        reason: str,
        authority_epoch: int,
        selection_group_ids: tuple[str, ...],
        source_event_ids: tuple[str, ...],
        force_domain_scope: bool = False,
    ) -> tuple[
        ExecutionCancellation,
        ExecutionCancellationReceipt,
        DispatchPermitValidationReceipt,
        tuple[ControlTraceFact, ...],
    ]:
        cancelled_groups = tuple(dict.fromkeys(selection_group_ids))
        domain_scope = bool(
            force_domain_scope
            or not self.config.execution_capability.supports_local_group_stop
        )
        request = ExecutionCancellationRequest(
            reason=reason,
            authority_epoch=authority_epoch,
            requested_scope="control_domain" if domain_scope else "selection_groups",
            selection_group_ids=() if domain_scope else cancelled_groups,
            invalidated_decision_ids=(
                () if self._previous_decision_id is None else (self._previous_decision_id,)
            ),
            invalidated_reference_segment_ids=tuple(
                sorted(
                    {
                        item.segment_id
                        for item in (
                            *self._active_segment_by_group.values(),
                            *self._pending_segment_by_group.values(),
                        )
                    }
                )
            ),
            source_event_ids=tuple(dict.fromkeys(source_event_ids)),
        )
        cancellation = self.execution_epochs.cancel(
            request,
            capability=self.config.execution_capability,
        )
        if cancellation.effective_scope == "control_domain":
            self.dispatch_owners.observe_execution_cancellation(cancellation)
            self._active_segment_by_group.clear()
            self._pending_segment_by_group.clear()
        else:
            for group in cancellation.selection_group_ids:
                self._active_segment_by_group.pop(group, None)
                self._pending_segment_by_group.pop(group, None)

        self.cancellation_acknowledgements.register(cancellation)
        facts: list[ControlTraceFact] = [
            project_execution_cancellation(self.trace_context, cancellation)
        ]
        permit = self.dispatch_owners.issue_safety_control_permit(
            operation="hold",
            evidence=cancellation,
            safety_source="execution_cancellation_runtime",
        )
        validation = self.dispatch_owners.authorize(
            permit,
            current_authority_epoch=authority_epoch,
            current_execution_epoch=cancellation.execution_epoch,
        )
        facts.append(
            project_dispatch_permit_validation(
                self.trace_context,
                permit=permit,
                validation=validation,
            )
        )

        acknowledgements: dict[str, bool] = {}
        failure_status: str | None = None
        if validation.accepted:
            try:
                acknowledgements = {
                    str(level): bool(accepted)
                    for level, accepted in self.backend_hold(cancellation).items()
                }
            except Exception as exc:  # noqa: BLE001 - explicit failed cancellation evidence.
                failure_status = f"backend_hold_exception_{type(exc).__name__}"
        else:
            failure_status = f"safety_permit_{validation.status}"

        if failure_status is not None:
            acknowledgements = {
                level: False
                for level in cancellation.required_ack_levels
            }
        for level, accepted in acknowledgements.items():
            acknowledgement = self.cancellation_acknowledgements.acknowledge(
                cancellation.cancellation_id,
                acknowledgement_level=level,
                clock_domain_id=self.config.clock_domain.clock_domain_id,
                source_id=self.backend_name,
                accepted=accepted,
                status=(
                    "backend_hold_acknowledged"
                    if accepted
                    else failure_status or "backend_hold_rejected"
                ),
            )
            facts.append(
                project_execution_cancellation_ack(
                    self.trace_context,
                    acknowledgement,
                )
            )
        receipt = self.cancellation_acknowledgements.receipt(
            cancellation.cancellation_id
        )
        self._backend_hold_active = receipt.complete
        return cancellation, receipt, validation, tuple(facts)

    def run_decision(self, *, now: float | None = None) -> AuthoritativeControlStep:
        timestamp = float(self.clock() if now is None else now)
        snapshot = capture_arbitration_input(
            coordinator=self.coordinator,
            proposal_buffer=self.proposals,
            producer_epochs=self.producer_epochs,
            authority_store=self.authority_store,
            execution_gate_store=self.execution_gate_store,
            scheduler=self.scheduler,
            embodiment_instance_id=self.config.embodiment_instance_id,
            component_catalog=self.config.component_catalog,
            active_selection_groups=self.config.active_selection_groups,
            previous_authoritative_decision_id=self._previous_decision_id,
            previous_selection_by_group=self._previous_selection_by_group,
            candidate_authorization_store=self._candidate_authorizations,
            target_runtime_contract_hash=self.config.target_runtime_contract_hash,
            required_authorization_producer_kinds=(
                self.config.required_authorization_producer_kinds
            ),
            now=timestamp,
        )
        if snapshot is None:
            return AuthoritativeControlStep(status="not_due")

        step_facts: list[ControlTraceFact] = list(
            project_arbitration_input(self.trace_context, snapshot)
        )
        computation, metrics = self.arbitration_runtime.compute(
            self.arbiter,
            snapshot,
            clock=self.clock,
        )
        step_facts.append(
            project_arbitration_computation(self.trace_context, computation)
        )
        attempted_at = float(self.clock())
        publication_attempt, decision = self.arbitration_runtime.publish(
            computation,
            attempted_at=attempted_at,
            live_gate_snapshot=self._live_gate_snapshot(now=attempted_at),
            input_snapshot=snapshot,
        )
        step_facts.extend(
            project_publication(
                self.trace_context,
                publication_attempt,
                decision,
                selected_action_hash=computation.selected_action_hash,
                computation_metrics=metrics,
            )
        )
        if (
            decision is None
            and computation.output_status == "selected"
            and publication_attempt.required_fallback is None
        ):
            raise RuntimeError(
                "A rejected selected result requires an explicit publication fallback."
            )
        directive = build_execution_directive(
            computation,
            active_execution=ActiveExecutionSnapshot(
                control_domain_id=self.config.control_domain_id,
                execution_epoch=self.execution_epoch,
                active_segment_by_group=self._active_segment_by_group,
                backend_hold_active=self._backend_hold_active,
                fault_latched=self.execution_epochs.snapshot().fault_latched,
            ),
            policy=self.config.persistence_policy,
            now=attempted_at,
            supports_local_group_stop=(
                self.config.execution_capability.supports_local_group_stop
            ),
            publication_fallback=(
                None if decision is not None else publication_attempt.required_fallback
            ),
        )
        step_facts.append(
            project_execution_directive(
                self.trace_context,
                directive,
                decision_id=None if decision is None else decision.decision_id,
                publication_attempt_id=publication_attempt.publication_attempt_id,
                execution_epoch=self.execution_epoch,
                occurred_at=attempted_at,
            )
        )
        needs_cancellation = any(
            item.directive == "cancel_group_and_hold"
            for item in directive.directive_by_selection_group.values()
        )
        if decision is None or directive.global_status != "normal":
            cancellation = None
            cancellation_receipt = None
            cancellation_validation = None
            if needs_cancellation:
                cancellation_reason = (
                    "fault" if directive.fault_latched else "hold"
                )
                (
                    cancellation,
                    cancellation_receipt,
                    cancellation_validation,
                    cancellation_facts,
                ) = self._cancel_execution(
                    reason=cancellation_reason,
                    authority_epoch=computation.authority_epoch,
                    selection_group_ids=tuple(
                        group
                        for group, item in directive.directive_by_selection_group.items()
                        if item.directive == "cancel_group_and_hold"
                    ),
                    source_event_ids=(
                        computation.result_id,
                        publication_attempt.publication_attempt_id,
                        directive.directive_id,
                    ),
                )
                step_facts.extend(cancellation_facts)
            if cancellation_receipt is not None:
                status = (
                    "execution_cancelled"
                    if cancellation_receipt.complete
                    else "execution_cancellation_incomplete"
                )
            elif directive.global_status != "normal":
                status = "execution_hold_active"
            elif any(
                item.directive == "keep_backend_hold"
                for item in directive.directive_by_selection_group.values()
            ):
                status = "execution_hold_active"
            else:
                status = "execution_kept_existing"
            self._emit(step_facts)
            return AuthoritativeControlStep(
                status=status,
                snapshot=snapshot,
                computation=computation,
                publication_attempt=publication_attempt,
                decision=decision,
                directive=directive,
                cancellation=cancellation,
                cancellation_receipt=cancellation_receipt,
                cancellation_dispatch_validation=cancellation_validation,
                trace_facts=tuple(step_facts),
            )

        compound_profile = self._compound_profile_for_decision(decision)
        try:
            safety_result = self.shared_safety(
                decision.selected_action,
                decision.decision_id,
            )
        except Exception:
            if compound_profile is None:
                raise
            failure_fact = self._compound_failure_fact(
                compound_profile,
                stage="shared_safety_exception",
                source_event_ids=(
                    computation.result_id,
                    publication_attempt.publication_attempt_id,
                    directive.directive_id,
                ),
            )
            step_facts.append(failure_fact)
            (
                cancellation,
                cancellation_receipt,
                cancellation_validation,
                cancellation_facts,
            ) = self._cancel_execution(
                reason="hold",
                authority_epoch=decision.authority_epoch,
                selection_group_ids=self.config.active_selection_groups,
                source_event_ids=(
                    computation.result_id,
                    publication_attempt.publication_attempt_id,
                    directive.directive_id,
                    failure_fact.fact_id,
                ),
                force_domain_scope=True,
            )
            step_facts.extend(cancellation_facts)
            self._emit(step_facts)
            return AuthoritativeControlStep(
                status="compound_safety_exception_hold",
                snapshot=snapshot,
                computation=computation,
                publication_attempt=publication_attempt,
                decision=decision,
                directive=directive,
                cancellation=cancellation,
                cancellation_receipt=cancellation_receipt,
                cancellation_dispatch_validation=cancellation_validation,
                trace_facts=tuple(step_facts),
            )
        if safety_result.input_action_id != decision.decision_id:
            raise ValueError("Shared safety result does not reference this decision action.")
        safety_id, safety_fact = project_shared_safety_result(
            self.trace_context,
            safety_result,
            decision_id=decision.decision_id,
            directive_id=directive.directive_id,
            input_action_hash=str(computation.selected_action_hash),
            occurred_at=float(self.clock()),
        )
        step_facts.append(safety_fact)
        if safety_result.safe_action is None or safety_result.status in {"hold", "fault"}:
            cancellation = None
            cancellation_receipt = None
            cancellation_validation = None
            should_cancel = not self._backend_hold_active or (
                safety_result.status == "fault"
                and not self.execution_epochs.snapshot().fault_latched
            )
            if should_cancel:
                (
                    cancellation,
                    cancellation_receipt,
                    cancellation_validation,
                    cancellation_facts,
                ) = self._cancel_execution(
                    reason="fault" if safety_result.status == "fault" else "hold",
                    authority_epoch=decision.authority_epoch,
                    selection_group_ids=self.config.active_selection_groups,
                    source_event_ids=(
                        computation.result_id,
                        publication_attempt.publication_attempt_id,
                        directive.directive_id,
                        safety_id,
                    ),
                    force_domain_scope=True,
                )
                step_facts.extend(cancellation_facts)
            if safety_result.status == "fault":
                _gate_event, gate_fact = self._admit_execution_gate(
                    status="fault",
                    reason="shared_canonical_safety_fault",
                    source_event_ids=(safety_id,),
                    produced_at=None,
                    emit=False,
                )
                step_facts.append(gate_fact)
            self._emit(step_facts)
            return AuthoritativeControlStep(
                status=f"safety_{safety_result.status}",
                snapshot=snapshot,
                computation=computation,
                publication_attempt=publication_attempt,
                decision=decision,
                directive=directive,
                safety_result=safety_result,
                cancellation=cancellation,
                cancellation_receipt=cancellation_receipt,
                cancellation_dispatch_validation=cancellation_validation,
                trace_facts=tuple(step_facts),
            )

        if compound_profile is not None:
            expected_effectors = tuple(
                component.effector_name
                for component in self.config.component_catalog.components
            )
            if (
                tuple(safety_result.safe_action.active_effectors)
                != expected_effectors
            ):
                failure_fact = self._compound_failure_fact(
                    compound_profile,
                    stage="safe_action_scope_mismatch",
                    source_event_ids=(
                        computation.result_id,
                        directive.directive_id,
                        safety_id,
                    ),
                )
                step_facts.append(failure_fact)
                (
                    cancellation,
                    cancellation_receipt,
                    cancellation_validation,
                    cancellation_facts,
                ) = self._cancel_execution(
                    reason="hold",
                    authority_epoch=decision.authority_epoch,
                    selection_group_ids=self.config.active_selection_groups,
                    source_event_ids=(
                        computation.result_id,
                        publication_attempt.publication_attempt_id,
                        directive.directive_id,
                        safety_id,
                        failure_fact.fact_id,
                    ),
                    force_domain_scope=True,
                )
                step_facts.extend(cancellation_facts)
                self._emit(step_facts)
                return AuthoritativeControlStep(
                    status="compound_safety_scope_hold",
                    snapshot=snapshot,
                    computation=computation,
                    publication_attempt=publication_attempt,
                    decision=decision,
                    directive=directive,
                    safety_result=safety_result,
                    cancellation=cancellation,
                    cancellation_receipt=cancellation_receipt,
                    cancellation_dispatch_validation=cancellation_validation,
                    trace_facts=tuple(step_facts),
                )
        try:
            command = self.resolver.resolve(safety_result.safe_action)
        except Exception:
            if compound_profile is None:
                raise
            failure_fact = self._compound_failure_fact(
                compound_profile,
                stage="resolution_failure",
                source_event_ids=(
                    computation.result_id,
                    directive.directive_id,
                    safety_id,
                ),
            )
            step_facts.append(failure_fact)
            (
                cancellation,
                cancellation_receipt,
                cancellation_validation,
                cancellation_facts,
            ) = self._cancel_execution(
                reason="hold",
                authority_epoch=decision.authority_epoch,
                selection_group_ids=self.config.active_selection_groups,
                source_event_ids=(
                    computation.result_id,
                    publication_attempt.publication_attempt_id,
                    directive.directive_id,
                    safety_id,
                    failure_fact.fact_id,
                ),
                force_domain_scope=True,
            )
            step_facts.extend(cancellation_facts)
            self._emit(step_facts)
            return AuthoritativeControlStep(
                status="compound_resolution_failure_hold",
                snapshot=snapshot,
                computation=computation,
                publication_attempt=publication_attempt,
                decision=decision,
                directive=directive,
                safety_result=safety_result,
                cancellation=cancellation,
                cancellation_receipt=cancellation_receipt,
                cancellation_dispatch_validation=cancellation_validation,
                trace_facts=tuple(step_facts),
            )
        created_at = float(self.clock())
        command_id, _command_hash, command_fact = project_command(
            self.trace_context,
            command,
            decision_id=decision.decision_id,
            directive_id=directive.directive_id,
            safety_result_id=safety_id,
            scheduled_for=computation.target_effective_at,
            created_at=created_at,
            authority_epoch=decision.authority_epoch,
            execution_epoch=self.execution_epoch,
        )
        step_facts.append(command_fact)
        selected_directives = {
            group: item
            for group, item in directive.directive_by_selection_group.items()
            if item.directive == "publish_selected"
        }
        permit = self.dispatch_owners.issue_motion_permit(
            token=self.dispatch_owner_token,
            authority_epoch=decision.authority_epoch,
            execution_epoch=self.execution_epoch,
            command_lineage_id=command_id,
            directive_lineage_id=directive.directive_id,
        )
        dispatch_context = build_authoritative_command_dispatch_context(
            runtime_session_id=self.config.runtime_session_id,
            control_domain_id=self.config.control_domain_id,
            embodiment_instance_id=self.config.embodiment_instance_id,
            decision_id=decision.decision_id,
            directive_id=directive.directive_id,
            safety_result_id=safety_id,
            command_id=command_id,
            dispatch_permit_id=permit.permit_id,
            owner_token_id=permit.token_id,
            owner_generation=permit.owner_generation,
            target_effective_at=computation.target_effective_at,
            authority_epoch=decision.authority_epoch,
            execution_epoch=self.execution_epoch,
            group_segment_generation_by_group={
                group: item.group_segment_generation
                for group, item in selected_directives.items()
            },
            selected_action_ref_by_group={
                group: str(item.selected_action_ref)
                for group, item in selected_directives.items()
            },
            safe_action=safety_result.safe_action,
        )
        # This is the dispatch linearization point: no trace sink or other
        # mutable control work may occur between authorization and the backend.
        validation = self.dispatch_owners.authorize(
            permit,
            current_authority_epoch=decision.authority_epoch,
            current_execution_epoch=self.execution_epoch,
        )
        receipts: Mapping[str, CommandDispatchReceipt] | None = None
        if validation.accepted:
            dispatch_time = float(self.clock())
            try:
                if self.contextual_dispatcher is None:
                    receipts = dict(self.dispatcher(command))
                else:
                    receipts = dict(
                        self.contextual_dispatcher(command, dispatch_context)
                    )
            except Exception as exc:  # noqa: BLE001 - backend failure becomes typed evidence.
                receipts = {
                    name: CommandDispatchReceipt(
                        effector_name=name,
                        accepted=False,
                        operation="authoritative_backend_dispatch",
                        status="backend_exception",
                        timestamp=dispatch_time,
                        result_text=f"{type(exc).__name__}: {exc}",
                    )
                    for name in command.commands
                }
            missing_receipts = set(command.commands) - set(receipts)
            if missing_receipts:
                receipts = {
                    **dict(receipts),
                    **{
                        name: CommandDispatchReceipt(
                            effector_name=name,
                            accepted=False,
                            operation="authoritative_backend_dispatch",
                            status="missing_backend_receipt",
                            timestamp=dispatch_time,
                            result_text="Backend returned no receipt for this command.",
                        )
                        for name in missing_receipts
                    },
                }
        dispatch_owner_fact = project_dispatch_owner_validation(
            self.trace_context,
            token=self.dispatch_owner_token,
            permit=permit,
            validation=validation,
        )
        step_facts.append(dispatch_owner_fact)
        dispatch_receipt_id: str | None = None
        if receipts is not None:
            dispatch_receipt_id, receipt_fact = project_dispatch_receipts(
                self.trace_context,
                receipts,
                command_id=command_id,
                dispatch_event_id=dispatch_owner_fact.fact_id,
                backend_name=self.backend_name,
                dispatched_at=float(self.clock()),
            )
            step_facts.append(receipt_fact)

        backend_accepted = bool(receipts) and set(command.commands).issubset(receipts) and all(
            item.accepted for item in receipts.values()
        )
        cancellation = None
        cancellation_receipt = None
        cancellation_validation = None
        if compound_profile is not None and not (
            validation.accepted and backend_accepted
        ):
            failure_sources = tuple(
                value
                for value in (
                    computation.result_id,
                    publication_attempt.publication_attempt_id,
                    directive.directive_id,
                    safety_id,
                    command_id,
                    dispatch_receipt_id,
                )
                if value is not None
            )
            failure_fact = self._compound_failure_fact(
                compound_profile,
                stage="downstream_admission_failure",
                source_event_ids=failure_sources,
            )
            step_facts.append(failure_fact)
            (
                cancellation,
                cancellation_receipt,
                cancellation_validation,
                cancellation_facts,
            ) = self._cancel_execution(
                reason="hold",
                authority_epoch=decision.authority_epoch,
                selection_group_ids=self.config.active_selection_groups,
                source_event_ids=(*failure_sources, failure_fact.fact_id),
                force_domain_scope=True,
            )
            step_facts.extend(cancellation_facts)
        if validation.accepted and backend_accepted:
            for group, group_directive in directive.directive_by_selection_group.items():
                if group_directive.directive != "publish_selected":
                    continue
                assert group_directive.selected_action_ref is not None
                assert group_directive.segment_expiry is not None
                segment = ActiveExecutionSegment(
                    selection_group_id=group,
                    segment_id=f"segment:{decision.decision_id}:{group}",
                    selected_action_ref=group_directive.selected_action_ref,
                    expires_at=group_directive.segment_expiry,
                    group_segment_generation=group_directive.group_segment_generation,
                )
                if self.config.defer_selected_segment_activation:
                    self._pending_segment_by_group[group] = segment
                else:
                    self._active_segment_by_group[group] = segment
            self._previous_decision_id = decision.decision_id
            self._previous_selection_by_group = dict(decision.selected_by_group)
            self._backend_hold_active = False

        self._emit(step_facts)
        return AuthoritativeControlStep(
            status=(
                "dispatched"
                if validation.accepted and backend_accepted
                else "compound_downstream_failure_hold"
                if cancellation is not None
                else "dispatch_failed"
                if validation.accepted
                else "dispatch_rejected"
            ),
            snapshot=snapshot,
            computation=computation,
            publication_attempt=publication_attempt,
            decision=decision,
            directive=directive,
            safety_result=safety_result,
            command=command,
            dispatch_context=dispatch_context,
            dispatch_validation=validation,
            dispatch_receipts=receipts,
            command_id=command_id,
            dispatch_receipt_id=dispatch_receipt_id,
            cancellation=cancellation,
            cancellation_receipt=cancellation_receipt,
            cancellation_dispatch_validation=cancellation_validation,
            trace_facts=tuple(step_facts),
        )

    def record_measurement(
        self,
        step: AuthoritativeControlStep,
        *,
        measurement: Mapping[str, object],
        measured_at: float | None = None,
    ) -> ControlTraceFact:
        if (
            step.decision is None
            or step.command_id is None
            or step.dispatch_receipt_id is None
        ):
            raise ValueError("Measurement requires a dispatched authoritative step.")
        fact = project_measurement_alignment(
            self.trace_context,
            decision_id=step.decision.decision_id,
            command_id=step.command_id,
            dispatch_receipt_id=step.dispatch_receipt_id,
            measured_at=float(self.clock() if measured_at is None else measured_at),
            measurement=measurement,
        )
        self._emit((fact,))
        return fact

    def close_trace(self, *, status: str = "complete", reason: str = "normal") -> ControlTraceFact:
        if self._terminal:
            raise RuntimeError("Authoritative control trace is already terminal.")
        occurred_at = float(self.clock())
        fact = self.trace_context.fact(
            fact_id=f"runtime-trace-terminal:{self.config.runtime_session_id}",
            event_type="runtime_trace_terminal",
            occurred_at=occurred_at,
            payload={
                "terminal_status": str(status),
                "reason": str(reason),
                "occurred_at": occurred_at,
            },
        )
        self._terminal = True
        self._emit((fact,))
        return fact


__all__ = [
    "AuthoritativeControlConfig",
    "AuthoritativeControlRuntime",
    "AuthoritativeControlStep",
    "ExecutionGateTransitionOutcome",
]
