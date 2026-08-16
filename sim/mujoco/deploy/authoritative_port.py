"""Narrow application port for one lower-owned MuJoCo authority runtime.

The port exposes immutable profile evidence, bounded snapshots, and explicit
admission operations.  It deliberately does not expose the underlying stores,
registries, dispatcher, scheduler, proposal buffer, or mutable runtime core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.action import (
    ActionProposalDraft,
    AuthoritativeControlRuntime,
    AuthoritativeReceiverStateSnapshot,
    AuthorityEvent,
    RobotAction,
    SharedCanonicalSafetyResult,
)
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext
from robot_motion.action.proposal import ProposalAdmissionReceipt
from robot_motion.runtime.cancellation import ExecutionEpochSnapshot
from robot_motion.runtime.control_admission import ControlCut
from robot_motion.runtime.dispatch_owner import DispatchOwnerSnapshot
from robot_motion.runtime.execution_gate import ExecutionGateSnapshot
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochActivation


@dataclass(frozen=True)
class MujocoAuthoritativeProfileSnapshot:
    """Immutable application-visible identity of the authoritative profile."""

    initial_control_mode: str
    selection_group_ids: tuple[str, ...]
    authority_policy_hash: str
    decision_schedule_hash: str
    takeover_allowed_producers: tuple[str, ...]
    decision_period_sec: float
    target_lead_time_sec: float


class MujocoAuthoritativeTargetPort:
    """Capability-limited facade over the target-owned authoritative core.

    The wrapped core remains the sole owner of authority, epochs, safety,
    proposal admission, dispatch ownership, and trace persistence.  Callers
    receive values or invoke explicit operations; they cannot obtain the
    mutable owner objects themselves.
    """

    __slots__ = ("_core", "_profile")

    def __init__(self, core: AuthoritativeControlRuntime) -> None:
        if not isinstance(core, AuthoritativeControlRuntime):
            raise TypeError("core must be an AuthoritativeControlRuntime.")
        self._core = core
        config = core.config
        self._profile = MujocoAuthoritativeProfileSnapshot(
            initial_control_mode=config.initial_control_mode,
            selection_group_ids=tuple(config.active_selection_groups),
            authority_policy_hash=config.authority_policy.semantic_content_hash,
            decision_schedule_hash=config.decision_schedule.semantic_content_hash,
            takeover_allowed_producers=tuple(
                config.authority_policy.takeover_allowed_producers
            ),
            decision_period_sec=float(config.decision_schedule.period_sec),
            target_lead_time_sec=float(
                config.decision_schedule.target_lead_time_sec
            ),
        )

    @property
    def profile(self) -> MujocoAuthoritativeProfileSnapshot:
        return self._profile

    @property
    def trace_context(self) -> AuthoritativeTraceContext:
        return self._core.trace_context

    @property
    def clock_domain(self) -> ClockDomainDescriptor:
        return self._core.coordinator.clock_domain

    def capture_receiver_state(self) -> AuthoritativeReceiverStateSnapshot:
        return self._core.capture_receiver_state()

    def capture_execution_gate_snapshot(
        self,
        *,
        now: float | None = None,
    ) -> ExecutionGateSnapshot:
        return self._core.capture_execution_gate_snapshot(now=now)

    def capture_execution_epoch_snapshot(self) -> ExecutionEpochSnapshot:
        return self._core.execution_epochs.snapshot()

    def capture_dispatch_owner_snapshot(self) -> DispatchOwnerSnapshot:
        return self._core.dispatch_owners.snapshot()

    def capture_control_cut(self, *, cutoff_at: float) -> ControlCut:
        return self._core.coordinator.capture_cut(cutoff_at=float(cutoff_at)).cut

    def current_producer_epoch(self, producer_id: str) -> int | None:
        return self._core.producer_epochs.current_epoch(producer_id)

    def activate_producer(
        self,
        *,
        producer_id: str,
        producer_epoch: int,
        produced_at: float | None = None,
        produced_clock_domain_id: str | None = None,
        reason: str = "startup",
    ) -> ProducerEpochActivation:
        return self._core.activate_producer(
            producer_id=producer_id,
            producer_epoch=producer_epoch,
            produced_at=produced_at,
            produced_clock_domain_id=produced_clock_domain_id,
            reason=reason,
        )

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
        return self._core.admit_authority_event(
            event_type=event_type,
            reason=reason,
            selection_group_id=selection_group_id,
            owner_producer_id=owner_producer_id,
            control_mode=control_mode,
            expires_at=expires_at,
            expires_clock_domain_id=expires_clock_domain_id,
            produced_at=produced_at,
        )

    def admit_proposal(
        self,
        draft: ActionProposalDraft,
        *,
        source_event_id: str,
    ) -> ProposalAdmissionReceipt:
        return self._core.admit_proposal(
            draft,
            source_event_id=source_event_id,
        )

    def record_external_facts(
        self,
        facts: Sequence[ControlTraceFact],
    ) -> None:
        self._core.record_external_facts(tuple(facts))

    def evaluate_canonical_safety(
        self,
        action: RobotAction,
        *,
        input_action_id: str,
    ) -> SharedCanonicalSafetyResult:
        return self._core.shared_safety(action, input_action_id)


__all__ = [
    "MujocoAuthoritativeProfileSnapshot",
    "MujocoAuthoritativeTargetPort",
]
