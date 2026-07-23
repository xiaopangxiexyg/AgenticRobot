from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from robot_motion.position_control.capability import ExecutionCapabilityProfile
from robot_motion.runtime.cancellation import (
    CancellationAcknowledgementTracker,
    ExecutionCancellation,
    ExecutionCancellationReceipt,
    ExecutionCancellationRequest,
    ExecutionEpochRegistry,
)
from robot_motion.runtime.dispatch_owner import (
    AdmittedSafetyControlEvidence,
    DispatchOwnerRegistry,
    LifecycleControlAuthorization,
)


class ManualClock:
    def __init__(self, value: float = 20.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def capability() -> ExecutionCapabilityProfile:
    return ExecutionCapabilityProfile(
        profile_id="dispatch-owner-test-v1",
        validated_max_selected_keyframe_hz=60.0,
        configured_selected_keyframe_hz=30.0,
        validated_max_coupled_ik_hz=60.0,
        configured_coupled_ik_hz=60.0,
        reference_resample_hz=75.0,
        command_hz=150.0,
        backend_servo_hz=150.0,
        validated_action_spaces=("bimanual_ee_pose_v1",),
        validated_selection_groups=("arms",),
        supports_local_group_stop=False,
        acceptance_evidence_id="synthetic-m1d-only",
        decision_runtime_budget_sec=0.005,
        ik_runtime_budget_sec=0.012,
        end_to_end_control_budget_sec=0.02,
        cancellation_ack_deadline_sec=0.1,
        required_cancellation_ack_levels=("software_queue", "sdk_request"),
    )


def owner_registry(clock: ManualClock | None = None) -> DispatchOwnerRegistry:
    return DispatchOwnerRegistry(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        initial_authority_epoch=1,
        initial_execution_epoch=3,
        clock=ManualClock() if clock is None else clock,
    )


def execution_registry(clock: ManualClock | None = None) -> ExecutionEpochRegistry:
    return ExecutionEpochRegistry(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        clock_domain_id="control-clock",
        initial_execution_epoch=3,
        clock=ManualClock() if clock is None else clock,
    )


def cancel(
    registry: ExecutionEpochRegistry,
    *,
    reason: str,
    authority_epoch: int,
) -> ExecutionCancellation:
    return registry.cancel(
        ExecutionCancellationRequest(
            reason=reason,
            authority_epoch=authority_epoch,
            requested_scope="control_domain",
            invalidated_queued_command_ids=("queued-command",),
            source_event_ids=(f"{reason}-request",),
        ),
        capability=capability(),
    )


def complete_receipt(
    event: ExecutionCancellation,
    *,
    clock: ManualClock,
) -> ExecutionCancellationReceipt:
    tracker = CancellationAcknowledgementTracker(clock=clock)
    tracker.register(event)
    tracker.acknowledge(
        event.cancellation_id,
        acknowledgement_level="software_queue",
        clock_domain_id="control-clock",
        source_id="fake-command-worker",
    )
    tracker.acknowledge(
        event.cancellation_id,
        acknowledgement_level="sdk_request",
        clock_domain_id="control-clock",
        source_id="fake-sdk-hold",
    )
    return tracker.receipt(event.cancellation_id)


def motion_permit(registry: DispatchOwnerRegistry, token, *, suffix: str = "0"):
    return registry.issue_motion_permit(
        token=token,
        authority_epoch=registry.snapshot().current_authority_epoch,
        execution_epoch=registry.snapshot().current_execution_epoch,
        command_lineage_id=f"command-{suffix}",
        directive_lineage_id=f"directive-{suffix}",
    )


def test_initial_owner_is_unique_and_sources_cannot_be_dispatch_owners() -> None:
    value = owner_registry()
    assert value.snapshot().state == "unowned"
    token = value.grant_initial_owner(
        owner_runtime_id="legacy-runtime",
        owner_kind="legacy_control_path",
    )

    assert value.snapshot().current_token == token
    with pytest.raises(RuntimeError, match="already"):
        value.grant_initial_owner(
            owner_runtime_id="shared-runtime",
            owner_kind="shared_arbitration_runtime",
        )

    for source_kind in ("human", "policy", "replay"):
        other = owner_registry()
        with pytest.raises(ValueError, match="legacy or shared"):
            other.grant_initial_owner(
                owner_runtime_id=f"{source_kind}-source",
                owner_kind=source_kind,
            )


def test_concurrent_initial_grant_never_creates_two_tokens() -> None:
    value = owner_registry()

    def grant(index: int):
        try:
            return value.grant_initial_owner(
                owner_runtime_id=f"legacy-{index}",
                owner_kind="legacy_control_path",
            )
        except RuntimeError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        tokens = tuple(pool.map(grant, range(32)))

    granted = tuple(token for token in tokens if token is not None)
    assert len(granted) == 1
    assert value.snapshot().current_token == granted[0]


def test_motion_permit_is_one_shot_and_rejection_is_typed() -> None:
    value = owner_registry()
    token = value.grant_initial_owner(
        owner_runtime_id="legacy-runtime",
        owner_kind="legacy_control_path",
    )
    permit = motion_permit(value, token)

    accepted = value.authorize(
        permit,
        current_authority_epoch=1,
        current_execution_epoch=3,
    )
    replayed = value.authorize(
        permit,
        current_authority_epoch=1,
        current_execution_epoch=3,
    )

    assert accepted.accepted
    assert accepted.status == "accepted"
    assert not replayed.accepted
    assert replayed.status == "permit_already_consumed"


def test_concurrent_authorization_consumes_a_motion_permit_once() -> None:
    value = owner_registry()
    token = value.grant_initial_owner(
        owner_runtime_id="legacy-runtime",
        owner_kind="legacy_control_path",
    )
    permit = motion_permit(value, token)

    def authorize_once(_index: int):
        return value.authorize(
            permit,
            current_authority_epoch=1,
            current_execution_epoch=3,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = tuple(pool.map(authorize_once, range(32)))

    assert sum(receipt.accepted for receipt in receipts) == 1
    assert {receipt.status for receipt in receipts} == {
        "accepted",
        "permit_already_consumed",
    }


def test_wrong_domain_and_unissued_permits_are_rejected_before_backend() -> None:
    value = owner_registry()
    token = value.grant_initial_owner(
        owner_runtime_id="legacy-runtime",
        owner_kind="legacy_control_path",
    )
    permit = motion_permit(value, token)

    wrong_domain = replace(permit, control_domain_id="robot-1")
    unissued = replace(permit, permit_id="forged-permit")
    assert value.authorize(
        wrong_domain,
        current_authority_epoch=1,
        current_execution_epoch=3,
    ).status == "wrong_control_domain"
    assert value.authorize(
        unissued,
        current_authority_epoch=1,
        current_execution_epoch=3,
    ).status == "permit_not_issued_by_registry"


def test_live_authority_and_execution_epoch_changes_invalidate_old_motion() -> None:
    clock = ManualClock()
    value = owner_registry(clock)
    epochs = execution_registry(clock)
    token = value.grant_initial_owner(
        owner_runtime_id="legacy-runtime",
        owner_kind="legacy_control_path",
    )
    old_authority = motion_permit(value, token, suffix="authority")
    value.observe_authority_epoch(authority_epoch=2, evidence_event_id="authority-event-2")
    assert value.authorize(
        old_authority,
        current_authority_epoch=2,
        current_execution_epoch=3,
    ).status == "authority_epoch_mismatch"

    current_authority = motion_permit(value, token, suffix="execution")
    cancellation = cancel(epochs, reason="hold", authority_epoch=2)
    value.observe_execution_cancellation(cancellation)
    assert value.authorize(
        current_authority,
        current_authority_epoch=2,
        current_execution_epoch=4,
    ).status == "execution_epoch_mismatch"


def test_recovery_observation_allows_only_fresh_epoch_motion() -> None:
    clock = ManualClock()
    value = owner_registry(clock)
    epochs = execution_registry(clock)
    token = value.grant_initial_owner(
        owner_runtime_id="legacy-runtime",
        owner_kind="legacy_control_path",
    )
    cancellation = cancel(epochs, reason="fault", authority_epoch=1)
    value.observe_execution_cancellation(cancellation)
    stale = value.issue_motion_permit(
        token=token,
        authority_epoch=1,
        execution_epoch=4,
        command_lineage_id="fault-epoch-command",
        directive_lineage_id="fault-epoch-directive",
    )
    recovery = epochs.recover(
        authority_epoch=1,
        source_event_ids=("explicit-rearm",),
    )
    value.observe_execution_recovery(recovery)

    assert value.authorize(
        stale,
        current_authority_epoch=1,
        current_execution_epoch=5,
    ).status == "execution_epoch_mismatch"
    fresh = value.issue_motion_permit(
        token=token,
        authority_epoch=1,
        execution_epoch=5,
        command_lineage_id="recovered-command",
        directive_lineage_id="recovered-directive",
    )
    assert value.authorize(
        fresh,
        current_authority_epoch=1,
        current_execution_epoch=5,
    ).accepted


def test_owner_handoff_has_an_ownerless_hold_interval_and_complete_ack_gate() -> None:
    clock = ManualClock()
    value = owner_registry(clock)
    epochs = execution_registry(clock)
    legacy = value.grant_initial_owner(
        owner_runtime_id="legacy-runtime",
        owner_kind="legacy_control_path",
    )
    legacy_permit = motion_permit(value, legacy)
    cancellation = cancel(epochs, reason="authority_handoff", authority_epoch=1)

    transition = value.begin_transition(
        current_token=legacy,
        transition_kind="authority_handoff",
        cancellation=cancellation,
        gate_status="hold",
        gate_epoch=4,
    )
    during = value.snapshot()
    assert during.state == "transitioning_hold"
    assert during.current_token is None
    assert transition.revoked_token.revoked_at is not None
    assert value.authorize(
        legacy_permit,
        current_authority_epoch=1,
        current_execution_epoch=4,
    ).status == "owner_token_revoked"

    incomplete = ExecutionCancellationReceipt(
        cancellation_id=cancellation.cancellation_id,
        required_ack_levels=("software_queue", "sdk_request"),
        acknowledged_levels=("software_queue",),
        pending_levels=("sdk_request",),
        rejected_levels=(),
        acknowledgement_deadline_at=20.1,
        observed_at=20.0,
        complete=False,
        timed_out=False,
        requires_fault_escalation=False,
    )
    with pytest.raises(RuntimeError, match="complete cancellation"):
        value.complete_transition(
            transition_id=transition.transition_id,
            cancellation_receipt=incomplete,
            owner_runtime_id="shared-runtime",
            owner_kind="shared_arbitration_runtime",
            new_authority_epoch=2,
            gate_status="hold",
        )

    complete = complete_receipt(cancellation, clock=clock)
    with pytest.raises(ValueError, match="legacy or shared"):
        value.complete_transition(
            transition_id=transition.transition_id,
            cancellation_receipt=complete,
            owner_runtime_id="policy-node",
            owner_kind="policy",
            new_authority_epoch=2,
            gate_status="hold",
        )
    assert value.snapshot().state == "transitioning_hold"

    shared = value.complete_transition(
        transition_id=transition.transition_id,
        cancellation_receipt=complete,
        owner_runtime_id="shared-runtime",
        owner_kind="shared_arbitration_runtime",
        new_authority_epoch=2,
        gate_status="transitioning",
    )
    assert shared.owner_generation == 2
    assert shared.execution_epoch_at_acquire == 4
    assert value.snapshot().current_token == shared
    backend_calls: list[str] = []
    shared_permit = motion_permit(value, shared, suffix="first-shared")
    dispatch_receipt = value.authorize(
        shared_permit,
        current_authority_epoch=2,
        current_execution_epoch=4,
    )
    if dispatch_receipt.accepted:
        backend_calls.append(shared_permit.command_lineage_id)
    assert backend_calls == ["command-first-shared"]


def test_safety_hold_remains_authorized_after_motion_owner_is_revoked() -> None:
    clock = ManualClock()
    value = owner_registry(clock)
    epochs = execution_registry(clock)
    legacy = value.grant_initial_owner(
        owner_runtime_id="legacy-runtime",
        owner_kind="legacy_control_path",
    )
    cancellation = cancel(epochs, reason="authority_handoff", authority_epoch=1)
    value.begin_transition(
        current_token=legacy,
        transition_kind="authority_handoff",
        cancellation=cancellation,
        gate_status="hold",
        gate_epoch=5,
    )

    safety_permit = value.issue_safety_control_permit(
        operation="hold",
        evidence=cancellation,
        safety_source="execution_cancellation_runtime",
    )
    receipt = value.authorize(
        safety_permit,
        current_authority_epoch=1,
        current_execution_epoch=4,
    )
    assert receipt.accepted
    assert value.snapshot().current_token is None


def test_safety_control_is_not_a_generic_bypass() -> None:
    value = owner_registry()
    evidence = AdmittedSafetyControlEvidence(
        event_id="safety-event-1",
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        execution_epoch=3,
        evidence_kind="safety_event",
        safety_source="shared_safety_guard",
        control_admission_sequence=12,
    )
    with pytest.raises(ValueError, match="Unsupported safety-control operation"):
        value.issue_safety_control_permit(
            operation="move_pose",
            evidence=evidence,
            safety_source="shared_safety_guard",
        )
    with pytest.raises(ValueError, match="source"):
        value.issue_safety_control_permit(
            operation="hold",
            evidence=evidence,
            safety_source="policy",
        )
    valid = value.issue_safety_control_permit(
        operation="collision_stop",
        evidence=evidence,
        safety_source="shared_safety_guard",
    )
    assert value.authorize(
        valid,
        current_authority_epoch=1,
        current_execution_epoch=3,
    ).accepted


@pytest.mark.parametrize("requester_kind", ("human", "policy", "replay"))
def test_sources_cannot_construct_lifecycle_control(requester_kind: str) -> None:
    value = owner_registry()
    authorization = LifecycleControlAuthorization(
        authorization_event_id="lifecycle-auth-1",
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        lifecycle_state_machine_id="startup-state-machine",
        requester_kind=requester_kind,
        operation="home",
        target_execution_epoch=3,
        operator_or_safety_policy_id="operator-confirmation-1",
    )
    with pytest.raises(ValueError, match="lifecycle state machine"):
        value.issue_lifecycle_control_permit(authorization)


def test_lifecycle_control_is_owner_independent_but_epoch_bound() -> None:
    value = owner_registry()
    authorization = LifecycleControlAuthorization(
        authorization_event_id="lifecycle-auth-1",
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        lifecycle_state_machine_id="startup-state-machine",
        requester_kind="lifecycle_state_machine",
        operation="home",
        target_execution_epoch=3,
        operator_or_safety_policy_id="operator-confirmation-1",
    )
    permit = value.issue_lifecycle_control_permit(authorization)
    assert value.snapshot().state == "unowned"
    assert value.authorize(
        permit,
        current_authority_epoch=1,
        current_execution_epoch=3,
    ).accepted


def test_owner_rollback_requires_new_cancellation_before_legacy_regrant() -> None:
    clock = ManualClock()
    value = owner_registry(clock)
    epochs = execution_registry(clock)
    legacy = value.grant_initial_owner(
        owner_runtime_id="legacy-runtime",
        owner_kind="legacy_control_path",
    )
    handoff = cancel(epochs, reason="authority_handoff", authority_epoch=1)
    handoff_transition = value.begin_transition(
        current_token=legacy,
        transition_kind="authority_handoff",
        cancellation=handoff,
        gate_status="hold",
        gate_epoch=1,
    )
    shared = value.complete_transition(
        transition_id=handoff_transition.transition_id,
        cancellation_receipt=complete_receipt(handoff, clock=clock),
        owner_runtime_id="shared-runtime",
        owner_kind="shared_arbitration_runtime",
        new_authority_epoch=2,
        gate_status="hold",
    )
    shared_permit = motion_permit(value, shared, suffix="shared")

    rollback = cancel(epochs, reason="owner_rollback", authority_epoch=2)
    rollback_transition = value.begin_transition(
        current_token=shared,
        transition_kind="owner_rollback",
        cancellation=rollback,
        gate_status="transitioning",
        gate_epoch=2,
    )
    assert value.authorize(
        shared_permit,
        current_authority_epoch=2,
        current_execution_epoch=5,
    ).status == "owner_token_revoked"
    restored = value.complete_transition(
        transition_id=rollback_transition.transition_id,
        cancellation_receipt=complete_receipt(rollback, clock=clock),
        owner_runtime_id="legacy-runtime",
        owner_kind="legacy_control_path",
        new_authority_epoch=3,
        gate_status="hold",
    )

    assert restored.owner_kind == "legacy_control_path"
    assert restored.owner_generation == 3
    assert restored.execution_epoch_at_acquire == 5


def test_stale_owner_generation_is_rejected_after_switch() -> None:
    clock = ManualClock()
    value = owner_registry(clock)
    epochs = execution_registry(clock)
    legacy = value.grant_initial_owner(
        owner_runtime_id="legacy-runtime",
        owner_kind="legacy_control_path",
    )
    stale = motion_permit(value, legacy)
    handoff = cancel(epochs, reason="authority_handoff", authority_epoch=1)
    transition = value.begin_transition(
        current_token=legacy,
        transition_kind="authority_handoff",
        cancellation=handoff,
        gate_status="hold",
        gate_epoch=1,
    )
    value.complete_transition(
        transition_id=transition.transition_id,
        cancellation_receipt=complete_receipt(handoff, clock=clock),
        owner_runtime_id="shared-runtime",
        owner_kind="shared_arbitration_runtime",
        new_authority_epoch=2,
        gate_status="hold",
    )
    assert value.authorize(
        stale,
        current_authority_epoch=2,
        current_execution_epoch=4,
    ).status == "owner_generation_mismatch"


def test_transition_rejects_ready_gate_and_mismatched_cancellation_reason() -> None:
    clock = ManualClock()
    value = owner_registry(clock)
    epochs = execution_registry(clock)
    legacy = value.grant_initial_owner(
        owner_runtime_id="legacy-runtime",
        owner_kind="legacy_control_path",
    )
    handoff = cancel(epochs, reason="authority_handoff", authority_epoch=1)
    with pytest.raises(ValueError, match="HOLD"):
        value.begin_transition(
            current_token=legacy,
            transition_kind="authority_handoff",
            cancellation=handoff,
            gate_status="ready",
            gate_epoch=1,
        )
    with pytest.raises(ValueError, match="reason"):
        value.begin_transition(
            current_token=legacy,
            transition_kind="owner_rollback",
            cancellation=handoff,
            gate_status="hold",
            gate_epoch=1,
        )
