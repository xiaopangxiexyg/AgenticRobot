from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from control_composition.activation import build_composition_activation_scope
from control_composition.composition import (
    CompositionAuthorityGateSpec,
    CompositionInputDraft,
    CompositionInputStore,
    CompositionSpec,
    DeterministicComposer,
    build_composition_input_snapshot,
    build_robot_action_v1_composition_input_draft,
    decide_composition_authority_disposition,
    project_composition_inputs,
)
from control_composition.eligibility import (
    CompositionSourceEligibilityDraft,
    CompositionSourceEligibilitySpec,
    CompositionSourceEligibilityStore,
    build_composition_source_eligibility_snapshot,
    project_composition_source_eligibility,
)
from control_composition.events import CompositionDomainEvent
from control_composition.retention import TimeWindowRetentionSpec
from control_composition.runtime import ControlCompositionRuntime
from dataset_interface.composition_control_trace import (
    project_composition_domain_event,
)
from robot_motion.action import (
    BIMANUAL_QPOS_V1,
    ActionComponentCatalog,
    ActionComponentSpec,
    RobotAction,
    build_robot_action_v1_runtime_schema,
)
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry


pytestmark = [pytest.mark.unit, pytest.mark.contract]


class Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def retention_spec() -> TimeWindowRetentionSpec:
    return TimeWindowRetentionSpec(
        spec_id="source-eligibility-test-window-v1",
        history_window_sec=0.0,
        max_future_lead_sec=2.0,
        max_effective_duration_sec=4.0,
    )


def composition_spec() -> CompositionSpec:
    return CompositionSpec(
        spec_id="eligible-base-v1",
        algebra="component_union",
        input_roles=("base",),
        required_roles=("base",),
    )


def eligibility_spec(
    spec: CompositionSpec | None = None,
) -> CompositionSourceEligibilitySpec:
    bound_spec = composition_spec() if spec is None else spec
    return CompositionSourceEligibilitySpec(
        spec_id="eligible-base-source-v1",
        composition_spec_hash=bound_spec.semantic_content_hash,
        source_slot_by_role={"base": "human-primary"},
        producer_id_by_source_slot={"human-primary": "human-producer"},
    )


def qpos_action(value: float = 0.1) -> RobotAction:
    return RobotAction(
        action_space=BIMANUAL_QPOS_V1,
        arm_qpos_by_side={
            "left": np.full(6, value, dtype=np.float32),
        },
        source="source-eligibility-fixture",
    )


def component_catalog() -> ActionComponentCatalog:
    return ActionComponentCatalog(
        catalog_id="source-eligibility-left-arm-v1",
        action_space=BIMANUAL_QPOS_V1,
        components=(
            ActionComponentSpec("left_arm", "left_arm", "left_arm"),
        ),
    )


def action_schema():
    return build_robot_action_v1_runtime_schema(
        component_catalog=component_catalog(),
    )


def input_draft() -> CompositionInputDraft:
    return build_robot_action_v1_composition_input_draft(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        producer_id="human-producer",
        producer_epoch=1,
        sequence=0,
        input_role="base",
        action=qpos_action(),
        action_schema=action_schema(),
        produced_at=9.9,
        produced_clock_domain_id="human.monotonic",
        effective_from=10.0,
        effective_until=12.0,
        effective_clock_domain_id="control.monotonic",
    )


def source_state(
    *,
    producer_epoch: int = 1,
    status: str = "available",
    status_epoch: int = 1,
    sequence: int = 0,
    effective_from: float = 10.0,
    lease_expires_at: float = 11.0,
    rearm_required: bool = False,
    fault_code: str | None = None,
    rearm_receipt_id: str | None = None,
    rearm_receipt_hash: str | None = None,
) -> CompositionSourceEligibilityDraft:
    return CompositionSourceEligibilityDraft(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        source_slot_id="human-primary",
        producer_id="human-producer",
        producer_epoch=producer_epoch,
        status_epoch=status_epoch,
        sequence=sequence,
        status=status,
        observed_at=9.99,
        observed_clock_domain_id="human.monotonic",
        effective_from=effective_from,
        lease_expires_at=lease_expires_at,
        effective_clock_domain_id="control.monotonic",
        rearm_required=rearm_required,
        fault_code=fault_code,
        rearm_receipt_id=rearm_receipt_id,
        rearm_receipt_hash=rearm_receipt_hash,
    )


def stores():
    clock = Clock()
    coordinator = ControlAdmissionCoordinator(
        "robot-0",
        clock_domain=ClockDomainDescriptor(
            clock_domain_id="control.monotonic",
            runtime_session_id="session-0",
            boot_id="boot-0",
            reset_epoch=0,
        ),
        clock=clock,
    )
    epochs = ProducerEpochRegistry(
        runtime_session_id="session-0",
        coordinator=coordinator,
    )
    epochs.activate(
        producer_id="human-producer",
        producer_epoch=1,
        produced_at=9.9,
        produced_clock_domain_id="human.monotonic",
        reason="startup",
    )
    inputs = CompositionInputStore(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        retention_spec=retention_spec(),
        action_schema=action_schema(),
    )
    states = CompositionSourceEligibilityStore(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        retention_spec=retention_spec(),
    )
    assert inputs.admit(input_draft()).accepted
    return clock, coordinator, epochs, inputs, states


def capture(
    coordinator: ControlAdmissionCoordinator,
    epochs: ProducerEpochRegistry,
    inputs: CompositionInputStore,
    states: CompositionSourceEligibilityStore,
    *,
    target: float,
):
    spec = composition_spec()
    state_spec = eligibility_spec(spec)
    cut = coordinator.capture_cut(cutoff_at=target).cut
    epoch_snapshot = epochs.snapshot(cut)
    source_snapshot = build_composition_source_eligibility_snapshot(
        project_composition_source_eligibility(states, cut),
        producer_epochs=epoch_snapshot,
        spec=state_spec,
        composition_spec=spec,
        target_effective_at=target,
    )
    input_snapshot = build_composition_input_snapshot(
        project_composition_inputs(inputs, cut),
        producer_epochs=epoch_snapshot,
        spec=spec,
        target_effective_at=target,
        source_eligibility_snapshot=source_snapshot,
    )
    return spec, source_snapshot, input_snapshot


def test_unscoped_snapshot_keeps_legacy_hash_and_selection() -> None:
    _, coordinator, epochs, inputs, states = stores()
    assert states.admit(source_state(status="paused")).accepted
    cut = coordinator.capture_cut(cutoff_at=10.1).cut
    projection = project_composition_inputs(inputs, cut)
    epoch_snapshot = epochs.snapshot(cut)
    first = build_composition_input_snapshot(
        projection,
        producer_epochs=epoch_snapshot,
        spec=composition_spec(),
        target_effective_at=10.1,
    )
    second = build_composition_input_snapshot(
        projection,
        producer_epochs=epoch_snapshot,
        spec=composition_spec(),
        target_effective_at=10.1,
        source_eligibility_snapshot=None,
    )
    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.composition_input_snapshot_id == second.composition_input_snapshot_id
    assert tuple(first.selected_input_by_role) == ("base",)
    assert first.source_eligibility_snapshot_id is None
    assert not first.source_ineligibility_reason_by_role


def test_activation_and_source_eligibility_share_one_v3_snapshot() -> None:
    _, coordinator, epochs, inputs, states = stores()
    spec = composition_spec()
    activation = build_composition_activation_scope(
        operation_spec_id="operation:source-eligibility",
        operation_spec_hash="operation-hash",
        session_epoch=1,
        activated_at=10.0,
        clock_domain_id="control.monotonic",
        input_barrier_at=10.0,
        required_roles=spec.required_roles,
    )
    bound_input = inputs.admit(
        replace(
            input_draft(),
            composition_activation_id=activation.activation_id,
            composition_activation_hash=activation.semantic_content_hash,
        )
    )
    assert bound_input.accepted
    assert states.admit(source_state()).accepted
    cut = coordinator.capture_cut(cutoff_at=10.1).cut
    epoch_snapshot = epochs.snapshot(cut)
    source_snapshot = build_composition_source_eligibility_snapshot(
        project_composition_source_eligibility(states, cut),
        producer_epochs=epoch_snapshot,
        spec=eligibility_spec(spec),
        composition_spec=spec,
        target_effective_at=10.1,
    )
    snapshot = build_composition_input_snapshot(
        project_composition_inputs(inputs, cut),
        producer_epochs=epoch_snapshot,
        spec=spec,
        target_effective_at=10.1,
        activation_scope=activation,
        source_eligibility_snapshot=source_snapshot,
    )
    assert snapshot.expected_activation_id == activation.activation_id
    assert snapshot.source_eligibility_snapshot_id == (
        source_snapshot.source_eligibility_snapshot_id
    )
    assert snapshot.selected_input_by_role["base"].composition_input_id == (
        bound_input.composition_input_id
    )


@pytest.mark.parametrize(
    ("status", "rearm_required", "fault_code", "reason"),
    (
        ("paused", True, None, "source_status:paused"),
        ("fault", True, "deadman_lost", "source_status:fault"),
        ("disconnected", True, None, "source_status:disconnected"),
        ("rearming", True, None, "source_status:rearming"),
    ),
)
def test_unavailable_source_invalidates_unexpired_action_and_formal_gate_holds(
    status: str,
    rearm_required: bool,
    fault_code: str | None,
    reason: str,
) -> None:
    _, coordinator, epochs, inputs, states = stores()
    assert states.admit(
        source_state(
            status=status,
            rearm_required=rearm_required,
            fault_code=fault_code,
        )
    ).accepted
    spec, source_snapshot, input_snapshot = capture(
        coordinator,
        epochs,
        inputs,
        states,
        target=10.1,
    )
    assert input_draft().effective_until > input_snapshot.target_effective_at
    assert source_snapshot.ineligible_reason_by_role["base"] == reason
    assert input_snapshot.missing_roles == ("base",)
    assert input_snapshot.source_ineligibility_reason_by_role["base"] == reason
    result = DeterministicComposer().compose(input_snapshot, spec=spec)
    assert result.output_status == "no_action"
    disposition = decide_composition_authority_disposition(
        input_snapshot,
        result,
        spec=spec,
        gate_spec=CompositionAuthorityGateSpec(
            gate_spec_id="eligible-base-gate-v1",
            composition_spec_hash=spec.semantic_content_hash,
            composer_producer_id="composer",
            missing_behavior_by_role={"base": "hold"},
            proposal_ttl_sec=0.1,
        ),
    )
    assert disposition.action == "hold"
    assert disposition.reason == "all_inputs_missing"


def test_retryable_source_pause_can_request_no_action_without_granting_authority(
) -> None:
    _, coordinator, epochs, inputs, states = stores()
    assert states.admit(
        source_state(status="paused", rearm_required=False)
    ).accepted
    spec, _source_snapshot, input_snapshot = capture(
        coordinator,
        epochs,
        inputs,
        states,
        target=10.1,
    )
    result = DeterministicComposer().compose(input_snapshot, spec=spec)
    disposition = decide_composition_authority_disposition(
        input_snapshot,
        result,
        spec=spec,
        gate_spec=CompositionAuthorityGateSpec(
            gate_spec_id="retryable-base-gate-v1",
            composition_spec_hash=spec.semantic_content_hash,
            composer_producer_id="composer",
            missing_behavior_by_role={"base": "no_action"},
            proposal_ttl_sec=0.1,
            all_inputs_missing_behavior="no_action",
        ),
    )

    assert disposition.action == "no_action"
    assert disposition.downstream_request == "no_action"
    assert disposition.reason == "all_inputs_missing"
    assert disposition.authoritative is False


def test_expired_source_lease_invalidates_unexpired_action() -> None:
    _, coordinator, epochs, inputs, states = stores()
    assert states.admit(
        source_state(lease_expires_at=10.05)
    ).accepted
    _, source_snapshot, input_snapshot = capture(
        coordinator,
        epochs,
        inputs,
        states,
        target=10.1,
    )
    assert source_snapshot.ineligible_reason_by_role["base"] == (
        "source_lease_expired"
    )
    assert input_snapshot.missing_roles == ("base",)


def test_enabled_scope_requires_an_exact_source_state() -> None:
    _, coordinator, epochs, inputs, states = stores()
    _, source_snapshot, input_snapshot = capture(
        coordinator,
        epochs,
        inputs,
        states,
        target=10.1,
    )
    assert source_snapshot.ineligible_reason_by_role["base"] == (
        "source_state_missing"
    )
    assert input_snapshot.missing_roles == ("base",)


def test_control_cut_prevents_late_pause_from_rewriting_earlier_snapshot() -> None:
    _, coordinator, epochs, inputs, states = stores()
    assert states.admit(source_state()).accepted
    earlier_cut = coordinator.capture_cut(cutoff_at=10.1).cut
    assert states.admit(
        source_state(
            status="paused",
            status_epoch=2,
            rearm_required=True,
        )
    ).accepted
    spec = composition_spec()
    state_spec = eligibility_spec(spec)
    earlier = build_composition_source_eligibility_snapshot(
        project_composition_source_eligibility(states, earlier_cut),
        producer_epochs=epochs.snapshot(earlier_cut),
        spec=state_spec,
        composition_spec=spec,
        target_effective_at=10.1,
    )
    assert earlier.is_role_eligible("base")

    later_cut = coordinator.capture_cut(cutoff_at=10.1).cut
    later = build_composition_source_eligibility_snapshot(
        project_composition_source_eligibility(states, later_cut),
        producer_epochs=epochs.snapshot(later_cut),
        spec=state_spec,
        composition_spec=spec,
        target_effective_at=10.1,
    )
    assert later.ineligible_reason_by_role["base"] == "source_status:paused"


def test_old_status_epoch_and_late_heartbeat_cannot_revive_source() -> None:
    _, coordinator, epochs, inputs, states = stores()
    assert states.admit(source_state()).accepted
    assert states.admit(
        source_state(
            status="paused",
            status_epoch=2,
            rearm_required=True,
        )
    ).accepted
    late = states.admit(
        source_state(
            status="available",
            status_epoch=1,
            sequence=1,
            lease_expires_at=12.0,
        )
    )
    assert not late.accepted
    assert late.rejection_code == "status_epoch_rollback"
    _, source_snapshot, input_snapshot = capture(
        coordinator,
        epochs,
        inputs,
        states,
        target=10.1,
    )
    assert source_snapshot.ineligible_reason_by_role["base"] == (
        "source_status:paused"
    )
    assert input_snapshot.missing_roles == ("base",)


def test_rearm_required_source_needs_new_epoch_and_typed_receipt() -> None:
    _, coordinator, epochs, inputs, states = stores()
    assert states.admit(source_state()).accepted
    assert states.admit(
        source_state(
            status="fault",
            status_epoch=2,
            rearm_required=True,
            fault_code="device_fault",
        )
    ).accepted
    no_receipt = states.admit(
        source_state(
            status="available",
            status_epoch=3,
        )
    )
    assert not no_receipt.accepted
    assert no_receipt.rejection_code == "rearm_receipt_required"
    recovered = states.admit(
        source_state(
            status="available",
            status_epoch=3,
            rearm_receipt_id="source-rearm-receipt:3",
            rearm_receipt_hash="source-rearm-hash-3",
        )
    )
    assert recovered.accepted
    _, source_snapshot, input_snapshot = capture(
        coordinator,
        epochs,
        inputs,
        states,
        target=10.1,
    )
    assert source_snapshot.is_role_eligible("base")
    assert tuple(input_snapshot.selected_input_by_role) == ("base",)


def test_intermediate_unavailable_state_cannot_clear_rearm_latch() -> None:
    _, _, _, _, states = stores()
    assert states.admit(
        source_state(
            status="paused",
            status_epoch=1,
            rearm_required=True,
        )
    ).accepted
    clear_attempt = states.admit(
        source_state(
            status="disconnected",
            status_epoch=2,
            rearm_required=False,
        )
    )
    assert not clear_attempt.accepted
    assert clear_attempt.rejection_code == "rearm_latch_cannot_clear"
    available_attempt = states.admit(
        source_state(
            status="available",
            status_epoch=2,
        )
    )
    assert not available_attempt.accepted
    assert available_attempt.rejection_code == "rearm_receipt_required"


def test_profile_may_allow_explicit_new_epoch_recovery_without_rearm() -> None:
    _, _, _, _, states = stores()
    assert states.admit(source_state()).accepted
    assert states.admit(
        source_state(
            status="paused",
            status_epoch=2,
            rearm_required=False,
        )
    ).accepted
    recovered = states.admit(
        source_state(
            status="available",
            status_epoch=3,
        )
    )
    assert recovered.accepted


def test_same_epoch_heartbeat_may_extend_lease_but_not_change_status() -> None:
    _, _, _, _, states = stores()
    assert states.admit(source_state()).accepted
    heartbeat = states.admit(
        source_state(
            sequence=1,
            lease_expires_at=12.0,
        )
    )
    assert heartbeat.accepted
    transition_without_epoch = states.admit(
        source_state(
            status="paused",
            sequence=2,
            rearm_required=True,
        )
    )
    assert not transition_without_epoch.accepted
    assert (
        transition_without_epoch.rejection_code
        == "status_epoch_transition_required"
    )


def test_latest_unavailable_state_is_retained_as_fail_closed_tombstone() -> None:
    _, coordinator, epochs, _, states = stores()
    assert states.admit(
        source_state(lease_expires_at=13.0)
    ).accepted
    assert states.admit(
        source_state(
            status="paused",
            status_epoch=2,
            lease_expires_at=10.2,
            rearm_required=True,
        )
    ).accepted
    spec = composition_spec()
    state_spec = eligibility_spec(spec)
    cut = coordinator.capture_cut(cutoff_at=13.1).cut
    snapshot = states.capture_snapshot(
        cut,
        producer_epochs=epochs.snapshot(cut),
        spec=state_spec,
        composition_spec=spec,
        target_effective_at=13.1,
    )
    assert snapshot.ineligible_reason_by_role["base"] == "source_status:paused"
    health = states.retention_snapshot()
    assert health.retained_record_count == 1
    assert health.pruned_record_count == 1


def test_producer_restart_invalidates_old_source_state() -> None:
    clock, coordinator, epochs, inputs, states = stores()
    assert states.admit(
        source_state(
            status="fault",
            rearm_required=True,
            fault_code="device_fault",
        )
    ).accepted
    clock.value = 10.2
    epochs.activate(
        producer_id="human-producer",
        producer_epoch=2,
        produced_at=10.2,
        produced_clock_domain_id="human.monotonic",
        reason="restart",
    )
    _, source_snapshot, input_snapshot = capture(
        coordinator,
        epochs,
        inputs,
        states,
        target=10.2,
    )
    assert source_snapshot.ineligible_reason_by_role["base"] == (
        "source_state_missing"
    )
    assert input_snapshot.missing_roles == ("base",)
    restart_without_rearm = states.admit(
        source_state(
            producer_epoch=2,
            status_epoch=1,
            status="available",
            effective_from=10.2,
            lease_expires_at=10.4,
        )
    )
    assert not restart_without_rearm.accepted
    assert restart_without_rearm.rejection_code == "rearm_receipt_required"
    restarted = states.admit(
        source_state(
            producer_epoch=2,
            status_epoch=1,
            status="available",
            effective_from=10.2,
            lease_expires_at=10.4,
            rearm_receipt_id="source-rearm-receipt:restart",
            rearm_receipt_hash="source-rearm-hash-restart",
        )
    )
    assert restarted.accepted


def test_runtime_opt_in_events_are_externally_projectable() -> None:
    clock, coordinator, epochs, _, _ = stores()
    events: list[CompositionDomainEvent] = []
    catalog = component_catalog()
    runtime = ControlCompositionRuntime(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="fixture-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        action_schema=build_robot_action_v1_runtime_schema(
            component_catalog=catalog,
        ),
        event_emitter=lambda batch: events.extend(batch),
        proposal_admitter=lambda *_args, **_kwargs: None,
        context_retention_spec=retention_spec(),
        composition_input_retention_spec=retention_spec(),
        source_eligibility_retention_spec=retention_spec(),
        clock=clock,
    )
    assert runtime.admit_composition_input(input_draft()).accepted
    assert runtime.admit_source_eligibility(
        source_state(
            status="disconnected",
            rearm_required=True,
        )
    ).accepted
    state_spec = eligibility_spec()
    evaluation = runtime.evaluate_composition(
        spec=composition_spec(),
        source_eligibility_spec=state_spec,
        target_effective_at=10.1,
    )
    assert evaluation.composition_evaluation.output_status == "no_action"
    assert evaluation.input_snapshot.missing_roles == ("base",)
    assert evaluation.source_eligibility_snapshot is not None
    assert evaluation.evaluation_record is not None
    assert evaluation.evaluation_record.source_eligibility_spec is state_spec
    kinds = tuple(event.event_kind for event in events)
    assert "composition_source_eligibility_spec" in kinds
    assert "composition_source_eligibility_snapshot" in kinds

    trace_context = AuthoritativeTraceContext(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="fixture-0",
        clock_domain_id="control.monotonic",
    )
    facts = tuple(
        project_composition_domain_event(trace_context, event)
        for event in events
    )
    assert len(facts) == len(events)
    assert any(
        fact.event_type == "composition_source_eligibility_snapshot"
        for fact in facts
    )


def test_runtime_composition_support_is_capped_by_source_lease() -> None:
    clock, coordinator, epochs, _, _ = stores()
    runtime = ControlCompositionRuntime(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="fixture-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        action_schema=action_schema(),
        event_emitter=lambda _batch: None,
        proposal_admitter=lambda *_args, **_kwargs: None,
        context_retention_spec=retention_spec(),
        composition_input_retention_spec=retention_spec(),
        source_eligibility_retention_spec=retention_spec(),
        clock=clock,
    )
    assert runtime.admit_composition_input(input_draft()).accepted
    assert runtime.admit_source_eligibility(
        source_state(lease_expires_at=10.15)
    ).accepted
    evaluation = runtime.evaluate_composition(
        spec=composition_spec(),
        source_eligibility_spec=eligibility_spec(),
        target_effective_at=10.1,
    )

    assert evaluation.composition_evaluation.output_status == "ready"
    assert (
        evaluation.composition_evaluation.composition_supported_until
        == 10.15
    )


def test_disabled_runtime_rejects_source_state_without_changing_input_path() -> None:
    clock, coordinator, epochs, _, _ = stores()
    catalog = component_catalog()
    runtime = ControlCompositionRuntime(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="fixture-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        action_schema=build_robot_action_v1_runtime_schema(
            component_catalog=catalog,
        ),
        event_emitter=lambda _batch: None,
        proposal_admitter=lambda *_args, **_kwargs: None,
        context_retention_spec=retention_spec(),
        composition_input_retention_spec=retention_spec(),
        clock=clock,
    )
    disabled = runtime.admit_source_eligibility(source_state())
    assert not disabled.accepted
    assert disabled.rejection_code == "source_eligibility_disabled"
    assert runtime.admit_composition_input(input_draft()).accepted
    evaluation = runtime.evaluate_composition(
        spec=composition_spec(),
        target_effective_at=10.1,
    )
    assert evaluation.composition_evaluation.output_status == "ready"
    assert evaluation.source_eligibility_snapshot is None
