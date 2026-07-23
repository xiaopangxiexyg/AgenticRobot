from __future__ import annotations

from dataclasses import replace

import pytest

from robot_motion.action import (
    ActionComponentCatalog,
    ActionComponentSpec,
    DecisionScheduleSpec,
    build_m6_candidate_policy_arm_qpos_manifest,
    build_scheduled_reference_segment_activation_spec,
)
from robot_motion.position_control import ExecutionCapabilityProfile
from robot_motion.runtime import (
    DispatchSeamRejectedError,
    ExecutionCancellationRequest,
    ExecutionWorkItem,
)
from teleop_runtime.action.legacy_execution_context import (
    LegacyControlDomainExecutionContext,
)
from teleop_runtime.backends.arms.rm65_dispatch_inventory import (
    assess_rm65_arms_only_dispatch_inventory,
    build_rm65_arms_only_dispatch_inventory,
)


class ManualClock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def _capability() -> ExecutionCapabilityProfile:
    return ExecutionCapabilityProfile(
        profile_id="legacy-context-capability-v1",
        validated_max_selected_keyframe_hz=60.0,
        configured_selected_keyframe_hz=30.0,
        validated_max_coupled_ik_hz=None,
        configured_coupled_ik_hz=None,
        reference_resample_hz=60.0,
        command_hz=120.0,
        backend_servo_hz=1000.0,
        validated_action_spaces=("bimanual_qpos_v1",),
        validated_selection_groups=("arms",),
        supports_local_group_stop=False,
        acceptance_evidence_id="legacy-context-scoped-fixture-v1",
        decision_runtime_budget_sec=0.005,
        ik_runtime_budget_sec=1.0 / 60.0,
        end_to_end_control_budget_sec=1.0 / 30.0,
        required_cancellation_ack_levels=("software_queue",),
    )


def _manifest():
    catalog = ActionComponentCatalog(
        catalog_id="legacy-context-arms-v1",
        action_space="bimanual_qpos_v1",
        components=(
            ActionComponentSpec("left_arm", "arms", "left_arm", coupled=True),
            ActionComponentSpec("right_arm", "arms", "right_arm", coupled=True),
        ),
    )
    capability = _capability()
    schedule = DecisionScheduleSpec(
        spec_id="legacy-context-schedule-v1",
        decision_hz=30.0,
        target_lead_time_sec=0.05,
    )
    activation = build_scheduled_reference_segment_activation_spec(
        schedule,
        capability,
        minimum_activation_margin_sec=1.0 / 60.0,
        activation_margin_evidence_id="legacy-context-margin-v1",
    )
    return build_m6_candidate_policy_arm_qpos_manifest(
        catalog=catalog,
        capability=capability,
        activation=activation,
    )


def _context() -> LegacyControlDomainExecutionContext:
    manifest = _manifest()
    inventory = build_rm65_arms_only_dispatch_inventory(manifest)
    assessment = assess_rm65_arms_only_dispatch_inventory(inventory)
    return LegacyControlDomainExecutionContext.create_fake_only_unbound(
        runtime_session_id="session-legacy-context",
        control_domain_id="rm65-arms",
        embodiment_instance_id="rm65-bimanual-0",
        clock_domain_id="control-monotonic",
        owner_runtime_id="legacy-policy-frontend",
        execution_manifest=manifest,
        dispatch_inventory=inventory,
        inventory_assessment=assessment,
        initial_authority_epoch=0,
        initial_execution_epoch=0,
        clock=ManualClock(),
    )


def _work_item(*, generation: int = 0, execution_epoch: int = 0):
    return ExecutionWorkItem(
        work_item_id=f"backend-command-{generation}-{execution_epoch}",
        work_kind="backend_command",
        control_domain_id="rm65-arms",
        authority_epoch=0,
        execution_epoch=execution_epoch,
        segment_generation_by_group={"arms": generation},
        source_decision_id="decision-0",
        source_reference_segment_id="segment-0",
    )


def test_context_owns_one_unbound_legacy_scope_without_hardware_authorization() -> None:
    context = _context()
    snapshot = context.snapshot()

    assert snapshot.composition_mode == "fake_only_unbound"
    assert snapshot.current_owner_kind == "legacy_control_path"
    assert snapshot.current_owner_token_id == context.legacy_owner_token.token_id
    assert snapshot.segment_generation_by_group == {"arms": 0}
    assert not snapshot.production_binding_enabled
    assert snapshot.hardware_authorization_id is None
    assert not context.inventory_assessment.legacy_mediation_ready
    assert not context.inventory_assessment.confers_execution_authorization


def test_final_motion_fake_composition_checks_work_then_consumes_one_permit() -> None:
    context = _context()
    seam_id = "rm65.motion.pre_sdk.send_joint_target"
    prepared = context.prepare_fake_motion_invocation(
        seam_id=seam_id,
        work_item=_work_item(),
        command_lineage_id="command-0",
        directive_lineage_id="directive-0",
    )
    gate = context.fake_motion_gate(seam_id)
    calls: list[tuple[float, ...]] = []

    result = gate.invoke(
        prepared.invocation.permit,
        current_authority_epoch=prepared.invocation.current_authority_epoch,
        current_execution_epoch=prepared.invocation.current_execution_epoch,
        backend_call=lambda: calls.append((1.0, 2.0)) or "fake-sdk-result",
    )

    assert prepared.epoch_check.accepted
    assert not prepared.production_binding_enabled
    assert result.result == "fake-sdk-result"
    assert calls == [(1.0, 2.0)]
    with pytest.raises(DispatchSeamRejectedError, match="permit_already_consumed"):
        gate.invoke(
            prepared.invocation.permit,
            current_authority_epoch=0,
            current_execution_epoch=0,
            backend_call=lambda: calls.append((3.0, 4.0)),
        )
    assert calls == [(1.0, 2.0)]


def test_stale_group_generation_fails_before_a_permit_is_prepared() -> None:
    context = _context()
    with pytest.raises(RuntimeError, match="segment_generation_mismatch"):
        context.prepare_fake_motion_invocation(
            seam_id="rm65.motion.pre_sdk.send_joint_target",
            work_item=_work_item(generation=1),
            command_lineage_id="command-stale",
            directive_lineage_id="directive-stale",
        )


def test_logical_admission_cannot_be_misused_as_a_final_motion_gate() -> None:
    context = _context()
    with pytest.raises(ValueError, match="one-shot pre-SDK motion seam"):
        context.fake_motion_gate("rm65.motion.target_admission.send_command")


def test_context_rejects_an_assessment_for_another_inventory() -> None:
    manifest = _manifest()
    inventory = build_rm65_arms_only_dispatch_inventory(manifest)
    assessment = assess_rm65_arms_only_dispatch_inventory(inventory)
    mismatched = replace(assessment, inventory_hash="different-inventory-hash")

    with pytest.raises(ValueError, match="does not describe this inventory"):
        LegacyControlDomainExecutionContext.create_fake_only_unbound(
            runtime_session_id="session-legacy-context",
            control_domain_id="rm65-arms",
            embodiment_instance_id="rm65-bimanual-0",
            clock_domain_id="control-monotonic",
            owner_runtime_id="legacy-policy-frontend",
            execution_manifest=manifest,
            dispatch_inventory=inventory,
            inventory_assessment=mismatched,
            clock=ManualClock(),
        )


@pytest.mark.parametrize("reason", ["hold", "fault"])
def test_quick_stop_ticks_use_new_epoch_safety_work_and_one_shot_permits(
    reason: str,
) -> None:
    context = _context()
    cancellation = context.execution_epochs.cancel(
        ExecutionCancellationRequest(
            reason=reason,
            authority_epoch=0,
            requested_scope="control_domain",
            source_event_ids=(f"{reason}-request",),
        ),
        capability=_capability(),
    )
    synchronized = context.observe_fake_cancellation(cancellation)
    operation = context.admit_fake_rm65_quick_stop(
        cancellation=cancellation,
        operation_id=f"quick-stop-{reason}",
        frozen_stop_target_hash="frozen-last-accepted-qpos",
        maximum_duration_sec=0.1,
        maximum_tick_count=2,
    )
    prepared = context.prepare_fake_rm65_quick_stop_tick(
        seam_id="rm65.safety.quick_stop.tick.pre_sdk",
        operation_id=operation.operation_id,
        tick_sequence=1,
        command_lineage_id="quick-stop-qcmd-1",
    )
    gate = context.fake_safety_gate(prepared.seam_id)
    calls: list[str] = []

    result = gate.invoke(
        prepared.invocation.permit,
        current_authority_epoch=prepared.invocation.current_authority_epoch,
        current_execution_epoch=prepared.invocation.current_execution_epoch,
        backend_call=lambda: calls.append("fake-quick-stop-sdk") or 0,
    )

    assert synchronized.execution_epoch == cancellation.execution_epoch == 1
    assert prepared.work_item.work_kind == "safety_backend_command"
    assert prepared.work_item.execution_epoch == cancellation.execution_epoch
    assert prepared.work_item.source_decision_id is None
    assert prepared.work_item.source_safety_operation_id == operation.operation_id
    assert prepared.epoch_check.accepted
    assert result.validation.accepted
    assert calls == ["fake-quick-stop-sdk"]
    with pytest.raises(DispatchSeamRejectedError, match="permit_already_consumed"):
        gate.invoke(
            prepared.invocation.permit,
            current_authority_epoch=0,
            current_execution_epoch=1,
            backend_call=lambda: calls.append("reused"),
        )
    assert calls == ["fake-quick-stop-sdk"]


def test_quick_stop_cannot_be_admitted_before_owner_registry_observes_cancel() -> None:
    context = _context()
    cancellation = context.execution_epochs.cancel(
        ExecutionCancellationRequest(
            reason="hold",
            authority_epoch=0,
            requested_scope="control_domain",
            source_event_ids=("hold-request",),
        ),
        capability=_capability(),
    )

    with pytest.raises(ValueError, match="synchronized new execution epoch"):
        context.admit_fake_rm65_quick_stop(
            cancellation=cancellation,
            operation_id="quick-stop-too-early",
            frozen_stop_target_hash="frozen-last-accepted-qpos",
            maximum_duration_sec=0.1,
            maximum_tick_count=1,
        )
