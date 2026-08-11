from __future__ import annotations

import numpy as np
import pytest

from control_composition import (
    ResidualAnchorComponentBinding,
    ResidualAnchorReceipt,
    ResidualAnchorRequest,
    TimeWindowRetentionSpec,
    build_composition_activation_scope,
)
from control_composition.composition import (
    build_relative_baseline_residual_contracts,
)
from control_composition.events import CompositionDomainEvent
from control_composition.observability import SourceDifferenceSpec
from control_composition.runtime import ControlCompositionRuntime
from control_composition.trajectory import (
    FrozenTrajectoryResidualInput,
    ResidualTemporalWarpSpec,
    TrajectoryCompositionContributorStamp,
    activate_residual_session,
    empty_trajectory_editing_state,
    build_trajectory_corrected_preview,
    capture_frozen_trajectory_composition_projection,
    evaluate_trajectory_composition_shadow,
    offer_nominal_trajectory,
    trajectory_composition_shadow_domain_event,
)
from control_station.composition.trajectory_shadow import (
    TrajectoryCompositionShadowController,
    TrajectoryCompositionShadowProfile,
)
from dataset_interface.composition_control_trace import (
    project_composition_domain_event,
)
from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentAction,
    ComponentActionSchema,
    ComponentActionTrajectory,
    ComponentActionTrajectoryKeyframe,
    ControlComponentSpecV2,
    ProposalAdmissionReceipt,
)
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext
from robot_motion.runtime.control_admission import (
    ControlAdmissionCoordinator,
)
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry


pytestmark = [pytest.mark.unit, pytest.mark.contract]


NOMINAL_ROLE = "planned_reference"
RESIDUAL_ROLE = "operator_offset"
NOMINAL_PRODUCER = "local-planner-instance"
RESIDUAL_PRODUCER = "tracking-device-instance"


class Clock:
    def __init__(self, value: float = 9.95) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def _schemas() -> tuple[ComponentActionSchema, ComponentActionSchema]:
    common = {"schema_version": "development", "dtype": "float64"}
    absolute = {
        "tool_tip": ActionPayloadSchema(
            schema_id="tool-tip-absolute",
            representation_family="cartesian_pose",
            shape=(7,),
            coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
            units=("m", "m", "m", "1", "1", "1", "1"),
            value_semantics="absolute_target",
            reference_frame="task_world",
            controlled_frame="tool_tip",
            **common,
        ),
        "claw_joints": ActionPayloadSchema(
            schema_id="claw-joints-absolute",
            representation_family="joint_position",
            shape=(2,),
            coordinate_names=("joint_a", "joint_b"),
            units=("rad", "rad"),
            value_semantics="absolute_target",
            **common,
        ),
    }
    relative = {
        "tool_tip": ActionPayloadSchema(
            schema_id="tool-tip-relative",
            representation_family="cartesian_pose",
            shape=(7,),
            coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
            units=("m", "m", "m", "1", "1", "1", "1"),
            value_semantics="relative_delta",
            reference_frame="task_world",
            controlled_frame="tool_tip",
            semantic_parameters={
                "translation_application": "reference_frame_additive",
                "rotation_application": "baseline_local_right_multiply",
            },
            **common,
        ),
        "claw_joints": ActionPayloadSchema(
            schema_id="claw-joints-relative",
            representation_family="joint_position",
            shape=(2,),
            coordinate_names=("joint_a", "joint_b"),
            units=("rad", "rad"),
            value_semantics="relative_delta",
            semantic_parameters={"application": "elementwise_additive"},
            **common,
        ),
    }
    catalog = ActionComponentCatalogV2(
        catalog_id="arbitrary-topology-fixture",
        catalog_version="development",
        components=tuple(
            ControlComponentSpecV2(
                component_id=component_id,
                controlled_entity_id=f"entity:{component_id}",
                command_channel_id=f"channel:{component_id}",
                selection_group_id=f"group:{component_id}",
                resource_claims=(f"resource:{component_id}",),
                accepted_payload_schema_hashes=(
                    absolute[component_id].semantic_content_hash,
                    relative[component_id].semantic_content_hash,
                ),
            )
            for component_id in absolute
        ),
    )
    return (
        ComponentActionSchema(
            schema_id="arbitrary-topology-absolute",
            schema_version="development",
            component_catalog=catalog,
            payload_schema_by_component=absolute,
        ),
        ComponentActionSchema(
            schema_id="arbitrary-topology-relative",
            schema_version="development",
            component_catalog=catalog,
            payload_schema_by_component=relative,
        ),
    )


def _action(
    schema: ComponentActionSchema,
    *,
    x: float,
    joints: tuple[float, float],
) -> ComponentAction:
    return ComponentAction(
        action_schema=schema,
        values_by_component={
            "tool_tip": (x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            "claw_joints": joints,
        },
    )


def _trajectory(
    schema: ComponentActionSchema,
    *,
    revision: int = 0,
    offset: float = 0.0,
) -> ComponentActionTrajectory:
    keyframes = tuple(
        ComponentActionTrajectoryKeyframe(
            target_effective_at=10.0 + 0.1 * index,
            effective_until=10.1 + 0.1 * index,
            action=_action(
                schema,
                x=offset + float(index),
                joints=(0.2 + 0.1 * index, 0.4 + 0.1 * index),
            ),
            source_index=index,
            source_fact_ids=(f"source-step:{revision}:{index}",),
            source_fact_hashes=(f"source-step-hash:{revision}:{index}",),
        )
        for index in range(3)
    )
    return ComponentActionTrajectory(
        trajectory_id=f"canonical-plan:{revision}",
        revision=revision,
        source_kind="generic_planned_action",
        source_instance_id=NOMINAL_PRODUCER,
        clock_domain_id="control.monotonic",
        created_at=9.5 + 0.01 * revision,
        keyframes=keyframes,
        source_fact_ids=(f"source-plan:{revision}",),
        source_fact_hashes=(f"source-plan-hash:{revision}",),
        source_activation_id="nominal-source-activation:1",
        source_activation_hash="nominal-source-activation-hash:1",
    )


def _runtime_fixture():
    absolute, relative = _schemas()
    contracts = build_relative_baseline_residual_contracts(
        contract_id="generic-plan-plus-offset",
        output_action_schema=absolute,
        residual_action_schema=relative,
        baseline_role=NOMINAL_ROLE,
        residual_role=RESIDUAL_ROLE,
    )
    clock = Clock()
    coordinator = ControlAdmissionCoordinator(
        "virtual-domain",
        clock_domain=ClockDomainDescriptor(
            clock_domain_id="control.monotonic",
            runtime_session_id="virtual-session",
            boot_id="virtual-boot",
            reset_epoch=0,
        ),
        clock=clock,
    )
    epochs = ProducerEpochRegistry(
        runtime_session_id="virtual-session",
        coordinator=coordinator,
    )
    for producer_id in (NOMINAL_PRODUCER, RESIDUAL_PRODUCER):
        epochs.activate(
            producer_id=producer_id,
            producer_epoch=1,
            produced_at=9.8,
            produced_clock_domain_id="control.monotonic",
            reason="virtual_start",
        )
    events: list[CompositionDomainEvent] = []
    proposals = []

    def admit_proposal(draft, **_kwargs):
        proposals.append(draft)
        return ProposalAdmissionReceipt(accepted=True)

    retention = TimeWindowRetentionSpec(
        spec_id="trajectory-shadow-window",
        history_window_sec=0.2,
        max_future_lead_sec=1.0,
        max_effective_duration_sec=1.0,
    )
    runtime = ControlCompositionRuntime(
        runtime_session_id="virtual-session",
        control_domain_id="virtual-domain",
        embodiment_instance_id="arbitrary-embodiment",
        coordinator=coordinator,
        producer_epochs=epochs,
        action_schema=absolute,
        input_action_schema_by_role={
            NOMINAL_ROLE: absolute,
            RESIDUAL_ROLE: relative,
        },
        event_emitter=events.extend,
        proposal_admitter=admit_proposal,
        context_retention_spec=retention,
        composition_input_retention_spec=retention,
        composition_strategy=contracts.strategy,
        clock=clock,
    )
    return runtime, coordinator, contracts, absolute, relative, events, proposals


def _state(coordinator, absolute, relative, *, residual_active: bool):
    state = offer_nominal_trajectory(
        empty_trajectory_editing_state(reducer_id="generic-trajectory-editor"),
        _trajectory(absolute),
        offer_sequence=1,
    )
    if not residual_active:
        return state
    scope = build_composition_activation_scope(
        operation_spec_id="generic-residual-operation",
        operation_spec_hash="generic-residual-operation-hash",
        session_epoch=1,
        activated_at=9.7,
        clock_domain_id="control.monotonic",
        input_barrier_at=9.7,
        required_roles=(NOMINAL_ROLE, RESIDUAL_ROLE),
    )
    request = ResidualAnchorRequest.for_activation(
        scope,
        request_generation=0,
        source_role=RESIDUAL_ROLE,
        component_ids=absolute.component_ids,
        source_binding_id="tracking-source-binding",
        source_binding_hash="tracking-source-binding-hash",
        absolute_action_schema_hash=absolute.semantic_content_hash,
        relative_action_schema_hash=relative.semantic_content_hash,
        requested_at=9.75,
    )
    receipt = ResidualAnchorReceipt.capture(
        request,
        source_action_id="tracking-source-action",
        source_action_hash="tracking-source-action-hash",
        source_available_at=9.85,
        captured_at=9.9,
        component_bindings=tuple(
            ResidualAnchorComponentBinding(
                component_id=component_id,
                source_sample_id=f"tracking-sample:{component_id}",
                source_sample_hash=f"tracking-sample-hash:{component_id}",
                absolute_payload_schema_hash=(
                    absolute.payload_schema_by_component[
                        component_id
                    ].semantic_content_hash
                ),
                anchor_id=f"tracking-anchor:{component_id}",
                anchor_hash=f"tracking-anchor-hash:{component_id}",
            )
            for component_id in absolute.component_ids
        ),
    )
    cut = coordinator.capture_cut(cutoff_at=9.95).cut
    return activate_residual_session(
        state,
        residual_session_id="generic-residual-session",
        activation_scope=scope,
        control_cut=cut,
        residual_anchor_receipt=receipt,
        first_editable_target_at=10.0,
    )


def _stamp(producer_id: str, sequence: int):
    return TrajectoryCompositionContributorStamp(
        producer_id=producer_id,
        producer_epoch=1,
        sequence=sequence,
        produced_at=9.9,
        produced_clock_domain_id="control.monotonic",
    )


def _evaluate(*, residual_active: bool = True, residual_until: float = 10.2):
    runtime, coordinator, contracts, absolute, relative, events, proposals = (
        _runtime_fixture()
    )
    state = _state(
        coordinator,
        absolute,
        relative,
        residual_active=residual_active,
    )
    anchor_receipt = state.active_residual_anchor_receipt
    result = evaluate_trajectory_composition_shadow(
        runtime,
        state=state,
        target_effective_at=10.1,
        composition_spec=contracts.composition_spec,
        nominal_role=NOMINAL_ROLE,
        residual_role=RESIDUAL_ROLE,
        nominal_stamp=_stamp(NOMINAL_PRODUCER, 0),
        residual_stamp=(
            _stamp(RESIDUAL_PRODUCER, 0) if residual_active else None
        ),
        residual_action=(
            _action(relative, x=0.2, joints=(0.4, -0.2))
            if residual_active
            else None
        ),
        residual_fact_id=("tracking-residual:0" if residual_active else None),
        residual_fact_hash=(
            "tracking-residual-hash:0" if residual_active else None
        ),
        residual_anchor_receipt_id=(
            None if anchor_receipt is None else anchor_receipt.receipt_id
        ),
        residual_anchor_receipt_hash=(
            None
            if anchor_receipt is None
            else anchor_receipt.semantic_content_hash
        ),
        residual_effective_until=(residual_until if residual_active else None),
        residual_warp_spec=(
            ResidualTemporalWarpSpec(
                spec_id="linear-entry-ramp",
                ramp_duration_sec=0.2,
            )
            if residual_active
            else None
        ),
        source_difference_spec=(
            SourceDifferenceSpec(
                spec_id="corrected-minus-nominal",
                composition_spec_hash=contracts.composition_spec.semantic_content_hash,
                compared_reference_role_pairs=(),
                output_residual_roles=(NOMINAL_ROLE,),
            )
            if residual_active
            else None
        ),
    )
    return result, state, runtime, events, proposals


def test_pass_through_preview_does_not_touch_composition_runtime() -> None:
    result, _state_value, runtime, events, proposals = _evaluate(
        residual_active=False
    )
    assert result.output_status == "ready"
    assert result.reason == "pass_through_nominal_preview"
    assert result.corrected_action is result.nominal_action
    assert result.executable_action is None
    assert not result.authoritative
    assert not result.proposal_admitted
    assert result.nominal_input_admission is None
    assert result.composition_submission is None
    assert not events
    assert not proposals
    assert runtime.coordinator.committed_high_watermark == 2


def test_residual_shadow_runs_real_composition_without_publication() -> None:
    result, state, _runtime, events, proposals = _evaluate()
    assert result.output_status == "ready"
    assert result.residual_sample is not None
    assert result.residual_sample.weight == pytest.approx(0.5)
    assert result.nominal_input_admission.accepted
    assert result.residual_input_admission.accepted
    assert result.composition_submission.proposal_admission is None
    assert result.executable_action is None
    assert state.last_emitted_target_at is None
    assert not proposals

    corrected = result.corrected_action
    assert corrected is not None
    np.testing.assert_allclose(
        corrected.value("tool_tip").values[:3],
        (1.1, 0.0, 0.0),
    )
    np.testing.assert_allclose(
        corrected.value("claw_joints").values,
        (0.5, 0.4),
    )
    difference = result.composition_submission.source_difference_observation
    assert difference is not None
    assert len(difference.output_residuals) == 1
    metric_by_component = {
        item.component_id: item
        for item in difference.output_residuals[0].component_metrics
    }
    assert metric_by_component["tool_tip"].translation_norm_m == pytest.approx(0.1)
    assert metric_by_component["claw_joints"].qpos_delta == pytest.approx(
        (0.2, -0.1)
    )
    assert any(event.event_kind == "composition_evaluation" for event in events)


def test_expired_residual_returns_no_action_before_input_admission() -> None:
    result, state, runtime, events, proposals = _evaluate(residual_until=10.1)
    assert result.output_status == "no_action"
    assert result.reason == "residual_support_expired"
    assert result.nominal_input_admission is None
    assert result.residual_input_admission is None
    assert state.last_emitted_target_at is None
    assert runtime.coordinator.committed_high_watermark == 2
    assert not events
    assert not proposals


@pytest.mark.parametrize(
    ("target_effective_at", "expected_reason"),
    (
        (10.1, "residual_input_unavailable"),
        (10.3, "trajectory_preparation:trajectory_support_expired"),
    ),
)
def test_missing_residual_or_nominal_support_never_partially_admits(
    target_effective_at: float,
    expected_reason: str,
) -> None:
    runtime, coordinator, contracts, absolute, relative, events, proposals = (
        _runtime_fixture()
    )
    state = _state(coordinator, absolute, relative, residual_active=True)
    result = evaluate_trajectory_composition_shadow(
        runtime,
        state=state,
        target_effective_at=target_effective_at,
        composition_spec=contracts.composition_spec,
        nominal_role=NOMINAL_ROLE,
        residual_role=RESIDUAL_ROLE,
        nominal_stamp=_stamp(NOMINAL_PRODUCER, 0),
    )
    assert result.output_status == "no_action"
    assert result.reason == expected_reason
    assert result.nominal_input_admission is None
    assert result.residual_input_admission is None
    assert runtime.coordinator.committed_high_watermark == 2
    assert not events
    assert not proposals


def test_latest_shadow_is_preview_only_and_does_not_change_publication_basis() -> None:
    runtime, coordinator, contracts, absolute, relative, events, proposals = (
        _runtime_fixture()
    )
    state = _state(coordinator, absolute, relative, residual_active=True)
    basis_before = state.publication_basis_hash
    state = offer_nominal_trajectory(
        state,
        _trajectory(absolute, revision=1, offset=100.0),
        offer_sequence=2,
    )
    anchor_receipt = state.active_residual_anchor_receipt
    assert anchor_receipt is not None
    assert state.publication_basis_hash == basis_before
    result = evaluate_trajectory_composition_shadow(
        runtime,
        state=state,
        target_effective_at=10.1,
        composition_spec=contracts.composition_spec,
        nominal_role=NOMINAL_ROLE,
        residual_role=RESIDUAL_ROLE,
        nominal_stamp=_stamp(NOMINAL_PRODUCER, 0),
        residual_stamp=_stamp(RESIDUAL_PRODUCER, 0),
        residual_action=_action(relative, x=0.2, joints=(0.4, -0.2)),
        residual_fact_id="tracking-residual:0",
        residual_fact_hash="tracking-residual-hash:0",
        residual_anchor_receipt_id=anchor_receipt.receipt_id,
        residual_anchor_receipt_hash=anchor_receipt.semantic_content_hash,
        residual_effective_until=10.2,
        residual_warp_spec=ResidualTemporalWarpSpec(
            spec_id="linear-entry-ramp",
            ramp_duration_sec=0.2,
        ),
    )
    assert result.output_status == "ready"
    assert result.shadow_action is not None
    assert result.shadow_action.value("tool_tip").values[0] == pytest.approx(101.0)
    assert result.nominal_action.value("tool_tip").values[0] == pytest.approx(1.0)
    assert result.corrected_action.value("tool_tip").values[0] == pytest.approx(1.1)
    assert result.preparation.publication_basis_hash == basis_before
    assert not proposals


def test_shadow_summary_event_retains_non_authoritative_identity() -> None:
    result, _state_value, _runtime, _events, _proposals = _evaluate()
    event = trajectory_composition_shadow_domain_event(
        result,
        occurred_at=9.96,
    )
    assert event.event_kind == "trajectory_composition_shadow_result"
    assert event.subject is result
    assert event.occurred_at == pytest.approx(9.96)
    assert not result.authoritative
    assert result.executable_action is None
    fact = project_composition_domain_event(
        AuthoritativeTraceContext(
            runtime_session_id="virtual-session",
            control_domain_id="virtual-domain",
            embodiment_instance_id="arbitrary-embodiment",
            clock_domain_id="control.monotonic",
        ),
        event,
    )
    assert fact.event_type == "trajectory_composition_shadow_result"
    assert fact.payload["nominal_action_hash"] == (
        result.nominal_action.semantic_content_hash
    )
    assert fact.payload["corrected_action_hash"] == (
        result.corrected_action.semantic_content_hash
    )
    assert fact.payload["proposal_admitted"] is False
    assert fact.payload["authoritative"] is False
    assert "nominal_action" not in fact.payload


def test_station_shadow_profile_is_default_off_and_lazy() -> None:
    emitted = []
    called = []
    controller = TrajectoryCompositionShadowController(
        profile=TrajectoryCompositionShadowProfile(),
        clock=lambda: 9.96,
        event_emitter=emitted.extend,
    )

    def should_not_run():
        called.append(True)
        raise AssertionError("default-off shadow fetched a source")

    cycle = controller.run_shadow_cycle(should_not_run)
    assert cycle.status == "disabled"
    assert cycle.evaluation is None
    assert not called
    assert not emitted


def test_station_enabled_shadow_emits_only_non_authoritative_summary() -> None:
    result, _state_value, _runtime, _events, _proposals = _evaluate()
    emitted = []
    controller = TrajectoryCompositionShadowController(
        profile=TrajectoryCompositionShadowProfile(
            profile_id="virtual-trajectory-residual-shadow",
            enabled=True,
            send_actions=False,
        ),
        clock=lambda: 9.96,
        event_emitter=emitted.extend,
    )
    cycle = controller.run_shadow_cycle(lambda: result)
    assert cycle.status == "evaluated"
    assert cycle.evaluation is result
    assert emitted == [cycle.domain_event]
    assert cycle.domain_event.event_kind == (
        "trajectory_composition_shadow_result"
    )
    assert cycle.evaluation.executable_action is None


def test_station_shadow_rejects_send_actions_true() -> None:
    with pytest.raises(ValueError, match="send_actions=true"):
        TrajectoryCompositionShadowProfile(
            profile_id="invalid-motion-profile",
            enabled=True,
            send_actions=True,
        )
    with pytest.raises(TypeError, match="must be booleans"):
        TrajectoryCompositionShadowProfile(
            profile_id="misparsed-profile",
            enabled="false",
            send_actions=False,
        )


def _capture_corrected_projection(*, residual_until: float = 10.25):
    runtime, coordinator, contracts, absolute, relative, events, proposals = (
        _runtime_fixture()
    )
    state = _state(
        coordinator,
        absolute,
        relative,
        residual_active=True,
    )
    anchor = state.active_residual_anchor_receipt
    assert anchor is not None
    residual = FrozenTrajectoryResidualInput(
        input_role=RESIDUAL_ROLE,
        action=_action(relative, x=0.2, joints=(0.4, -0.2)),
        contributor_stamp=_stamp(RESIDUAL_PRODUCER, 0),
        residual_fact_id="tracking-residual:frozen",
        residual_fact_hash="tracking-residual-frozen-hash",
        residual_anchor_receipt_id=anchor.receipt_id,
        residual_anchor_receipt_hash=anchor.semantic_content_hash,
        effective_until=residual_until,
        warp_spec=ResidualTemporalWarpSpec(
            spec_id="frozen-linear-entry-ramp",
            ramp_duration_sec=0.2,
        ),
    )
    before = (
        runtime.composition_inputs.retention_snapshot(),
        runtime.contexts.retention_snapshot(),
        len(runtime.composition_inputs),
    )
    projection = capture_frozen_trajectory_composition_projection(
        runtime,
        state=state,
        composition_spec=contracts.composition_spec,
        nominal_role=NOMINAL_ROLE,
        nominal_stamp=_stamp(NOMINAL_PRODUCER, 0),
        residual_input=residual,
        cutoff_at=9.95,
    )
    after_capture = (
        runtime.composition_inputs.retention_snapshot(),
        runtime.contexts.retention_snapshot(),
        len(runtime.composition_inputs),
    )
    return (
        projection,
        runtime,
        events,
        proposals,
        before,
        after_capture,
    )


def test_corrected_horizon_uses_one_non_closing_frozen_basis() -> None:
    projection, runtime, events, proposals, before, after_capture = (
        _capture_corrected_projection()
    )

    first = build_trajectory_corrected_preview(projection)
    replay = build_trajectory_corrected_preview(projection)

    assert tuple(point.output_status for point in first.points) == (
        "ready",
        "ready",
        "ready",
    )
    assert first.semantic_content_hash == replay.semantic_content_hash
    assert before == after_capture
    assert runtime.composition_inputs.retention_snapshot() == before[0]
    assert runtime.contexts.retention_snapshot() == before[1]
    assert len(runtime.composition_inputs) == 0
    assert not events
    assert not proposals
    assert first.promotable is False
    assert first.authoritative is False
    assert all(point.promotable is False for point in first.points)
    assert all(point.execution_guaranteed is False for point in first.points)
    activation = projection.state.residual_session_activation_binding
    assert activation is not None
    assert all(
        point.input_snapshot is not None
        and point.input_snapshot.producer_epoch_snapshot_hash
        and point.input_snapshot.expected_activation_id
        == activation.activation_scope.activation_id
        and point.input_snapshot.expected_activation_hash
        == activation.activation_scope.semantic_content_hash
        for point in first.points
    )
    assert len(
        {
            point.input_snapshot.control_cut_hash
            for point in first.points
            if point.input_snapshot is not None
        }
    ) == 1
    assert not hasattr(
        first.points[0].projected_inputs.inputs[0],
        "control_admission_sequence",
    )
    assert not hasattr(
        first.points[0].projected_inputs.inputs[0],
        "composition_input_id",
    )


def test_corrected_horizon_reapplies_one_residual_without_integrating_it() -> None:
    projection, *_ = _capture_corrected_projection()

    preview = build_trajectory_corrected_preview(projection)
    corrected = tuple(point.corrected_action for point in preview.points)
    assert all(action is not None for action in corrected)
    np.testing.assert_allclose(
        corrected[0].value("tool_tip").values[:3],
        (0.0, 0.0, 0.0),
    )
    np.testing.assert_allclose(
        corrected[1].value("tool_tip").values[:3],
        (1.1, 0.0, 0.0),
    )
    np.testing.assert_allclose(
        corrected[2].value("tool_tip").values[:3],
        (2.2, 0.0, 0.0),
    )
    np.testing.assert_allclose(
        corrected[1].value("claw_joints").values,
        (0.5, 0.4),
    )
    np.testing.assert_allclose(
        corrected[2].value("claw_joints").values,
        (0.8, 0.4),
    )


def test_corrected_horizon_stops_at_frozen_residual_support() -> None:
    projection, *_ = _capture_corrected_projection(residual_until=10.15)

    preview = build_trajectory_corrected_preview(projection)

    assert tuple(point.output_status for point in preview.points) == (
        "ready",
        "ready",
        "unavailable",
    )
    assert preview.points[2].reason == "residual_support_expired"
    assert preview.points[2].corrected_action is None
