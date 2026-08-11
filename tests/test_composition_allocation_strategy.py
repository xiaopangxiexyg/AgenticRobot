from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from control_composition.allocation import (
    COMPOSITION_ALLOCATION_CONTEXT_CONTRACT_HASH_DEVELOPMENT,
    ComponentAllocationEvidencePolicy,
    ComponentAllocationPolicy,
    ComponentResidualMagnitudeObservation,
    ComponentSourceActivityObservation,
    CompositionAllocationSpec,
    AllocationTransitionContinuityPolicy,
    build_composition_allocation_context_draft,
    empty_composition_allocation_state,
    evaluate_composition_allocation,
)
from control_composition.allocation.composition_profile import (
    build_allocated_composition_contracts,
)
from control_composition import (
    RESIDUAL_ANCHOR_LIFECYCLE_PENDING,
    RESIDUAL_ANCHOR_LIFECYCLE_READY,
    ResidualAnchorComponentBinding,
    ResidualAnchorLifecycleDraft,
    ResidualAnchorReceipt,
    ResidualAnchorRequest,
    build_composition_activation_scope,
    build_residual_anchor_lifecycle_snapshot,
)
from control_composition.composition import (
    FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
    AllocatedPoseComponentCompositionRule,
    AllocatedQposComponentCompositionRule,
    ComponentCompositionRuleSet,
    CompositionInputDraft,
    CompositionSpec,
    CompositionContextProjection,
    FormalComponentCompositionStrategyV2,
    FormalCompositionStrategySpec,
    ProjectedCompositionInput,
    ProjectedCompositionStrategyInput,
    build_composition_context_snapshot,
    build_projected_composition_input_snapshot,
    evaluate_projected_composition_strategy,
    frozen_control_cut_semantic_hash,
)
from control_composition.information import (
    CompositionContextRequirement,
    CompositionInformationPlan,
)
from control_composition.retention import TimeWindowRetentionSpec
from control_composition.runtime import ControlCompositionRuntime
from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentAction,
    ComponentActionSchema,
    ControlComponentSpecV2,
    ProposalAdmissionReceipt,
)
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry


pytestmark = [pytest.mark.unit, pytest.mark.contract]


class Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def _schemas() -> tuple[ComponentActionSchema, ComponentActionSchema]:
    absolute_payloads = {
        "arm": ActionPayloadSchema(
            schema_id="allocated-arm-absolute",
            schema_version="development",
            representation_family="cartesian_pose",
            dtype="float64",
            shape=(7,),
            coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
            units=("m", "m", "m", "1", "1", "1", "1"),
            value_semantics="absolute_target",
            reference_frame="world",
            controlled_frame="tool",
        ),
        "hand": ActionPayloadSchema(
            schema_id="allocated-hand-absolute",
            schema_version="development",
            representation_family="joint_position",
            dtype="float64",
            shape=(2,),
            coordinate_names=("finger_0", "finger_1"),
            units=("rad", "rad"),
            value_semantics="absolute_target",
        ),
    }
    delta_payloads = {
        "arm": ActionPayloadSchema(
            schema_id="allocated-arm-relative",
            schema_version="development",
            representation_family="cartesian_pose",
            dtype="float64",
            shape=(7,),
            coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
            units=("m", "m", "m", "1", "1", "1", "1"),
            value_semantics="relative_delta",
            reference_frame="world",
            controlled_frame="tool",
        ),
        "hand": ActionPayloadSchema(
            schema_id="allocated-hand-relative",
            schema_version="development",
            representation_family="joint_position",
            dtype="float64",
            shape=(2,),
            coordinate_names=("finger_0", "finger_1"),
            units=("rad", "rad"),
            value_semantics="relative_delta",
        ),
    }
    catalog = ActionComponentCatalogV2(
        catalog_id="allocated-arm-hand",
        catalog_version="development",
        components=tuple(
            ControlComponentSpecV2(
                component_id=component_id,
                controlled_entity_id=component_id,
                command_channel_id=component_id,
                selection_group_id=component_id,
                resource_claims=(f"resource:{component_id}",),
                accepted_payload_schema_hashes=(
                    absolute_payloads[component_id].semantic_content_hash,
                    delta_payloads[component_id].semantic_content_hash,
                ),
            )
            for component_id in ("arm", "hand")
        ),
    )
    return (
        ComponentActionSchema(
            schema_id="allocated-absolute-output",
            schema_version="development",
            component_catalog=catalog,
            payload_schema_by_component=absolute_payloads,
        ),
        ComponentActionSchema(
            schema_id="allocated-relative-input",
            schema_version="development",
            component_catalog=catalog,
            payload_schema_by_component=delta_payloads,
        ),
    )


def _allocation_spec() -> CompositionAllocationSpec:
    return CompositionAllocationSpec(
        allocation_spec_id="allocated-arm-hand",
        base_absolute_role="base_absolute",
        relative_overlay_role="operator_relative",
        absolute_overlay_role="operator_absolute",
        component_policies=tuple(
            ComponentAllocationPolicy(
                component_id=component_id,
                motion_source_role="operator_activity",
                residual_source_role="operator_relative",
                motion_policy=ComponentAllocationEvidencePolicy(
                    relative_enter_level=1.0,
                    relative_exit_level=0.5,
                    absolute_enter_level=3.0,
                    absolute_exit_level=2.0,
                ),
                residual_magnitude_policy=ComponentAllocationEvidencePolicy(
                    relative_enter_level=1.0,
                    relative_exit_level=0.5,
                    absolute_enter_level=3.0,
                    absolute_exit_level=2.0,
                ),
            )
            for component_id in ("arm", "hand")
        ),
    )


def _contract():
    output_schema, delta_schema = _schemas()
    allocation_spec = _allocation_spec()
    contracts = build_allocated_composition_contracts(
        contract_id="activity-allocated-arm-hand",
        output_action_schema=output_schema,
        relative_action_schema=delta_schema,
        allocation_spec=allocation_spec,
        allocation_context_channel="source_allocation",
        maximum_decision_latency_sec=0.1,
    )
    return (
        output_schema,
        delta_schema,
        allocation_spec,
        contracts.composition_spec,
        contracts.strategy,
    )


def _continuity_contract():
    output_schema, delta_schema = _schemas()
    continuity_policy = AllocationTransitionContinuityPolicy(
        policy_id="allocated-arm-hand-continuity",
        component_ids=output_schema.component_ids,
        handoff_duration_sec=0.2,
    )
    allocation_spec = replace(
        _allocation_spec(),
        transition_continuity_policy_hash=(
            continuity_policy.semantic_content_hash
        ),
    )
    contracts = build_allocated_composition_contracts(
        contract_id="continuous-allocated-arm-hand",
        output_action_schema=output_schema,
        relative_action_schema=delta_schema,
        allocation_spec=allocation_spec,
        allocation_context_channel="source_allocation",
        transition_continuity_policy=continuity_policy,
    )
    return output_schema, delta_schema, allocation_spec, contracts


def test_generic_allocated_contract_builder_matches_three_fixed_roles() -> None:
    output_schema, delta_schema = _schemas()
    allocation_spec = _allocation_spec()

    contracts = build_allocated_composition_contracts(
        contract_id="generic-allocated-arm-hand",
        output_action_schema=output_schema,
        relative_action_schema=delta_schema,
        allocation_spec=allocation_spec,
        allocation_context_channel="source_allocation",
    )

    assert contracts.composition_spec.input_roles == (
        "base_absolute",
        "operator_relative",
        "operator_absolute",
    )
    assert contracts.composition_spec.required_roles == (
        "base_absolute",
        "operator_relative",
        "operator_absolute",
    )
    assert contracts.composition_spec.missing_required_behavior == "no_action"
    assert contracts.strategy.spec.required_context_channels == (
        "source_allocation",
    )
    assert tuple(
        rule.representation for rule in contracts.component_rule_set.rules
    ) == ("pose", "qpos")


def _runtime(output_schema, delta_schema, strategy, *, lifecycle_ready: bool = True):
    clock = Clock()
    coordinator = ControlAdmissionCoordinator(
        "robot",
        clock_domain=ClockDomainDescriptor(
            clock_domain_id="control.monotonic",
            runtime_session_id="session",
            boot_id="boot",
            reset_epoch=0,
        ),
        clock=clock,
    )
    epochs = ProducerEpochRegistry(
        runtime_session_id="session",
        coordinator=coordinator,
    )
    for producer in (
        "base_absolute",
        "operator_relative",
        "operator_absolute",
        "allocation_reducer",
        "anchor_lifecycle",
        "formal-composer",
    ):
        epochs.activate(
            producer_id=producer,
            producer_epoch=1,
            produced_at=9.9,
            produced_clock_domain_id=f"{producer}.monotonic",
            reason="startup",
        )
    retention = TimeWindowRetentionSpec(
        spec_id="allocation-strategy-window",
        history_window_sec=0.0,
        max_future_lead_sec=2.0,
        max_effective_duration_sec=2.0,
    )
    information_plan = CompositionInformationPlan(
        plan_id="allocation-strategy-information",
        context_requirements=(
            CompositionContextRequirement(
                channel="source_allocation",
                accepted_payload_contract_hashes=(
                    COMPOSITION_ALLOCATION_CONTEXT_CONTRACT_HASH_DEVELOPMENT,
                ),
                payload_access="inline",
                maximum_age_sec=1.0,
                required=not bool(strategy.spec.optional_context_channels),
            ),
        ),
    )
    runtime = ControlCompositionRuntime(
        runtime_session_id="session",
        control_domain_id="robot",
        embodiment_instance_id="virtual",
        coordinator=coordinator,
        producer_epochs=epochs,
        action_schema=output_schema,
        input_action_schema_by_role={
            "base_absolute": output_schema,
            "operator_relative": delta_schema,
            "operator_absolute": output_schema,
        },
        event_emitter=lambda _events: None,
        proposal_admitter=lambda _draft, **_kwargs: ProposalAdmissionReceipt(
            accepted=True
        ),
        context_retention_spec=retention,
        composition_input_retention_spec=retention,
        information_plan=information_plan,
        composition_strategy=strategy,
        clock=clock,
    )
    activation = build_composition_activation_scope(
        operation_spec_id="allocation-test",
        operation_spec_hash="allocation-test-hash",
        session_epoch=1,
        activated_at=9.9,
        clock_domain_id="control.monotonic",
        input_barrier_at=9.9,
        required_roles=strategy.composition_spec.required_roles,
    )
    request = ResidualAnchorRequest.for_activation(
        activation,
        request_generation=0,
        source_role="operator_relative",
        component_ids=output_schema.component_ids,
        source_binding_id="test-operator",
        source_binding_hash="test-operator-hash",
        absolute_action_schema_hash=output_schema.semantic_content_hash,
        relative_action_schema_hash=delta_schema.semantic_content_hash,
        requested_at=9.9,
    )
    receipt = ResidualAnchorReceipt.capture(
        request,
        source_action_id="operator-anchor:0",
        source_action_hash="operator-anchor-hash:0",
        source_available_at=9.9,
        captured_at=9.9,
        component_bindings=tuple(
            ResidualAnchorComponentBinding(
                component_id=component_id,
                source_sample_id=f"operator-anchor:{component_id}:0",
                source_sample_hash=f"operator-anchor-hash:{component_id}:0",
                absolute_payload_schema_hash=(
                    output_schema.payload_schema_by_component[
                        component_id
                    ].semantic_content_hash
                ),
                anchor_id=f"anchor:{component_id}:0",
                anchor_hash=f"anchor-hash:{component_id}:0",
            )
            for component_id in output_schema.component_ids
        ),
    )
    lifecycle_spec = strategy.residual_anchor_lifecycle_spec
    lifecycle_records = [(0, RESIDUAL_ANCHOR_LIFECYCLE_PENDING, None)]
    if lifecycle_ready:
        lifecycle_records.append((1, RESIDUAL_ANCHOR_LIFECYCLE_READY, receipt))
    for sequence, phase, anchor_receipt in lifecycle_records:
        lifecycle_admission = runtime.admit_residual_anchor_lifecycle(
            ResidualAnchorLifecycleDraft(
                runtime_session_id="session",
                control_domain_id="robot",
                producer_id="anchor_lifecycle",
                producer_epoch=1,
                sequence=sequence,
                lifecycle_scope_id="human-allocation",
                lifecycle_spec_hash=lifecycle_spec.semantic_content_hash,
                phase=phase,
                effective_for_target_at=9.9,
                observed_at=9.9,
                observed_clock_domain_id="control.monotonic",
                anchor_request=request,
                anchor_receipt=anchor_receipt,
            ),
            spec=lifecycle_spec,
            activation_scope=activation,
        )
        assert lifecycle_admission.accepted, lifecycle_admission
    return runtime, clock, activation, receipt


def test_optional_allocation_context_uses_only_real_baseline_input() -> None:
    output_schema, delta_schema = _schemas()
    allocation_spec = _allocation_spec()
    contracts = build_allocated_composition_contracts(
        contract_id="optional-allocation-context",
        output_action_schema=output_schema,
        relative_action_schema=delta_schema,
        allocation_spec=allocation_spec,
        allocation_context_channel="source_allocation",
        required_roles=("base_absolute",),
        allow_missing_allocation_context=True,
    )
    runtime, _clock, activation, anchor_receipt = _runtime(
        output_schema,
        delta_schema,
        contracts.strategy,
    )
    baseline = ComponentAction(
        action_schema=output_schema,
        values_by_component={
            "arm": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            "hand": (0.2, 0.3),
        },
    )
    assert runtime.admit_composition_input(
        _input_draft(
            "base_absolute",
            baseline,
            anchor_receipt,
            activation,
        ),
        activation_scope=activation,
    ).accepted

    submission = runtime.evaluate_composition(
        spec=contracts.composition_spec,
        target_effective_at=10.05,
        activation_scope=activation,
        residual_anchor_lifecycle_scope_id="human-allocation",
    )
    evaluation = submission.composition_evaluation

    assert evaluation.output_status == "ready"
    assert evaluation.composed_candidate.semantic_content_hash == (
        baseline.semantic_content_hash
    )
    assert tuple(submission.input_snapshot.selected_input_by_role) == (
        "base_absolute",
    )
    assert evaluation.strategy_result_data[
        "allocation_selected_mode_by_component"
    ] == {"arm": "base_only", "hand": "base_only"}
    assert evaluation.strategy_used_context_ids == ()


def test_pending_anchor_lifecycle_blocks_an_otherwise_ready_allocation() -> None:
    output_schema, delta_schema, allocation_spec, spec, strategy = _contract()
    runtime, clock, activation, anchor_receipt = _runtime(
        output_schema,
        delta_schema,
        strategy,
        lifecycle_ready=False,
    )
    for role, schema in (
        ("base_absolute", output_schema),
        ("operator_relative", delta_schema),
        ("operator_absolute", output_schema),
    ):
        action = ComponentAction(
            action_schema=schema,
            values_by_component={
                "arm": (
                    0.1 if role == "operator_relative" else 1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ),
                "hand": (
                    (0.1, 0.1)
                    if role == "operator_relative"
                    else (0.2, 0.2)
                ),
            },
        )
        assert runtime.admit_composition_input(
            _input_draft(
                role,
                action,
                anchor_receipt,
                activation,
            ),
            activation_scope=activation,
        ).accepted
    _admit_allocation_context(
        runtime,
        clock,
        empty_composition_allocation_state(
            "allocation-reducer",
            spec=allocation_spec,
        ),
        allocation_spec,
        activation,
        anchor_receipt,
        level=1.5,
        at=10.0,
        sequence=0,
    )

    result = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.05,
        required_context_channels=("source_allocation",),
        activation_scope=activation,
        residual_anchor_lifecycle_scope_id="human-allocation",
    ).composition_evaluation
    assert result.output_status == "no_action"
    assert result.composed_candidate is None
    assert "residual_anchor_rebase_pending" in result.reason


def _input_draft(
    role: str,
    action: ComponentAction,
    anchor_receipt: ResidualAnchorReceipt,
    activation,
) -> CompositionInputDraft:
    anchor_ids = (
        {}
        if role == "base_absolute"
        else {"residual_anchor": anchor_receipt.receipt_id}
    )
    anchor_hashes = (
        {}
        if role == "base_absolute"
        else {"residual_anchor": anchor_receipt.semantic_content_hash}
    )
    return CompositionInputDraft(
        runtime_session_id="session",
        control_domain_id="robot",
        producer_id=role,
        producer_epoch=1,
        sequence=0,
        input_role=role,
        action=action,
        produced_at=9.9,
        produced_clock_domain_id=f"{role}.monotonic",
        effective_from=10.0,
        effective_until=11.0,
        effective_clock_domain_id="control.monotonic",
        semantic_binding_id_by_kind=anchor_ids,
        semantic_binding_hash_by_kind=anchor_hashes,
        composition_activation_id=activation.activation_id,
        composition_activation_hash=activation.semantic_content_hash,
    )


def _activity(component_id: str, *, level: float, at: float):
    return ComponentSourceActivityObservation(
        source_role="operator_activity",
        metric_provider_id="test.activity.v1",
        metric_provider_hash="test-activity-provider-hash:v1",
        component_id=component_id,
        representation_family="test.normalized_activity",
        window_started_at=at - 0.2,
        window_ended_at=at - 0.05,
        available_at=at - 0.01,
        valid_until=at + 0.3,
        clock_domain_id="control.monotonic",
        metric_values={"motion": level},
        normalization_scale_by_metric={"motion": 1.0},
        source_fact_ids=(f"activity:{component_id}:{at}",),
        source_fact_hashes=(f"activity-hash:{component_id}:{at}:{level}",),
    )


def _magnitude(
    component_id: str,
    *,
    level: float,
    at: float,
    anchor_receipt: ResidualAnchorReceipt | None = None,
):
    receipt_id = (
        "residual-anchor-receipt:0"
        if anchor_receipt is None
        else anchor_receipt.receipt_id
    )
    receipt_hash = (
        "residual-anchor-receipt-hash:0"
        if anchor_receipt is None
        else anchor_receipt.semantic_content_hash
    )
    receipt_generation = (
        0 if anchor_receipt is None else anchor_receipt.request_generation
    )
    return ComponentResidualMagnitudeObservation(
        source_role="operator_relative",
        metric_provider_id="test.residual-magnitude.v1",
        metric_provider_hash="test-residual-magnitude-provider-hash:v1",
        component_id=component_id,
        representation_family="test.normalized_residual",
        sampled_at=at - 0.02,
        available_at=at - 0.01,
        valid_until=at + 0.3,
        clock_domain_id="control.monotonic",
        anchor_receipt_id=receipt_id,
        anchor_receipt_hash=receipt_hash,
        anchor_request_generation=receipt_generation,
        residual_action_id=f"residual:{component_id}:{at}",
        residual_action_hash=f"residual-hash:{component_id}:{at}:{level}",
        metric_values={"magnitude": level},
        normalization_scale_by_metric={"magnitude": 1.0},
    )


def _admit_allocation_context(
    runtime,
    clock,
    allocation_state,
    allocation_spec,
    activation,
    anchor_receipt,
    *,
    level: float,
    at: float,
    sequence: int,
):
    evaluation = evaluate_composition_allocation(
        allocation_state,
        spec=allocation_spec,
        activity_by_component={
            component_id: _activity(component_id, level=level, at=at)
            for component_id in allocation_spec.component_ids
        },
        residual_magnitude_by_component={
            component_id: _magnitude(
                component_id,
                level=level,
                at=at,
                anchor_receipt=anchor_receipt,
            )
            for component_id in allocation_spec.component_ids
        },
        evaluated_at=at,
        clock_domain_id="control.monotonic",
    )
    clock.value = at
    receipt = runtime.admit_composition_context(
        build_composition_allocation_context_draft(
            evaluation.decision,
            runtime_session_id="session",
            control_domain_id="robot",
            producer_id="allocation_reducer",
            producer_epoch=1,
            sequence=sequence,
            context_channel="source_allocation",
            produced_at=at,
            produced_clock_domain_id="control.monotonic",
            effective_from=at,
            effective_clock_domain_id="control.monotonic",
            allocation_scope_id="human-allocation",
            activity_basis_id=f"allocation-basis:{sequence}",
            activity_basis_hash=f"allocation-basis-hash:{sequence}",
            control_cut_hash=f"allocation-cut-hash:{sequence}",
            activation_scope=activation,
        )
    )
    assert receipt.accepted, receipt
    return evaluation.state


def test_projected_allocation_uses_only_context_from_its_frozen_cut() -> None:
    output_schema, delta_schema, allocation_spec, spec, strategy = _contract()
    runtime, clock, activation, anchor_receipt = _runtime(
        output_schema, delta_schema, strategy
    )
    actions = {
        "base_absolute": ComponentAction(
            action_schema=output_schema,
            values_by_component={
                "arm": (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
                "hand": (0.2, 0.4),
            },
        ),
        "operator_relative": ComponentAction(
            action_schema=delta_schema,
            values_by_component={
                "arm": (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                "hand": (0.05, -0.1),
            },
        ),
        "operator_absolute": ComponentAction(
            action_schema=output_schema,
            values_by_component={
                "arm": (9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 1.0),
                "hand": (0.9, 0.8),
            },
        ),
    }
    allocation_state = empty_composition_allocation_state(
        "allocation-reducer",
        spec=allocation_spec,
    )
    allocation_state = _admit_allocation_context(
        runtime,
        clock,
        allocation_state,
        allocation_spec,
        activation,
        anchor_receipt,
        level=0.0,
        at=10.0,
        sequence=0,
    )
    before = (
        runtime.contexts.retention_snapshot(),
        runtime.composition_inputs.retention_snapshot(),
    )
    capture = runtime.capture_non_closing_projection_cut(
        cutoff_at=10.0,
        include_residual_anchor_lifecycle=True,
    )
    cut = capture.cut
    epochs = capture.projections["producer_epochs"]
    basis_context = build_composition_context_snapshot(
        capture.projections["contexts"],
        producer_epochs=epochs,
        target_effective_at=10.05,
        required_channels=("source_allocation",),
    )
    frozen_contexts = CompositionContextProjection(
        control_domain_id=cut.control_domain_id,
        cutoff_at=cut.cutoff_at,
        cutoff_admission_sequence=cut.cutoff_admission_sequence,
        contexts=tuple(basis_context.selected_context_by_channel.values()),
    )
    basis_hash = frozen_control_cut_semantic_hash(cut)
    projected = tuple(
        ProjectedCompositionInput(
            projection_basis_hash=basis_hash,
            runtime_session_id=cut.runtime_session_id,
            control_domain_id=cut.control_domain_id,
            input_role=role,
            action=action,
            producer_id=role,
            producer_epoch=1,
            produced_at=9.9,
            produced_clock_domain_id=f"{role}.monotonic",
            effective_from=10.0,
            effective_until=11.0,
            effective_clock_domain_id=cut.clock_domain_id,
            projection_sequence=index,
            source_kind="virtual_frozen_source",
            source_fact_ids=(f"source:{role}:0",),
            source_fact_hashes=(f"source-hash:{role}:0",),
            semantic_binding_id_by_kind=(
                {}
                if role == "base_absolute"
                else {"residual_anchor": anchor_receipt.receipt_id}
            ),
            semantic_binding_hash_by_kind=(
                {}
                if role == "base_absolute"
                else {
                    "residual_anchor": anchor_receipt.semantic_content_hash
                }
            ),
            composition_activation_id=activation.activation_id,
            composition_activation_hash=activation.semantic_content_hash,
        )
        for index, (role, action) in enumerate(actions.items())
    )

    def evaluate_at(target: float):
        input_snapshot = build_projected_composition_input_snapshot(
            projected,
            projection_basis_hash=basis_hash,
            control_cut=cut,
            producer_epochs=epochs,
            spec=spec,
            target_effective_at=target,
            activation_scope=activation,
            temporal_selection_spec=runtime.temporal_selection_spec,
        )
        context_snapshot = build_composition_context_snapshot(
            frozen_contexts,
            producer_epochs=epochs,
            target_effective_at=target,
            required_channels=("source_allocation",),
        )
        strategy_input = ProjectedCompositionStrategyInput(
            runtime_session_id=cut.runtime_session_id,
            control_domain_id=cut.control_domain_id,
            clock_domain_id=cut.clock_domain_id,
            strategy_spec=strategy.spec,
            composition_spec=spec,
            input_snapshot=input_snapshot,
            context_snapshot=context_snapshot,
            information_plan=runtime.information_plan,
            residual_anchor_lifecycle_snapshot=(
                build_residual_anchor_lifecycle_snapshot(
                    capture.projections["residual_anchor_lifecycle"],
                    producer_epochs=epochs,
                    spec=strategy.residual_anchor_lifecycle_spec,
                    activation_scope=activation,
                    lifecycle_scope_id="human-allocation",
                    target_effective_at=target,
                )
            ),
        )
        return evaluate_projected_composition_strategy(
            strategy,
            strategy_input,
            support_limits=tuple(
                value.effective_until
                for value in context_snapshot.selected_context_by_channel.values()
            ),
        )

    first = evaluate_at(10.05)
    np.testing.assert_allclose(
        first.composed_candidate.value("arm").values[:3],
        (1.0, 2.0, 3.0),
    )

    _admit_allocation_context(
        runtime,
        clock,
        allocation_state,
        allocation_spec,
        activation,
        anchor_receipt,
        level=4.0,
        at=10.10,
        sequence=1,
    )
    replay = evaluate_at(10.15)
    np.testing.assert_allclose(
        replay.composed_candidate.value("arm").values[:3],
        (1.0, 2.0, 3.0),
    )
    assert runtime.contexts.retention_snapshot().closed_target_through is None
    assert runtime.composition_inputs.retention_snapshot() == before[1]


def _admit_component_allocation_context(
    runtime,
    clock,
    allocation_state,
    allocation_spec,
    activation,
    anchor_receipt,
    *,
    level_by_component: dict[str, float],
    at: float,
    sequence: int,
):
    evaluation = evaluate_composition_allocation(
        allocation_state,
        spec=allocation_spec,
        activity_by_component={
            component_id: _activity(
                component_id,
                level=level_by_component[component_id],
                at=at,
            )
            for component_id in allocation_spec.component_ids
        },
        residual_magnitude_by_component={
            component_id: _magnitude(
                component_id,
                level=level_by_component[component_id],
                at=at,
                anchor_receipt=anchor_receipt,
            )
            for component_id in allocation_spec.component_ids
        },
        evaluated_at=at,
        clock_domain_id="control.monotonic",
    )
    clock.value = at
    receipt = runtime.admit_composition_context(
        build_composition_allocation_context_draft(
            evaluation.decision,
            runtime_session_id="session",
            control_domain_id="robot",
            producer_id="allocation_reducer",
            producer_epoch=1,
            sequence=sequence,
            context_channel="source_allocation",
            produced_at=at,
            produced_clock_domain_id="control.monotonic",
            effective_from=at,
            effective_clock_domain_id="control.monotonic",
            allocation_scope_id="human-allocation",
            activity_basis_id=f"allocation-basis:{sequence}",
            activity_basis_hash=f"allocation-basis-hash:{sequence}",
            control_cut_hash=f"allocation-cut-hash:{sequence}",
            activation_scope=activation,
        )
    )
    assert receipt.accepted, receipt
    return evaluation.state


def test_formal_composer_consumes_base_relative_and_absolute_modes() -> None:
    output_schema, delta_schema, allocation_spec, spec, strategy = _contract()
    runtime, clock, activation, anchor_receipt = _runtime(
        output_schema, delta_schema, strategy
    )
    baseline = ComponentAction(
        action_schema=output_schema,
        values_by_component={
            "arm": (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
            "hand": (0.2, 0.4),
        },
    )
    relative = ComponentAction(
        action_schema=delta_schema,
        values_by_component={
            "arm": (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            "hand": (0.05, -0.1),
        },
    )
    absolute = ComponentAction(
        action_schema=output_schema,
        values_by_component={
            "arm": (9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 1.0),
            "hand": (0.9, 0.8),
        },
    )
    for role, action in (
        ("base_absolute", baseline),
        ("operator_relative", relative),
        ("operator_absolute", absolute),
    ):
        assert runtime.admit_composition_input(
            _input_draft(role, action, anchor_receipt, activation),
            activation_scope=activation,
        ).accepted

    allocation_state = empty_composition_allocation_state(
        "allocation-reducer",
        spec=allocation_spec,
    )
    allocation_state = _admit_allocation_context(
        runtime,
        clock,
        allocation_state,
        allocation_spec,
        activation,
        anchor_receipt,
        level=0.0,
        at=10.0,
        sequence=0,
    )
    base = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.05,
        required_context_channels=("source_allocation",),
        activation_scope=activation,
        residual_anchor_lifecycle_scope_id="human-allocation",
    ).composition_evaluation
    assert base.output_status == "ready"
    np.testing.assert_allclose(
        base.composed_candidate.value("arm").values[:3], (1, 2, 3)
    )
    np.testing.assert_allclose(base.composed_candidate.value("hand").values, (0.2, 0.4))

    allocation_state = _admit_allocation_context(
        runtime,
        clock,
        allocation_state,
        allocation_spec,
        activation,
        anchor_receipt,
        level=1.5,
        at=10.10,
        sequence=1,
    )
    relative_result = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.15,
        required_context_channels=("source_allocation",),
        activation_scope=activation,
        residual_anchor_lifecycle_scope_id="human-allocation",
    ).composition_evaluation
    assert relative_result.output_status == "ready"
    np.testing.assert_allclose(
        relative_result.composed_candidate.value("arm").values[:3],
        (1.1, 2.0, 3.0),
    )
    np.testing.assert_allclose(
        relative_result.composed_candidate.value("hand").values,
        (0.25, 0.3),
    )

    allocation_state = _admit_allocation_context(
        runtime,
        clock,
        allocation_state,
        allocation_spec,
        activation,
        anchor_receipt,
        level=4.0,
        at=10.20,
        sequence=2,
    )
    absolute_result = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.25,
        required_context_channels=("source_allocation",),
        activation_scope=activation,
        residual_anchor_lifecycle_scope_id="human-allocation",
    ).composition_evaluation
    assert absolute_result.output_status == "ready"
    np.testing.assert_allclose(
        absolute_result.composed_candidate.value("arm").values[:3],
        (9.0, 8.0, 7.0),
    )
    np.testing.assert_allclose(
        absolute_result.composed_candidate.value("hand").values,
        (0.9, 0.8),
    )


def test_continuity_enabled_strategy_aligns_absolute_entry_and_records_receipt() -> None:
    output_schema, delta_schema, allocation_spec, contracts = (
        _continuity_contract()
    )
    strategy = contracts.strategy
    runtime, clock, activation, anchor_receipt = _runtime(
        output_schema,
        delta_schema,
        strategy,
    )
    strategy.activate_allocation_transition_continuity(activation)
    actions = {
        "base_absolute": ComponentAction(
            action_schema=output_schema,
            values_by_component={
                "arm": (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
                "hand": (0.2, 0.4),
            },
        ),
        "operator_relative": ComponentAction(
            action_schema=delta_schema,
            values_by_component={
                "arm": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                "hand": (0.0, 0.0),
            },
        ),
        "operator_absolute": ComponentAction(
            action_schema=output_schema,
            values_by_component={
                "arm": (9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 1.0),
                "hand": (0.9, 0.8),
            },
        ),
    }
    for role, action in actions.items():
        assert runtime.admit_composition_input(
            _input_draft(role, action, anchor_receipt, activation),
            activation_scope=activation,
        ).accepted

    state = empty_composition_allocation_state(
        "allocation-reducer",
        spec=allocation_spec,
    )
    state = _admit_allocation_context(
        runtime,
        clock,
        state,
        allocation_spec,
        activation,
        anchor_receipt,
        level=0.0,
        at=10.0,
        sequence=0,
    )
    base_submission = runtime.evaluate_composition(
        spec=contracts.composition_spec,
        target_effective_at=10.05,
        required_context_channels=("source_allocation",),
        activation_scope=activation,
        residual_anchor_lifecycle_scope_id="human-allocation",
    )
    base = base_submission.composition_evaluation
    continuity_runtime = next(
        iter(strategy.allocation_transition_continuity_runtime_by_spec_hash.values())
    )
    assert continuity_runtime.state is not None
    assert continuity_runtime.state.revision == 0
    base_submission = runtime.admit_evaluated_composition(
        base_submission,
        runtime_session_id="session",
        producer_id="formal-composer",
        producer_epoch=1,
        sequence=0,
        effective_until=10.2,
    )
    assert base_submission.proposal_admission is not None
    assert base_submission.proposal_admission.accepted
    assert continuity_runtime.state is not None
    assert continuity_runtime.state.revision == 1

    state = _admit_allocation_context(
        runtime,
        clock,
        state,
        allocation_spec,
        activation,
        anchor_receipt,
        level=4.0,
        at=10.10,
        sequence=1,
    )
    takeover_submission = runtime.evaluate_composition(
        spec=contracts.composition_spec,
        target_effective_at=10.15,
        required_context_channels=("source_allocation",),
        activation_scope=activation,
        residual_anchor_lifecycle_scope_id="human-allocation",
    )
    takeover = takeover_submission.composition_evaluation
    np.testing.assert_allclose(
        takeover.composed_candidate.value("arm").values,
        base.composed_candidate.value("arm").values,
    )
    np.testing.assert_allclose(
        takeover.composed_candidate.value("hand").values,
        base.composed_candidate.value("hand").values,
    )
    receipt = takeover.strategy_result_data[
        "allocation_transition_continuity"
    ]
    assert receipt["transition_by_component"] == {
        "arm": "base_only->absolute_overlay_owner",
        "hand": "base_only->absolute_overlay_owner",
    }
    assert receipt["raw_candidate_hash"] != receipt["output_candidate_hash"]
    assert continuity_runtime.state is not None
    assert continuity_runtime.state.revision == 1
    takeover_submission = runtime.admit_evaluated_composition(
        takeover_submission,
        runtime_session_id="session",
        producer_id="formal-composer",
        producer_epoch=1,
        sequence=1,
        effective_until=10.3,
    )
    assert takeover_submission.proposal_admission is not None
    assert takeover_submission.proposal_admission.accepted
    assert continuity_runtime.state is not None
    assert continuity_runtime.state.revision == 2

    moved_absolute = ComponentAction(
        action_schema=output_schema,
        values_by_component={
            "arm": (10.0, 8.0, 7.0, 0.0, 0.0, 0.0, 1.0),
            "hand": (1.0, 0.5),
        },
    )
    moved_draft = replace(
        _input_draft(
            "operator_absolute",
            moved_absolute,
            anchor_receipt,
            activation,
        ),
        sequence=1,
        produced_at=10.16,
        effective_from=10.16,
    )
    clock.value = 10.16
    assert runtime.admit_composition_input(
        moved_draft,
        activation_scope=activation,
    ).accepted
    moved_submission = runtime.evaluate_composition(
        spec=contracts.composition_spec,
        target_effective_at=10.20,
        required_context_channels=("source_allocation",),
        activation_scope=activation,
        residual_anchor_lifecycle_scope_id="human-allocation",
    )
    moved = moved_submission.composition_evaluation
    np.testing.assert_allclose(
        moved.composed_candidate.value("arm").values[:3],
        (2.0, 2.0, 3.0),
    )
    np.testing.assert_allclose(
        moved.composed_candidate.value("hand").values,
        (0.3, 0.1),
    )
    # A ready shadow/evaluation is still only a preparation. If an authority
    # gate declines or never submits it, it cannot advance the handoff baseline.
    assert continuity_runtime.state is not None
    assert continuity_runtime.state.revision == 2

    replacement_submission = runtime.evaluate_composition(
        spec=contracts.composition_spec,
        target_effective_at=10.21,
        required_context_channels=("source_allocation",),
        activation_scope=activation,
        residual_anchor_lifecycle_scope_id="human-allocation",
    )
    with pytest.raises(ValueError, match="not the current preparation"):
        runtime.admit_evaluated_composition(
            moved_submission,
            runtime_session_id="session",
            producer_id="formal-composer",
            producer_epoch=1,
            sequence=2,
            effective_until=10.35,
        )
    assert replacement_submission.proposal_admission is None
    assert continuity_runtime.state is not None
    assert continuity_runtime.state.revision == 2


def test_relative_mode_rejects_input_bound_to_another_anchor() -> None:
    output_schema, delta_schema, allocation_spec, spec, strategy = _contract()
    runtime, clock, activation, anchor_receipt = _runtime(
        output_schema, delta_schema, strategy
    )
    actions = {
        "base_absolute": ComponentAction(
            action_schema=output_schema,
            values_by_component={
                "arm": (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
                "hand": (0.2, 0.4),
            },
        ),
        "operator_relative": ComponentAction(
            action_schema=delta_schema,
            values_by_component={
                "arm": (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                "hand": (0.05, -0.1),
            },
        ),
        "operator_absolute": ComponentAction(
            action_schema=output_schema,
            values_by_component={
                "arm": (9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 1.0),
                "hand": (0.9, 0.8),
            },
        ),
    }
    for role, action in actions.items():
        draft = _input_draft(role, action, anchor_receipt, activation)
        if role == "operator_relative":
            draft = replace(
                draft,
                semantic_binding_id_by_kind={
                    "residual_anchor": "residual-anchor-receipt:other"
                },
                semantic_binding_hash_by_kind={
                    "residual_anchor": "residual-anchor-receipt-hash:other"
                },
            )
        assert runtime.admit_composition_input(
            draft,
            activation_scope=activation,
        ).accepted
    state = empty_composition_allocation_state(
        "allocation-reducer",
        spec=allocation_spec,
    )
    _admit_allocation_context(
        runtime,
        clock,
        state,
        allocation_spec,
        activation,
        anchor_receipt,
        level=1.5,
        at=10.0,
        sequence=0,
    )

    result = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.05,
        required_context_channels=("source_allocation",),
        activation_scope=activation,
        residual_anchor_lifecycle_scope_id="human-allocation",
    ).composition_evaluation

    assert result.output_status == "no_action"
    assert result.composed_candidate is None
    assert "allocation_selected_input_anchor_mismatch" in result.reason


def test_formal_composer_selects_modes_independently_by_component() -> None:
    output_schema, delta_schema, allocation_spec, spec, strategy = _contract()
    runtime, clock, activation, anchor_receipt = _runtime(
        output_schema, delta_schema, strategy
    )
    baseline = ComponentAction(
        action_schema=output_schema,
        values_by_component={
            "arm": (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0),
            "hand": (0.2, 0.4),
        },
    )
    relative = ComponentAction(
        action_schema=delta_schema,
        values_by_component={
            "arm": (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            "hand": (0.05, -0.1),
        },
    )
    absolute = ComponentAction(
        action_schema=output_schema,
        values_by_component={
            "arm": (9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 1.0),
            "hand": (0.9, 0.8),
        },
    )
    for role, action in (
        ("base_absolute", baseline),
        ("operator_relative", relative),
        ("operator_absolute", absolute),
    ):
        assert runtime.admit_composition_input(
            _input_draft(role, action, anchor_receipt, activation),
            activation_scope=activation,
        ).accepted

    state = empty_composition_allocation_state(
        "allocation-reducer",
        spec=allocation_spec,
    )
    _admit_component_allocation_context(
        runtime,
        clock,
        state,
        allocation_spec,
        activation,
        anchor_receipt,
        level_by_component={"arm": 4.0, "hand": 1.5},
        at=10.0,
        sequence=0,
    )

    result = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.05,
        required_context_channels=("source_allocation",),
        activation_scope=activation,
        residual_anchor_lifecycle_scope_id="human-allocation",
    ).composition_evaluation

    assert result.output_status == "ready"
    np.testing.assert_allclose(
        result.composed_candidate.value("arm").values[:3],
        (9.0, 8.0, 7.0),
    )
    np.testing.assert_allclose(
        result.composed_candidate.value("hand").values,
        (0.25, 0.3),
    )


def test_expired_allocation_context_returns_no_action_without_fallback() -> None:
    output_schema, delta_schema, allocation_spec, spec, strategy = _contract()
    runtime, clock, activation, anchor_receipt = _runtime(
        output_schema, delta_schema, strategy
    )
    for role, schema in (
        ("base_absolute", output_schema),
        ("operator_relative", delta_schema),
        ("operator_absolute", output_schema),
    ):
        action = ComponentAction(
            action_schema=schema,
            values_by_component={
                "arm": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                "hand": (0.0, 0.0),
            },
        )
        assert runtime.admit_composition_input(
            _input_draft(role, action, anchor_receipt, activation),
            activation_scope=activation,
        ).accepted
    state = empty_composition_allocation_state(
        "allocation-reducer",
        spec=allocation_spec,
    )
    _admit_allocation_context(
        runtime,
        clock,
        state,
        allocation_spec,
        activation,
        anchor_receipt,
        level=0.0,
        at=10.0,
        sequence=0,
    )

    result = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.31,
        required_context_channels=("source_allocation",),
        activation_scope=activation,
        residual_anchor_lifecycle_scope_id="human-allocation",
    ).composition_evaluation

    assert result.output_status == "no_action"
    assert result.composed_candidate is None
    assert "missing_context" in result.reason


def test_allocation_context_cannot_be_backdated_before_its_decision() -> None:
    allocation_spec = _allocation_spec()
    state = empty_composition_allocation_state(
        "allocation-reducer",
        spec=allocation_spec,
    )
    evaluation = evaluate_composition_allocation(
        state,
        spec=allocation_spec,
        activity_by_component={
            component_id: _activity(component_id, level=0.0, at=10.0)
            for component_id in allocation_spec.component_ids
        },
        residual_magnitude_by_component={
            component_id: _magnitude(component_id, level=0.0, at=10.0)
            for component_id in allocation_spec.component_ids
        },
        evaluated_at=10.0,
        clock_domain_id="control.monotonic",
    )

    with pytest.raises(
        ValueError,
        match="effective_from cannot predate its decision",
    ):
        build_composition_allocation_context_draft(
            evaluation.decision,
            runtime_session_id="session",
            control_domain_id="robot",
            producer_id="allocation_reducer",
            producer_epoch=1,
            sequence=0,
            context_channel="source_allocation",
            produced_at=10.0,
            produced_clock_domain_id="control.monotonic",
            effective_from=9.99,
            effective_clock_domain_id="control.monotonic",
        )
