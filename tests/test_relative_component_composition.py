from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from control_composition.composition import (
    FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
    ComponentCompositionRuleSet,
    CompositionInputDraft,
    CompositionSpec,
    FormalComponentCompositionStrategyV2,
    FormalCompositionStrategySpec,
    RelativePoseComponentCompositionRule,
    RelativeQposComponentCompositionRule,
    build_relative_baseline_residual_contracts,
)
from control_composition.events import CompositionDomainEvent
from control_composition.temporal_sampling import (
    GuaranteedNextComponentAction,
    sample_component_action_source,
)
from control_composition.retention import TimeWindowRetentionSpec
from control_composition.runtime import ControlCompositionRuntime
from dataset_interface.composition_control_trace import (
    project_composition_input_snapshot,
    project_composition_spec,
    project_composition_domain_event,
)
from dataset_contracts.completeness import (
    evaluate_control_trace_lineage,
    trace_requirement_profile,
)
from robot_motion.action import (
    ActionProposalDraft,
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentAction,
    ComponentActionSchema,
    ControlComponentSpecV2,
    ProposalAdmissionReceipt,
)
from robot_motion.action.authoritative_trace import AuthoritativeTraceContext
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry
from teleop_runtime.action.human_residual import (
    HumanPoseSample,
    HumanResidualPoseAdapter,
)
from teleop_runtime.action.human_joint_residual import (
    HumanJointPositionResidualAdapter,
    HumanJointPositionSample,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


class Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def _payloads() -> tuple[
    dict[str, ActionPayloadSchema],
    dict[str, ActionPayloadSchema],
]:
    common = {
        "schema_version": "development",
        "dtype": "float64",
    }
    absolute = {
        "arm": ActionPayloadSchema(
            schema_id="arm-absolute",
            representation_family="cartesian_pose",
            shape=(7,),
            coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
            units=("m", "m", "m", "1", "1", "1", "1"),
            value_semantics="absolute_target",
            reference_frame="world",
            controlled_frame="tool",
            **common,
        ),
        "hand": ActionPayloadSchema(
            schema_id="hand-absolute",
            representation_family="joint_position",
            shape=(2,),
            coordinate_names=("finger_0", "finger_1"),
            units=("rad", "rad"),
            value_semantics="absolute_target",
            **common,
        ),
    }
    delta = {
        "arm": ActionPayloadSchema(
            schema_id="arm-world-translation-local-rotation-delta",
            representation_family="cartesian_pose",
            shape=(7,),
            coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
            units=("m", "m", "m", "1", "1", "1", "1"),
            value_semantics="relative_delta",
            reference_frame="world",
            controlled_frame="tool",
            semantic_parameters={
                "translation_application": "reference_frame_additive",
                "rotation_application": "baseline_local_right_multiply",
            },
            **common,
        ),
        "hand": ActionPayloadSchema(
            schema_id="hand-additive-delta",
            representation_family="joint_position",
            shape=(2,),
            coordinate_names=("finger_0", "finger_1"),
            units=("rad", "rad"),
            value_semantics="relative_delta",
            semantic_parameters={"application": "elementwise_additive"},
            **common,
        ),
    }
    return absolute, delta


def _schemas() -> tuple[
    ComponentActionSchema,
    ComponentActionSchema,
    dict[str, ActionPayloadSchema],
    dict[str, ActionPayloadSchema],
]:
    absolute, delta = _payloads()
    return _schemas_from_payloads(absolute=absolute, delta=delta)


def _schemas_from_payloads(
    *,
    absolute: dict[str, ActionPayloadSchema],
    delta: dict[str, ActionPayloadSchema],
) -> tuple[
    ComponentActionSchema,
    ComponentActionSchema,
    dict[str, ActionPayloadSchema],
    dict[str, ActionPayloadSchema],
]:
    catalog = ActionComponentCatalogV2(
        catalog_id="relative-composition-fixture",
        catalog_version="development",
        components=tuple(
            ControlComponentSpecV2(
                component_id=component_id,
                controlled_entity_id=component_id,
                command_channel_id=component_id,
                selection_group_id=component_id,
                resource_claims=(f"resource:{component_id}",),
                accepted_payload_schema_hashes=(
                    absolute[component_id].semantic_content_hash,
                    delta[component_id].semantic_content_hash,
                ),
            )
            for component_id in ("arm", "hand")
        ),
    )
    return (
        ComponentActionSchema(
            schema_id="relative-composition-absolute-output",
            schema_version="development",
            component_catalog=catalog,
            payload_schema_by_component=absolute,
        ),
        ComponentActionSchema(
            schema_id="relative-composition-delta-input",
            schema_version="development",
            component_catalog=catalog,
            payload_schema_by_component=delta,
        ),
        absolute,
        delta,
    )


def _contract(
    *,
    required_roles: tuple[str, ...] = ("policy", "human"),
    schema_bundle: tuple[
        ComponentActionSchema,
        ComponentActionSchema,
        dict[str, ActionPayloadSchema],
        dict[str, ActionPayloadSchema],
    ]
    | None = None,
    qpos_delta_contract_hash: str | None = None,
):
    output_schema, delta_schema, absolute, delta = (
        _schemas() if schema_bundle is None else schema_bundle
    )
    rules = ComponentCompositionRuleSet(
        rule_set_id="follow-policy-with-human-delta-v1",
        action_space="component-keyed",
        rules=(
            RelativePoseComponentCompositionRule(
                component_id="arm",
                baseline_role="policy",
                delta_role="human",
            ),
            RelativeQposComponentCompositionRule(
                component_id="hand",
                baseline_role="policy",
                delta_role="human",
                baseline_value_contract_hash=(
                    absolute["hand"].semantic_content_hash
                ),
                delta_value_contract_hash=(
                    delta["hand"].semantic_content_hash
                    if qpos_delta_contract_hash is None
                    else qpos_delta_contract_hash
                ),
            ),
        ),
    )
    spec = CompositionSpec(
        spec_id="follow-policy-with-human-delta-v1",
        algebra=FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
        input_roles=("policy", "human"),
        required_roles=required_roles,
        depends_on_roles_by_role={"human": ("policy",)},
        component_rule_set_hash=rules.semantic_content_hash,
        action_schema_hash=output_schema.semantic_content_hash,
        input_action_schema_hash_by_role={
            "policy": output_schema.semantic_content_hash,
            "human": delta_schema.semantic_content_hash,
        },
    )
    strategy_spec = FormalCompositionStrategySpec(
        strategy_id="follow-policy-with-human-delta-v1",
        strategy_version="development",
        composition_spec_hash=spec.semantic_content_hash,
        implementation_contract_hashes=(rules.semantic_content_hash,),
        maximum_decision_latency_sec=0.01,
    )
    strategy = FormalComponentCompositionStrategyV2(
        strategy_spec=strategy_spec,
        composition_spec=spec,
        component_rule_set=rules,
    )
    return output_schema, delta_schema, spec, strategy


def _runtime(
    output_schema: ComponentActionSchema,
    delta_schema: ComponentActionSchema,
    strategy: FormalComponentCompositionStrategyV2,
    captured_proposals: list[ActionProposalDraft] | None = None,
    captured_events: list[CompositionDomainEvent] | None = None,
    *,
    baseline_role: str = "policy",
    residual_role: str = "human",
) -> ControlCompositionRuntime:
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
    for producer_id in (baseline_role, residual_role):
        epochs.activate(
            producer_id=producer_id,
            producer_epoch=1,
            produced_at=9.9,
            produced_clock_domain_id=f"{producer_id}.monotonic",
            reason="startup",
        )
    retention = TimeWindowRetentionSpec(
        spec_id="relative-composition-window",
        history_window_sec=0.0,
        max_future_lead_sec=2.0,
        max_effective_duration_sec=2.0,
    )
    def admit_proposal(
        draft: ActionProposalDraft,
        **_kwargs,
    ) -> ProposalAdmissionReceipt:
        if captured_proposals is not None:
            captured_proposals.append(draft)
        return ProposalAdmissionReceipt(accepted=True)

    return ControlCompositionRuntime(
        runtime_session_id="session",
        control_domain_id="robot",
        embodiment_instance_id="virtual",
        coordinator=coordinator,
        producer_epochs=epochs,
        action_schema=output_schema,
        input_action_schema_by_role={
            baseline_role: output_schema,
            residual_role: delta_schema,
        },
        event_emitter=(
            (lambda _events: None)
            if captured_events is None
            else captured_events.extend
        ),
        proposal_admitter=admit_proposal,
        context_retention_spec=retention,
        composition_input_retention_spec=retention,
        composition_strategy=strategy,
        clock=clock,
    )


def _draft(
    *,
    role: str,
    sequence: int,
    action: ComponentAction,
    effective_from: float = 10.0,
    effective_until: float = 11.0,
) -> CompositionInputDraft:
    return CompositionInputDraft(
        runtime_session_id="session",
        control_domain_id="robot",
        producer_id=role,
        producer_epoch=1,
        sequence=sequence,
        input_role=role,
        action=action,
        produced_at=9.9,
        produced_clock_domain_id=f"{role}.monotonic",
        effective_from=effective_from,
        effective_until=effective_until,
        effective_clock_domain_id="control.monotonic",
    )


def test_follow_baseline_applies_world_translation_local_rotation_and_qpos_delta() -> None:
    output_schema, delta_schema, spec, strategy = _contract()
    runtime = _runtime(output_schema, delta_schema, strategy)
    baseline_rotation = Rotation.from_euler("z", 90.0, degrees=True)
    delta_rotation = Rotation.from_euler("x", 90.0, degrees=True)
    baseline = ComponentAction(
        action_schema=output_schema,
        values_by_component={
            "arm": np.r_[[1.0, 2.0, 3.0], baseline_rotation.as_quat()],
            "hand": [0.2, 0.4],
        },
    )
    delta = ComponentAction(
        action_schema=delta_schema,
        values_by_component={
            "arm": np.r_[[0.1, 0.0, 0.0], delta_rotation.as_quat()],
            "hand": [0.05, -0.1],
        },
    )
    assert runtime.admit_composition_input(
        _draft(role="policy", sequence=0, action=baseline)
    ).accepted
    assert runtime.admit_composition_input(
        _draft(role="human", sequence=0, action=delta)
    ).accepted

    first_submission = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    )
    first = first_submission.composition_evaluation
    assert first.output_status == "ready"
    assert first.composed_candidate is not None
    np.testing.assert_allclose(
        first.composed_candidate.value("arm").values[:3],
        [1.1, 2.0, 3.0],
    )
    expected_rotation = baseline_rotation * delta_rotation
    actual_rotation = Rotation.from_quat(
        first.composed_candidate.value("arm").values[3:7]
    )
    assert (expected_rotation.inv() * actual_rotation).magnitude() < 1e-10
    np.testing.assert_allclose(
        first.composed_candidate.value("hand").values,
        [0.25, 0.3],
    )
    assert first.composition_supported_until == 11.0
    trace_context = AuthoritativeTraceContext(
        runtime_session_id="session",
        control_domain_id="robot",
        embodiment_instance_id="virtual",
        clock_domain_id="control.monotonic",
    )
    spec_fact = project_composition_spec(
        trace_context,
        spec,
        occurred_at=10.0,
    )
    snapshot_fact = project_composition_input_snapshot(
        trace_context,
        first_submission.input_snapshot,
    )
    assert spec_fact.payload["action_schema_hash"] == (
        output_schema.semantic_content_hash
    )
    assert dict(spec_fact.payload["input_action_schema_hash_by_role"]) == dict(
        spec.input_action_schema_hash_by_role
    )
    assert dict(
        snapshot_fact.payload["input_action_schema_hash_by_role"]
    ) == dict(spec.input_action_schema_hash_by_role)

    second = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.2,
    ).composition_evaluation
    np.testing.assert_allclose(
        second.composed_candidate.value("arm").values,
        first.composed_candidate.value("arm").values,
    )
    np.testing.assert_allclose(
        second.composed_candidate.value("hand").values,
        first.composed_candidate.value("hand").values,
    )
    assert second.composition_supported_until == first.composition_supported_until


def test_follow_baseline_reapplies_held_delta_to_new_baseline_without_integration() -> None:
    output_schema, delta_schema, spec, strategy = _contract()
    runtime = _runtime(output_schema, delta_schema, strategy)
    delta = ComponentAction(
        action_schema=delta_schema,
        values_by_component={
            "arm": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "hand": [0.05, -0.1],
        },
    )
    first_baseline = ComponentAction(
        action_schema=output_schema,
        values_by_component={
            "arm": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "hand": [0.2, 0.4],
        },
    )
    next_baseline = ComponentAction(
        action_schema=output_schema,
        values_by_component={
            "arm": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "hand": [0.3, 0.5],
        },
    )
    assert runtime.admit_composition_input(
        _draft(role="policy", sequence=0, action=first_baseline)
    ).accepted
    assert runtime.admit_composition_input(
        _draft(role="human", sequence=0, action=delta)
    ).accepted
    first = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    ).composition_evaluation
    assert runtime.admit_composition_input(
        _draft(
            role="policy",
            sequence=1,
            action=next_baseline,
            effective_from=10.15,
        )
    ).accepted
    second = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.2,
    ).composition_evaluation

    np.testing.assert_allclose(
        first.composed_candidate.value("arm").values[:3],
        [1.1, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        second.composed_candidate.value("arm").values[:3],
        [2.1, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        second.composed_candidate.value("hand").values,
        [0.35, 0.4],
    )


def test_optional_delta_uses_baseline_but_dependency_never_uses_delta_alone() -> None:
    output_schema, delta_schema, spec, strategy = _contract(
        required_roles=("policy",)
    )
    runtime = _runtime(output_schema, delta_schema, strategy)
    baseline = ComponentAction(
        action_schema=output_schema,
        values_by_component={
            "arm": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "hand": [0.2, 0.4],
        },
    )
    assert runtime.admit_composition_input(
        _draft(role="policy", sequence=0, action=baseline)
    ).accepted
    result = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    ).composition_evaluation
    assert result.output_status == "ready"
    np.testing.assert_allclose(
        result.composed_candidate.value("arm").values,
        baseline.value("arm").values,
    )
    np.testing.assert_allclose(
        result.composed_candidate.value("hand").values,
        baseline.value("hand").values,
    )

    with pytest.raises(
        ValueError,
        match="required role may depend only on required roles",
    ):
        CompositionSpec(
            spec_id="invalid-required-delta",
            algebra=FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
            input_roles=("policy", "human"),
            required_roles=("human",),
            depends_on_roles_by_role={"human": ("policy",)},
            component_rule_set_hash=strategy.component_rule_set.semantic_content_hash,
            action_schema_hash=output_schema.semantic_content_hash,
            input_action_schema_hash_by_role={
                "policy": output_schema.semantic_content_hash,
                "human": delta_schema.semantic_content_hash,
            },
        )


def test_relative_delta_uses_only_an_explicitly_guaranteed_next_step() -> None:
    _, delta_schema, _, _ = _contract()
    current = ComponentAction(
        action_schema=delta_schema,
        values_by_component={
            "arm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "hand": [0.0, 0.0],
        },
    )
    next_action = ComponentAction(
        action_schema=delta_schema,
        values_by_component={
            "arm": [
                0.2,
                0.0,
                0.0,
                *Rotation.from_euler("z", 90.0, degrees=True).as_quat(),
            ],
            "hand": [0.2, -0.2],
        },
    )
    fallback = sample_component_action_source(
        current_action=current,
        current_target_effective_at=10.0,
        current_source_fact_id="human-step-0",
        current_source_fact_hash="human-step-0-hash",
        target_effective_at=10.5,
        clock_domain_id="control.monotonic",
        guaranteed_next_step=None,
    )
    assert fallback.action is current
    assert fallback.derivation_kind == "current_step_fallback"

    interpolated = sample_component_action_source(
        current_action=current,
        current_target_effective_at=10.0,
        current_source_fact_id="human-step-0",
        current_source_fact_hash="human-step-0-hash",
        target_effective_at=10.5,
        clock_domain_id="control.monotonic",
        guaranteed_next_step=GuaranteedNextComponentAction(
            guarantee_id="human-step-1-guaranteed",
            source_fact_id="human-step-1",
            source_fact_hash="human-step-1-hash",
            action=next_action,
            target_effective_at=11.0,
            clock_domain_id="control.monotonic",
        ),
    )
    assert interpolated.derivation_kind == "interpolated_guaranteed_next"
    np.testing.assert_allclose(
        interpolated.action.value("arm").values[:3],
        [0.1, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        interpolated.action.value("hand").values,
        [0.1, -0.1],
    )
    expected = Rotation.from_euler("z", 45.0, degrees=True)
    actual = Rotation.from_quat(
        interpolated.action.value("arm").values[3:7]
    )
    assert (expected.inv() * actual).magnitude() < 1e-10


def test_relative_result_promotes_only_the_absolute_output_schema() -> None:
    output_schema, delta_schema, spec, strategy = _contract()
    captured: list[ActionProposalDraft] = []
    runtime = _runtime(
        output_schema,
        delta_schema,
        strategy,
        captured_proposals=captured,
    )
    assert runtime.admit_composition_input(
        _draft(
            role="policy",
            sequence=0,
            action=ComponentAction(
                action_schema=output_schema,
                values_by_component={
                    "arm": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "hand": [0.2, 0.4],
                },
            ),
        )
    ).accepted
    assert runtime.admit_composition_input(
        _draft(
            role="human",
            sequence=0,
            action=ComponentAction(
                action_schema=delta_schema,
                values_by_component={
                    "arm": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "hand": [0.05, -0.1],
                },
            ),
        )
    ).accepted
    evaluation = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    )
    promoted = runtime.admit_evaluated_composition(
        evaluation,
        runtime_session_id="session",
        producer_id="composer",
        producer_epoch=1,
        sequence=0,
        effective_until=10.5,
    )
    assert promoted.proposal_admission is not None
    assert promoted.proposal_admission.accepted
    assert len(captured) == 1
    assert captured[0].action.action_schema.semantic_content_hash == (
        output_schema.semantic_content_hash
    )
    assert captured[0].action.action_schema.semantic_content_hash != (
        delta_schema.semantic_content_hash
    )


def test_role_schema_swap_is_rejected_before_relative_evaluation() -> None:
    output_schema, delta_schema, _, strategy = _contract()
    runtime = _runtime(output_schema, delta_schema, strategy)
    absolute = ComponentAction(
        action_schema=output_schema,
        values_by_component={
            "arm": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "hand": [0.2, 0.4],
        },
    )
    delta = ComponentAction(
        action_schema=delta_schema,
        values_by_component={
            "arm": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "hand": [0.05, -0.1],
        },
    )

    policy_receipt = runtime.admit_composition_input(
        _draft(role="policy", sequence=0, action=delta)
    )
    human_receipt = runtime.admit_composition_input(
        _draft(role="human", sequence=0, action=absolute)
    )
    assert not policy_receipt.accepted
    assert policy_receipt.rejection_code == "action_schema_mismatch"
    assert not human_receipt.accepted
    assert human_receipt.rejection_code == "action_schema_mismatch"


def test_relative_pose_frame_mismatch_is_invalid_and_cannot_publish() -> None:
    absolute, delta = _payloads()
    delta["arm"] = ActionPayloadSchema(
        schema_id="arm-wrong-frame-delta",
        schema_version="development",
        representation_family="cartesian_pose",
        dtype="float64",
        shape=(7,),
        coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
        units=("m", "m", "m", "1", "1", "1", "1"),
        value_semantics="relative_delta",
        reference_frame="another_world",
        controlled_frame="tool",
    )
    bundle = _schemas_from_payloads(absolute=absolute, delta=delta)
    output_schema, delta_schema, spec, strategy = _contract(
        schema_bundle=bundle
    )
    captured: list[ActionProposalDraft] = []
    runtime = _runtime(
        output_schema,
        delta_schema,
        strategy,
        captured_proposals=captured,
    )
    assert runtime.admit_composition_input(
        _draft(
            role="policy",
            sequence=0,
            action=ComponentAction(
                action_schema=output_schema,
                values_by_component={
                    "arm": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "hand": [0.2, 0.4],
                },
            ),
        )
    ).accepted
    assert runtime.admit_composition_input(
        _draft(
            role="human",
            sequence=0,
            action=ComponentAction(
                action_schema=delta_schema,
                values_by_component={
                    "arm": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "hand": [0.05, -0.1],
                },
            ),
        )
    ).accepted

    evaluation = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    )
    assert evaluation.composition_evaluation.output_status == "invalid"
    assert evaluation.composition_evaluation.reason == (
        "component_invalid:arm:relative_pose_frame_mismatch"
    )
    assert evaluation.composition_evaluation.composed_candidate is None
    with pytest.raises(ValueError, match="Only a ready"):
        runtime.admit_evaluated_composition(
            evaluation,
            runtime_session_id="session",
            producer_id="composer",
            producer_epoch=1,
            sequence=0,
            effective_until=10.5,
        )
    assert captured == []


def test_relative_qpos_contract_mismatch_is_invalid() -> None:
    output_schema, delta_schema, spec, strategy = _contract(
        qpos_delta_contract_hash="f" * 64
    )
    runtime = _runtime(output_schema, delta_schema, strategy)
    assert runtime.admit_composition_input(
        _draft(
            role="policy",
            sequence=0,
            action=ComponentAction(
                action_schema=output_schema,
                values_by_component={
                    "arm": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "hand": [0.2, 0.4],
                },
            ),
        )
    ).accepted
    assert runtime.admit_composition_input(
        _draft(
            role="human",
            sequence=0,
            action=ComponentAction(
                action_schema=delta_schema,
                values_by_component={
                    "arm": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "hand": [0.05, -0.1],
                },
            ),
        )
    ).accepted

    result = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    ).composition_evaluation
    assert result.output_status == "invalid"
    assert result.reason == (
        "component_invalid:hand:relative_qpos_value_contract_mismatch"
    )
    assert result.composed_candidate is None


def test_required_delta_expiry_uses_half_open_ttl_and_returns_no_action() -> None:
    output_schema, delta_schema, spec, strategy = _contract()
    captured: list[ActionProposalDraft] = []
    runtime = _runtime(
        output_schema,
        delta_schema,
        strategy,
        captured_proposals=captured,
    )
    assert runtime.admit_composition_input(
        _draft(
            role="policy",
            sequence=0,
            action=ComponentAction(
                action_schema=output_schema,
                values_by_component={
                    "arm": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "hand": [0.2, 0.4],
                },
            ),
        )
    ).accepted
    assert runtime.admit_composition_input(
        _draft(
            role="human",
            sequence=0,
            action=ComponentAction(
                action_schema=delta_schema,
                values_by_component={
                    "arm": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "hand": [0.05, -0.1],
                },
            ),
            effective_until=10.2,
        )
    ).accepted

    evaluation = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.2,
    )
    result = evaluation.composition_evaluation
    assert result.output_status == "no_action"
    assert result.reason == "missing_required:human"
    assert result.composed_candidate is None
    assert result.composition_supported_until is None
    with pytest.raises(ValueError, match="Only a ready"):
        runtime.admit_evaluated_composition(
            evaluation,
            runtime_session_id="session",
            producer_id="composer",
            producer_epoch=1,
            sequence=0,
            effective_until=10.5,
        )
    assert captured == []


def test_missing_optional_delta_component_uses_only_that_baseline_component() -> None:
    output_schema, delta_schema, spec, strategy = _contract()
    runtime = _runtime(output_schema, delta_schema, strategy)
    baseline = ComponentAction(
        action_schema=output_schema,
        values_by_component={
            "arm": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "hand": [0.2, 0.4],
        },
    )
    arm_only_delta = ComponentAction(
        action_schema=delta_schema,
        values_by_component={
            "arm": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
    )
    assert runtime.admit_composition_input(
        _draft(role="policy", sequence=0, action=baseline)
    ).accepted
    assert runtime.admit_composition_input(
        _draft(role="human", sequence=0, action=arm_only_delta)
    ).accepted

    result = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    ).composition_evaluation
    assert result.output_status == "ready"
    np.testing.assert_allclose(
        result.composed_candidate.value("arm").values[:3],
        [1.1, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        result.composed_candidate.value("hand").values,
        baseline.value("hand").values,
    )


def test_delta_without_its_optional_baseline_is_removed_by_dependency_closure() -> None:
    output_schema, delta_schema, spec, strategy = _contract(
        required_roles=()
    )
    captured: list[ActionProposalDraft] = []
    runtime = _runtime(
        output_schema,
        delta_schema,
        strategy,
        captured_proposals=captured,
    )
    assert runtime.admit_composition_input(
        _draft(
            role="human",
            sequence=0,
            action=ComponentAction(
                action_schema=delta_schema,
                values_by_component={
                    "arm": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "hand": [0.05, -0.1],
                },
            ),
        )
    ).accepted

    evaluation = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    )
    snapshot = evaluation.input_snapshot
    result = evaluation.composition_evaluation
    assert "human" in snapshot.selected_input_before_dependency_by_role
    assert "human" not in snapshot.selected_input_by_role
    assert snapshot.dependency_ineligibility_reason_by_role["human"] == (
        "prerequisite_unavailable:policy"
    )
    assert result.output_status == "no_action"
    assert result.reason == "relative_baseline_missing:arm:policy"
    assert captured == []


@pytest.mark.parametrize(
    "bad_arm",
    (
        [0.0, 0.0, 0.0, math.nan, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
    ),
)
def test_invalid_relative_pose_values_are_rejected_at_carrier_boundary(
    bad_arm: list[float],
) -> None:
    _, delta_schema, _, _ = _contract()
    with pytest.raises(ValueError):
        ComponentAction(
            action_schema=delta_schema,
            values_by_component={
                "arm": bad_arm,
                "hand": [0.0, 0.0],
            },
        )


def test_equivalent_delta_quaternion_signs_produce_identical_pose_output() -> None:
    output_schema, delta_schema, spec, strategy = _contract()
    baseline = ComponentAction(
        action_schema=output_schema,
        values_by_component={
            "arm": [
                1.0,
                0.0,
                0.0,
                *Rotation.from_euler("y", 30.0, degrees=True).as_quat(),
            ],
            "hand": [0.2, 0.4],
        },
    )
    quaternion = Rotation.from_euler("x", 40.0, degrees=True).as_quat()
    outputs = []
    for index, sign in enumerate((1.0, -1.0)):
        runtime = _runtime(output_schema, delta_schema, strategy)
        assert runtime.admit_composition_input(
            _draft(role="policy", sequence=0, action=baseline)
        ).accepted
        assert runtime.admit_composition_input(
            _draft(
                role="human",
                sequence=0,
                action=ComponentAction(
                    action_schema=delta_schema,
                    values_by_component={
                        "arm": [
                            0.1,
                            0.0,
                            0.0,
                            *(sign * quaternion),
                        ],
                        "hand": [0.05, -0.1],
                    },
                ),
            )
        ).accepted
        result = runtime.evaluate_composition(
            spec=spec,
            target_effective_at=10.1 + 0.01 * index,
        ).composition_evaluation
        assert result.output_status == "ready"
        outputs.append(result.composed_candidate.value("arm").values)

    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-12)


def test_relative_actions_detach_mutable_source_arrays() -> None:
    output_schema, delta_schema, spec, strategy = _contract()
    runtime = _runtime(output_schema, delta_schema, strategy)
    baseline_values = np.asarray(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float64,
    )
    delta_values = np.asarray(
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float64,
    )
    baseline = ComponentAction(
        action_schema=output_schema,
        values_by_component={
            "arm": baseline_values,
            "hand": [0.2, 0.4],
        },
    )
    delta = ComponentAction(
        action_schema=delta_schema,
        values_by_component={
            "arm": delta_values,
            "hand": [0.05, -0.1],
        },
    )
    baseline_values[0] = 100.0
    delta_values[0] = 200.0
    assert runtime.admit_composition_input(
        _draft(role="policy", sequence=0, action=baseline)
    ).accepted
    assert runtime.admit_composition_input(
        _draft(role="human", sequence=0, action=delta)
    ).accepted

    result = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    ).composition_evaluation
    np.testing.assert_allclose(
        result.composed_candidate.value("arm").values[:3],
        [1.1, 0.0, 0.0],
    )


def test_seeded_virtual_policy_and_human_residual_sweep_matches_contract() -> None:
    """Exercise varying low-rate baselines and residuals without hardware."""

    output_schema, delta_schema, spec, strategy = _contract()
    runtime = _runtime(output_schema, delta_schema, strategy)
    rng = np.random.default_rng(20260728)

    for sequence in range(32):
        target = 10.01 + 0.02 * sequence
        baseline_translation = rng.uniform(-0.5, 0.5, size=3)
        delta_translation = rng.uniform(-0.05, 0.05, size=3)
        baseline_rotation = Rotation.from_rotvec(
            rng.uniform(-1.0, 1.0, size=3)
        )
        delta_rotation = Rotation.from_rotvec(
            rng.uniform(-0.25, 0.25, size=3)
        )
        baseline_qpos = rng.uniform(-0.8, 0.8, size=2)
        delta_qpos = rng.uniform(-0.1, 0.1, size=2)
        baseline = ComponentAction(
            action_schema=output_schema,
            values_by_component={
                "arm": np.r_[
                    baseline_translation,
                    baseline_rotation.as_quat(),
                ],
                "hand": baseline_qpos,
            },
        )
        delta = ComponentAction(
            action_schema=delta_schema,
            values_by_component={
                "arm": np.r_[delta_translation, delta_rotation.as_quat()],
                "hand": delta_qpos,
            },
        )
        assert runtime.admit_composition_input(
            _draft(
                role="policy",
                sequence=sequence,
                action=baseline,
                effective_from=target,
                effective_until=11.5,
            )
        ).accepted
        assert runtime.admit_composition_input(
            _draft(
                role="human",
                sequence=sequence,
                action=delta,
                effective_from=target,
                effective_until=11.5,
            )
        ).accepted

        result = runtime.evaluate_composition(
            spec=spec,
            target_effective_at=target,
        ).composition_evaluation
        assert result.output_status == "ready"
        output = result.composed_candidate
        np.testing.assert_allclose(
            output.value("arm").values[:3],
            baseline_translation + delta_translation,
            atol=1e-12,
        )
        expected_rotation = baseline_rotation * delta_rotation
        actual_rotation = Rotation.from_quat(
            output.value("arm").values[3:7]
        )
        assert (
            expected_rotation.inv() * actual_rotation
        ).magnitude() < 1e-10
        np.testing.assert_allclose(
            output.value("hand").values,
            baseline_qpos + delta_qpos,
            atol=1e-12,
        )
        assert result.composition_supported_until == 11.5


def test_phase_offset_sources_are_selected_only_at_their_effective_time() -> None:
    output_schema, delta_schema, spec, strategy = _contract()
    runtime = _runtime(output_schema, delta_schema, strategy)
    assert runtime.admit_composition_input(
        _draft(
            role="policy",
            sequence=0,
            action=ComponentAction(
                action_schema=output_schema,
                values_by_component={
                    "arm": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "hand": [0.2, 0.4],
                },
            ),
        )
    ).accepted
    assert runtime.admit_composition_input(
        _draft(
            role="human",
            sequence=0,
            action=ComponentAction(
                action_schema=delta_schema,
                values_by_component={
                    "arm": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "hand": [0.05, -0.1],
                },
            ),
            effective_from=10.1,
            effective_until=10.4,
        )
    ).accepted

    before_delta = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.05,
    ).composition_evaluation
    assert before_delta.output_status == "no_action"
    assert before_delta.reason == "missing_required:human"

    at_delta = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    ).composition_evaluation
    assert at_delta.output_status == "ready"
    np.testing.assert_allclose(
        at_delta.composed_candidate.value("arm").values[:3],
        [1.1, 0.0, 0.0],
    )
    assert at_delta.composition_supported_until == 10.4


def test_relative_trace_publishes_self_describing_action_schemas_once() -> None:
    output_schema, delta_schema, spec, strategy = _contract()
    events: list[CompositionDomainEvent] = []
    runtime = _runtime(
        output_schema,
        delta_schema,
        strategy,
        captured_events=events,
    )
    assert runtime.admit_composition_input(
        _draft(
            role="policy",
            sequence=0,
            action=ComponentAction(
                action_schema=output_schema,
                values_by_component={
                    "arm": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "hand": [0.2, 0.4],
                },
            ),
        )
    ).accepted
    assert runtime.admit_composition_input(
        _draft(
            role="human",
            sequence=0,
            action=ComponentAction(
                action_schema=delta_schema,
                values_by_component={
                    "arm": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "hand": [0.05, -0.1],
                },
            ),
        )
    ).accepted
    for target in (10.1, 10.2):
        result = runtime.evaluate_composition(
            spec=spec,
            target_effective_at=target,
        ).composition_evaluation
        assert result.output_status == "ready"

    payload_schema_events = tuple(
        event
        for event in events
        if event.event_kind == "action_payload_schema"
    )
    action_schema_events = tuple(
        event
        for event in events
        if event.event_kind == "component_action_schema"
    )
    assert len(payload_schema_events) == 4
    assert len(action_schema_events) == 2
    assert {
        event.subject.semantic_content_hash
        for event in action_schema_events
    } == {
        output_schema.semantic_content_hash,
        delta_schema.semantic_content_hash,
    }

    trace_context = AuthoritativeTraceContext(
        runtime_session_id="session",
        control_domain_id="robot",
        embodiment_instance_id="virtual",
        clock_domain_id="control.monotonic",
    )
    facts = tuple(
        project_composition_domain_event(trace_context, event)
        for event in events
    )
    payload_facts = tuple(
        fact for fact in facts if fact.event_type == "action_payload_schema"
    )
    assert {
        fact.payload["value_semantics"] for fact in payload_facts
    } == {"absolute_target", "relative_delta"}
    relative_pose_fact = next(
        fact
        for fact in payload_facts
        if fact.payload["representation_family"] == "cartesian_pose"
        and fact.payload["value_semantics"] == "relative_delta"
    )
    assert relative_pose_fact.payload["reference_frame"] == "world"
    assert relative_pose_fact.payload["controlled_frame"] == "tool"
    schema_issue_codes = {
        issue.code
        for issue in evaluate_control_trace_lineage(
            facts,
            profile=trace_requirement_profile("plain_dry_run_v1"),
        )
        if "action_schema" in issue.code or "payload_schema" in issue.code
    }
    assert schema_issue_codes == set()

    missing_relative_pose = tuple(
        fact
        for fact in facts
        if fact.fact_id != relative_pose_fact.fact_id
    )
    missing_issue_codes = {
        issue.code
        for issue in evaluate_control_trace_lineage(
            missing_relative_pose,
            profile=trace_requirement_profile("plain_dry_run_v1"),
        )
    }
    assert "component_action_schema_missing_payload_schema" in (
        missing_issue_codes
    )


def test_virtual_policy_composes_with_canonicalized_pose_and_joint_residuals() -> None:
    output_schema, delta_schema, spec, strategy = _contract()
    runtime = _runtime(output_schema, delta_schema, strategy)
    graph = CoordinateFrameGraph(
        graph_id="virtual-policy-human-residual",
        frames=tuple(
            CoordinateFrameSpec(frame_id)
            for frame_id in ("tracker_world", "world", "tracker", "tool")
        ),
        transforms=(
            RigidFrameTransformSpec(
                transform_id="tracker-world-to-world",
                source_frame_id="tracker_world",
                target_frame_id="world",
                matrix_target_source=np.eye(4),
            ),
            RigidFrameTransformSpec(
                transform_id="tool-to-tracker",
                source_frame_id="tool",
                target_frame_id="tracker",
                matrix_target_source=np.eye(4),
            ),
        ),
    )
    residual_adapter = HumanResidualPoseAdapter(
        output_action_schema=delta_schema,
        frame_graph=graph,
        translation_gain_by_component={"arm": 2.0},
    )

    def sample(sample_id: str, measured_at: float, x: float) -> HumanPoseSample:
        pose = np.eye(4)
        pose[0, 3] = x
        return HumanPoseSample(
            component_id="arm",
            sample_id=sample_id,
            measured_at=measured_at,
            measured_clock_domain_id="tracker.source",
            source_reference_frame_id="tracker_world",
            source_controlled_frame_id="tracker",
            source_coordinate_binding_id="virtual-calibration:0",
            source_coordinate_binding_hash="virtual-calibration-hash:0",
            pose_source_reference_source_controlled=pose,
        )

    anchor = residual_adapter.capture_anchor(
        (sample("anchor", 9.9, 0.2),),
        anchor_generation=0,
        captured_at=9.9,
        captured_clock_domain_id="control.monotonic",
    )
    human_pose = residual_adapter.adapt(
        (sample("current", 10.0, 0.3),),
        anchor=anchor,
        adapted_at=10.0,
        adapted_clock_domain_id="control.monotonic",
    ).relative_action
    joint_adapter = HumanJointPositionResidualAdapter(
        output_action_schema=delta_schema,
        source_payload_schema_by_component={
            "hand": output_schema.payload_schema_by_component["hand"],
        },
    )

    def joint_sample(
        sample_id: str,
        measured_at: float,
        values: tuple[float, float],
    ) -> HumanJointPositionSample:
        return HumanJointPositionSample(
            component_id="hand",
            sample_id=sample_id,
            measured_at=measured_at,
            measured_clock_domain_id="q6.source",
            source_value_binding_id="virtual-q6-retarget:0",
            source_value_binding_hash="virtual-q6-retarget-hash:0",
            source_payload_schema=(
                output_schema.payload_schema_by_component["hand"]
            ),
            values=values,
        )

    joint_anchor = joint_adapter.capture_anchor(
        (joint_sample("joint-anchor", 9.9, (0.4, 0.6)),),
        anchor_generation=0,
        captured_at=9.9,
        captured_clock_domain_id="control.monotonic",
    )
    human_joint = joint_adapter.adapt(
        (joint_sample("joint-current", 10.0, (0.5, 0.4)),),
        anchor=joint_anchor,
        adapted_at=10.0,
        adapted_clock_domain_id="control.monotonic",
    ).relative_action
    human = ComponentAction(
        action_schema=delta_schema,
        values_by_component={
            "arm": human_pose.value("arm").values,
            "hand": human_joint.value("hand").values,
        },
    )
    policy = ComponentAction(
        action_schema=output_schema,
        values_by_component={
            "arm": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "hand": [0.2, 0.4],
        },
    )
    assert runtime.admit_composition_input(
        _draft(role="policy", sequence=0, action=policy)
    ).accepted
    assert runtime.admit_composition_input(
        _draft(role="human", sequence=0, action=human)
    ).accepted

    evaluation = runtime.evaluate_composition(
        spec=spec,
        target_effective_at=10.1,
    ).composition_evaluation

    assert evaluation.output_status == "ready"
    np.testing.assert_allclose(
        evaluation.composed_candidate.value("arm").values[:3],
        [1.2, 0.0, 0.0],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        evaluation.composed_candidate.value("hand").values,
        [0.3, 0.2],
    )


def test_relative_contract_builder_does_not_assume_policy_or_human_roles() -> None:
    output_schema, delta_schema, _, _ = _contract()
    contracts = build_relative_baseline_residual_contracts(
        contract_id="planner-with-correction",
        output_action_schema=output_schema,
        residual_action_schema=delta_schema,
        baseline_role="planner",
        residual_role="correction",
    )
    runtime = _runtime(
        output_schema,
        delta_schema,
        contracts.strategy,
        baseline_role="planner",
        residual_role="correction",
    )
    baseline = ComponentAction(
        action_schema=output_schema,
        values_by_component={
            "arm": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "hand": [0.2, 0.4],
        },
    )
    residual = ComponentAction(
        action_schema=delta_schema,
        values_by_component={
            "arm": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "hand": [0.05, -0.1],
        },
    )
    assert runtime.admit_composition_input(
        _draft(role="planner", sequence=0, action=baseline)
    ).accepted
    assert runtime.admit_composition_input(
        _draft(role="correction", sequence=0, action=residual)
    ).accepted

    result = runtime.evaluate_composition(
        spec=contracts.composition_spec,
        target_effective_at=10.1,
    ).composition_evaluation

    assert result.output_status == "ready"
    np.testing.assert_allclose(
        result.composed_candidate.value("arm").values[:3],
        (1.1, 0.0, 0.0),
    )
    np.testing.assert_allclose(
        result.composed_candidate.value("hand").values,
        (0.25, 0.3),
    )
