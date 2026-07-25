from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from control_composition.composition import (
    FORMAL_COMPOSITION_STRATEGY_SPEC_HASH_DOMAIN_V2,
    FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
    ComponentCompositionRuleSet,
    CompositionInputDraft,
    CompositionSpec,
    FormalComponentCompositionStrategyV2,
    FormalCompositionStrategySpec,
    PoseComponentCompositionRule,
    QposComponentCompositionRule,
    hand_action_spec_semantic_hash,
)
from control_composition.composition.core import (
    COMPOSITION_SPEC_HASH_DOMAIN_V1,
    COMPOSITION_SPEC_HASH_DOMAIN_V2,
)
from control_composition.retention import TimeWindowRetentionSpec
from control_composition.runtime import ControlCompositionRuntime
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_interface.composition_control_trace import (
    project_composition_domain_event,
)
from robot_motion.action import (
    BIMANUAL_EE_POSE_HAND16_V1,
    BIMANUAL_EE_POSE_V1,
    ActionComponentCatalog,
    ActionComponentSpec,
    HandActionSpec,
    ProposalBuffer,
    RobotAction,
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
        spec_id="formal-component-test-window-v1",
        history_window_sec=0.0,
        max_future_lead_sec=2.0,
        max_effective_duration_sec=2.0,
    )


def build_contract(
    *,
    action_space: str = BIMANUAL_EE_POSE_V1,
    hand_spec: HandActionSpec | None = None,
    missing_behavior: str = "no_action",
):
    value_spec = hand_spec or HandActionSpec.inspire_tac("left")
    rule_set = ComponentCompositionRuleSet(
        rule_set_id="virtual-human-policy-pose-qpos-v1",
        action_space=action_space,
        rules=(
            PoseComponentCompositionRule(
                component_id="left_arm",
                contributing_roles=("human", "policy"),
                weight_by_role={"human": 0.75, "policy": 0.25},
                missing_behavior=missing_behavior,
                max_translation_disagreement_m=0.10,
                max_rotation_disagreement_rad=math.radians(30.0),
            ),
            QposComponentCompositionRule(
                component_id="left_hand",
                contributing_roles=("human", "policy"),
                weight_by_role={"human": 0.25, "policy": 0.75},
                missing_behavior=missing_behavior,
                value_contract_hash=hand_action_spec_semantic_hash(value_spec),
            ),
        ),
    )
    composition_spec = CompositionSpec(
        spec_id="virtual-human-policy-pose-qpos-v2",
        algebra=FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
        input_roles=("human", "policy"),
        required_roles=("human",),
        component_rule_set_hash=rule_set.semantic_content_hash,
    )
    strategy_spec = FormalCompositionStrategySpec(
        strategy_id="virtual-human-policy-component-strategy-v2",
        strategy_version="2.0",
        composition_spec_hash=composition_spec.semantic_content_hash,
        implementation_contract_hashes=(
            rule_set.semantic_content_hash,
        ),
        maximum_decision_latency_sec=0.01,
    )
    strategy = FormalComponentCompositionStrategyV2(
        strategy_spec=strategy_spec,
        composition_spec=composition_spec,
        component_rule_set=rule_set,
        qpos_value_spec_by_component={"left_hand": value_spec},
    )
    return value_spec, rule_set, composition_spec, strategy


def build_runtime(
    strategy,
    *,
    action_space: str = BIMANUAL_EE_POSE_V1,
):
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
    for producer_id in ("human-0", "policy-0", "composer-0"):
        epochs.activate(
            producer_id=producer_id,
            producer_epoch=1,
            produced_at=9.9,
            produced_clock_domain_id=f"{producer_id}.monotonic",
            reason="startup",
        )
    proposals = ProposalBuffer(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        capacity=16,
    )
    events = []
    runtime = ControlCompositionRuntime(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="virtual-robot-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        component_catalog=ActionComponentCatalog(
            catalog_id="virtual-left-arm-hand-v1",
            action_space=action_space,
            components=(
                ActionComponentSpec("left_arm", "left_arm", "left_arm"),
                ActionComponentSpec(
                    "left_hand",
                    "left_hand",
                    "left_hand",
                ),
            ),
        ),
        event_emitter=lambda batch: events.extend(batch),
        proposal_admitter=lambda draft, **_kwargs: proposals.admit(draft),
        context_retention_spec=retention_spec(),
        composition_input_retention_spec=retention_spec(),
        composition_strategy=strategy,
        clock=clock,
    )
    return runtime, proposals, events


def input_draft(
    *,
    producer_id: str,
    role: str,
    sequence: int,
    arm_x: float,
    hand_values: np.ndarray,
    action_space: str = BIMANUAL_EE_POSE_V1,
    hand_spec: HandActionSpec | None = None,
) -> CompositionInputDraft:
    metadata = (
        {}
        if hand_spec is None
        else {"hand_action_specs_by_side": {"left": hand_spec.to_metadata()}}
    )
    return CompositionInputDraft(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        producer_id=producer_id,
        producer_epoch=1,
        sequence=sequence,
        input_role=role,
        action=RobotAction(
            action_space=action_space,
            arm_pose_by_side={
                "left": np.asarray(
                    [arm_x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    dtype=np.float32,
                )
            },
            hand_qpos_by_side={"left": hand_values},
            active_effectors=("left_arm", "left_hand"),
            frame_by_effector={"left_arm": "left_arm_base"},
            controlled_frame_by_effector={"left_arm": "tool_tcp"},
            source=f"virtual-{role}",
            metadata=metadata,
        ),
        produced_at=9.9,
        produced_clock_domain_id=f"{producer_id}.monotonic",
        effective_from=10.0,
        effective_until=11.0,
        effective_clock_domain_id="control.monotonic",
    )


def test_existing_composition_spec_hash_is_byte_compatible_v1() -> None:
    spec = CompositionSpec(
        spec_id="component-union-v1",
        algebra="component_union",
        input_roles=("base",),
        required_roles=("base",),
    )
    assert spec.schema_version == 1
    assert spec.semantic_content_hash == canonical_content_hash(
        {
            "spec_id": "component-union-v1",
            "algebra": "component_union",
            "input_roles": ("base",),
            "required_roles": ("base",),
            "missing_required_behavior": "no_action",
            "missing_residual_behavior": "zero_residual",
            "residual_limit_by_component": {},
        },
        domain=COMPOSITION_SPEC_HASH_DOMAIN_V1,
    )


def test_additive_v2_spec_has_isolated_hash_and_forbids_v1_parameters() -> None:
    _, rule_set, spec, _ = build_contract()
    assert spec.schema_version == 2
    assert spec.semantic_content_hash == canonical_content_hash(
        {
            "spec_id": spec.spec_id,
            "algebra": FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
            "input_roles": ("human", "policy"),
            "required_roles": ("human",),
            "missing_required_behavior": "no_action",
            "component_rule_set_hash": rule_set.semantic_content_hash,
        },
        domain=COMPOSITION_SPEC_HASH_DOMAIN_V2,
    )
    with pytest.raises(ValueError, match="component_rule_set_hash"):
        CompositionSpec(
            spec_id="invalid-formal-component-v2",
            algebra=FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
            input_roles=("human",),
            required_roles=("human",),
        )
    with pytest.raises(ValueError, match="deterministic-v1"):
        CompositionSpec(
            spec_id="invalid-formal-component-fields-v2",
            algebra=FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
            input_roles=("human",),
            required_roles=("human",),
            weight_by_role={"human": 1.0},
            component_rule_set_hash=rule_set.semantic_content_hash,
        )
    with pytest.raises(ValueError, match="requires formal_component_strategy_v2"):
        CompositionSpec(
            spec_id="invalid-v1-rule-binding",
            algebra="component_union",
            input_roles=("human",),
            required_roles=("human",),
            component_rule_set_hash=rule_set.semantic_content_hash,
        )


def test_existing_formal_strategy_spec_hash_is_unchanged_without_contracts() -> None:
    strategy_spec = FormalCompositionStrategySpec(
        strategy_id="existing-formal-strategy-v2",
        strategy_version="2.0",
        composition_spec_hash="a" * 64,
        required_context_channels=("robot.state",),
        parameter_payload={"implementation_gain": 0.5},
        maximum_decision_latency_sec=0.002,
    )
    assert strategy_spec.semantic_content_hash == canonical_content_hash(
        {
            "strategy_id": "existing-formal-strategy-v2",
            "strategy_version": "2.0",
            "execution_class": "inline_bounded",
            "composition_spec_hash": "a" * 64,
            "required_context_channels": ("robot.state",),
            "parameter_payload": {"implementation_gain": 0.5},
            "maximum_decision_latency_sec": 0.002,
        },
        domain=FORMAL_COMPOSITION_STRATEGY_SPEC_HASH_DOMAIN_V2,
    )


def test_virtual_human_policy_pose_and_q6_form_one_result() -> None:
    hand_spec, _, composition_spec, strategy = build_contract()
    runtime, proposals, events = build_runtime(strategy)
    assert runtime.admit_composition_input(
        input_draft(
            producer_id="human-0",
            role="human",
            sequence=0,
            arm_x=0.0,
            hand_values=np.zeros(6, dtype=np.float32),
            hand_spec=hand_spec,
        )
    ).accepted
    assert runtime.admit_composition_input(
        input_draft(
            producer_id="policy-0",
            role="policy",
            sequence=0,
            arm_x=0.04,
            hand_values=np.ones(6, dtype=np.float32),
            hand_spec=hand_spec,
        )
    ).accepted

    evaluation = runtime.evaluate_composition(
        spec=composition_spec,
        target_effective_at=10.1,
    )
    result = evaluation.composition_result
    assert result.output_status == "ready"
    assert result.reason == "formal_component_composition"
    assert result.composed_action is not None
    assert result.composed_action.active_effectors == (
        "left_arm",
        "left_hand",
    )
    np.testing.assert_allclose(
        result.composed_action.arm_pose_by_side["left"][:3],
        np.asarray([0.01, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        result.composed_action.hand_qpos_by_side["left"],
        np.full(6, 0.75, dtype=np.float32),
    )
    assert result.strategy_spec_hash == strategy.spec.semantic_content_hash
    assert len(proposals) == 0
    rule_events = tuple(
        event
        for event in events
        if event.event_kind == "component_composition_rule_set"
    )
    assert len(rule_events) == 1
    trace_context = AuthoritativeTraceContext(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="virtual-robot-0",
        clock_domain_id="control.monotonic",
    )
    rule_fact = project_composition_domain_event(
        trace_context,
        rule_events[0],
    )
    assert rule_fact.event_type == "component_composition_rule_set"
    assert rule_fact.payload["semantic_content_hash"] == (
        strategy.component_rule_set.semantic_content_hash
    )
    binding_events = tuple(
        event
        for event in events
        if event.event_kind == "formal_component_composition_binding"
    )
    assert len(binding_events) == 1
    binding_fact = project_composition_domain_event(
        trace_context,
        binding_events[0],
    )
    assert binding_fact.payload["composition_spec_hash"] == (
        composition_spec.semantic_content_hash
    )
    assert binding_fact.payload["component_rule_set_hash"] == (
        strategy.component_rule_set.semantic_content_hash
    )
    assert binding_fact.payload["formal_strategy_spec_hash"] == (
        strategy.spec.semantic_content_hash
    )
    event_kinds = tuple(event.event_kind for event in events)
    assert event_kinds.index("composition_spec") < event_kinds.index(
        "component_composition_rule_set"
    )
    assert event_kinds.index(
        "component_composition_rule_set"
    ) < event_kinds.index("formal_composition_strategy_spec")
    assert event_kinds.index(
        "formal_composition_strategy_spec"
    ) < event_kinds.index("formal_component_composition_binding")

    promoted = runtime.admit_evaluated_composition(
        evaluation,
        runtime_session_id="session-0",
        producer_id="composer-0",
        producer_epoch=1,
        sequence=0,
        effective_until=10.2,
        component_coverage=("left_arm", "left_hand"),
        selection_groups=("left_arm", "left_hand"),
        policy_node_id="virtual-component-policy",
    )
    assert promoted.proposal_admission is not None
    assert promoted.proposal_admission.accepted
    assert len(proposals) == 1


def test_v2_without_formal_strategy_fails_closed() -> None:
    hand_spec, _, composition_spec, _ = build_contract()
    runtime, proposals, _ = build_runtime(None)
    assert runtime.admit_composition_input(
        input_draft(
            producer_id="human-0",
            role="human",
            sequence=0,
            arm_x=0.0,
            hand_values=np.zeros(6, dtype=np.float32),
            hand_spec=hand_spec,
        )
    ).accepted
    result = runtime.evaluate_composition(
        spec=composition_spec,
        target_effective_at=10.1,
    ).composition_result
    assert result.output_status == "invalid"
    assert result.reason == "formal_component_strategy_required"
    assert result.composed_action is None
    assert len(proposals) == 0


def test_v2_strategy_cannot_redefine_component_semantics_in_parameters() -> None:
    hand_spec, rule_set, composition_spec, _ = build_contract()
    strategy_spec = FormalCompositionStrategySpec(
        strategy_id="invalid-duplicated-component-semantics-v2",
        strategy_version="2.0",
        composition_spec_hash=composition_spec.semantic_content_hash,
        implementation_contract_hashes=(
            rule_set.semantic_content_hash,
        ),
        parameter_payload={"weight_by_role": {"human": 1.0}},
    )
    strategy = FormalComponentCompositionStrategyV2(
        strategy_spec=strategy_spec,
        composition_spec=composition_spec,
        component_rule_set=rule_set,
        qpos_value_spec_by_component={"left_hand": hand_spec},
    )
    runtime, _, _ = build_runtime(strategy)
    with pytest.raises(ValueError, match="must not redefine"):
        runtime.evaluate_composition(
            spec=composition_spec,
            target_effective_at=10.1,
        )


def test_missing_policy_can_only_renormalize_when_rule_contract_allows_it() -> None:
    hand_spec, _, composition_spec, strategy = build_contract(
        missing_behavior="renormalize_available"
    )
    runtime, _, _ = build_runtime(strategy)
    human_hand = np.linspace(0.0, 0.5, 6, dtype=np.float32)
    assert runtime.admit_composition_input(
        input_draft(
            producer_id="human-0",
            role="human",
            sequence=0,
            arm_x=0.02,
            hand_values=human_hand,
            hand_spec=hand_spec,
        )
    ).accepted
    result = runtime.evaluate_composition(
        spec=composition_spec,
        target_effective_at=10.1,
    ).composition_result
    assert result.output_status == "ready"
    assert result.reason == (
        "formal_component_composition_renormalized:left_arm,left_hand"
    )
    np.testing.assert_allclose(
        result.composed_action.arm_pose_by_side["left"][0],
        0.02,
    )
    np.testing.assert_allclose(
        result.composed_action.hand_qpos_by_side["left"],
        human_hand,
    )


def test_component_missing_from_present_role_obeys_rule_fallback() -> None:
    hand_spec, _, composition_spec, strategy = build_contract()
    runtime, _, _ = build_runtime(strategy)
    assert runtime.admit_composition_input(
        input_draft(
            producer_id="human-0",
            role="human",
            sequence=0,
            arm_x=0.0,
            hand_values=np.zeros(6, dtype=np.float32),
            hand_spec=hand_spec,
        )
    ).accepted
    policy_pose_only = input_draft(
        producer_id="policy-0",
        role="policy",
        sequence=0,
        arm_x=0.01,
        hand_values=np.ones(6, dtype=np.float32),
        hand_spec=hand_spec,
    )
    policy_pose_only = replace(
        policy_pose_only,
        action=RobotAction(
            action_space=BIMANUAL_EE_POSE_V1,
            arm_pose_by_side=policy_pose_only.action.arm_pose_by_side,
            active_effectors=("left_arm",),
            frame_by_effector={"left_arm": "left_arm_base"},
            controlled_frame_by_effector={"left_arm": "tool_tcp"},
            source="virtual-policy-pose-only",
        ),
    )
    assert runtime.admit_composition_input(policy_pose_only).accepted
    result = runtime.evaluate_composition(
        spec=composition_spec,
        target_effective_at=10.1,
    ).composition_result
    assert result.output_status == "no_action"
    assert result.reason == "component_missing:left_hand:policy"
    assert result.composed_action is None


def test_pose_disagreement_and_frame_mismatch_fail_closed() -> None:
    hand_spec, _, composition_spec, strategy = build_contract()
    runtime, _, _ = build_runtime(strategy)
    assert runtime.admit_composition_input(
        input_draft(
            producer_id="human-0",
            role="human",
            sequence=0,
            arm_x=0.0,
            hand_values=np.zeros(6, dtype=np.float32),
            hand_spec=hand_spec,
        )
    ).accepted
    assert runtime.admit_composition_input(
        input_draft(
            producer_id="policy-0",
            role="policy",
            sequence=0,
            arm_x=0.11,
            hand_values=np.ones(6, dtype=np.float32),
            hand_spec=hand_spec,
        )
    ).accepted
    disagreement = runtime.evaluate_composition(
        spec=composition_spec,
        target_effective_at=10.1,
    ).composition_result
    assert disagreement.output_status == "invalid"
    assert disagreement.reason.startswith(
        "component_invalid:left_arm:translation_disagreement:"
    )

    hand_spec2, _, composition_spec2, strategy2 = build_contract()
    runtime2, _, _ = build_runtime(strategy2)
    human = input_draft(
        producer_id="human-0",
        role="human",
        sequence=0,
        arm_x=0.0,
        hand_values=np.zeros(6, dtype=np.float32),
        hand_spec=hand_spec2,
    )
    policy = input_draft(
        producer_id="policy-0",
        role="policy",
        sequence=0,
        arm_x=0.01,
        hand_values=np.ones(6, dtype=np.float32),
        hand_spec=hand_spec2,
    )
    policy = replace(
        policy,
        action=RobotAction(
            action_space=BIMANUAL_EE_POSE_V1,
            arm_pose_by_side=policy.action.arm_pose_by_side,
            hand_qpos_by_side=policy.action.hand_qpos_by_side,
            active_effectors=("left_arm", "left_hand"),
            frame_by_effector={"left_arm": "different_arm_base"},
            controlled_frame_by_effector={"left_arm": "tool_tcp"},
            source="virtual-policy-wrong-frame",
            metadata=policy.action.metadata,
        ),
    )
    assert runtime2.admit_composition_input(human).accepted
    assert runtime2.admit_composition_input(policy).accepted
    mismatch = runtime2.evaluate_composition(
        spec=composition_spec2,
        target_effective_at=10.1,
    ).composition_result
    assert mismatch.output_status == "invalid"
    assert mismatch.reason == (
        "component_invalid:left_arm:frame_or_action_space_mismatch"
    )


def test_q16_active_mask_preserves_inactive_padding() -> None:
    hand_spec = HandActionSpec.q16(
        "left",
        hand_model="virtual-q6-in-q16",
        active_dof=6,
    )
    _, _, composition_spec, strategy = build_contract(
        action_space=BIMANUAL_EE_POSE_HAND16_V1,
        hand_spec=hand_spec,
    )
    runtime, _, _ = build_runtime(
        strategy,
        action_space=BIMANUAL_EE_POSE_HAND16_V1,
    )
    human = np.full(16, np.nan, dtype=np.float32)
    policy = np.full(16, np.nan, dtype=np.float32)
    human[:6] = 0.2
    policy[:6] = 0.6
    for producer_id, role, values in (
        ("human-0", "human", human),
        ("policy-0", "policy", policy),
    ):
        assert runtime.admit_composition_input(
            input_draft(
                producer_id=producer_id,
                role=role,
                sequence=0,
                arm_x=0.0,
                hand_values=values,
                action_space=BIMANUAL_EE_POSE_HAND16_V1,
                hand_spec=hand_spec,
            )
        ).accepted
    result = runtime.evaluate_composition(
        spec=composition_spec,
        target_effective_at=10.1,
    ).composition_result
    output = result.composed_action.hand_qpos_by_side["left"]
    np.testing.assert_allclose(output[:6], np.full(6, 0.5))
    assert np.all(np.isnan(output[6:]))
