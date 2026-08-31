from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from control_composition.composition import (
    FORMAL_COMPOSITION_STRATEGY_SPEC_HASH_DOMAIN_V2,
    FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
    FORMAL_COMPONENT_STRATEGY_MODE_DISCRETE_OWNER_V1,
    FORMAL_COMPONENT_STRATEGY_MODE_WEIGHTED_V1,
    ComponentCompositionRuleSet,
    CompositionInputDraft,
    CompositionDynamicParameterContract,
    CompositionDynamicParameterDraft,
    CompositionSpec,
    CompositionStrategyControl,
    CompositionStrategyRevisionDraft,
    FormalComponentCompositionStrategyV2,
    FormalCompositionStrategySpec,
    PoseComponentCompositionRule,
    QposComponentCompositionRule,
    build_initial_strategy_revision,
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
    build_robot_action_v1_runtime_schema,
    project_robot_action_v1_to_component_action,
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
    max_translation_disagreement_m: float | None = 0.10,
    max_rotation_disagreement_rad: float | None = math.radians(30.0),
    maximum_decision_latency_sec: float | None = 0.01,
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
                max_translation_disagreement_m=(
                    max_translation_disagreement_m
                ),
                max_rotation_disagreement_rad=max_rotation_disagreement_rad,
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
        maximum_decision_latency_sec=maximum_decision_latency_sec,
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
    enable_hot_control: bool = False,
    hot_parameter_lifecycle: str = "latched",
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
    component_catalog = ActionComponentCatalog(
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
    )
    hand_spec = (
        None
        if strategy is None
        else getattr(
            strategy,
            "qpos_value_spec_by_component",
            {},
        ).get("left_hand")
    )
    if hand_spec is None:
        hand_spec = (
            HandActionSpec.q16("left")
            if action_space == BIMANUAL_EE_POSE_HAND16_V1
            else HandActionSpec.inspire_tac("left")
        )
    strategy_control = None
    if enable_hot_control:
        if strategy is None:
            raise ValueError("Hot control requires a formal strategy.")
        component_ids = tuple(
            rule.component_id for rule in strategy.component_rule_set.rules
        )
        parameter_contract = CompositionDynamicParameterContract(
            parameter_set_id="human-policy-alpha-v1",
            base_role="policy",
            overlay_role="human",
            component_ids=component_ids,
            default_alpha_by_component={
                component_id: 0.0 for component_id in component_ids
            },
            lifecycle=hot_parameter_lifecycle,
        )
        initial_revision = build_initial_strategy_revision(
            runtime_session_id="session-0",
            control_domain_id="robot-0",
            strategy_instance_id="component-strategy-instance-0",
            request_id="initial",
            strategy_mode=FORMAL_COMPONENT_STRATEGY_MODE_WEIGHTED_V1,
            composition_spec=strategy.composition_spec,
            strategy_spec=strategy.spec,
            component_rule_set=strategy.component_rule_set,
            dynamic_parameter_contract=parameter_contract,
            initial_alpha_by_component={
                component_id: 0.0 for component_id in component_ids
            },
            available_at=9.8,
            available_clock_domain_id="control.monotonic",
        )
        strategy_control = CompositionStrategyControl(
            runtime_session_id="session-0",
            control_domain_id="robot-0",
            strategy_instance_id="component-strategy-instance-0",
            coordinator=coordinator,
            producer_epochs=epochs,
            retention_spec=retention_spec(),
            supported_strategy_modes=strategy.supported_strategy_modes,
            initial_revision=initial_revision,
        )
    runtime = ControlCompositionRuntime(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="virtual-robot-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        action_schema=build_robot_action_v1_runtime_schema(
            component_catalog=component_catalog,
            hand_action_specs_by_side={"left": hand_spec},
        ),
        event_emitter=lambda batch: events.extend(batch),
        proposal_admitter=lambda draft, **_kwargs: proposals.admit(draft),
        context_retention_spec=retention_spec(),
        composition_input_retention_spec=retention_spec(),
        composition_strategy=strategy,
        strategy_control=strategy_control,
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
    robot_action = RobotAction(
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
    )
    component_catalog = ActionComponentCatalog(
        catalog_id="virtual-left-arm-hand-v1",
        action_space=action_space,
        components=(
            ActionComponentSpec("left_arm", "left_arm", "left_arm"),
            ActionComponentSpec("left_hand", "left_hand", "left_hand"),
        ),
    )
    action_schema = build_robot_action_v1_runtime_schema(
        component_catalog=component_catalog,
        hand_action_specs_by_side={
            "left": hand_spec
            or (
                HandActionSpec.q16("left")
                if action_space == BIMANUAL_EE_POSE_HAND16_V1
                else HandActionSpec.inspire_tac("left")
            )
        },
    )
    return CompositionInputDraft(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        producer_id=producer_id,
        producer_epoch=1,
        sequence=sequence,
        input_role=role,
        action=project_robot_action_v1_to_component_action(
            robot_action,
            action_schema=action_schema,
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
    result = evaluation.composition_evaluation
    assert result.output_status == "ready"
    assert result.reason == "formal_component_composition"
    assert result.composed_candidate is not None
    assert result.composed_candidate.component_coverage == (
        "left_arm",
        "left_hand",
    )
    np.testing.assert_allclose(
        result.composed_candidate.value("left_arm").values[:3],
        np.asarray([0.01, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        result.composed_candidate.value("left_hand").values,
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
    ).composition_evaluation
    assert result.output_status == "invalid"
    assert result.reason == "formal_component_strategy_required"
    assert result.composed_candidate is None
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
    ).composition_evaluation
    assert result.output_status == "ready"
    assert result.reason == (
        "formal_component_composition_renormalized:left_arm,left_hand"
    )
    np.testing.assert_allclose(
        result.composed_candidate.value("left_arm").values[0],
        0.02,
    )
    np.testing.assert_allclose(
        result.composed_candidate.value("left_hand").values,
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
        action=policy_pose_only.action.project(("left_arm",)),
    )
    assert runtime.admit_composition_input(policy_pose_only).accepted
    result = runtime.evaluate_composition(
        spec=composition_spec,
        target_effective_at=10.1,
    ).composition_evaluation
    assert result.output_status == "no_action"
    assert result.reason == "component_missing:left_hand:policy"
    assert result.composed_candidate is None


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
    ).composition_evaluation
    assert disagreement.output_status == "invalid"
    assert disagreement.reason == (
        "component_invalid:left_arm:translation_disagreement"
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
    wrong_frame = RobotAction(
            action_space=BIMANUAL_EE_POSE_V1,
            arm_pose_by_side={"left": policy.action.value("left_arm").values},
            hand_qpos_by_side={"left": policy.action.value("left_hand").values},
            active_effectors=("left_arm", "left_hand"),
            frame_by_effector={"left_arm": "different_arm_base"},
            controlled_frame_by_effector={"left_arm": "tool_tcp"},
            source="virtual-policy-wrong-frame",
            metadata={
                "hand_action_specs_by_side": {
                    "left": hand_spec2.to_metadata()
                }
            },
        )
    assert runtime2.admit_composition_input(human).accepted
    with pytest.raises(ValueError, match="does not match"):
        project_robot_action_v1_to_component_action(
            wrong_frame,
            action_schema=policy.action.action_schema,
        )


def test_pose_disagreement_can_be_observe_only_without_rejection() -> None:
    hand_spec, _, composition_spec, strategy = build_contract(
        max_translation_disagreement_m=None,
        max_rotation_disagreement_rad=None,
    )
    runtime, _, _ = build_runtime(strategy)
    for producer_id, role, arm_x, hand_value in (
        ("human-0", "human", 0.0, 0.0),
        ("policy-0", "policy", 1.0, 1.0),
    ):
        assert runtime.admit_composition_input(
            input_draft(
                producer_id=producer_id,
                role=role,
                sequence=0,
                arm_x=arm_x,
                hand_values=np.full(6, hand_value, dtype=np.float32),
                hand_spec=hand_spec,
            )
        ).accepted

    result = runtime.evaluate_composition(
        spec=composition_spec,
        target_effective_at=10.1,
    ).composition_evaluation

    assert result.output_status == "ready"
    assert result.composed_candidate is not None
    np.testing.assert_allclose(
        result.composed_candidate.value("left_arm").values[:3],
        np.asarray([0.25, 0.0, 0.0], dtype=np.float32),
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
    ).composition_evaluation
    output = result.composed_candidate.value("left_hand").values
    np.testing.assert_allclose(output[:6], np.full(6, 0.5))
    assert np.all(np.isnan(output[6:]))


def _admit_hot_control_inputs(
    runtime: ControlCompositionRuntime,
    *,
    hand_spec: HandActionSpec,
) -> None:
    for producer_id, role, arm_x, hand_value in (
        ("human-0", "human", 0.08, 1.0),
        ("policy-0", "policy", 0.00, 0.0),
    ):
        assert runtime.admit_composition_input(
            input_draft(
                producer_id=producer_id,
                role=role,
                sequence=0,
                arm_x=arm_x,
                hand_values=np.full(6, hand_value, dtype=np.float32),
                hand_spec=hand_spec,
            )
        ).accepted


def _parameter_draft(
    runtime: ControlCompositionRuntime,
    *,
    sequence: int,
    alpha: float,
    effective_until: float | None = None,
) -> CompositionDynamicParameterDraft:
    control = runtime.strategy_control
    assert control is not None
    revision = control.latest_revision
    contract = revision.dynamic_parameter_contract
    assert contract is not None
    return CompositionDynamicParameterDraft(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        producer_id="human-0",
        producer_generation=1,
        sequence=sequence,
        strategy_revision_hash=revision.semantic_content_hash,
        parameter_contract_hash=contract.semantic_content_hash,
        alpha_by_component={
            component_id: alpha for component_id in contract.component_ids
        },
        produced_at=10.0,
        produced_clock_domain_id="human-0.monotonic",
        effective_from=10.0,
        effective_until=effective_until,
        effective_clock_domain_id="control.monotonic",
    )


def test_hot_alpha_uses_latest_record_from_the_same_control_cut() -> None:
    hand_spec, _, _, strategy = build_contract()
    runtime, _, events = build_runtime(
        strategy,
        enable_hot_control=True,
    )
    _admit_hot_control_inputs(runtime, hand_spec=hand_spec)

    before_update = runtime.coordinator.capture_cut(cutoff_at=10.0).cut
    receipt = runtime.admit_dynamic_parameters(
        _parameter_draft(runtime, sequence=0, alpha=0.75)
    )
    assert receipt.accepted

    old_cut = runtime.evaluate_hot_composition(
        target_effective_at=10.1,
        control_cut=before_update,
    )
    new_cut = runtime.evaluate_hot_composition(
        target_effective_at=10.1,
        cutoff_at=10.0,
    )
    old_action = old_cut.composition_evaluation.composed_candidate
    new_action = new_cut.composition_evaluation.composed_candidate
    np.testing.assert_allclose(
        old_action.value("left_arm").values[0],
        0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        new_action.value("left_arm").values[0],
        0.06,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        new_action.value("left_hand").values,
        np.full(6, 0.75),
        atol=1e-6,
    )
    assert (
        old_cut.evaluation_record.strategy_control_binding
        .parameter_snapshot.selected_parameter_record_id
        is None
    )
    assert (
        new_cut.evaluation_record.strategy_control_binding
        .parameter_snapshot.selected_parameter_record_id
        == receipt.parameter_record.parameter_record_id
    )
    assert new_cut.composition_evaluation.strategy_control_binding_hash == (
        new_cut.evaluation_record.strategy_control_binding.semantic_content_hash
    )
    kinds = tuple(event.event_kind for event in events)
    assert "composition_dynamic_parameter" in kinds
    assert "composition_dynamic_parameter_snapshot" in kinds
    assert "composition_strategy_control_binding" in kinds
    trace_context = AuthoritativeTraceContext(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="virtual-robot-0",
        clock_domain_id="control.monotonic",
    )
    hot_event_kinds = {
        "composition_dynamic_parameter_contract",
        "composition_strategy_revision",
        "composition_dynamic_parameter",
        "composition_dynamic_parameter_snapshot",
        "composition_strategy_control_binding",
        "hot_composition_strategy_input",
    }
    projected = {
        event.event_kind: project_composition_domain_event(
            trace_context,
            event,
        )
        for event in events
        if event.event_kind in hot_event_kinds
    }
    assert set(projected) == hot_event_kinds
    assert (
        projected["hot_composition_strategy_input"].payload[
            "strategy_control_binding_hash"
        ]
        == new_cut.composition_evaluation.strategy_control_binding_hash
    )


def test_invalid_hot_parameter_keeps_the_previous_applied_value() -> None:
    hand_spec, _, _, strategy = build_contract()
    runtime, _, _ = build_runtime(strategy, enable_hot_control=True)
    _admit_hot_control_inputs(runtime, hand_spec=hand_spec)
    accepted = runtime.admit_dynamic_parameters(
        _parameter_draft(runtime, sequence=0, alpha=0.4)
    )
    assert accepted.accepted
    rejected = runtime.admit_dynamic_parameters(
        _parameter_draft(runtime, sequence=1, alpha=1.2)
    )
    assert not rejected.accepted

    result = runtime.evaluate_hot_composition(
        target_effective_at=10.1,
        cutoff_at=10.0,
    )
    np.testing.assert_allclose(
        result.composition_evaluation.composed_candidate
        .value("left_arm")
        .values[0],
        0.032,
        atol=1e-6,
    )
    snapshot = result.evaluation_record.strategy_control_binding.parameter_snapshot
    assert snapshot.selected_parameter_record_id == (
        accepted.parameter_record.parameter_record_id
    )


def test_low_frequency_revision_switch_is_atomic_with_initial_parameters() -> None:
    hand_spec, _, _, strategy = build_contract()
    runtime, _, _ = build_runtime(strategy, enable_hot_control=True)
    _admit_hot_control_inputs(runtime, hand_spec=hand_spec)
    old_parameter = runtime.admit_dynamic_parameters(
        _parameter_draft(runtime, sequence=0, alpha=0.25)
    )
    assert old_parameter.accepted
    control = runtime.strategy_control
    assert control is not None
    previous = control.latest_revision
    revised_rules = replace(
        previous.component_rule_set,
        rule_set_id="virtual-human-policy-pose-qpos-revision-1",
    )
    revised_spec = replace(
        previous.composition_spec,
        spec_id="virtual-human-policy-pose-qpos-revision-1",
        component_rule_set_hash=revised_rules.semantic_content_hash,
    )
    revised_strategy_spec = replace(
        previous.strategy_spec,
        composition_spec_hash=revised_spec.semantic_content_hash,
        implementation_contract_hashes=(
            revised_rules.semantic_content_hash,
        ),
    )
    revision_receipt = runtime.admit_strategy_revision(
        CompositionStrategyRevisionDraft(
            request_id="switch-to-human-default",
            expected_previous_revision_hash=previous.semantic_content_hash,
            strategy_mode=(
                FORMAL_COMPONENT_STRATEGY_MODE_DISCRETE_OWNER_V1
            ),
            composition_spec=revised_spec,
            strategy_spec=revised_strategy_spec,
            component_rule_set=revised_rules,
            dynamic_parameter_contract=(
                previous.dynamic_parameter_contract
            ),
            initial_alpha_by_component={
                "left_arm": 0.6,
                "left_hand": 0.6,
            },
        )
    )
    assert revision_receipt.accepted

    result = runtime.evaluate_hot_composition(
        target_effective_at=10.1,
        cutoff_at=10.0,
    )
    binding = result.evaluation_record.strategy_control_binding
    assert binding.revision.semantic_content_hash == (
        revision_receipt.revision.semantic_content_hash
    )
    assert binding.parameter_snapshot.selected_parameter_record_id is None
    np.testing.assert_allclose(
        result.composition_evaluation.composed_candidate
        .value("left_arm")
        .values[0],
        0.08,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        binding.parameter_snapshot.alpha_by_component["left_hand"],
        0.6,
    )
    assert result.composition_evaluation.reason.endswith(
        FORMAL_COMPONENT_STRATEGY_MODE_DISCRETE_OWNER_V1
    )


def test_revision_rejects_stale_cas_and_runtime_boundary_change() -> None:
    _, _, _, strategy = build_contract()
    runtime, _, _ = build_runtime(strategy, enable_hot_control=True)
    control = runtime.strategy_control
    assert control is not None
    previous = control.latest_revision

    stale = runtime.admit_strategy_revision(
        CompositionStrategyRevisionDraft(
            request_id="stale",
            expected_previous_revision_hash="not-current",
            strategy_mode=previous.strategy_mode,
            composition_spec=previous.composition_spec,
            strategy_spec=previous.strategy_spec,
            component_rule_set=previous.component_rule_set,
            dynamic_parameter_contract=(
                previous.dynamic_parameter_contract
            ),
            initial_alpha_by_component=(
                previous.initial_alpha_by_component
            ),
        )
    )
    assert not stale.accepted
    assert stale.rejection_code == "stale_previous_revision"

    topology_change = replace(
        previous.component_rule_set,
        rule_set_id="invalid-topology-change",
        rules=(previous.component_rule_set.rules[0],),
    )
    incompatible_spec = replace(
        previous.composition_spec,
        component_rule_set_hash=topology_change.semantic_content_hash,
    )
    incompatible_strategy = replace(
        previous.strategy_spec,
        composition_spec_hash=incompatible_spec.semantic_content_hash,
        implementation_contract_hashes=(
            topology_change.semantic_content_hash,
        ),
    )
    incompatible = runtime.admit_strategy_revision(
        CompositionStrategyRevisionDraft(
            request_id="topology-change",
            expected_previous_revision_hash=previous.semantic_content_hash,
            strategy_mode=previous.strategy_mode,
            composition_spec=incompatible_spec,
            strategy_spec=incompatible_strategy,
            component_rule_set=topology_change,
            dynamic_parameter_contract=None,
        )
    )
    assert not incompatible.accepted
    assert "fixed runtime compatibility boundary" in incompatible.message

    pending = runtime.admit_strategy_revision(
        CompositionStrategyRevisionDraft(
            request_id="one-pending",
            expected_previous_revision_hash=previous.semantic_content_hash,
            strategy_mode=previous.strategy_mode,
            composition_spec=previous.composition_spec,
            strategy_spec=previous.strategy_spec,
            component_rule_set=previous.component_rule_set,
            dynamic_parameter_contract=(
                previous.dynamic_parameter_contract
            ),
            initial_alpha_by_component=(
                previous.initial_alpha_by_component
            ),
        )
    )
    assert pending.accepted
    second_pending = runtime.admit_strategy_revision(
        CompositionStrategyRevisionDraft(
            request_id="second-pending",
            expected_previous_revision_hash=(
                pending.revision.semantic_content_hash
            ),
            strategy_mode=previous.strategy_mode,
            composition_spec=previous.composition_spec,
            strategy_spec=previous.strategy_spec,
            component_rule_set=previous.component_rule_set,
            dynamic_parameter_contract=(
                previous.dynamic_parameter_contract
            ),
            initial_alpha_by_component=(
                previous.initial_alpha_by_component
            ),
        )
    )
    assert not second_pending.accepted
    assert second_pending.rejection_code == "pending_revision_exists"


def test_ttl_hot_parameter_expires_to_revision_default() -> None:
    hand_spec, _, _, strategy = build_contract()
    runtime, _, _ = build_runtime(
        strategy,
        enable_hot_control=True,
        hot_parameter_lifecycle="ttl",
    )
    _admit_hot_control_inputs(runtime, hand_spec=hand_spec)
    receipt = runtime.admit_dynamic_parameters(
        _parameter_draft(
            runtime,
            sequence=0,
            alpha=1.0,
            effective_until=10.15,
        )
    )
    assert receipt.accepted
    before_expiry = runtime.evaluate_hot_composition(
        target_effective_at=10.1,
        cutoff_at=10.0,
    )
    after_expiry = runtime.evaluate_hot_composition(
        target_effective_at=10.2,
        cutoff_at=10.0,
    )
    assert before_expiry.composition_evaluation.composed_candidate.value(
        "left_arm"
    ).values[0] == pytest.approx(0.08)
    assert after_expiry.composition_evaluation.composed_candidate.value(
        "left_arm"
    ).values[0] == pytest.approx(0.0)
    assert (
        after_expiry.evaluation_record.strategy_control_binding
        .parameter_snapshot.selected_parameter_record_id
        is None
    )


def test_concurrent_hot_updates_never_expose_a_torn_component_vector() -> None:
    hand_spec, _, _, strategy = build_contract(
        maximum_decision_latency_sec=None,
    )
    runtime, _, _ = build_runtime(strategy, enable_hot_control=True)
    _admit_hot_control_inputs(runtime, hand_spec=hand_spec)
    observations: list[tuple[float, float, str]] = []

    def publish() -> None:
        for sequence in range(100):
            alpha = float(sequence % 11) / 10.0
            receipt = runtime.admit_dynamic_parameters(
                _parameter_draft(
                    runtime,
                    sequence=sequence,
                    alpha=alpha,
                )
            )
            assert receipt.accepted

    def evaluate() -> None:
        for _ in range(100):
            result = runtime.evaluate_hot_composition(
                target_effective_at=10.1,
                cutoff_at=10.0,
            )
            action = result.composition_evaluation.composed_candidate
            arm_alpha = float(action.value("left_arm").values[0]) / 0.08
            hand_alpha = float(action.value("left_hand").values[0])
            observations.append(
                (
                    arm_alpha,
                    hand_alpha,
                    result.composition_evaluation
                    .strategy_control_binding_hash,
                )
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (pool.submit(publish), pool.submit(evaluate))
        for future in futures:
            future.result()

    assert len(observations) == 100
    for arm_alpha, hand_alpha, binding_hash in observations:
        assert arm_alpha == pytest.approx(hand_alpha, abs=2e-6)
        assert binding_hash
