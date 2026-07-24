from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from control_composition.composition import (
    CompositionContext,
    DiscreteOwnerPoseMixingStrategy,
    HandQposMixingStrategySpec,
    LinearHandQposInterpolationStrategy,
    PoseMixingContext,
    PoseMixingStrategySpec,
    WeightedSE3PoseFusionStrategy,
    assemble_compound_component_action,
    build_hand_qpos_mixing_decision,
    build_hand_qpos_mixing_input_snapshot,
    build_pose_mixing_decision,
    build_pose_mixing_input_snapshot,
    evaluate_component_mixing_shadow,
)
from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    BIMANUAL_EE_POSE_HAND16_V1,
    ActionComponentCatalog,
    ActionComponentSpec,
    ActionProposalDraft,
    ActionArbiter,
    AuthorityEventDraft,
    AuthorityPolicySpec,
    AuthorityStore,
    DecisionScheduleSpec,
    DecisionScheduler,
    HandActionSpec,
    ProposalBuffer,
    RobotAction,
)
from robot_motion.action.arbitration_input import capture_arbitration_input
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.execution_gate import ExecutionGateStore
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry


class Clock:
    def __init__(self, value: float = 5.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return float(self.value)


def _pose(x: float, *, source: str) -> RobotAction:
    left = np.asarray([x, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    right = np.asarray([x, -0.1, 0.4, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return RobotAction.bimanual_ee_pose(
        left_arm=left,
        right_arm=right,
        source=source,
    )


def _capture(
    *,
    owner: str = "replay-main",
    include_hands: bool = False,
    action_by_producer=None,
):
    clock = Clock()
    coordinator = ControlAdmissionCoordinator(
        "robot-0",
        clock_domain=ClockDomainDescriptor(
            "control.monotonic",
            "session-0",
            "boot-0",
            0,
        ),
        clock=clock,
    )
    epochs = ProducerEpochRegistry(
        runtime_session_id="session-0",
        coordinator=coordinator,
    )
    for producer in ("human-main", "replay-main"):
        epochs.activate(
            producer_id=producer,
            producer_epoch=1,
            produced_at=1.0,
            produced_clock_domain_id=f"{producer}.clock",
            reason="startup",
        )
    components = [
        ActionComponentSpec("left_arm", "arms", "left_arm", coupled=True),
        ActionComponentSpec("right_arm", "arms", "right_arm", coupled=True),
    ]
    if include_hands:
        components.extend(
            (
                ActionComponentSpec("left_hand", "hands", "left_hand", coupled=False),
                ActionComponentSpec("right_hand", "hands", "right_hand", coupled=False),
            )
        )
    catalog = ActionComponentCatalog(
        "pose-arms-v1",
        BIMANUAL_EE_POSE_V1,
        tuple(components),
    )
    proposals = ProposalBuffer(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        capacity=16,
    )
    for sequence, (producer, kind, x) in enumerate(
        (
            ("human-main", "human", 0.35),
            ("replay-main", "replay", 0.45),
        )
    ):
        receipt = proposals.admit(
            ActionProposalDraft(
                runtime_session_id="session-0",
                control_domain_id="robot-0",
                embodiment_instance_id="rm65-pair-0",
                producer_kind=kind,
                producer_id=producer,
                producer_epoch=1,
                sequence=sequence,
                action=(
                    _pose(x, source=kind)
                    if action_by_producer is None
                    else action_by_producer[producer]
                ),
                produced_at=4.0,
                produced_clock_domain_id=f"{producer}.clock",
                effective_from=10.0,
                effective_until=11.0,
                target_effective_at=10.5,
                effective_clock_domain_id="control.monotonic",
                component_schema_hash=catalog.semantic_content_hash,
                component_coverage=(
                    "left_arm",
                    "right_arm",
                    *(("left_hand", "right_hand") if include_hands else ()),
                ),
                selection_groups=("arms",),
            )
        )
        assert receipt.accepted
    policy = AuthorityPolicySpec(
        policy_id="mixed-pose-v1",
        priority_by_group={"arms": {"kind:human": 20, "kind:replay": 10}},
        fallback_by_group={"arms": "no_action"},
        takeover_allowed_producers=("human-main", "replay-main"),
    )
    authority = AuthorityStore(
        runtime_session_id="session-0",
        coordinator=coordinator,
        policy=policy,
        initial_mode="hybrid",
    )
    authority.admit(
        AuthorityEventDraft(
            runtime_session_id="session-0",
            control_domain_id="robot-0",
            sequence=0,
            event_type="takeover",
            produced_at=5.0,
            produced_clock_domain_id="control.monotonic",
            selection_group_id="arms",
            owner_producer_id=owner,
            reason="fixture owner",
        )
    )
    gate = ExecutionGateStore(
        runtime_session_id="session-0",
        coordinator=coordinator,
    )
    clock.value = 10.5
    snapshot = capture_arbitration_input(
        coordinator=coordinator,
        proposal_buffer=proposals,
        producer_epochs=epochs,
        authority_store=authority,
        execution_gate_store=gate,
        scheduler=DecisionScheduler(
            DecisionScheduleSpec(spec_id="30hz", decision_hz=30.0),
            clock=clock,
        ),
        embodiment_instance_id="rm65-pair-0",
        component_catalog=catalog,
        active_selection_groups=("arms",),
    )
    assert snapshot is not None
    return snapshot


def _mixing_snapshot(
    arbitration_input,
    *,
    contexts=(),
    component_dependencies=None,
):
    return build_pose_mixing_input_snapshot(
        arbitration_input,
        source_slot_by_producer={
            "human-main": "human-slot",
            "replay-main": "replay-slot",
        },
        clock_domain_id="control.monotonic",
        contexts=tuple(contexts),
        component_dependencies=component_dependencies,
    )


def _compound_arbitration_input_with_left_hand():
    arbitration_input = _capture(include_hands=True)
    for item, value in zip(
        arbitration_input.proposal_snapshot.candidates,
        (0.2, 0.8),
    ):
        proposal = item.proposal
        base = proposal.action
        hand = np.zeros(6, dtype=np.float32)
        hand[0] = value
        object.__setattr__(
            proposal,
            "action",
            RobotAction(
                action_space=BIMANUAL_EE_POSE_V1,
                arm_pose_by_side=base.arm_pose_by_side,
                hand_qpos_by_side={"left": hand},
                active_effectors=("left_arm", "right_arm", "left_hand"),
                frame_by_effector=base.frame_by_effector,
                controlled_frame_by_effector=base.controlled_frame_by_effector,
                source=base.source,
                metadata=base.metadata,
            ),
        )
    return arbitration_input


def test_discrete_owner_strategy_is_pose_passthrough_equivalent_to_arbiter() -> None:
    arbitration_input = _capture(owner="replay-main")
    normal = ActionArbiter().arbitrate(arbitration_input)
    snapshot = _mixing_snapshot(arbitration_input)
    mixed = DiscreteOwnerPoseMixingStrategy().decide(snapshot)

    assert normal.output_status == "selected"
    assert mixed.status == "selected"
    assert mixed.selected_source_slot_ids == ("replay-slot",)
    assert mixed.selected_proposal_ids == (normal.selected_by_group["arms"],)
    assert mixed.output_action is not None
    assert mixed.output_action.action_space == BIMANUAL_EE_POSE_V1
    np.testing.assert_array_equal(
        mixed.output_action.arm_pose_by_side["left"],
        normal.composed_action.arm_pose_by_side["left"],
    )


def test_context_snapshot_is_frozen_and_rejects_late_information() -> None:
    arbitration_input = _capture()
    payload = {"operator_label": "prefer_human", "scores": [0.1, 0.9]}
    context = PoseMixingContext(
        context_id="label:7",
        context_kind="label",
        source_fact_id="dataset-label:7",
        source_fact_hash="sha256:label-7",
        payload_contract_hash="sha256:label-contract-v1",
        semantic_payload=payload,
        available_at=10.4,
        available_clock_domain_id="control.monotonic",
        control_admission_sequence=(
            arbitration_input.proposal_snapshot.cutoff_admission_sequence
        ),
        effective_from=10.0,
        effective_until=11.0,
    )
    snapshot = _mixing_snapshot(arbitration_input, contexts=(context,))
    payload["operator_label"] = "prefer_replay"
    payload["scores"][1] = 0.0

    assert snapshot.contexts[0].semantic_payload["operator_label"] == "prefer_human"
    assert tuple(snapshot.contexts[0].semantic_payload["scores"]) == (0.1, 0.9)

    late = PoseMixingContext(
        context_id="future:late",
        context_kind="future_action",
        source_fact_id="plan:late",
        source_fact_hash="sha256:plan-late",
        payload_contract_hash="sha256:future-plan-v1",
        semantic_payload={"keyframes": (3, 4)},
        available_at=10.6,
        available_clock_domain_id="control.monotonic",
        control_admission_sequence=(
            arbitration_input.proposal_snapshot.cutoff_admission_sequence
        ),
    )
    with pytest.raises(ValueError, match="not available at the decision cut"):
        _mixing_snapshot(arbitration_input, contexts=(late,))


def test_generic_context_accepts_new_modalities_and_enforces_target_ttl() -> None:
    arbitration_input = _capture()
    vision = CompositionContext(
        context_id="vision:embedding:7",
        context_kind="vision.embedding.v2",
        source_fact_id="camera-feature:7",
        source_fact_hash="sha256:camera-feature-7",
        payload_contract_hash="sha256:vision-embedding-v2",
        semantic_payload={"blob_ref": "features/7.npy", "quality": 0.92},
        available_at=10.4,
        available_clock_domain_id="control.monotonic",
        control_admission_sequence=(
            arbitration_input.proposal_snapshot.cutoff_admission_sequence
        ),
        effective_from=10.0,
        effective_until=11.0,
        blob_refs=("features/7.npy",),
    )
    snapshot = _mixing_snapshot(arbitration_input, contexts=(vision,))
    candidate = snapshot.candidate_by_source_slot["human-slot"]
    decision = build_pose_mixing_decision(
        snapshot,
        spec=PoseMixingStrategySpec(
            strategy_id="fixture-vision-router-v1",
            algorithm="select_with_frozen_vision_context",
            accepted_context_kinds=("vision.embedding.v2",),
        ),
        status="selected",
        reason="vision_context_admitted",
        selected=(candidate,),
        used_context_ids=(vision.context_id,),
        output_action=candidate.action,
    )

    assert decision.used_context_ids == ("vision:embedding:7",)
    assert snapshot.contexts[0].semantic_payload["quality"] == pytest.approx(0.92)

    stale = CompositionContext(
        context_id="state:stale",
        context_kind="state",
        source_fact_id="state:6",
        source_fact_hash="sha256:state-6",
        payload_contract_hash="sha256:state-v1",
        semantic_payload={"phase": "approach"},
        available_at=10.4,
        available_clock_domain_id="control.monotonic",
        control_admission_sequence=(
            arbitration_input.proposal_snapshot.cutoff_admission_sequence
        ),
        effective_from=9.0,
        effective_until=10.0,
    )
    with pytest.raises(ValueError, match="not effective at the decision target"):
        _mixing_snapshot(arbitration_input, contexts=(stale,))


class _LabelRoutingStrategy:
    def __init__(self) -> None:
        self._spec = PoseMixingStrategySpec(
            strategy_id="fixture-label-router-v1",
            algorithm="select_slot_from_admitted_label",
            accepted_context_kinds=("label",),
        )

    @property
    def spec(self) -> PoseMixingStrategySpec:
        return self._spec

    def decide(self, snapshot):
        label = next(item for item in snapshot.contexts if item.context_kind == "label")
        slot = str(label.semantic_payload["source_slot_id"])
        candidate = snapshot.candidate_by_source_slot[slot]
        return build_pose_mixing_decision(
            snapshot,
            spec=self.spec,
            status="selected",
            reason="admitted_label_route",
            selected=(candidate,),
            used_context_ids=(label.context_id,),
            output_action=candidate.action,
        )


def test_strategy_is_replaceable_and_may_use_admitted_label_context() -> None:
    arbitration_input = _capture(owner="replay-main")
    label = PoseMixingContext(
        context_id="label:human",
        context_kind="label",
        source_fact_id="operator-label:1",
        source_fact_hash="sha256:operator-label-1",
        payload_contract_hash="sha256:source-route-label-v1",
        semantic_payload={"source_slot_id": "human-slot"},
        available_at=10.4,
        available_clock_domain_id="control.monotonic",
        control_admission_sequence=(
            arbitration_input.proposal_snapshot.cutoff_admission_sequence
        ),
    )
    snapshot = _mixing_snapshot(arbitration_input, contexts=(label,))
    decision = _LabelRoutingStrategy().decide(snapshot)

    assert decision.selected_source_slot_ids == ("human-slot",)
    assert decision.used_context_ids == ("label:human",)
    assert decision.output_action is not None
    assert decision.output_action.source == "human"


def _rotated_pose(
    x: float,
    rotation_degrees: float,
    *,
    source: str,
    quaternion_sign: float = 1.0,
    frame_by_effector=None,
) -> RobotAction:
    quaternion = Rotation.from_euler(
        "z",
        rotation_degrees,
        degrees=True,
    ).as_quat()
    quaternion *= float(quaternion_sign)
    left = np.concatenate(([x, 0.1, 0.4], quaternion)).astype(np.float32)
    right = np.concatenate(([x, -0.1, 0.4], quaternion)).astype(np.float32)
    return RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={"left": left, "right": right},
        frame_by_effector=dict(frame_by_effector or {}),
        source=source,
    )


def test_weighted_se3_pose_fusion_uses_shortest_rotation_path() -> None:
    arbitration_input = _capture(
        action_by_producer={
            "human-main": _rotated_pose(0.35, 0.0, source="human"),
            "replay-main": _rotated_pose(0.45, 90.0, source="replay"),
        }
    )
    snapshot = _mixing_snapshot(arbitration_input)
    strategy = WeightedSE3PoseFusionStrategy(
        weights_by_source_slot={
            "human-slot": 0.5,
            "replay-slot": 0.5,
        },
        max_rotation_disagreement_rad=np.deg2rad(100.0),
    )

    first = strategy.decide(snapshot)
    second = strategy.decide(snapshot)

    assert first.decision_id == second.decision_id
    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.status == "selected"
    assert first.selected_source_slot_ids == ("human-slot", "replay-slot")
    assert first.output_action is not None
    assert first.output_action.timestamp == snapshot.target_effective_at
    np.testing.assert_allclose(
        first.output_action.arm_pose_by_side["left"][:3],
        [0.4, 0.1, 0.4],
        atol=1e-7,
    )
    fused_rotation = Rotation.from_quat(
        first.output_action.arm_pose_by_side["left"][3:7]
    )
    assert fused_rotation.magnitude() == pytest.approx(
        np.deg2rad(45.0),
        abs=1e-6,
    )
    assert first.output_action.source == (
        "pose_fusion:weighted_se3_pose_fusion_v1"
    )


@pytest.mark.parametrize("component", ("left_arm", "right_arm"))
def test_pose_mixing_supports_one_explicit_arm_component(component: str) -> None:
    dependencies = (component,)
    snapshot = _mixing_snapshot(
        _capture(owner="human-main"),
        component_dependencies=dependencies,
    )
    discrete = DiscreteOwnerPoseMixingStrategy(
        component_dependencies=dependencies
    ).decide(snapshot)
    fused = WeightedSE3PoseFusionStrategy(
        weights_by_source_slot={
            "human-slot": 0.25,
            "replay-slot": 0.75,
        },
        component_dependencies=dependencies,
    ).decide(snapshot)
    side = component.removesuffix("_arm")

    assert discrete.status == "selected"
    assert discrete.output_action is not None
    assert discrete.output_action.active_effectors == dependencies
    assert set(discrete.output_action.arm_pose_by_side) == {side}
    assert fused.status == "selected"
    assert fused.output_action is not None
    assert fused.output_action.active_effectors == dependencies
    assert set(fused.output_action.arm_pose_by_side) == {side}
    np.testing.assert_allclose(
        fused.output_action.arm_pose_by_side[side][0],
        0.425,
        atol=1e-7,
    )


def test_pose_strategy_rejects_component_manifest_mismatch() -> None:
    snapshot = _mixing_snapshot(
        _capture(),
        component_dependencies=("left_arm",),
    )
    with pytest.raises(ValueError, match="dependency mismatch"):
        WeightedSE3PoseFusionStrategy(
            weights_by_source_slot={
                "human-slot": 0.5,
                "replay-slot": 0.5,
            },
        ).decide(snapshot)


def test_pose_mixing_rejects_non_arm_component_subset() -> None:
    with pytest.raises(ValueError, match="only left_arm/right_arm"):
        _mixing_snapshot(
            _capture(),
            component_dependencies=("left_hand",),
        )


def test_weighted_se3_pose_fusion_handles_antipodal_quaternions() -> None:
    arbitration_input = _capture(
        action_by_producer={
            "human-main": _rotated_pose(
                0.4,
                0.0,
                source="human",
                quaternion_sign=1.0,
            ),
            "replay-main": _rotated_pose(
                0.4,
                0.0,
                source="replay",
                quaternion_sign=-1.0,
            ),
        }
    )
    decision = WeightedSE3PoseFusionStrategy(
        weights_by_source_slot={
            "human-slot": 0.25,
            "replay-slot": 0.75,
        }
    ).decide(_mixing_snapshot(arbitration_input))

    assert decision.status == "selected"
    assert decision.output_action is not None
    quaternion = decision.output_action.arm_pose_by_side["left"][3:7]
    assert abs(float(quaternion[3])) == pytest.approx(1.0, abs=1e-7)
    np.testing.assert_allclose(quaternion[:3], 0.0, atol=1e-7)


def test_weighted_se3_pose_fusion_fails_closed_on_source_conflict() -> None:
    arbitration_input = _capture(
        action_by_producer={
            "human-main": _rotated_pose(0.0, 0.0, source="human"),
            "replay-main": _rotated_pose(0.5, 0.0, source="replay"),
        }
    )
    decision = WeightedSE3PoseFusionStrategy(
        weights_by_source_slot={
            "human-slot": 0.5,
            "replay-slot": 0.5,
        },
        max_translation_disagreement_m=0.1,
    ).decide(_mixing_snapshot(arbitration_input))

    assert decision.status == "invalid"
    assert decision.output_action is None
    assert "translation_disagreement" in decision.reason


def test_weighted_se3_pose_fusion_rejects_implicit_frame_conversion() -> None:
    arbitration_input = _capture(
        action_by_producer={
            "human-main": _rotated_pose(0.35, 0.0, source="human"),
            "replay-main": _rotated_pose(
                0.45,
                0.0,
                source="replay",
                frame_by_effector={
                    "left_arm": "canonical_desk",
                    "right_arm": "canonical_desk",
                },
            ),
        }
    )
    decision = WeightedSE3PoseFusionStrategy(
        weights_by_source_slot={
            "human-slot": 0.5,
            "replay-slot": 0.5,
        },
    ).decide(_mixing_snapshot(arbitration_input))

    assert decision.status == "invalid"
    assert decision.reason == "pose_fusion_frame_or_action_space_mismatch"


def test_weighted_se3_pose_fusion_missing_source_policy_is_explicit() -> None:
    snapshot = _mixing_snapshot(_capture())
    fail_closed = WeightedSE3PoseFusionStrategy(
        weights_by_source_slot={
            "human-slot": 0.25,
            "not-connected": 0.75,
        },
    ).decide(snapshot)
    renormalized = WeightedSE3PoseFusionStrategy(
        weights_by_source_slot={
            "human-slot": 0.25,
            "not-connected": 0.75,
        },
        missing_candidate_behavior="renormalize_available",
    ).decide(snapshot)

    assert fail_closed.status == "no_action"
    assert "not-connected" in fail_closed.reason
    assert renormalized.status == "selected"
    assert renormalized.selected_source_slot_ids == ("human-slot",)
    assert renormalized.output_action is not None
    np.testing.assert_array_equal(
        renormalized.output_action.arm_pose_by_side["left"],
        snapshot.candidate_by_source_slot["human-slot"].action.arm_pose_by_side[
            "left"
        ],
    )


def test_pose_mixing_rejects_qpos_candidate_at_the_boundary() -> None:
    arbitration_input = _capture()
    candidate = arbitration_input.proposal_snapshot.candidates[0].proposal
    object.__setattr__(
        candidate,
        "action",
        RobotAction.bimanual_qpos(
            left_arm=np.zeros(6, dtype=np.float32),
            right_arm=np.zeros(6, dtype=np.float32),
        ),
    )
    with pytest.raises(ValueError, match="requires bimanual_ee_pose_v1"):
        _mixing_snapshot(arbitration_input)


def test_arm_pose_mixing_rejects_silently_carried_hand_qpos() -> None:
    arbitration_input = _capture()
    candidate = arbitration_input.proposal_snapshot.candidates[0].proposal
    pose_action = candidate.action
    object.__setattr__(
        candidate,
        "action",
        RobotAction(
            action_space=BIMANUAL_EE_POSE_V1,
            arm_pose_by_side=pose_action.arm_pose_by_side,
            hand_qpos_by_side={
                "left": np.zeros(6, dtype=np.float32),
                "right": np.zeros(6, dtype=np.float32),
            },
            active_effectors=(
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
            ),
            frame_by_effector=pose_action.frame_by_effector,
            controlled_frame_by_effector=(
                pose_action.controlled_frame_by_effector
            ),
        ),
    )
    with pytest.raises(ValueError, match="excludes hand/end-effector payloads"):
        _mixing_snapshot(arbitration_input)


def _hand_input_snapshot(
    *,
    hand_spec: HandActionSpec,
    action_space: str = BIMANUAL_EE_POSE_V1,
    contexts=(),
):
    arbitration_input = _capture()
    values = (0.2, 0.8)
    for item, value in zip(arbitration_input.proposal_snapshot.candidates, values):
        proposal = item.proposal
        left_arm = proposal.action.arm_pose_by_side["left"]
        right_arm = proposal.action.arm_pose_by_side["right"]
        if action_space == BIMANUAL_EE_POSE_HAND16_V1:
            left_hand = np.full(16, np.nan, dtype=np.float32)
            left_hand[: hand_spec.active_dof] = np.linspace(
                value,
                value + 0.1,
                hand_spec.active_dof,
                dtype=np.float32,
            )
            action = RobotAction.bimanual_ee_pose_hand16(
                left_arm=left_arm,
                right_arm=right_arm,
                left_hand=left_hand,
                active_effectors=("left_arm", "right_arm", "left_hand"),
                source=proposal.action.source,
            )
        else:
            left_hand = np.zeros(6, dtype=np.float32)
            left_hand[0] = value
            action = RobotAction.bimanual_ee_pose(
                left_arm=left_arm,
                right_arm=right_arm,
                left_hand=left_hand,
                active_effectors=("left_arm", "right_arm", "left_hand"),
                source=proposal.action.source,
            )
        object.__setattr__(proposal, "action", action)
    return build_hand_qpos_mixing_input_snapshot(
        arbitration_input,
        source_slot_by_producer={
            "human-main": "human-slot",
            "replay-main": "replay-slot",
        },
        side="left",
        hand_spec=hand_spec,
        clock_domain_id="control.monotonic",
        contexts=tuple(contexts),
    )


def test_hand_qpos_interpolation_uses_active_slots_and_preserves_lineage() -> None:
    hand_spec = HandActionSpec.dmtac_gripper("left")
    snapshot = _hand_input_snapshot(hand_spec=hand_spec)
    strategy = LinearHandQposInterpolationStrategy(
        hand_spec,
        action_space=BIMANUAL_EE_POSE_V1,
        weights_by_source_slot={"human-slot": 0.25, "replay-slot": 0.75},
    )
    decision = strategy.decide(snapshot)

    assert decision.status == "selected"
    assert decision.selected_source_slot_ids == ("human-slot", "replay-slot")
    assert decision.selected_proposal_ids == (
        snapshot.candidate_by_source_slot["human-slot"].proposal.proposal_id,
        snapshot.candidate_by_source_slot["replay-slot"].proposal.proposal_id,
    )
    assert decision.output_hand_qpos is not None
    np.testing.assert_allclose(decision.output_hand_qpos, [0.65, 0, 0, 0, 0, 0])
    assert decision.output_hand_qpos_hash is not None
    assert decision.input_snapshot_id == snapshot.snapshot_id
    assert decision.input_snapshot_hash == snapshot.semantic_content_hash
    assert decision.semantic_content_hash


def test_hand_qpos_strategy_can_consume_the_same_generic_context_contract() -> None:
    hand_spec = HandActionSpec.dmtac_gripper("left")
    arbitration_input = _capture()
    context = CompositionContext(
        context_id="state:grasp-phase:1",
        context_kind="task_state",
        source_fact_id="task-state:1",
        source_fact_hash="sha256:task-state-1",
        payload_contract_hash="sha256:task-state-v1",
        semantic_payload={"phase": "pregrasp"},
        available_at=10.4,
        available_clock_domain_id="control.monotonic",
        control_admission_sequence=(
            arbitration_input.proposal_snapshot.cutoff_admission_sequence
        ),
        effective_from=10.0,
        effective_until=11.0,
    )
    snapshot = _hand_input_snapshot(
        hand_spec=hand_spec,
        contexts=(context,),
    )
    candidate = snapshot.candidate_by_source_slot["human-slot"]
    decision = build_hand_qpos_mixing_decision(
        snapshot,
        spec=HandQposMixingStrategySpec(
            strategy_id="fixture-contextual-qpos-v1",
            algorithm="select_human_qpos_in_pregrasp",
            component_id=snapshot.component_id,
            action_space=snapshot.action_space,
            hand_action_spec_hash=snapshot.hand_action_spec_hash,
            weights_by_source_slot={"human-slot": 1.0},
            accepted_context_kinds=("task_state",),
        ),
        status="selected",
        reason="task_state_pregrasp",
        selected=(candidate,),
        normalized_weights={"human-slot": 1.0},
        output_hand_qpos=candidate.action.hand_qpos_by_side["left"],
        used_context_ids=(context.context_id,),
    )

    assert snapshot.contexts == (context,)
    assert decision.used_context_ids == (context.context_id,)


def test_hand_qpos_interpolation_preserves_q16_nan_padding() -> None:
    hand_spec = HandActionSpec.q16("left", active_dof=2)
    snapshot = _hand_input_snapshot(
        hand_spec=hand_spec,
        action_space=BIMANUAL_EE_POSE_HAND16_V1,
    )
    strategy = LinearHandQposInterpolationStrategy(
        hand_spec,
        action_space=BIMANUAL_EE_POSE_HAND16_V1,
        weights_by_source_slot={"human-slot": 0.5, "replay-slot": 0.5},
    )
    decision = strategy.decide(snapshot)

    assert decision.status == "selected"
    assert decision.output_hand_qpos is not None
    np.testing.assert_allclose(decision.output_hand_qpos[:2], [0.5, 0.6])
    assert np.all(np.isnan(decision.output_hand_qpos[2:]))


def test_hand_qpos_interpolation_fails_closed_on_missing_weighted_candidate() -> None:
    hand_spec = HandActionSpec.dmtac_gripper("left")
    snapshot = _hand_input_snapshot(hand_spec=hand_spec)
    strategy = LinearHandQposInterpolationStrategy(
        hand_spec,
        action_space=BIMANUAL_EE_POSE_V1,
        weights_by_source_slot={"human-slot": 0.5, "missing-slot": 0.5},
    )

    decision = strategy.decide(snapshot)
    assert decision.status == "no_action"
    assert decision.output_hand_qpos is None
    assert "missing-slot" in decision.reason


def test_component_shadow_projects_arm_from_compound_action_and_joins_hand_qpos() -> None:
    arbitration_input = _compound_arbitration_input_with_left_hand()
    hand_spec = HandActionSpec.dmtac_gripper("left")
    decision = evaluate_component_mixing_shadow(
        arbitration_input,
        source_slot_by_producer={
            "human-main": "human-slot",
            "replay-main": "replay-slot",
        },
        clock_domain_id="control.monotonic",
        arm_strategy=DiscreteOwnerPoseMixingStrategy(),
        hand_strategy_by_side={
            "left": LinearHandQposInterpolationStrategy(
                hand_spec,
                action_space=BIMANUAL_EE_POSE_V1,
                weights_by_source_slot={"human-slot": 0.25, "replay-slot": 0.75},
            )
        },
        hand_spec_by_side={"left": hand_spec},
    )

    assert decision.status == "selected"
    assert decision.output_action is not None
    assert decision.output_action.active_effectors == (
        "left_arm",
        "right_arm",
        "left_hand",
    )
    np.testing.assert_allclose(
        decision.output_action.hand_qpos_by_side["left"],
        [0.65, 0, 0, 0, 0, 0],
    )
    assert not decision.arm_decision.output_action.hand_qpos_by_side
    assert decision.component_decision_hash_by_id["left_hand"]
    assert decision.output_action_hash


def test_component_shadow_can_join_se3_arm_fusion_with_hand_qpos_interpolation() -> None:
    arbitration_input = _compound_arbitration_input_with_left_hand()
    hand_spec = HandActionSpec.dmtac_gripper("left")
    decision = evaluate_component_mixing_shadow(
        arbitration_input,
        source_slot_by_producer={
            "human-main": "human-slot",
            "replay-main": "replay-slot",
        },
        clock_domain_id="control.monotonic",
        arm_strategy=WeightedSE3PoseFusionStrategy(
            weights_by_source_slot={
                "human-slot": 0.5,
                "replay-slot": 0.5,
            },
        ),
        hand_strategy_by_side={
            "left": LinearHandQposInterpolationStrategy(
                hand_spec,
                action_space=BIMANUAL_EE_POSE_V1,
                weights_by_source_slot={
                    "human-slot": 0.25,
                    "replay-slot": 0.75,
                },
            ),
        },
        hand_spec_by_side={"left": hand_spec},
    )

    assert decision.status == "selected"
    assert decision.output_action is not None
    np.testing.assert_allclose(
        decision.output_action.arm_pose_by_side["left"][:3],
        [0.4, 0.1, 0.4],
    )
    np.testing.assert_allclose(
        decision.output_action.hand_qpos_by_side["left"],
        [0.65, 0, 0, 0, 0, 0],
    )
    assert decision.arm_decision.selected_source_slot_ids == (
        "human-slot",
        "replay-slot",
    )


def test_component_shadow_fails_closed_when_hand_component_is_missing() -> None:
    arbitration_input = _compound_arbitration_input_with_left_hand()
    hand_spec = HandActionSpec.dmtac_gripper("left")
    strategy = LinearHandQposInterpolationStrategy(
        hand_spec,
        action_space=BIMANUAL_EE_POSE_V1,
        weights_by_source_slot={"human-slot": 0.5, "missing-slot": 0.5},
    )
    decision = evaluate_component_mixing_shadow(
        arbitration_input,
        source_slot_by_producer={
            "human-main": "human-slot",
            "replay-main": "replay-slot",
        },
        clock_domain_id="control.monotonic",
        arm_strategy=DiscreteOwnerPoseMixingStrategy(),
        hand_strategy_by_side={"left": strategy},
        hand_spec_by_side={"left": hand_spec},
    )

    assert decision.status == "no_action"
    assert decision.output_action is None
    assert decision.reason == "hand_strategy_no_action"


def test_compound_assembly_is_strict_and_does_not_duplicate_hand_ownership() -> None:
    base = _pose(0.25, source="arm")
    assembled = assemble_compound_component_action(
        base,
        hand_qpos_by_side={"left": np.asarray([0.3] * 6, dtype=np.float32)},
        expected_action_space=BIMANUAL_EE_POSE_V1,
        timestamp=12.0,
    )

    assert assembled.active_effectors == ("left_arm", "right_arm", "left_hand")
    assert assembled.timestamp == 12.0
    np.testing.assert_allclose(assembled.hand_qpos_by_side["left"], [0.3] * 6)
    with pytest.raises(ValueError, match="duplicate hand ownership"):
        assemble_compound_component_action(
            assembled,
            hand_qpos_by_side={"left": np.asarray([0.4] * 6, dtype=np.float32)},
        )
