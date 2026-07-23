from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    BIMANUAL_QPOS_V1,
    ActionComponentCatalog,
    ActionComponentSpec,
    CompositionInputDraft,
    CompositionInputStore,
    CompositionSpec,
    DeterministicComposer,
    ProposalBuffer,
    RobotAction,
    build_composite_action_proposal_draft,
    build_composition_input_snapshot,
    project_composition_inputs,
)
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry
from robot_motion.action.proposal import ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V2
from teleop_runtime.control_trace.projection import (
    project_composition_input,
    project_composition_input_admission,
    project_composition_input_snapshot,
    project_composition_result,
)


class Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def runtime():
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
        runtime_session_id="session-0", coordinator=coordinator
    )
    for producer in ("slow-base", "fast-residual", "composer"):
        epochs.activate(
            producer_id=producer,
            producer_epoch=1,
            produced_at=1.0,
            produced_clock_domain_id=f"{producer}.monotonic",
            reason="startup",
        )
    store = CompositionInputStore(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        capacity=32,
    )
    proposals = ProposalBuffer(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        capacity=16,
    )
    catalog = ActionComponentCatalog(
        catalog_id="left-arm-v1",
        action_space=BIMANUAL_QPOS_V1,
        components=(ActionComponentSpec("left_arm", "left_arm", "left_arm"),),
    )
    return clock, coordinator, epochs, store, proposals, catalog


def qpos_action(value, *, include_hand: bool = False) -> RobotAction:
    arm = np.full(6, value, dtype=np.float32)
    hand = {"left": np.full(6, value, dtype=np.float32)} if include_hand else {}
    return RobotAction(
        action_space=BIMANUAL_QPOS_V1,
        arm_qpos_by_side={"left": arm},
        hand_qpos_by_side=hand,
        source="fixture",
    )


def pose_action(x: float, yaw_degrees: float = 0.0) -> RobotAction:
    quaternion = Rotation.from_euler(
        "z",
        math.radians(yaw_degrees),
    ).as_quat()
    return RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={
            side: np.asarray(
                [x, y, 0.3, *quaternion],
                dtype=np.float32,
            )
            for side, y in (("left", -0.1), ("right", 0.1))
        },
        active_effectors=("left_arm", "right_arm"),
        frame_by_effector={
            "left_arm": "left_arm_base",
            "right_arm": "right_arm_base",
        },
        controlled_frame_by_effector={
            "left_arm": "tool_tcp",
            "right_arm": "tool_tcp",
        },
        source="fixture",
    )


def input_draft(
    *,
    producer: str,
    role: str,
    action: RobotAction,
    sequence: int = 0,
    context_ids=(),
) -> CompositionInputDraft:
    return CompositionInputDraft(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        producer_id=producer,
        producer_epoch=1,
        sequence=sequence,
        input_role=role,
        action=action,
        produced_at=9.9,
        produced_clock_domain_id=f"{producer}.monotonic",
        effective_from=10.0,
        effective_until=11.0,
        effective_clock_domain_id="control.monotonic",
        context_ids=tuple(context_ids),
        observation_cut_id=f"cut:{producer}:{sequence}",
        policy_output_id=f"output:{producer}:{sequence}",
    )


def residual_spec(*, missing: str = "zero_residual") -> CompositionSpec:
    return CompositionSpec(
        spec_id="base-plus-residual-v1",
        algebra="base_plus_bounded_residual",
        input_roles=("base", "residual"),
        required_roles=("base",),
        missing_residual_behavior=missing,
        residual_limit_by_component={"left_arm": 0.05},
    )


def test_internal_inputs_are_not_authority_candidates_and_only_composite_is_proposed() -> None:
    _, coordinator, epochs, store, proposals, catalog = runtime()
    base = store.admit(
        input_draft(
            producer="slow-base",
            role="base",
            action=qpos_action(0.1),
            context_ids=("slow-context-7",),
        )
    )
    residual = store.admit(
        input_draft(
            producer="fast-residual",
            role="residual",
            action=qpos_action(0.2),
            context_ids=("slow-context-7",),
        )
    )
    assert base.accepted and residual.accepted
    assert len(proposals) == 0
    cut = coordinator.capture_cut(cutoff_at=10.0).cut
    snapshot = build_composition_input_snapshot(
        project_composition_inputs(store, cut),
        producer_epochs=epochs.snapshot(cut),
        spec=residual_spec(),
        target_effective_at=10.5,
    )
    result = DeterministicComposer().compose(snapshot, spec=residual_spec())
    assert result.output_status == "ready"
    assert result.composed_action is not None
    np.testing.assert_allclose(
        result.composed_action.arm_qpos_by_side["left"],
        np.full(6, 0.15, dtype=np.float32),
    )
    proposal = proposals.admit(
        build_composite_action_proposal_draft(
            result,
            runtime_session_id="session-0",
            control_domain_id="robot-0",
            embodiment_instance_id="rm65-pair-0",
            producer_id="composer",
            producer_epoch=1,
            sequence=0,
            produced_at=10.0,
            produced_clock_domain_id="control.monotonic",
            effective_from=10.0,
            effective_until=11.0,
            effective_clock_domain_id="control.monotonic",
            component_schema_hash=catalog.semantic_content_hash,
            component_coverage=("left_arm",),
            selection_groups=("left_arm",),
            policy_node_id="fast-composer",
        )
    ).proposal
    assert proposal is not None
    assert len(proposals) == 1
    assert proposal.composition_input_ids == result.input_ids
    assert proposal.composition_input_hashes == result.input_hashes
    assert proposal.composition_spec_hash == result.composition_spec_hash
    assert proposal.semantic_hash_domain == ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V2
    assert proposal.context_ids == ("slow-context-7",)


def test_snapshot_is_immutable_and_late_residual_only_changes_next_cut() -> None:
    clock, coordinator, epochs, store, _, _ = runtime()
    store.admit(
        input_draft(producer="slow-base", role="base", action=qpos_action(0.1))
    )
    first = store.admit(
        input_draft(
            producer="fast-residual", role="residual", action=qpos_action(0.01)
        )
    ).composition_input
    assert first is not None
    cut = coordinator.capture_cut(cutoff_at=10.0).cut
    projection = project_composition_inputs(store, cut)
    epoch_snapshot = epochs.snapshot(cut)
    clock.value = 10.01
    second = store.admit(
        input_draft(
            producer="fast-residual",
            role="residual",
            action=qpos_action(0.02),
            sequence=1,
        )
    ).composition_input
    assert second is not None
    frozen = build_composition_input_snapshot(
        projection,
        producer_epochs=epoch_snapshot,
        spec=residual_spec(),
        target_effective_at=10.5,
    )
    assert frozen.selected_input_by_role["residual"].composition_input_id == (
        first.composition_input_id
    )
    next_cut = coordinator.capture_cut(cutoff_at=10.02).cut
    current = build_composition_input_snapshot(
        project_composition_inputs(store, next_cut),
        producer_epochs=epochs.snapshot(next_cut),
        spec=residual_spec(),
        target_effective_at=10.5,
    )
    assert current.selected_input_by_role["residual"].composition_input_id == (
        second.composition_input_id
    )


def test_missing_residual_has_explicit_zero_or_no_action_semantics() -> None:
    _, coordinator, epochs, store, _, _ = runtime()
    store.admit(
        input_draft(producer="slow-base", role="base", action=qpos_action(0.1))
    )
    cut = coordinator.capture_cut(cutoff_at=10.0).cut
    projection = project_composition_inputs(store, cut)
    epoch_snapshot = epochs.snapshot(cut)
    zero_spec = residual_spec(missing="zero_residual")
    zero_snapshot = build_composition_input_snapshot(
        projection,
        producer_epochs=epoch_snapshot,
        spec=zero_spec,
        target_effective_at=10.5,
    )
    zero = DeterministicComposer().compose(zero_snapshot, spec=zero_spec)
    assert zero.output_status == "ready"
    assert zero.reason == "zero_residual"
    np.testing.assert_allclose(
        zero.composed_action.arm_qpos_by_side["left"], np.full(6, 0.1)
    )

    no_action_spec = residual_spec(missing="no_action")
    no_action_snapshot = build_composition_input_snapshot(
        projection,
        producer_epochs=epoch_snapshot,
        spec=no_action_spec,
        target_effective_at=10.5,
    )
    no_action = DeterministicComposer().compose(
        no_action_snapshot, spec=no_action_spec
    )
    assert no_action.output_status == "no_action"
    assert no_action.composed_action is None


def test_component_union_rejects_overlap_and_preserves_disjoint_components() -> None:
    _, coordinator, epochs, store, _, _ = runtime()
    arm_only = RobotAction(
        action_space=BIMANUAL_QPOS_V1,
        arm_qpos_by_side={"left": np.ones(6, dtype=np.float32)},
        source="arm",
    )
    hand_only = RobotAction(
        action_space=BIMANUAL_QPOS_V1,
        hand_qpos_by_side={"left": np.full(6, 0.2, dtype=np.float32)},
        source="hand",
    )
    store.admit(input_draft(producer="slow-base", role="arm", action=arm_only))
    store.admit(
        input_draft(producer="fast-residual", role="hand", action=hand_only)
    )
    spec = CompositionSpec(
        spec_id="component-union-v1",
        algebra="component_union",
        input_roles=("arm", "hand"),
        required_roles=("arm", "hand"),
    )
    cut = coordinator.capture_cut(cutoff_at=10.0).cut
    snapshot = build_composition_input_snapshot(
        project_composition_inputs(store, cut),
        producer_epochs=epochs.snapshot(cut),
        spec=spec,
        target_effective_at=10.5,
    )
    result = DeterministicComposer().compose(snapshot, spec=spec)
    assert result.output_status == "ready"
    assert result.composed_action.active_effectors == ("left_arm", "left_hand")


def test_weighted_se3_pose_composition_is_generic_and_deterministic() -> None:
    _, coordinator, epochs, store, _, _ = runtime()
    store.admit(
        input_draft(
            producer="slow-base",
            role="policy_pose",
            action=pose_action(0.2, 0.0),
        )
    )
    store.admit(
        input_draft(
            producer="fast-residual",
            role="human_pose",
            action=pose_action(0.4, 90.0),
        )
    )
    spec = CompositionSpec(
        spec_id="weighted-se3-pose-v1",
        algebra="weighted_se3_pose_fusion",
        input_roles=("human_pose", "policy_pose"),
        required_roles=("human_pose", "policy_pose"),
        weight_by_role={"human_pose": 0.5, "policy_pose": 0.5},
        max_translation_disagreement_m=0.3,
        max_rotation_disagreement_rad=math.radians(100.0),
    )
    cut = coordinator.capture_cut(cutoff_at=10.0).cut
    snapshot = build_composition_input_snapshot(
        project_composition_inputs(store, cut),
        producer_epochs=epochs.snapshot(cut),
        spec=spec,
        target_effective_at=10.5,
    )
    first = DeterministicComposer().compose(snapshot, spec=spec)
    second = DeterministicComposer().compose(snapshot, spec=spec)

    assert first.output_status == "ready"
    assert first.reason == "weighted_se3_pose_fusion"
    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.composed_action is not None
    np.testing.assert_allclose(
        first.composed_action.arm_pose_by_side["left"][:3],
        np.asarray([0.3, -0.1, 0.3]),
        atol=1e-6,
    )
    yaw = Rotation.from_quat(
        first.composed_action.arm_pose_by_side["left"][3:7]
    ).as_euler("zyx", degrees=True)[0]
    assert math.isclose(yaw, 45.0, abs_tol=1e-4)


def test_weighted_se3_pose_composition_missing_role_and_conflict_fail_closed() -> None:
    _, coordinator, epochs, store, _, _ = runtime()
    store.admit(
        input_draft(
            producer="slow-base",
            role="policy_pose",
            action=pose_action(0.2),
        )
    )
    fail_closed_spec = CompositionSpec(
        spec_id="weighted-se3-required-v1",
        algebra="weighted_se3_pose_fusion",
        input_roles=("human_pose", "policy_pose"),
        required_roles=("human_pose", "policy_pose"),
        weight_by_role={"human_pose": 0.5, "policy_pose": 0.5},
        max_translation_disagreement_m=0.2,
        max_rotation_disagreement_rad=math.radians(60.0),
    )
    cut = coordinator.capture_cut(cutoff_at=10.0).cut
    fail_closed_snapshot = build_composition_input_snapshot(
        project_composition_inputs(store, cut),
        producer_epochs=epochs.snapshot(cut),
        spec=fail_closed_spec,
        target_effective_at=10.5,
    )
    fail_closed = DeterministicComposer().compose(
        fail_closed_snapshot,
        spec=fail_closed_spec,
    )
    assert fail_closed.output_status == "no_action"
    assert fail_closed.reason == "missing_required:human_pose"

    renormalize_spec = CompositionSpec(
        spec_id="weighted-se3-renormalize-v1",
        algebra="weighted_se3_pose_fusion",
        input_roles=("human_pose", "policy_pose"),
        required_roles=("policy_pose",),
        weight_by_role={"human_pose": 0.5, "policy_pose": 0.5},
        max_translation_disagreement_m=0.2,
        max_rotation_disagreement_rad=math.radians(60.0),
    )
    renormalize_snapshot = build_composition_input_snapshot(
        project_composition_inputs(store, cut),
        producer_epochs=epochs.snapshot(cut),
        spec=renormalize_spec,
        target_effective_at=10.5,
    )
    renormalized = DeterministicComposer().compose(
        renormalize_snapshot,
        spec=renormalize_spec,
    )
    assert renormalized.output_status == "ready"
    assert renormalized.reason == "weighted_se3_pose_fusion_renormalized"

    store.admit(
        input_draft(
            producer="fast-residual",
            role="human_pose",
            action=pose_action(0.6),
            sequence=1,
        )
    )
    conflict_cut = coordinator.capture_cut(cutoff_at=10.0).cut
    conflict_snapshot = build_composition_input_snapshot(
        project_composition_inputs(store, conflict_cut),
        producer_epochs=epochs.snapshot(conflict_cut),
        spec=fail_closed_spec,
        target_effective_at=10.5,
    )
    conflict = DeterministicComposer().compose(
        conflict_snapshot,
        spec=fail_closed_spec,
    )
    assert conflict.output_status == "invalid"
    assert "translation_disagreement" in conflict.reason


def test_existing_composition_spec_hash_does_not_gain_pose_fusion_fields() -> None:
    spec = residual_spec()
    expected = canonical_content_hash(
        {
            "spec_id": spec.spec_id,
            "algebra": spec.algebra,
            "input_roles": spec.input_roles,
            "required_roles": spec.required_roles,
            "missing_required_behavior": spec.missing_required_behavior,
            "missing_residual_behavior": spec.missing_residual_behavior,
            "residual_limit_by_component": spec.residual_limit_by_component,
        },
        domain="dexteleop.composition_spec.v1",
    )
    assert spec.semantic_content_hash == expected


def test_composition_trace_records_keep_input_ids_and_hashes() -> None:
    _, coordinator, epochs, store, _, _ = runtime()
    receipt = store.admit(
        input_draft(producer="slow-base", role="base", action=qpos_action(0.1))
    )
    assert receipt.composition_input is not None
    spec = residual_spec()
    cut = coordinator.capture_cut(cutoff_at=10.0).cut
    snapshot = build_composition_input_snapshot(
        project_composition_inputs(store, cut),
        producer_epochs=epochs.snapshot(cut),
        spec=spec,
        target_effective_at=10.5,
    )
    result = DeterministicComposer().compose(snapshot, spec=spec)
    admission_fact = project_composition_input_admission(
        receipt,
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="rm65-pair-0",
        occurred_at=10.0,
        clock_domain_id="control.monotonic",
    )
    input_fact = project_composition_input(
        receipt.composition_input, embodiment_instance_id="rm65-pair-0"
    )
    snapshot_fact = project_composition_input_snapshot(
        snapshot,
        runtime_session_id="session-0",
        embodiment_instance_id="rm65-pair-0",
        clock_domain_id="control.monotonic",
    )
    result_fact = project_composition_result(
        result,
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="rm65-pair-0",
        occurred_at=10.0,
        clock_domain_id="control.monotonic",
    )
    assert admission_fact.event_type == "composition_input_admission"
    assert input_fact.payload["authority_candidate"] is False
    assert snapshot_fact.payload["selected_input_ids_by_role"]["base"] == (
        receipt.composition_input_id
    )
    assert result_fact.payload["input_ids"] == result.input_ids
    assert result_fact.payload["input_hashes"] == result.input_hashes
