from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dataset_contracts.canonical_encoding import canonical_content_hash
from control_composition.activation import (
    ACTIVATION_LOCAL_STRICT_SOURCE_FRESHNESS_POLICY_V2,
    build_composition_activation_scope,
    evaluate_composition_input_activation,
)
from control_composition.composition import (
    CompositionAuthorityGateSpec,
    CompositionInputDraft,
    CompositionInputStore,
    CompositionSpec,
    DeterministicComposer,
    build_composite_action_proposal_draft,
    build_composition_input_snapshot,
    build_robot_action_v1_composition_input_draft,
    decide_composition_authority_disposition,
    project_composition_inputs,
)
from control_composition.retention import TimeWindowRetentionSpec
from control_composition.temporal import CompositionTemporalSelectionSpec
from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    BIMANUAL_QPOS_V1,
    ActionComponentCatalog,
    ActionComponentSpec,
    ProposalBuffer,
    RobotAction,
    build_robot_action_v1_runtime_schema,
)
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry
from robot_motion.action.proposal import ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V4
from teleop_runtime.control_trace.projection import (
    project_composition_input,
    project_composition_input_admission,
    project_composition_input_snapshot,
    project_composition_evaluation,
    project_composition_spec,
)


class Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def component_catalog(
    *,
    action_space: str,
    active_effectors: tuple[str, ...],
) -> ActionComponentCatalog:
    return ActionComponentCatalog(
        catalog_id=(
            f"action-composition:{action_space}:"
            f"{'+'.join(active_effectors)}"
        ),
        action_space=action_space,
        components=tuple(
            ActionComponentSpec(effector, effector, effector)
            for effector in active_effectors
        ),
    )


def runtime(
    *,
    action_space: str = BIMANUAL_QPOS_V1,
    active_effectors: tuple[str, ...] = ("left_arm",),
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
    catalog = component_catalog(
        action_space=action_space,
        active_effectors=active_effectors,
    )
    action_schema = build_robot_action_v1_runtime_schema(
        component_catalog=catalog,
    )
    store = CompositionInputStore(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        retention_spec=TimeWindowRetentionSpec(
            spec_id="action-composition-test-window-v1",
            history_window_sec=0.0,
            max_future_lead_sec=2.0,
            max_effective_duration_sec=2.0,
        ),
        action_schema=action_schema,
    )
    proposals = ProposalBuffer(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        capacity=16,
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
    source_fact_ids=(),
    source_fact_hashes=(),
    action_schema=None,
) -> CompositionInputDraft:
    bound_action_schema = action_schema
    if bound_action_schema is None:
        bound_action_schema = build_robot_action_v1_runtime_schema(
            component_catalog=component_catalog(
                action_space=action.action_space,
                active_effectors=tuple(action.active_effectors),
            ),
        )
    return build_robot_action_v1_composition_input_draft(
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        producer_id=producer,
        producer_epoch=1,
        sequence=sequence,
        input_role=role,
        action=action,
        action_schema=bound_action_schema,
        produced_at=9.9,
        produced_clock_domain_id=f"{producer}.monotonic",
        effective_from=10.0,
        effective_until=11.0,
        effective_clock_domain_id="control.monotonic",
        context_ids=tuple(context_ids),
        observation_cut_id=f"cut:{producer}:{sequence}",
        policy_output_id=f"output:{producer}:{sequence}",
        source_fact_ids=tuple(source_fact_ids),
        source_fact_hashes=tuple(source_fact_hashes),
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


def test_composition_source_participation_defaults_and_dependency_validation() -> None:
    default_required = CompositionSpec(
        spec_id="default-required",
        algebra="component_union",
        input_roles=("baseline", "overlay"),
    )
    assert default_required.required_roles == ("baseline", "overlay")
    assert dict(default_required.participation_requirement_by_role) == {
        "baseline": "required",
        "overlay": "required",
    }

    optional_overlay = CompositionSpec(
        spec_id="optional-overlay",
        algebra="component_union",
        input_roles=("baseline", "overlay"),
        required_roles=("baseline",),
        depends_on_roles_by_role={"overlay": ("baseline",)},
    )
    assert dict(optional_overlay.participation_requirement_by_role) == {
        "baseline": "required",
        "overlay": "optional",
    }

    with pytest.raises(ValueError, match="required role may depend"):
        CompositionSpec(
            spec_id="invalid-required-dependency",
            algebra="component_union",
            input_roles=("baseline", "overlay"),
            required_roles=("overlay",),
            depends_on_roles_by_role={"overlay": ("baseline",)},
        )
    with pytest.raises(ValueError, match="acyclic"):
        CompositionSpec(
            spec_id="cyclic-dependency",
            algebra="component_union",
            input_roles=("baseline", "overlay"),
            required_roles=(),
            depends_on_roles_by_role={
                "baseline": ("overlay",),
                "overlay": ("baseline",),
            },
        )


def test_dependency_closure_excludes_overlay_but_preserves_preclosure_lineage() -> None:
    _, coordinator, epochs, store, _, _ = runtime()
    receipt = store.admit(
        input_draft(
            producer="fast-residual",
            role="overlay",
            action=qpos_action(0.2),
        )
    )
    assert receipt.accepted
    cut = coordinator.capture_cut(cutoff_at=10.0).cut
    spec = CompositionSpec(
        spec_id="optional-relative-branch",
        algebra="component_union",
        input_roles=("baseline", "overlay"),
        required_roles=(),
        depends_on_roles_by_role={"overlay": ("baseline",)},
    )
    snapshot = build_composition_input_snapshot(
        project_composition_inputs(store, cut),
        producer_epochs=epochs.snapshot(cut),
        spec=spec,
        target_effective_at=10.5,
    )

    assert snapshot.selected_input_by_role == {}
    assert tuple(snapshot.selected_input_before_dependency_by_role) == (
        "overlay",
    )
    assert snapshot.dependency_ineligibility_reason_by_role == {
        "overlay": "prerequisite_unavailable:baseline"
    }
    assert snapshot.missing_roles == ("baseline", "overlay")


def test_optional_dependent_missing_does_not_remove_available_baseline() -> None:
    _, coordinator, epochs, store, _, _ = runtime()
    receipt = store.admit(
        input_draft(
            producer="slow-base",
            role="baseline",
            action=qpos_action(0.1),
        )
    )
    assert receipt.accepted
    cut = coordinator.capture_cut(cutoff_at=10.0).cut
    spec = CompositionSpec(
        spec_id="optional-relative-branch",
        algebra="component_union",
        input_roles=("baseline", "overlay"),
        required_roles=("baseline",),
        depends_on_roles_by_role={"overlay": ("baseline",)},
    )
    snapshot = build_composition_input_snapshot(
        project_composition_inputs(store, cut),
        producer_epochs=epochs.snapshot(cut),
        spec=spec,
        target_effective_at=10.5,
    )

    assert tuple(snapshot.selected_input_by_role) == ("baseline",)
    assert snapshot.missing_roles == ("overlay",)
    assert snapshot.dependency_ineligibility_reason_by_role == {}


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
    assert result.composed_candidate is not None
    np.testing.assert_allclose(
        result.composed_candidate.value("left_arm").values,
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
            policy_node_id="fast-composer",
        )
    ).proposal
    assert proposal is not None
    assert len(proposals) == 1
    assert proposal.composition_input_ids == result.input_ids
    assert proposal.composition_input_hashes == result.input_hashes
    assert proposal.composition_spec_hash == result.composition_spec_hash
    assert proposal.semantic_hash_domain == ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V4
    assert proposal.context_ids == ("slow-context-7",)


def test_unscoped_composition_evaluation_uses_new_candidate_hash_domain() -> None:
    _, coordinator, epochs, store, _, _ = runtime()
    receipt = store.admit(
        input_draft(
            producer="slow-base",
            role="base",
            action=qpos_action(0.1),
        )
    )
    assert receipt.accepted
    cut = coordinator.capture_cut(cutoff_at=10.0).cut
    spec = residual_spec()
    snapshot = build_composition_input_snapshot(
        project_composition_inputs(store, cut),
        producer_epochs=epochs.snapshot(cut),
        spec=spec,
        target_effective_at=10.5,
    )
    result = DeterministicComposer().compose(snapshot, spec=spec)

    assert receipt.semantic_content_hash == (
        "18c50efff3a8db702ce2854e1145eb0e39b9b02050ec6bd358b4b86c666d5bb5"
    )
    assert snapshot.semantic_content_hash == (
        "9e763408d5a4765305f9d7e13bd40320588267e4cf79b323a667613f96357352"
    )
    assert result.semantic_content_hash == (
        "7304caeee0d3bb0f9fe938506c5eceb7a5bba44f24d7c03b49ccf3232bdefc94"
    )
    assert snapshot.expected_activation_id is None
    assert result.composition_activation_id is None


def test_temporal_selection_uses_receiver_admission_for_equal_source_intervals() -> None:
    _, coordinator, epochs, store, _, _ = runtime()
    first = store.admit(
        input_draft(
            producer="slow-base",
            role="base",
            action=qpos_action(0.1),
        )
    )
    second = store.admit(
        input_draft(
            producer="fast-residual",
            role="base",
            action=qpos_action(0.2),
        )
    )
    assert first.accepted and second.accepted
    cut = coordinator.capture_cut(cutoff_at=10.0).cut
    spec = CompositionSpec(
        spec_id="single-base-temporal-selection",
        algebra="component_union",
        input_roles=("base",),
        required_roles=("base",),
    )
    snapshot = build_composition_input_snapshot(
        project_composition_inputs(store, cut),
        producer_epochs=epochs.snapshot(cut),
        spec=spec,
        target_effective_at=10.5,
    )
    assert snapshot.selected_input_by_role["base"].producer_id == (
        "fast-residual"
    )

    with pytest.raises(ValueError, match="Multiple composition inputs"):
        build_composition_input_snapshot(
            project_composition_inputs(store, cut),
            producer_epochs=epochs.snapshot(cut),
            spec=spec,
            target_effective_at=10.5,
            temporal_selection_spec=CompositionTemporalSelectionSpec(
                spec_id="reject-overlap",
                overlap_behavior="reject_multiple_covering",
            ),
        )


def test_temporal_selection_bounds_target_against_receiver_cut() -> None:
    _, coordinator, epochs, store, _, _ = runtime()
    cut = coordinator.capture_cut(cutoff_at=10.0).cut
    with pytest.raises(ValueError, match="lead bound"):
        build_composition_input_snapshot(
            project_composition_inputs(store, cut),
            producer_epochs=epochs.snapshot(cut),
            spec=residual_spec(),
            target_effective_at=10.5,
            temporal_selection_spec=CompositionTemporalSelectionSpec(
                spec_id="bounded-target",
                maximum_target_lead_sec=0.1,
                maximum_target_lag_sec=0.1,
            ),
        )


def test_activation_v2_excludes_unbound_and_prior_activation_inputs() -> None:
    clock, coordinator, epochs, store, _, _ = runtime()
    old = store.admit(
        input_draft(
            producer="slow-base",
            role="base",
            action=qpos_action(0.1),
        )
    )
    assert old.accepted
    clock.value = 10.2
    spec = CompositionSpec(
        spec_id="activation-local-component-union-v2",
        algebra="component_union",
        input_roles=("base",),
        required_roles=("base",),
    )
    activation = build_composition_activation_scope(
        operation_spec_id="composition-session-spec:test",
        operation_spec_hash="operation-spec-hash:test",
        session_epoch=1,
        activated_at=10.2,
        clock_domain_id="control.monotonic",
        input_barrier_at=10.1,
        required_roles=spec.required_roles,
    )
    bound_draft = replace(
        input_draft(
            producer="slow-base",
            role="base",
            action=qpos_action(0.2),
            sequence=1,
        ),
        produced_at=10.2,
        effective_from=10.2,
        effective_until=11.2,
        composition_activation_id=activation.activation_id,
        composition_activation_hash=activation.semantic_content_hash,
    )
    bound = store.admit(bound_draft)
    assert bound.accepted
    assert bound.semantic_content_hash != old.semantic_content_hash
    cut = coordinator.capture_cut(cutoff_at=10.2).cut
    snapshot = build_composition_input_snapshot(
        project_composition_inputs(store, cut),
        producer_epochs=epochs.snapshot(cut),
        spec=spec,
        target_effective_at=10.5,
        activation_scope=activation,
    )
    assert snapshot.selected_input_by_role["base"].composition_input_id == (
        bound.composition_input_id
    )
    assert snapshot.expected_activation_id == activation.activation_id
    result = DeterministicComposer().compose(snapshot, spec=spec)
    assert result.composition_activation_id == activation.activation_id
    assert result.composition_activation_hash == activation.semantic_content_hash

    next_activation = build_composition_activation_scope(
        operation_spec_id=activation.operation_spec_id,
        operation_spec_hash=activation.operation_spec_hash,
        session_epoch=2,
        activated_at=10.2,
        clock_domain_id=activation.clock_domain_id,
        input_barrier_at=10.1,
        required_roles=spec.required_roles,
    )
    assert next_activation.activation_id != activation.activation_id
    missing = build_composition_input_snapshot(
        project_composition_inputs(store, cut),
        producer_epochs=epochs.snapshot(cut),
        spec=spec,
        target_effective_at=10.5,
        activation_scope=next_activation,
    )
    assert missing.selected_input_by_role == {}
    assert missing.missing_roles == ("base",)
    blocked = DeterministicComposer().compose(missing, spec=spec)
    assert blocked.output_status == "no_action"
    assert blocked.reason == "missing_required:base"


def test_activation_freshness_is_receiver_local_and_strict_source_is_same_clock() -> None:
    receiver_scope = build_composition_activation_scope(
        operation_spec_id="composition-session-spec:freshness",
        operation_spec_hash="operation-spec-hash:freshness",
        session_epoch=0,
        activated_at=10.0,
        clock_domain_id="control.monotonic",
        input_barrier_at=10.0,
        required_roles=("base",),
    )
    receiver_local = evaluate_composition_input_activation(
        receiver_scope,
        composition_activation_id=receiver_scope.activation_id,
        composition_activation_hash=receiver_scope.semantic_content_hash,
        produced_at=1.0,
        produced_clock_domain_id="producer.monotonic",
        available_at=10.0,
        available_clock_domain_id="control.monotonic",
        effective_from=10.0,
        effective_clock_domain_id="control.monotonic",
    )
    assert receiver_local.accepted

    stale_effective = evaluate_composition_input_activation(
        receiver_scope,
        composition_activation_id=receiver_scope.activation_id,
        composition_activation_hash=receiver_scope.semantic_content_hash,
        produced_at=10.1,
        produced_clock_domain_id="control.monotonic",
        available_at=10.1,
        available_clock_domain_id="control.monotonic",
        effective_from=9.99,
        effective_clock_domain_id="control.monotonic",
    )
    assert not stale_effective.accepted
    assert (
        stale_effective.rejection_code
        == "activation_effective_barrier_violation"
    )

    strict_scope = build_composition_activation_scope(
        operation_spec_id=receiver_scope.operation_spec_id,
        operation_spec_hash=receiver_scope.operation_spec_hash,
        session_epoch=1,
        activated_at=10.0,
        clock_domain_id=receiver_scope.clock_domain_id,
        input_barrier_at=10.0,
        required_roles=receiver_scope.required_roles,
        freshness_policy=(
            ACTIVATION_LOCAL_STRICT_SOURCE_FRESHNESS_POLICY_V2
        ),
    )
    cross_clock = evaluate_composition_input_activation(
        strict_scope,
        composition_activation_id=strict_scope.activation_id,
        composition_activation_hash=strict_scope.semantic_content_hash,
        produced_at=10.1,
        produced_clock_domain_id="producer.monotonic",
        available_at=10.1,
        available_clock_domain_id="control.monotonic",
        effective_from=10.1,
        effective_clock_domain_id="control.monotonic",
    )
    assert not cross_clock.accepted
    assert cross_clock.rejection_code == "activation_source_clock_mismatch"

    on_barrier = evaluate_composition_input_activation(
        strict_scope,
        composition_activation_id=strict_scope.activation_id,
        composition_activation_hash=strict_scope.semantic_content_hash,
        produced_at=10.0,
        produced_clock_domain_id="control.monotonic",
        available_at=10.0,
        available_clock_domain_id="control.monotonic",
        effective_from=10.0,
        effective_clock_domain_id="control.monotonic",
    )
    assert not on_barrier.accepted
    assert on_barrier.rejection_code == "activation_source_barrier_violation"


def test_activation_freshness_is_enforced_by_generic_input_admission() -> None:
    _, coordinator, _, store, _, _ = runtime()
    receiver_scope = build_composition_activation_scope(
        operation_spec_id="composition-session-spec:store-freshness",
        operation_spec_hash="operation-spec-hash:store-freshness",
        session_epoch=0,
        activated_at=10.0,
        clock_domain_id="control.monotonic",
        input_barrier_at=10.0,
        required_roles=("base",),
    )
    receiver_local = store.admit(
        replace(
            input_draft(
                producer="slow-base",
                role="base",
                action=qpos_action(0.05),
            ),
            produced_at=1.0,
            produced_clock_domain_id="producer.monotonic",
            composition_activation_id=receiver_scope.activation_id,
            composition_activation_hash=(
                receiver_scope.semantic_content_hash
            ),
        ),
        activation_scope=receiver_scope,
    )
    assert receiver_local.accepted

    scope = build_composition_activation_scope(
        operation_spec_id="composition-session-spec:store-freshness",
        operation_spec_hash="operation-spec-hash:store-freshness",
        session_epoch=1,
        activated_at=10.0,
        clock_domain_id="control.monotonic",
        input_barrier_at=10.0,
        required_roles=("base",),
        freshness_policy=(
            ACTIVATION_LOCAL_STRICT_SOURCE_FRESHNESS_POLICY_V2
        ),
    )
    draft = replace(
        input_draft(
            producer="slow-base",
            role="base",
            action=qpos_action(0.1),
        ),
        produced_at=10.0,
        produced_clock_domain_id="control.monotonic",
        composition_activation_id=scope.activation_id,
        composition_activation_hash=scope.semantic_content_hash,
    )

    rejected = store.admit(draft, activation_scope=scope)

    assert not rejected.accepted
    assert rejected.rejection_code == "activation_source_barrier_violation"
    assert coordinator.committed_high_watermark == 4
    assert len(store) == 1

    fresh = store.admit(
        replace(draft, produced_at=10.01),
        activation_scope=scope,
    )
    assert fresh.accepted
    assert coordinator.committed_high_watermark == 5
    assert len(store) == 2


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
        zero.composed_candidate.value("left_arm").values,
        np.full(6, 0.1),
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
    assert no_action.composed_candidate is None


def test_component_union_rejects_overlap_and_preserves_disjoint_components() -> None:
    _, coordinator, epochs, store, _, _ = runtime(
        active_effectors=("left_arm", "left_hand"),
    )
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
    store.admit(
        input_draft(
            producer="slow-base",
            role="arm",
            action=arm_only,
            action_schema=store.action_schema,
        )
    )
    store.admit(
        input_draft(
            producer="fast-residual",
            role="hand",
            action=hand_only,
            action_schema=store.action_schema,
        )
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
    assert result.composed_candidate.component_coverage == (
        "left_arm",
        "left_hand",
    )


def test_weighted_se3_pose_composition_is_generic_and_deterministic() -> None:
    _, coordinator, epochs, store, _, _ = runtime(
        action_space=BIMANUAL_EE_POSE_V1,
        active_effectors=("left_arm", "right_arm"),
    )
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
    assert first.composed_candidate is not None
    np.testing.assert_allclose(
        first.composed_candidate.value("left_arm").values[:3],
        np.asarray([0.3, -0.1, 0.3]),
        atol=1e-6,
    )
    yaw = Rotation.from_quat(
        first.composed_candidate.value("left_arm").values[3:7]
    ).as_euler("zyx", degrees=True)[0]
    assert math.isclose(yaw, 45.0, abs_tol=1e-4)


def test_weighted_pose_composition_supports_single_arm_manifest() -> None:
    _, coordinator, epochs, store, _, _ = runtime(
        action_space=BIMANUAL_EE_POSE_V1,
        active_effectors=("left_arm",),
    )
    for producer, role, x in (
        ("slow-base", "policy_pose", 0.2),
        ("fast-residual", "human_pose", 0.4),
    ):
        store.admit(
            input_draft(
                producer=producer,
                role=role,
                action=RobotAction(
                    action_space=BIMANUAL_EE_POSE_V1,
                    arm_pose_by_side={
                        "left": np.asarray(
                            [x, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0],
                            dtype=np.float32,
                        )
                    },
                    active_effectors=("left_arm",),
                    source=role,
                ),
            )
        )
    spec = CompositionSpec(
        spec_id="weighted-se3-left-arm-v1",
        algebra="weighted_se3_pose_fusion",
        input_roles=("human_pose", "policy_pose"),
        required_roles=("human_pose", "policy_pose"),
        weight_by_role={"human_pose": 0.5, "policy_pose": 0.5},
        max_translation_disagreement_m=0.3,
        max_rotation_disagreement_rad=math.radians(60.0),
        pose_component_dependencies=("left_arm",),
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
    assert result.composed_candidate is not None
    assert result.composed_candidate.component_coverage == ("left_arm",)
    np.testing.assert_allclose(
        result.composed_candidate.value("left_arm").values[:3],
        [0.3, -0.1, 0.3],
        atol=1e-7,
    )


def test_default_weighted_pose_spec_hash_keeps_legacy_payload() -> None:
    spec = CompositionSpec(
        spec_id="weighted-se3-pose-v1",
        algebra="weighted_se3_pose_fusion",
        input_roles=("human_pose", "policy_pose"),
        required_roles=("human_pose", "policy_pose"),
        weight_by_role={"human_pose": 0.5, "policy_pose": 0.5},
        max_translation_disagreement_m=0.3,
        max_rotation_disagreement_rad=math.radians(100.0),
    )
    expected = canonical_content_hash(
        {
            "spec_id": spec.spec_id,
            "algebra": spec.algebra,
            "input_roles": spec.input_roles,
            "required_roles": spec.required_roles,
            "missing_required_behavior": spec.missing_required_behavior,
            "missing_residual_behavior": spec.missing_residual_behavior,
            "residual_limit_by_component": spec.residual_limit_by_component,
            "weight_by_role": spec.weight_by_role,
            "max_translation_disagreement_m": (
                spec.max_translation_disagreement_m
            ),
            "max_rotation_disagreement_rad": (
                spec.max_rotation_disagreement_rad
            ),
            "rotation_fusion": (
                "deterministic_reference_shortest_path_so3_tangent_mean_v1"
            ),
        },
        domain="dexteleop.composition_spec.v1",
    )

    assert spec.semantic_content_hash == expected


def test_weighted_se3_pose_composition_missing_role_and_conflict_fail_closed() -> None:
    _, coordinator, epochs, store, _, _ = runtime(
        action_space=BIMANUAL_EE_POSE_V1,
        active_effectors=("left_arm", "right_arm"),
    )
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


def test_formal_composition_gate_publishes_only_ready_result() -> None:
    _, coordinator, epochs, store, _, _ = runtime(
        action_space=BIMANUAL_EE_POSE_V1,
        active_effectors=("left_arm", "right_arm"),
    )
    for producer, role, value in (
        ("slow-base", "human_pose", 0.2),
        ("fast-residual", "replay_pose", 0.3),
    ):
        assert store.admit(
            input_draft(
                producer=producer,
                role=role,
                action=pose_action(value),
            )
        ).accepted
    spec = CompositionSpec(
        spec_id="formal-human-replay-se3-v1",
        algebra="weighted_se3_pose_fusion",
        input_roles=("human_pose", "replay_pose"),
        required_roles=("human_pose", "replay_pose"),
        weight_by_role={"human_pose": 0.5, "replay_pose": 0.5},
        max_translation_disagreement_m=0.2,
        max_rotation_disagreement_rad=math.radians(60.0),
    )
    gate = CompositionAuthorityGateSpec(
        gate_spec_id="formal-human-replay-authority-v1",
        composition_spec_hash=spec.semantic_content_hash,
        composer_producer_id="composer",
        missing_behavior_by_role={
            "human_pose": "hold",
            "replay_pose": "hold",
        },
        proposal_ttl_sec=0.1,
    )
    cut = coordinator.capture_cut(cutoff_at=10.0).cut
    snapshot = build_composition_input_snapshot(
        project_composition_inputs(store, cut),
        producer_epochs=epochs.snapshot(cut),
        spec=spec,
        target_effective_at=10.5,
    )
    result = DeterministicComposer().compose(snapshot, spec=spec)
    first = decide_composition_authority_disposition(
        snapshot,
        result,
        spec=spec,
        gate_spec=gate,
    )
    second = decide_composition_authority_disposition(
        snapshot,
        result,
        spec=spec,
        gate_spec=gate,
    )
    assert first.action == "publish_proposal"
    assert first.downstream_request == "submit_candidate"
    assert not first.authoritative
    assert first.reason == "ready_composition"
    assert first.semantic_content_hash == second.semantic_content_hash


def test_formal_composition_gate_holds_on_required_ttl_or_conflict() -> None:
    _, coordinator, epochs, store, _, _ = runtime(
        action_space=BIMANUAL_EE_POSE_V1,
        active_effectors=("left_arm", "right_arm"),
    )
    assert store.admit(
        input_draft(
            producer="slow-base",
            role="replay_pose",
            action=pose_action(0.2),
        )
    ).accepted
    spec = CompositionSpec(
        spec_id="formal-human-replay-fail-closed-v1",
        algebra="weighted_se3_pose_fusion",
        input_roles=("human_pose", "replay_pose"),
        required_roles=("human_pose", "replay_pose"),
        weight_by_role={"human_pose": 0.5, "replay_pose": 0.5},
        max_translation_disagreement_m=0.05,
        max_rotation_disagreement_rad=math.radians(30.0),
    )
    gate = CompositionAuthorityGateSpec(
        gate_spec_id="formal-human-replay-gate-v1",
        composition_spec_hash=spec.semantic_content_hash,
        composer_producer_id="composer",
        missing_behavior_by_role={
            "human_pose": "hold",
            "replay_pose": "hold",
        },
        proposal_ttl_sec=0.1,
    )
    cut = coordinator.capture_cut(cutoff_at=10.0).cut
    missing_snapshot = build_composition_input_snapshot(
        project_composition_inputs(store, cut),
        producer_epochs=epochs.snapshot(cut),
        spec=spec,
        target_effective_at=10.5,
    )
    missing_result = DeterministicComposer().compose(
        missing_snapshot,
        spec=spec,
    )
    missing = decide_composition_authority_disposition(
        missing_snapshot,
        missing_result,
        spec=spec,
        gate_spec=gate,
    )
    assert missing.action == "hold"
    assert missing.downstream_request == "request_hold"
    assert not missing.authoritative
    assert missing.reason == "required_contributor_missing"

    assert store.admit(
        input_draft(
            producer="fast-residual",
            role="human_pose",
            action=pose_action(0.5),
            sequence=1,
        )
    ).accepted
    conflict_cut = coordinator.capture_cut(cutoff_at=10.0).cut
    conflict_snapshot = build_composition_input_snapshot(
        project_composition_inputs(store, conflict_cut),
        producer_epochs=epochs.snapshot(conflict_cut),
        spec=spec,
        target_effective_at=10.5,
    )
    conflict_result = DeterministicComposer().compose(
        conflict_snapshot,
        spec=spec,
    )
    conflict = decide_composition_authority_disposition(
        conflict_snapshot,
        conflict_result,
        spec=spec,
        gate_spec=gate,
    )
    assert conflict.action == "hold"
    assert conflict.reason == "composition_invalid"


def test_formal_composition_gate_rejects_unsafe_required_fallback() -> None:
    spec = CompositionSpec(
        spec_id="formal-required-role-v1",
        algebra="weighted_se3_pose_fusion",
        input_roles=("human_pose", "replay_pose"),
        required_roles=("human_pose",),
        weight_by_role={"human_pose": 0.5, "replay_pose": 0.5},
        max_translation_disagreement_m=0.2,
        max_rotation_disagreement_rad=math.radians(60.0),
    )
    gate = CompositionAuthorityGateSpec(
        gate_spec_id="unsafe-formal-fallback-v1",
        composition_spec_hash=spec.semantic_content_hash,
        composer_producer_id="composer",
        missing_behavior_by_role={
            "human_pose": "drop_and_renormalize",
            "replay_pose": "drop_and_renormalize",
        },
        proposal_ttl_sec=0.1,
    )
    with np.testing.assert_raises_regex(
        ValueError,
        "Required formal contributors must HOLD",
    ):
        gate.validate_composition_spec(spec)


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
    result_fact = project_composition_evaluation(
        result,
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="rm65-pair-0",
        occurred_at=10.0,
        clock_domain_id="control.monotonic",
    )
    spec_fact = project_composition_spec(
        spec,
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        embodiment_instance_id="rm65-pair-0",
        occurred_at=10.0,
        clock_domain_id="control.monotonic",
    )
    assert admission_fact.event_type == "composition_input_admission"
    assert input_fact.event_type == "composition_input"
    assert "source_fact_ids" not in input_fact.payload
    assert input_fact.payload["authority_candidate"] is False
    assert snapshot_fact.payload["selected_input_ids_by_role"]["base"] == (
        receipt.composition_input_id
    )
    assert result_fact.payload["input_ids"] == result.input_ids
    assert result_fact.payload["input_hashes"] == result.input_hashes
    assert spec_fact.event_type == "composition_spec"
    assert spec_fact.payload["algebra"] == spec.algebra
    assert spec_fact.payload["semantic_content_hash"] == (
        spec.semantic_content_hash
    )


def test_generic_contributor_input_has_explicit_source_lineage() -> None:
    _, _, _, store, _, _ = runtime(
        action_space=BIMANUAL_EE_POSE_V1,
        active_effectors=("left_arm", "right_arm"),
    )
    source_ids = ("human-emission:0", "human-adapter-receipt:0")
    source_hashes = ("a" * 64, "b" * 64)
    receipt = store.admit(
        input_draft(
            producer="slow-base",
            role="human_pose",
            action=pose_action(0.2),
            source_fact_ids=source_ids,
            source_fact_hashes=source_hashes,
        )
    )
    assert receipt.accepted and receipt.composition_input is not None
    fact = project_composition_input(
        receipt.composition_input,
        embodiment_instance_id="rm65-pair-0",
    )
    assert fact.event_type == "composition_contributor_input"
    assert fact.payload["source_fact_ids"] == source_ids
    assert fact.payload["source_fact_hashes"] == source_hashes
    assert fact.payload["context_ids"] == ()


def test_input_snapshot_closes_time_window_without_losing_sequence_tombstone() -> None:
    clock, coordinator, epochs, store, _, _ = runtime()
    original_draft = replace(
        input_draft(
            producer="slow-base",
            role="base",
            action=qpos_action(0.1),
        ),
        effective_from=10.0,
        effective_until=10.2,
    )
    receipt = store.admit(original_draft)
    assert receipt.accepted

    clock.value = 10.15
    cut = coordinator.capture_cut(cutoff_at=clock.value).cut
    selected = store.capture_snapshot(
        cut,
        producer_epochs=epochs.snapshot(cut),
        spec=residual_spec(),
        target_effective_at=clock.value,
    )
    assert selected.selected_input_by_role["base"].composition_input_id == (
        receipt.composition_input_id
    )
    assert len(store) == 1

    clock.value = 10.25
    later_cut = coordinator.capture_cut(cutoff_at=clock.value).cut
    missing = store.capture_snapshot(
        later_cut,
        producer_epochs=epochs.snapshot(later_cut),
        spec=residual_spec(),
        target_effective_at=clock.value,
    )
    assert "base" in missing.missing_roles
    health = store.retention_snapshot()
    assert health.closed_target_through == clock.value
    assert health.retained_record_count == 0
    assert health.pruned_record_count == 1
    assert health.producer_epoch_tombstone_count == 1
    assert (
        selected.selected_input_by_role["base"]
        .action.value("left_arm")
        .payload_schema.representation_family
        == "joint_position"
    )

    duplicate = store.admit(
        replace(
            original_draft,
            effective_from=10.25,
            effective_until=10.5,
        )
    )
    assert not duplicate.accepted
    assert duplicate.rejection_code == "duplicate"
