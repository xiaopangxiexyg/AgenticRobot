from __future__ import annotations

import pytest

from control_composition.trajectory import (
    TRAJECTORY_EDITING_MODE_RESIDUAL_ACTIVE,
    TRAJECTORY_TARGET_STATUS_ALREADY_EMITTED,
    TRAJECTORY_TARGET_STATUS_NO_ACTION,
    TRAJECTORY_TARGET_STATUS_READY,
    activate_residual_session,
    commit_trajectory_target_emission,
    empty_trajectory_editing_state,
    offer_nominal_trajectory,
    prepare_trajectory_target,
)
from control_composition import (
    ResidualAnchorComponentBinding,
    ResidualAnchorReceipt,
    ResidualAnchorRequest,
    build_composition_activation_scope,
)
from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentAction,
    ComponentActionSchema,
    ComponentActionTrajectory,
    ComponentActionTrajectoryKeyframe,
    ControlComponentSpecV2,
)
from robot_motion.runtime.control_admission import ControlCut


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _schema() -> ComponentActionSchema:
    payload = ActionPayloadSchema(
        schema_id="editing-arm-pose",
        schema_version="development",
        representation_family="cartesian_pose",
        dtype="float64",
        shape=(7,),
        coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
        units=("m", "m", "m", "1", "1", "1", "1"),
        value_semantics="absolute_target",
        reference_frame="world",
        controlled_frame="tool",
    )
    catalog = ActionComponentCatalogV2(
        catalog_id="editing-fixture",
        catalog_version="development",
        components=(
            ControlComponentSpecV2(
                component_id="arm",
                controlled_entity_id="arm",
                command_channel_id="arm",
                selection_group_id="arm",
                resource_claims=("resource:arm",),
                accepted_payload_schema_hashes=(payload.semantic_content_hash,),
            ),
        ),
    )
    return ComponentActionSchema(
        schema_id="editing-fixture",
        schema_version="development",
        component_catalog=catalog,
        payload_schema_by_component={"arm": payload},
    )


def _trajectory(revision: int, offset: float) -> ComponentActionTrajectory:
    keyframes = []
    for index in range(3):
        action = ComponentAction(
            action_schema=_schema(),
            values_by_component={
                "arm": (
                    offset + index,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ),
            },
        )
        keyframes.append(
            ComponentActionTrajectoryKeyframe(
                target_effective_at=10.0 + index * 0.1,
                effective_until=10.1 + index * 0.1,
                action=action,
                source_index=index,
                source_fact_ids=(f"plan:{revision}:step:{index}",),
                source_fact_hashes=(f"plan:{revision}:hash:{index}",),
            )
        )
    return ComponentActionTrajectory(
        trajectory_id="virtual-policy-plan",
        revision=revision,
        source_kind="virtual_policy",
        source_instance_id="virtual-policy",
        clock_domain_id="control.monotonic",
        created_at=9.0 + revision * 0.01,
        keyframes=tuple(keyframes),
        source_fact_ids=(f"plan:{revision}",),
        source_fact_hashes=(f"plan-hash:{revision}",),
        source_activation_id="virtual-policy-activation:1",
        source_activation_hash="virtual-policy-activation-hash:1",
    )


def _activate(state, *, boundary: float):
    assert state.active_trajectory is not None
    scope = build_composition_activation_scope(
        operation_spec_id="trajectory-residual-test",
        operation_spec_hash="trajectory-residual-test-hash",
        session_epoch=1,
        activated_at=9.7,
        clock_domain_id="control.monotonic",
        input_barrier_at=9.6,
        required_roles=("baseline", "residual"),
    )
    request = ResidualAnchorRequest.for_activation(
        scope,
        request_generation=0,
        source_role="residual",
        component_ids=("arm",),
        source_binding_id="virtual-human:1",
        source_binding_hash="virtual-human-hash:1",
        absolute_action_schema_hash=state.active_trajectory.action_schema_hash,
        relative_action_schema_hash="relative-arm-schema-hash",
        requested_at=9.75,
    )
    receipt = ResidualAnchorReceipt.capture(
        request,
        source_action_id="virtual-human-action:1",
        source_action_hash="virtual-human-action-hash:1",
        source_available_at=9.8,
        captured_at=9.81,
        component_bindings=(
            ResidualAnchorComponentBinding(
                component_id="arm",
                source_sample_id="virtual-human-arm:1",
                source_sample_hash="virtual-human-arm-hash:1",
                absolute_payload_schema_hash=(
                    state.active_trajectory.keyframes[0]
                    .action.value("arm")
                    .payload_schema.semantic_content_hash
                ),
                anchor_id="virtual-human-anchor:1",
                anchor_hash="virtual-human-anchor-hash:1",
            ),
        ),
    )
    return activate_residual_session(
        state,
        residual_session_id="human:1",
        activation_scope=scope,
        control_cut=ControlCut(
            runtime_session_id="test-runtime",
            control_domain_id="test-control",
            clock_domain_id="control.monotonic",
            clock_reset_epoch=0,
            cutoff_at=9.9,
            cutoff_admission_sequence=20,
        ),
        residual_anchor_receipt=receipt,
        first_editable_target_at=boundary,
    )


def test_pass_through_preparation_is_exact_and_commit_is_separate() -> None:
    state = empty_trajectory_editing_state(reducer_id="editor")
    state = offer_nominal_trajectory(
        state,
        _trajectory(1, 1.0),
        offer_sequence=1,
    )
    prepared = prepare_trajectory_target(
        state,
        target_effective_at=10.0,
    )

    assert prepared.status == TRAJECTORY_TARGET_STATUS_READY
    assert prepared.action is state.active_trajectory.keyframes[0].action
    assert state.last_emitted_target_at is None

    committed = commit_trajectory_target_emission(
        state,
        prepared,
        emission_receipt_id="emission:0",
        emission_receipt_hash="emission-hash:0",
    )
    duplicate = prepare_trajectory_target(
        committed,
        target_effective_at=10.0,
    )
    assert duplicate.status == TRAJECTORY_TARGET_STATUS_ALREADY_EMITTED


def test_commit_can_bind_the_formally_published_corrected_action_hash() -> None:
    state = offer_nominal_trajectory(
        empty_trajectory_editing_state(reducer_id="editor"),
        _trajectory(1, 1.0),
        offer_sequence=1,
    )
    prepared = prepare_trajectory_target(state, target_effective_at=10.0)

    committed = commit_trajectory_target_emission(
        state,
        prepared,
        emission_receipt_id="proposal:corrected:0",
        emission_receipt_hash="proposal-receipt-hash:corrected:0",
        emitted_action_hash="formally-composed-action-hash:0",
    )

    assert committed.last_emitted_action_hash == (
        "formally-composed-action-hash:0"
    )
    assert committed.last_emitted_action_hash != (
        prepared.action.semantic_content_hash
    )


def test_residual_session_latches_base_and_keeps_only_latest_shadow() -> None:
    base = _trajectory(1, 1.0)
    state = offer_nominal_trajectory(
        empty_trajectory_editing_state(reducer_id="editor"),
        base,
        offer_sequence=1,
    )
    state = _activate(state, boundary=10.1)
    first_shadow = _trajectory(2, 20.0)
    latest_shadow = _trajectory(3, 30.0)
    state = offer_nominal_trajectory(
        state,
        first_shadow,
        offer_sequence=2,
    )
    state = offer_nominal_trajectory(
        state,
        latest_shadow,
        offer_sequence=3,
    )

    assert state.mode == TRAJECTORY_EDITING_MODE_RESIDUAL_ACTIVE
    assert state.active_trajectory.semantic_content_hash == (
        base.semantic_content_hash
    )
    assert state.shadow_trajectory.semantic_content_hash == (
        latest_shadow.semantic_content_hash
    )
    prepared = prepare_trajectory_target(
        state,
        target_effective_at=10.1,
    )
    assert prepared.status == TRAJECTORY_TARGET_STATUS_READY
    assert prepared.action.semantic_content_hash == (
        base.keyframes[1].action.semantic_content_hash
    )


def test_stale_offer_and_stale_preparation_cannot_change_state() -> None:
    state = offer_nominal_trajectory(
        empty_trajectory_editing_state(reducer_id="editor"),
        _trajectory(1, 1.0),
        offer_sequence=4,
    )
    prepared = prepare_trajectory_target(
        state,
        target_effective_at=10.0,
    )
    with pytest.raises(ValueError, match="strictly increase"):
        offer_nominal_trajectory(
            state,
            _trajectory(2, 2.0),
            offer_sequence=4,
        )

    advanced = offer_nominal_trajectory(
        state,
        _trajectory(2, 2.0),
        offer_sequence=5,
    )
    with pytest.raises(ValueError, match="another publication basis"):
        commit_trajectory_target_emission(
            advanced,
            prepared,
            emission_receipt_id="emission:stale",
            emission_receipt_hash="emission-hash:stale",
        )


def test_shadow_arrival_does_not_invalidate_active_emission_preparation() -> None:
    state = offer_nominal_trajectory(
        empty_trajectory_editing_state(reducer_id="editor"),
        _trajectory(1, 1.0),
        offer_sequence=1,
    )
    state = _activate(state, boundary=10.1)
    prepared = prepare_trajectory_target(
        state,
        target_effective_at=10.0,
    )
    with_shadow = offer_nominal_trajectory(
        state,
        _trajectory(2, 20.0),
        offer_sequence=2,
    )

    assert with_shadow.semantic_content_hash != state.semantic_content_hash
    assert with_shadow.publication_basis_hash == state.publication_basis_hash
    committed = commit_trajectory_target_emission(
        with_shadow,
        prepared,
        emission_receipt_id="emission:0",
        emission_receipt_hash="emission-hash:0",
    )
    assert committed.last_emitted_target_at == 10.0


def test_missing_and_off_lattice_targets_are_explicit_no_action() -> None:
    empty = empty_trajectory_editing_state(reducer_id="editor")
    missing = prepare_trajectory_target(empty, target_effective_at=10.0)
    assert missing.status == TRAJECTORY_TARGET_STATUS_NO_ACTION
    assert missing.reason == "no_active_trajectory"

    state = offer_nominal_trajectory(
        empty,
        _trajectory(1, 1.0),
        offer_sequence=1,
    )
    off_lattice = prepare_trajectory_target(
        state,
        target_effective_at=10.05,
    )
    expired = prepare_trajectory_target(
        state,
        target_effective_at=10.3,
    )
    assert off_lattice.reason == "target_not_on_trajectory_lattice"
    assert expired.reason == "trajectory_support_expired"


def test_single_step_trajectory_does_not_invent_future_support() -> None:
    source = _trajectory(1, 1.0)
    single_step = ComponentActionTrajectory(
        trajectory_id="virtual-policy-single-step",
        revision=source.revision,
        source_kind="virtual_policy_single_step",
        source_instance_id=source.source_instance_id,
        clock_domain_id=source.clock_domain_id,
        created_at=source.created_at,
        keyframes=(source.keyframes[0],),
        source_fact_ids=("single-policy-action:0",),
        source_fact_hashes=("single-policy-action-hash:0",),
        source_activation_id=source.source_activation_id,
        source_activation_hash=source.source_activation_hash,
    )
    state = offer_nominal_trajectory(
        empty_trajectory_editing_state(reducer_id="single-step-editor"),
        single_step,
        offer_sequence=0,
    )

    current = prepare_trajectory_target(state, target_effective_at=10.0)
    unsupported_future = prepare_trajectory_target(
        state,
        target_effective_at=10.05,
    )

    assert current.status == TRAJECTORY_TARGET_STATUS_READY
    assert unsupported_future.status == TRAJECTORY_TARGET_STATUS_NO_ACTION
    assert unsupported_future.reason == "target_not_on_trajectory_lattice"


def test_replaying_equal_events_produces_equal_state_hash() -> None:
    def replay():
        state = empty_trajectory_editing_state(reducer_id="editor")
        state = offer_nominal_trajectory(
            state,
            _trajectory(1, 1.0),
            offer_sequence=1,
        )
        return _activate(state, boundary=10.1)

    assert replay().semantic_content_hash == replay().semantic_content_hash
