from __future__ import annotations

from dataclasses import replace

import pytest

from control_composition import (
    ResidualAnchorComponentBinding,
    ResidualAnchorReceipt,
    ResidualAnchorRequest,
    build_composition_activation_scope,
)
from control_composition.trajectory import (
    RESIDUAL_REBASE_CAPTURE_TRIGGER_ASSOCIATED_SAMPLE,
    TRAJECTORY_TARGET_STATUS_NO_ACTION,
    TRAJECTORY_TARGET_STATUS_READY,
    ResidualTemporalWarpSpec,
    activate_residual_session,
    commit_trajectory_target_emission,
    complete_residual_rebase,
    empty_trajectory_editing_state,
    offer_nominal_trajectory,
    prepare_trajectory_target,
    request_residual_rebase,
    sample_residual_for_trajectory_target,
)
from control_station import StationIntent, StationIntentAdmissionReceipt
from control_station.composition import (
    StationResidualRebaseCommand,
    StationResidualRebaseController,
    StationResidualRebaseProfile,
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


def _schemas() -> tuple[ComponentActionSchema, ComponentActionSchema]:
    absolute_payload = ActionPayloadSchema(
        schema_id="rebase-arm-absolute",
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
    relative_payload = ActionPayloadSchema(
        schema_id="rebase-arm-relative",
        schema_version="development",
        representation_family="cartesian_pose",
        dtype="float64",
        shape=(7,),
        coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
        units=("m", "m", "m", "1", "1", "1", "1"),
        value_semantics="relative_delta",
        reference_frame="world",
        controlled_frame="tool",
    )
    catalog = ActionComponentCatalogV2(
        catalog_id="rebase-fixture",
        catalog_version="development",
        components=(
            ControlComponentSpecV2(
                component_id="arm",
                controlled_entity_id="arm",
                command_channel_id="arm",
                selection_group_id="arm",
                resource_claims=("resource:arm",),
                accepted_payload_schema_hashes=(
                    absolute_payload.semantic_content_hash,
                    relative_payload.semantic_content_hash,
                ),
            ),
        ),
    )
    return (
        ComponentActionSchema(
            schema_id="rebase-absolute",
            schema_version="development",
            component_catalog=catalog,
            payload_schema_by_component={"arm": absolute_payload},
        ),
        ComponentActionSchema(
            schema_id="rebase-relative",
            schema_version="development",
            component_catalog=catalog,
            payload_schema_by_component={"arm": relative_payload},
        ),
    )


def _trajectory() -> ComponentActionTrajectory:
    absolute, _ = _schemas()
    return ComponentActionTrajectory(
        trajectory_id="rebase-policy-plan",
        revision=0,
        source_kind="virtual_policy",
        source_instance_id="virtual-policy:1",
        clock_domain_id="control.monotonic",
        created_at=9.0,
        keyframes=tuple(
            ComponentActionTrajectoryKeyframe(
                target_effective_at=10.0 + 0.1 * index,
                effective_until=10.1 + 0.1 * index,
                action=ComponentAction(
                    action_schema=absolute,
                    values_by_component={
                        "arm": (
                            1.0 + index,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                        )
                    },
                ),
                source_index=index,
                source_fact_ids=(f"policy-step:{index}",),
                source_fact_hashes=(f"policy-step-hash:{index}",),
            )
            for index in range(3)
        ),
        source_fact_ids=("policy-plan:0",),
        source_fact_hashes=("policy-plan-hash:0",),
        source_activation_id="policy-activation:1",
        source_activation_hash="policy-activation-hash:1",
    )


def _binding(component_suffix: str) -> ResidualAnchorComponentBinding:
    absolute, _ = _schemas()
    return ResidualAnchorComponentBinding(
        component_id="arm",
        source_sample_id=f"human-arm:{component_suffix}",
        source_sample_hash=f"human-arm-hash:{component_suffix}",
        absolute_payload_schema_hash=(
            absolute.payload_schema_by_component["arm"].semantic_content_hash
        ),
        anchor_id=f"human-anchor:{component_suffix}",
        anchor_hash=f"human-anchor-hash:{component_suffix}",
    )


def _initial_state():
    absolute, relative = _schemas()
    scope = build_composition_activation_scope(
        operation_spec_id="rebase-residual-operation",
        operation_spec_hash="rebase-residual-operation-hash",
        session_epoch=1,
        activated_at=9.7,
        clock_domain_id="control.monotonic",
        input_barrier_at=9.7,
        required_roles=("baseline", "residual"),
    )
    request = ResidualAnchorRequest.for_activation(
        scope,
        request_generation=0,
        source_role="residual",
        component_ids=("arm",),
        source_binding_id="virtual-human:1",
        source_binding_hash="virtual-human-hash:1",
        absolute_action_schema_hash=absolute.semantic_content_hash,
        relative_action_schema_hash=relative.semantic_content_hash,
        requested_at=9.75,
    )
    receipt = ResidualAnchorReceipt.capture(
        request,
        source_action_id="human-action:initial",
        source_action_hash="human-action-hash:initial",
        source_available_at=9.8,
        captured_at=9.81,
        component_bindings=(_binding("initial"),),
    )
    state = activate_residual_session(
        offer_nominal_trajectory(
            empty_trajectory_editing_state(reducer_id="rebase-editor"),
            _trajectory(),
            offer_sequence=0,
        ),
        residual_session_id="residual-session:1",
        activation_scope=scope,
        control_cut=ControlCut(
            runtime_session_id="runtime:1",
            control_domain_id="control:1",
            clock_domain_id="control.monotonic",
            clock_reset_epoch=0,
            cutoff_at=9.9,
            cutoff_admission_sequence=10,
        ),
        residual_anchor_receipt=receipt,
        first_editable_target_at=10.0,
    )
    return state, scope, request, receipt


def _successor_request(scope, previous, *, requested_at: float = 10.01):
    absolute, relative = _schemas()
    return ResidualAnchorRequest.for_activation(
        scope,
        request_generation=previous.request_generation + 1,
        source_role="residual",
        component_ids=("arm",),
        source_binding_id="virtual-human:1",
        source_binding_hash="virtual-human-hash:1",
        absolute_action_schema_hash=absolute.semantic_content_hash,
        relative_action_schema_hash=relative.semantic_content_hash,
        requested_at=requested_at,
        previous_request=previous,
    )


def _receipt(
    request: ResidualAnchorRequest,
    *,
    action_id: str = "human-action:rebase",
    action_hash: str = "human-action-hash:rebase",
    available_at: float = 10.02,
    captured_at: float = 10.03,
) -> ResidualAnchorReceipt:
    return ResidualAnchorReceipt.capture(
        request,
        source_action_id=action_id,
        source_action_hash=action_hash,
        source_available_at=available_at,
        captured_at=captured_at,
        component_bindings=(_binding("rebase"),),
    )


def _request_rebase(state, anchor_request, **overrides):
    values = {
        "effective_not_before_target_at": 10.1,
        "capture_deadline": 10.08,
        "trigger_kind": "operator.button.rebase_and_clear",
        "trigger_fact_id": "operator-trigger:1",
        "trigger_fact_hash": "operator-trigger-hash:1",
    }
    values.update(overrides)
    return request_residual_rebase(
        state,
        anchor_request=anchor_request,
        **values,
    )


def test_pending_rebase_fences_old_anchor_and_invalidates_prior_preparation() -> None:
    state, scope, initial_request, _initial_receipt = _initial_state()
    prepared_before_trigger = prepare_trajectory_target(
        state,
        target_effective_at=10.1,
    )
    old_basis = state.publication_basis_hash
    anchor_request = _successor_request(scope, initial_request)

    pending = _request_rebase(state, anchor_request)

    assert pending.publication_basis_hash != old_basis
    assert pending.pending_residual_rebase is not None
    blocked = prepare_trajectory_target(pending, target_effective_at=10.1)
    assert blocked.status == TRAJECTORY_TARGET_STATUS_NO_ACTION
    assert blocked.reason == "residual_rebase_pending"
    assert _request_rebase(pending, anchor_request) is pending
    with pytest.raises(ValueError, match="another publication basis"):
        commit_trajectory_target_emission(
            pending,
            prepared_before_trigger,
            emission_receipt_id="old-anchor-emission",
            emission_receipt_hash="old-anchor-emission-hash",
        )


def test_pending_rebase_rejects_a_different_trigger_transaction() -> None:
    state, scope, initial_request, _initial_receipt = _initial_state()
    anchor_request = _successor_request(scope, initial_request)
    pending = _request_rebase(state, anchor_request)

    with pytest.raises(ValueError, match="already pending"):
        _request_rebase(
            pending,
            anchor_request,
            trigger_fact_id="operator-trigger:2",
            trigger_fact_hash="operator-trigger-hash:2",
        )


def test_fresh_receipt_advances_anchor_revision_without_replacing_session() -> None:
    state, scope, initial_request, initial_receipt = _initial_state()
    activation_hash = state.residual_session_activation_binding.semantic_content_hash
    anchor_request = _successor_request(scope, initial_request)
    pending = _request_rebase(state, anchor_request)
    request = pending.pending_residual_rebase
    assert request is not None
    receipt = _receipt(anchor_request)

    rebased = complete_residual_rebase(
        pending,
        request,
        receipt,
        effective_for_target_at=10.1,
    )

    binding = rebased.active_residual_anchor_binding
    assert binding is not None
    assert binding.binding_revision == 1
    assert binding.residual_session_id == state.residual_session_id
    assert binding.residual_anchor_receipt is receipt
    assert rebased.pending_residual_rebase is None
    assert (
        rebased.residual_session_activation_binding.semantic_content_hash
        == activation_hash
    )
    assert (
        rebased.residual_session_activation_binding.residual_anchor_receipt
        is initial_receipt
    )
    assert complete_residual_rebase(
        rebased,
        request,
        receipt,
        effective_for_target_at=10.1,
    ) is rebased
    assert (
        prepare_trajectory_target(rebased, target_effective_at=10.1).status
        == TRAJECTORY_TARGET_STATUS_READY
    )


def test_residual_sampling_requires_the_new_active_anchor_identity() -> None:
    state, scope, initial_request, initial_receipt = _initial_state()
    anchor_request = _successor_request(scope, initial_request)
    pending = _request_rebase(state, anchor_request)
    request = pending.pending_residual_rebase
    assert request is not None
    receipt = _receipt(anchor_request)
    rebased = complete_residual_rebase(
        pending,
        request,
        receipt,
        effective_for_target_at=10.1,
    )
    preparation = prepare_trajectory_target(rebased, target_effective_at=10.1)
    _, relative = _schemas()
    residual = ComponentAction(
        action_schema=relative,
        values_by_component={"arm": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)},
    )
    warp = ResidualTemporalWarpSpec(spec_id="rebase-ramp", ramp_duration_sec=0.1)

    with pytest.raises(ValueError, match="another active anchor"):
        sample_residual_for_trajectory_target(
            rebased,
            preparation,
            residual_action=residual,
            residual_fact_id="residual-after-rebase:1",
            residual_fact_hash="residual-after-rebase-hash:1",
            residual_anchor_receipt_id=initial_receipt.receipt_id,
            residual_anchor_receipt_hash=initial_receipt.semantic_content_hash,
            warp_spec=warp,
        )
    sample = sample_residual_for_trajectory_target(
        rebased,
        preparation,
        residual_action=residual,
        residual_fact_id="residual-after-rebase:1",
        residual_fact_hash="residual-after-rebase-hash:1",
        residual_anchor_receipt_id=receipt.receipt_id,
        residual_anchor_receipt_hash=receipt.semantic_content_hash,
        warp_spec=warp,
    )
    assert sample.anchor_request_generation == 1
    assert sample.residual_anchor_receipt_id == receipt.receipt_id


def test_late_or_pre_barrier_anchor_sample_cannot_complete_rebase() -> None:
    state, scope, initial_request, _initial_receipt = _initial_state()
    anchor_request = _successor_request(scope, initial_request)
    with pytest.raises(ValueError, match="before its request barrier"):
        _receipt(anchor_request, available_at=10.0, captured_at=10.02)

    pending = _request_rebase(
        state,
        anchor_request,
        capture_deadline=10.04,
    )
    request = pending.pending_residual_rebase
    assert request is not None
    late = _receipt(anchor_request, available_at=10.02, captured_at=10.05)
    with pytest.raises(ValueError, match="missed its capture deadline"):
        complete_residual_rebase(
            pending,
            request,
            late,
            effective_for_target_at=10.1,
        )


def test_successor_anchor_cannot_take_effect_before_its_capture() -> None:
    state, scope, initial_request, _initial_receipt = _initial_state()
    anchor_request = _successor_request(scope, initial_request)
    pending = _request_rebase(
        state,
        anchor_request,
        capture_deadline=10.2,
    )
    request = pending.pending_residual_rebase
    assert request is not None
    receipt = _receipt(
        anchor_request,
        available_at=10.02,
        captured_at=10.11,
    )

    with pytest.raises(ValueError, match="cannot activate before capture"):
        complete_residual_rebase(
            pending,
            request,
            receipt,
            effective_for_target_at=10.1,
        )


def test_trigger_associated_capture_rejects_another_fresh_sample() -> None:
    state, scope, initial_request, _initial_receipt = _initial_state()
    anchor_request = _successor_request(scope, initial_request)
    pending = _request_rebase(
        state,
        anchor_request,
        capture_mode=RESIDUAL_REBASE_CAPTURE_TRIGGER_ASSOCIATED_SAMPLE,
        trigger_source_action_id="human-action:trigger-packet",
        trigger_source_action_hash="human-action-hash:trigger-packet",
    )
    request = pending.pending_residual_rebase
    assert request is not None
    with pytest.raises(ValueError, match="not the trigger-associated"):
        complete_residual_rebase(
            pending,
            request,
            _receipt(anchor_request),
            effective_for_target_at=10.1,
        )

    associated = _receipt(
        anchor_request,
        action_id="human-action:trigger-packet",
        action_hash="human-action-hash:trigger-packet",
    )
    rebased = complete_residual_rebase(
        pending,
        request,
        associated,
        effective_for_target_at=10.1,
    )
    assert rebased.active_residual_anchor_receipt is associated


def test_rebase_boundary_cannot_rewrite_an_emitted_target() -> None:
    state, scope, initial_request, _initial_receipt = _initial_state()
    preparation = prepare_trajectory_target(state, target_effective_at=10.1)
    committed = commit_trajectory_target_emission(
        state,
        preparation,
        emission_receipt_id="emission:10.1",
        emission_receipt_hash="emission-hash:10.1",
    )
    anchor_request = _successor_request(scope, initial_request)

    with pytest.raises(ValueError, match="follow the publication barrier"):
        _request_rebase(committed, anchor_request)


def test_rebase_transaction_replay_is_hash_deterministic() -> None:
    state_a, scope_a, initial_request_a, _initial_receipt_a = _initial_state()
    state_b, scope_b, initial_request_b, _initial_receipt_b = _initial_state()
    anchor_request_a = _successor_request(scope_a, initial_request_a)
    anchor_request_b = _successor_request(scope_b, initial_request_b)
    pending_a = _request_rebase(state_a, anchor_request_a)
    pending_b = _request_rebase(state_b, anchor_request_b)
    request_a = pending_a.pending_residual_rebase
    request_b = pending_b.pending_residual_rebase
    assert request_a is not None and request_b is not None

    rebased_a = complete_residual_rebase(
        pending_a,
        request_a,
        _receipt(anchor_request_a),
        effective_for_target_at=10.1,
    )
    rebased_b = complete_residual_rebase(
        pending_b,
        request_b,
        _receipt(anchor_request_b),
        effective_for_target_at=10.1,
    )

    assert pending_a.semantic_content_hash == pending_b.semantic_content_hash
    assert rebased_a.semantic_content_hash == rebased_b.semantic_content_hash
    assert (
        rebased_a.active_residual_anchor_binding.semantic_content_hash
        == rebased_b.active_residual_anchor_binding.semantic_content_hash
    )


def _accepted_station_rebase_intent(state, *, profile_id: str):
    anchor = state.active_residual_anchor_binding
    assert anchor is not None
    intent = StationIntent(
        intent_id="station-intent:residual-rebase:1",
        runtime_session_id="runtime:1",
        control_domain_id="control:1",
        client_id="operator:1",
        client_sequence=1,
        intent_kind="request_residual_rebase_and_clear",
        produced_at=9.99,
        produced_clock_domain_id="control.monotonic",
        available_at=10.0,
        available_clock_domain_id="control.monotonic",
        admission_sequence=20,
        expected_state_revision=1,
        manifest_hash="station-manifest-hash:1",
        subject_id="composer-primary",
        resource_binding_id=anchor.anchor_binding_id,
        operation_spec_id=profile_id,
        reason="operator_button",
        semantic_content_hash="station-intent-hash:residual-rebase:1",
    )
    receipt = StationIntentAdmissionReceipt(
        intent_id=intent.intent_id,
        runtime_session_id=intent.runtime_session_id,
        control_domain_id=intent.control_domain_id,
        client_id=intent.client_id,
        client_sequence=intent.client_sequence,
        intent_kind=intent.intent_kind,
        available_at=intent.available_at,
        available_clock_domain_id=intent.available_clock_domain_id,
        admission_sequence=intent.admission_sequence,
        accepted=True,
        status="residual_rebase_intent_accepted",
        state_revision_before=1,
        state_revision_after=2,
        manifest_hash=intent.manifest_hash,
        resulting_state_hash="station-state-hash:2",
        intent_hash=intent.semantic_content_hash,
    )
    return intent, receipt


def test_station_rebase_port_is_lazy_when_default_off() -> None:
    state, _scope, _initial_request, _initial_receipt = _initial_state()
    intent, receipt = _accepted_station_rebase_intent(state, profile_id="off")
    called = False

    def command_factory():
        nonlocal called
        called = True
        raise AssertionError("disabled port must not evaluate the command supplier")

    cycle = StationResidualRebaseController(
        profile=StationResidualRebaseProfile()
    ).request_rebase(
        state=state,
        intent=intent,
        intent_receipt=receipt,
        command_factory=command_factory,
    )

    assert cycle.status == "disabled"
    assert not called
    assert state.pending_residual_rebase is None


def test_station_rebase_port_binds_accepted_intent_as_trigger_lineage() -> None:
    state, scope, initial_request, _initial_receipt = _initial_state()
    profile_id = "residual-rebase-profile:1"
    intent, receipt = _accepted_station_rebase_intent(
        state,
        profile_id=profile_id,
    )
    anchor_request = _successor_request(scope, initial_request)
    controller = StationResidualRebaseController(
        profile=StationResidualRebaseProfile(
            profile_id=profile_id,
            composer_source_slot_id="composer-primary",
            enabled=True,
        )
    )

    cycle = controller.request_rebase(
        state=state,
        intent=intent,
        intent_receipt=receipt,
        command_factory=lambda: StationResidualRebaseCommand(
            anchor_request=anchor_request,
            effective_not_before_target_at=10.1,
            capture_deadline=10.08,
        ),
    )

    assert cycle.status == "requested"
    assert cycle.state_after.pending_residual_rebase is cycle.rebase_request
    assert cycle.rebase_request.trigger_kind == intent.intent_kind
    assert cycle.rebase_request.trigger_fact_id == intent.intent_id
    assert cycle.rebase_request.trigger_fact_hash == intent.semantic_content_hash


def test_station_rebase_port_rejects_receipt_for_another_intent_payload() -> None:
    state, scope, initial_request, _initial_receipt = _initial_state()
    profile_id = "residual-rebase-profile:1"
    intent, receipt = _accepted_station_rebase_intent(
        state,
        profile_id=profile_id,
    )
    controller = StationResidualRebaseController(
        profile=StationResidualRebaseProfile(
            profile_id=profile_id,
            composer_source_slot_id="composer-primary",
            enabled=True,
        )
    )

    with pytest.raises(ValueError, match="accepted Station intent"):
        controller.request_rebase(
            state=state,
            intent=intent,
            intent_receipt=replace(
                receipt,
                intent_hash="another-intent-hash",
            ),
            command_factory=lambda: StationResidualRebaseCommand(
                anchor_request=_successor_request(scope, initial_request),
                effective_not_before_target_at=10.1,
                capture_deadline=10.08,
            ),
        )


def test_station_rebase_port_rejects_unmapped_trigger_clock() -> None:
    state, scope, initial_request, _initial_receipt = _initial_state()
    profile_id = "residual-rebase-profile:1"
    intent, receipt = _accepted_station_rebase_intent(
        state,
        profile_id=profile_id,
    )
    cross_clock_intent = replace(
        intent,
        available_clock_domain_id="station.unmapped",
        semantic_content_hash="station-intent-hash:unmapped",
    )
    cross_clock_receipt = replace(
        receipt,
        available_clock_domain_id="station.unmapped",
        intent_hash=cross_clock_intent.semantic_content_hash,
    )
    controller = StationResidualRebaseController(
        profile=StationResidualRebaseProfile(
            profile_id=profile_id,
            composer_source_slot_id="composer-primary",
            enabled=True,
        )
    )

    with pytest.raises(ValueError, match="one mapped clock"):
        controller.request_rebase(
            state=state,
            intent=cross_clock_intent,
            intent_receipt=cross_clock_receipt,
            command_factory=lambda: StationResidualRebaseCommand(
                anchor_request=_successor_request(scope, initial_request),
                effective_not_before_target_at=10.1,
                capture_deadline=10.08,
            ),
        )
