from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from control_composition import (
    ResidualAnchorComponentBinding,
    ResidualAnchorReceipt,
    ResidualAnchorRequest,
    build_composition_activation_scope,
)
from control_composition.trajectory import (
    TRAJECTORY_BASE_LIFECYCLE_EXHAUSTED,
    TRAJECTORY_BASE_LIFECYCLE_REFRESH_PENDING,
    TRAJECTORY_BASE_LIFECYCLE_STABLE,
    TRAJECTORY_INTERACTION_MODE_COMPONENT_CONTROL_REQUESTED,
    TRAJECTORY_INTERACTION_MODE_HAND_BACK,
    TRAJECTORY_INTERACTION_MODE_PASS_THROUGH,
    TRAJECTORY_INTERACTION_MODE_RESIDUAL_ACTIVE,
    TRAJECTORY_TARGET_STATUS_NO_ACTION,
    TRAJECTORY_TRANSITION_CAPABILITY_HOLD_BOUNDARY_ONLY,
    TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_AND_MEASUREMENT,
    TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_ONLY,
    TRAJECTORY_TRANSITION_DECISION_AUTO_ELIGIBLE,
    TRAJECTORY_TRANSITION_DECISION_HARD_INCOMPATIBLE,
    TRAJECTORY_TRANSITION_DECISION_OPERATOR_REVIEW,
    TRAJECTORY_TRANSITION_KIND_BASE_REFRESH,
    NominalBaseBinding,
    NominalBaseTransitionAssessment,
    TrajectoryContinuityCorrection,
    TrajectoryTransitionCapabilitySpec,
    activate_residual_session,
    build_successor_nominal_base_binding,
    commit_trajectory_target_emission,
    empty_trajectory_editing_state,
    offer_nominal_trajectory,
    prepare_trajectory_target,
    revalidate_nominal_base_transition,
    transition_trajectory_session_state,
    validate_trajectory_interaction_base_state,
)
from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentAction,
    ComponentActionSchema,
    ComponentActionTrajectory,
    ComponentActionTrajectoryKeyframe,
    ControlComponentSpecV2,
    ExecutionMeasurementObservation,
    ExecutionObservationCapabilitySpec,
    ExecutionObservationPlan,
    ExecutionReferenceFrameObservation,
    bind_execution_observation_plan,
)
from robot_motion.runtime.control_admission import ControlCut


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _schema() -> ComponentActionSchema:
    payload = ActionPayloadSchema(
        schema_id="lifecycle-arm-pose",
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
        catalog_id="lifecycle-fixture",
        catalog_version="development",
        components=(
            ControlComponentSpecV2(
                component_id="arm",
                controlled_entity_id="arm",
                command_channel_id="arm",
                selection_group_id="arm",
                resource_claims=("resource:arm",),
                accepted_payload_schema_hashes=(
                    payload.semantic_content_hash,
                ),
            ),
        ),
    )
    return ComponentActionSchema(
        schema_id="lifecycle-fixture",
        schema_version="development",
        component_catalog=catalog,
        payload_schema_by_component={"arm": payload},
    )


def _relative_correction_action() -> ComponentAction:
    payload = ActionPayloadSchema(
        schema_id="lifecycle-arm-correction",
        schema_version="development",
        representation_family="cartesian_pose",
        dtype="float64",
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
    )
    catalog = ActionComponentCatalogV2(
        catalog_id="lifecycle-correction-fixture",
        catalog_version="development",
        components=(
            ControlComponentSpecV2(
                component_id="arm",
                controlled_entity_id="arm",
                command_channel_id="arm",
                selection_group_id="arm",
                resource_claims=("resource:arm",),
                accepted_payload_schema_hashes=(
                    payload.semantic_content_hash,
                ),
            ),
        ),
    )
    schema = ComponentActionSchema(
        schema_id="lifecycle-correction-fixture",
        schema_version="development",
        component_catalog=catalog,
        payload_schema_by_component={"arm": payload},
    )
    return ComponentAction(
        action_schema=schema,
        values_by_component={
            "arm": (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        },
    )


def _trajectory(
    revision: int,
    offset: float,
    *,
    source_activation: str = "policy-activation:1",
) -> ComponentActionTrajectory:
    schema = _schema()
    keyframes = tuple(
        ComponentActionTrajectoryKeyframe(
            target_effective_at=10.0 + index * 0.1,
            effective_until=10.1 + index * 0.1,
            action=ComponentAction(
                action_schema=schema,
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
            ),
            source_index=index,
            source_fact_ids=(f"plan:{revision}:step:{index}",),
            source_fact_hashes=(f"plan:{revision}:step-hash:{index}",),
        )
        for index in range(3)
    )
    return ComponentActionTrajectory(
        trajectory_id=f"policy-plan:{revision}",
        revision=revision,
        source_kind="virtual_policy",
        source_instance_id="virtual-policy-worker:1",
        clock_domain_id="control.monotonic",
        created_at=9.0 + revision * 0.01,
        keyframes=keyframes,
        source_fact_ids=(f"plan:{revision}",),
        source_fact_hashes=(f"plan-hash:{revision}",),
        source_activation_id=source_activation,
        source_activation_hash=f"{source_activation}:hash",
    )


def _cut(at: float, sequence: int = 20) -> ControlCut:
    return ControlCut(
        runtime_session_id="runtime:1",
        control_domain_id="control:1",
        clock_domain_id="control.monotonic",
        clock_reset_epoch=0,
        cutoff_at=at,
        cutoff_admission_sequence=sequence,
    )


def _activation_facts(*, captured_at: float = 9.81):
    scope = build_composition_activation_scope(
        operation_spec_id="policy-residual",
        operation_spec_hash="policy-residual-hash",
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
        absolute_action_schema_hash=_schema().semantic_content_hash,
        relative_action_schema_hash="relative-arm-schema-hash",
        requested_at=9.75,
    )
    receipt = ResidualAnchorReceipt.capture(
        request,
        source_action_id="human-action:1",
        source_action_hash="human-action-hash:1",
        source_available_at=9.8,
        captured_at=captured_at,
        component_bindings=(
            ResidualAnchorComponentBinding(
                component_id="arm",
                source_sample_id="human-arm:1",
                source_sample_hash="human-arm-hash:1",
                absolute_payload_schema_hash=(
                    _schema()
                    .payload_schema_by_component["arm"]
                    .semantic_content_hash
                ),
                anchor_id="human-anchor:1",
                anchor_hash="human-anchor-hash:1",
            ),
        ),
    )
    return scope, receipt


def _active_state():
    state = offer_nominal_trajectory(
        empty_trajectory_editing_state(reducer_id="trajectory-editor"),
        _trajectory(1, 1.0),
        offer_sequence=1,
    )
    scope, receipt = _activation_facts()
    return activate_residual_session(
        state,
        residual_session_id="human-residual:1",
        activation_scope=scope,
        control_cut=_cut(9.9),
        residual_anchor_receipt=receipt,
        first_editable_target_at=10.0,
    )


def _observation_capability(*, measurement: bool):
    schema_hash = (
        _schema().payload_schema_by_component["arm"].semantic_content_hash
    )
    lower = ExecutionObservationCapabilitySpec(
        capability_id=(
            "sim-reference-measurement" if measurement else "sim-reference"
        ),
        target_handler_id="virtual-target",
        clock_domain_id="control.monotonic",
        supported_reference_modes=("off", "full_frames"),
        supported_measurement_modes=(
            ("off", "full_frames") if measurement else ("off",)
        ),
        representation_by_component={"arm": "cartesian_pose"},
        schema_hash_by_component={"arm": schema_hash},
        reference_stage_id="scheduled-pose-reference",
        measurement_stage_id=("sim-measurement" if measurement else None),
    )
    plan = ExecutionObservationPlan(
        plan_id="transition-seam-observations",
        reference_mode="full_frames",
        measurement_mode=("full_frames" if measurement else "off"),
        component_ids=("arm",),
    )
    binding = bind_execution_observation_plan(plan, lower, bound_at=9.5)
    return lower, binding


def _transition_capability(level: str):
    if level == TRAJECTORY_TRANSITION_CAPABILITY_HOLD_BOUNDARY_ONLY:
        return TrajectoryTransitionCapabilitySpec(
            capability_id="hold-boundary-only",
            level=level,
            component_ids=("arm",),
            allow_automatic_hand_back=False,
            allow_automatic_base_refresh=False,
        )
    measurement = (
        level
        == TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_AND_MEASUREMENT
    )
    lower, binding = _observation_capability(measurement=measurement)
    return TrajectoryTransitionCapabilitySpec(
        capability_id=f"transition:{level}",
        level=level,
        component_ids=("arm",),
        allow_automatic_hand_back=True,
        allow_automatic_base_refresh=True,
        observation_binding=binding,
        observation_capability=lower,
        maximum_reference_age_sec=0.2,
        maximum_measurement_age_sec=(0.2 if measurement else None),
        reference_tracking_assumption_id=(
            None if measurement else "deterministic-sim-tracking"
        ),
        reference_tracking_assumption_hash=(
            None if measurement else "deterministic-sim-tracking-hash"
        ),
    )


def _reference(capability: TrajectoryTransitionCapabilitySpec):
    binding = capability.observation_binding
    assert binding is not None
    schema_hash = (
        _schema().payload_schema_by_component["arm"].semantic_content_hash
    )
    return ExecutionReferenceFrameObservation(
        reference_frame_id="reference:10.1",
        reference_stage_id="scheduled-pose-reference",
        source_reference_id="reference-source:10.1",
        source_reference_hash="reference-source-hash:10.1",
        target_fact_id="target:10.1",
        target_fact_hash="target-hash:10.1",
        control_domain_id="control:1",
        selection_group_id="arm",
        execution_fence_sequence=1,
        reference_revision=1,
        reference_sequence=11,
        temporal_semantics="point_sample",
        scheduled_for=10.1,
        generated_at=9.91,
        effective_from=None,
        effective_until=None,
        available_at=9.92,
        clock_domain_id="control.monotonic",
        value_by_component={"arm": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)},
        representation_by_component={"arm": "cartesian_pose"},
        schema_hash_by_component={"arm": schema_hash},
        observation_binding_id=binding.binding_id,
        observation_plan_hash=binding.plan_hash,
    )


def _measurement(capability: TrajectoryTransitionCapabilitySpec):
    binding = capability.observation_binding
    assert binding is not None
    schema_hash = (
        _schema().payload_schema_by_component["arm"].semantic_content_hash
    )
    return ExecutionMeasurementObservation(
        measurement_id="measurement:9.91",
        source_measurement_hash="measurement-source-hash:9.91",
        control_domain_id="control:1",
        measured_at=9.91,
        available_at=9.93,
        clock_domain_id="control.monotonic",
        value_by_component={"arm": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)},
        representation_by_component={"arm": "cartesian_pose"},
        schema_hash_by_component={"arm": schema_hash},
        observation_binding_id=binding.binding_id,
        observation_plan_hash=binding.plan_hash,
    )


def _assessment(
    state,
    *,
    capability: TrajectoryTransitionCapabilitySpec,
    candidate: ComponentActionTrajectory | None = None,
    decision: str = TRAJECTORY_TRANSITION_DECISION_AUTO_ELIGIBLE,
):
    assert state.active_base_binding is not None
    candidate = _trajectory(2, 20.0) if candidate is None else candidate
    reference = (
        None
        if capability.level
        == TRAJECTORY_TRANSITION_CAPABILITY_HOLD_BOUNDARY_ONLY
        else _reference(capability)
    )
    measurement = (
        _measurement(capability)
        if capability.level
        == TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_AND_MEASUREMENT
        else None
    )
    return NominalBaseTransitionAssessment(
        transition_kind=TRAJECTORY_TRANSITION_KIND_BASE_REFRESH,
        residual_session_id=state.residual_session_id,
        active_base_binding=state.active_base_binding,
        candidate_trajectory=candidate,
        candidate_offer_sequence=2,
        assessment_control_cut=_cut(9.95, 30),
        residual_fact_id="residual:1",
        residual_fact_hash="residual-hash:1",
        residual_effective_until=10.2,
        switch_at=10.1,
        available_at=9.96,
        publication_basis_hash=state.publication_basis_hash,
        transition_capability=capability,
        old_corrected_action=(
            state.active_trajectory.keyframe_at(10.1).action
        ),
        new_corrected_action=candidate.keyframe_at(10.1).action,
        decision=decision,
        seam_metrics={"translation_l2": 0.01},
        reference_observation=reference,
        measurement_observation=measurement,
        tracking_assessment_id=(
            "tracking-assessment:1" if measurement is not None else None
        ),
        tracking_assessment_hash=(
            "tracking-assessment-hash:1" if measurement is not None else None
        ),
    )


def _revalidate(state, assessment, **overrides):
    values = {
        "checked_control_cut": _cut(10.0, 40),
        "current_base_binding": state.active_base_binding,
        "current_candidate_trajectory": assessment.candidate_trajectory,
        "current_candidate_offer_sequence": assessment.candidate_offer_sequence,
        "current_residual_fact_id": assessment.residual_fact_id,
        "current_residual_fact_hash": assessment.residual_fact_hash,
        "current_residual_effective_until": assessment.residual_effective_until,
        "current_publication_basis_hash": state.publication_basis_hash,
        "current_publication_finalized_target_at": 10.0,
        "current_capability": assessment.transition_capability,
        "current_reference_observation": assessment.reference_observation,
        "current_measurement_observation": assessment.measurement_observation,
        "current_tracking_assessment_id": assessment.tracking_assessment_id,
        "current_tracking_assessment_hash": assessment.tracking_assessment_hash,
    }
    values.update(overrides)
    return revalidate_nominal_base_transition(assessment, **values)


def test_interaction_and_base_lifecycle_matrix_has_profile_gated_exception() -> None:
    validate_trajectory_interaction_base_state(
        TRAJECTORY_INTERACTION_MODE_PASS_THROUGH,
        TRAJECTORY_BASE_LIFECYCLE_STABLE,
    )
    with pytest.raises(ValueError, match="Illegal"):
        validate_trajectory_interaction_base_state(
            TRAJECTORY_INTERACTION_MODE_PASS_THROUGH,
            TRAJECTORY_BASE_LIFECYCLE_REFRESH_PENDING,
        )
    with pytest.raises(ValueError, match="Illegal"):
        validate_trajectory_interaction_base_state(
            TRAJECTORY_INTERACTION_MODE_HAND_BACK,
            TRAJECTORY_BASE_LIFECYCLE_REFRESH_PENDING,
        )
    with pytest.raises(ValueError, match="Illegal"):
        validate_trajectory_interaction_base_state(
            TRAJECTORY_INTERACTION_MODE_COMPONENT_CONTROL_REQUESTED,
            TRAJECTORY_BASE_LIFECYCLE_STABLE,
        )
    validate_trajectory_interaction_base_state(
        TRAJECTORY_INTERACTION_MODE_COMPONENT_CONTROL_REQUESTED,
        TRAJECTORY_BASE_LIFECYCLE_STABLE,
        allow_component_control_on_stable=True,
    )


def test_activation_binding_is_immutable_and_rejects_future_anchor() -> None:
    state = _active_state()
    binding = state.residual_session_activation_binding
    assert binding is not None
    with pytest.raises(FrozenInstanceError):
        binding.residual_session_id = "changed"

    pass_through = offer_nominal_trajectory(
        empty_trajectory_editing_state(reducer_id="trajectory-editor"),
        _trajectory(1, 1.0),
        offer_sequence=1,
    )
    scope, future_receipt = _activation_facts(captured_at=9.91)
    with pytest.raises(ValueError, match="future anchor"):
        activate_residual_session(
            pass_through,
            residual_session_id="human-residual:2",
            activation_scope=scope,
            control_cut=_cut(9.9),
            residual_anchor_receipt=future_receipt,
            first_editable_target_at=10.0,
        )


def test_first_editable_target_must_follow_initial_publication_barrier() -> None:
    state = offer_nominal_trajectory(
        empty_trajectory_editing_state(reducer_id="trajectory-editor"),
        _trajectory(1, 1.0),
        offer_sequence=1,
    )
    preparation = prepare_trajectory_target(state, target_effective_at=10.0)
    committed = commit_trajectory_target_emission(
        state,
        preparation,
        emission_receipt_id="emission:10.0",
        emission_receipt_hash="emission-hash:10.0",
    )
    scope, receipt = _activation_facts()
    with pytest.raises(ValueError, match="follow the initial publication"):
        activate_residual_session(
            committed,
            residual_session_id="human-residual:1",
            activation_scope=scope,
            control_cut=_cut(9.9),
            residual_anchor_receipt=receipt,
            first_editable_target_at=10.0,
        )


def test_source_activation_is_required_for_nominal_base() -> None:
    trajectory = replace(
        _trajectory(1, 1.0),
        source_activation_id=None,
        source_activation_hash=None,
    )
    with pytest.raises(ValueError, match="source activation"):
        NominalBaseBinding(
            residual_session_id="human-residual:1",
            binding_revision=0,
            trajectory=trajectory,
            active_offer_sequence=1,
            bound_for_target_at=10.0,
        )


def test_anchor_from_another_composition_activation_is_rejected() -> None:
    state = offer_nominal_trajectory(
        empty_trajectory_editing_state(reducer_id="trajectory-editor"),
        _trajectory(1, 1.0),
        offer_sequence=1,
    )
    current_scope, _ = _activation_facts()
    stale_scope = build_composition_activation_scope(
        operation_spec_id="policy-residual",
        operation_spec_hash="policy-residual-hash",
        session_epoch=2,
        activated_at=9.7,
        clock_domain_id="control.monotonic",
        input_barrier_at=9.6,
        required_roles=("baseline", "residual"),
    )
    stale_request = ResidualAnchorRequest.for_activation(
        stale_scope,
        request_generation=0,
        source_role="residual",
        component_ids=("arm",),
        source_binding_id="virtual-human:1",
        source_binding_hash="virtual-human-hash:1",
        absolute_action_schema_hash=_schema().semantic_content_hash,
        relative_action_schema_hash="relative-arm-schema-hash",
        requested_at=9.75,
    )
    stale_receipt = ResidualAnchorReceipt.capture(
        stale_request,
        source_action_id="human-action:stale",
        source_action_hash="human-action-hash:stale",
        source_available_at=9.8,
        captured_at=9.81,
        component_bindings=(
            ResidualAnchorComponentBinding(
                component_id="arm",
                source_sample_id="human-arm:stale",
                source_sample_hash="human-arm-hash:stale",
                absolute_payload_schema_hash=(
                    _schema()
                    .payload_schema_by_component["arm"]
                    .semantic_content_hash
                ),
                anchor_id="human-anchor:stale",
                anchor_hash="human-anchor-hash:stale",
            ),
        ),
    )
    with pytest.raises(ValueError, match="another composition activation"):
        activate_residual_session(
            state,
            residual_session_id="human-residual:1",
            activation_scope=current_scope,
            control_cut=_cut(9.9),
            residual_anchor_receipt=stale_receipt,
            first_editable_target_at=10.0,
        )


def test_lifecycle_transition_is_stale_checked_and_exhaustion_is_no_action() -> None:
    state = _active_state()
    with pytest.raises(ValueError, match="stale state"):
        transition_trajectory_session_state(
            state,
            expected_state_hash="old-state-hash",
            interaction_mode=TRAJECTORY_INTERACTION_MODE_RESIDUAL_ACTIVE,
            base_lifecycle_status=TRAJECTORY_BASE_LIFECYCLE_EXHAUSTED,
        )
    exhausted = transition_trajectory_session_state(
        state,
        expected_state_hash=state.semantic_content_hash,
        interaction_mode=TRAJECTORY_INTERACTION_MODE_RESIDUAL_ACTIVE,
        base_lifecycle_status=TRAJECTORY_BASE_LIFECYCLE_EXHAUSTED,
    )
    preparation = prepare_trajectory_target(
        exhausted,
        target_effective_at=10.1,
    )
    assert preparation.status == TRAJECTORY_TARGET_STATUS_NO_ACTION
    assert preparation.reason == "base_support_exhausted"


def test_transition_capability_levels_reject_overclaims() -> None:
    with pytest.raises(ValueError, match="cannot allow automatic"):
        TrajectoryTransitionCapabilitySpec(
            capability_id="invalid-level-zero",
            level=TRAJECTORY_TRANSITION_CAPABILITY_HOLD_BOUNDARY_ONLY,
            component_ids=("arm",),
            allow_automatic_hand_back=True,
            allow_automatic_base_refresh=False,
        )
    lower, binding = _observation_capability(measurement=False)
    with pytest.raises(ValueError, match="tracking assumption"):
        TrajectoryTransitionCapabilitySpec(
            capability_id="invalid-reference-only",
            level=TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_ONLY,
            component_ids=("arm",),
            allow_automatic_hand_back=True,
            allow_automatic_base_refresh=True,
            observation_binding=binding,
            observation_capability=lower,
            maximum_reference_age_sec=0.2,
        )
    with pytest.raises(ValueError, match="requires measurements"):
        TrajectoryTransitionCapabilitySpec(
            capability_id="invalid-reference-measurement",
            level=TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_AND_MEASUREMENT,
            component_ids=("arm",),
            allow_automatic_hand_back=True,
            allow_automatic_base_refresh=True,
            observation_binding=binding,
            observation_capability=lower,
            maximum_reference_age_sec=0.2,
            maximum_measurement_age_sec=0.2,
        )


def test_nonautomatic_assessment_cannot_create_successor_without_receipt() -> None:
    state = _active_state()
    capability = _transition_capability(
        TRAJECTORY_TRANSITION_CAPABILITY_HOLD_BOUNDARY_ONLY
    )
    assessment = _assessment(
        state,
        capability=capability,
        decision=TRAJECTORY_TRANSITION_DECISION_OPERATOR_REVIEW,
    )
    revalidation = _revalidate(state, assessment)
    assert revalidation.accepted
    with pytest.raises(ValueError, match="separate operator"):
        build_successor_nominal_base_binding(assessment, revalidation)


def test_hard_incompatible_candidate_can_be_recorded_without_fake_seam() -> None:
    state = _active_state()
    candidate = _trajectory(2, 20.0)
    assessment = NominalBaseTransitionAssessment(
        transition_kind=TRAJECTORY_TRANSITION_KIND_BASE_REFRESH,
        residual_session_id=state.residual_session_id,
        active_base_binding=state.active_base_binding,
        candidate_trajectory=candidate,
        candidate_offer_sequence=2,
        assessment_control_cut=_cut(9.95, 30),
        residual_fact_id="residual:1",
        residual_fact_hash="residual-hash:1",
        residual_effective_until=10.2,
        switch_at=10.1,
        available_at=9.96,
        publication_basis_hash=state.publication_basis_hash,
        transition_capability=_transition_capability(
            TRAJECTORY_TRANSITION_CAPABILITY_HOLD_BOUNDARY_ONLY
        ),
        old_corrected_action=None,
        new_corrected_action=None,
        decision=TRAJECTORY_TRANSITION_DECISION_HARD_INCOMPATIBLE,
        reason_codes=("frame_adapter_missing",),
    )
    assert assessment.decision == (
        TRAJECTORY_TRANSITION_DECISION_HARD_INCOMPATIBLE
    )
    assert assessment.old_corrected_action is None


def test_auto_assessment_revalidation_builds_immutable_successor_lineage() -> None:
    state = _active_state()
    capability = _transition_capability(
        TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_ONLY
    )
    assessment = _assessment(state, capability=capability)
    revalidation = _revalidate(state, assessment)
    assert revalidation.accepted

    successor = build_successor_nominal_base_binding(
        assessment,
        revalidation,
    )
    assert successor.binding_revision == 1
    assert successor.predecessor_binding_hash == (
        state.active_base_binding.semantic_content_hash
    )
    assert successor.transition_assessment_hash == (
        assessment.semantic_content_hash
    )
    assert successor.transition_revalidation_hash == (
        revalidation.semantic_content_hash
    )
    assert state.active_base_binding.binding_revision == 0


def test_continuity_correction_has_separate_residual_and_base_lineage() -> None:
    state = _active_state()
    assessment = _assessment(
        state,
        capability=_transition_capability(
            TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_ONLY
        ),
    )
    revalidation = _revalidate(state, assessment)
    successor = build_successor_nominal_base_binding(
        assessment,
        revalidation,
    )
    correction = TrajectoryContinuityCorrection(
        assessment=assessment,
        revalidation=revalidation,
        from_base_binding=state.active_base_binding,
        to_base_binding=successor,
        seam_reference_id=assessment.reference_observation.reference_frame_id,
        seam_reference_hash=(
            assessment.reference_observation.semantic_content_hash
        ),
        residual_fact_id_at_activation=assessment.residual_fact_id,
        residual_fact_hash_at_activation=assessment.residual_fact_hash,
        correction_action=_relative_correction_action(),
        decay_profile_id="linear-decay:1",
        decay_profile_hash="linear-decay-hash:1",
        effective_until=10.2,
    )
    assert correction.from_base_binding.binding_revision == 0
    assert correction.to_base_binding.binding_revision == 1
    assert correction.residual_fact_hash_at_activation == "residual-hash:1"
    assert state.residual_session_activation_binding.semantic_content_hash != (
        correction.semantic_content_hash
    )


def test_reference_and_measurement_transition_binds_tracking_facts() -> None:
    state = _active_state()
    assessment = _assessment(
        state,
        capability=_transition_capability(
            TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_AND_MEASUREMENT
        ),
    )
    accepted = _revalidate(state, assessment)
    assert accepted.accepted

    changed_tracking = _revalidate(
        state,
        assessment,
        current_tracking_assessment_id="tracking-assessment:2",
        current_tracking_assessment_hash="tracking-assessment-hash:2",
    )
    assert not changed_tracking.accepted
    assert "tracking_assessment_changed" in changed_tracking.reason_codes

    stale_measurement = replace(
        assessment.measurement_observation,
        measured_at=9.7,
    )
    with pytest.raises(ValueError, match="Measurement observation is stale"):
        replace(
            assessment,
            measurement_observation=stale_measurement,
        )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (
            {"current_residual_fact_hash": "new-residual-hash"},
            "residual_basis_changed_requires_reassessment",
        ),
        (
            {"current_publication_finalized_target_at": 10.1},
            "publication_barrier_reached_switch",
        ),
        (
            {
                "current_candidate_trajectory": _trajectory(3, 30.0),
                "current_candidate_offer_sequence": 3,
            },
            "candidate_shadow_changed",
        ),
    ],
)
def test_transition_revalidation_rejects_changed_basis(overrides, reason) -> None:
    state = _active_state()
    assessment = _assessment(
        state,
        capability=_transition_capability(
            TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_ONLY
        ),
    )
    result = _revalidate(state, assessment, **overrides)
    assert not result.accepted
    assert reason in result.reason_codes


def test_transition_revalidation_rejects_changed_base_and_capability() -> None:
    state = _active_state()
    assessment = _assessment(
        state,
        capability=_transition_capability(
            TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_ONLY
        ),
    )
    changed_base = NominalBaseBinding(
        residual_session_id=state.residual_session_id,
        binding_revision=0,
        trajectory=_trajectory(1, 2.0),
        active_offer_sequence=1,
        bound_for_target_at=10.0,
    )
    base_result = _revalidate(
        state,
        assessment,
        current_base_binding=changed_base,
    )
    assert not base_result.accepted
    assert "active_base_binding_changed" in base_result.reason_codes

    capability_result = _revalidate(
        state,
        assessment,
        current_capability=_transition_capability(
            TRAJECTORY_TRANSITION_CAPABILITY_HOLD_BOUNDARY_ONLY
        ),
    )
    assert not capability_result.accepted
    assert "transition_capability_changed" in capability_result.reason_codes
    assert (
        "reference_not_supported_by_current_capability"
        in capability_result.reason_codes
    )


def test_auto_transition_cannot_cross_source_activation() -> None:
    state = _active_state()
    candidate = _trajectory(
        2,
        20.0,
        source_activation="policy-activation:2",
    )
    with pytest.raises(ValueError, match="cross source activation"):
        _assessment(
            state,
            capability=_transition_capability(
                TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_ONLY
            ),
            candidate=candidate,
        )


def test_assessment_must_arrive_before_switch_and_revalidate_before_boundary() -> None:
    state = _active_state()
    capability = _transition_capability(
        TRAJECTORY_TRANSITION_CAPABILITY_REFERENCE_ONLY
    )
    assessment = _assessment(state, capability=capability)
    with pytest.raises(ValueError, match="before switch_at"):
        replace(assessment, available_at=10.1)

    result = _revalidate(
        state,
        assessment,
        checked_control_cut=_cut(10.1, 50),
    )
    assert not result.accepted
    assert "switch_boundary_reached" in result.reason_codes
