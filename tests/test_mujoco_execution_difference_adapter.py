from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.contract]

from control_composition.composition import CompositionInputSnapshot, CompositionEvaluation
from control_composition.observability import (
    DIFFERENCE_MODULE_POST_COMPOSITION_EXECUTION,
    associate_measurement_to_reference,
    validate_difference_observation_lineage,
)
from robot_motion.action import (
    admit_reference_segment,
    build_reference_endpoint_parent,
    build_reference_segment,
    build_reference_segment_terminal,
)
from sim.mujoco.adapters.reference_lineage import (
    build_mujoco_reference_target_activation_observation,
)
from sim.mujoco.deploy.execution_difference import (
    build_mujoco_qpos_execution_difference_observation,
    project_mujoco_execution_measurement_observation,
    project_mujoco_execution_reference_frame,
    project_mujoco_execution_reference_sample,
)
from sim.mujoco.deploy.measurement import MujocoQposMeasurement


def _snapshot() -> CompositionInputSnapshot:
    return CompositionInputSnapshot(
        composition_input_snapshot_id="input-snapshot:sim-execution",
        control_domain_id="robot-0",
        cutoff_at=9.0,
        cutoff_admission_sequence=4,
        target_effective_at=10.0,
        composition_spec_hash="composition-spec:1",
        action_schema_hash="action-schema:1",
        target_clock_domain_id="control.monotonic",
        selected_input_by_role={},
        missing_roles=(),
        producer_epoch_snapshot_hash="epochs:1",
        semantic_content_hash="input-snapshot-hash:sim-execution",
    )


def _result(snapshot: CompositionInputSnapshot) -> CompositionEvaluation:
    return CompositionEvaluation(
        composition_evaluation_id="composition-evaluation:sim-execution",
        composition_input_snapshot_id=snapshot.composition_input_snapshot_id,
        composition_input_snapshot_hash=snapshot.semantic_content_hash,
        composition_spec_hash=snapshot.composition_spec_hash,
        action_schema_hash=snapshot.action_schema_hash,
        target_effective_at=snapshot.target_effective_at,
        target_clock_domain_id=snapshot.target_clock_domain_id,
        output_status="no_action",
        reason="fixture",
        composed_candidate=None,
        composed_candidate_hash=None,
        input_ids=(),
        input_hashes=(),
        context_ids=(),
        semantic_content_hash="composition-evaluation-hash:sim-execution",
    )


def _reference_facts(*, terminal_at: float | None = None):
    start = build_reference_endpoint_parent(
        role="start",
        source_kind="measured_rebase",
        source_fact_id="measurement-rebase:1",
        projected_group_action_hash="start-action-hash",
        action_schema_hash="action-schema",
        frame_schema_hash="frame-schema",
        target_effective_at=10.0,
    )
    end = build_reference_endpoint_parent(
        role="end",
        source_kind="current_safe_target",
        source_fact_id="safety-result:1",
        projected_group_action_hash="end-action-hash",
        action_schema_hash="action-schema",
        frame_schema_hash="frame-schema",
        target_effective_at=11.0,
    )
    segment = build_reference_segment(
        control_domain_id="robot-0",
        selection_group_id="arms",
        execution_epoch=3,
        group_segment_generation=4,
        authority_kind="authoritative_shared",
        start_parent=start,
        end_parent=end,
        component_dependencies=("right_arm",),
        created_at=9.0,
        clock_domain_id="control.monotonic",
        interpolation_contract_hash="interpolation:1",
        resolver_contract_hash="resolver:1",
        input_action_schema_hash="action-schema",
        output_reference_schema_hash="right-arm-qpos-schema",
    )
    admission = admit_reference_segment(
        segment,
        receiver_id="mujoco-receiver",
        admitted_at=9.5,
        admission_sequence=8,
        current_execution_epoch=3,
        expected_group_segment_generation=4,
        achieved_admission_guarantee="interpolation_segment_accepted",
    )
    activation = build_mujoco_reference_target_activation_observation(
        reference_segment_id=segment.reference_segment_id,
        generation_activation_id="generation-activation:4",
        reference_sequence=12,
        execution_epoch=3,
        group_segment_generation=4,
        scheduled_for=10.0,
        activated_at_worker_time=10.1,
        observed_at=10.11,
        qref_by_side={"right": (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)},
        activation_status_by_side={"right": "tracking"},
    )
    terminal = None
    if terminal_at is not None:
        terminal = build_reference_segment_terminal(
            segment,
            status="canceled",
            effective_at=terminal_at,
            available_at=terminal_at + 0.01,
            terminal_admission_sequence=9,
        )
    return segment, admission, activation, terminal


def _measurement(*, measured_at: float = 10.2) -> MujocoQposMeasurement:
    values = (0.5, 1.0, 2.0, 3.0, 4.0, 4.0)
    return MujocoQposMeasurement(
        measurement_id=f"mujoco-measurement:{measured_at}",
        qpos_by_side={"right": values},
        measured_at=measured_at,
        clock_domain_id="control.monotonic",
        physics_tick=20,
        semantic_content_hash=f"mujoco-measurement-hash:{measured_at}",
    )


def test_mujoco_adapter_projects_internal_generation_to_public_execution_identity() -> None:
    segment, admission, activation, terminal = _reference_facts()
    reference = project_mujoco_execution_reference_frame(
        segment=segment,
        admission=admission,
        activation=activation,
        terminal=terminal,
        side_to_component_id={"right": "right_arm"},
        component_schema_hash="right-arm-qpos-schema",
    )
    measurement = project_mujoco_execution_measurement_observation(
        _measurement(),
        control_domain_id="robot-0",
        side_to_component_id={"right": "right_arm"},
        component_schema_hash="right-arm-qpos-schema",
        available_at=10.21,
    )
    association = associate_measurement_to_reference(
        measurement,
        reference,
        selection_group_id="arms",
        component_ids=("right_arm",),
    )

    assert association.status == "available"
    assert association.execution_fence_sequence == 3
    assert association.reference_revision == 4
    assert tuple(item.stage for item in association.receipt_references) == (
        "reference_segment_admission",
        "reference_activation",
        "actual_reference_frame",
        "measurement",
        "measurement_observation",
    )
    assert association.component_difference["right_arm"]["signed"] == pytest.approx(
        (0.5, 0.0, 0.0, 0.0, 0.0, -1.0)
    )

    snapshot = _snapshot()
    result = _result(snapshot)
    observation = build_mujoco_qpos_execution_difference_observation(
        input_snapshot=snapshot,
        composition_evaluation=result,
        association=association,
        observed_at=10.22,
    )
    assert observation.status == "available"
    assert observation.payload["difference_scope"] == "reference_to_measured"
    validate_difference_observation_lineage(
        observation,
        module_id=DIFFERENCE_MODULE_POST_COMPOSITION_EXECUTION,
        input_snapshot=snapshot,
        composition_evaluation=result,
    )


def test_mujoco_terminal_prevents_late_measurement_from_reusing_canceled_reference() -> None:
    segment, admission, activation, terminal = _reference_facts(terminal_at=10.3)
    reference = project_mujoco_execution_reference_frame(
        segment=segment,
        admission=admission,
        activation=activation,
        terminal=terminal,
        side_to_component_id={"right": "right_arm"},
        component_schema_hash="right-arm-qpos-schema",
    )
    measurement = project_mujoco_execution_measurement_observation(
        _measurement(measured_at=10.3),
        control_domain_id="robot-0",
        side_to_component_id={"right": "right_arm"},
        component_schema_hash="right-arm-qpos-schema",
        available_at=10.31,
    )
    association = associate_measurement_to_reference(
        measurement,
        reference,
        selection_group_id="arms",
        component_ids=("right_arm",),
    )

    assert association.status == "unavailable"
    assert association.reason == "reference_not_effective_at_measurement"
    assert association.reference_frame_id is None
    assert tuple(item.stage for item in association.receipt_references) == (
        "reference_segment_admission",
        "reference_activation",
        "reference_segment_terminal",
        "actual_reference_frame",
        "measurement",
        "measurement_observation",
    )


def test_mujoco_owner_can_return_measurement_coupled_reference_point_sample() -> None:
    segment, admission, activation, _ = _reference_facts()
    reference = project_mujoco_execution_reference_sample(
        segment=segment,
        admission=admission,
        activation=activation,
        sampled_at=10.2,
        available_at=10.21,
        side_to_component_id={"right": "right_arm"},
        component_schema_hash="right-arm-qpos-schema",
        observation_binding_id="observation-binding:1",
        observation_plan_hash="observation-plan-hash:1",
    )
    measurement = project_mujoco_execution_measurement_observation(
        _measurement(measured_at=10.2),
        control_domain_id="robot-0",
        side_to_component_id={"right": "right_arm"},
        component_schema_hash="right-arm-qpos-schema",
        available_at=10.22,
        observation_binding_id="observation-binding:1",
        observation_plan_hash="observation-plan-hash:1",
    )
    association = associate_measurement_to_reference(
        measurement,
        reference,
        selection_group_id="arms",
        component_ids=("right_arm",),
    )

    assert reference.temporal_semantics == "point_sample"
    assert association.status == "available"
    assert association.reason == "downstream_actual_reference_at_measurement"


def test_mujoco_adapter_rejects_rejected_or_quick_stop_reference_as_actual_qref() -> None:
    segment, admission, activation, _ = _reference_facts()
    rejected = admit_reference_segment(
        segment,
        receiver_id="mujoco-receiver",
        admitted_at=9.5,
        admission_sequence=8,
        current_execution_epoch=99,
        expected_group_segment_generation=4,
        achieved_admission_guarantee="interpolation_segment_accepted",
    )
    with pytest.raises(ValueError, match="rejected segment"):
        project_mujoco_execution_reference_frame(
            segment=segment,
            admission=rejected,
            activation=activation,
            side_to_component_id={"right": "right_arm"},
            component_schema_hash="right-arm-qpos-schema",
        )

    quick_stop = build_mujoco_reference_target_activation_observation(
        reference_segment_id=segment.reference_segment_id,
        generation_activation_id="generation-activation:4",
        reference_sequence=12,
        execution_epoch=3,
        group_segment_generation=4,
        scheduled_for=10.0,
        activated_at_worker_time=10.1,
        observed_at=10.11,
        qref_by_side={"right": (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)},
        activation_status_by_side={"right": "quick_stop"},
    )
    with pytest.raises(ValueError, match="tracking/degraded"):
        project_mujoco_execution_reference_frame(
            segment=segment,
            admission=admission,
            activation=quick_stop,
            side_to_component_id={"right": "right_arm"},
            component_schema_hash="right-arm-qpos-schema",
        )
