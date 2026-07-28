from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.contract]

from control_composition.composition import CompositionInputSnapshot, CompositionEvaluation
from control_composition.observability import (
    DIFFERENCE_MODULE_COMPOSITION_IMMEDIATE,
    DIFFERENCE_MODULE_POST_COMPOSITION_EXECUTION,
    DifferenceObservationEnvelope,
    DifferenceReceiptReference,
    DisabledCompositionImmediateDifferenceObserver,
    DisabledPostCompositionExecutionDifferenceObserver,
    associate_measurement_to_reference,
    validate_difference_observation_lineage,
)
from robot_motion.action import (
    ExecutionMeasurementObservation,
    ExecutionReferenceFrameObservation,
)


def _snapshot() -> CompositionInputSnapshot:
    return CompositionInputSnapshot(
        composition_input_snapshot_id="input-snapshot:1",
        control_domain_id="robot-0",
        cutoff_at=10.0,
        cutoff_admission_sequence=4,
        target_effective_at=10.2,
        composition_spec_hash="composition-spec:1",
        action_schema_hash="action-schema:1",
        target_clock_domain_id="control.monotonic",
        selected_input_by_role={},
        missing_roles=("human",),
        producer_epoch_snapshot_hash="epochs:1",
        semantic_content_hash="input-snapshot-hash:1",
    )


def _result(snapshot: CompositionInputSnapshot) -> CompositionEvaluation:
    return CompositionEvaluation(
        composition_evaluation_id="composition-evaluation:1",
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
        semantic_content_hash="composition-evaluation-hash:1",
    )


def test_immediate_disabled_observer_is_input_only_and_lineage_bound() -> None:
    snapshot = _snapshot()
    observation = DisabledCompositionImmediateDifferenceObserver().observe(
        snapshot,
        observed_at=10.1,
        clock_domain_id="control.monotonic",
    )

    assert observation.module_id == DIFFERENCE_MODULE_COMPOSITION_IMMEDIATE
    assert observation.status == "unavailable"
    assert observation.reason == "disabled"
    assert observation.composition_evaluation_id is None
    validate_difference_observation_lineage(
        observation,
        module_id=DIFFERENCE_MODULE_COMPOSITION_IMMEDIATE,
        input_snapshot=snapshot,
    )


def test_post_observer_keeps_result_and_opaque_receipt_lineage() -> None:
    snapshot = _snapshot()
    result = _result(snapshot)
    reference = DifferenceReceiptReference(
        stage="dispatch",
        receipt_id="dispatch:1",
        receipt_hash="dispatch-hash:1",
        observed_at=10.3,
        clock_domain_id="control.monotonic",
    )
    observation = DisabledPostCompositionExecutionDifferenceObserver().observe(
        snapshot,
        result,
        observed_at=10.4,
        clock_domain_id="control.monotonic",
        receipt_references=(reference,),
    )

    assert observation.module_id == DIFFERENCE_MODULE_POST_COMPOSITION_EXECUTION
    assert observation.composition_evaluation_id == result.composition_evaluation_id
    assert observation.receipt_references == (reference,)
    validate_difference_observation_lineage(
        observation,
        module_id=DIFFERENCE_MODULE_POST_COMPOSITION_EXECUTION,
        input_snapshot=snapshot,
        composition_evaluation=result,
    )


def test_envelope_preserves_opaque_payload_without_freezing_metric_catalogue() -> None:
    snapshot = _snapshot()
    observation = DifferenceObservationEnvelope(
        module_id=DIFFERENCE_MODULE_COMPOSITION_IMMEDIATE,
        capability_id="fixture.pose-qpos.experimental",
        status="available",
        reason="fixture",
        control_domain_id=snapshot.control_domain_id,
        clock_domain_id="control.monotonic",
        cutoff_at=snapshot.cutoff_at,
        cutoff_admission_sequence=snapshot.cutoff_admission_sequence,
        target_effective_at=snapshot.target_effective_at,
        observed_at=10.1,
        valid_until=10.5,
        input_snapshot_id=snapshot.composition_input_snapshot_id,
        input_snapshot_hash=snapshot.semantic_content_hash,
        payload_contract_id="fixture-difference-payload",
        payload_contract_hash="fixture-difference-payload-hash",
        payload={"metric_family": "pose", "future_extension": {"value": 1}},
    )

    assert observation.payload["metric_family"] == "pose"
    with pytest.raises(TypeError):
        observation.payload["new"] = "must-not-mutate"  # type: ignore[index]
    assert observation.observation_id.startswith("difference-observation:")


def test_lineage_rejects_result_on_input_only_observation() -> None:
    snapshot = _snapshot()
    result = _result(snapshot)
    observation = DisabledCompositionImmediateDifferenceObserver().observe(
        snapshot,
        observed_at=10.1,
        clock_domain_id="control.monotonic",
    )
    with pytest.raises(ValueError, match="requires result lineage"):
        validate_difference_observation_lineage(
            observation,
            module_id=DIFFERENCE_MODULE_COMPOSITION_IMMEDIATE,
            input_snapshot=snapshot,
            composition_evaluation=result,
        )


def _reference(
    *,
    frame_id: str,
    reference_time: float,
    effective_from: float,
    effective_until: float | None,
    value: tuple[float, ...],
    clock_domain_id: str = "control.monotonic",
    schema_hash: str = "joint-schema:arbitrary",
    generation: int = 1,
) -> ExecutionReferenceFrameObservation:
    return ExecutionReferenceFrameObservation(
        reference_frame_id=frame_id,
        reference_stage_id="fixture.active_qref",
        source_reference_id=f"{frame_id}:source",
        source_reference_hash=f"{frame_id}:source-hash",
        target_fact_id=f"{frame_id}:target",
        target_fact_hash=f"{frame_id}:target-hash",
        control_domain_id="robot-0",
        selection_group_id="arms",
        execution_fence_sequence=2,
        reference_revision=generation,
        reference_sequence=generation,
        temporal_semantics="active_interval",
        scheduled_for=reference_time,
        generated_at=effective_from,
        effective_from=effective_from,
        effective_until=effective_until,
        available_at=effective_from,
        clock_domain_id=clock_domain_id,
        value_by_component={"tooling.primary": value},
        representation_by_component={"tooling.primary": "joint_position"},
        schema_hash_by_component={"tooling.primary": schema_hash},
    )


def _measurement(
    *,
    measured_at: float,
    value: tuple[float, ...],
    clock_domain_id: str = "control.monotonic",
    schema_hash: str = "joint-schema:arbitrary",
    available_at: float | None = None,
) -> ExecutionMeasurementObservation:
    return ExecutionMeasurementObservation(
        measurement_id=f"measurement:{measured_at}",
        source_measurement_hash=f"measurement-hash:{measured_at}",
        control_domain_id="robot-0",
        measured_at=measured_at,
        available_at=measured_at if available_at is None else available_at,
        clock_domain_id=clock_domain_id,
        value_by_component={"tooling.primary": value},
        representation_by_component={"tooling.primary": "joint_position"},
        schema_hash_by_component={"tooling.primary": schema_hash},
    )


def test_execution_difference_requires_one_downstream_paired_actual_reference() -> None:
    old = _reference(
        frame_id="reference:old",
        reference_time=10.0,
        effective_from=10.0,
        effective_until=10.5,
        value=(0.0, 1.0, 2.0),
    )
    # Its logical target time is closer to the measurement, but the worker has
    # not activated it yet.
    late_future = _reference(
        frame_id="reference:late-future",
        reference_time=10.39,
        effective_from=10.5,
        effective_until=None,
        value=(9.0, 9.0, 9.0),
        generation=2,
    )
    measurement = _measurement(measured_at=10.4, value=(0.5, 0.0, 2.0))
    with pytest.raises(
        TypeError,
        match="ExecutionReferenceFrameObservation or None",
    ):
        associate_measurement_to_reference(
            measurement,
            (late_future, old),  # type: ignore[arg-type]
            selection_group_id="arms",
            component_ids=("tooling.primary",),
        )
    association = associate_measurement_to_reference(
        measurement,
        old,
        selection_group_id="arms",
        component_ids=("tooling.primary",),
    )

    assert association.status == "available"
    assert association.reference_frame_id == old.reference_frame_id
    difference = association.component_difference["tooling.primary"]
    assert difference["signed"] == pytest.approx((0.5, -1.0, 0.0))
    assert difference["linf"] == pytest.approx(1.0)


def test_execution_difference_uses_half_open_reference_interval() -> None:
    association = associate_measurement_to_reference(
        _measurement(measured_at=10.5, value=(0.0,)),
        _reference(
            frame_id="reference:expired",
            reference_time=10.0,
            effective_from=10.0,
            effective_until=10.5,
            value=(0.0,),
        ),
        selection_group_id="arms",
        component_ids=("tooling.primary",),
    )

    assert association.status == "unavailable"
    assert association.reason == "reference_not_effective_at_measurement"
    assert association.reference_frame_id is None


def test_execution_difference_rejects_cross_clock_and_schema_guessing() -> None:
    cross_clock = associate_measurement_to_reference(
        _measurement(measured_at=10.2, value=(0.0,)),
        _reference(
            frame_id="reference:clock",
            reference_time=10.0,
            effective_from=10.0,
            effective_until=None,
            value=(0.0,),
            clock_domain_id="worker.monotonic",
        ),
        selection_group_id="arms",
        component_ids=("tooling.primary",),
    )
    assert cross_clock.status == "incomparable"
    assert cross_clock.reason == "clock_domain_mismatch"

    schema_mismatch = associate_measurement_to_reference(
        _measurement(
            measured_at=10.2,
            value=(0.0,),
            schema_hash="measurement-schema",
        ),
        _reference(
            frame_id="reference:schema",
            reference_time=10.0,
            effective_from=10.0,
            effective_until=None,
            value=(0.0,),
            schema_hash="reference-schema",
        ),
        selection_group_id="arms",
        component_ids=("tooling.primary",),
    )
    assert schema_mismatch.status == "incomparable"
    assert schema_mismatch.reason == "schema_mismatch:tooling.primary"


def test_execution_difference_requires_reference_fact_to_be_observable_causally() -> None:
    reference = ExecutionReferenceFrameObservation(
        reference_frame_id="reference:late-observation",
        reference_stage_id="fixture.active_qref",
        source_reference_id="reference:late-observation:source",
        source_reference_hash="reference:late-observation:source-hash",
        target_fact_id="target:late-observation",
        target_fact_hash="target:late-observation:hash",
        control_domain_id="robot-0",
        selection_group_id="arms",
        execution_fence_sequence=2,
        reference_revision=1,
        reference_sequence=1,
        temporal_semantics="active_interval",
        scheduled_for=10.0,
        generated_at=10.0,
        effective_from=10.0,
        effective_until=None,
        available_at=10.3,
        clock_domain_id="control.monotonic",
        value_by_component={"tooling.primary": (0.0,)},
        representation_by_component={"tooling.primary": "joint_position"},
        schema_hash_by_component={"tooling.primary": "joint-schema:arbitrary"},
    )
    association = associate_measurement_to_reference(
        _measurement(
            measured_at=10.1,
            available_at=10.2,
            value=(0.0,),
        ),
        reference,
        selection_group_id="arms",
        component_ids=("tooling.primary",),
    )

    assert association.status == "unavailable"
    assert (
        association.reason
        == "reference_fact_not_available_at_measurement_observation"
    )
