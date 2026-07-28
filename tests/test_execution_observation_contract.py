from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.contract]

from robot_motion.action import (
    ExecutionMeasurementObservation,
    ExecutionObservationCapabilitySpec,
    ExecutionObservationPlan,
    ExecutionObservationWatermark,
    ExecutionReferenceFrameObservation,
    bind_execution_observation_plan,
    validate_execution_observation_binding,
)


def _frame(**overrides: object) -> ExecutionReferenceFrameObservation:
    values = {
        "reference_frame_id": "reference-frame:1",
        "reference_stage_id": "fixture.active_qref",
        "source_reference_id": "source-reference:1",
        "source_reference_hash": "source-reference-hash:1",
        "target_fact_id": "execution-target:1",
        "target_fact_hash": "execution-target-hash:1",
        "control_domain_id": "robot-0",
        "selection_group_id": "arms",
        "execution_fence_sequence": 3,
        "reference_revision": 4,
        "reference_sequence": 5,
        "temporal_semantics": "active_interval",
        "scheduled_for": 10.0,
        "generated_at": 10.01,
        "effective_from": 10.01,
        "effective_until": 10.02,
        "available_at": 10.011,
        "clock_domain_id": "control.monotonic",
        "value_by_component": {"right_arm": (0.0, 1.0)},
        "representation_by_component": {"right_arm": "joint_position"},
        "schema_hash_by_component": {"right_arm": "right-arm-qpos-schema"},
    }
    values.update(overrides)
    return ExecutionReferenceFrameObservation(**values)  # type: ignore[arg-type]


def test_execution_observation_plan_is_default_off_and_demand_bounded() -> None:
    disabled = ExecutionObservationPlan(plan_id="execution-observation:off")
    assert disabled.reference_mode == "off"
    assert disabled.maximum_reference_publish_hz is None

    decimated = ExecutionObservationPlan(
        plan_id="execution-observation:gui",
        reference_mode="decimated_frames",
        component_ids=("right_arm",),
        maximum_reference_publish_hz=20.0,
    )
    assert decimated.reference_mode == "decimated_frames"
    assert decimated.maximum_reference_publish_hz == 20.0
    assert disabled.semantic_content_hash != decimated.semantic_content_hash

    with pytest.raises(ValueError, match="requires maximum"):
        ExecutionObservationPlan(
            plan_id="execution-observation:invalid",
            reference_mode="decimated_frames",
        )


def test_reference_frame_is_stage_qualified_and_uses_half_open_interval() -> None:
    frame = _frame()
    assert frame.reference_stage_id == "fixture.active_qref"
    assert frame.covers(10.01)
    assert not frame.covers(10.02)


def test_reference_frame_cannot_be_available_before_it_is_generated() -> None:
    with pytest.raises(ValueError, match="available before generation"):
        _frame(generated_at=10.02, available_at=10.01)


def test_point_sample_cannot_claim_an_active_interval() -> None:
    point = _frame(
        temporal_semantics="point_sample",
        effective_from=None,
        effective_until=None,
    )
    assert not point.covers(10.01)
    with pytest.raises(ValueError, match="cannot claim an active interval"):
        _frame(temporal_semantics="point_sample")


def _capability() -> ExecutionObservationCapabilitySpec:
    return ExecutionObservationCapabilitySpec(
        capability_id="fixture-observation-capability",
        target_handler_id="fixture-target",
        clock_domain_id="control.monotonic",
        supported_reference_modes=(
            "off",
            "decimated_frames",
            "full_frames",
        ),
        supported_measurement_modes=("off", "decimated_frames", "full_frames"),
        representation_by_component={"right_arm": "joint_position"},
        schema_hash_by_component={"right_arm": "right-arm-qpos-schema"},
        reference_stage_id="fixture.active_qref",
        measurement_stage_id="fixture.measured_qpos",
    )


def test_non_off_plan_requires_exact_capability_binding() -> None:
    plan = ExecutionObservationPlan(
        plan_id="execution-observation:required",
        reference_mode="full_frames",
        measurement_mode="full_frames",
        component_ids=("right_arm",),
    )
    capability = _capability()
    binding = bind_execution_observation_plan(
        plan,
        capability,
        bound_at=9.0,
    )
    assert binding.plan_hash == plan.semantic_content_hash
    assert binding.capability_hash == capability.semantic_content_hash

    with pytest.raises(ValueError, match="does not cover components"):
        bind_execution_observation_plan(
            ExecutionObservationPlan(
                plan_id="execution-observation:unknown-component",
                reference_mode="full_frames",
                component_ids=("left_arm",),
            ),
            capability,
            bound_at=9.0,
        )
    with pytest.raises(ValueError, match="off observation plan"):
        bind_execution_observation_plan(
            ExecutionObservationPlan(plan_id="execution-observation:off"),
            capability,
            bound_at=9.0,
        )


def test_bound_observation_and_watermark_make_fulfillment_auditable() -> None:
    plan = ExecutionObservationPlan(
        plan_id="execution-observation:required",
        reference_mode="full_frames",
        measurement_mode="full_frames",
        component_ids=("right_arm",),
    )
    capability = _capability()
    binding = bind_execution_observation_plan(plan, capability, bound_at=9.0)
    frame = _frame(
        observation_binding_id=binding.binding_id,
        observation_plan_hash=binding.plan_hash,
    )
    validate_execution_observation_binding(frame, binding, capability)

    measurement = ExecutionMeasurementObservation(
        measurement_id="measurement:1",
        source_measurement_hash="measurement-hash:1",
        control_domain_id="robot-0",
        measured_at=10.015,
        available_at=10.016,
        clock_domain_id="control.monotonic",
        value_by_component={"right_arm": (0.0, 1.0)},
        representation_by_component={"right_arm": "joint_position"},
        schema_hash_by_component={"right_arm": "right-arm-qpos-schema"},
        observation_binding_id=binding.binding_id,
        observation_plan_hash=binding.plan_hash,
    )
    validate_execution_observation_binding(measurement, binding, capability)

    with pytest.raises(ValueError, match="exactly cover"):
        validate_execution_observation_binding(
            ExecutionMeasurementObservation(
                measurement_id="measurement:missing-component",
                source_measurement_hash="measurement-hash:missing-component",
                control_domain_id="robot-0",
                measured_at=10.015,
                available_at=10.016,
                clock_domain_id="control.monotonic",
                value_by_component={"left_arm": (0.0, 1.0)},
                representation_by_component={"left_arm": "joint_position"},
                schema_hash_by_component={
                    "left_arm": "left-arm-qpos-schema"
                },
                observation_binding_id=binding.binding_id,
                observation_plan_hash=binding.plan_hash,
            ),
            binding,
            capability,
        )

    watermark = ExecutionObservationWatermark(
        observation_binding_id=binding.binding_id,
        observation_plan_hash=binding.plan_hash,
        stream="measurement",
        processed_through=10.015,
        latest_sequence=1,
        emitted_count=1,
        dropped_count=0,
        status="healthy",
        reason="all_selected_frames_published",
        available_at=10.016,
        clock_domain_id="control.monotonic",
    )
    assert watermark.status == "healthy"

    with pytest.raises(ValueError, match="cannot report drops"):
        ExecutionObservationWatermark(
            observation_binding_id=binding.binding_id,
            observation_plan_hash=binding.plan_hash,
            stream="measurement",
            processed_through=10.015,
            latest_sequence=1,
            emitted_count=1,
            dropped_count=1,
            status="healthy",
            reason="invalid",
            available_at=10.016,
            clock_domain_id="control.monotonic",
        )
