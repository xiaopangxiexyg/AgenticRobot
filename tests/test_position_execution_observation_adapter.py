from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.contract]

from control_composition.observability import associate_measurement_to_reference
from robot_motion.action import ExecutionMeasurementObservation
from robot_motion.position_control import ResampledMotionReference
from teleop_runtime.position_control import (
    POSITION_PRE_RESOLVER_REFERENCE_STAGE,
    LegacyReferenceDerivation,
    LegacyReferenceSourceFact,
    project_position_pre_resolver_reference_frame,
)


def _derivation() -> LegacyReferenceDerivation:
    start = LegacyReferenceSourceFact(
        source_index=0,
        source_timestamp=10.0,
        source_stage="current_observation_anchor",
        target_payload_hash="target:0",
    )
    end = LegacyReferenceSourceFact(
        source_index=1,
        source_timestamp=10.1,
        source_stage="safe_keyframe",
        target_payload_hash="target:1",
    )
    return LegacyReferenceDerivation(
        legacy_worker_generation=3,
        modality="qpos",
        scheduled_for=10.02,
        emitted_at=10.021,
        reference=ResampledMotionReference(
            sequence=4,
            timestamp=10.025,
            left_source_index=0,
            right_source_index=1,
            left_source_timestamp=10.0,
            right_source_timestamp=10.1,
            interpolation_alpha=0.25,
            qpos_by_side={"right": np.asarray((0.0, 1.0, 2.0))},
        ),
        start_source_fact=start,
        end_source_fact=end,
        interpolation_contract_hash="linear-qpos:fixture",
    )


def test_position_projection_preserves_exact_value_without_claiming_activation() -> None:
    frame = project_position_pre_resolver_reference_frame(
        _derivation(),
        control_domain_id="robot-0",
        selection_group_id="arms",
        execution_fence_sequence=7,
        reference_revision=3,
        target_fact_id="position-work-item:1",
        target_fact_hash="position-work-item-hash:1",
        clock_domain_id="control.monotonic",
        side_to_component_id={"right": "right_arm"},
        component_schema_hash="right-arm-qpos-schema",
    )

    assert frame.reference_stage_id == POSITION_PRE_RESOLVER_REFERENCE_STAGE
    assert frame.temporal_semantics == "point_sample"
    assert frame.value_by_component["right_arm"] == pytest.approx(
        (0.0, 1.0, 2.0)
    )

    measurement = ExecutionMeasurementObservation(
        measurement_id="measurement:1",
        source_measurement_hash="measurement-source-hash:1",
        control_domain_id="robot-0",
        measured_at=10.03,
        available_at=10.031,
        clock_domain_id="control.monotonic",
        value_by_component={"right_arm": (0.0, 1.0, 2.0)},
        representation_by_component={"right_arm": "joint_position"},
        schema_hash_by_component={"right_arm": "right-arm-qpos-schema"},
    )
    association = associate_measurement_to_reference(
        measurement,
        frame,
        selection_group_id="arms",
        component_ids=("right_arm",),
    )
    assert association.status == "incomparable"
    assert association.reason == "reference_has_no_active_interval"
