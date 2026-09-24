from __future__ import annotations

import pytest

from agent_gateway.profiles.back_projection import (
    build_back_projection_registration,
)
from agent_gateway.profiles.placement import build_placement_registration
from agent_gateway.profiles.segmentation import build_segmentation_registration

pytestmark = [pytest.mark.unit]


def _registration(target_kind: str = "sim"):
    current_mode = "real_active" if target_kind == "real" else "sim_active"
    identity = {
        "runtime_session_id": "session:test",
        "control_domain_id": "domain:test",
        "target_binding_id": "target:test",
        "station_manifest_hash": "station:test",
        "execution_manifest_hash": "execution:test",
    }
    segmentation = build_segmentation_registration(
        target_kind=target_kind,
        current_mode=current_mode,
        source_id_by_role={"main": "front"},
        model_profile_id="sam2:test",
        model_profile_hash="sam2-model:test",
        inference_implementation_hash="sam2-runtime:test",
        **identity,
    )
    back_projection = build_back_projection_registration(
        target_kind=target_kind,
        current_mode=current_mode,
        source_catalog=(
            {
                "source_id": "front",
                "role": "main",
                "native_width": 640,
                "native_height": 480,
            },
        ),
        reference_frame_id="canonical_station",
        reference_frame_spec_hash="frame:station:test",
        frame_graph_hash="frame-graph:test",
        **identity,
    )
    return build_placement_registration(
        segmentation=segmentation,
        back_projection=back_projection,
        component_id="right_arm",
        destination_role="main",
        **identity,
    )


def _planned_result() -> dict[str, object]:
    return {
        "planning_status": "planned",
        "reason_code": "placement_geometry_planned",
        "held_segment_operation_id": "operation:held",
        "destination_segment_operation_id": "operation:destination",
        "station_revision": 3,
        "component_id": "right_arm",
        "reference_frame_id": "canonical_station",
        "reference_frame_spec_hash": "frame:station:test",
        "frame_graph_hash": "frame-graph:test",
        "target_eef_xy": [0.4, -0.1],
        "minimum_lateral_eef_z": 0.3,
        "fits_with_clearance": True,
        "clearance_m": 0.01,
        "held_gripper_close_fraction": None,
        "held_gripper_state_checked": False,
        "destination_geometry_provisional": False,
        "geometry_eligible_for_motion_planning": True,
        "execution_requires_explicit_motion_skill": True,
        "motion_or_hold_performed": False,
    }


def test_plan_placement_schema_is_simple_read_only_and_cross_validated() -> None:
    registration = _registration()
    route = registration.skill.routes[0]
    schema = registration.skill.schema_entry.parameter_schema
    assert "oneOf" not in repr(schema)
    assert "anyOf" not in repr(schema)
    assert route.control_effect_class == "read_only"
    assert not route.may_move_robot
    assert route.max_operation_duration_ms == 125_000
    parameters = {
        "held_segment_operation_id": "operation:held",
        "destination_segment_operation_id": "operation:destination",
        "clearance_m": 0.01,
    }
    registration.payload_validator.validate_parameters(parameters)
    with pytest.raises(ValueError, match="must differ"):
        registration.payload_validator.validate_parameters(
            {
                **parameters,
                "destination_segment_operation_id": "operation:held",
            }
        )

    result = _planned_result()
    registration.payload_validator.validate_execution_result(result)
    registration.payload_validator.validate_execution_result_for_parameters(
        parameters,
        result,
    )
    with pytest.raises(ValueError, match="eligibility contradicts"):
        registration.payload_validator.validate_execution_result(
            {**result, "fits_with_clearance": False}
        )


def test_real_placement_result_fails_closed_without_gripper_measurement() -> None:
    registration = _registration()
    validator = registration.payload_validator.validate_execution_result
    real_registration = _registration("real")
    result = _planned_result()
    validator(result)
    with pytest.raises(ValueError, match="measured closed gripper"):
        real_registration.payload_validator.validate_execution_result(result)
