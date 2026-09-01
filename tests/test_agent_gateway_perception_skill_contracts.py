from __future__ import annotations

import pytest

from agent_gateway.skills.perception import (
    capture_depth_schema_entry_v2,
    capture_depth_skill_definition_v2,
    capture_rgb_schema_entry_v2,
    capture_rgb_skill_definition_v1,
    capture_rgb_skill_definition_v2,
    validate_capture_rgb_result_for_parameters_v2,
    validate_capture_rgb_result_v2,
)

pytestmark = [pytest.mark.unit]


def _rgb_v2_result() -> dict[str, object]:
    return {
        "snapshot_id": "rgb-snapshot:test",
        "station_revision": 7,
        "capture_window_started_at": 9.95,
        "capture_window_ended_at": 9.97,
        "frames": [
            {
                "ordinal": 0,
                "source_id": "left_wrist",
                "role": "wrist_left",
                "acquired_at": 9.97,
                "media_type": "image/jpeg",
                "width": 32,
                "height": 24,
                "media_content_hash": "a" * 64,
            },
            {
                "ordinal": 1,
                "source_id": "front",
                "role": "main",
                "acquired_at": 9.95,
                "media_type": "image/jpeg",
                "width": 32,
                "height": 24,
                "media_content_hash": "b" * 64,
            },
        ],
        "total_bytes": 1024,
    }


def test_perception_v2_schemas_are_public_and_v1_rgb_remains_available() -> None:
    rgb_v1 = capture_rgb_skill_definition_v1()
    rgb_v2 = capture_rgb_skill_definition_v2()
    depth_v2 = capture_depth_skill_definition_v2()
    rgb_schema = capture_rgb_schema_entry_v2(
        skill_definition_hash=rgb_v2.semantic_content_hash,
        parameter_profile_id="rgb:test:v2",
        parameter_profile_hash="a" * 64,
        allowed_roles=("main", "wrist_left"),
        max_total_bytes=4096,
    )
    depth_schema = capture_depth_schema_entry_v2(
        skill_definition_hash=depth_v2.semantic_content_hash,
        parameter_profile_id="depth:test:v2",
        parameter_profile_hash="b" * 64,
        allowed_roles=("main",),
        reference_frame_id="canonical_station",
        reference_frame_spec_hash="c" * 64,
        frame_graph_hash="d" * 64,
    )

    assert rgb_v1.skill_version == "1"
    assert rgb_v2.skill_version == "2"
    assert rgb_schema.skill_version == "2"
    assert "capture_window_started_at" in rgb_schema.execution_result_schema["required"]
    assert (
        "acquired_at"
        in rgb_schema.execution_result_schema["properties"]["frames"]["items"][
            "required"
        ]
    )
    assert depth_schema.skill_version == "2"
    assert "frame_graph_hash" in depth_schema.execution_result_schema["required"]


def test_rgb_v2_preserves_requested_order_and_bounds_each_frame_to_window() -> None:
    parameters = {
        "selected_roles": ["wrist_left", "main"],
        "width": 32,
        "height": 24,
        "max_total_bytes": 2048,
    }
    result = _rgb_v2_result()

    validate_capture_rgb_result_v2(
        result,
        source_id_by_role={"main": "front", "wrist_left": "left_wrist"},
        max_total_bytes=4096,
    )
    validate_capture_rgb_result_for_parameters_v2(parameters, result)

    outside = _rgb_v2_result()
    outside["frames"][0]["acquired_at"] = 9.98
    with pytest.raises(ValueError, match="outside the capture window"):
        validate_capture_rgb_result_v2(
            outside,
            source_id_by_role={"main": "front", "wrist_left": "left_wrist"},
            max_total_bytes=4096,
        )

    reordered = _rgb_v2_result()
    reordered["frames"] = list(reversed(reordered["frames"]))
    with pytest.raises(ValueError, match="roles differ from the request"):
        validate_capture_rgb_result_for_parameters_v2(parameters, reordered)
