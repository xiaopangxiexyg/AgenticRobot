from __future__ import annotations

from dataclasses import replace

import pytest

from agent_gateway.profiles.mujoco.depth_deployment import (
    MUJOCO_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1,
    MUJOCO_CAPTURE_DEPTH_ROUTE_ID_V2,
    MujocoDepthCaptureProfile,
    build_mujoco_depth_capture_registration,
    build_mujoco_depth_capture_registration_v2,
)
from agent_gateway.profiles.mujoco.motion_factory import (
    MujocoAgentArtifactStorageConfig,
)
from agent_gateway.skills.perception.definition import CAPTURE_DEPTH_SKILL_ID_V1


def _registration():
    return build_mujoco_depth_capture_registration(
        source_catalog=(
            {
                "source_id": "front",
                "role": "main",
                "native_width": 640,
                "native_height": 480,
            },
            {
                "source_id": "right_wrist",
                "role": "wrist_right",
                "native_width": 320,
                "native_height": 240,
            },
        ),
        runtime_session_id="session:mujoco-depth",
        control_domain_id="domain:mujoco",
        target_binding_id="target:mujoco",
        station_manifest_hash="station:hash",
        execution_manifest_hash="execution:hash",
    )


def _result(*, source_id="front", role="main", width=640, height=480):
    return {
        "source_id": source_id,
        "role": role,
        "captured_at": 1.25,
        "width": width,
        "height": height,
        "depth_unit": "meter",
        "invalid_depth_value": 0.0,
        "camera_pose_world_xyz_quat_xyzw": [0, 0, 1, 0, 0, 0, 1],
        "intrinsics": {
            "width": width,
            "height": height,
            "fx": 300.0,
            "fy": 300.0,
            "cx": width / 2,
            "cy": height / 2,
            "fovy_deg": 60.0,
        },
    }


def _registration_v2():
    return build_mujoco_depth_capture_registration_v2(
        source_catalog=(
            {
                "source_id": "front",
                "role": "main",
                "native_width": 640,
                "native_height": 480,
            },
        ),
        runtime_session_id="session:mujoco-depth-v2",
        control_domain_id="domain:mujoco",
        target_binding_id="target:mujoco",
        station_manifest_hash="station:hash",
        execution_manifest_hash="execution:hash",
        source_frame_id="mujoco_sim_hil/world",
        source_frame_spec_hash="frame:world:hash",
        reference_frame_id="mujoco_sim_hil/station",
        reference_frame_spec_hash="frame:station:hash",
        frame_graph_hash="frame:graph:hash",
    )


def _result_v2(**overrides):
    result = {
        key: value
        for key, value in _result().items()
        if key != "camera_pose_world_xyz_quat_xyzw"
    }
    result.update(
        {
            "station_revision": 7,
            "reference_frame_id": "mujoco_sim_hil/station",
            "reference_frame_spec_hash": "frame:station:hash",
            "frame_graph_hash": "frame:graph:hash",
            "camera_pose_reference_xyz_quat_xyzw": [
                0,
                0,
                1,
                0,
                0,
                0,
                1,
            ],
        }
    )
    result.update(overrides)
    return result


def test_depth_registration_is_strictly_read_only_and_artifact_bearing() -> None:
    bound = _registration()
    template = bound.registration
    route = template.routes[0]
    slot = template.io_contract.output_artifact_slots[0]

    assert template.definition.skill_id == CAPTURE_DEPTH_SKILL_ID_V1
    assert template.definition.skill_kind == "inspection"
    assert template.definition.controlled_component_ids == ()
    assert route.control_effect_class == "read_only"
    assert route.privilege_tier == "t0_read_only"
    assert not route.may_move_robot
    assert not route.may_open_publication
    assert not route.requires_operator_confirmation
    assert not route.requires_deadman
    assert route.cancellation_policy == "cancel_only"
    assert not route.requires_task_evaluation
    assert slot.slot_id == MUJOCO_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1
    assert slot.direction == "output"
    assert (slot.min_count, slot.max_count) == (1, 1)
    assert route.skill_io_contract_hash == template.io_contract.semantic_content_hash


def test_depth_v2_binds_reference_frame_identity_and_preserves_v1() -> None:
    legacy = _registration()
    assert legacy.registration.definition.skill_version == "1"

    bound = _registration_v2()
    template = bound.registration
    route = template.routes[0]
    assert template.definition.skill_id == CAPTURE_DEPTH_SKILL_ID_V1
    assert template.definition.skill_version == "2"
    assert route.route_id == MUJOCO_CAPTURE_DEPTH_ROUTE_ID_V2
    assert route.route_version == "2"
    assert bound.deployment_binding.source_frame_id == "mujoco_sim_hil/world"
    assert bound.deployment_binding.reference_frame_id == (
        "mujoco_sim_hil/station"
    )
    assert template.schema_entry.execution_result_schema["properties"][
        "frame_graph_hash"
    ] == {"const": "frame:graph:hash"}
    bound.payload_validator.validate_parameters({"role": "main"})
    bound.payload_validator.validate_execution_result(_result_v2())

    with pytest.raises(ValueError, match="reference-frame identity"):
        bound.payload_validator.validate_execution_result(
            _result_v2(frame_graph_hash="other:graph")
        )
    with pytest.raises(ValueError, match="station_revision"):
        bound.payload_validator.validate_execution_result(
            _result_v2(station_revision=True)
        )


def test_depth_registration_validates_semantic_role_and_contextual_result() -> None:
    bound = _registration()
    validator = bound.payload_validator
    validator.validate_parameters({"role": "main"})
    validator.validate_execution_result(_result())
    assert validator.validate_execution_result_for_parameters is not None
    validator.validate_execution_result_for_parameters(
        {"role": "main"},
        _result(),
    )

    with pytest.raises(ValueError, match="not registered"):
        validator.validate_parameters({"role": "missing"})
    with pytest.raises(ValueError, match="registered role"):
        validator.validate_execution_result(_result(source_id="other"))
    with pytest.raises(ValueError, match="dimensions"):
        validator.validate_execution_result(_result(width=320))
    with pytest.raises(ValueError, match="differs"):
        validator.validate_execution_result_for_parameters(
            {"role": "wrist_right"},
            _result(),
        )

    for field_name, invalid_value, message in (
        ("width", True, "positive integers"),
        ("invalid_depth_value", False, "invalid-value"),
        (
            "camera_pose_world_xyz_quat_xyzw",
            [0, 0, 1, 0, 0, 0, 2],
            "normalized",
        ),
    ):
        invalid = {**_result(), field_name: invalid_value}
        with pytest.raises(ValueError, match=message):
            validator.validate_execution_result(invalid)


def test_depth_profile_rejects_duplicates_and_oversized_native_frames() -> None:
    for invalid_limit in (True, 0, -1):
        with pytest.raises(ValueError, match="max_artifact_bytes"):
            MujocoDepthCaptureProfile(
                sources=(("front", "main", 2, 2),),
                max_artifact_bytes=invalid_limit,
            )
    with pytest.raises(ValueError, match="unique"):
        MujocoDepthCaptureProfile(
            sources=(("front", "main", 2, 2), ("front", "aux", 2, 2)),
        )
    with pytest.raises(ValueError, match="unique semantic roles"):
        MujocoDepthCaptureProfile(
            sources=(("front", "main", 2, 2), ("side", "main", 2, 2)),
        )
    with pytest.raises(ValueError, match="Artifact bound"):
        MujocoDepthCaptureProfile(
            sources=(("front", "main", 4096, 4096),),
        )


def test_depth_registration_rejects_malformed_source_catalog() -> None:
    with pytest.raises(ValueError, match="unexpected fields"):
        build_mujoco_depth_capture_registration(
            source_catalog=(({"source_id": "front", "native_width": 2}),),
            runtime_session_id="session:mujoco-depth",
            control_domain_id="domain:mujoco",
            target_binding_id="target:mujoco",
            station_manifest_hash="station:hash",
            execution_manifest_hash="execution:hash",
        )

    bound = _registration()
    with pytest.raises(ValueError, match="Read-only Routes"):
        replace(bound.registration.routes[0], may_move_robot=True)


def test_artifact_storage_partition_is_session_bound_and_limits_are_finite(
    tmp_path,
) -> None:
    config = MujocoAgentArtifactStorageConfig(
        root=tmp_path,
        quota_bytes=4096,
        max_artifact_bytes=2048,
        max_chunk_bytes=1024,
    )
    first = config.store_root_for("runtime:first")
    assert first == config.store_root_for("runtime:first")
    assert first != config.store_root_for("runtime:second")
    assert first.parent == tmp_path
    with pytest.raises(ValueError, match="runtime_session_id"):
        config.store_root_for(" runtime:first")

    with pytest.raises(ValueError, match="max_artifact_bytes"):
        MujocoAgentArtifactStorageConfig(
            root=tmp_path,
            quota_bytes=1024,
            max_artifact_bytes=2048,
        )
    with pytest.raises(ValueError, match="max_chunk_bytes"):
        MujocoAgentArtifactStorageConfig(
            root=tmp_path,
            quota_bytes=4096,
            max_artifact_bytes=2048,
            max_chunk_bytes=4096,
        )
