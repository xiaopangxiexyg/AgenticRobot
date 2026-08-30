from __future__ import annotations

import json
import math

import numpy as np
import pytest

from agent_gateway.profiles.real.grasp_pose_adapter import (
    CAMERA_RELATIVE_GRASP_PROPOSAL_SCHEMA_V1,
    CameraRelativeGraspProposal,
    load_camera_relative_grasp_proposal,
    resolve_camera_relative_grasp_target,
)
from agent_gateway.profiles.real.perception_deployment import (
    REAL_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1,
    REAL_CAPTURE_DEPTH_ROUTE_ID_V2,
)
from agent_gateway.protocol.artifact_contracts import (
    AgentArtifactBinding,
    AgentArtifactRef,
)
from agent_gateway.skills.perception.definition import CAPTURE_DEPTH_SKILL_ID_V1
from agent_gateway.skills.perception.schemas import (
    DEPTH_FRAME_COMPRESSION_V1,
    DEPTH_FRAME_ENCODING_V1,
    DEPTH_FRAME_MEDIA_TYPE_V1,
    DEPTH_FRAME_SCHEMA_ID_V1,
)


def _depth_binding() -> AgentArtifactBinding:
    return AgentArtifactBinding(
        slot_id=REAL_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1,
        ordinal=0,
        artifact=AgentArtifactRef(
            media_type=DEPTH_FRAME_MEDIA_TYPE_V1,
            schema_id=DEPTH_FRAME_SCHEMA_ID_V1,
            encoding=DEPTH_FRAME_ENCODING_V1,
            compression=DEPTH_FRAME_COMPRESSION_V1,
            byte_length=4096,
            sha256="a" * 64,
        ),
        production_receipt_ref="real-depth-capture:" + "b" * 64,
    )


def _payload() -> dict[str, object]:
    half = math.sqrt(0.5)
    return {
        "schema": CAMERA_RELATIVE_GRASP_PROPOSAL_SCHEMA_V1,
        "runtime_session_id": "real-session:test",
        "skill_route_manifest_hash": "route-manifest:test",
        "source_operation_id": "depth-operation:test",
        "source_operation_hash": "depth-operation-hash:test",
        "perception_route_id": REAL_CAPTURE_DEPTH_ROUTE_ID_V2,
        "perception_route_hash": "depth-route:test",
        "perception_deployment_binding_hash": "depth-binding:test",
        "component_id": "right_arm",
        "station_revision": 8,
        "reference_frame_id": "canonical_station",
        "reference_frame_spec_hash": "station-frame:test",
        "frame_graph_hash": "frame-graph:test",
        "camera_pose_reference_xyz_quat_xyzw": [
            1.0,
            2.0,
            3.0,
            0.0,
            0.0,
            half,
            half,
        ],
        "tool_tcp_pose_camera_xyz_quat_xyzw": [
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0,
        ],
        "depth_artifact_binding": _depth_binding().to_wire(),
        "capture_group_id": "c" * 64,
    }


def _manifest() -> dict[str, object]:
    return {
        "runtime_session_id": "real-session:test",
        "route_manifest_hash": "route-manifest:test",
        "skill_routes": [
            {
                "route": {
                    "skill_id": CAPTURE_DEPTH_SKILL_ID_V1,
                    "skill_version": "2",
                    "route_id": REAL_CAPTURE_DEPTH_ROUTE_ID_V2,
                    "route_hash": "depth-route:test",
                },
                "deployment_binding": {
                    "deployment_binding_hash": "depth-binding:test",
                },
            }
        ],
    }


def _snapshot() -> dict[str, object]:
    return {
        "station_revision": 8,
        "frame_graph_hash": "frame-graph:test",
        "measurement_summary": {
            "reference_frame_id": "canonical_station",
            "reference_frame_spec_hash": "station-frame:test",
        },
    }


def _completed_depth_operation() -> dict[str, object]:
    return {
        "operation_id": "depth-operation:test",
        "operation_hash": "depth-operation-hash:test",
        "runtime_session_id": "real-session:test",
        "status": "completed",
        "terminal": True,
        "execution_outcome": "completed",
        "station_revision_at_admission": 8,
        "execution_result": {
            "station_revision": 8,
            "reference_frame_id": "canonical_station",
            "reference_frame_spec_hash": "station-frame:test",
            "frame_graph_hash": "frame-graph:test",
            "camera_pose_reference_xyz_quat_xyzw": [
                1.0,
                2.0,
                3.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        },
        "output_artifact_bindings": [_depth_binding().to_wire()],
    }


def test_camera_relative_grasp_composes_station_camera_and_camera_tcp() -> None:
    proposal = CameraRelativeGraspProposal.from_mapping(_payload())

    resolved = resolve_camera_relative_grasp_target(
        proposal=proposal,
        route_manifest=_manifest(),
        motion_snapshot=_snapshot(),
        expected_component_id="right_arm",
    )

    np.testing.assert_allclose(
        resolved.target_pose_reference_xyz_quat_xyzw[:3],
        [1.0, 2.1, 3.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.abs(resolved.target_pose_reference_xyz_quat_xyzw[3:]),
        [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
        atol=1e-6,
    )
    assert resolved.component_id == "right_arm"
    assert resolved.station_revision == 8
    assert resolved.capture_group_id == "c" * 64
    assert len(resolved.semantic_content_hash) == 64


def test_proposal_builds_directly_from_completed_depth_operation() -> None:
    proposal = CameraRelativeGraspProposal.from_depth_operation_wire(
        route_manifest=_manifest(),
        depth_operation=_completed_depth_operation(),
        component_id="right_arm",
        tool_tcp_pose_camera_xyz_quat_xyzw=[
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    )

    assert CameraRelativeGraspProposal.from_mapping(
        proposal.to_mapping()
    ) == proposal
    assert proposal.camera_pose_reference_xyz_quat_xyzw[:3] == (1.0, 2.0, 3.0)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ({"terminal": False}, "terminal and completed"),
        ({"runtime_session_id": "other"}, "another Gateway session"),
        ({"station_revision_at_admission": 9}, "differs from admission"),
    ),
)
def test_proposal_builder_rejects_non_authoritative_depth_operation(
    mutation: dict[str, object],
    match: str,
) -> None:
    operation = {**_completed_depth_operation(), **mutation}

    with pytest.raises((ValueError, RuntimeError), match=match):
        CameraRelativeGraspProposal.from_depth_operation_wire(
            route_manifest=_manifest(),
            depth_operation=operation,
            component_id="right_arm",
            tool_tcp_pose_camera_xyz_quat_xyzw=[0, 0, 0, 0, 0, 0, 1],
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ({"station_revision": 9}, "revision is stale"),
        ({"frame_graph_hash": "changed"}, "FrameGraph changed"),
        (
            {
                "measurement_summary": {
                    "reference_frame_id": "canonical_station",
                    "reference_frame_spec_hash": "changed",
                }
            },
            "frame identity changed",
        ),
    ),
)
def test_camera_grasp_rejects_stale_motion_identity(
    mutation: dict[str, object],
    match: str,
) -> None:
    snapshot = {**_snapshot(), **mutation}

    with pytest.raises(RuntimeError, match=match):
        resolve_camera_relative_grasp_target(
            proposal=CameraRelativeGraspProposal.from_mapping(_payload()),
            route_manifest=_manifest(),
            motion_snapshot=snapshot,
            expected_component_id="right_arm",
        )


def test_camera_grasp_rejects_wrong_arm_and_changed_perception_binding() -> None:
    proposal = CameraRelativeGraspProposal.from_mapping(_payload())
    with pytest.raises(ValueError, match="different arm"):
        resolve_camera_relative_grasp_target(
            proposal=proposal,
            route_manifest=_manifest(),
            motion_snapshot=_snapshot(),
            expected_component_id="left_arm",
        )

    manifest = _manifest()
    manifest["skill_routes"][0]["deployment_binding"][
        "deployment_binding_hash"
    ] = "changed"
    with pytest.raises(RuntimeError, match="perception deployment changed"):
        resolve_camera_relative_grasp_target(
            proposal=proposal,
            route_manifest=manifest,
            motion_snapshot=_snapshot(),
            expected_component_id="right_arm",
        )


def test_camera_grasp_proposal_requires_typed_depth_artifact_and_receipt() -> None:
    payload = _payload()
    payload["depth_artifact_binding"] = {
        **payload["depth_artifact_binding"],
        "production_receipt_ref": None,
    }
    with pytest.raises(ValueError, match="artifact_binding_hash"):
        CameraRelativeGraspProposal.from_mapping(payload)

    binding_without_receipt = AgentArtifactBinding(
        slot_id=REAL_CAPTURE_DEPTH_OUTPUT_SLOT_ID_V1,
        ordinal=0,
        artifact=_depth_binding().artifact,
        production_receipt_ref=None,
    )
    payload = _payload()
    payload["depth_artifact_binding"] = binding_without_receipt.to_wire()
    with pytest.raises(ValueError, match="capture receipt"):
        CameraRelativeGraspProposal.from_mapping(payload)


def test_camera_grasp_proposal_loader_is_bounded_and_exact(tmp_path) -> None:
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    proposal = load_camera_relative_grasp_proposal(path)

    assert proposal.component_id == "right_arm"
    payload = _payload()
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_camera_relative_grasp_proposal(path)
