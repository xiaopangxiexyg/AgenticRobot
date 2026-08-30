from __future__ import annotations

import json
from argparse import Namespace
from typing import ClassVar

import pytest

from agent_gateway.profiles.real import one_shot_client
from agent_gateway.profiles.real.grasp_pose_adapter import (
    CAMERA_RELATIVE_GRASP_PROPOSAL_SCHEMA_V1,
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


class _FakeClient:
    instances: ClassVar[list[_FakeClient]] = []

    def __init__(self, *, socket_path: str, timeout_sec: float) -> None:
        self.socket_path = socket_path
        self.timeout_sec = timeout_sec
        self.calls = []
        self.closed = False
        self.__class__.instances.append(self)

    def close(self) -> None:
        self.closed = True

    def call(self, method, payload):
        self.calls.append((method, payload))
        if method == "get_skill_route_manifest":
            return {
                "runtime_session_id": "real-session:test",
                "route_manifest_hash": "route-manifest:test",
                "skill_routes": [
                    {
                        "route": {
                            "skill_id": "motion.step_tcp_forward",
                            "skill_version": "1",
                            "skill_definition_hash": "skill-definition:test",
                            "route_id": "real.rm65.agent_skill_step_tcp_forward",
                            "route_version": "1",
                            "route_hash": "route:test",
                            "parameter_schema_hash": "parameter-schema-hash:test",
                            "allowed_parameter_profiles": [
                                {
                                    "parameter_profile_id": "profile:test",
                                    "parameter_profile_hash": "profile-hash:test",
                                    "parameter_schema_hash": "parameter-schema-hash:test",
                                }
                            ],
                        },
                        "deployment_binding": {
                            "deployment_binding_hash": "deployment-binding:test",
                            "safety_envelope_refs": [
                                {
                                    "object_id": "envelope:test",
                                    "content_hash": "envelope-hash:test",
                                    "constraint_set_hash": "constraint-hash:test",
                                    "allowed_route_ids": [
                                        "real.rm65.agent_skill_step_tcp_forward"
                                    ],
                                }
                            ],
                        },
                    }
                ],
                "skill_schema_catalog_hash": "schema-catalog:test",
            }
        if method == "get_agent_skill_schema_catalog":
            return {
                "skill_schema_catalog_hash": "schema-catalog:test",
                "schema_entries": [
                    {
                        "skill_id": "motion.step_tcp_forward",
                        "skill_version": "1",
                        "parameter_schema_hash": "parameter-schema-hash:test",
                        "parameter_schema": {
                            "properties": {
                                "component_id": {"enum": ["left_arm"]},
                                "direction": {
                                    "enum": ["forward"]
                                },
                                "reference_frame_id": {
                                    "enum": ["canonical_station"]
                                },
                                "reference_frame_spec_hash": {
                                    "const": "station-frame:test"
                                },
                                "controlled_frame_id": {"const": "tool_tcp"},
                                "frame_graph_hash": {"const": "frame-graph:test"},
                                "tool_config_hash": {"const": "tool-config:test"},
                                "calibration_hash": {"const": "calibration:test"},
                                "primitive_catalog_hash": {
                                    "const": "primitive-catalog:test"
                                },
                            }
                        },
                        "parameter_profiles": [
                            {
                                "parameter_profile_id": "profile:test",
                                "parameter_profile_hash": "profile-hash:test",
                                "parameter_schema_hash": "parameter-schema-hash:test",
                            }
                        ],
                    }
                ],
            }
        if method == "get_observation_snapshot":
            return {
                "snapshot_id": "snapshot:test",
                "station_revision": 4,
                "lifecycle_state": "running",
                "execution_gate": "ready",
                "frame_graph_hash": "frame-graph:test",
                "tool_config_hash": "tool-config:test",
                "calibration_hash": "calibration:test",
                "measurement_summary": {
                    "pose_xyz_quat_xyzw_by_component": {"left_arm": [0] * 7},
                },
            }
        if method == "submit_skill_request":
            assert payload["skill_execution_grant_id"] is None
            assert payload["parameters"]["direction"] == "forward"
            return {
                "accepted": True,
                "reason_code": "request_admitted",
                "operation": {
                    "operation_id": "operation:test",
                    "status": "running",
                },
            }
        if method == "get_own_operation":
            return {
                "accepted": True,
                "reason_code": "own_operation_found",
                "operation": {
                    "operation_id": "operation:test",
                    "status": "completed",
                },
            }
        raise AssertionError(method)


class _FakeAbsolutePoseClient(_FakeClient):
    expected_pose: ClassVar[list[float]] = [
        0.31,
        -0.12,
        0.28,
        0.0,
        0.0,
        0.0,
        1.0,
    ]

    def call(self, method, payload):
        self.calls.append((method, payload))
        if method == "get_skill_route_manifest":
            return {
                "runtime_session_id": "real-session:absolute-pose-test",
                "route_manifest_hash": "route-manifest:absolute-pose-test",
                "skill_routes": [
                    {
                        "route": {
                            "skill_id": "motion.move_tcp_to_pose",
                            "skill_version": "1",
                            "skill_definition_hash": "skill-definition:absolute-pose",
                            "route_id": "real.rm65.agent_skill_tcp_pose_target",
                            "route_version": "1",
                            "route_hash": "route:absolute-pose",
                            "parameter_schema_hash": "schema:absolute-pose",
                            "allowed_parameter_profiles": [
                                {
                                    "parameter_profile_id": (
                                        "real-tool-tcp-absolute-grasp-pose-v1"
                                    ),
                                    "parameter_profile_hash": "profile:absolute-pose",
                                    "parameter_schema_hash": "schema:absolute-pose",
                                }
                            ],
                        },
                        "deployment_binding": {
                            "deployment_binding_hash": "binding:absolute-pose",
                            "safety_envelope_refs": [
                                {
                                    "object_id": "envelope:absolute-pose",
                                    "content_hash": "envelope-hash:absolute-pose",
                                    "constraint_set_hash": (
                                        "constraint-hash:absolute-pose"
                                    ),
                                    "allowed_route_ids": [
                                        "real.rm65.agent_skill_tcp_pose_target"
                                    ],
                                }
                            ],
                        },
                    },
                    {
                        "route": {
                            "skill_id": CAPTURE_DEPTH_SKILL_ID_V1,
                            "skill_version": "2",
                            "route_id": REAL_CAPTURE_DEPTH_ROUTE_ID_V2,
                            "route_hash": "depth-route:absolute-pose",
                        },
                        "deployment_binding": {
                            "deployment_binding_hash": (
                                "depth-binding:absolute-pose"
                            ),
                        },
                    },
                ],
                "skill_schema_catalog_hash": "catalog:absolute-pose",
            }
        if method == "get_agent_skill_schema_catalog":
            return {
                "skill_schema_catalog_hash": "catalog:absolute-pose",
                "schema_entries": [
                    {
                        "skill_id": "motion.move_tcp_to_pose",
                        "skill_version": "1",
                        "parameter_schema_hash": "schema:absolute-pose",
                        "parameter_schema": {
                            "properties": {
                                "pose_targets_by_component": {
                                    "properties": {
                                        "right_arm": {
                                            "type": "array",
                                            "minItems": 7,
                                            "maxItems": 7,
                                        }
                                    }
                                },
                                "reference_frame_id": {
                                    "enum": ["canonical_station"]
                                },
                                "reference_frame_spec_hash": {
                                    "type": "string"
                                },
                                "controlled_frame_id": {"const": "tool_tcp"},
                                "frame_graph_hash": {"type": "string"},
                                "tool_config_hash": {"type": "string"},
                                "calibration_hash": {"type": "string"},
                                "speed_scale": {
                                    "type": "number",
                                    "minimum": 0.05,
                                    "maximum": 1.0,
                                },
                            }
                        },
                        "parameter_profiles": [
                            {
                                "parameter_profile_id": (
                                    "real-tool-tcp-absolute-grasp-pose-v1"
                                ),
                                "parameter_profile_hash": "profile:absolute-pose",
                                "parameter_schema_hash": "schema:absolute-pose",
                            }
                        ],
                    }
                ],
            }
        if method == "get_observation_snapshot":
            return {
                "snapshot_id": "snapshot:absolute-pose",
                "station_revision": 8,
                "lifecycle_state": "running",
                "execution_gate": "ready",
                "frame_graph_hash": "frame-graph:absolute-pose",
                "tool_config_hash": "tool-config:absolute-pose",
                "calibration_hash": "calibration:absolute-pose",
                "measurement_summary": {
                    "pose_xyz_quat_xyzw_by_component": {
                        "right_arm": [0.0] * 7
                    },
                    "reference_frame_id": "canonical_station",
                    "reference_frame_spec_hash": "station-frame:absolute-pose",
                },
            }
        if method == "submit_skill_request":
            parameters = payload["parameters"]
            assert payload["requested_timeout_ms"] == 30_000
            assert "component_id" not in parameters
            assert parameters["reference_frame_id"] == "canonical_station"
            assert parameters["reference_frame_spec_hash"] == (
                "station-frame:absolute-pose"
            )
            assert set(parameters["pose_targets_by_component"]) == {"right_arm"}
            assert parameters["pose_targets_by_component"]["right_arm"] == (
                pytest.approx(self.expected_pose, abs=1e-6)
            )
            assert parameters["speed_scale"] == 0.2
            return {
                "accepted": True,
                "reason_code": "request_admitted",
                "operation": {
                    "operation_id": "operation:absolute-pose",
                    "status": "running",
                },
            }
        if method == "get_own_operation":
            return {
                "accepted": True,
                "reason_code": "own_operation_found",
                "operation": {
                    "operation_id": "operation:absolute-pose",
                    "status": "completed",
                },
            }
        raise AssertionError(method)


class _FakeCameraGraspClient(_FakeAbsolutePoseClient):
    expected_pose: ClassVar[list[float]] = [0.3, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]


def test_one_shot_client_unwraps_owned_operation_outcome(monkeypatch) -> None:
    _FakeClient.instances.clear()
    monkeypatch.setattr(one_shot_client, "_Client", _FakeClient)
    args = Namespace(
        confirm="EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE",
        socket_path="/tmp/primitive.sock",
        side="left",
        action="forward",
        client_id="dexteleop-local-primitive-operator",
        timeout_sec=1.0,
        operation_timeout_sec=None,
    )

    operation = one_shot_client.execute_one_shot(args)

    assert operation["status"] == "completed"
    client = _FakeClient.instances[0]
    assert client.closed is True
    assert [method for method, _ in client.calls] == [
        "get_skill_route_manifest",
        "get_agent_skill_schema_catalog",
        "get_observation_snapshot",
        "submit_skill_request",
        "get_own_operation",
    ]
    request = next(
        payload
        for method, payload in client.calls
        if method == "submit_skill_request"
    )
    assert request["parameters"]["reference_frame_id"] == "canonical_station"
    assert request["parameters"]["reference_frame_spec_hash"] == (
        "station-frame:test"
    )


def test_one_shot_client_submits_absolute_dmtac_tcp_in_canonical_station(
    monkeypatch,
) -> None:
    _FakeClient.instances.clear()
    monkeypatch.setattr(one_shot_client, "_Client", _FakeAbsolutePoseClient)
    args = Namespace(
        confirm="EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE",
        socket_path="/tmp/absolute-pose.sock",
        side="right",
        action="move_tcp_to_pose",
        target_xyz_m=(0.31, -0.12, 0.28),
        target_quat_xyzw=(0.0, 0.0, 0.0, 2.0),
        speed_scale=0.2,
        client_id="dexteleop-local-primitive-operator",
        timeout_sec=1.0,
        operation_timeout_sec=None,
    )

    operation = one_shot_client.execute_one_shot(args)

    assert operation["status"] == "completed"
    assert _FakeClient.instances[0].closed is True


def test_absolute_pose_client_requires_a_complete_pose_before_connect(
    monkeypatch,
) -> None:
    _FakeClient.instances.clear()
    monkeypatch.setattr(one_shot_client, "_Client", _FakeAbsolutePoseClient)
    args = Namespace(
        confirm="EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE",
        socket_path="/tmp/absolute-pose.sock",
        side="right",
        action="move_tcp_to_pose",
        target_xyz_m=(0.31, -0.12, 0.28),
        target_quat_xyzw=None,
        speed_scale=0.2,
        client_id="dexteleop-local-primitive-operator",
        timeout_sec=1.0,
        operation_timeout_sec=1.0,
    )

    try:
        one_shot_client.execute_one_shot(args)
    except ValueError as error:
        assert "requires both" in str(error)
    else:
        raise AssertionError("incomplete absolute TCP pose must be rejected")

    assert _FakeClient.instances == []


def test_one_shot_client_resolves_camera_grasp_before_motion_submit(
    monkeypatch,
    tmp_path,
) -> None:
    _FakeClient.instances.clear()
    monkeypatch.setattr(one_shot_client, "_Client", _FakeCameraGraspClient)
    artifact_binding = AgentArtifactBinding(
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
    proposal_path = tmp_path / "camera-grasp.json"
    proposal_path.write_text(
        json.dumps(
            {
                "schema": CAMERA_RELATIVE_GRASP_PROPOSAL_SCHEMA_V1,
                "runtime_session_id": "real-session:absolute-pose-test",
                "skill_route_manifest_hash": (
                    "route-manifest:absolute-pose-test"
                ),
                "source_operation_id": "depth-operation:test",
                "source_operation_hash": "depth-operation-hash:test",
                "perception_route_id": REAL_CAPTURE_DEPTH_ROUTE_ID_V2,
                "perception_route_hash": "depth-route:absolute-pose",
                "perception_deployment_binding_hash": (
                    "depth-binding:absolute-pose"
                ),
                "component_id": "right_arm",
                "station_revision": 8,
                "reference_frame_id": "canonical_station",
                "reference_frame_spec_hash": "station-frame:absolute-pose",
                "frame_graph_hash": "frame-graph:absolute-pose",
                "camera_pose_reference_xyz_quat_xyzw": [
                    0.1,
                    0.2,
                    0.3,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "tool_tcp_pose_camera_xyz_quat_xyzw": [
                    0.2,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "depth_artifact_binding": artifact_binding.to_wire(),
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(
        confirm="EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE",
        socket_path="/tmp/absolute-pose.sock",
        side="right",
        action="move_tcp_to_pose",
        target_xyz_m=None,
        target_quat_xyzw=None,
        camera_grasp_proposal=str(proposal_path),
        speed_scale=0.2,
        client_id="dexteleop-local-primitive-operator",
        timeout_sec=1.0,
        operation_timeout_sec=None,
    )

    operation = one_shot_client.execute_one_shot(args)

    assert operation["status"] == "completed"
    assert _FakeClient.instances[0].closed is True


def test_camera_grasp_is_mutually_exclusive_with_direct_target(tmp_path) -> None:
    args = Namespace(
        action="move_tcp_to_pose",
        target_xyz_m=(0.1, 0.2, 0.3),
        target_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        camera_grasp_proposal=str(tmp_path / "unused.json"),
        speed_scale=0.2,
    )

    try:
        one_shot_client._absolute_tcp_input_from_args(args)
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("camera and direct target modes must be exclusive")
