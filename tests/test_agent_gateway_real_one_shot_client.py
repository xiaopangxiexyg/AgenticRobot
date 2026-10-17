from __future__ import annotations

import fcntl
import json
import subprocess
import sys
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
    skill_id: ClassVar[str] = "motion.move_tcp_to_pose"
    route_id: ClassVar[str] = "real.rm65.agent_skill_tcp_pose_target"
    requested_timeout_ms: ClassVar[int] = 30_000
    expected_pose: ClassVar[list[float]] = [
        0.31,
        -0.12,
        0.28,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    measured_pose: ClassVar[list[float]] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    def call(self, method, payload):
        self.calls.append((method, payload))
        if method == "get_skill_route_manifest":
            return {
                "runtime_session_id": "real-session:absolute-pose-test",
                "route_manifest_hash": "route-manifest:absolute-pose-test",
                "skill_routes": [
                    {
                        "route": {
                            "skill_id": self.skill_id,
                            "skill_version": "1",
                            "skill_definition_hash": "skill-definition:absolute-pose",
                            "route_id": self.route_id,
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
                                        self.route_id
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
                        "skill_id": self.skill_id,
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
                        "right_arm": list(self.measured_pose)
                    },
                    "reference_frame_id": "canonical_station",
                    "reference_frame_spec_hash": "station-frame:absolute-pose",
                },
            }
        if method == "submit_skill_request":
            parameters = payload["parameters"]
            assert payload["skill_id"] == self.skill_id
            assert payload["route_id"] == self.route_id
            assert payload["requested_timeout_ms"] == self.requested_timeout_ms
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


class _FakeRelativePoseClient(_FakeAbsolutePoseClient):
    measured_pose: ClassVar[list[float]] = [
        0.25,
        -0.10,
        0.30,
        0.0,
        0.0,
        0.0,
        2.0,
    ]
    expected_pose: ClassVar[list[float]] = [
        0.25,
        -0.02,
        0.30,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def test_one_shot_client_unwraps_owned_operation_outcome(monkeypatch, tmp_path) -> None:
    _FakeClient.instances.clear()
    monkeypatch.setattr(one_shot_client, "_Client", _FakeClient)
    args = Namespace(
        confirm="EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE",
        socket_path=str(tmp_path / "primitive.sock"),
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


@pytest.mark.parametrize("statuses", (
    ("completed",), ("failed",), ("cancelled",), ("timed_out",),
    ("running", "cancelling", "completed"),
))
def test_one_shot_client_queries_before_pacing_and_waits_for_terminal_owner(
    monkeypatch, tmp_path, statuses,
):
    events = []
    pending = iter(statuses)

    class Client(_FakeClient):
        def call(self, method, payload):
            response = super().call(method, payload)
            if method == "get_own_operation":
                status = next(pending)
                events.append(("query", status))
                response["operation"]["status"] = status
            return response

    monkeypatch.setattr(one_shot_client, "_Client", Client)
    monkeypatch.setattr(one_shot_client.time, "sleep", lambda delay: events.append(("sleep", delay)))
    args = Namespace(
        confirm="EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE",
        socket_path=str(tmp_path / "poll.sock"), side="left", action="forward",
        client_id="operator", timeout_sec=1.0, operation_timeout_sec=1.0,
    )
    operation = one_shot_client.execute_one_shot(args)
    expected = []
    for index, status in enumerate(statuses):
        if index:
            expected.append(("sleep", .05))
        expected.append(("query", status))
    assert events == expected
    assert operation["status"] == statuses[-1]
    client = Client.instances[-1]
    assert client.closed
    assert sum(method == "submit_skill_request" for method, _ in client.calls) == 1
    for method, payload in client.calls:
        if method == "get_own_operation":
            assert payload["operation_id"] == "operation:test"
            assert payload["runtime_session_id"] == "real-session:test"


def test_one_shot_client_pending_poll_does_not_sleep_or_query_past_deadline(
    monkeypatch, tmp_path,
):
    clock = [0.0]
    sleeps = []

    class Client(_FakeClient):
        def call(self, method, payload):
            response = super().call(method, payload)
            if method == "get_own_operation":
                response["operation"]["status"] = "running"
                clock[0] = 3.98  # One-second operation budget plus HOLD grace.
            return response

    def sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(one_shot_client, "_Client", Client)
    monkeypatch.setattr(one_shot_client.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(one_shot_client.time, "sleep", sleep)
    args = Namespace(
        confirm="EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE",
        socket_path=str(tmp_path / "timeout.sock"), side="left", action="forward",
        client_id="operator", timeout_sec=1.0, operation_timeout_sec=1.0,
    )
    with pytest.raises(RuntimeError, match="terminal operation/HOLD"):
        one_shot_client.execute_one_shot(args)
    assert sleeps == pytest.approx([.02])
    client = Client.instances[-1]
    assert client.closed
    assert sum(method == "get_own_operation" for method, _ in client.calls) == 1


@pytest.mark.parametrize("action", ["move_tcp_to_pose", "move_tcp_to_pose_safe"])
def test_one_shot_client_submits_absolute_dmtac_tcp_in_canonical_station(
    monkeypatch,
    tmp_path,
    action,
) -> None:
    class Client(_FakeAbsolutePoseClient):
        skill_id = "motion." + action
        route_id = "real.rm65.agent_skill_tcp_pose_target" + (
            "_safe" if action == "move_tcp_to_pose_safe" else ""
        )
        requested_timeout_ms = 120_000 if action == "move_tcp_to_pose_safe" else 30_000

    _FakeClient.instances.clear()
    monkeypatch.setattr(one_shot_client, "_Client", Client)
    args = Namespace(
        confirm="EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE",
        socket_path=str(tmp_path / "absolute-pose.sock"),
        side="right",
        action=action,
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


def test_one_shot_client_resolves_relative_station_delta_from_same_snapshot(
    monkeypatch,
    tmp_path,
) -> None:
    _FakeClient.instances.clear()
    monkeypatch.setattr(one_shot_client, "_Client", _FakeRelativePoseClient)
    args = Namespace(
        confirm="EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE",
        socket_path=str(tmp_path / "relative-pose.sock"),
        side="right",
        action="move_tcp_to_pose",
        target_xyz_m=None,
        target_quat_xyzw=None,
        camera_grasp_proposal=None,
        relative_delta_xyz_m=(0.0, 0.08, 0.0),
        speed_scale=0.2,
        client_id="dexteleop-local-primitive-operator",
        timeout_sec=1.0,
        operation_timeout_sec=None,
    )

    operation = one_shot_client.execute_one_shot(args)

    assert operation["status"] == "completed"
    client = _FakeClient.instances[0]
    snapshot_call_index = next(
        index
        for index, (method, _) in enumerate(client.calls)
        if method == "get_observation_snapshot"
    )
    submit_call_index = next(
        index
        for index, (method, _) in enumerate(client.calls)
        if method == "submit_skill_request"
    )
    assert snapshot_call_index < submit_call_index
    assert client.closed is True


def test_absolute_pose_client_requires_a_complete_pose_before_connect(
    monkeypatch,
    tmp_path,
) -> None:
    _FakeClient.instances.clear()
    monkeypatch.setattr(one_shot_client, "_Client", _FakeAbsolutePoseClient)
    args = Namespace(
        confirm="EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE",
        socket_path=str(tmp_path / "absolute-pose.sock"),
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
        socket_path=str(tmp_path / "absolute-pose.sock"),
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


def test_grasp_approach_accepts_the_absolute_tcp_input_contract() -> None:
    pose, proposal, delta = one_shot_client._absolute_tcp_input_from_args(
        Namespace(
            action="approach_grasp_pose",
            target_xyz_m=(0.1, -0.2, 0.3),
            target_quat_xyzw=(0.0, 0.0, 0.0, 2.0),
            camera_grasp_proposal=None,
            relative_delta_xyz_m=None,
            speed_scale=0.25,
        )
    )

    assert pose == (0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0)
    assert proposal is None
    assert delta is None


def test_relative_delta_is_mutually_exclusive_with_direct_target() -> None:
    args = Namespace(
        action="move_tcp_to_pose",
        target_xyz_m=(0.1, 0.2, 0.3),
        target_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        camera_grasp_proposal=None,
        relative_delta_xyz_m=(0.0, 0.08, 0.0),
        speed_scale=0.2,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        one_shot_client._absolute_tcp_input_from_args(args)


def test_relative_delta_rejects_zero_without_connecting(monkeypatch, tmp_path) -> None:
    _FakeClient.instances.clear()
    monkeypatch.setattr(one_shot_client, "_Client", _FakeAbsolutePoseClient)
    args = Namespace(
        confirm="EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE",
        socket_path=str(tmp_path / "relative-pose.sock"),
        side="left",
        action="move_tcp_to_pose",
        target_xyz_m=None,
        target_quat_xyzw=None,
        camera_grasp_proposal=None,
        relative_delta_xyz_m=(0.0, 0.0, 0.0),
        speed_scale=0.2,
        client_id="dexteleop-local-primitive-operator",
        timeout_sec=1.0,
        operation_timeout_sec=None,
    )

    with pytest.raises(ValueError, match="must be nonzero"):
        one_shot_client.execute_one_shot(args)

    assert _FakeClient.instances == []


class _SubmissionClient:
    def __init__(self):
        self.requests = []
        self.error = None

    def call(self, method, payload):
        assert method == "submit_skill_request"
        self.requests.append(payload)
        if self.error is not None:
            raise self.error
        return {"accepted": False, "reason_code": "active_operation_exists_no_queue"}

def _sequence_request(client, socket_path, *, session="session:one", client_id="operator"):
    return one_shot_client._submit_with_client_sequence(
        client, {"runtime_session_id": session, "client_id": client_id},
        socket_path=str(socket_path), timeout_sec=1,
    )

def test_client_sequence_persists_and_is_scoped_to_session_client_and_endpoint(tmp_path):
    endpoint = tmp_path / "gateway.sock"
    client = _SubmissionClient()
    _sequence_request(client, endpoint)
    # A rejected RPC still consumes its reserved sequence; no automatic retry.
    _sequence_request(client, endpoint)
    _sequence_request(client, endpoint, session="session:two")
    _sequence_request(client, endpoint, client_id="another")
    _sequence_request(client, tmp_path / "another.sock")
    _sequence_request(client, endpoint)
    assert [request["client_sequence"] for request in client.requests] == [1, 2, 1, 1, 1, 3]
    assert (tmp_path / "gateway.sock.client-sequences.json").stat().st_mode & 0o777 == 0o600

def test_client_sequence_is_not_reused_after_lost_rpc_response(tmp_path):
    client = _SubmissionClient()
    client.error = TimeoutError("lost response")
    with pytest.raises(TimeoutError, match="lost response"):
        _sequence_request(client, tmp_path / "gateway.sock")
    assert len(client.requests) == 1
    client.error = None
    _sequence_request(client, tmp_path / "gateway.sock")
    assert [request["client_sequence"] for request in client.requests] == [1, 2]

@pytest.mark.parametrize("contents", ["", "{", "[]", '{"s":{"c":true}}', '{"s":{"c":0}}'])
def test_invalid_client_sequence_state_fails_without_submission(tmp_path, contents):
    (tmp_path / "gateway.sock.client-sequences.json").write_text(contents)
    client = _SubmissionClient()
    with pytest.raises(ValueError):
        _sequence_request(client, tmp_path / "gateway.sock")
    assert not client.requests

@pytest.mark.parametrize("suffix", [".client-sequences.json", ".client-sequences.json.lock"])
def test_client_sequence_refuses_symlink_state_or_lock(tmp_path, suffix):
    target = tmp_path / "other.json"
    target.write_text("{}")
    (tmp_path / ("gateway.sock" + suffix)).symlink_to(target)
    client = _SubmissionClient()
    with pytest.raises(OSError):
        _sequence_request(client, tmp_path / "gateway.sock")
    assert not client.requests
    assert target.read_text() == "{}"

def test_sequence_storage_failure_does_not_submit_or_discard_existing_state(monkeypatch, tmp_path):
    client = _SubmissionClient()
    endpoint = tmp_path / "gateway.sock"
    _sequence_request(client, endpoint)
    state_path = tmp_path / "gateway.sock.client-sequences.json"
    previous = state_path.read_bytes()

    def fail_replace(*args):
        raise OSError("storage unavailable")

    monkeypatch.setattr(one_shot_client.os, "replace", fail_replace)
    with pytest.raises(OSError, match="storage unavailable"):
        _sequence_request(client, endpoint)
    assert len(client.requests) == 1
    assert state_path.read_bytes() == previous
    assert set(tmp_path.iterdir()) == {state_path, tmp_path / (state_path.name + ".lock")}

def test_client_sequence_lock_wait_is_bounded_and_does_not_submit(tmp_path):
    lock_path = tmp_path / "gateway.sock.client-sequences.json.lock"
    with lock_path.open("w") as lock:
        lock_path.chmod(0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        client = _SubmissionClient()
        with pytest.raises(TimeoutError, match="submission lock"):
            one_shot_client._submit_with_client_sequence(
                client, {"runtime_session_id": "s", "client_id": "c"},
                socket_path=str(tmp_path / "gateway.sock"), timeout_sec=0.02,
            )
        assert not client.requests


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_client_sequence_timeout_does_not_reserve_or_submit(tmp_path, timeout):
    client = _SubmissionClient()
    with pytest.raises(ValueError, match="finite and positive"):
        one_shot_client._submit_with_client_sequence(
            client, {"runtime_session_id": "s", "client_id": "c"},
            socket_path=str(tmp_path / "gateway.sock"), timeout_sec=timeout,
        )
    assert client.requests == []
    assert list(tmp_path.iterdir()) == []

@pytest.mark.parametrize("suffix", [".client-sequences.json", ".client-sequences.json.lock"])
def test_client_sequence_rejects_other_user_writable_files(tmp_path, suffix):
    path = tmp_path / ("gateway.sock" + suffix)
    path.write_text("{}")
    path.chmod(0o666)
    client = _SubmissionClient()
    with pytest.raises(ValueError, match="writable by other users"):
        _sequence_request(client, tmp_path / "gateway.sock")
    assert not client.requests

def test_concurrent_processes_reserve_and_submit_in_sequence_order(tmp_path):
    code = '''
import json, sys, time
from pathlib import Path
from agent_gateway.profiles.real.one_shot_client import _submit_with_client_sequence
class Client:
    def call(self, method, payload):
        time.sleep(0.02)
        with Path(sys.argv[2]).open("a") as stream:
            stream.write(json.dumps(payload) + "\\n")
        return {"accepted": False}
_submit_with_client_sequence(Client(), {"runtime_session_id":"same", "client_id":"same"},
    socket_path=sys.argv[1], timeout_sec=5)
'''
    output = tmp_path / "submissions.jsonl"
    children = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(tmp_path / "gateway.sock"), str(output)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(6)
    ]
    try:
        for child in children:
            _, error = child.communicate(timeout=10)
            assert child.returncode == 0, error
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=3)
    requests = [json.loads(line) for line in output.read_text().splitlines()]
    assert [request["client_sequence"] for request in requests] == list(range(1, 7))

@pytest.mark.parametrize(
    "lifecycle,gate,allowed",
    [
        ("hold", "hold", True),
        ("running", "ready", True),
        ("fault", "fault", False),
        ("hold", "fault", False),
        ("closed", "hold", False),
    ],
)
def test_motion_client_reenters_confirmed_hold_without_gripper_route(
    monkeypatch, tmp_path, lifecycle, gate, allowed,
):
    class Client(_FakeAbsolutePoseClient):
        def call(self, method, payload):
            result = super().call(method, payload)
            if method == "get_observation_snapshot":
                result["lifecycle_state"] = lifecycle
                result["execution_gate"] = gate
            return result

    _FakeClient.instances.clear()
    monkeypatch.setattr(one_shot_client, "_Client", Client)
    args = Namespace(
        confirm="EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE",
        socket_path=str(tmp_path / "state.sock"), side="right", action="move_tcp_to_pose",
        target_xyz_m=(0.31, -0.12, 0.28), target_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        speed_scale=0.2, client_id="operator", timeout_sec=1.0, operation_timeout_sec=None,
    )
    if allowed:
        assert one_shot_client.execute_one_shot(args)["status"] == "completed"
    else:
        with pytest.raises(RuntimeError, match="Lower real Skill owner"):
            one_shot_client.execute_one_shot(args)
        assert not (tmp_path / "state.sock.client-sequences.json").exists()
    assert _FakeClient.instances[0].closed


def test_repeated_one_shot_calls_do_not_replay_the_first_sequence(monkeypatch, tmp_path):
    sequences = []

    class Client(_FakeAbsolutePoseClient):
        def call(self, method, payload):
            if method == "submit_skill_request":
                sequence = payload["client_sequence"]
                if sequences and sequence <= sequences[-1]:
                    return {"accepted": False, "reason_code": "client_sequence_not_monotonic"}
                sequences.append(sequence)
            return super().call(method, payload)

    monkeypatch.setattr(one_shot_client, "_Client", Client)
    args = Namespace(
        confirm="EXECUTE_AUTHORIZED_REAL_TCP_PRIMITIVE",
        socket_path=str(tmp_path / "repeat.sock"), side="right", action="move_tcp_to_pose",
        target_xyz_m=(0.31, -0.12, 0.28), target_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        speed_scale=0.2, client_id="operator", timeout_sec=1.0, operation_timeout_sec=None,
    )
    for _ in range(2):
        assert one_shot_client.execute_one_shot(args)["status"] == "completed"
    assert sequences == [1, 2]


def test_simple_client_entry_does_not_import_server_or_motion_stacks():
    from pathlib import Path

    script = r'''
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
blocked = ("scipy", "hardware", "sim", "teleop_runtime", "deployments")
class NoApplicationImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError("unneeded client import: " + fullname)
sys.meta_path.insert(0, NoApplicationImports())
from agent_gateway.profiles.real import one_shot_client as client
for side in ("right",):
    for state in ("open", "closed"):
        args = client.build_parser().parse_args([
            "--socket-path", "/unused-offline.sock", "--side", side,
            "--action", "set_gripper_state", "--target-state", state])
        assert client._absolute_tcp_input_from_args(args) == (None, None, None)
        parameters = client._parameters(action=args.action,
            skill_id="manipulation.set_gripper_state", side=side, snapshot={},
            target_state=state, schema_entry={"parameter_schema": {"properties": {
                "target_state": {"enum": ["open", "closed"]},
                "component_id": {"enum": ["left_gripper", "right_gripper"]}}}})
        assert parameters == {"target_state": state}
args = client.build_parser().parse_args([
    "--socket-path", "/unused-offline.sock", "--side", "right",
    "--action", "move_tcp_to_pose", "--relative-delta-xyz-m", "0", "0", ".03"])
assert client._absolute_tcp_input_from_args(args) == (None, None, (0., 0., .03))
assert client._relative_tcp_target_from_snapshot(snapshot={"measurement_summary": {
    "pose_xyz_quat_xyzw_by_component": {"right_arm": [0., 0., .3, 0., 0., 0., 1.]}}},
    side="right", delta_xyz_m=(0., 0., .03)) == (0., 0., .3 + .03, 0., 0., 0., 1.)
assert all(name not in sys.modules for name in blocked)
'''
    completed = subprocess.run([sys.executable, "-I", "-c", script,
        str(Path(__file__).resolve().parents[1])], capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_fixed_orientation_client_uses_the_same_skill_ids_as_the_server():
    from agent_gateway.profiles.real.motion_primitive_profiles import (
        REAL_AGENT_TCP_FIXED_ORIENTATION_HEADING_BY_SKILL_ID_V1,
    )
    assert set(one_shot_client.FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1) == set(
        REAL_AGENT_TCP_FIXED_ORIENTATION_HEADING_BY_SKILL_ID_V1)
