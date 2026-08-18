from __future__ import annotations

from argparse import Namespace
from typing import ClassVar

from agent_gateway.profiles.real import one_shot_client


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
        operation_timeout_sec=1.0,
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
